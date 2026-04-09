#!/usr/bin/env python3
"""
compute_success_distribution.py — Bimodal beta mixture modeling of success metrics.

Computes the distribution of success metrics across trajectories and models it
as a mixture of two beta distributions. Uses Monte Carlo sampling to estimate:
  1. True distribution: from ground truth offline demos (directly calculable)
  2. Estimated distribution: from world model outcomes
  3. Parameter distributions: α, β, π are themselves distributed

Success metric is bounded in [0, 1] as: max(0, 1 - green_pixel_ratio)

The script:
  - Samples trajectories from the validation dataset
  - Computes success metric for both GT and WM trajectories
  - Fits bimodal beta mixture models to the distributions
  - Uses Monte Carlo sampling to estimate parameter uncertainty
  - Saves statistics and visualizations

Usage:
    python compute_success_distribution.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name pusht \\
        --n_eval 200 \\
        --n_monte_carlo 1000 \\
        --rollout_length 10 \\
        --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/success_distribution
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from scipy import stats
from scipy.optimize import minimize

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from direct_success_metric import count_green_pixels
from utils import seed


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

class Tee:
    """Mirror stdout to a log file."""
    def __init__(self, path):
        self._file = open(path, "w", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._file.write(data)
        self._stdout.write(data)

    def flush(self):
        self._file.flush()
        self._stdout.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1, 1] → (H, W, C) uint8."""
    img = frame_chw.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().float().numpy()
    return (img * 255).astype(np.uint8)


def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(T, C, H, W) in [-1, 1] → (T, H, W, C) uint8."""
    arr = frames.cpu().float().numpy()          # (T, C, H, W)
    arr = arr.transpose(0, 2, 3, 1)             # (T, H, W, C)
    arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_gt_actions_rollout(wm, obs, act, frameskip, num_hist, rollout_length, device):
    """
    Open-loop WM rollout driven by GT actions.

    Args:
        wm:             world model (with decoder)
        obs:            dict {"visual": (T_raw, C, H, W), "proprio": (T_raw, D), ...}
        act:            (T_raw, action_dim) raw actions from dataset
        frameskip:      frameskip used during WM training
        num_hist:       number of context frames for the WM encoder
        rollout_length: number of WM steps to predict
        device:         torch device

    Returns:
        pred_frames : (rollout_length+1, C, H, W) tensor in [-1, 1] — WM decoded frames
        gt_frames   : (rollout_length+1, C, H, W) tensor in [-1, 1] — GT frames (subsampled)
    """
    # Subsample at frameskip intervals
    obs_sub = {
        k: v[0 : rollout_length * frameskip + 1 : frameskip]
        for k, v in obs.items()
    }
    # Group raw actions → (rollout_length, action_dim * frameskip)
    act_sub = act[0 : rollout_length * frameskip]
    act_sub = rearrange(act_sub, "(h f) d -> h (f d)", f=frameskip)

    obs_0 = {k: v[:num_hist].unsqueeze(0).to(device) for k, v in obs_sub.items()}
    actions = act_sub.unsqueeze(0).to(device)   # (1, rollout_length, action_dim*frameskip)

    with torch.no_grad():
        z_obses, _ = wm.rollout(obs_0, actions)
        pred_obs, _ = wm.decode_obs(z_obses)
        pred_frames = pred_obs["visual"][0].cpu()   # (T, C, H, W)

    gt_frames = obs_sub["visual"]                   # (rollout_length+1, C, H, W)

    # Align lengths (pred may have T = rollout_length+1 starting from context end)
    T = min(pred_frames.shape[0], gt_frames.shape[0])
    return pred_frames[-T:], gt_frames[-T:]


def sample_valid_trajectory(dset, rollout_length, frameskip, rng):
    """Return the first trajectory in a shuffled pass that is long enough."""
    min_raw = rollout_length * frameskip + 1
    indices = list(range(len(dset)))
    rng.shuffle(indices)
    for idx in indices:
        obs, act, *_ = dset[idx]
        if obs["visual"].shape[0] >= min_raw and act.shape[0] >= rollout_length * frameskip:
            return obs, act
    raise RuntimeError(
        f"No trajectory has >= {min_raw} raw frames for "
        f"rollout_length={rollout_length}, frameskip={frameskip}."
    )


# ---------------------------------------------------------------------------
# Bimodal Beta Mixture Model
# ---------------------------------------------------------------------------

class BimodalBetaMixture:
    """
    Bimodal beta mixture: p(x) = π * Beta(x; α₁, β₁) + (1-π) * Beta(x; α₂, β₂)
    where x ∈ [0, 1], π ∈ [0, 1], and α₁, β₁, α₂, β₂ > 0.
    """
    def __init__(self):
        self.alpha1 = None
        self.beta1 = None
        self.alpha2 = None
        self.beta2 = None
        self.pi = None

    def fit(self, data, max_iters=100, tol=1e-6):
        """
        Fit bimodal beta mixture to data using EM algorithm.

        Args:
            data: (N,) array of samples in [0, 1]
            max_iters: maximum EM iterations
            tol: convergence tolerance
        """
        data = np.asarray(data).flatten()
        # Clip to (1e-6, 1-1e-6) to avoid boundary issues with beta distribution
        data = np.clip(data, 1e-6, 1.0 - 1e-6)
        N = len(data)

        # Initialize with k-means-like split on sorted data
        sorted_data = np.sort(data)
        split_idx = N // 2
        data_1 = sorted_data[:split_idx]
        data_2 = sorted_data[split_idx:]

        # Initialize parameters using method of moments for beta distribution
        def moments_to_beta(samples):
            """Convert sample mean/var to alpha/beta parameters."""
            mean = np.mean(samples)
            var = np.var(samples)
            # Avoid division by zero and ensure positive parameters
            var = max(var, 1e-6)
            denom = mean * (1 - mean) / var - 1
            denom = max(denom, 1e-3)  # Avoid very small denominators
            alpha = mean * denom
            beta = (1 - mean) * denom
            return max(alpha, 0.1), max(beta, 0.1)

        self.alpha1, self.beta1 = moments_to_beta(data_1)
        self.alpha2, self.beta2 = moments_to_beta(data_2)
        self.pi = 0.5

        # EM iterations
        for iteration in range(max_iters):
            # E-step: compute responsibilities
            logp1 = stats.beta.logpdf(data, self.alpha1, self.beta1)
            logp2 = stats.beta.logpdf(data, self.alpha2, self.beta2)

            # Numerically stable log-sum-exp
            max_logp = np.maximum(logp1, logp2)
            logp1_norm = logp1 - max_logp
            logp2_norm = logp2 - max_logp

            p1 = self.pi * np.exp(logp1_norm)
            p2 = (1 - self.pi) * np.exp(logp2_norm)
            r1 = p1 / (p1 + p2 + 1e-10)
            r2 = 1 - r1

            # M-step: update parameters
            N1 = np.sum(r1) + 1e-10
            N2 = np.sum(r2) + 1e-10

            # Update mixture weight
            pi_new = N1 / N

            # Update beta parameters using method of moments
            mean1 = np.sum(r1 * data) / N1
            var1 = np.sum(r1 * (data - mean1) ** 2) / N1
            mean2 = np.sum(r2 * data) / N2
            var2 = np.sum(r2 * (data - mean2) ** 2) / N2

            # Method of moments for beta: E[X] = α/(α+β), Var[X] = αβ/((α+β)²(α+β+1))
            def estimate_beta_params(mean, var):
                """Estimate alpha and beta from empirical mean and variance."""
                var = max(var, 1e-6)
                mean = np.clip(mean, 1e-3, 1 - 1e-3)
                denom = (mean * (1 - mean) / var) - 1
                denom = max(denom, 1e-3)
                alpha = mean * denom
                beta = (1 - mean) * denom
                return max(alpha, 0.1), max(beta, 0.1)

            alpha1_new, beta1_new = estimate_beta_params(mean1, var1)
            alpha2_new, beta2_new = estimate_beta_params(mean2, var2)

            # Check convergence
            converged = (
                abs(self.alpha1 - alpha1_new) < tol and
                abs(self.beta1 - beta1_new) < tol and
                abs(self.alpha2 - alpha2_new) < tol and
                abs(self.beta2 - beta2_new) < tol and
                abs(self.pi - pi_new) < tol
            )

            self.alpha1 = alpha1_new
            self.beta1 = beta1_new
            self.alpha2 = alpha2_new
            self.beta2 = beta2_new
            self.pi = pi_new

            if converged:
                break

    def pdf(self, x):
        """Evaluate bimodal mixture PDF at x."""
        x = np.asarray(x)
        return (
            self.pi * stats.beta.pdf(x, self.alpha1, self.beta1) +
            (1 - self.pi) * stats.beta.pdf(x, self.alpha2, self.beta2)
        )

    def sample(self, n, rng=None):
        """Draw n samples from the bimodal mixture."""
        if rng is None:
            rng = np.random.RandomState()

        # Choose which component
        comp = rng.binomial(1, self.pi, n)
        samples = np.zeros(n)
        samples[comp == 1] = rng.beta(self.alpha1, self.beta1, np.sum(comp == 1))
        samples[comp == 0] = rng.beta(self.alpha2, self.beta2, np.sum(comp == 0))
        return samples

    def to_dict(self):
        """Serialize to dict."""
        return {
            "alpha1": float(self.alpha1),
            "beta1": float(self.beta1),
            "alpha2": float(self.alpha2),
            "beta2": float(self.beta2),
            "pi": float(self.pi),
        }



# ---------------------------------------------------------------------------
# Parameter Distribution via Monte Carlo
# ---------------------------------------------------------------------------

def estimate_parameter_distribution(data, n_monte_carlo, rng=None):
    """
    Estimate distribution of bimodal beta mixture parameters via Monte Carlo.

    Bootstrap-resamples the data n_monte_carlo times, fits a bimodal beta mixture
    to each resample, and returns the resulting parameter samples.

    Args:
        data: (N,) array of samples in [0, 1]
        n_monte_carlo: number of bootstrap resamples
        rng: numpy random state

    Returns:
        param_samples: dict with keys alpha1, beta1, alpha2, beta2, pi
                       each mapping to (n_monte_carlo,) arrays
    """
    if rng is None:
        rng = np.random.RandomState()

    data = np.asarray(data).flatten()
    N = len(data)

    param_samples = {
        "alpha1": [],
        "beta1": [],
        "alpha2": [],
        "beta2": [],
        "pi": [],
    }

    for i in range(n_monte_carlo):
        # Bootstrap resample
        indices = rng.choice(N, N, replace=True)
        data_boot = data[indices]

        # Fit mixture
        mixture = BimodalBetaMixture()
        mixture.fit(data_boot)

        param_samples["alpha1"].append(mixture.alpha1)
        param_samples["beta1"].append(mixture.beta1)
        param_samples["alpha2"].append(mixture.alpha2)
        param_samples["beta2"].append(mixture.beta2)
        param_samples["pi"].append(mixture.pi)

    for key in param_samples:
        param_samples[key] = np.array(param_samples[key])

    return param_samples


def summarize_parameter_distribution(param_samples):
    """Compute mean, std, and credible intervals for parameter samples."""
    summary = {}
    for key, samples in param_samples.items():
        summary[key] = {
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "median": float(np.median(samples)),
            "q05": float(np.percentile(samples, 5)),
            "q95": float(np.percentile(samples, 95)),
        }
    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_success_distributions(gt_scores, wm_scores, gt_mixture, wm_mixture, output_dir):
    """Save histograms and fitted mixtures for GT and WM success scores."""
    if not HAVE_MATPLOTLIB:
        print("[Warning] matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # GT distribution
    ax = axes[0]
    ax.hist(gt_scores, bins=20, density=True, alpha=0.6, color="steelblue", label="GT data")
    x = np.linspace(min(gt_scores) - 0.1, max(gt_scores) + 0.1, 200)
    ax.plot(x, gt_mixture.pdf(x), "r-", linewidth=2, label="Fitted mixture")
    ax.set_xlabel("Success metric")
    ax.set_ylabel("Density")
    ax.set_title("GT Success Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # WM distribution
    ax = axes[1]
    ax.hist(wm_scores, bins=20, density=True, alpha=0.6, color="darkorange", label="WM data")
    x = np.linspace(min(wm_scores) - 0.1, max(wm_scores) + 0.1, 200)
    ax.plot(x, wm_mixture.pdf(x), "r-", linewidth=2, label="Fitted mixture")
    ax.set_xlabel("Success metric")
    ax.set_ylabel("Density")
    ax.set_title("WM Success Distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "success_distributions.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved distributions plot: {out_path}")


def plot_wm_parameter_distributions(param_samples_wm, output_dir):
    """Save WM parameter distributions (alpha, beta, pi) from Monte Carlo bootstrap."""
    if not HAVE_MATPLOTLIB:
        print("[Warning] matplotlib not available; skipping parameter plots")
        return

    params_to_plot = ["alpha1", "alpha2", "beta1", "beta2", "pi"]
    fig, axes = plt.subplots(len(params_to_plot), 1, figsize=(6, 3 * len(params_to_plot)))

    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        ax.hist(param_samples_wm[param], bins=30, alpha=0.7, color="darkorange", edgecolor="black")
        ax.set_title(f"WM {param}")
        ax.set_xlabel(param)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "wm_parameter_distributions.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved WM parameter distributions plot: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_success_distribution_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(cfg.output_dir, "compute_success_distribution.log"))

    # ── World model ───────────────────────────────────────────────────────
    model_path = Path(cfg.ckpt_base_path) / "outputs" / cfg.model_name
    with open(model_path / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    _, dset_split = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset_split["valid"]
    print(f"Validation dataset: {len(dset)} trajectories")

    model_ckpt = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    wm = _load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    if wm.decoder is None:
        raise RuntimeError("World model has no decoder — pixel decoding required for success metric.")

    num_hist  = model_cfg.num_hist
    frameskip = model_cfg.frameskip
    rollout_length = cfg.rollout_length

    print(f"\nWorld model '{cfg.model_name}' (epoch={cfg.model_epoch})")
    print(f"  rollout_length={rollout_length}, frameskip={frameskip}, num_hist={num_hist}")
    print(f"  Monte Carlo samples for parameter distribution: {cfg.n_monte_carlo}")

    print(f"\nRunning {cfg.n_eval} evaluations ...\n")

    # ── Evaluation loop ───────────────────────────────────────────────────
    gt_scores = []
    wm_scores = []
    records = []

    for i in range(cfg.n_eval):
        obs, act = sample_valid_trajectory(dset, rollout_length, frameskip, rng)

        # WM open-loop rollout with GT actions
        pred_frames, gt_frames = run_gt_actions_rollout(
            wm, obs, act, frameskip, num_hist, rollout_length, device
        )

        # Success metric: 1 - (green-pixel ratio). Lower green pixels = higher success.
        gt_first_np = _to_hwc_uint8(gt_frames[0])
        gt_final_np = _to_hwc_uint8(gt_frames[-1])
        wm_first_np = _to_hwc_uint8(pred_frames[0])
        wm_final_np = _to_hwc_uint8(pred_frames[-1])

        gt_score = max(0.0, 1.0 - (count_green_pixels(gt_final_np) / max(count_green_pixels(gt_first_np), 1)))
        wm_score = max(0.0, 1.0 - (count_green_pixels(wm_final_np) / max(count_green_pixels(wm_first_np), 1)))

        gt_scores.append(gt_score)
        wm_scores.append(wm_score)

        records.append({
            "traj_idx": i,
            "gt_score": gt_score,
            "wm_score": wm_score,
        })

        if (i + 1) % max(1, cfg.n_eval // 10) == 0:
            print(f"  [{i+1:3d}/{cfg.n_eval}]  GT: {np.mean(gt_scores):.3f} ± {np.std(gt_scores):.3f}  "
                  f"WM: {np.mean(wm_scores):.3f} ± {np.std(wm_scores):.3f}")

    gt_scores = np.array(gt_scores)
    wm_scores = np.array(wm_scores)

    # ── Fit bimodal mixtures ──────────────────────────────────────────────
    print(f"\nFitting bimodal beta mixtures ...")
    gt_mixture = BimodalBetaMixture()
    gt_mixture.fit(gt_scores)

    wm_mixture = BimodalBetaMixture()
    wm_mixture.fit(wm_scores)

    print(f"  GT mixture: π={gt_mixture.pi:.3f}, α₁={gt_mixture.alpha1:.3f}, β₁={gt_mixture.beta1:.3f}")
    print(f"              α₂={gt_mixture.alpha2:.3f}, β₂={gt_mixture.beta2:.3f}")
    print(f"  WM mixture: π={wm_mixture.pi:.3f}, α₁={wm_mixture.alpha1:.3f}, β₁={wm_mixture.beta1:.3f}")
    print(f"              α₂={wm_mixture.alpha2:.3f}, β₂={wm_mixture.beta2:.3f}")

    # ── Monte Carlo parameter distribution (WM only) ───────────────────────
    # GT parameters are fixed (directly computed from the dataset); no uncertainty estimation needed.
    print(f"\nEstimating WM parameter distributions via Monte Carlo ({cfg.n_monte_carlo} samples) ...")
    param_samples_wm = estimate_parameter_distribution(wm_scores, cfg.n_monte_carlo, rng)

    param_summary_wm = summarize_parameter_distribution(param_samples_wm)

    print(f"\nWM parameter summary:")
    for param, stats_dict in sorted(param_summary_wm.items()):
        print(f"  {param}: mean={stats_dict['mean']:.4f}, std={stats_dict['std']:.4f}, "
              f"[{stats_dict['q05']:.4f}, {stats_dict['q95']:.4f}]")

    # ── Save plots ────────────────────────────────────────────────────────
    plot_success_distributions(gt_scores, wm_scores, gt_mixture, wm_mixture, cfg.output_dir)
    plot_wm_parameter_distributions(param_samples_wm, cfg.output_dir)

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "metadata": {
            "model_name": cfg.model_name,
            "model_epoch": cfg.model_epoch,
            "n_eval": cfg.n_eval,
            "n_monte_carlo": cfg.n_monte_carlo,
            "rollout_length": rollout_length,
            "frameskip": frameskip,
        },
        "gt_statistics": {
            "mean": float(np.mean(gt_scores)),
            "std": float(np.std(gt_scores)),
            "min": float(np.min(gt_scores)),
            "max": float(np.max(gt_scores)),
            "median": float(np.median(gt_scores)),
        },
        "wm_statistics": {
            "mean": float(np.mean(wm_scores)),
            "std": float(np.std(wm_scores)),
            "min": float(np.min(wm_scores)),
            "max": float(np.max(wm_scores)),
            "median": float(np.median(wm_scores)),
        },
        "gt_bimodal_mixture": gt_mixture.to_dict(),
        "wm_bimodal_mixture": wm_mixture.to_dict(),
        "wm_parameter_summary": param_summary_wm,
        "trajectories": records,
    }

    out_json = os.path.join(cfg.output_dir, "success_distribution.json")
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_json}")

    tee.close()

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute bimodal beta mixture model of success metrics."
    )
    parser.add_argument("--ckpt_base_path", required=True,
                        help="Base directory containing outputs/<model_name>/")
    parser.add_argument("--model_name", required=True,
                        help="WM model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest",
                        help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--n_eval", type=int, default=200,
                        help="Number of trajectories to evaluate (default: 200)")
    parser.add_argument("--n_monte_carlo", type=int, default=1000,
                        help="Number of Monte Carlo bootstrap samples for parameter distribution (default: 1000)")
    parser.add_argument("--rollout_length", type=int, default=10,
                        help="WM rollout steps per trajectory (default: 10)")
    parser.add_argument("--output_dir", default="./eval_results/success_distribution",
                        help="Directory for JSON and plots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_success_distribution_main(args)
