#!/usr/bin/env python3
"""
policy_ranking.py — Rank noised policies using the world model as a simulator.

For each noise level σ, wraps the base diffusion policy in NoisedPolicyWrapper(σ),
generates actions from GT observations, runs those actions open-loop through the
world model, measures success on the final decoded frame using the green-pixel
metric, and plots average success vs noise level.

Usage:
    python policy_ranking.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name pusht \\
        --model_epoch latest \\
        --n_eval 100 \\
        --rollout_length 10 \\
        --noise_levels 0.0 0.1 0.2 0.5 1.0 2.0 \\
        --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/policy_ranking \\
        --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image as PILImage

try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from lerobot_utils import NoisedPolicyWrapper
from direct_success_metric import count_green_pixels
from utils import seed


# ---------------------------------------------------------------------------
# Logging helper  (mirrors compare_success.py)
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
# Frame helpers  (mirrors compare_success.py)
# ---------------------------------------------------------------------------

def _to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1, 1] → (H, W, C) uint8."""
    img = frame_chw.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().float().numpy()
    return (img * 255).astype(np.uint8)


def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(T, C, H, W) in [-1, 1] → (T, H, W, C) uint8."""
    arr = frames.cpu().float().numpy()
    arr = arr.transpose(0, 2, 3, 1)
    arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Trajectory sampling  (mirrors compare_success.py)
# ---------------------------------------------------------------------------

def sample_valid_trajectory(dset, rollout_length, frameskip, rng):
    """Return a trajectory from a shuffled pass that is long enough."""
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
# Policy-driven WM rollout
# ---------------------------------------------------------------------------

def run_policy_wm_rollout(
    wm,
    obs,
    act,
    frameskip: int,
    num_hist: int,
    rollout_length: int,
    device: torch.device,
    policy,
    policy_img_size: int = 96,
):
    """
    Open-loop WM rollout driven by a policy's actions on GT observations.

    For each timestep t, the policy receives the GT observation (image + proprio)
    at that step and produces an action.  Actions are then grouped by frameskip
    and fed into the WM exactly as in compare_success.run_gt_actions_rollout —
    except the actions come from the policy (possibly noised) rather than the
    dataset.

    Args:
        wm:             world model (with decoder)
        obs:            dict {"visual": (T_raw, C, H, W), "proprio": (T_raw, D), ...}
        act:            (T_raw, action_dim) raw GT actions (used only for context frames)
        frameskip:      frameskip used during WM training
        num_hist:       number of context frames for the WM encoder
        rollout_length: number of WM steps to predict
        device:         torch device
        policy:         PolicyWrapper (or NoisedPolicyWrapper) instance
        policy_img_size: image size expected by the policy

    Returns:
        pred_frames : (rollout_length+1, C, H, W) tensor in [-1, 1] — WM decoded frames
        gt_frames   : (rollout_length+1, C, H, W) tensor in [-1, 1] — GT frames
    """
    T_need = rollout_length * frameskip + 1

    # Subsample at frameskip intervals
    obs_sub = {k: v[:T_need:frameskip] for k, v in obs.items()}
    # obs_sub["visual"]: (rollout_length+1, C, H, W)

    # ── Generate policy actions from GT observations ──────────────────────
    # We generate one raw action per raw timestep, then group by frameskip.
    n_raw_steps = rollout_length * frameskip  # number of raw actions needed
    policy_actions = []

    visual_raw = obs["visual"][:T_need]  # (T_need, C, H, W)
    proprio_raw = obs.get("proprio", obs.get("state", None))
    if proprio_raw is not None:
        proprio_raw = proprio_raw[:T_need]  # (T_need, D)

    with torch.no_grad():
        for t in range(n_raw_steps):
            # Build image tensor for policy
            frame = visual_raw[t]  # (C, H, W) in [-1, 1]
            img_01 = (frame.clamp(-1, 1) + 1.0) / 2.0  # [0, 1]
            img = F.interpolate(
                img_01.unsqueeze(0).to(device),
                size=(policy_img_size, policy_img_size),
                mode="bilinear",
                align_corners=False,
            )  # (1, C, H, W)

            # Build state tensor for policy
            if proprio_raw is not None:
                state = proprio_raw[t].unsqueeze(0).to(device)  # (1, D)
            else:
                state = torch.zeros(1, 2, device=device)

            action = policy.predict(state, img)  # (1, action_dim)
            policy_actions.append(action.squeeze(0).cpu())  # (action_dim,)

    # Stack → (n_raw_steps, action_dim)
    act_policy = torch.stack(policy_actions, dim=0)

    # Group raw actions → (rollout_length, action_dim * frameskip)
    act_wm = rearrange(act_policy, "(h f) d -> h (f d)", f=frameskip)

    # ── WM rollout ─────────────────────────────────────────────────────────
    obs_0 = {k: v[:num_hist].unsqueeze(0).to(device) for k, v in obs_sub.items()}
    actions = act_wm.unsqueeze(0).to(device)  # (1, rollout_length, action_dim*frameskip)

    with torch.no_grad():
        z_obses, _ = wm.rollout(obs_0, actions)
        pred_obs, _ = wm.decode_obs(z_obses)
        pred_frames = pred_obs["visual"][0].cpu()  # (T, C, H, W)

    gt_frames = obs_sub["visual"]  # (rollout_length+1, C, H, W)

    T = min(pred_frames.shape[0], gt_frames.shape[0])
    return pred_frames[-T:], gt_frames[-T:]


# ---------------------------------------------------------------------------
# Per-noise-level evaluation
# ---------------------------------------------------------------------------

def evaluate_noise_level(
    noise_scale: float,
    wm,
    dset,
    frameskip: int,
    num_hist: int,
    rollout_length: int,
    n_eval: int,
    device: torch.device,
    policy_model_name: str,
    policy_img_size: int,
    rng: np.random.RandomState,
    verbose: bool = True,
) -> dict:
    """
    Run n_eval WM rollouts with a NoisedPolicy at the given noise_scale and
    return per-trajectory scores plus the mean success score.
    """
    policy = NoisedPolicyWrapper(
        model_name=policy_model_name,
        noise_scale=noise_scale,
    )

    scores = []
    for i in range(n_eval):
        obs, act = sample_valid_trajectory(dset, rollout_length, frameskip, rng)
        pred_frames, gt_frames = run_policy_wm_rollout(
            wm, obs, act, frameskip, num_hist, rollout_length, device,
            policy, policy_img_size,
        )

        gt_first_np  = _to_hwc_uint8(gt_frames[0])
        wm_final_np  = _to_hwc_uint8(pred_frames[-1])

        gt_first_green = max(count_green_pixels(gt_first_np), 1)
        wm_score = max(0.0, 1.0 - (count_green_pixels(wm_final_np) / gt_first_green))
        scores.append(wm_score)

        if verbose:
            print(
                f"  σ={noise_scale:.3f}  [{i+1:3d}/{n_eval}]  "
                f"WM score: {wm_score:.3f}"
            )

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    return {"noise_scale": noise_scale, "scores": scores, "mean": mean_score, "std": std_score}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_ranking_plot(results_by_noise, output_dir, model_name, rollout_length):
    """
    Bar + error-bar plot of mean success score vs noise level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_levels = [r["noise_scale"] for r in results_by_noise]
    means        = [r["mean"]        for r in results_by_noise]
    stds         = [r["std"]         for r in results_by_noise]
    n_eval       = len(results_by_noise[0]["scores"]) if results_by_noise else 0

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(noise_levels))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color="steelblue", edgecolor="white", width=0.6,
                  error_kw={"ecolor": "black", "elinewidth": 1.2})
    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05 + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"σ={σ}" for σ in noise_levels], rotation=30, ha="right")
    ax.set_ylabel("Mean WM success score")
    ax.set_title(
        f"{model_name} — WM success vs policy noise (rl={rollout_length}, n={n_eval})"
    )
    ax.set_ylim(0, max(max(means) * 1.2, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"policy_ranking_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved ranking plot: {out_path}")
    return out_path


def save_line_plot(results_by_noise, output_dir, model_name, rollout_length):
    """
    Line plot with shaded ±1 std band: success score vs noise level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    noise_levels = np.array([r["noise_scale"] for r in results_by_noise])
    means        = np.array([r["mean"]        for r in results_by_noise])
    stds         = np.array([r["std"]         for r in results_by_noise])
    n_eval       = len(results_by_noise[0]["scores"]) if results_by_noise else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(noise_levels, means, marker="o", color="steelblue", linewidth=2, label="mean score")
    ax.fill_between(noise_levels, means - stds, means + stds,
                    alpha=0.25, color="steelblue", label="±1 std")
    ax.set_xlabel("Policy noise scale (σ)")
    ax.set_ylabel("Mean WM success score")
    ax.set_title(
        f"{model_name} — Policy ranking via WM (rl={rollout_length}, n={n_eval})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"policy_ranking_line_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved line plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def policy_ranking_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(cfg.output_dir, f"policy_ranking_rl{cfg.rollout_length}.log"))

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
        raise RuntimeError("World model has no decoder — pixel decoding required.")

    num_hist  = model_cfg.num_hist
    frameskip = model_cfg.frameskip

    print(f"\nWorld model '{cfg.model_name}' (epoch={cfg.model_epoch})")
    print(f"  rollout_length={cfg.rollout_length}, frameskip={frameskip}, num_hist={num_hist}")
    print(f"  Policy: {cfg.policy_model_name}")
    print(f"  Noise levels: {cfg.noise_levels}")
    print(f"  n_eval per level: {cfg.n_eval}\n")

    # ── Sweep over noise levels ───────────────────────────────────────────
    results_by_noise = []
    for noise_scale in cfg.noise_levels:
        print(f"\n=== Noise scale σ={noise_scale:.3f} ===")
        result = evaluate_noise_level(
            noise_scale=noise_scale,
            wm=wm,
            dset=dset,
            frameskip=frameskip,
            num_hist=num_hist,
            rollout_length=cfg.rollout_length,
            n_eval=cfg.n_eval,
            device=device,
            policy_model_name=cfg.policy_model_name,
            policy_img_size=cfg.policy_img_size,
            rng=rng,
            verbose=True,
        )
        results_by_noise.append(result)
        print(
            f"  → mean={result['mean']:.4f}  std={result['std']:.4f}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Policy Ranking Summary ===")
    print(f"{'σ':>8}  {'mean score':>12}  {'std':>8}")
    print("-" * 34)
    for r in results_by_noise:
        print(f"{r['noise_scale']:>8.3f}  {r['mean']:>12.4f}  {r['std']:>8.4f}")

    # ── Save plots ────────────────────────────────────────────────────────
    save_ranking_plot(results_by_noise, cfg.output_dir, cfg.model_name, cfg.rollout_length)
    save_line_plot(results_by_noise, cfg.output_dir, cfg.model_name, cfg.rollout_length)

    # ── Save JSON ─────────────────────────────────────────────────────────
    summary = {
        "model_name":       cfg.model_name,
        "model_epoch":      cfg.model_epoch,
        "rollout_length":   cfg.rollout_length,
        "n_eval":           cfg.n_eval,
        "policy_model_name": cfg.policy_model_name,
        "noise_levels":     cfg.noise_levels,
        "results": [
            {"noise_scale": r["noise_scale"], "mean": r["mean"], "std": r["std"]}
            for r in results_by_noise
        ],
    }
    out_json = os.path.join(cfg.output_dir, f"policy_ranking_rl{cfg.rollout_length}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": summary, "per_trajectory": results_by_noise}, f, indent=2)
    print(f"\nResults saved to: {out_json}")
    tee.close()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank noised policies using the world model as a simulator."
    )
    parser.add_argument("--ckpt_base_path", required=True,
                        help="Base directory containing outputs/<model_name>/")
    parser.add_argument("--model_name", required=True,
                        help="WM model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest",
                        help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--policy_model_name", default="lerobot/diffusion_pusht",
                        help="LeRobot policy model name (default: lerobot/diffusion_pusht)")
    parser.add_argument("--policy_img_size", type=int, default=96,
                        help="Image size expected by the policy (default: 96)")
    parser.add_argument("--noise_levels", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
                        help="List of Gaussian noise scales to evaluate (default: 0.0 0.1 0.2 0.5 1.0 2.0)")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of trajectories to evaluate per noise level (default: 100)")
    parser.add_argument("--rollout_length", type=int, default=10,
                        help="WM rollout steps per trajectory (default: 10)")
    parser.add_argument("--output_dir", default="./eval_results/policy_ranking",
                        help="Directory for JSON and plots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy_ranking_main(args)
