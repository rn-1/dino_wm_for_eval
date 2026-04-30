#!/usr/bin/env python3
"""
policy_ranking.py — Rank inhibited policies using the world model as a simulator.

For each inhibition coefficient c, wraps the base diffusion policy in
InhibitedPolicyWrapper(c), generates actions from GT observations, runs those
actions open-loop through the world model, measures success on the final decoded
frame using the green-pixel metric, and plots average success vs inhibition coeff.

Usage:
    python policy_ranking.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name pusht \\
        --model_epoch latest \\
        --n_eval 100 \\
        --rollout_length 10 \\
        --inhibition_coeffs 0.0 0.2 0.4 0.6 0.8 1.0 \\
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
from lerobot_utils import InhibitedPolicyWrapper
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
# Gym (ground-truth) rollout helpers
# ---------------------------------------------------------------------------

def _load_pusht_env():
    """Try to load the PushT gym environment."""
    try:
        import gym
        return gym.make("PushT-v0")
    except Exception:
        from env.pusht.pusht_wrapper import PushTWrapper
        return PushTWrapper()


def run_single_gym_episode(
    policy,
    env,
    rollout_length: int,
    device: torch.device,
    policy_img_size: int = 96,
):
    """
    Run one episode with *policy* in *env* for up to *rollout_length* steps.

    Returns a list of (C, H, W) float tensors in [-1, 1]:
        frames[0]  — initial frame
        frames[-1] — final frame after rollout
    """
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    frames = []

    for _ in range(rollout_length):
        # ── extract image & state ─────────────────────────────────────────
        if isinstance(obs, dict):
            img = obs.get("image", obs.get("visual", next(iter(obs.values()))))
            state = obs.get("state", obs.get("proprio", np.zeros(2, dtype=np.float32)))
        else:
            img = obs
            state = np.zeros(2, dtype=np.float32)

        # ── convert to float tensors ──────────────────────────────────────
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if not isinstance(state, torch.Tensor):
            state = (torch.from_numpy(state).float()
                     if isinstance(state, np.ndarray)
                     else torch.tensor(state, dtype=torch.float32))

        # Ensure (C, H, W)
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3 and img.shape[0] not in (1, 3):
            img = img.permute(2, 0, 1)

        # Normalise to [-1, 1]
        if img.max() > 1.5:
            img = img / 127.5 - 1.0

        frames.append(img.cpu())

        # ── policy action ─────────────────────────────────────────────────
        img_t = F.interpolate(
            img.unsqueeze(0).to(device),
            size=(policy_img_size, policy_img_size),
            mode="bilinear",
            align_corners=False,
        )
        state_t = state.unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy.predict(state_t, img_t)
        action_np = action.squeeze(0).cpu().numpy()

        # ── step env ──────────────────────────────────────────────────────
        step_result = env.step(action_np)
        if len(step_result) == 5:
            obs, _, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            obs, _, done, _ = step_result

        if done:
            break

    # Capture final frame
    if isinstance(obs, dict):
        final_img = obs.get("image", obs.get("visual", None))
    else:
        final_img = obs
    if final_img is not None:
        if isinstance(final_img, np.ndarray):
            final_img = torch.from_numpy(final_img).float()
        if final_img.ndim == 3 and final_img.shape[0] not in (1, 3):
            final_img = final_img.permute(2, 0, 1)
        if final_img.max() > 1.5:
            final_img = final_img / 127.5 - 1.0
        frames.append(final_img.cpu())

    return frames


def evaluate_inhibition_level_gym(
    inhibition_coeff: float,
    n_eval: int,
    rollout_length: int,
    device: torch.device,
    policy_model_name: str,
    policy_img_size: int,
    verbose: bool = True,
) -> dict:
    """
    Run n_eval gym episodes with InhibitedPolicyWrapper at *inhibition_coeff*.
    Returns per-trajectory scores and summary statistics.
    """
    env = _load_pusht_env()
    policy = InhibitedPolicyWrapper(
        model_name=policy_model_name,
        inhibition_coeff=inhibition_coeff,
    )

    scores = []
    for i in range(n_eval):
        frames = run_single_gym_episode(policy, env, rollout_length, device, policy_img_size)

        if len(frames) >= 2:
            first_np = _to_hwc_uint8(frames[0])
            final_np = _to_hwc_uint8(frames[-1])
            gt_first_green = max(count_green_pixels(first_np), 1)
            score = max(0.0, 1.0 - (count_green_pixels(final_np) / gt_first_green))
        else:
            score = 0.0

        scores.append(score)
        if verbose:
            print(
                f"  c={inhibition_coeff:.3f}  [{i+1:3d}/{n_eval}]  "
                f"Gym score: {score:.3f}"
            )

    env.close()
    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    return {"inhibition_coeff": inhibition_coeff, "scores": scores, "mean": mean_score, "std": std_score}


# ---------------------------------------------------------------------------
# Per-noise-level evaluation
# ---------------------------------------------------------------------------

def evaluate_inhibition_level(
    inhibition_coeff: float,
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
    Run n_eval WM rollouts with an InhibitedPolicy at the given inhibition_coeff and
    return per-trajectory scores plus the mean success score.
    """
    policy = InhibitedPolicyWrapper(
        model_name=policy_model_name,
        inhibition_coeff=inhibition_coeff,
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
                f"  c={inhibition_coeff:.3f}  [{i+1:3d}/{n_eval}]  "
                f"WM score: {wm_score:.3f}"
            )

    mean_score = float(np.mean(scores))
    std_score  = float(np.std(scores))
    return {"inhibition_coeff": inhibition_coeff, "scores": scores, "mean": mean_score, "std": std_score}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_ranking_plot(results_by_coeff, output_dir, model_name, rollout_length,
                      gt_results_by_coeff=None):
    """
    Bar + error-bar plot of mean success score vs inhibition coefficient.
    When *gt_results_by_coeff* is provided, GT bars are plotted side-by-side with WM bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeffs    = [r["inhibition_coeff"] for r in results_by_coeff]
    wm_means  = [r["mean"] for r in results_by_coeff]
    wm_stds   = [r["std"]  for r in results_by_coeff]
    n_eval    = len(results_by_coeff[0]["scores"]) if results_by_coeff else 0

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(coeffs))

    if gt_results_by_coeff is not None:
        gt_means = [r["mean"] for r in gt_results_by_coeff]
        gt_stds  = [r["std"]  for r in gt_results_by_coeff]
        width = 0.35
        wm_bars = ax.bar(x - width / 2, wm_means, yerr=wm_stds, capsize=4,
                         color="steelblue", edgecolor="white", width=width,
                         error_kw={"ecolor": "black", "elinewidth": 1.2},
                         label="World-model")
        gt_bars = ax.bar(x + width / 2, gt_means, yerr=gt_stds, capsize=4,
                         color="tomato", edgecolor="white", width=width,
                         error_kw={"ecolor": "black", "elinewidth": 1.2},
                         label="GT gym")
        for bar, val in zip(wm_bars, wm_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="steelblue")
        for bar, val in zip(gt_bars, gt_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="tomato")
        all_means = wm_means + gt_means
        ax.legend(fontsize=9)
    else:
        width = 0.6
        bars = ax.bar(x, wm_means, yerr=wm_stds, capsize=4,
                      color="steelblue", edgecolor="white", width=width,
                      error_kw={"ecolor": "black", "elinewidth": 1.2})
        for bar, val in zip(bars, wm_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(wm_stds) * 0.05 + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=9)
        all_means = wm_means

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in coeffs], rotation=30, ha="right")
    ax.set_ylabel("Mean success score")
    ax.set_title(
        f"{model_name} — success vs inhibition coeff (rl={rollout_length}, n={n_eval})"
    )
    ax.set_ylim(0, max(max(all_means) * 1.2, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"policy_ranking_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved ranking plot: {out_path}")
    return out_path


def save_line_plot(results_by_coeff, output_dir, model_name, rollout_length,
                   gt_results_by_coeff=None):
    """
    Line plot with shaded ±1 std band: success score vs inhibition coefficient.
    When *gt_results_by_coeff* is provided, GT is overlaid as a second line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeffs   = np.array([r["inhibition_coeff"] for r in results_by_coeff])
    wm_means = np.array([r["mean"] for r in results_by_coeff])
    wm_stds  = np.array([r["std"]  for r in results_by_coeff])
    n_eval   = len(results_by_coeff[0]["scores"]) if results_by_coeff else 0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(coeffs, wm_means, marker="o", color="steelblue", linewidth=2, label="WM (world model)")
    ax.fill_between(coeffs, wm_means - wm_stds, wm_means + wm_stds,
                    alpha=0.25, color="steelblue")

    if gt_results_by_coeff is not None:
        gt_means = np.array([r["mean"] for r in gt_results_by_coeff])
        gt_stds  = np.array([r["std"]  for r in gt_results_by_coeff])
        ax.plot(coeffs, gt_means, marker="s", color="tomato", linewidth=2, label="GT gym")
        ax.fill_between(coeffs, gt_means - gt_stds, gt_means + gt_stds,
                        alpha=0.25, color="tomato")

    ax.set_xlabel("Inhibition coefficient (c)")
    ax.set_ylabel("Mean success score")
    ax.set_title(
        f"{model_name} — Policy ranking WM vs GT (rl={rollout_length}, n={n_eval})"
    )
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"policy_ranking_line_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved line plot: {out_path}")
    return out_path


def save_comparison_plot(wm_results, gt_results, output_dir, model_name, rollout_length):
    """
    Two-panel figure: left = WM scores, right = GT gym scores.
    Useful for visually verifying that WM ranking correlates with ground truth.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeffs   = np.array([r["inhibition_coeff"] for r in wm_results])
    wm_means = np.array([r["mean"] for r in wm_results])
    wm_stds  = np.array([r["std"]  for r in wm_results])
    gt_means = np.array([r["mean"] for r in gt_results])
    gt_stds  = np.array([r["std"]  for r in gt_results])
    n_eval   = len(wm_results[0]["scores"]) if wm_results else 0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, means, stds, color, title in [
        (axes[0], wm_means, wm_stds, "steelblue", "World-model rollout"),
        (axes[1], gt_means, gt_stds, "tomato",    "GT gym rollout"),
    ]:
        x = np.arange(len(coeffs))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=color, edgecolor="white", width=0.6,
                      error_kw={"ecolor": "black", "elinewidth": 1.2})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"c={c}" for c in coeffs], rotation=30, ha="right")
        ax.set_ylabel("Mean success score")
        ax.set_title(f"{title} (n={n_eval})")
        ax.set_ylim(0, max(max(means) * 1.2, 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"{model_name} — WM vs GT ranking (rl={rollout_length})", fontsize=12
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"policy_ranking_comparison_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved comparison plot: {out_path}")
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
    print(f"  Inhibition coefficients: {cfg.inhibition_coeffs}")
    print(f"  n_eval per level: {cfg.n_eval}\n")

    # ── Sweep over inhibition coefficients ───────────────────────────────
    results_by_coeff = []
    for inhibition_coeff in cfg.inhibition_coeffs:
        print(f"\n=== Inhibition coeff c={inhibition_coeff:.3f} ===")
        result = evaluate_inhibition_level(
            inhibition_coeff=inhibition_coeff,
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
        results_by_coeff.append(result)
        print(
            f"  → mean={result['mean']:.4f}  std={result['std']:.4f}"
        )

    # ── GT gym sweep (optional) ───────────────────────────────────────────
    gt_results_by_coeff = None
    if getattr(cfg, "run_gym", False):
        print("\n" + "=" * 60)
        print("GT GYM ROLLOUT SWEEP")
        print("=" * 60)
        gt_results_by_coeff = []
        for inhibition_coeff in cfg.inhibition_coeffs:
            print(f"\n=== [Gym] Inhibition coeff c={inhibition_coeff:.3f} ===")
            gt_result = evaluate_inhibition_level_gym(
                inhibition_coeff=inhibition_coeff,
                n_eval=cfg.n_eval,
                rollout_length=cfg.rollout_length,
                device=device,
                policy_model_name=cfg.policy_model_name,
                policy_img_size=cfg.policy_img_size,
                verbose=True,
            )
            gt_results_by_coeff.append(gt_result)
            print(
                f"  → mean={gt_result['mean']:.4f}  std={gt_result['std']:.4f}"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Policy Ranking Summary ===")
    header = f"{'c':>8}  {'WM mean':>10}  {'WM std':>8}"
    sep    = "-" * (34 + (22 if gt_results_by_coeff else 0))
    if gt_results_by_coeff:
        header += f"  {'GT mean':>10}  {'GT std':>8}"
    print(header)
    print(sep)
    for i, r in enumerate(results_by_coeff):
        line = f"{r['inhibition_coeff']:>8.3f}  {r['mean']:>10.4f}  {r['std']:>8.4f}"
        if gt_results_by_coeff:
            g = gt_results_by_coeff[i]
            line += f"  {g['mean']:>10.4f}  {g['std']:>8.4f}"
        print(line)

    # ── Save plots ────────────────────────────────────────────────────────
    save_ranking_plot(results_by_coeff, cfg.output_dir, cfg.model_name, cfg.rollout_length,
                      gt_results_by_coeff=gt_results_by_coeff)
    save_line_plot(results_by_coeff, cfg.output_dir, cfg.model_name, cfg.rollout_length,
                   gt_results_by_coeff=gt_results_by_coeff)
    if gt_results_by_coeff:
        save_comparison_plot(results_by_coeff, gt_results_by_coeff,
                             cfg.output_dir, cfg.model_name, cfg.rollout_length)

    # ── Save JSON ─────────────────────────────────────────────────────────
    summary = {
        "model_name":         cfg.model_name,
        "model_epoch":        cfg.model_epoch,
        "rollout_length":     cfg.rollout_length,
        "n_eval":             cfg.n_eval,
        "policy_model_name":  cfg.policy_model_name,
        "inhibition_coeffs":  cfg.inhibition_coeffs,
        "wm_results": [
            {"inhibition_coeff": r["inhibition_coeff"], "mean": r["mean"], "std": r["std"]}
            for r in results_by_coeff
        ],
    }
    if gt_results_by_coeff:
        summary["gt_results"] = [
            {"inhibition_coeff": r["inhibition_coeff"], "mean": r["mean"], "std": r["std"]}
            for r in gt_results_by_coeff
        ]

    out_json = os.path.join(cfg.output_dir, f"policy_ranking_rl{cfg.rollout_length}.json")
    payload = {"summary": summary, "wm_per_trajectory": results_by_coeff}
    if gt_results_by_coeff:
        payload["gt_per_trajectory"] = gt_results_by_coeff
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
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
    parser.add_argument("--inhibition_coeffs", type=float, nargs="+",
                        default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                        help="List of inhibition coefficients to evaluate (default: 0.0 0.2 0.4 0.6 0.8 1.0)")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of trajectories to evaluate per noise level (default: 100)")
    parser.add_argument("--rollout_length", type=int, default=10,
                        help="WM rollout steps per trajectory (default: 10)")
    parser.add_argument("--output_dir", default="./eval_results/policy_ranking",
                        help="Directory for JSON and plots.")
    parser.add_argument("--run_gym", action="store_true",
                        help="Also run GT gym rollouts and overlay results in plots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy_ranking_main(args)
