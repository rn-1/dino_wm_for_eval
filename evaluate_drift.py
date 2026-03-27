#!/usr/bin/env python3
"""
evaluate_drift.py — Combined WM frame and action drift analysis.

For each trajectory, runs two rollouts from the same starting condition:
  - WM rollout:  policy conditioned on WM-decoded frames → pred_frames, actions_wm
  - GT rollout:  policy conditioned on GT frames         →              actions_gt

Per-step metrics:
  frame_mse[t]  = MSE(pred_frames[t], gt_frames[t])   — how far WM visuals drift
  action_mse[t] = MSE(actions_wm[t],  actions_gt[t])  — how far policy actions drift

Together these two curves characterise the world model's overall drift: frame_mse
shows where the visual prediction goes wrong; action_mse shows whether that visual
error actually changes what the policy decides to do.

Usage:
    python evaluate_drift.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name pusht \\
        --n_eval 50 \\
        --rollout_length 10
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from lerobot_utils import PolicyWrapper
from utils import seed


# ---------------------------------------------------------------------------
# Logging / visualisation helpers
# ---------------------------------------------------------------------------

class Tee:
    """Mirrors stdout to a log file."""
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


def _to_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1, 1]  →  (H, W, C) or (H, W) numpy array in [0, 1]."""
    img = frame_chw.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy()
    return img[:, :, 0] if img.shape[2] == 1 else img


def save_example_frames(pred_frames, gt_frames, mse_val, out_path, label=""):
    """
    Save a two-row PNG grid: GT frames (top) vs predicted frames (bottom).

    Args:
        pred_frames: (T, C, H, W) float in [-1, 1]
        gt_frames:   (T, C, H, W) float in [-1, 1]
        mse_val:     scalar frame MSE for the trajectory (used in title)
        out_path:    destination file path (.png)
        label:       optional prefix for the figure title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = pred_frames.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(2.5 * T, 5))
    if T == 1:
        axes = axes.reshape(2, 1)

    cmap = "gray" if pred_frames.shape[1] == 1 else None
    for t in range(T):
        step_mse = ((pred_frames[t] - gt_frames[t]) ** 2).mean().item()
        axes[0, t].imshow(_to_hwc(gt_frames[t]), cmap=cmap, vmin=0, vmax=1)
        axes[0, t].set_title(f"t={t}", fontsize=8)
        axes[0, t].axis("off")
        axes[1, t].imshow(_to_hwc(pred_frames[t]), cmap=cmap, vmin=0, vmax=1)
        axes[1, t].set_title(f"{step_mse:.4f}", fontsize=7)
        axes[1, t].axis("off")

    axes[0, 0].set_ylabel("GT", fontsize=10)
    axes[1, 0].set_ylabel("Pred", fontsize=10)
    fig.suptitle(f"{label}  frame MSE={mse_val:.5f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_drift_plots(per_step_frame_mse, per_step_action_mse,
                      all_frame_mse, all_action_mse,
                      output_dir, model_name, rollout_length):
    """
    Save three PNGs:
      1. drift_per_step_rl{N}.png      — frame MSE + action MSE vs prediction step
      2. drift_frame_traj_rl{N}.png    — per-trajectory frame MSE distribution
      3. drift_action_traj_rl{N}.png   — per-trajectory action MSE distribution
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_frame  = list(range(len(per_step_frame_mse)))
    steps_action = list(range(len(per_step_action_mse)))
    frame_vals   = per_step_frame_mse.tolist()
    action_vals  = per_step_action_mse.tolist()

    # -- Plot 1: per-step frame MSE and action MSE on shared x-axis ----------
    fig, (ax_f, ax_a) = plt.subplots(1, 2, figsize=(12, 4))

    ax_f.plot(steps_frame, frame_vals, marker="o", linewidth=1.5, markersize=4, color="steelblue")
    ax_f.set_xlabel("Prediction step")
    ax_f.set_ylabel("Mean frame MSE")
    ax_f.set_title(f"{model_name} — Frame MSE vs step")
    ax_f.grid(True, linestyle="--", alpha=0.5)

    ax_a.plot(steps_action, action_vals, marker="s", linewidth=1.5, markersize=4, color="darkorange")
    ax_a.set_xlabel("Prediction step")
    ax_a.set_ylabel("Mean action MSE")
    ax_a.set_title(f"{model_name} — Action MSE vs step")
    ax_a.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"Drift degradation over horizon (rl={rollout_length})", fontsize=11)
    fig.tight_layout()
    out1 = os.path.join(output_dir, f"drift_per_step_rl{rollout_length}.png")
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"Saved per-step drift plot    : {out1}")

    # -- Plot 2: per-trajectory frame MSE distribution (sorted + histogram) --
    traj_frame_vals = sorted([t.mean().item() for t in all_frame_mse])
    mean_frame = sum(traj_frame_vals) / len(traj_frame_vals)
    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
    ax_s.plot(traj_frame_vals, linewidth=1.2, color="steelblue")
    ax_s.axhline(mean_frame, color="red", linestyle="--", linewidth=1,
                 label=f"mean={mean_frame:.5f}")
    ax_s.set_xlabel("Trajectory rank (sorted)")
    ax_s.set_ylabel("Frame MSE")
    ax_s.set_title("Per-traj frame MSE (sorted)")
    ax_s.legend(fontsize=8)
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_h.hist(traj_frame_vals, bins=20, edgecolor="white", linewidth=0.4, color="steelblue")
    ax_h.axvline(mean_frame, color="red", linestyle="--", linewidth=1,
                 label=f"mean={mean_frame:.5f}")
    ax_h.set_xlabel("Frame MSE")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Per-traj frame MSE dist.")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(f"{model_name} — per-trajectory frame MSE (rl={rollout_length})", fontsize=11)
    fig.tight_layout()
    out2 = os.path.join(output_dir, f"drift_frame_traj_rl{rollout_length}.png")
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Saved per-traj frame plot    : {out2}")

    # -- Plot 3: per-trajectory action MSE distribution -----------------------
    traj_action_vals = sorted([t.mean().item() for t in all_action_mse])
    mean_action = sum(traj_action_vals) / len(traj_action_vals)
    fig, (ax_s, ax_h) = plt.subplots(1, 2, figsize=(11, 4))
    ax_s.plot(traj_action_vals, linewidth=1.2, color="darkorange")
    ax_s.axhline(mean_action, color="red", linestyle="--", linewidth=1,
                 label=f"mean={mean_action:.5f}")
    ax_s.set_xlabel("Trajectory rank (sorted)")
    ax_s.set_ylabel("Action MSE")
    ax_s.set_title("Per-traj action MSE (sorted)")
    ax_s.legend(fontsize=8)
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_h.hist(traj_action_vals, bins=20, edgecolor="white", linewidth=0.4, color="darkorange")
    ax_h.axvline(mean_action, color="red", linestyle="--", linewidth=1,
                 label=f"mean={mean_action:.5f}")
    ax_h.set_xlabel("Action MSE")
    ax_h.set_ylabel("Count")
    ax_h.set_title("Per-traj action MSE dist.")
    ax_h.legend(fontsize=8)
    ax_h.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle(f"{model_name} — per-trajectory action MSE (rl={rollout_length})", fontsize=11)
    fig.tight_layout()
    out3 = os.path.join(output_dir, f"drift_action_traj_rl{rollout_length}.png")
    fig.savefig(out3, dpi=120)
    plt.close(fig)
    print(f"Saved per-traj action plot   : {out3}")


# ---------------------------------------------------------------------------
# Action sanity checks
# ---------------------------------------------------------------------------

def _check_actions_not_constant(actions: torch.Tensor, label: str = ""):
    """
    Warn if all actions in a rollout are identical (per-dim std == 0).
    actions: (T, action_dim)
    """
    if actions.shape[0] < 2:
        return
    std_per_dim = actions.std(dim=0)
    if (std_per_dim == 0).all():
        print(f"  WARNING [{label}]: all {actions.shape[0]} actions are identical — "
              f"policy may be returning a cached/dummy action. "
              f"actions[0]={actions[0].numpy()}")
    elif (std_per_dim < 1e-8).any():
        zero_dims = (std_per_dim < 1e-8).nonzero(as_tuple=True)[0].tolist()
        print(f"  WARNING [{label}]: action dims {zero_dims} are constant across all steps.")


# ---------------------------------------------------------------------------
# Shared helpers (same preprocessing as rollout.py:477-485)
# ---------------------------------------------------------------------------

def _img_for_policy(frame_chw: torch.Tensor, policy_img_size: int) -> torch.Tensor:
    """(C,H,W) in [-1,1]  →  (1,C,size,size) in [0,1]."""
    img_01 = (frame_chw.clamp(-1.0, 1.0) + 1.0) / 2.0
    return F.interpolate(
        img_01.unsqueeze(0),
        size=(policy_img_size, policy_img_size),
        mode="bilinear",
        align_corners=False,
    )


def reset_policy(policy: PolicyWrapper):
    """Reset the policy's internal observation queue between rollouts."""
    try:
        policy.policy.reset()
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Single-rollout functions (mirrors rollout.py:467-509)
# ---------------------------------------------------------------------------

def run_wm_rollout(wm, policy, obs, frameskip, n_past, rollout_length,
                   policy_img_size, device, verbose=False):
    """
    Closed-loop rollout: policy conditioned on WM-decoded frames.

    Returns:
        pred_frames : (rollout_length+1, C, H, W)  decoded WM frames in [-1, 1]
        actions_wm  : (rollout_length, action_dim)  policy actions at each step
    """
    wm_action_dim = wm.action_encoder.in_chans

    obs_0 = {k: v[:n_past].unsqueeze(0).to(device) for k, v in obs.items()}
    act_0 = torch.zeros(1, n_past, wm_action_dim, device=device)

    decoded_frames, actions = [], []

    with torch.no_grad():
        z = wm.encode(obs_0, act_0)

        for t in range(rollout_length):
            z_obs_cur, _ = wm.separate_emb(z[:, -1:])
            decoded, _   = wm.decode_obs(z_obs_cur)
            img = decoded["visual"][0, 0]          # (C, H, W) in [-1, 1]
            decoded_frames.append(img.cpu())

            img_policy  = _img_for_policy(img, policy_img_size)
            state_idx   = min(t + n_past - 1, obs["proprio"].shape[0] - 1)
            state_input = obs["proprio"][state_idx].unsqueeze(0).to(device)

            action = policy.predict(observation=state_input, image=img_policy)
            action_cpu = action.squeeze(0).cpu()
            actions.append(action_cpu)    # (action_dim,)
            if verbose:
                print(f"      [WM t={t}] action={action_cpu.numpy()}")

            action_grouped = action.repeat(1, frameskip).unsqueeze(1)
            z_pred = wm.predict(z[:, -wm.num_hist:])
            z_new  = z_pred[:, -1:, ...]
            z_new  = wm.replace_actions_from_z(z_new, action_grouped)
            z      = torch.cat([z, z_new], dim=1)

        # Final frame
        z_obs_last, _ = wm.separate_emb(z[:, -1:])
        decoded_last, _ = wm.decode_obs(z_obs_last)
        decoded_frames.append(decoded_last["visual"][0, 0].cpu())

    actions_t = torch.stack(actions)   # (rollout_length, action_dim)
    _check_actions_not_constant(actions_t, label="WM")
    return (
        torch.stack(decoded_frames),   # (rollout_length+1, C, H, W)
        actions_t,
    )


def run_gt_rollout(policy, obs, n_past, rollout_length, policy_img_size, device, verbose=False):
    """
    Policy conditioned on GT frames at each step.

    Returns:
        actions_gt : (rollout_length, action_dim)
    """
    actions = []
    with torch.no_grad():
        for t in range(rollout_length):
            frame_idx   = min(t + n_past - 1, obs["visual"].shape[0] - 1)
            img         = obs["visual"][frame_idx].to(device)
            img_policy  = _img_for_policy(img, policy_img_size)

            state_idx   = min(t + n_past - 1, obs["proprio"].shape[0] - 1)
            state_input = obs["proprio"][state_idx].unsqueeze(0).to(device)

            action = policy.predict(observation=state_input, image=img_policy)
            action_cpu = action.squeeze(0).cpu()
            actions.append(action_cpu)
            if verbose:
                print(f"      [GT t={t}] action={action_cpu.numpy()}")

    actions_t = torch.stack(actions)   # (rollout_length, action_dim)
    _check_actions_not_constant(actions_t, label="GT")
    return actions_t


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def evaluate_trajectory(wm, policy, obs, frameskip, n_past, rollout_length,
                        policy_img_size, device, verbose=False):
    """
    Run WM rollout and GT rollout for one trajectory and compute per-step drift.

    Returns dict:
        frame_mse   : (rollout_length+1,) — pixel MSE between WM frames and GT frames
        action_mse  : (rollout_length,)   — MSE between WM-conditioned and GT-conditioned actions
        actions_wm  : (rollout_length, action_dim) — raw WM policy actions
        actions_gt  : (rollout_length, action_dim) — raw GT policy actions
        pred_frames : (T, C, H, W) CPU tensor — decoded WM frames in [-1, 1]
        gt_frames   : (T, C, H, W) CPU tensor — aligned GT frames in [-1, 1]
    """
    # WM rollout
    reset_policy(policy)
    pred_frames, actions_wm = run_wm_rollout(
        wm, policy, obs, frameskip, n_past, rollout_length, policy_img_size, device,
        verbose=verbose,
    )

    # GT rollout
    reset_policy(policy)
    actions_gt = run_gt_rollout(
        policy, obs, n_past, rollout_length, policy_img_size, device, verbose=verbose,
    )

    # Frame MSE: align GT frames to the predicted window
    T_out    = pred_frames.shape[0]   # rollout_length + 1
    gt_start = n_past - 1
    gt_frames = obs["visual"][gt_start : gt_start + T_out]
    min_T     = min(pred_frames.shape[0], gt_frames.shape[0])
    frame_mse = ((pred_frames[:min_T] - gt_frames[:min_T]) ** 2).mean(dim=[1, 2, 3])

    # Action MSE: element-wise squared error averaged over action dims
    action_mse = ((actions_wm - actions_gt) ** 2).mean(dim=-1)   # (rollout_length,)

    return {
        "frame_mse":   frame_mse,
        "action_mse":  action_mse,
        "actions_wm":  actions_wm,
        "actions_gt":  actions_gt,
        "pred_frames": pred_frames[:min_T].cpu(),
        "gt_frames":   gt_frames[:min_T].cpu(),
    }


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def sample_valid_trajectory(dset, rollout_length, frameskip, n_past, rng):
    min_frames = (n_past + rollout_length) * frameskip
    indices = list(range(len(dset)))
    rng.shuffle(indices)
    for idx in indices:
        obs, *_ = dset[idx]
        if obs["visual"].shape[0] >= min_frames:
            return obs
    raise RuntimeError(
        f"No trajectory in dataset has >= {min_frames} raw frames "
        f"(n_past={n_past}, rollout_length={rollout_length}, frameskip={frameskip})."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_drift_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(cfg.output_dir, "drift_eval.log"))

    # ── Load world model ──────────────────────────────────────────────────
    model_path = Path(cfg.ckpt_base_path) / "outputs" / cfg.model_name
    with open(model_path / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    _, dset = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset["valid"]
    print(f"Validation dataset: {len(dset)} trajectories")

    model_ckpt = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    wm = _load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    if wm.decoder is None:
        raise RuntimeError("World model has no decoder — pixel decoding required.")

    # ── Load policy ───────────────────────────────────────────────────────
    n_action_steps = getattr(cfg, "n_action_steps", None)
    policy = PolicyWrapper(model_name=cfg.policy_model_name, n_action_steps=n_action_steps)
    print(f"  policy n_action_steps={policy.policy.config.n_action_steps} "
          f"(actions served per diffusion inference pass)")

    n_past         = model_cfg.num_hist
    frameskip      = model_cfg.frameskip
    rollout_length = cfg.rollout_length
    policy_img_size = cfg.policy_img_size

    print(f"\nEvaluating '{cfg.model_name}' (epoch={cfg.model_epoch}) "
          f"with policy '{cfg.policy_model_name}'")
    print(f"  n_eval={cfg.n_eval}, rollout_length={rollout_length}, "
          f"frameskip={frameskip}, n_past={n_past}")

    # ── Evaluation loop ───────────────────────────────────────────────────
    all_frame_mse  = []   # list of (rollout_length+1,) tensors
    all_action_mse = []   # list of (rollout_length,) tensors
    # For saving high-loss examples: list of (frame_mse_float, pred_frames, gt_frames)
    all_examples = []

    verbose_first = getattr(cfg, "verbose_actions", False)
    for traj_idx in range(cfg.n_eval):
        obs = sample_valid_trajectory(dset, rollout_length, frameskip, n_past, rng)
        # Print per-step action values for the first trajectory (or all if --verbose_actions)
        verbose_this = verbose_first or (traj_idx == 0)
        metrics = evaluate_trajectory(
            wm, policy, obs, frameskip, n_past, rollout_length, policy_img_size, device,
            verbose=verbose_this,
        )
        all_frame_mse.append(metrics["frame_mse"])
        all_action_mse.append(metrics["action_mse"])
        all_examples.append((
            metrics["frame_mse"].mean().item(),
            metrics["pred_frames"],
            metrics["gt_frames"],
        ))

        act_wm = metrics["actions_wm"]   # (T, action_dim)
        act_gt = metrics["actions_gt"]
        print(f"  [{traj_idx+1:3d}/{cfg.n_eval}]  "
              f"frame MSE: {metrics['frame_mse'].mean():.5f}  "
              f"action MSE: {metrics['action_mse'].mean():.5f}  "
              f"act_wm std={act_wm.std():.4f} range=[{act_wm.min():.3f},{act_wm.max():.3f}]  "
              f"act_gt std={act_gt.std():.4f} range=[{act_gt.min():.3f},{act_gt.max():.3f}]")

    # ── Aggregate ─────────────────────────────────────────────────────────
    per_step_frame_mse  = torch.stack(all_frame_mse).mean(dim=0)    # (rollout_length+1,)
    per_step_action_mse = torch.stack(all_action_mse).mean(dim=0)   # (rollout_length,)
    overall_frame_mse   = per_step_frame_mse.mean().item()
    overall_action_mse  = per_step_action_mse.mean().item()

    print(f"\n=== Drift Evaluation Results ===")
    print(f"  Overall frame MSE    : {overall_frame_mse:.6f}")
    print(f"  Per-step frame MSE   : {[f'{v:.5f}' for v in per_step_frame_mse.tolist()]}")
    print(f"  Overall action MSE   : {overall_action_mse:.6f}")
    print(f"  Per-step action MSE  : {[f'{v:.5f}' for v in per_step_action_mse.tolist()]}")

    # ── Save plots ────────────────────────────────────────────────────────
    _save_drift_plots(
        per_step_frame_mse, per_step_action_mse,
        all_frame_mse, all_action_mse,
        cfg.output_dir, cfg.model_name, rollout_length,
    )

    # ── Save high-loss example images ─────────────────────────────────────
    if cfg.n_save_examples > 0:
        examples_dir = os.path.join(cfg.output_dir, "high_loss_examples")
        os.makedirs(examples_dir, exist_ok=True)
        top_examples = sorted(all_examples, key=lambda x: x[0], reverse=True)
        for rank, (mse_val, pred_f, gt_f) in enumerate(top_examples[: cfg.n_save_examples]):
            out_img = os.path.join(examples_dir, f"rank{rank + 1:02d}_framemse{mse_val:.5f}.png")
            save_example_frames(pred_f, gt_f, mse_val, out_img,
                                label=f"{cfg.model_name} rank#{rank + 1}")
        print(f"\nSaved {min(cfg.n_save_examples, len(top_examples))} "
              f"high-loss example images to: {examples_dir}/")

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "model_name":           cfg.model_name,
        "model_epoch":          cfg.model_epoch,
        "policy_model_name":    cfg.policy_model_name,
        "n_eval":               cfg.n_eval,
        "rollout_length":       rollout_length,
        "frameskip":            frameskip,
        "n_past":               n_past,
        "overall_frame_mse":    overall_frame_mse,
        "per_step_frame_mse":   per_step_frame_mse.tolist(),
        "overall_action_mse":   overall_action_mse,
        "per_step_action_mse":  per_step_action_mse.tolist(),
        "per_traj_frame_mse":   [t.mean().item() for t in all_frame_mse],
        "per_traj_action_mse":  [t.mean().item() for t in all_action_mse],
    }
    out_path = os.path.join(cfg.output_dir, f"drift_eval_results_rl{rollout_length}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Log saved to:     {os.path.join(cfg.output_dir, 'drift_eval.log')}")
    tee.close()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate WM frame drift and policy action drift in a single run."
    )
    parser.add_argument("--ckpt_base_path", required=True,
                        help="Base directory containing outputs/<model_name>/")
    parser.add_argument("--model_name", required=True,
                        help="Model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest",
                        help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--policy_model_name", default="lerobot/diffusion_pusht",
                        help="LeRobot policy model name or local path")
    parser.add_argument("--n_eval", type=int, default=50,
                        help="Number of trajectories to evaluate (default: 50)")
    parser.add_argument("--rollout_length", type=int, default=10,
                        help="Steps to roll out per trajectory (default: 10)")
    parser.add_argument("--policy_img_size", type=int, default=96,
                        help="Image size for policy input in pixels (default: 96)")
    parser.add_argument("--output_dir", default="./eval_results",
                        help="Directory to save results JSON (default: ./eval_results)")
    parser.add_argument("--n_save_examples", type=int, default=5,
                        help="Number of highest-frame-MSE trajectories to save as PNG grids (0=none)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Override policy n_action_steps (actions per diffusion inference pass). "
                             "Use 1 for fresh inference at every step. Default: use policy's trained value.")
    parser.add_argument("--verbose_actions", action="store_true",
                        help="Print per-step action values for every trajectory (default: first only)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_drift_main(args)
