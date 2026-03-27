#!/usr/bin/env python3
"""
evaluate_wm.py — Batched evaluation of world model prediction quality vs. ground truth.

Runs GT-action rollouts in batches, decodes predicted frames, and computes pixel-space
MSE against ground truth observations from the validation dataset.

Usage:
    python evaluate_wm.py \
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
        --model_name pusht \
        --n_eval 100 \
        --batch_size 8 \
        --rollout_length 10
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

# Ensure the dino_wm package directory is on sys.path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
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
        mse_val:     scalar MSE for the trajectory (used in title)
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
    fig.suptitle(f"{label}  MSE={mse_val:.5f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def sample_batch(dset, batch_size, rollout_length, frameskip, num_hist, device, rng):
    """
    Sample `batch_size` trajectories from the dataset that each have at least
    `rollout_length` world-model steps after frameskip subsampling.

    Returns:
        obs_0_batch:     dict {"visual": (B, num_hist, C, H, W),
                               "proprio": (B, num_hist, D)}
        actions_batch:   (B, rollout_length, action_dim * frameskip)
        gt_frames_batch: (B, rollout_length+1, C, H, W)  in [-1, 1]
    """
    min_raw_len = rollout_length * frameskip + 1  # raw frames required

    obs_0_list = []
    actions_list = []
    gt_frames_list = []

    # Build list of valid indices (trajectories with enough frames)
    valid_indices = [
        i for i in range(len(dset))
        if dset[i][0]["visual"].shape[0] >= min_raw_len
    ]
    if len(valid_indices) == 0:
        raise RuntimeError(
            f"No trajectories in dataset have >= {min_raw_len} raw frames "
            f"(rollout_length={rollout_length}, frameskip={frameskip}). "
            f"Dataset has {len(dset)} trajectories total."
        )

    # Sample with replacement if dataset is smaller than batch_size
    collected = 0
    while collected < batch_size:
        idx = valid_indices[rng.randint(len(valid_indices))]

        obs, act, state, _ = dset[idx]

        # Subsample observations at frameskip intervals → (rollout_length+1, ...)
        obs_sub = {
            k: obs[k][0 : rollout_length * frameskip + 1 : frameskip]
            for k in obs.keys()
        }

        # Subsample and group actions → (rollout_length, action_dim * frameskip)
        act_sub = act[0 : rollout_length * frameskip]
        act_sub = rearrange(act_sub, "(h f) d -> h (f d)", f=frameskip)

        obs_0 = {k: obs_sub[k][:num_hist] for k in obs_sub.keys()}  # context frames
        gt_frames = obs_sub["visual"]  # (rollout_length+1, C, H, W) in [-1, 1]

        obs_0_list.append(obs_0)
        actions_list.append(act_sub)
        gt_frames_list.append(gt_frames)
        collected += 1

    # Stack into batch tensors and move to device
    obs_0_batch = {
        k: torch.stack([o[k] for o in obs_0_list], dim=0).to(device)
        for k in obs_0_list[0].keys()
    }
    actions_batch = torch.stack(actions_list, dim=0).to(device)
    gt_frames_batch = torch.stack(gt_frames_list, dim=0).to(device)

    return obs_0_batch, actions_batch, gt_frames_batch


def evaluate_batch(wm, obs_0_batch, actions_batch, gt_frames_batch):
    """
    Run one batched GT rollout and compute per-step pixel-space MSE vs ground truth.

    wm.rollout returns z_obses with T = rollout_length + 1 frames (initial context
    frames followed by predicted frames). We align GT to the decoded prediction length.

    Returns:
        per_step_mse: (T,) tensor — MSE averaged over batch and spatial dims per step
        per_traj_mse: (B,) tensor — MSE averaged over time and spatial dims per trajectory
        pred_frames:  (B, T, C, H, W) CPU tensor — decoded WM frames in [-1, 1]
        gt_aligned:   (B, T, C, H, W) CPU tensor — aligned GT frames in [-1, 1]
    """
    z_obses, _ = wm.rollout(obs_0_batch, actions_batch)
    pred_obs, _ = wm.decode_obs(z_obses)
    pred_frames = pred_obs["visual"]  # (B, T, C, H, W) in [-1, 1]

    # Align GT to predicted length (both should be rollout_length+1, but guard for safety)
    T = pred_frames.shape[1]
    gt_aligned = gt_frames_batch[:, -T:]  # (B, T, C, H, W)

    sq_err = (pred_frames - gt_aligned) ** 2   # (B, T, C, H, W)
    per_step_mse = sq_err.mean(dim=[0, 2, 3, 4])   # (T,)
    per_traj_mse = sq_err.mean(dim=[1, 2, 3, 4])   # (B,)

    return per_step_mse, per_traj_mse, pred_frames.cpu(), gt_aligned.cpu()


def _save_mse_plots(per_step_mse, per_traj_mse, output_dir, model_name, rollout_length):
    """
    Save two MSE plots to output_dir:
      1. per_step_mse.png  — mean MSE at each prediction step (horizon plot)
      2. per_traj_mse.png  — MSE distribution across trajectories (sorted + histogram)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = list(range(len(per_step_mse)))
    vals = per_step_mse.tolist()

    # -- Plot 1: per-step MSE over prediction horizon -------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, vals, marker="o", linewidth=1.5, markersize=4)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Mean MSE")
    ax.set_title(f"{model_name} — MSE vs prediction horizon")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out1 = os.path.join(output_dir, f"per_step_mse_rl{rollout_length}.png")
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"Saved per-step MSE plot  : {out1}")

    # -- Plot 2: per-trajectory MSE distribution (sorted + histogram) ---------
    traj_vals = sorted(per_traj_mse.tolist())
    fig, (ax_sorted, ax_hist) = plt.subplots(1, 2, figsize=(11, 4))

    ax_sorted.plot(traj_vals, linewidth=1.2)
    ax_sorted.axhline(float(per_traj_mse.mean()), color="red",
                      linestyle="--", linewidth=1, label=f"mean={per_traj_mse.mean():.5f}")
    ax_sorted.set_xlabel("Trajectory rank (sorted by MSE)")
    ax_sorted.set_ylabel("MSE")
    ax_sorted.set_title("Per-trajectory MSE (sorted)")
    ax_sorted.legend(fontsize=8)
    ax_sorted.grid(True, linestyle="--", alpha=0.5)

    ax_hist.hist(traj_vals, bins=20, edgecolor="white", linewidth=0.4)
    ax_hist.axvline(float(per_traj_mse.mean()), color="red",
                    linestyle="--", linewidth=1, label=f"mean={per_traj_mse.mean():.5f}")
    ax_hist.set_xlabel("MSE")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Per-trajectory MSE distribution")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"{model_name} — trajectory MSE", fontsize=11)
    fig.tight_layout()
    out2 = os.path.join(output_dir, f"per_traj_mse_rl{rollout_length}.png")
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Saved per-traj MSE plot  : {out2}")


def evaluate_wm_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(cfg.output_dir, "wm_eval.log"))

    # ── Load model config ─────────────────────────────────────────────────────
    model_path = Path(cfg.ckpt_base_path) / "outputs" / cfg.model_name
    with open(model_path / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    # ── Load validation dataset ───────────────────────────────────────────────
    _, dset = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset["valid"]
    print(f"Validation dataset size: {len(dset)} trajectories")

    # ── Load world model checkpoint ───────────────────────────────────────────
    model_ckpt = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    wm = _load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    if wm.decoder is None:
        raise RuntimeError(
            "World model has no decoder — pixel-space MSE requires a decoder. "
            "Check that has_decoder=True in the model config."
        )

    num_hist = model_cfg.num_hist
    frameskip = model_cfg.frameskip
    rollout_length = cfg.rollout_length
    batch_size = cfg.batch_size
    n_eval = cfg.n_eval

    print(f"\nEvaluating '{cfg.model_name}' (epoch={cfg.model_epoch})")
    print(f"  n_eval={n_eval}, batch_size={batch_size}, "
          f"rollout_length={rollout_length}, frameskip={frameskip}, num_hist={num_hist}")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_per_step_mse = []
    all_per_traj_mse = []
    # For saving high-loss examples: list of (mse_float, pred_frames_T, gt_frames_T)
    all_examples = []
    n_batches = (n_eval + batch_size - 1) // batch_size
    collected = 0

    for batch_idx in range(n_batches):
        this_batch = min(batch_size, n_eval - collected)

        obs_0_batch, actions_batch, gt_frames_batch = sample_batch(
            dset, this_batch, rollout_length, frameskip, num_hist, device, rng
        )

        with torch.no_grad():
            per_step_mse, per_traj_mse, pred_frames, gt_aligned = evaluate_batch(
                wm, obs_0_batch, actions_batch, gt_frames_batch
            )

        all_per_step_mse.append(per_step_mse.cpu())
        all_per_traj_mse.append(per_traj_mse.cpu())

        for i in range(this_batch):
            all_examples.append((
                per_traj_mse[i].item(),
                pred_frames[i],   # (T, C, H, W)
                gt_aligned[i],    # (T, C, H, W)
            ))

        collected += this_batch
        running_mean = torch.cat(all_per_traj_mse).mean().item()
        print(f"  Batch {batch_idx + 1:3d}/{n_batches} — "
              f"batch MSE: {per_traj_mse.mean().item():.6f}  "
              f"running mean: {running_mean:.6f}")

    # ── Aggregate results ─────────────────────────────────────────────────────
    # Average per-step MSE across batches (weighted equally since all batches same size
    # except possibly the last; close enough for reporting)
    per_step_mse_all = torch.stack(all_per_step_mse).mean(dim=0)  # (T,)
    per_traj_mse_all = torch.cat(all_per_traj_mse)                # (n_eval,)
    overall_mse = per_traj_mse_all.mean().item()

    print(f"\n=== World Model Evaluation Results ===")
    print(f"  Model          : {cfg.model_name} (epoch={cfg.model_epoch})")
    print(f"  Trajectories   : {collected}")
    print(f"  Rollout length : {rollout_length} steps")
    print(f"  Overall MSE    : {overall_mse:.6f}")
    print(f"  Per-step MSE   : {[f'{v:.6f}' for v in per_step_mse_all.tolist()]}")

    # ── Save high-loss example images ─────────────────────────────────────────
    if cfg.n_save_examples > 0:
        examples_dir = os.path.join(cfg.output_dir, "high_loss_examples")
        os.makedirs(examples_dir, exist_ok=True)
        top_examples = sorted(all_examples, key=lambda x: x[0], reverse=True)
        for rank, (mse_val, pred_f, gt_f) in enumerate(top_examples[: cfg.n_save_examples]):
            out_img = os.path.join(examples_dir, f"rank{rank + 1:02d}_mse{mse_val:.5f}.png")
            save_example_frames(pred_f, gt_f, mse_val, out_img,
                                label=f"{cfg.model_name} rank#{rank + 1}")
        print(f"\nSaved {min(cfg.n_save_examples, len(top_examples))} "
              f"high-loss example images to: {examples_dir}/")

    # ── Save MSE plots ────────────────────────────────────────────────────────
    _save_mse_plots(per_step_mse_all, per_traj_mse_all, cfg.output_dir, cfg.model_name, rollout_length)

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "model_name": cfg.model_name,
        "model_epoch": cfg.model_epoch,
        "n_eval": collected,
        "rollout_length": rollout_length,
        "batch_size": batch_size,
        "frameskip": frameskip,
        "num_hist": num_hist,
        "overall_mse": overall_mse,
        "per_step_mse": per_step_mse_all.tolist(),
        "per_traj_mse": per_traj_mse_all.tolist(),
    }
    out_path = os.path.join(cfg.output_dir, f"wm_eval_results_rl{rollout_length}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Log saved to:     {os.path.join(cfg.output_dir, 'wm_eval.log')}")
    tee.close()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate world model prediction quality via batched GT rollouts."
    )
    parser.add_argument(
        "--ckpt_base_path", required=True,
        help="Base directory containing outputs/<model_name>/  (same as rollout.py's ckpt_base_path)"
    )
    parser.add_argument("--model_name", required=True, help="Model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest", help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--n_eval", type=int, default=100, help="Total trajectories to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Trajectories per forward pass")
    parser.add_argument("--rollout_length", type=int, default=10, help="Steps to predict per trajectory")
    parser.add_argument("--output_dir", default="./eval_results", help="Directory to save results JSON")
    parser.add_argument("--n_save_examples", type=int, default=5,
                        help="Number of highest-MSE trajectories to save as PNG grids (0=none)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_wm_main(args)
