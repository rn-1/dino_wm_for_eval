#!/usr/bin/env python3
"""
compare_success_robomimic.py — Robometer-progress comparison: GT vs WM trajectory.

For each sampled trajectory:
  1. Run an open-loop WM rollout with GT actions → WM-predicted frames.
  2. Retrieve the corresponding GT frames from the dataset.
  3. Score both videos end-to-end with Robometer to extract per-frame progress.
  4. Compare progress (final-frame and mean-over-clip) between GT and WM.

The trajectories sampled here are not full point-to-point episodes, so a
binary success judgement on the final frame is not meaningful. Progress is a
continuous signal in [0, 1] that reflects how far along the task the clip has
gotten, which is the appropriate quantity to compare WM-decoded video against
the dataset's GT video.

Outputs (in --output_dir):
  compare_success_rl{N}.json  — per-trajectory + aggregate progress
  compare_success_rl{N}.png   — bar plot (GT vs WM progress, MAE)
  compare_success_scatter_rl{N}.png — per-traj GT vs WM final progress
  disagreements_top/          — PNG grids for trajectories with the largest
                                 |gt_progress_final − wm_progress_final|

Usage:
    python compare_success_robomimic.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name robomimic \\
        --n_eval 100 \\
        --rollout_length 10 \\
        --task_prompt "Pick up the red cube and place it in the bin." \\
        --output_dir /project2/.../eval_results/compare_success_robomimic
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
from PIL import Image as PILImage

try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from use_robometer import load_robometer, infer_robometer
from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from utils import seed


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

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


def _save_rollout_outputs(gt_frames, wm_frames, base_path, fps=4):
    """
    Save GT and WM rollout frames as .mp4 videos and .npy arrays.

    gt_frames / wm_frames : (T, C, H, W) float tensors in [-1, 1]
    base_path             : path prefix; writes <base_path>_gt.{mp4,npy}
                            and <base_path>_wm.{mp4,npy}
    """
    gt_np = _frames_to_uint8(gt_frames)   # (T, H, W, C) uint8
    wm_np = _frames_to_uint8(wm_frames)

    np.save(f"{base_path}_gt.npy", gt_np)
    np.save(f"{base_path}_wm.npy", wm_np)

    if HAVE_IMAGEIO:
        imageio.mimsave(f"{base_path}_gt.mp4", gt_np, fps=fps)
        imageio.mimsave(f"{base_path}_wm.mp4", wm_np, fps=fps)
    else:
        print(f"[Warning] imageio not available; skipping video save for {base_path}")


def _save_comparison_grid(gt_frames, wm_frames, gt_success, wm_success, out_path, title=""):
    """
    Save a two-row PNG grid: GT frames (top) vs WM frames (bottom),
    with success labels in the title.

    gt_frames / wm_frames: (T, C, H, W) float tensors in [-1, 1]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = min(gt_frames.shape[0], wm_frames.shape[0])
    cmap = "gray" if gt_frames.shape[1] == 1 else None
    fig, axes = plt.subplots(2, T, figsize=(2.5 * T, 5))
    if T == 1:
        axes = axes.reshape(2, 1)

    for t in range(T):
        axes[0, t].imshow(_to_hwc_uint8(gt_frames[t]), cmap=cmap, vmin=0, vmax=255)
        axes[0, t].set_title(f"t={t}", fontsize=8)
        axes[0, t].axis("off")
        axes[1, t].imshow(_to_hwc_uint8(wm_frames[t]), cmap=cmap, vmin=0, vmax=255)
        axes[1, t].set_title(f"t={t}", fontsize=8)
        axes[1, t].axis("off")

    gt_label = "GT ✓" if gt_success else "GT ✗"
    wm_label = "WM ✓" if wm_success else "WM ✗"
    axes[0, 0].set_ylabel(gt_label, fontsize=10)
    axes[1, 0].set_ylabel(wm_label, fontsize=10)
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_summary_plot(results, output_dir, model_name, rollout_length):
    """
    Save a summary bar chart comparing GT vs WM Robometer progress.

    Bars: GT progress (final-frame mean), WM progress (final-frame mean),
          GT progress (per-frame mean), WM progress (per-frame mean), MAE.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_final = results["gt_progress_final_mean"]
    wm_final = results["wm_progress_final_mean"]
    gt_mean  = results["gt_progress_mean_of_mean"]
    wm_mean  = results["wm_progress_mean_of_mean"]
    mae_final = results["progress_final_mae"]
    n = results["n_eval"]

    labels = ["GT final", "WM final", "GT mean", "WM mean", "MAE (final)"]
    values = [gt_final, wm_final, gt_mean, wm_mean, mae_final]
    colors = ["steelblue", "darkorange", "skyblue", "navajowhite", "tomato"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylim(0, max(1.05, max(values) * 1.2))
    ax.set_ylabel("Robometer progress (∈ [0, 1])")
    corr = results.get("progress_final_pearson_r")
    corr_str = f", r={corr:.3f}" if corr is not None and np.isfinite(corr) else ""
    ax.set_title(
        f"{model_name} — Robometer progress GT vs WM "
        f"(rl={rollout_length}, n={n}{corr_str})"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"compare_success_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved summary plot: {out_path}")


def _save_scatter_plot(records, output_dir, model_name, rollout_length, pearson_r):
    """Scatter of per-trajectory GT vs WM final-frame progress, with y=x reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.array([r["gt_progress_final"] for r in records])
    wm = np.array([r["wm_progress_final"] for r in records])
    n = len(records)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(gt, wm, s=20, alpha=0.6, color="steelblue", edgecolor="white")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("GT final progress")
    ax.set_ylabel("WM final progress")
    title = f"{model_name} — GT vs WM final progress (rl={rollout_length}, n={n})"
    if pearson_r is not None and np.isfinite(pearson_r):
        title += f"  |  Pearson r={pearson_r:.3f}"
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir,
                            f"compare_success_scatter_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved scatter plot: {out_path}")


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
# Main
# ---------------------------------------------------------------------------

def compare_success_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(cfg.output_dir, "compare_success.log"))

    # ── Robometer judge ───────────────────────────────────────────────────
    print("Loading Robometer ...")
    robometer_bundle = load_robometer(device=device)
    if robometer_bundle[3] is None:
        raise RuntimeError(
            "Robometer model failed to load — required for progress scoring. "
            "Check .hf_cache snapshot and the sibling 'robometer' repo."
        )

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
        raise RuntimeError("World model has no decoder — pixel decoding required for Robometer scoring.")

    num_hist  = model_cfg.num_hist
    frameskip = model_cfg.frameskip
    rollout_length = cfg.rollout_length

    print(f"\nWorld model '{cfg.model_name}' (epoch={cfg.model_epoch})")
    print(f"  rollout_length={rollout_length}, frameskip={frameskip}, num_hist={num_hist}")
    print(f"  Robometer prompt: {cfg.task_prompt!r}")

    print(f"Running {cfg.n_eval} evaluations ...\n")

    # ── Rollout output directory ──────────────────────────────────────────
    rollouts_dir = None
    if cfg.save_rollouts:
        rollouts_dir = os.path.join(cfg.output_dir, f"rollouts_rl{rollout_length}")
        os.makedirs(rollouts_dir, exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────
    records = []

    for i in range(cfg.n_eval):
        obs, act = sample_valid_trajectory(dset, rollout_length, frameskip, rng)

        # WM open-loop rollout with GT actions
        pred_frames, gt_frames = run_gt_actions_rollout(
            wm, obs, act, frameskip, num_hist, rollout_length, device
        )

        if rollouts_dir is not None:
            base = os.path.join(rollouts_dir, f"traj_{i:03d}")
            _save_rollout_outputs(gt_frames, pred_frames, base, fps=cfg.video_fps)

        # Robometer scoring: full-clip progress for GT and WM videos
        gt_u8 = _frames_to_uint8(gt_frames)        # (T, H, W, C) uint8
        wm_u8 = _frames_to_uint8(pred_frames)
        gt_progress, _ = infer_robometer(*robometer_bundle, gt_u8,
                                         cfg.task_prompt, device=device)
        wm_progress, _ = infer_robometer(*robometer_bundle, wm_u8,
                                         cfg.task_prompt, device=device)

        gt_progress_final = float(gt_progress[-1]) if len(gt_progress) else 0.0
        wm_progress_final = float(wm_progress[-1]) if len(wm_progress) else 0.0
        gt_progress_mean  = float(np.mean(gt_progress)) if len(gt_progress) else 0.0
        wm_progress_mean  = float(np.mean(wm_progress)) if len(wm_progress) else 0.0

        progress_diff_final = abs(gt_progress_final - wm_progress_final)
        agree = progress_diff_final < cfg.agreement_tol

        records.append({
            "traj_idx":            i,
            "gt_progress_final":   gt_progress_final,
            "wm_progress_final":   wm_progress_final,
            "gt_progress_mean":    gt_progress_mean,
            "wm_progress_mean":    wm_progress_mean,
            "progress_diff_final": progress_diff_final,
            "agreement":           bool(agree),
            "gt_progress_seq":     [float(x) for x in gt_progress],
            "wm_progress_seq":     [float(x) for x in wm_progress],
        })

        print(
            f"  [{i+1:3d}/{cfg.n_eval}]  "
            f"GT prog: {gt_progress_final:.3f}  WM prog: {wm_progress_final:.3f}  "
            f"|Δ|={progress_diff_final:.3f}  "
            f"{'AGREE' if agree else 'DISAGREE'}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    n = len(records)
    gt_finals = np.array([r["gt_progress_final"] for r in records])
    wm_finals = np.array([r["wm_progress_final"] for r in records])
    gt_means  = np.array([r["gt_progress_mean"]  for r in records])
    wm_means  = np.array([r["wm_progress_mean"]  for r in records])
    diffs     = np.abs(gt_finals - wm_finals)

    progress_final_mae   = float(np.mean(diffs))
    progress_final_rmse  = float(np.sqrt(np.mean(diffs ** 2)))
    agreement_rate       = float(np.mean([r["agreement"] for r in records]))

    if n >= 2 and gt_finals.std() > 0 and wm_finals.std() > 0:
        pearson_r = float(np.corrcoef(gt_finals, wm_finals)[0, 1])
    else:
        pearson_r = float("nan")

    print(f"\n=== Robometer Progress Comparison (rl={rollout_length}) ===")
    print(f"  GT progress  (final mean) : {gt_finals.mean():.3f}  std={gt_finals.std():.3f}")
    print(f"  WM progress  (final mean) : {wm_finals.mean():.3f}  std={wm_finals.std():.3f}")
    print(f"  GT progress  (per-frame)  : {gt_means.mean():.3f}")
    print(f"  WM progress  (per-frame)  : {wm_means.mean():.3f}")
    print(f"  MAE  (final)              : {progress_final_mae:.3f}")
    print(f"  RMSE (final)              : {progress_final_rmse:.3f}")
    print(f"  Pearson r (final)         : {pearson_r:.3f}")
    print(f"  Agreement (|Δ|<{cfg.agreement_tol}): {agreement_rate:.3f}")

    # ── Save grids for trajectories with the largest progress gap ────────
    sorted_records = sorted(
        enumerate(records), key=lambda kv: kv[1]["progress_diff_final"], reverse=True,
    )
    top_disagreements = sorted_records[: cfg.n_save_examples]
    if top_disagreements:
        dis_dir = os.path.join(cfg.output_dir, "disagreements_top")
        os.makedirs(dis_dir, exist_ok=True)
        # Re-sample frames for the top-N — we didn't keep them above to save memory.
        # The idx in records corresponds to the order of the for-loop, not the dataset.
        # We re-run the rollout for these specific records.
        for rank, (rec_idx, rec) in enumerate(top_disagreements):
            # Replay using the same rng path is non-trivial; instead, just sample
            # a fresh trajectory and label clearly. To keep behaviour faithful,
            # we already have the sequences in the records dict for plotting per-
            # frame progress. We omit re-rendering the videos; users can inspect
            # frames via --save_rollouts.
            _save_progress_curve(
                rec["gt_progress_seq"], rec["wm_progress_seq"],
                out_path=os.path.join(dis_dir, f"rank{rank+1:02d}_traj{rec_idx:03d}.png"),
                title=(f"Traj {rec_idx} — GT {rec['gt_progress_final']:.2f}  "
                       f"WM {rec['wm_progress_final']:.2f}  |Δ|={rec['progress_diff_final']:.2f}"),
            )
        print(f"\nSaved top-{len(top_disagreements)} progress-curve plots to: {dis_dir}/")

    # ── Save summary plots ────────────────────────────────────────────────
    agg = {
        "model_name":                cfg.model_name,
        "model_epoch":               cfg.model_epoch,
        "task_prompt":               cfg.task_prompt,
        "n_eval":                    n,
        "rollout_length":            rollout_length,
        "frameskip":                 frameskip,
        "gt_progress_final_mean":    float(gt_finals.mean()),
        "wm_progress_final_mean":    float(wm_finals.mean()),
        "gt_progress_final_std":     float(gt_finals.std()),
        "wm_progress_final_std":     float(wm_finals.std()),
        "gt_progress_mean_of_mean":  float(gt_means.mean()),
        "wm_progress_mean_of_mean":  float(wm_means.mean()),
        "progress_final_mae":        progress_final_mae,
        "progress_final_rmse":       progress_final_rmse,
        "progress_final_pearson_r":  pearson_r,
        "agreement_tol":             cfg.agreement_tol,
        "agreement_rate":            agreement_rate,
    }
    _save_summary_plot(agg, cfg.output_dir, cfg.model_name, rollout_length)
    _save_scatter_plot(records, cfg.output_dir, cfg.model_name, rollout_length, pearson_r)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = os.path.join(cfg.output_dir, f"compare_success_rl{rollout_length}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": agg, "trajectories": records}, f, indent=2)
    print(f"Results saved to: {out_json}")
    tee.close()

    return agg


def _save_progress_curve(gt_seq, wm_seq, out_path, title=""):
    """Plot per-frame Robometer progress for GT vs WM on the same axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt = np.asarray(gt_seq, dtype=float)
    wm = np.asarray(wm_seq, dtype=float)
    T = max(len(gt), len(wm))
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if len(gt):
        ax.plot(x[: len(gt)], gt, marker="o", color="steelblue", linewidth=2, label="GT")
    if len(wm):
        ax.plot(x[: len(wm)], wm, marker="s", color="tomato",   linewidth=2, label="WM")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Robometer progress")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Robometer-judged progress: GT trajectory vs WM trajectory (GT actions)."
    )
    parser.add_argument("--ckpt_base_path", required=True,
                        help="Base directory containing outputs/<model_name>/")
    parser.add_argument("--model_name", required=True,
                        help="WM model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest",
                        help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of trajectories to evaluate (default: 100)")
    parser.add_argument("--rollout_length", type=int, default=10,
                        help="WM rollout steps per trajectory (default: 10)")
    parser.add_argument("--task_prompt",
                        default="Pick up the red cube and place it in the bin.",
                        help="Task prompt fed to Robometer for progress scoring.")
    parser.add_argument("--output_dir", default="./eval_results/compare_success_robomimic",
                        help="Directory for JSON, plots, and example images.")
    parser.add_argument("--n_save_examples", type=int, default=10,
                        help="Max top-disagreement progress-curve plots to save (default: 10)")
    parser.add_argument("--agreement_tol", type=float, default=0.2,
                        help="|gt_progress_final - wm_progress_final| below this counts "
                             "as agreement (default: 0.2)")
    parser.add_argument("--save_rollouts", action="store_true",
                        help="Save GT and WM frames as .mp4 videos and .npy arrays.")
    parser.add_argument("--video_fps", type=int, default=4,
                        help="FPS for saved rollout videos (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_success_main(args)
