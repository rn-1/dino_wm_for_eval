#!/usr/bin/env python3
"""
compare_success.py — VLM-judged success comparison: GT trajectory vs WM trajectory.

For each sampled trajectory:
  1. Run an open-loop WM rollout with GT actions → WM-predicted frames.
  2. Retrieve the corresponding GT frames from the dataset.
  3. Feed the final frame of each trajectory to the VLM judge (Qwen3-VL-4B).
  4. Record whether the GT and WM trajectories are judged as successes.

Outputs (in --output_dir):
  compare_success_rl{N}.json  — per-trajectory + aggregate results
  compare_success_rl{N}.png   — bar / comparison plot
  high_disagreement/          — PNG grids for GT-success / WM-failure cases
  low_loss_examples/          — optional: side-by-side frames for top examples

Usage:
    python compare_success.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name pusht \\
        --n_eval 100 \\
        --rollout_length 10 \\
        --task_prompt "Push the T block to the goal position." \\
        --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/compare_success
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

from vlm_judge import load_vlm_judge, judge_frame
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
# VLM response parsing
# ---------------------------------------------------------------------------

def _parse_vlm_score(raw_response: str) -> float:
    """
    Parse VLM response to extract numerical score in [0, 1].

    VLM is instructed to respond with a number between 0.0 and 1.0.
    Extract the first number found in the response.
    """
    import re
    # Find all floating-point numbers in the response
    numbers = re.findall(r'0?\.\d+|[0-1](?:\.\d+)?', raw_response.strip())
    if numbers:
        try:
            score = float(numbers[0])
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except ValueError:
            pass
    # Fallback: return 0.0
    return 0.0


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
    Save a summary bar chart comparing GT vs WM success rates.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gt_rate = results["gt_mean_score"]
    wm_rate = results["wm_mean_score"]
    agree_rate = results["agreement_rate"]
    n = results["n_eval"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["GT score", "WM score", "Agreement"],
        [gt_rate, wm_rate, agree_rate],
        color=["steelblue", "darkorange", "seagreen"],
        width=0.5,
        edgecolor="white",
    )
    for bar, val in zip(bars, [gt_rate, wm_rate, agree_rate]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_ylim(0, max(1.1, gt_rate * 1.1, wm_rate * 1.1))
    ax.set_ylabel("Score / Rate")
    ax.set_title(f"{model_name} — green-pixel success score (rl={rollout_length}, n={n})")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"compare_success_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved summary plot: {out_path}")


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

    # ── VLM judge ─────────────────────────────────────────────────────────
    vlm_processor, vlm_model = load_vlm_judge(
        model_name=cfg.vlm_model_name,
        device=device,
        hf_cache_dir=os.path.join(cfg.ckpt_base_path, ".hf_cache"),
        local_files_only=cfg.local_files_only,
    )
    if vlm_model is None:
        raise RuntimeError("Could not load VLM judge model.")

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
        raise RuntimeError("World model has no decoder — pixel decoding required for VLM judging.")

    num_hist  = model_cfg.num_hist
    frameskip = model_cfg.frameskip
    rollout_length = cfg.rollout_length

    print(f"\nWorld model '{cfg.model_name}' (epoch={cfg.model_epoch})")
    print(f"  rollout_length={rollout_length}, frameskip={frameskip}, num_hist={num_hist}")
    print(f"  VLM judge: {cfg.vlm_model_name}")

    print(f"Running {cfg.n_eval} evaluations ...\n")

    # ── Rollout output directory ──────────────────────────────────────────
    rollouts_dir = None
    if cfg.save_rollouts:
        rollouts_dir = os.path.join(cfg.output_dir, f"rollouts_rl{rollout_length}")
        os.makedirs(rollouts_dir, exist_ok=True)

    # ── Evaluation loop ───────────────────────────────────────────────────
    records = []
    disagreements = []   # (gt_score, wm_score, gt_frames, wm_frames) where WM significantly underperforms

    for i in range(cfg.n_eval):
        obs, act = sample_valid_trajectory(dset, rollout_length, frameskip, rng)

        # WM open-loop rollout with GT actions
        pred_frames, gt_frames = run_gt_actions_rollout(
            wm, obs, act, frameskip, num_hist, rollout_length, device
        )

        # Save rollout videos and arrays
        if rollouts_dir is not None:
            base = os.path.join(rollouts_dir, f"traj_{i:03d}")
            _save_rollout_outputs(gt_frames, pred_frames, base, fps=cfg.video_fps)

        # VLM success scoring: judge final frames
        gt_final_np = _to_hwc_uint8(gt_frames[-1])
        wm_final_np = _to_hwc_uint8(pred_frames[-1])

        gt_judge = judge_frame(vlm_processor, vlm_model, gt_final_np, cfg.task_prompt, device=device)
        wm_judge = judge_frame(vlm_processor, vlm_model, wm_final_np, cfg.task_prompt, device=device)

        gt_score = _parse_vlm_score(gt_judge["raw_response"])
        wm_score = _parse_vlm_score(wm_judge["raw_response"])
        gt_success = gt_score > cfg.success_threshold
        wm_success = wm_score > cfg.success_threshold
        agree    = abs(gt_score - wm_score) < cfg.agreement_tol

        records.append({
            "traj_idx":    i,
            "gt_score":    gt_score,
            "wm_score":    wm_score,
            "gt_success":  gt_success,
            "wm_success":  wm_success,
            "gt_response": gt_judge["raw_response"],
            "wm_response": wm_judge["raw_response"],
            "agreement":   agree,
        })

        if gt_success and not wm_success:
            disagreements.append((i, gt_frames, pred_frames, gt_score, wm_score))

        print(
            f"  [{i+1:3d}/{cfg.n_eval}]  "
            f"GT: {gt_score:.3f}  WM: {wm_score:.3f}  "
            f"{'AGREE' if agree else 'DISAGREE'}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────
    n = len(records)
    gt_success_rate = sum(r["gt_success"] for r in records) / n
    wm_success_rate = sum(r["wm_success"] for r in records) / n
    agreement_rate  = sum(r["agreement"]  for r in records) / n

    # Rates conditioned on GT outcome
    gt_pos = [r for r in records if r["gt_success"]]
    gt_neg = [r for r in records if not r["gt_success"]]
    wm_given_gt_pos = sum(r["wm_success"] for r in gt_pos) / len(gt_pos) if gt_pos else float("nan")
    wm_given_gt_neg = sum(r["wm_success"] for r in gt_neg) / len(gt_neg) if gt_neg else float("nan")

    print(f"\n=== VLM Success Comparison (rl={rollout_length}) ===")
    print(f"  GT success rate         : {gt_success_rate:.3f}  ({sum(r['gt_success'] for r in records)}/{n})")
    print(f"  WM success rate         : {wm_success_rate:.3f}  ({sum(r['wm_success'] for r in records)}/{n})")
    print(f"  Agreement rate          : {agreement_rate:.3f}")
    print(f"  WM success | GT success : {wm_given_gt_pos:.3f}  (n={len(gt_pos)})")
    print(f"  WM success | GT failure : {wm_given_gt_neg:.3f}  (n={len(gt_neg)})")

    # ── Save disagreement grids ───────────────────────────────────────────
    if disagreements:
        dis_dir = os.path.join(cfg.output_dir, "disagreements_gt_pass_wm_fail")
        os.makedirs(dis_dir, exist_ok=True)
        for rank, (traj_i, gt_f, wm_f, gt_s, wm_s) in enumerate(disagreements[: cfg.n_save_examples]):
            out_img = os.path.join(dis_dir, f"rank{rank+1:02d}_traj{traj_i:03d}.png")
            _save_comparison_grid(
                gt_f, wm_f,
                gt_success=True, wm_success=False,
                out_path=out_img,
                title=f"Traj {traj_i} — GT {gt_s:.2f}  WM {wm_s:.2f}  |  {cfg.task_prompt[:50]}",
            )
        print(f"\nSaved {min(len(disagreements), cfg.n_save_examples)} "
              f"disagreement grids to: {dis_dir}/")

    # ── Save summary plot ─────────────────────────────────────────────────
    agg = {
        "model_name":        cfg.model_name,
        "model_epoch":       cfg.model_epoch,
        "vlm_model_name":    cfg.vlm_model_name,
        "task_prompt":       cfg.task_prompt,
        "n_eval":            n,
        "rollout_length":    rollout_length,
        "frameskip":         frameskip,
        "gt_success_rate":   gt_success_rate,
        "wm_success_rate":   wm_success_rate,
        "agreement_rate":    agreement_rate,
        "wm_success_given_gt_success": wm_given_gt_pos,
        "wm_success_given_gt_failure": wm_given_gt_neg,
    }
    _save_summary_plot(agg, cfg.output_dir, cfg.model_name, rollout_length)

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_json = os.path.join(cfg.output_dir, f"compare_success_rl{rollout_length}.json")
    with open(out_json, "w") as f:
        json.dump({"summary": agg, "trajectories": records}, f, indent=2)
    print(f"Results saved to: {out_json}")
    tee.close()

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare VLM-judged success: GT trajectory vs WM trajectory (GT actions)."
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
    parser.add_argument("--task_prompt", default="Push the T block to the goal position.",
                        help="Language description of the task for the VLM judge.")
    parser.add_argument("--vlm_model_name", default="Qwen/Qwen3-VL-4B-Instruct",
                        help="HuggingFace VLM model ID (default: Qwen/Qwen3-VL-4B-Instruct)")
    parser.add_argument("--local_files_only", action="store_true",
                        help="Do not contact HuggingFace (requires cached weights).")
    parser.add_argument("--output_dir", default="./eval_results/compare_success",
                        help="Directory for JSON, plots, and example images.")
    parser.add_argument("--n_save_examples", type=int, default=10,
                        help="Max disagreement grids to save (default: 10)")
    parser.add_argument("--success_threshold", type=float, default=0.5,
                        help="VLM score above which a frame is judged a success (default: 0.5)")
    parser.add_argument("--agreement_tol", type=float, default=0.2,
                        help="|gt_score - wm_score| below this counts as agreement (default: 0.2)")
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
