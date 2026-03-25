#!/usr/bin/env python3
"""
evaluate_wm_robomimic.py — Batched world-model evaluation for a dino_wm model
trained on Robomimic data.

Runs GT-action rollouts in batches, decodes predicted frames via the WM decoder,
and computes pixel-space MSE against ground-truth observations from the validation
set.  Mirrors evaluate_wm.py but provides Robomimic-specific defaults and
sys.path setup.

Usage:
    python robomimic_code/evaluate_wm_robomimic.py \
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
        --model_name robomimic_lift \
        --n_eval 100 \
        --batch_size 8 \
        --rollout_length 10

The script delegates entirely to evaluate_wm.evaluate_wm_main(), so the full
evaluation pipeline (batched rollout, MSE computation, per-step/per-traj plots,
high-loss example images, JSON results) is identical to the PushT pipeline.
"""

import os
import sys

# ── Make dino_wm root importable ─────────────────────────────────────────────
_DINO_WM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _DINO_WM_ROOT not in sys.path:
    sys.path.insert(0, _DINO_WM_ROOT)

# ── Make robomimic_code root importable ──────────────────────────────────────
_ROBOMIMIC_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROBOMIMIC_CODE_ROOT not in sys.path:
    sys.path.insert(0, _ROBOMIMIC_CODE_ROOT)

import argparse

# Import the shared evaluation engine from the main dino_wm package
from evaluate_wm import evaluate_wm_main


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a dino_wm world model trained on Robomimic data via "
            "batched GT-action rollouts and pixel-space MSE."
        )
    )
    p.add_argument(
        "--ckpt_base_path", required=True,
        help="Base directory containing outputs/<model_name>/ "
             "(same layout as the main dino_wm codebase)",
    )
    p.add_argument(
        "--model_name", required=True,
        help="Model sub-directory under outputs/ (e.g. 'robomimic_lift')",
    )
    p.add_argument(
        "--model_epoch", default="latest",
        help="Checkpoint epoch tag (default: latest)",
    )
    p.add_argument(
        "--n_eval", type=int, default=100,
        help="Total number of trajectories to evaluate",
    )
    p.add_argument(
        "--batch_size", type=int, default=8,
        help="Number of trajectories per forward pass",
    )
    p.add_argument(
        "--rollout_length", type=int, default=10,
        help="Number of WM prediction steps per trajectory",
    )
    p.add_argument(
        "--output_dir", default="./eval_results/robomimic",
        help="Directory to write results JSON, plots, and log",
    )
    p.add_argument(
        "--n_save_examples", type=int, default=5,
        help="Number of highest-MSE trajectories to save as PNG comparison grids",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for trajectory sampling",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = evaluate_wm_main(args)

    print("\n=== evaluate_wm_robomimic done ===")
    print(f"  Overall MSE : {results['overall_mse']:.6f}")
    print(f"  Results dir : {args.output_dir}")
