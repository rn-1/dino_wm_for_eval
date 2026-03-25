#!/bin/bash
# Submit all robomimic evaluation jobs to SLURM.
# Run from the dino_wm root directory:
#
#   bash robomimic_code/run_jobs_robomimic.sh
#
# Jobs submitted:
#   rollout_robomimic          — open-loop WM rollout with GT actions
#   evaluate_wm_robomimic      — batched pixel-MSE WM evaluation (rl 10/20/40)
#   rollout_robomimic_policy   — closed-loop WM + policy rollout with Robometer (rl 10/20/40)

export HYDRA_FULL_ERROR=1

sbatch robomimic_code/rollout_robomimic.sh
sbatch robomimic_code/evaluate_wm_robomimic.sh
sbatch robomimic_code/rollout_robomimic_policy.sh
