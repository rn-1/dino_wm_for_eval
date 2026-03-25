#!/bin/bash

export HYDRA_FULL_ERROR=1

sbatch rollout.sh
sbatch evaluate_drift.sh
sbatch evaluate_policy.sh
sbatch evaluate_wm.sh
