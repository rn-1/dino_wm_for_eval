#!/bin/bash

export HYDRA_FULL_ERROR=1

sbatch dvry_rollout.sh
sbatch dvry_evaluate_drift.sh
sbatch dvry_evaluate_policy.sh
sbatch dvry_evaluate_wm.sh
