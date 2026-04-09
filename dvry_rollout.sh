#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --job-name=rollout

set -euo pipefail

module purge
module load conda

module load apptainer

# IMPORTANT: initialize conda for non-interactive shells
# (this path depends on your cluster's conda module; these are common options)

eval "$(conda shell.bash hook)"


conda activate dino_wm
conda env list

# Sanity checks (these should show your conda env + python3.9)
echo "=== Sanity ==="
which python
python -V
python -c "import sys,site; print('exe:',sys.executable); print('usersite:', site.getusersitepackages()); print('site:', site.getsitepackages())"
echo "pip:"
python -m pip -V


cd /project2/jessetho_1732/rl_eval_wm/dino_wm

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/project2/jessetho_1732/rl_eval_wm/dino_wm/mujoco
export MUJOCO_PATH=/project2/jessetho_1732/rl_eval_wm/dino_wm/mujoco

export DATASET_DIR="/project2/jessetho_1732/rl_eval_wm/dino_wm/data"
export HF_HOME="/project2/jessetho_1732/rl_eval_wm/dino_wm/.hf_cache"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

export WANDB_API_KEY="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"

# python rollout.py --config-name=plan_pusht.yaml

# rm -f /scratch1/rnene/dinoenv.sif
# apptainer build --fakeroot /scratch1/rnene/dinoenv.sif container.def
apptainer exec --nv --fakeroot --writable-tmpfs /scratch1/rnene/dinoenv.sif bash -lc "
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
	export HF_HOME=/project2/jessetho_1732/rl_eval_wm/dino_wm/.hf_cache
	export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
	export TRANSFORMERS_CACHE=\$HF_HOME/transformers
	mkdir -p \"\$HUGGINGFACE_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\"
	cd /project2/jessetho_1732/rl_eval_wm/dino_wm && python rollout.py --config-name=plan_pusht.yaml"
