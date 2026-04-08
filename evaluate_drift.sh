#!/bin/bash

#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --job-name=evaluate_drift
#SBATCH --output=evaluate_drift.out
#SBATCH --error=evaluate_drift.err

set -euo pipefail

module purge
module load conda
module load apptainer

eval "$(conda shell.bash hook)"

conda activate dino_wm
conda env list

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
export TORCH_HOME="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

export WANDB_API_KEY="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"

# rm -f /scratch1/rnene/dinoenv.sif
# apptainer build --fakeroot /scratch1/rnene/dinoenv.sif container.def
apptainer exec --nv --fakeroot --writable-tmpfs --bind /apps:/apps /scratch1/rnene/dinoenv.sif bash -lc "
  export PATH=/opt/micromamba/envs/app/bin:/usr/local/bin:/usr/bin:/bin
  export CPATH=/usr/include/x86_64-linux-gnu:${CPATH:-}
  export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
  # Keep Apptainer --nv CUDA libs on the path, then append MuJoCo.
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
  export HF_HOME=/project2/jessetho_1732/rl_eval_wm/dino_wm/.hf_cache
  export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
  export TRANSFORMERS_CACHE=\$HF_HOME/transformers
  export TORCH_HOME=\$HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  mkdir -p \"\$HUGGINGFACE_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\"
  pip install -q -U 'cython<3'
  cd /project2/jessetho_1732/rl_eval_wm/dino_wm
  python cache_pusht_model.py
  python evaluate_drift.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name pusht \
    --model_epoch latest \
    --policy_model_name lerobot/diffusion_pusht \
    --n_eval 10000 \
    --rollout_length 5 \
    --policy_img_size 96 \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/drift \
    --seed 42 \
    --device cuda &
  PID1=\$!
  python evaluate_drift.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name pusht \
    --model_epoch latest \
    --policy_model_name lerobot/diffusion_pusht \
    --n_eval 10000 \
    --rollout_length 8 \
    --policy_img_size 96 \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/drift \
    --seed 42 \
    --device cuda &
  PID2=\$!
  python evaluate_drift.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name pusht \
    --model_epoch latest \
    --policy_model_name lerobot/diffusion_pusht \
    --n_eval 10000 \
    --rollout_length 10 \
    --policy_img_size 96 \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/drift \
    --seed 42 \
    --device cuda &
  PID3=\$!
  wait \$PID1 || exit 1
  wait \$PID2 || exit 1
  wait \$PID3 || exit 1
"