#!/bin/bash

#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=evaluate_wm_robomimic
#SBATCH --output=evaluate_wm_robomimic.out
#SBATCH --error=evaluate_wm_robomimic.err

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

apptainer exec --nv --fakeroot --writable-tmpfs --bind /apps:/apps /scratch1/rnene/dinoenv.sif bash -lc "
  export PATH=/opt/micromamba/envs/app/bin:/usr/local/bin:/usr/bin:/bin
  export CPATH=/usr/include/x86_64-linux-gnu:${CPATH:-}
  export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
  export HF_HOME=/project2/jessetho_1732/rl_eval_wm/dino_wm/.hf_cache
  export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
  export TRANSFORMERS_CACHE=\$HF_HOME/transformers
  export TORCH_HOME=\$HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  mkdir -p \"\$HUGGINGFACE_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\"
  pip install -q -U 'cython<3'
  cd /project2/jessetho_1732/rl_eval_wm/dino_wm
  python robomimic_code/evaluate_wm_robomimic.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name robomimic \
    --model_epoch latest \
    --n_eval 10000 \
    --batch_size 8 \
    --rollout_length 10 \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/robomimic_wm/rl10 \
    --seed 42
  python robomimic_code/evaluate_wm_robomimic.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name robomimic \
    --model_epoch latest \
    --n_eval 10000 \
    --batch_size 8 \
    --rollout_length 20 \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/robomimic_wm/rl20 \
    --seed 42
  python robomimic_code/evaluate_wm_robomimic.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name robomimic \
    --model_epoch latest \
    --n_eval 10000 \
    --batch_size 8 \
    --rollout_length 40 \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/robomimic_wm/rl40 \
    --seed 42
"
