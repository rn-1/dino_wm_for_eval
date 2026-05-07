#!/bin/bash

#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --job-name=policy_ranking_robomimic
#SBATCH --output=policy_ranking_robomimic.out
#SBATCH --error=policy_ranking_robomimic.err

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

# Directory of robomimic-style policy checkpoints — each file is one model
# evaluated as a separate ranking entry by --policy_ckpt_dir.
POLICY_CKPT_DIR="/project2/jessetho_1732/rl_eval_wm/dino_wm/20260423193411_diffusion_policy_lift-mh-image_v15_Ta=1/20260423193411/models"
# Glob pattern matching checkpoint files inside POLICY_CKPT_DIR (override if .pth/.pt).
CKPT_GLOB="*.ckpt"

apptainer exec --nv --fakeroot --writable-tmpfs --bind /apps:/apps /scratch1/rnene/dinoenv.sif bash -lc "
  export PATH=/opt/micromamba/envs/app/bin:/usr/local/bin:/usr/bin:/bin
  export CPATH=/usr/include/x86_64-linux-gnu:${CPATH:-}
  export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
  export DATASET_DIR=/project2/jessetho_1732/rl_eval_wm/dino_wm/data
  export HF_HOME=/project2/jessetho_1732/rl_eval_wm/dino_wm/.hf_cache
  export HUGGINGFACE_HUB_CACHE=\$HF_HOME/hub
  export TRANSFORMERS_CACHE=\$HF_HOME/transformers
  export TORCH_HOME=\$HF_HOME
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  pip install git+https://github.com/aravindr93/mjrl.git
  mkdir -p \"\$HUGGINGFACE_HUB_CACHE\" \"\$TRANSFORMERS_CACHE\"
  export MUJOCO_GL=egl
  pip install -q -U 'cython<3'
  ( cd /project2/jessetho_1732/rl_eval_wm/robometer && pip install -e . ) \
    || ( cd /project2/jessetho_1732/rl_eval_wm/robometer && pip install . ) \
    || export PYTHONPATH=/project2/jessetho_1732/rl_eval_wm/robometer:\${PYTHONPATH:-}
  sed -i 's/field(init=False, metadata=/field(init=False, default=None, metadata=/g' \
    /opt/micromamba/envs/app/lib/python3.12/site-packages/lerobot/policies/groot/groot_n1.py
  cd /project2/jessetho_1732/rl_eval_wm/dino_wm
  python robomimic_code/policy_ranking_robomimic.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name robomimic \
    --model_epoch latest \
    --policy_ckpt_dir \"$POLICY_CKPT_DIR\" \
    --ckpt_glob \"$CKPT_GLOB\" \
    --ckpt_inhibition_coeff 1.0 \
    --n_eval 1000 \
    --rollout_length 10 \
    --robometer_prompt 'Pick up the red cube and place it in the bin.' \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/policy_ranking_robomimic \
    --seed 42 \
    --run_gym
  python robomimic_code/policy_ranking_robomimic.py \
    --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \
    --model_name robomimic \
    --model_epoch latest \
    --policy_ckpt_dir \"$POLICY_CKPT_DIR\" \
    --ckpt_glob \"$CKPT_GLOB\" \
    --ckpt_inhibition_coeff 1.0 \
    --n_eval 1000 \
    --rollout_length 20 \
    --robometer_prompt 'Pick up the red cube and place it in the bin.' \
    --output_dir /project2/jessetho_1732/rl_eval_wm/dino_wm/eval_results/policy_ranking_robomimic \
    --seed 42 \
    --run_gym
"
