#!/bin/bash

#SBATCH --partition=nlp_hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --job-name=pusht_tune

module purge
module load gcc/11.3.0
module load conda

source ~/.bashrc

conda init
conda activate dino_wm

export DATASET_DIR="/project2/jessetho_1732/rl_eval_wm/dino_wm/data"

export WANDB_API_KEY="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"

wandb login

python train.py --config-name train.yaml env=pusht frameskip=5 num_hist=3

