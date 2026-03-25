#!/bin/bash

apptainer exec --nv --fakeroot --writable-tmpfs --bind /apps:/apps /scratch1/rnene/dinoenv.sif bash -lc "
  export PATH=/opt/micromamba/envs/app/bin:/usr/local/bin:/usr/bin:/bin
  export CPATH=/usr/include/x86_64-linux-gnu:${CPATH:-}
  export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
  pip install -q -U 'cython<3'
  cd /project2/jessetho_1732/rl_eval_wm/dino_wm
  python -c \"
import torch
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

import numpy; print('numpy:', numpy.__version__)
import einops; print('einops OK')
import omegaconf; print('omegaconf OK')
import hydra; print('hydra OK')
import matplotlib; print('matplotlib OK')
import cv2; print('cv2 OK')
import imageio; print('imageio OK')
import gym; print('gym OK')
import wandb; print('wandb OK')
import decord; print('decord OK')

# local imports
import sys, os
sys.path.insert(0, '.')
import utils; print('utils OK')
import rollout; print('rollout OK')
import lerobot_utils; print('lerobot_utils OK')
import evaluate_drift; print('evaluate_drift OK')
import evaluate_policy; print('evaluate_policy OK')
import evaluate_wm; print('evaluate_wm OK')

# dino_datasets import + basic usage
from dino_datasets.img_transforms import default_transform
from dino_datasets.traj_dset import TrajDataset, TrajSlicerDataset

class _DummyTraj(TrajDataset):
  action_dim = 2
  state_dim = 3
  proprio_dim = 1

  def __len__(self):
    return 1

  def get_seq_length(self, idx):
    return 4

  def __getitem__(self, idx):
    obs = {
      'visual': torch.zeros(4, 3, 8, 8),
      'proprio': torch.zeros(4, 1),
    }
    act = torch.zeros(4, 2)
    state = torch.zeros(4, 3)
    return obs, act, state, {}

dummy = _DummyTraj()
sliced = TrajSlicerDataset(dummy, num_frames=2, frameskip=1)
obs, act, state = sliced[0]
assert act.shape == (2, 2)

transform = default_transform(img_size=8)
img = torch.zeros(3, 8, 8)
img_out = transform(img)
assert img_out.shape == (3, 8, 8)
print('dino_datasets import/use OK')

print('All imports OK')
\"
"
