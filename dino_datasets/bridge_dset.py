import torch
import decord
import numpy as np
from pathlib import Path
from einops import rearrange
from decord import VideoReader, gpu, cpu
from typing import Callable, Optional
from .traj_dset import TrajDataset, get_train_val_sliced
from typing import Optional, Callable, Any
decord.bridge.set_bridge("torch")
import json 

class BridgeDataset(torch.utils.data.Dataset):

    def __init__(self, directory, split
            transform Optional[Callable] = None,
            normalize_action: bool = False,
            n_rollouts: Optional[int],
            action_scale = 1.0,
        ):

        self.directory = directory
        self.transform = transform
        self.normalize_action = normalize_action
        self.files = g.glob(f"{directory}/bridge/annotation/{split}/*.json")[:n_rollouts]
        self.len = len(self.files)

    def __getitem__(self, idx):
        with json.load(files[idx]) as f:

            video_dir = f["videos"]["video_path"]# idk if there will be more than one
        
            vr = VideoReader(video_path, ctx=gpu(0))
            num_frames = len(vr)
            # Read all frames at once
            frames = vr.get_batch(range(num_frames))  # shape: (T, H, W, 3)
            
            # Convert to PyTorch tensor and rearrange dimensions
            video_tensor = frames.permute(0, 3, 1, 2).float() / 255.0
            video_tensor = torch.from_numpy(frames.asnumpy())

            if(self.transform):
                video_tensor = self.transform(video_tensor)

            actions = f["action"]
            states = f["state"]
            contin_grip = f["continuous_gripper_state"]

            obs = {
                "visual":video_tensor,
                "proprio": states
            }

            return obs, actions, states, contin_grip

    def __len__(self):
        return self.len
