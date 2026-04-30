"""
robomimic_dset.py — Dataset loader for Robomimic HDF5 files compatible with dino_wm.

Robomimic stores data as:
  data/
    demo_0/
      obs/
        agentview_image:        (T, H, W, C)  uint8
        robot0_eef_pos:         (T, 3)
        robot0_eef_quat:        (T, 4)
        robot0_gripper_qwidth:  (T, 2)
      actions:                  (T, action_dim)
    ...
  mask/
    train:  list of demo keys
    valid:  list of demo keys

The factory function `load_robomimic_slice_train_val` mirrors the interface of
`dino_datasets.pusht_dset.load_pusht_slice_train_val` so it can be referenced
directly in a Hydra env config as `_target_`.
"""

import os
import sys

# Make dino_wm parent importable for shared utilities
_DINO_WM_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _DINO_WM_ROOT not in sys.path:
    sys.path.insert(0, _DINO_WM_ROOT)

import h5py
import torch
import numpy as np
from pathlib import Path
from typing import Callable, List, Optional

from einops import rearrange
from dino_datasets.traj_dset import TrajDataset, TrajSlicerDataset


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class RobomimicDataset(TrajDataset):
    """
    TrajDataset over a Robomimic HDF5 file.

    Each trajectory (demo) in the file becomes one dataset item.
    Visual observations come from a single camera (first entry of camera_names).
    Proprioceptive observations are the concatenation of proprio_keys.

    Args:
        data_path:      Path to the .hdf5 file.
        demo_keys:      Which demos to include (e.g. from the train/valid mask).
        camera_names:   Camera observation keys (without the trailing '_image'
                        suffix; e.g. ['agentview']).
        proprio_keys:   Robot-state observation keys to concatenate into proprio.
        transform:      Optional torchvision transform applied to (T, C, H, W)
                        float images in [0, 1].  Typically default_transform().
        action_mean/std:  Pre-computed normalisation statistics (computed once
                          over the training split and shared with the val split).
        proprio_mean/std: Same for proprio.
    """

    def __init__(
        self,
        data_path: str,
        demo_keys: List[str],
        camera_names: List[str],
        proprio_keys: List[str],
        transform: Optional[Callable],
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        proprio_mean: torch.Tensor,
        proprio_std: torch.Tensor,
    ):
        self.data_path = Path(data_path)
        self.demo_keys = demo_keys
        self.camera_names = camera_names
        self.proprio_keys = proprio_keys
        self.transform = transform

        self.action_mean = action_mean
        self.action_std = action_std
        self.proprio_mean = proprio_mean
        self.proprio_std = proprio_std

        # Load actions and proprio into RAM; images stay on disk.
        self._actions: List[torch.Tensor] = []
        self._proprios: List[torch.Tensor] = []
        self._seq_lengths: List[int] = []

        with h5py.File(self.data_path, "r") as f:
            for key in demo_keys:
                demo = f["data"][key]

                actions = torch.from_numpy(demo["actions"][:]).float()
                self._seq_lengths.append(actions.shape[0])

                proprio_parts = []
                for pk in proprio_keys:
                    if pk in demo["obs"]:
                        arr = demo["obs"][pk][:]
                        if arr.ndim == 1:
                            arr = arr[:, None]
                        proprio_parts.append(arr)
                proprio = (
                    np.concatenate(proprio_parts, axis=-1)
                    if proprio_parts
                    else np.zeros((actions.shape[0], 1), dtype=np.float32)
                )
                self._actions.append(
                    (actions - action_mean) / action_std
                )
                self._proprios.append(
                    (torch.from_numpy(proprio).float() - proprio_mean) / proprio_std
                )

        self.action_dim = self._actions[0].shape[-1]
        self.proprio_dim = self._proprios[0].shape[-1]
        self.state_dim = self.proprio_dim
        # state stats alias (used by Preprocessor)
        self.state_mean = proprio_mean
        self.state_std = proprio_std

        print(
            f"RobomimicDataset: {len(demo_keys)} demos, "
            f"action_dim={self.action_dim}, proprio_dim={self.proprio_dim}"
        )

    # ------------------------------------------------------------------
    # TrajDataset interface
    # ------------------------------------------------------------------

    def get_seq_length(self, idx: int) -> int:
        return self._seq_lengths[idx]

    def __len__(self) -> int:
        return len(self.demo_keys)

    def get_frames(self, idx: int, frames):
        """
        Load a set of frames for demo `idx`.

        Returns:
            obs:   dict {"visual": (T, C, H, W) float in [-1, 1],
                         "proprio": (T, proprio_dim)}
            act:   (T, action_dim)  — normalised
            state: (T, state_dim)   — normalised
            meta:  {}
        """
        frames = list(frames)
        key = self.demo_keys[idx]

        # Resolve camera key (robomimic appends '_image' to camera names)
        cam_suffix = "_image"
        cam_key = self.camera_names[0] + cam_suffix

        with h5py.File(self.data_path, "r") as f:
            demo_obs = f["data"][key]["obs"]
            if cam_key not in demo_obs:
                # Fallback: try exact key without suffix
                cam_key = self.camera_names[0]
            images = demo_obs[cam_key][frames]  # (T, H, W, C) uint8

        images = torch.from_numpy(images.astype(np.float32)) / 255.0  # [0, 1]
        images = rearrange(images, "T H W C -> T C H W")
        if self.transform:
            images = self.transform(images)  # typically maps to [-1, 1]

        act = self._actions[idx][frames]
        proprio = self._proprios[idx][frames]

        obs = {"visual": images, "proprio": proprio}
        return obs, act, proprio, {}

    def __getitem__(self, idx: int):
        return self.get_frames(idx, range(self.get_seq_length(idx)))


# ---------------------------------------------------------------------------
# Helpers for computing normalisation statistics
# ---------------------------------------------------------------------------

def _compute_stats(data_path: str, demo_keys: List[str], proprio_keys: List[str]):
    """Compute mean/std of actions and proprio over the given demo keys."""
    all_actions = []
    all_proprios = []

    with h5py.File(data_path, "r") as f:
        for key in demo_keys:
            demo = f["data"][key]
            all_actions.append(torch.from_numpy(demo["actions"][:]).float())

            parts = []
            for pk in proprio_keys:
                if pk in demo["obs"]:
                    arr = demo["obs"][pk][:]
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    parts.append(arr)
            if parts:
                all_proprios.append(
                    torch.from_numpy(np.concatenate(parts, axis=-1)).float()
                )

    actions_cat = torch.cat(all_actions, dim=0)
    proprios_cat = torch.cat(all_proprios, dim=0) if all_proprios else torch.zeros(1, 1)

    action_mean = actions_cat.mean(0)
    action_std = actions_cat.std(0).clamp(min=1e-6)
    proprio_mean = proprios_cat.mean(0)
    proprio_std = proprios_cat.std(0).clamp(min=1e-6)

    return action_mean, action_std, proprio_mean, proprio_std


def _get_split_keys(data_path: str, split: str):
    """Return demo keys for a given split from the HDF5 mask group (or auto-split)."""
    with h5py.File(data_path, "r") as f:
        all_keys = sorted(f["data"].keys())
        if "mask" in f and split in f["mask"]:
            keys = [
                k.decode() if isinstance(k, bytes) else k
                for k in f["mask"][split][:]
            ]
            return keys
        # Auto-split: 90 / 10
        n_train = int(0.9 * len(all_keys))
        if split == "train":
            return all_keys[:n_train]
        else:
            return all_keys[n_train:]


# ---------------------------------------------------------------------------
# Factory function — matches dino_wm's dataset-loading interface
# ---------------------------------------------------------------------------

def load_robomimic_slice_train_val(
    transform: Optional[Callable] = None,
    data_path: str = "data/robomimic/lift/ph/low_dim_v15.hdf5",
    camera_names: Optional[List[str]] = None,
    proprio_keys: Optional[List[str]] = None,
    normalize_action: bool = True,
    num_hist: int = 3,
    num_pred: int = 1,
    frameskip: int = 1,
):
    """
    Factory function compatible with dino_wm's `hydra.utils.call(dataset_cfg, ...)`.

    Returns:
        (sliced_datasets, traj_datasets) where each is a dict with 'train' / 'valid'.
        sliced_datasets: TrajSlicerDataset wrappers (used for training).
        traj_datasets:   full RobomimicDataset objects (used for rollout eval).
    """
    camera_names = camera_names or ["agentview"]
    proprio_keys = proprio_keys or [
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qwidth",
    ]

    train_keys = _get_split_keys(data_path, "train")
    valid_keys = _get_split_keys(data_path, "valid")

    # Compute normalization statistics from the training split only
    if normalize_action:
        action_mean, action_std, proprio_mean, proprio_std = _compute_stats(
            data_path, train_keys, proprio_keys
        )
    else:
        # Determine dims by peeking at one demo
        with h5py.File(data_path, "r") as f:
            demo0 = f["data"][train_keys[0]]
            action_dim = demo0["actions"].shape[-1]
            parts = [
                demo0["obs"][pk][:1]
                for pk in proprio_keys
                if pk in demo0["obs"]
            ]
        proprio_dim = sum(
            (p.shape[-1] if p.ndim > 1 else 1) for p in parts
        )
        action_mean = torch.zeros(action_dim)
        action_std = torch.ones(action_dim)
        proprio_mean = torch.zeros(proprio_dim)
        proprio_std = torch.ones(proprio_dim)

    shared_kwargs = dict(
        data_path=data_path,
        camera_names=camera_names,
        proprio_keys=proprio_keys,
        transform=transform,
        action_mean=action_mean,
        action_std=action_std,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
    )

    train_dset = RobomimicDataset(demo_keys=train_keys, **shared_kwargs)
    val_dset = RobomimicDataset(demo_keys=valid_keys, **shared_kwargs)

    num_frames = num_hist + num_pred
    train_slices = TrajSlicerDataset(train_dset, num_frames, frameskip)
    val_slices = TrajSlicerDataset(val_dset, num_frames, frameskip)

    sliced_datasets = {"train": train_slices, "valid": val_slices}
    traj_datasets = {"train": train_dset, "valid": val_dset}

    return sliced_datasets, traj_datasets
