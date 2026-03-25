#!/usr/bin/env python3
"""
rollout_robomimic.py — Rollout and closed-loop evaluation for a dino_wm world
model trained on Robomimic data.

Mirrors the structure of dino_wm/rollout.py but is tailored for Robomimic:
  • Loads a RobomimicDataset (HDF5) as the replay source.
  • Optionally wraps a RobomimicWrapper environment for ground-truth sim rollouts.
  • Supports plugging in a robomimic-style pretrained policy (BC / diffusion)
    or replaying GT actions from the dataset.
  • Scores closed-loop rollouts with Robometer.

Usage (argparse — no Hydra required):
    python robomimic_code/rollout_robomimic.py \
        --ckpt_base_path /project2/.../dino_wm \
        --model_name robomimic_lift \
        --model_epoch latest \
        --rollout_mode open_loop \
        --n_rollouts 10 \
        --rollout_length 20 \
        --output_dir ./rollout_results

Or with Hydra (from the dino_wm root, using the bundled config):
    python robomimic_code/rollout_robomimic.py \
        --config-path robomimic_code/conf \
        --config-name rollout_robomimic
"""

import os
import sys
import json
import pickle
import random
import argparse
import warnings
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

# ── Make dino_wm root importable ─────────────────────────────────────────────
_DINO_WM_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _DINO_WM_ROOT not in sys.path:
    sys.path.insert(0, _DINO_WM_ROOT)

# ── Make robomimic_code root importable ──────────────────────────────────────
_ROBOMIMIC_CODE_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROBOMIMIC_CODE_ROOT not in sys.path:
    sys.path.insert(0, _ROBOMIMIC_CODE_ROOT)

try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    from PIL import Image as PILImage
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from use_robometer import load_robometer, infer_robometer
from utils import seed as set_seed
from preprocessor import Preprocessor

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

class RobomimicPolicyWrapper:
    """
    Wraps a robomimic-trained policy (BC / BCQ / HBC / Diffusion) for
    single-step inference inside the closed-loop rollout.

    Args:
        ckpt_path:   Path to the robomimic checkpoint (.pth).
        device:      Torch device string.
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        try:
            import robomimic.utils.file_utils as FileUtils
        except ImportError as exc:
            raise ImportError("robomimic is required for RobomimicPolicyWrapper.") from exc

        self.device = device
        self.policy, _ = FileUtils.policy_from_checkpoint(
            ckpt_path=ckpt_path, device=device, verbose=True
        )
        self.policy.start_episode()

    def predict(self, obs_dict: dict) -> np.ndarray:
        """
        obs_dict keys mirror the robomimic observation spec:
          "agentview_image":    (1, H, W, C) or (H, W, C) uint8
          "robot0_eef_pos":     (1, 3) or (3,)  float
          etc.

        Returns: action (action_dim,) numpy array.
        """
        action = self.policy(obs_dict)
        return np.array(action)

    def reset(self):
        self.policy.start_episode()


# ---------------------------------------------------------------------------
# Rollout workspace
# ---------------------------------------------------------------------------

class RobomimicRolloutWorkspace:
    """
    Coordinates data sampling, WM rollout, and optional policy + Robometer scoring
    for a world model trained on Robomimic data.
    """

    def __init__(
        self,
        cfg: dict,
        wm: torch.nn.Module,
        dset,
        frameskip: int,
    ):
        self.cfg = cfg
        self.wm = wm
        self.dset = dset
        self.frameskip = frameskip
        self.device = next(wm.parameters()).device

        self.n_rollouts = cfg.get("n_rollouts", 10)
        self.rollout_length = cfg.get("rollout_length", 20)
        self.save_video = cfg.get("save_video", True)
        self.policy_mode = cfg.get("policy_mode", "gt")
        self.output_dir = cfg.get("output_dir", "./rollout_results")
        os.makedirs(self.output_dir, exist_ok=True)

        self.policy = None
        if self.policy_mode == "ckpt":
            policy_ckpt = cfg.get("policy_ckpt")
            if policy_ckpt is None:
                raise ValueError("policy_mode='ckpt' requires --policy_ckpt to be set.")
            self.policy = RobomimicPolicyWrapper(
                ckpt_path=policy_ckpt,
                device=str(self.device),
            )

        # Preprocessor for action denorm (used if we need real env actions)
        self.preprocessor = Preprocessor(
            action_mean=dset.action_mean,
            action_std=dset.action_std,
            state_mean=dset.state_mean,
            state_std=dset.state_std,
            proprio_mean=dset.proprio_mean,
            proprio_std=dset.proprio_std,
            transform=dset.transform,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_trajectory(self, min_raw_len: int):
        """Sample a random trajectory from the dataset that is long enough."""
        indices = list(range(len(self.dset)))
        random.shuffle(indices)
        for idx in indices:
            obs, act, state, _ = self.dset[idx]
            if obs["visual"].shape[0] >= min_raw_len:
                return obs, act, state
        raise RuntimeError(
            f"Dataset has no trajectories with >= {min_raw_len} raw frames."
        )

    # ------------------------------------------------------------------
    # Open-loop rollout (GT actions → WM)
    # ------------------------------------------------------------------

    def perform_rollout(self):
        """
        Run WM with ground-truth actions and compare WM-decoded frames against
        ground-truth observations.  Saves videos and logs latent-space errors.
        """
        plots_dir = os.path.join(self.output_dir, "rollout_plots")
        os.makedirs(plots_dir, exist_ok=True)

        min_horizon = self.cfg.get("min_horizon", 4)
        rand_start_end = self.cfg.get("rand_start_end", True)
        num_hist = self.cfg.get("num_hist", 3)
        min_raw_len = min_horizon * self.frameskip + 1

        all_errors = []

        for idx in range(self.n_rollouts):
            obs, act, _ = self._sample_trajectory(min_raw_len)
            T_raw = obs["visual"].shape[0]

            if rand_start_end:
                start = random.randint(
                    0, max(0, T_raw - min_raw_len)
                )
                max_h = (T_raw - start - 1) // self.frameskip
                horizon = random.randint(min_horizon, max(min_horizon, max_h))
            else:
                start = 0
                horizon = (T_raw - 1) // self.frameskip

            # Subsample at frameskip
            obs_sub = {
                k: v[start: start + horizon * self.frameskip + 1: self.frameskip]
                for k, v in obs.items()
            }
            act_sub = act[start: start + horizon * self.frameskip]
            act_sub = rearrange(act_sub, "(h f) d -> h (f d)", f=self.frameskip)

            # Initial context and goal
            obs_0 = {k: v[:num_hist].unsqueeze(0).to(self.device)
                     for k, v in obs_sub.items()}
            actions = act_sub.unsqueeze(0).to(self.device)

            obs_g = {k: v[-1:].unsqueeze(0).to(self.device)
                     for k, v in obs_sub.items()}
            z_g = self.wm.encode_obs(obs_g)

            with torch.no_grad():
                z_obses, _ = self.wm.rollout(obs_0, actions)

            # Latent error at last step
            from utils import slice_trajdict_with_t
            z_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
            errors = {
                k: torch.norm(z_last[k] - z_g[k], p=2, dim=-1).mean().item()
                for k in z_last if k in z_g
            }
            all_errors.append(errors)
            print(f"  [Rollout {idx:3d}] latent errors: {errors}")

            # Decode and save video if decoder present
            if self.wm.decoder is not None:
                pred_frames = self.wm.decode_obs(z_obses)[0]["visual"][0].cpu()
                gt_frames = obs_sub["visual"]
                base = os.path.join(plots_dir, f"rollout_{idx:03d}")
                self._save_video(gt_frames, pred_frames, base)

        # Aggregate
        agg = {}
        for k in all_errors[0]:
            agg[k] = float(np.mean([e[k] for e in all_errors]))
        print(f"\n=== Open-loop rollout summary: {agg}")

        out = os.path.join(self.output_dir, "rollout_errors.json")
        with open(out, "w") as f:
            json.dump({"per_rollout": all_errors, "mean": agg}, f, indent=2)
        print(f"Saved rollout errors → {out}")
        return agg

    # ------------------------------------------------------------------
    # Closed-loop policy rollout (WM + policy + Robometer)
    # ------------------------------------------------------------------

    def perform_closed_loop_policy_rollout(self):
        """
        At each step:
          1. Decode the current WM latent to a pixel image.
          2. Feed image (+ GT proprio) to the policy to get an action.
          3. Step the WM with that action.
          4. Score each frame with Robometer.
        """
        plots_dir = os.path.join(self.output_dir, "rollout_plots")
        os.makedirs(plots_dir, exist_ok=True)

        policy_img_size = self.cfg.get("policy_img_size", 84)
        num_hist = self.cfg.get("num_hist", 3)
        robometer_prompt = self.cfg.get(
            "robometer_prompt",
            "Pick up the red cube and place it in the bin.",
        )
        min_raw_len = num_hist + 1

        robometer_model, robometer_processor = load_robometer()

        all_progress = []

        for idx in range(self.n_rollouts):
            obs, act, _ = self._sample_trajectory(min_raw_len)
            orig_action_dim = act.shape[-1]
            wm_action_dim = orig_action_dim * self.frameskip

            obs_0 = {k: v[:num_hist].unsqueeze(0).to(self.device)
                     for k, v in obs.items()}
            act_0 = torch.zeros(1, num_hist, wm_action_dim, device=self.device)

            decoded_frames = []

            if self.policy is not None:
                self.policy.reset()

            with torch.no_grad():
                z = self.wm.encode(obs_0, act_0)

                for t in range(self.rollout_length):
                    # Decode current WM state
                    z_obs_cur, _ = self.wm.separate_emb(z[:, -1:])
                    decoded, _ = self.wm.decode_obs(z_obs_cur)
                    img = decoded["visual"][0, 0]   # (C, H, W) in [-1, 1]
                    decoded_frames.append(img.cpu())

                    # Get action
                    if self.policy_mode == "gt":
                        raw_idx = min(
                            t * self.frameskip,
                            act.shape[0] - self.frameskip,
                        )
                        action_np = act[raw_idx: raw_idx + self.frameskip].numpy()
                        # Already normalised — denorm for policy input if needed
                        action_grouped = torch.from_numpy(
                            action_np.reshape(1, 1, -1)
                        ).float().to(self.device)
                    elif self.policy_mode == "ckpt":
                        img_01 = (img.clamp(-1, 1) + 1) / 2
                        img_policy = F.interpolate(
                            img_01.unsqueeze(0),
                            size=(policy_img_size, policy_img_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        state_idx = min(t + num_hist - 1, obs["proprio"].shape[0] - 1)
                        proprio_t = obs["proprio"][state_idx].numpy()
                        # Build a minimal obs dict for the robomimic policy
                        cam_key = self.cfg.get("camera_names", ["agentview"])[0] + "_image"
                        img_hwc = (
                            img_01.permute(1, 2, 0).cpu().numpy() * 255
                        ).astype(np.uint8)
                        policy_obs = {
                            cam_key: img_hwc[None],  # (1, H, W, C)
                            "robot0_eef_pos": proprio_t[:3][None],
                        }
                        action_np = self.policy.predict(policy_obs)
                        action_grouped = torch.from_numpy(
                            action_np.reshape(1, 1, -1).repeat(self.frameskip, axis=-1)
                        ).float().to(self.device)
                    else:
                        action_grouped = torch.zeros(
                            1, 1, wm_action_dim, device=self.device
                        )

                    # Step WM
                    z_pred = self.wm.predict(z[:, -self.wm.num_hist:])
                    z_new = z_pred[:, -1:]
                    z_new = self.wm.replace_actions_from_z(z_new, action_grouped)
                    z = torch.cat([z, z_new], dim=1)

                # Final frame
                z_obs_last, _ = self.wm.separate_emb(z[:, -1:])
                dec_last, _ = self.wm.decode_obs(z_obs_last)
                decoded_frames.append(dec_last["visual"][0, 0].cpu())

            pred_frames = torch.stack(decoded_frames, dim=0)  # (T+1, C, H, W)
            gt_frames = obs["visual"][: len(decoded_frames)]

            base = os.path.join(plots_dir, f"policy_rollout_{idx:03d}")
            self._save_video(gt_frames, pred_frames, base)

            # Robometer scoring
            progress_scores = []
            if robometer_model is not None and HAVE_PIL:
                frames_u8 = self._to_uint8(pred_frames)
                for frame in frames_u8:
                    pil_img = PILImage.fromarray(frame)
                    out = infer_robometer(
                        robometer_model, robometer_processor, pil_img, robometer_prompt
                    )
                    if hasattr(out, "logits"):
                        score = float(out.logits.squeeze())
                    elif hasattr(out, "last_hidden_state"):
                        score = float(out.last_hidden_state.mean())
                    else:
                        score = float(torch.as_tensor(out[0]).mean())
                    progress_scores.append(score)
            else:
                print("[Warning] Robometer not available; skipping scoring.")

            all_progress.append(progress_scores)
            print(
                f"  [Rollout {idx:3d}] Robometer ({len(progress_scores)} frames): "
                f"{progress_scores}"
            )

        scores_path = os.path.join(self.output_dir, "robometer_progress.json")
        with open(scores_path, "w") as f:
            json.dump(
                {"progress": all_progress, "n_rollouts": self.n_rollouts,
                 "prompt": robometer_prompt},
                f, indent=2,
            )
        print(f"Saved Robometer progress → {scores_path}")
        return all_progress

    # ------------------------------------------------------------------
    # Video / array utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(frames: torch.Tensor) -> np.ndarray:
        """(T, C, H, W) in [-1, 1] → (T, H, W, C) uint8."""
        arr = frames.cpu().float().numpy()
        arr = arr.transpose(0, 2, 3, 1)
        arr = np.clip(arr * 0.5 + 0.5, 0, 1)
        return (arr * 255).astype(np.uint8)

    def _save_video(self, gt_frames: torch.Tensor, pred_frames: torch.Tensor, base: str):
        fps = self.cfg.get("video_fps", 10)
        gt_u8 = self._to_uint8(gt_frames)
        pred_u8 = self._to_uint8(pred_frames)
        np.save(f"{base}_gt.npy", gt_u8)
        np.save(f"{base}_pred.npy", pred_u8)
        if HAVE_IMAGEIO:
            imageio.mimsave(f"{base}_gt.mp4", gt_u8, fps=fps)
            imageio.mimsave(f"{base}_pred.mp4", pred_u8, fps=fps)
        elif HAVE_CV2:
            for tag, arr in [("gt", gt_u8), ("pred", pred_u8)]:
                h, w = arr.shape[1], arr.shape[2]
                out = cv2.VideoWriter(
                    f"{base}_{tag}.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (w, h),
                )
                for frame in arr:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def rollout_main(cfg: dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))
    random.seed(cfg.get("seed", 42))

    model_path = Path(cfg["ckpt_base_path"]) / "outputs" / cfg["model_name"]
    with open(model_path / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    # Dataset (loaded from the model's own config — must have env.dataset pointing
    # to a robomimic loader, e.g. robomimic_code.dino_datasets.robomimic_dset.*)
    _, dset_dict = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset_dict["valid"]

    model_ckpt = model_path / "checkpoints" / f"model_{cfg.get('model_epoch', 'latest')}.pth"
    wm = _load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    workspace = RobomimicRolloutWorkspace(
        cfg=cfg,
        wm=wm,
        dset=dset,
        frameskip=model_cfg.frameskip,
    )

    rollout_mode = cfg.get("rollout_mode", "open_loop")
    with torch.no_grad():
        if rollout_mode == "closed_loop_policy":
            workspace.perform_closed_loop_policy_rollout()
        else:
            workspace.perform_rollout()


def parse_args():
    p = argparse.ArgumentParser(
        description="Rollout evaluation for a dino_wm model trained on Robomimic data."
    )
    p.add_argument("--ckpt_base_path", required=True,
                   help="Base dir containing outputs/<model_name>/")
    p.add_argument("--model_name", required=True)
    p.add_argument("--model_epoch", default="latest")
    p.add_argument("--rollout_mode", default="open_loop",
                   choices=["open_loop", "closed_loop_policy"])
    p.add_argument("--policy_mode", default="gt",
                   choices=["gt", "ckpt"])
    p.add_argument("--policy_ckpt", default=None,
                   help="Path to robomimic policy checkpoint (.pth)")
    p.add_argument("--n_rollouts", type=int, default=10)
    p.add_argument("--rollout_length", type=int, default=20)
    p.add_argument("--num_hist", type=int, default=3)
    p.add_argument("--min_horizon", type=int, default=4)
    p.add_argument("--policy_img_size", type=int, default=84)
    p.add_argument("--robometer_prompt", default="Pick up the red cube and place it in the bin.")
    p.add_argument("--output_dir", default="./rollout_results")
    p.add_argument("--save_video", action="store_true", default=True)
    p.add_argument("--video_fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return vars(p.parse_args())


if __name__ == "__main__":
    rollout_main(parse_args())
