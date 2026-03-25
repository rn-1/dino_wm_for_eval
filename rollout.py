import os
import gym
import json
import hydra
import random
import torch
import torch.nn.functional as F
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

try:
    from PIL import Image as PILImage
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

import sys as _sys
from lerobot_utils import PolicyWrapper

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    import imageio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed

from utils import slice_trajdict_with_t

from use_robometer import load_robometer, infer_robometer
from policy_eval import LeRobotSingleStepAgent, load_policy_ckpt

from plan import (
    load_model,
    load_ckpt,
    ALL_MODEL_KEYS,
    DummyWandbRun,
)


# define static ideas of dummy and zero action

ZERO_ACTION = torch.zeros(1, 2)  # (batch=1, action_dim=2) — PushT agent action (dx, dy)
DUMMY_ACTION = torch.ones(1, 2) # (batch=1, action_dim=2) — PushT agent action (dx, dy)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class RolloutWorkspace:
    """
    Perform a simple rollout using ground truth actions and collect observations/images.
    """

    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env,
        env_name: str,
        frameskip: int,
        wandb_run=None,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run or DummyWandbRun()
        self.device = next(wm.parameters()).device

        # Policy setup
        self.policy_mode = cfg_dict.get("policy_mode", "gt")
        self.policy = None
        if self.policy_mode == "ckpt":
            policy_model_name = cfg_dict.get("policy_model_name", "lerobot/diffusion_pusht")
            self.policy = PolicyWrapper(model_name=policy_model_name)

        # Setup for rollout
        self.n_rollouts = cfg_dict.get("n_rollouts", 1)
        self.rollout_length = cfg_dict.get("rollout_length", 10)
        self.save_video = cfg_dict.get("save_video", True)
        self.save_images = cfg_dict.get("save_images", False)
        
        # Data preprocessor for action denormalization
        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )


    def init_models(self):
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        if model_ckpt.exists():
            self.load_ckpt(model_ckpt)
            log.info(f"Resuming from epoch {self.epoch}: {model_ckpt}")

        if policy_ckpt := self.cfg_dict.get("policy_ckpt") is not None:
            self.policy = load_policy_ckpt(policy_ckpt)

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proprio_encoder = hydra.utils.instantiate(
            self.cfg.proprio_encoder,
            in_chans=self.datasets["train"].proprio_dim,
            emb_dim=self.cfg.proprio_emb_dim,
        )
        proprio_emb_dim = self.proprio_encoder.emb_dim
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        self.action_encoder = hydra.utils.instantiate(
            self.cfg.action_encoder,
            in_chans=self.datasets["train"].action_dim,
            emb_dim=self.cfg.action_emb_dim,
        )
        action_emb_dim = self.action_encoder.emb_dim
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
    )
        
    def sample_rollout_trajectories(self):
        """
        Sample initial states and ground truth actions from the dataset.
        """
        states = []
        actions = []
        observations = []
        env_info = []

        # Sample trajectories of sufficient length
        for i in range(self.n_rollouts):
            max_offset = -1
            while max_offset < 0:
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                required_len = self.frameskip * self.rollout_length + 1
                max_offset = obs["visual"].shape[0] - required_len

            state = state.numpy()
            offset = random.randint(0, max_offset)
            
            obs = {
                key: arr[offset : offset + required_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + required_len]
            act = act[offset : offset + self.frameskip * self.rollout_length]
            
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)

        return observations, states, actions, env_info

    def perform_rollout(self, rb_components=None):
        """
        Execute rollout using model predictions based on openloop_rollout logic.
        Uses the world model to predict observations from ground truth actions.

        Args:
            rb_components: optional 4-tuple (rb_config, rb_tokenizer, rb_processor, rb_model)
                           pre-loaded by the caller. If None, Robometer is loaded here.
        """
        np.random.seed(self.cfg_dict.get("seed", 0))
        min_horizon = self.cfg_dict.get("min_horizon", 2)
        rand_start_end = self.cfg_dict.get("rand_start_end", True)

        output_dir = self.cfg_dict.get("saved_folder", ".")
        plotting_dir = os.path.join(output_dir, "rollout_plots")
        if not os.path.exists(plotting_dir):
            os.makedirs(plotting_dir, exist_ok=True)

        # Load Robometer (or reuse pre-loaded components)
        if rb_components is not None:
            rb_config, rb_tokenizer, rb_processor, rb_model = rb_components
        else:
            rb_config, rb_tokenizer, rb_processor, rb_model = load_robometer()
        robometer_prompt = self.cfg_dict.get(
            "robometer_prompt", "Push the T block to the goal position."
        )

        logs = {}
        rollout_results = []
        all_rb_results = {}

        # Rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg_dict.get("num_hist", 2), ""), (1, "_1framestart")]

        # Sample and process trajectories
        for idx in range(self.n_rollouts):
            valid_traj = False
            while not valid_traj:
                traj_idx = np.random.randint(0, len(self.dset))
                obs, act, state, _ = self.dset[traj_idx]
                # perform inferenec using the obs and image w policy if exists, else replace with
                if self.policy is not None:
                    step_actions = []
                    policy_img_size = self.cfg_dict.get("policy_img_size", 96)
                    for t in range(obs["visual"].shape[0]):
                        obs_t = {k: obs[k][t] for k in obs.keys()}
                        state = obs_t.get("proprio", obs_t.get("state", next(iter(obs_t.values())))).unsqueeze(0).to(self.device)
                        if "visual" in obs_t:
                            raw_img = obs_t["visual"].unsqueeze(0).to(self.device)  # (1, C, H, W) in [-1, 1]
                            img_01 = (raw_img.clamp(-1.0, 1.0) + 1.0) / 2.0
                            image = F.interpolate(img_01, size=(policy_img_size, policy_img_size), mode="bilinear", align_corners=False)
                        else:
                            image = torch.zeros(1, 3, policy_img_size, policy_img_size, device=self.device)
                        act_t = self.policy.predict(state, image).squeeze(0)
                        step_actions.append(act_t)
                    act = torch.stack(step_actions, dim=0)
                elif self.policy_mode == "dummy":
                    act = DUMMY_ACTION.repeat(obs["visual"].shape[0], 1)
                elif self.policy_mode == "zero":
                    act = ZERO_ACTION.repeat(obs["visual"].shape[0], 1)

                # defaults to gt action

                act = act.to(self.device)
                
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.frameskip
                    max_horizon = min(max_horizon, self.rollout_length)
                    if max_horizon > min_horizon:
                        valid_traj = True
                        horizon = np.random.randint(min_horizon, max_horizon + 1)
                else:
                    valid_traj = True
                    start = 0
                    horizon = min(
                        (obs["visual"].shape[0] - 1) // self.frameskip,
                        self.rollout_length,
                    )

            # Extract subsampled observations at frameskip intervals
            obs_subsampled = {}
            for k in obs.keys():
                obs_subsampled[k] = obs[k][
                    start : 
                    start + horizon * self.frameskip + 1 : 
                    self.frameskip
                ]
            
            # Subsample actions
            act_subsampled = act[start : start + horizon * self.frameskip]
            act_subsampled = rearrange(act_subsampled, "(h f) d -> h (f d)", f=self.frameskip)

            # Get goal observation encoding
            obs_g = {}
            for k in obs_subsampled.keys():
                obs_g[k] = obs_subsampled[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.wm.encode_obs(obs_g)

            # Select actions based on policy_mode
            if self.policy_mode == "dummy":
                actions = torch.ones_like(act_subsampled).unsqueeze(0)
            elif self.policy_mode == "gt":
                actions = act_subsampled.unsqueeze(0)
            elif self.policy_mode == "ckpt":
                step_actions = []
                policy_img_size = self.cfg_dict.get("policy_img_size", 96)
                for t in range(horizon):
                    obs_t = {k: obs_subsampled[k][t] for k in obs_subsampled.keys()}
                    state = obs_t.get("proprio", obs_t.get("state", next(iter(obs_t.values())))).unsqueeze(0).to(self.device)
                    if "visual" in obs_t:
                        raw_img = obs_t["visual"].unsqueeze(0).to(self.device)  # (1, C, H, W) in [-1, 1]
                        img_01 = (raw_img.clamp(-1.0, 1.0) + 1.0) / 2.0
                        image = F.interpolate(img_01, size=(policy_img_size, policy_img_size), mode="bilinear", align_corners=False)
                    else:
                        image = torch.zeros(1, 3, policy_img_size, policy_img_size, device=self.device)
                    act_t = self.policy.predict(state, image).squeeze(0)
                    step_actions.append(act_t)
                actions = torch.stack(step_actions, dim=0).unsqueeze(0)  # (1, horizon, action_dim)
            else:
                raise ValueError(f"Unknown policy_mode: '{self.policy_mode}'. Expected 'dummy', 'gt', or 'ckpt'.")

            # Perform rollout with different numbers of history frames
            for past in num_past:
                n_past, postfix = past

                # Prepare initial observations
                obs_0 = {}
                for k in obs_subsampled.keys():
                    obs_0[k] = obs_subsampled[k][:n_past].unsqueeze(0).to(self.device)

                # Perform model rollout
                z_obses, z = self.wm.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                
                # Compute prediction error
                div_loss = self._compute_error(z_obs_last, z_g)

                # Collect logs
                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[log_key].append(div_loss[k])
                    else:
                        logs[log_key] = [div_loss[k]]

                # Decode and save visualizations if decoder available
                if self.wm.decoder is not None:
                    # pred_frames: (t, c, h, w) — all decoded frames for first batch item
                    pred_frames = self.wm.decode_obs(z_obses)[0]["visual"][0].cpu()
                    base_path = os.path.join(plotting_dir, f"rollout_{idx:03d}{postfix}")
                    self._save_rollout_outputs(obs_subsampled["visual"], pred_frames, base_path)

                    # Score the decoded rollout with Robometer
                    rb_key = f"{idx:03d}{postfix}"
                    frames_uint8 = self._frames_to_uint8(pred_frames)
                    progress_scores, success_scores = [], []
                    if rb_model is None:
                        print(f"[Warning] Robometer model not loaded; skipping scoring for rollout {rb_key}.")
                    else:
                        progress_arr, success_arr = infer_robometer(
                            rb_config, rb_tokenizer, rb_processor, rb_model,
                            frames_uint8, robometer_prompt,
                        )
                        progress_scores = progress_arr.tolist()
                        success_scores = success_arr.tolist()
                    all_rb_results[rb_key] = {
                        "progress": progress_scores,
                        "success": success_scores,
                        "final_success": success_scores[-1] if success_scores else None,
                        "max_success": float(max(success_scores)) if success_scores else None,
                        "mean_success": float(np.mean(success_scores)) if success_scores else None,
                    }

                rollout_results.append({
                    "trajectory_idx": idx,
                    "horizon": horizon,
                    "n_past": n_past,
                    "postfix": postfix,
                    "z_obses": z_obses.detach().cpu() if hasattr(z_obses, 'detach') else z_obses,
                    "z_g": z_g.detach().cpu() if hasattr(z_g, 'detach') else z_g,
                    "errors": div_loss,
                })

        # Average logs
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }

        # Save Robometer progress
        scores_path = os.path.join(output_dir, "robometer_progress.json")
        with open(scores_path, "w") as f:
            json.dump({"rollouts": all_rb_results, "n_rollouts": self.n_rollouts}, f, indent=2)
        print(f"Saved Robometer progress to {scores_path}")

        # Save results
        self.save_results({
            "logs": logs,
            "rollout_results": rollout_results,
            "n_rollouts": self.n_rollouts,
            "rollout_length": self.rollout_length,
        })

        self.wandb_run.log(logs)
        return logs

    def perform_closed_loop_policy_rollout(self, rb_components=None):
        """
        (a) Run the LeRobot PushT diffusion policy closed-loop on world model decoded images:
            at each step, decode the WM latent to an image, feed image + GT proprio to the
            diffusion policy to get an action, then step the WM with that action.
        (b) Save the resulting predicted-frame trajectory as a video.
        (c) Run each frame of the video through Robometer to score task progress.

        Args:
            rb_components: optional 4-tuple (rb_config, rb_tokenizer, rb_processor, rb_model)
                           pre-loaded by the caller to avoid reloading across rollout lengths.
                           If None, Robometer is loaded here.

        Returns:
            all_progress: list of per-rollout dicts with progress/success scores.
        """
        np.random.seed(self.cfg_dict.get("seed", 0))
        output_dir = self.cfg_dict.get("saved_folder", ".")
        plotting_dir = os.path.join(output_dir, "rollout_plots")
        os.makedirs(plotting_dir, exist_ok=True)

        # --- (a) Load diffusion policy ---
        policy_model_name = self.cfg_dict.get("policy_model_name", "lerobot/diffusion_pusht")
        policy = PolicyWrapper(model_name=policy_model_name)

        # --- (c) Load Robometer (or reuse pre-loaded components) ---
        if rb_components is not None:
            rb_config, rb_tokenizer, rb_processor, rb_model = rb_components
        else:
            rb_config, rb_tokenizer, rb_processor, rb_model = load_robometer()
        robometer_prompt = self.cfg_dict.get(
            "robometer_prompt", "Push the T block to the goal position."
        )

        # LeRobot diffusion_pusht expects 96×96 images
        policy_img_size = self.cfg_dict.get("policy_img_size", 96)
        n_past = self.cfg_dict.get("num_hist", 2)

        all_progress = []

        for idx in range(self.n_rollouts):
            # Sample a trajectory from the validation dataset
            traj_idx = np.random.randint(0, len(self.dset))
            obs, act, state, _ = self.dset[traj_idx]
            # act: (T, orig_action_dim), e.g. (T, 2) for PushT
            orig_action_dim = act.shape[-1]
            # WM groups frameskip consecutive actions into one vector
            wm_action_dim = orig_action_dim * self.frameskip

            if obs["visual"].shape[0] < n_past + 1:
                continue

            # Initial observation window: (1, n_past, C, H, W)
            obs_0 = {k: v[:n_past].unsqueeze(0).to(self.device) for k, v in obs.items()}
            # Zero initial actions for the encoder (only used for conditioning)
            act_0 = torch.zeros(1, n_past, wm_action_dim, device=self.device)

            decoded_frames = []

            with torch.no_grad():
                z = self.wm.encode(obs_0, act_0)  # (1, n_past, num_patches+2, emb_dim)

                for t in range(self.rollout_length):
                    # Decode the most recent WM latent to a pixel image
                    z_obs_cur, _ = self.wm.separate_emb(z[:, -1:])
                    decoded, _ = self.wm.decode_obs(z_obs_cur)
                    img = decoded["visual"][0, 0]  # (C, H, W) in [-1, 1]
                    decoded_frames.append(img.cpu())

                    # Convert [-1,1] → [0,1] for policy input
                    img_01 = (img.clamp(-1.0, 1.0) + 1.0) / 2.0  # (C, H, W)
                    # Resize to the policy's expected spatial resolution
                    img_policy = F.interpolate(
                        img_01.unsqueeze(0),
                        size=(policy_img_size, policy_img_size),
                        mode="bilinear",
                        align_corners=False,
                    )  # (1, C, policy_img_size, policy_img_size)

                    # Use ground-truth proprio as the agent state (WM has no proprio decoder)
                    state_idx = min(t + n_past - 1, obs["proprio"].shape[0] - 1)
                    state_input = obs["proprio"][state_idx].unsqueeze(0).to(self.device)  # (1, proprio_dim)

                    print("State input shape:", state_input.shape)
                    print("Image for policy shape:", img_policy.shape)

                    # Query the diffusion policy
                    action = policy.predict(
                        observation=state_input,
                        image=img_policy,
                    )  # (1, orig_action_dim)

                    # Tile action to match WM's frameskip-grouped action dimension
                    action_grouped = action.repeat(1, self.frameskip).unsqueeze(1)  # (1, 1, wm_action_dim)

                    # Step the world model: predict next latent and inject the action
                    z_pred = self.wm.predict(z[:, -self.wm.num_hist:])
                    z_new = z_pred[:, -1:, ...]  # (1, 1, num_patches+2, emb_dim)
                    z_new = self.wm.replace_actions_from_z(z_new, action_grouped)
                    z = torch.cat([z, z_new], dim=1)

                # Decode and append the final predicted frame
                z_obs_last, _ = self.wm.separate_emb(z[:, -1:])
                decoded_last, _ = self.wm.decode_obs(z_obs_last)
                decoded_frames.append(decoded_last["visual"][0, 0].cpu())

            pred_frames = torch.stack(decoded_frames, dim=0)  # (T+1, C, H, W)

            # --- (b) Save trajectory as video ---
            base_path = os.path.join(plotting_dir, f"policy_rollout_{idx:03d}")
            gt_frames = obs["visual"][: len(decoded_frames)]  # (T+1, C, H, W)
            self._save_rollout_outputs(gt_frames, pred_frames, base_path)

            # --- (c) Score the full rollout with Robometer (video model) ---
            frames_uint8 = self._frames_to_uint8(pred_frames)  # (T+1, H, W, C) uint8
            progress_scores = []
            success_scores = []
            if rb_model is None:
                print("[Warning] Robometer model not loaded; skipping Robometer scoring.")
            else:
                progress_arr, success_arr = infer_robometer(
                    rb_config, rb_tokenizer, rb_processor, rb_model,
                    frames_uint8, robometer_prompt,
                )
                progress_scores = progress_arr.tolist()
                success_scores = success_arr.tolist()

            all_progress.append({
                "progress": progress_scores,
                "success": success_scores,
                "final_success": success_scores[-1] if success_scores else None,
                "max_success": float(max(success_scores)) if success_scores else None,
                "mean_success": float(np.mean(success_scores)) if success_scores else None,
            })
            print(f"[Rollout {idx}] Robometer progress over {len(progress_scores)} frames: {progress_scores}")

        # Save progress + success scores to disk
        scores_path = os.path.join(output_dir, "robometer_progress.json")
        with open(scores_path, "w") as f:
            json.dump({"rollouts": all_progress, "n_rollouts": self.n_rollouts}, f, indent=2)
        print(f"Saved Robometer progress to {scores_path}")

        return all_progress

    def save_results(self, results):
        """
        Save rollout results to disk.
        """
        output_dir = self.cfg_dict.get("saved_folder", ".")
        
        # Save as pickle
        results_path = os.path.join(output_dir, "rollout_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved rollout results to {results_path}")

        # Save metadata as JSON
        metadata = {
            "rollout_length": results.get("rollout_length", self.rollout_length),
            "n_rollouts": results.get("n_rollouts", self.n_rollouts),
        }
        metadata_path = os.path.join(output_dir, "rollout_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved rollout metadata to {metadata_path}")

        # Save logs
        if "logs" in results:
            logs_path = os.path.join(output_dir, "rollout_logs.json")
            with open(logs_path, "w") as f:
                json.dump(results["logs"], f, indent=2)
            print(f"Saved rollout logs to {logs_path}")

    def _frames_to_uint8(self, frames):
        """Convert (T, C, H, W) tensor/array in [-1, 1] to (T, H, W, C) uint8 numpy."""
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().float().numpy()
        if frames.ndim == 4 and frames.shape[1] in (1, 3):
            frames = frames.transpose(0, 2, 3, 1)
        frames = np.clip(frames * 0.5 + 0.5, 0.0, 1.0)
        return (frames * 255).astype(np.uint8)

    def _save_rollout_outputs(self, gt_frames, pred_frames, base_path):
        """
        Save ground truth and predicted rollout frames as videos and .npy arrays.

        gt_frames:   (T, C, H, W) tensor, normalized in [-1, 1]
        pred_frames: (T, C, H, W) tensor, normalized in [-1, 1]
        base_path:   path prefix; files are written as
                       <base_path>_gt.mp4 / _gt.npy
                       <base_path>_pred.mp4 / _pred.npy
        """
        fps = self.cfg_dict.get("video_fps", 4)
        gt_np   = self._frames_to_uint8(gt_frames)    # (T, H, W, C) uint8
        pred_np = self._frames_to_uint8(pred_frames)  # (T, H, W, C) uint8

        # Save .npy arrays
        np.save(f"{base_path}_gt.npy",   gt_np)
        np.save(f"{base_path}_pred.npy", pred_np)

        # Save videos
        if HAVE_IMAGEIO:
            imageio.mimsave(f"{base_path}_gt.mp4",   gt_np,   fps=fps)
            imageio.mimsave(f"{base_path}_pred.mp4", pred_np, fps=fps)
            print(f"Saved ground-truth video : {base_path}_gt.mp4")
            print(f"Saved predicted video    : {base_path}_pred.mp4")
        elif HAVE_CV2:
            for tag, frames in [("gt", gt_np), ("pred", pred_np)]:
                h, w = frames.shape[1], frames.shape[2]
                out = cv2.VideoWriter(
                    f"{base_path}_{tag}.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (w, h),
                )
                for frame in frames:
                    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                out.release()
                print(f"Saved {tag} video: {base_path}_{tag}.mp4")
        else:
            print(f"[Warning] No video library (imageio/cv2) available; skipping video save.")

        print(f"Saved ground-truth array : {base_path}_gt.npy")
        print(f"Saved predicted array    : {base_path}_pred.npy")

    def _compute_error(self, z_obs_last, z_g):
        """
        Compute prediction error between predicted and goal embeddings.
        """
        errors = {}
        for k in z_obs_last.keys():
            if k in z_g:
                # L2 distance
                err = torch.norm(z_obs_last[k] - z_g[k], p=2, dim=-1).mean()
                errors[k] = err.item() if hasattr(err, 'item') else float(err)
        return errors

def _load_model(model_ckpt, train_cfg, num_action_repeat, device):
    """
    Load model from checkpoint with PyTorch 2.6+ compatibility (weights_only=False)
    and direct encoder instantiation to avoid hydra.utils.instantiate type errors.
    """
    result = {}
    if model_ckpt.exists():
        payload = torch.load(model_ckpt, map_location=device, weights_only=False)
        for k, v in payload.items():
            if k in ALL_MODEL_KEYS:
                result[k] = v.to(device)
        result["epoch"] = payload["epoch"]
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        # Use OmegaConf.to_container to get a plain dict, then import and
        # instantiate the encoder class directly — avoids Hydra's type-annotation
        # resolution which triggers `type | None` errors when the hub code is
        # loaded under certain environments.
        enc_cfg = OmegaConf.to_container(train_cfg.encoder, resolve=True)
        target = enc_cfg.pop("_target_")
        module_name, class_name = target.rsplit(".", 1)
        import importlib
        cls = getattr(importlib.import_module(module_name), class_name)
        result["encoder"] = cls(**enc_cfg)

    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            result["decoder"] = ckpt["decoder"] if isinstance(ckpt, dict) else ckpt
        else:
            raise ValueError("Decoder not in checkpoint and no decoder_path in config")
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


def _load_dataset_with_legacy_target_fallback(model_cfg):
    """
    Load dataset config and transparently remap legacy Hydra target paths.

    Older checkpoints may store `env.dataset._target_` as `datasets.*` while this
    repo now uses `dino_datasets.*`. This can fail in containerized runs where
    `datasets` is not importable from the working directory/PYTHONPATH.
    """
    def _rewrite_legacy_targets(node):
        rewritten_any = False
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "_target_" and isinstance(value, str) and value.startswith("datasets."):
                    node[key] = value.replace("datasets.", "dino_datasets.", 1)
                    rewritten_any = True
                else:
                    rewritten_any = _rewrite_legacy_targets(value) or rewritten_any
        elif isinstance(node, list):
            for item in node:
                rewritten_any = _rewrite_legacy_targets(item) or rewritten_any
        return rewritten_any

    dataset_cfg = model_cfg.env.dataset
    raw_dataset_cfg = OmegaConf.to_container(dataset_cfg, resolve=False)
    rewrote_legacy_targets = _rewrite_legacy_targets(raw_dataset_cfg)
    normalized_dataset_cfg = OmegaConf.create(raw_dataset_cfg)

    if rewrote_legacy_targets:
        log.warning(
            "Rewriting legacy dataset Hydra targets from 'datasets.*' to "
            "'dino_datasets.*' for compatibility with the renamed package."
        )

    try:
        return hydra.utils.call(
            normalized_dataset_cfg,
            num_hist=model_cfg.num_hist,
            num_pred=model_cfg.num_pred,
            frameskip=model_cfg.frameskip,
        )
    except Exception as exc:
        target = dataset_cfg.get("_target_", None)
        if isinstance(target, str) and target.startswith("datasets."):
            fallback_target = target.replace("datasets.", "dino_datasets.", 1)
            log.warning(
                "Dataset target '%s' failed to resolve; retrying with '%s'. "
                "This commonly happens with legacy checkpoints in containerized runs.",
                target,
                fallback_target,
            )
            return hydra.utils.call(
                normalized_dataset_cfg,
                num_hist=model_cfg.num_hist,
                num_pred=model_cfg.num_pred,
                frameskip=model_cfg.frameskip,
            )
        raise exc



def rollout_main(cfg_dict):
    """
    Main function to perform model-based rollout with ground truth actions.
    Uses the world model to predict observations from actions.
    Sweeps over rollout_lengths (default [10, 20, 40]) and writes outputs
    into per-length subdirectories rl{N}/ under saved_folder.
    """
    base_output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if cfg_dict.get("wandb_logging", False):
        wandb_run = wandb.init(
            project="rollout_model_predictions", config=cfg_dict
        )
    else:
        wandb_run = None

    # Load model configuration and checkpoint (shared across all lengths)
    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"

    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict.get("seed", 0))

    # Load dataset
    _, dset = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset["valid"]

    # Load model
    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict.get('model_epoch', 'final')}.pth"
    )
    model = _load_model(model_ckpt, model_cfg, num_action_repeat, device=device)
    model.eval()

    rollout_mode = cfg_dict.get("rollout_mode", "open_loop")

    # Determine lengths to sweep; fall back to the single configured length
    rollout_lengths = cfg_dict.get("rollout_lengths", None)
    if rollout_lengths is None:
        rollout_lengths = [cfg_dict.get("rollout_length", 10)]

    # Load Robometer once (4B model — expensive to reload per length)
    rb_components = load_robometer()

    all_logs = {}
    for rl in rollout_lengths:
        # Build per-length config and output directory
        cfg_rl = dict(cfg_dict)
        cfg_rl["rollout_length"] = rl
        cfg_rl["saved_folder"] = os.path.join(base_output_dir, f"rl{rl}")
        os.makedirs(cfg_rl["saved_folder"], exist_ok=True)

        rollout_workspace = RolloutWorkspace(
            cfg_dict=cfg_rl,
            wm=model,
            dset=dset,
            env=None,
            env_name=model_cfg.env.name,
            frameskip=model_cfg.frameskip,
            wandb_run=wandb_run,
        )

        with torch.no_grad():
            if rollout_mode == "closed_loop_policy":
                logs = rollout_workspace.perform_closed_loop_policy_rollout(
                    rb_components=rb_components
                )
            else:
                logs = rollout_workspace.perform_rollout(
                    rb_components=rb_components
                )

        all_logs[f"rl{rl}"] = logs

    if wandb_run:
        wandb_run.finish()

    return all_logs


@hydra.main(config_path="conf", config_name="plan_pusht")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Rollout result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = cfg_dict.get("wandb_logging", False)
    
    rollout_main(cfg_dict)



if __name__ == "__main__":
    main()




