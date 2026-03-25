from __future__ import annotations

import torch

import numpy as np

from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.utils import add_envs_task, preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors

from use_robometer import load_robometer, infer_robometer

# i love vibe coding
class LeRobotSingleStepAgent:
    """Turns a LeRobot policy into a single-step (obs -> action) callable for a Gymnasium env."""

    def __init__(
        self,
        *,
        policy_path: str = "lerobot/diffusion_pusht",
        env_type: str = "pusht",
        device: str = "cuda",
        use_amp: bool = False,
        rename_map=None,
        trust_remote_code: bool = False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but not available.")
        self.device = torch.device("cuda")
        self.use_amp = use_amp

        # Minimal cfg-like dicts (LeRobot internally is config-driven)
        self.env_cfg = {"type": env_type}
        self.policy_cfg = {
            "path": policy_path,
            "device": str(self.device),
            "use_amp": use_amp,
        }

        # Policy
        self.policy = make_policy(cfg=self.policy_cfg, env_cfg=self.env_cfg, rename_map=rename_map)
        self.policy.eval()

        # Policy pre/post processors (device placement, formatting, etc.)
        preprocessor_overrides = {
            # match eval script behavior: force inference device to detected hardware
            "device_processor": {"device": str(self.policy.config.device)},
            "rename_observations_processor": {"rename_map": rename_map},
        }
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy_cfg,
            pretrained_path=getattr(self.policy_cfg, "pretrained_path", None),
            preprocessor_overrides=preprocessor_overrides,
        )

        # Env-specific processors (safe to include; important for some envs, usually no-op for pusht)
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=self.policy_cfg
        )

        # Make policy state clean for inference
        self.policy.reset()

    def act(self, *, env, obs, deterministic: bool = True) -> np.ndarray:
        """
        Args:
            env: The *vector env* or env wrapper used in your loop (needed for add_envs_task()).
                 If you're using a non-vector env, you can skip add_envs_task or adapt it.
            obs: Raw observation as returned by env.reset()/env.step()
            deterministic: kept for API symmetry; diffusion policies may ignore this.
        Returns:
            action as numpy array with shape (batch, action_dim) if env is vectorized,
            or (action_dim,) if you squeeze it yourself.
        """
        # Convert numpy->torch dict, ensure batch dimension, etc.
        obs = preprocess_observation(obs)

        # Add inferred "task" from env attrs (eval does this; for PushT it’s usually fine)
        obs = add_envs_task(env, obs)

        # Env-specific + policy preprocessing
        obs = self.env_preprocessor(obs)
        obs = self.preprocessor(obs)

        with torch.inference_mode():
            ctx = (
                torch.autocast(device_type=self.device.type)
                if self.use_amp and self.device.type in ("cuda", "mps")
                else torch.no_grad()
            )
            with ctx:
                action = self.policy.select_action(obs)

        # Postprocess into env action space
        action = self.postprocessor(action)
        action = self.env_postprocessor({"action": action})["action"]

        return action.detach().to("cpu").numpy()


def load_policy_ckpt(path, cfg_dict=None):
    """
    Load a LeRobot policy checkpoint as a LeRobotSingleStepAgent.
    """
    cfg_dict = cfg_dict or {}
    return LeRobotSingleStepAgent(
        policy_path=path,
        env_type=cfg_dict.get("env_type", "pusht"),
        device=cfg_dict.get("device", "cuda"),
        use_amp=cfg_dict.get("use_amp", False),
        trust_remote_code=cfg_dict.get("trust_remote_code", False),
    )


# --- example usage ---
if __name__ == "__main__":
    import gymnasium as gym
    from lerobot.envs.factory import make_env

    # Create PushT vector env with batch_size=1 (LeRobot’s pipeline is batch-first)
    env = make_env(env_cfg={"type": "pusht"}, n_envs=1, use_async_envs=False, trust_remote_code=False)

    agent = LeRobotSingleStepAgent(
        policy_path="lerobot/diffusion_pusht",
        env_type="pusht",
        device="cuda",
        use_amp=False,
    )

    obs, info = env.reset(seed=0)
    action = agent.act(env=env, obs=obs)
    print("action shape:", action.shape)  # typically (1, action_dim)

    obs, reward, terminated, truncated, info = env.step(action)
    env.close()