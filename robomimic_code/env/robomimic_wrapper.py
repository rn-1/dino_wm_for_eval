"""
robomimic_wrapper.py — Gym-compatible environment wrapper for Robomimic tasks.

Wraps robomimic's EnvRobosuite to expose the same interface as PushTWrapper:
  - prepare(seed, init_state)  → (obs, state)
  - step_multiple(actions)     → (obses, rewards, dones, infos)
  - rollout(seed, init_state, actions) → (obses, states)

Observations returned by the wrapper follow dino_wm conventions:
  obs["visual"]  : (H, W, C) uint8  — first camera image
  obs["proprio"] : (proprio_dim,) float — concatenated robot state

Requires:
  pip install robomimic robosuite
"""

import os
import sys
import numpy as np

_DINO_WM_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _DINO_WM_ROOT not in sys.path:
    sys.path.insert(0, _DINO_WM_ROOT)

from utils import aggregate_dct


class RobomimicWrapper:
    """
    Thin wrapper around a robomimic EnvRobosuite environment.

    Args:
        dataset_path:   Path to the Robomimic HDF5 file.  Used to read
                        env metadata so the sim is created identically to
                        how the data was collected.
        camera_names:   Cameras whose images to include in observations.
        proprio_keys:   Robot-state keys to concatenate into the proprio vector.
        img_size:       (H, W) to which images are resized (None = raw resolution).
        render_offscreen: Whether to render images off-screen (headless server).
    """

    def __init__(
        self,
        dataset_path: str,
        camera_names=None,
        proprio_keys=None,
        img_size=None,
        render_offscreen: bool = True,
    ):
        try:
            import robomimic.utils.env_utils as EnvUtils
            import robomimic.utils.file_utils as FileUtils
        except ImportError as e:
            raise ImportError(
                "robomimic is required for RobomimicWrapper. "
                "Install with: pip install robomimic"
            ) from e

        self.camera_names = camera_names or ["agentview"]
        self.proprio_keys = proprio_keys or [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qwidth",
        ]
        self.img_size = img_size  # (H, W) or None

        # Build env from dataset metadata so settings match the data collection
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        self._env = EnvUtils.create_env_from_metadata(
            env_meta,
            render=False,
            render_offscreen=render_offscreen,
            use_image_obs=True,
            camera_names=self.camera_names,
            camera_height=img_size[0] if img_size else 84,
            camera_width=img_size[1] if img_size else 84,
        )

        self.action_dim = self._env.action_dimension

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _parse_obs(self, raw_obs: dict) -> dict:
        """Convert a raw robomimic obs dict → dino_wm obs dict."""
        # Visual: take the first camera
        cam_key = self.camera_names[0] + "_image"
        image = raw_obs[cam_key]  # (H, W, C) uint8

        # Proprio: concatenate requested keys
        parts = []
        for pk in self.proprio_keys:
            if pk in raw_obs:
                val = np.atleast_1d(raw_obs[pk]).astype(np.float32)
                parts.append(val)
        proprio = np.concatenate(parts) if parts else np.zeros(1, dtype=np.float32)

        return {"visual": image, "proprio": proprio}

    def _get_state(self) -> np.ndarray:
        """Return the current simulator state as a flat numpy array."""
        return self._env.get_state()["states"]

    # ------------------------------------------------------------------
    # Core interface (mirrors PushTWrapper)
    # ------------------------------------------------------------------

    def prepare(self, seed: int, init_state=None):
        """
        Reset the environment.

        Args:
            seed:       RNG seed passed to the env (for any randomness).
            init_state: If provided, reset the simulator to this state vector
                        (as returned by _get_state / rollout).

        Returns:
            obs:   dino_wm obs dict for the initial frame
            state: flat state numpy array
        """
        if init_state is not None:
            raw_obs = self._env.reset_to({"states": init_state})
        else:
            raw_obs = self._env.reset()

        obs = self._parse_obs(raw_obs)
        state = self._get_state()
        return obs, state

    def step_multiple(self, actions: np.ndarray):
        """
        Execute a sequence of actions.

        Args:
            actions: (T, action_dim) array

        Returns:
            obses:   dict of (T, ...) arrays
            rewards: (T,)
            dones:   (T,)
            infos:   dict of (T, ...) arrays
        """
        obses, rewards, dones, infos = [], [], [], []
        for action in actions:
            raw_obs, reward, done, info = self._env.step(action)
            obs = self._parse_obs(raw_obs)
            info["state"] = self._get_state()
            obses.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed: int, init_state, actions: np.ndarray):
        """
        Full rollout from init_state with the given action sequence.

        Args:
            seed:       RNG seed.
            init_state: Initial simulator state (from _get_state).
            actions:    (T, action_dim) numpy array.

        Returns:
            obses:  dict of (T+1, ...) arrays — initial frame + T stepped frames
            states: (T+1, state_dim) array
        """
        obs0, state0 = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)

        # Prepend the initial observation and state
        for k in obses:
            obses[k] = np.concatenate(
                [np.expand_dims(obs0[k], 0), obses[k]], axis=0
            )
        states = np.concatenate(
            [state0[None], infos["state"]], axis=0
        )
        return obses, states

    def eval_state(self, goal_state: np.ndarray, cur_state: np.ndarray) -> dict:
        """
        Check whether cur_state is close enough to goal_state.
        Delegates to the underlying robomimic env if it exposes is_success().

        Returns a dict with at minimum 'success' (bool) and 'state_dist' (float).
        """
        state_dist = float(np.linalg.norm(goal_state - cur_state))

        try:
            # robomimic envs expose task-specific success criteria
            success = bool(self._env.is_success()["task"])
        except Exception:
            # Fallback: use a simple distance threshold
            success = state_dist < 0.05

        return {"success": success, "state_dist": state_dist}

    def sample_random_init_goal_states(self, seed: int):
        """
        Return two random (init, goal) simulator states by resetting the env twice.
        """
        rng = np.random.RandomState(seed)

        _, init_state = self.prepare(int(rng.randint(0, 2**31)))
        _, goal_state = self.prepare(int(rng.randint(0, 2**31)))
        return init_state, goal_state

    # Allow use as a context manager (ensures viewer/sim cleanup)
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
