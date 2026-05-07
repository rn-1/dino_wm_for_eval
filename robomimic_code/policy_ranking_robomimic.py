#!/usr/bin/env python3
"""
policy_ranking_robomimic.py — Rank inhibited robomimic policies using a
dino_wm world model trained on Robomimic data as a simulator.

Mirrors the structure of dino_wm/policy_ranking.py, but:
  * Loads a RobomimicDataset (HDF5) for trajectory sampling.
  * Wraps a robomimic-trained policy checkpoint (BC / Diffusion / etc.)
    in InhibitedRobomimicPolicyWrapper(c).
  * Generates actions from GT observations, runs them open-loop through
    the WM, and scores the WM-decoded video with Robometer
    (final-frame success probability).
  * Optional --run_gym path uses RobomimicWrapper for ground-truth sim
    rollouts and scores them with the same Robometer metric, so WM and GT
    bars are directly comparable.

Usage:
    python robomimic_code/policy_ranking_robomimic.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name robomimic \\
        --model_epoch latest \\
        --policy_ckpt /project2/.../robomimic_policy.pth \\
        --inhibition_coeffs 0.0 0.1 0.2 0.5 1.0 2.0 \\
        --n_eval 100 \\
        --rollout_length 20 \\
        --robometer_prompt "Pick up the red cube and place it in the bin." \\
        --output_dir /project2/.../eval_results/policy_ranking_robomimic \\
        --seed 42
"""

import os
import re
import sys
import glob
import json
import argparse
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

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from rollout_robomimic import RobomimicPolicyWrapper
from use_robometer import load_robometer, infer_robometer
from utils import seed


# ---------------------------------------------------------------------------
# Logging helper  (mirrors policy_ranking.py)
# ---------------------------------------------------------------------------

class Tee:
    """Mirror stdout to a log file."""
    def __init__(self, path):
        self._file = open(path, "w", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self._file.write(data)
        self._stdout.write(data)

    def flush(self):
        self._file.flush()
        self._stdout.flush()

    def close(self):
        sys.stdout = self._stdout
        self._file.close()


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _to_hwc_uint8(frame_chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1, 1] → (H, W, C) uint8."""
    img = frame_chw.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().float().numpy()
    return (img * 255).astype(np.uint8)


def _frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """(T, C, H, W) in [-1, 1] → (T, H, W, C) uint8."""
    arr = frames.cpu().float().numpy()
    arr = arr.transpose(0, 2, 3, 1)
    arr = np.clip(arr * 0.5 + 0.5, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inhibited policy wrapper
# ---------------------------------------------------------------------------

class InhibitedRobomimicPolicyWrapper:
    """
    Wraps a RobomimicPolicyWrapper and scales its output actions by
    *inhibition_coeff* (multiplicative inhibition).  c=1.0 leaves the policy
    unchanged; c=0.0 zeroes out all actions; c=2.0 amplifies (overshoot).
    """

    def __init__(self, ckpt_path: str, device: str, inhibition_coeff: float):
        self.base = RobomimicPolicyWrapper(ckpt_path=ckpt_path, device=device)
        self.inhibition_coeff = inhibition_coeff

    def predict(self, obs_dict: dict) -> np.ndarray:
        action = self.base.predict(obs_dict)
        if self.inhibition_coeff is not None:
            action = action * float(self.inhibition_coeff)
        return action

    def reset(self):
        self.base.reset()


# ---------------------------------------------------------------------------
# Dataset / proprio helpers
# ---------------------------------------------------------------------------

def _resolve_proprio_dims(data_path: str, demo_key: str, proprio_keys):
    """Peek at one demo to determine per-key proprio component dimensions."""
    import h5py
    dims = []
    with h5py.File(data_path, "r") as f:
        demo = f["data"][demo_key]
        for pk in proprio_keys:
            if pk in demo["obs"]:
                arr = demo["obs"][pk][:1]
                dim = arr.shape[-1] if arr.ndim > 1 else 1
            else:
                dim = 0
            dims.append(dim)
    return dims


def _denormalize_proprio(proprio_norm: torch.Tensor,
                         proprio_mean: torch.Tensor,
                         proprio_std: torch.Tensor) -> np.ndarray:
    """Undo dataset normalisation: (x - mean) / std → x * std + mean."""
    return (proprio_norm * proprio_std + proprio_mean).cpu().numpy()


def _build_robomimic_obs_dict(image_uint8_hwc: np.ndarray,
                              proprio_denorm: np.ndarray,
                              camera_names,
                              proprio_keys,
                              proprio_dims):
    """
    Construct a robomimic-style obs dict expected by `policy(obs_dict)`.

    image_uint8_hwc : (H, W, C) uint8
    proprio_denorm  : (proprio_total_dim,) float — denormalised
    """
    cam_key = camera_names[0] + "_image"
    obs_dict = {cam_key: image_uint8_hwc[None]}  # (1, H, W, C)

    cursor = 0
    for pk, dim in zip(proprio_keys, proprio_dims):
        if dim == 0:
            continue
        obs_dict[pk] = proprio_denorm[cursor: cursor + dim][None]  # (1, dim)
        cursor += dim
    return obs_dict


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def _ckpt_label(path: str) -> str:
    """Short display label for a policy checkpoint filename."""
    name = os.path.basename(path)
    m = re.search(r"epoch[=_-]?(\d+)", name, flags=re.IGNORECASE)
    if m:
        return f"ep{int(m.group(1))}"
    return os.path.splitext(name)[0]


def sample_valid_trajectory(dset, rollout_length, frameskip, rng):
    """Return a trajectory from a shuffled pass that is long enough."""
    min_raw = rollout_length * frameskip + 1
    indices = list(range(len(dset)))
    rng.shuffle(indices)
    for idx in indices:
        obs, act, *_ = dset[idx]
        if obs["visual"].shape[0] >= min_raw and act.shape[0] >= rollout_length * frameskip:
            return obs, act
    raise RuntimeError(
        f"No trajectory has >= {min_raw} raw frames for "
        f"rollout_length={rollout_length}, frameskip={frameskip}."
    )


# ---------------------------------------------------------------------------
# Robometer scoring
# ---------------------------------------------------------------------------

def _robometer_score(frames_chw: torch.Tensor,
                     robometer_bundle,
                     prompt: str,
                     device: torch.device) -> dict:
    """
    Score a rollout video with Robometer.

    Returns a dict with:
      success_final  : float in [0, 1]   — final-frame success probability
      progress_final : float in [0, 1]   — final-frame progress
      success_mean   : float in [0, 1]   — mean success across frames
      progress_mean  : float in [0, 1]
      success_seq    : list[float]       — full per-frame success array
      progress_seq   : list[float]
    """
    exp_config, tokenizer, processor, model = robometer_bundle
    if model is None:
        return {
            "success_final": 0.0, "progress_final": 0.0,
            "success_mean": 0.0,  "progress_mean": 0.0,
            "success_seq": [],    "progress_seq": [],
        }

    frames_u8 = _frames_to_uint8(frames_chw)  # (T, H, W, C) uint8
    progress, success = infer_robometer(
        exp_config, tokenizer, processor, model,
        frames_u8, prompt, device=device,
    )

    return {
        "success_final":  float(success[-1])  if len(success)  else 0.0,
        "progress_final": float(progress[-1]) if len(progress) else 0.0,
        "success_mean":   float(np.mean(success))  if len(success)  else 0.0,
        "progress_mean":  float(np.mean(progress)) if len(progress) else 0.0,
        "success_seq":    [float(x) for x in success],
        "progress_seq":   [float(x) for x in progress],
    }


# ---------------------------------------------------------------------------
# Policy-driven WM rollout
# ---------------------------------------------------------------------------

def run_policy_wm_rollout(
    wm,
    obs,
    act,
    frameskip: int,
    num_hist: int,
    rollout_length: int,
    device: torch.device,
    policy: InhibitedRobomimicPolicyWrapper,
    camera_names,
    proprio_keys,
    proprio_dims,
    proprio_mean: torch.Tensor,
    proprio_std: torch.Tensor,
):
    """
    Open-loop WM rollout driven by a robomimic policy's actions on GT observations.

    For each raw timestep t, the policy receives the GT image and proprio at
    that step (denormalised) and produces a raw action.  Actions are grouped by
    frameskip and fed into the WM exactly as in the GT-action open-loop
    rollout, except the actions come from the policy (possibly inhibited).

    Returns:
        pred_frames : (rollout_length+1, C, H, W) float in [-1, 1]
        gt_frames   : (rollout_length+1, C, H, W) float in [-1, 1]
    """
    T_need = rollout_length * frameskip + 1

    obs_sub = {k: v[:T_need:frameskip] for k, v in obs.items()}

    n_raw_steps = rollout_length * frameskip
    visual_raw  = obs["visual"][:T_need]      # (T_need, C, H, W) in [-1, 1]
    proprio_raw = obs["proprio"][:T_need]     # (T_need, D)       (normalised)

    policy.reset()

    policy_actions = []
    with torch.no_grad():
        for t in range(n_raw_steps):
            # Image → uint8 HWC for robomimic policy
            frame  = visual_raw[t]
            img_01 = (frame.clamp(-1, 1) + 1.0) / 2.0
            img_hwc = (img_01.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            # Proprio → denormalised flat numpy → split by key
            proprio_denorm = _denormalize_proprio(
                proprio_raw[t], proprio_mean, proprio_std,
            )
            obs_dict = _build_robomimic_obs_dict(
                img_hwc, proprio_denorm,
                camera_names, proprio_keys, proprio_dims,
            )

            action = policy.predict(obs_dict)            # (action_dim,)
            policy_actions.append(torch.from_numpy(np.asarray(action, dtype=np.float32)))

    # Stack → (n_raw_steps, action_dim)
    act_policy = torch.stack(policy_actions, dim=0)

    # Group raw actions → (rollout_length, action_dim * frameskip)
    act_wm = rearrange(act_policy, "(h f) d -> h (f d)", f=frameskip)

    # ── WM rollout ─────────────────────────────────────────────────────────
    obs_0 = {k: v[:num_hist].unsqueeze(0).to(device) for k, v in obs_sub.items()}
    actions = act_wm.unsqueeze(0).to(device)

    with torch.no_grad():
        z_obses, _ = wm.rollout(obs_0, actions)
        pred_obs, _ = wm.decode_obs(z_obses)
        pred_frames = pred_obs["visual"][0].cpu()

    gt_frames = obs_sub["visual"]

    T = min(pred_frames.shape[0], gt_frames.shape[0])
    return pred_frames[-T:], gt_frames[-T:]


# ---------------------------------------------------------------------------
# Per-inhibition-level WM evaluation
# ---------------------------------------------------------------------------

def evaluate_inhibition_level_wm(
    inhibition_coeff: float,
    wm,
    dset,
    frameskip: int,
    num_hist: int,
    rollout_length: int,
    n_eval: int,
    device: torch.device,
    policy_ckpt: str,
    camera_names,
    proprio_keys,
    proprio_dims,
    proprio_mean,
    proprio_std,
    robometer_bundle,
    robometer_prompt: str,
    rng: np.random.RandomState,
    verbose: bool = True,
) -> dict:
    """Run n_eval WM rollouts at the given inhibition level; score with Robometer."""
    policy = InhibitedRobomimicPolicyWrapper(
        ckpt_path=policy_ckpt,
        device=str(device),
        inhibition_coeff=inhibition_coeff,
    )

    scores         = []   # final-frame success
    progress_finals = []  # final-frame progress
    success_means   = []
    progress_means  = []

    for i in range(n_eval):
        obs, act = sample_valid_trajectory(dset, rollout_length, frameskip, rng)
        pred_frames, _ = run_policy_wm_rollout(
            wm, obs, act, frameskip, num_hist, rollout_length, device,
            policy, camera_names, proprio_keys, proprio_dims,
            proprio_mean, proprio_std,
        )

        rmt = _robometer_score(pred_frames, robometer_bundle, robometer_prompt, device)
        scores.append(rmt["success_final"])
        progress_finals.append(rmt["progress_final"])
        success_means.append(rmt["success_mean"])
        progress_means.append(rmt["progress_mean"])

        if verbose:
            print(
                f"  c={inhibition_coeff:.3f}  [{i+1:3d}/{n_eval}]  "
                f"WM success_final={rmt['success_final']:.3f}  "
                f"progress_final={rmt['progress_final']:.3f}"
            )

    return {
        "inhibition_coeff": inhibition_coeff,
        "scores":           scores,
        "mean":             float(np.mean(scores)),
        "std":              float(np.std(scores)),
        "progress_finals":  progress_finals,
        "success_means":    success_means,
        "progress_means":   progress_means,
    }


# ---------------------------------------------------------------------------
# GT gym rollout (RobomimicWrapper)
# ---------------------------------------------------------------------------

def _build_robomimic_env(data_path: str,
                         camera_names,
                         proprio_keys,
                         img_size_hw):
    """Construct a RobomimicWrapper for GT sim rollouts."""
    from env.robomimic_wrapper import RobomimicWrapper
    return RobomimicWrapper(
        dataset_path=data_path,
        camera_names=list(camera_names),
        proprio_keys=list(proprio_keys),
        img_size=tuple(img_size_hw),
        render_offscreen=True,
    )


def _to_chw_signed(image_hwc_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, C) uint8 → (C, H, W) float in [-1, 1] (matches dataset normalisation)."""
    arr = image_hwc_uint8.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _build_policy_obs_from_sim(env, camera_names):
    """
    Build a robomimic-style obs dict using raw values from the underlying sim,
    so per-proprio-key components retain their native shapes (no flat-vector
    splitting required).
    """
    cam_key = camera_names[0] + "_image"
    sim_obs = env._env.get_observation()
    obs_dict = {cam_key: np.asarray(sim_obs[cam_key])[None]}
    for pk in env.proprio_keys:
        if pk in sim_obs:
            val = np.atleast_1d(sim_obs[pk]).astype(np.float32)
            obs_dict[pk] = val[None]
    return obs_dict


def run_single_gym_episode(
    policy: InhibitedRobomimicPolicyWrapper,
    env,
    rollout_length: int,
    camera_names,
):
    """
    Run one episode in the robomimic env.  Returns:
        frames_chw   : list of (C, H, W) float tensors in [-1, 1]
        env_success  : bool — robomimic's own task-success indicator at last step
    """
    obs0, _ = env.prepare(seed=int(np.random.randint(0, 2**31 - 1)))

    frames_chw = [_to_chw_signed(obs0["visual"])]
    env_success = False
    policy.reset()

    for _ in range(rollout_length):
        obs_dict = _build_policy_obs_from_sim(env, camera_names)

        action = policy.predict(obs_dict)
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)

        obses, _, dones, _ = env.step_multiple(action_np[None])
        frames_chw.append(_to_chw_signed(obses["visual"][0]))

        if dones[-1]:
            break

    try:
        env_success = bool(env._env.is_success()["task"])
    except Exception:
        env_success = False

    return frames_chw, env_success


def evaluate_inhibition_level_gym(
    inhibition_coeff: float,
    n_eval: int,
    rollout_length: int,
    device: torch.device,
    policy_ckpt: str,
    data_path: str,
    camera_names,
    proprio_keys,
    img_size_hw,
    robometer_bundle,
    robometer_prompt: str,
    verbose: bool = True,
) -> dict:
    """Run n_eval GT gym episodes at the given inhibition level; score with Robometer."""
    env = _build_robomimic_env(data_path, camera_names, proprio_keys, img_size_hw)
    policy = InhibitedRobomimicPolicyWrapper(
        ckpt_path=policy_ckpt,
        device=str(device),
        inhibition_coeff=inhibition_coeff,
    )

    scores         = []
    progress_finals = []
    success_means   = []
    progress_means  = []
    env_successes   = []

    for i in range(n_eval):
        frames_chw, env_success = run_single_gym_episode(
            policy, env, rollout_length, camera_names,
        )
        env_successes.append(bool(env_success))

        if len(frames_chw) >= 2:
            video = torch.stack(frames_chw, dim=0)  # (T, C, H, W)
            rmt = _robometer_score(video, robometer_bundle, robometer_prompt, device)
        else:
            rmt = {"success_final": 0.0, "progress_final": 0.0,
                   "success_mean": 0.0,  "progress_mean": 0.0}

        scores.append(rmt["success_final"])
        progress_finals.append(rmt["progress_final"])
        success_means.append(rmt["success_mean"])
        progress_means.append(rmt["progress_mean"])

        if verbose:
            print(
                f"  c={inhibition_coeff:.3f}  [{i+1:3d}/{n_eval}]  "
                f"GT success_final={rmt['success_final']:.3f}  "
                f"env_success={env_success}"
            )

    env.close()
    return {
        "inhibition_coeff": inhibition_coeff,
        "scores":           scores,
        "mean":             float(np.mean(scores)),
        "std":              float(np.std(scores)),
        "progress_finals":  progress_finals,
        "success_means":    success_means,
        "progress_means":   progress_means,
        "env_success_rate": float(np.mean(env_successes)) if env_successes else 0.0,
        "env_successes":    env_successes,
    }


# ---------------------------------------------------------------------------
# Plotting (mirrors policy_ranking.py)
# ---------------------------------------------------------------------------

def save_ranking_plot(results_by_coeff, output_dir, model_name, rollout_length,
                      gt_results_by_coeff=None, tick_labels=None,
                      x_label_str="inhibition coeff", out_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeffs    = [r["inhibition_coeff"] for r in results_by_coeff]
    wm_means  = [r["mean"] for r in results_by_coeff]
    wm_stds   = [r["std"]  for r in results_by_coeff]
    n_eval    = len(results_by_coeff[0]["scores"]) if results_by_coeff else 0
    if tick_labels is None:
        tick_labels = [f"c={c}" for c in coeffs]

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(coeffs))

    if gt_results_by_coeff is not None:
        gt_means = [r["mean"] for r in gt_results_by_coeff]
        gt_stds  = [r["std"]  for r in gt_results_by_coeff]
        width = 0.35
        wm_bars = ax.bar(x - width / 2, wm_means, yerr=wm_stds, capsize=4,
                         color="steelblue", edgecolor="white", width=width,
                         error_kw={"ecolor": "black", "elinewidth": 1.2},
                         label="World-model")
        gt_bars = ax.bar(x + width / 2, gt_means, yerr=gt_stds, capsize=4,
                         color="tomato", edgecolor="white", width=width,
                         error_kw={"ecolor": "black", "elinewidth": 1.2},
                         label="GT gym")
        for bar, val in zip(wm_bars, wm_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="steelblue")
        for bar, val in zip(gt_bars, gt_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7, color="tomato")
        all_means = wm_means + gt_means
        ax.legend(fontsize=9)
    else:
        width = 0.6
        bars = ax.bar(x, wm_means, yerr=wm_stds, capsize=4,
                      color="steelblue", edgecolor="white", width=width,
                      error_kw={"ecolor": "black", "elinewidth": 1.2})
        for bar, val in zip(bars, wm_means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(wm_stds + [1e-6]) * 0.05 + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        all_means = wm_means

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel("Robometer success_final")
    ax.set_title(
        f"{model_name} — Robometer success vs {x_label_str} "
        f"(rl={rollout_length}, n={n_eval})"
    )
    ax.set_ylim(0, max(max(all_means) * 1.2, 0.1))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"policy_ranking_robomimic{out_suffix}_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved ranking plot: {out_path}")
    return out_path


def save_line_plot(results_by_coeff, output_dir, model_name, rollout_length,
                   gt_results_by_coeff=None, tick_labels=None,
                   x_label_str="Inhibition coefficient (c)", out_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeffs   = np.array([r["inhibition_coeff"] for r in results_by_coeff])
    wm_means = np.array([r["mean"] for r in results_by_coeff])
    wm_stds  = np.array([r["std"]  for r in results_by_coeff])
    n_eval   = len(results_by_coeff[0]["scores"]) if results_by_coeff else 0

    if tick_labels is not None:
        x_values = np.arange(len(coeffs), dtype=float)
    else:
        x_values = coeffs

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x_values, wm_means, marker="o", color="steelblue", linewidth=2,
            label="WM (world model)")
    ax.fill_between(x_values, wm_means - wm_stds, wm_means + wm_stds,
                    alpha=0.25, color="steelblue")

    if gt_results_by_coeff is not None:
        gt_means = np.array([r["mean"] for r in gt_results_by_coeff])
        gt_stds  = np.array([r["std"]  for r in gt_results_by_coeff])
        ax.plot(x_values, gt_means, marker="s", color="tomato", linewidth=2,
                label="GT gym")
        ax.fill_between(x_values, gt_means - gt_stds, gt_means + gt_stds,
                        alpha=0.25, color="tomato")

    ax.set_xlabel(x_label_str)
    ax.set_ylabel("Robometer success_final")
    ax.set_title(
        f"{model_name} — Policy ranking WM vs GT (rl={rollout_length}, n={n_eval})"
    )
    if tick_labels is not None:
        ax.set_xticks(x_values)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(output_dir,
                            f"policy_ranking_robomimic_line{out_suffix}_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved line plot: {out_path}")
    return out_path


def save_comparison_plot(wm_results, gt_results, output_dir, model_name, rollout_length,
                          tick_labels=None, out_suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coeffs   = np.array([r["inhibition_coeff"] for r in wm_results])
    wm_means = np.array([r["mean"] for r in wm_results])
    wm_stds  = np.array([r["std"]  for r in wm_results])
    gt_means = np.array([r["mean"] for r in gt_results])
    gt_stds  = np.array([r["std"]  for r in gt_results])
    n_eval   = len(wm_results[0]["scores"]) if wm_results else 0
    if tick_labels is None:
        tick_labels = [f"c={c}" for c in coeffs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    for ax, means, stds, color, title in [
        (axes[0], wm_means, wm_stds, "steelblue", "World-model rollout"),
        (axes[1], gt_means, gt_stds, "tomato",    "GT gym rollout"),
    ]:
        x = np.arange(len(coeffs))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=color, edgecolor="white", width=0.6,
                      error_kw={"ecolor": "black", "elinewidth": 1.2})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        ax.set_ylabel("Robometer success_final")
        ax.set_title(f"{title} (n={n_eval})")
        ax.set_ylim(0, max(max(means) * 1.2, 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    fig.suptitle(
        f"{model_name} — WM vs GT ranking (rl={rollout_length})", fontsize=12
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir,
                            f"policy_ranking_robomimic_comparison{out_suffix}_rl{rollout_length}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved comparison plot: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def policy_ranking_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    if bool(cfg.policy_ckpt) == bool(cfg.policy_ckpt_dir):
        raise RuntimeError(
            "Provide exactly one of --policy_ckpt or --policy_ckpt_dir."
        )
    ckpt_mode = cfg.policy_ckpt_dir is not None
    out_suffix = "_ckptsweep" if ckpt_mode else ""

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(
        cfg.output_dir,
        f"policy_ranking_robomimic{out_suffix}_rl{cfg.rollout_length}.log",
    ))

    # ── World model ───────────────────────────────────────────────────────
    model_path = Path(cfg.ckpt_base_path) / "outputs" / cfg.model_name
    with open(model_path / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    _, dset_split = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset_split["valid"]
    print(f"Validation dataset: {len(dset)} trajectories")

    model_ckpt = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    wm = _load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    if wm.decoder is None:
        raise RuntimeError("World model has no decoder — pixel decoding required.")

    num_hist  = model_cfg.num_hist
    frameskip = model_cfg.frameskip

    # ── Robomimic-specific config (camera + proprio keys + data_path) ────
    env_dataset_cfg = OmegaConf.to_container(model_cfg.env.dataset, resolve=True)
    data_path = env_dataset_cfg["data_path"]
    dataset_camera_names = env_dataset_cfg.get("camera_names", ["agentview"])
    dataset_proprio_keys = env_dataset_cfg.get(
        "proprio_keys",
        ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qwidth"],
    )

    env_kwargs = OmegaConf.to_container(model_cfg.env.get("kwargs", {}), resolve=True) \
        if model_cfg.env.get("kwargs", None) is not None else {}
    env_camera_names = env_kwargs.get("camera_names", dataset_camera_names)
    env_proprio_keys = env_kwargs.get("proprio_keys", dataset_proprio_keys)
    env_img_size_hw  = tuple(env_kwargs.get("img_size", [224, 224]))

    # Per-key proprio dims (from one demo) — needed to split flat proprio
    proprio_dims = _resolve_proprio_dims(
        data_path, dset.demo_keys[0], dataset_proprio_keys,
    )
    print(f"Proprio keys: {dataset_proprio_keys}  dims={proprio_dims}")

    # Normalisation stats live on the dataset
    proprio_mean = dset.proprio_mean
    proprio_std  = dset.proprio_std

    # ── Robometer judge ───────────────────────────────────────────────────
    print("\nLoading Robometer ...")
    robometer_bundle = load_robometer(device=device)
    if robometer_bundle[3] is None:
        raise RuntimeError(
            "Robometer model failed to load — required for scoring. "
            "Check .hf_cache snapshot and the sibling 'robometer' repo."
        )

    print(f"\nWorld model '{cfg.model_name}' (epoch={cfg.model_epoch})")
    print(f"  rollout_length={cfg.rollout_length}, frameskip={frameskip}, num_hist={num_hist}")
    print(f"  Robometer prompt: {cfg.robometer_prompt!r}")
    print(f"  n_eval per level: {cfg.n_eval}")

    # ── Build the sweep: list of (label, ckpt_path, inhibition_coeff) ────
    if ckpt_mode:
        ckpt_paths = sorted(glob.glob(os.path.join(cfg.policy_ckpt_dir, cfg.ckpt_glob)))
        if not ckpt_paths:
            raise RuntimeError(
                f"No checkpoints matching {cfg.ckpt_glob!r} in {cfg.policy_ckpt_dir}"
            )
        print(f"  Policy ckpt dir: {cfg.policy_ckpt_dir}")
        print(f"  Found {len(ckpt_paths)} checkpoints (ckpt_glob={cfg.ckpt_glob!r}, "
              f"ckpt_inhibition_coeff={cfg.ckpt_inhibition_coeff})")
        for p in ckpt_paths:
            print(f"    {os.path.basename(p)}")
        sweep = [(_ckpt_label(p), p, cfg.ckpt_inhibition_coeff) for p in ckpt_paths]
    else:
        print(f"  Policy ckpt: {cfg.policy_ckpt}")
        print(f"  Inhibition coefficients: {cfg.inhibition_coeffs}")
        sweep = [(f"c={c}", cfg.policy_ckpt, float(c)) for c in cfg.inhibition_coeffs]

    print()

    # ── WM sweep ─────────────────────────────────────────────────────────
    results_by_coeff = []
    for label, ckpt_path, inhibition_coeff in sweep:
        print(f"\n=== [WM] {label}  (ckpt={os.path.basename(ckpt_path)}, "
              f"c={inhibition_coeff:.3f}) ===")
        result = evaluate_inhibition_level_wm(
            inhibition_coeff=inhibition_coeff,
            wm=wm,
            dset=dset,
            frameskip=frameskip,
            num_hist=num_hist,
            rollout_length=cfg.rollout_length,
            n_eval=cfg.n_eval,
            device=device,
            policy_ckpt=ckpt_path,
            camera_names=dataset_camera_names,
            proprio_keys=dataset_proprio_keys,
            proprio_dims=proprio_dims,
            proprio_mean=proprio_mean,
            proprio_std=proprio_std,
            robometer_bundle=robometer_bundle,
            robometer_prompt=cfg.robometer_prompt,
            rng=rng,
            verbose=True,
        )
        result["label"] = label
        result["ckpt_path"] = ckpt_path
        results_by_coeff.append(result)
        print(f"  → mean={result['mean']:.4f}  std={result['std']:.4f}")

    # ── GT gym sweep (optional) ──────────────────────────────────────────
    gt_results_by_coeff = None
    if getattr(cfg, "run_gym", False):
        print("\n" + "=" * 60)
        print("GT GYM ROLLOUT SWEEP")
        print("=" * 60)
        gt_results_by_coeff = []
        for label, ckpt_path, inhibition_coeff in sweep:
            print(f"\n=== [Gym] {label}  (ckpt={os.path.basename(ckpt_path)}, "
                  f"c={inhibition_coeff:.3f}) ===")
            gt_result = evaluate_inhibition_level_gym(
                inhibition_coeff=inhibition_coeff,
                n_eval=cfg.n_eval,
                rollout_length=cfg.rollout_length,
                device=device,
                policy_ckpt=ckpt_path,
                data_path=data_path,
                camera_names=env_camera_names,
                proprio_keys=env_proprio_keys,
                img_size_hw=env_img_size_hw,
                robometer_bundle=robometer_bundle,
                robometer_prompt=cfg.robometer_prompt,
                verbose=True,
            )
            gt_result["label"] = label
            gt_result["ckpt_path"] = ckpt_path
            gt_results_by_coeff.append(gt_result)
            print(
                f"  → mean={gt_result['mean']:.4f}  std={gt_result['std']:.4f}  "
                f"env_success_rate={gt_result['env_success_rate']:.3f}"
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Policy Ranking Summary (Robomimic) ===")
    label_col = "checkpoint" if ckpt_mode else "c"
    label_w   = max(10, max(len(r["label"]) for r in results_by_coeff))
    header = f"{label_col:>{label_w}}  {'WM mean':>10}  {'WM std':>8}"
    if gt_results_by_coeff:
        header += f"  {'GT mean':>10}  {'GT std':>8}  {'GT env_succ':>11}"
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results_by_coeff):
        line = f"{r['label']:>{label_w}}  {r['mean']:>10.4f}  {r['std']:>8.4f}"
        if gt_results_by_coeff:
            g = gt_results_by_coeff[i]
            line += (f"  {g['mean']:>10.4f}  {g['std']:>8.4f}  "
                     f"{g['env_success_rate']:>11.4f}")
        print(line)

    # ── Save plots ────────────────────────────────────────────────────────
    plot_kwargs = {}
    if ckpt_mode:
        plot_kwargs = {
            "tick_labels": [r["label"] for r in results_by_coeff],
            "out_suffix":  out_suffix,
        }
    save_ranking_plot(results_by_coeff, cfg.output_dir, cfg.model_name,
                      cfg.rollout_length, gt_results_by_coeff=gt_results_by_coeff,
                      x_label_str="checkpoint" if ckpt_mode else "inhibition coeff",
                      **plot_kwargs)
    save_line_plot(results_by_coeff, cfg.output_dir, cfg.model_name,
                   cfg.rollout_length, gt_results_by_coeff=gt_results_by_coeff,
                   x_label_str="Checkpoint" if ckpt_mode else "Inhibition coefficient (c)",
                   **plot_kwargs)
    if gt_results_by_coeff:
        save_comparison_plot(results_by_coeff, gt_results_by_coeff,
                             cfg.output_dir, cfg.model_name, cfg.rollout_length,
                             **plot_kwargs)

    # ── Save JSON ─────────────────────────────────────────────────────────
    summary = {
        "model_name":         cfg.model_name,
        "model_epoch":        cfg.model_epoch,
        "rollout_length":     cfg.rollout_length,
        "n_eval":             cfg.n_eval,
        "robometer_prompt":   cfg.robometer_prompt,
        "sweep_mode":         "checkpoint" if ckpt_mode else "inhibition",
        "wm_results": [
            {"label":            r["label"],
             "ckpt_path":        r.get("ckpt_path"),
             "inhibition_coeff": r["inhibition_coeff"],
             "mean":             r["mean"],
             "std":              r["std"]}
            for r in results_by_coeff
        ],
    }
    if ckpt_mode:
        summary["policy_ckpt_dir"] = cfg.policy_ckpt_dir
        summary["ckpt_glob"]       = cfg.ckpt_glob
        summary["ckpt_inhibition_coeff"] = cfg.ckpt_inhibition_coeff
    else:
        summary["policy_ckpt"]       = cfg.policy_ckpt
        summary["inhibition_coeffs"] = list(cfg.inhibition_coeffs)
    if gt_results_by_coeff:
        summary["gt_results"] = [
            {"label":             r["label"],
             "ckpt_path":         r.get("ckpt_path"),
             "inhibition_coeff":  r["inhibition_coeff"],
             "mean":              r["mean"],
             "std":               r["std"],
             "env_success_rate":  r["env_success_rate"]}
            for r in gt_results_by_coeff
        ]

    out_json = os.path.join(
        cfg.output_dir,
        f"policy_ranking_robomimic{out_suffix}_rl{cfg.rollout_length}.json",
    )
    payload = {"summary": summary, "wm_per_trajectory": results_by_coeff}
    if gt_results_by_coeff:
        payload["gt_per_trajectory"] = gt_results_by_coeff
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved to: {out_json}")
    tee.close()

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rank inhibited robomimic policies using a dino_wm world "
                    "model trained on Robomimic data as a simulator."
    )
    parser.add_argument("--ckpt_base_path", required=True,
                        help="Base directory containing outputs/<model_name>/")
    parser.add_argument("--model_name", required=True,
                        help="WM model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest",
                        help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--policy_ckpt", default=None,
                        help="Path to a single robomimic policy checkpoint (.pth). "
                             "Mutually exclusive with --policy_ckpt_dir.")
    parser.add_argument("--policy_ckpt_dir", default=None,
                        help="Directory of policy checkpoints to sweep over. "
                             "Each checkpoint is evaluated as one ranking entry.")
    parser.add_argument("--ckpt_glob", default="*.pth",
                        help="Glob pattern for checkpoints under --policy_ckpt_dir "
                             "(default: *.pth)")
    parser.add_argument("--ckpt_inhibition_coeff", type=float, default=1.0,
                        help="Fixed inhibition coefficient applied during a "
                             "checkpoint sweep (default: 1.0 — no inhibition)")
    parser.add_argument("--inhibition_coeffs", type=float, nargs="+",
                        default=[0.0, 0.1, 0.2, 0.5, 1.0, 2.0],
                        help="Inhibition coefficients (used only in single-ckpt mode)")
    parser.add_argument("--n_eval", type=int, default=100,
                        help="Number of trajectories per inhibition level (default: 100)")
    parser.add_argument("--rollout_length", type=int, default=20,
                        help="WM rollout steps per trajectory (default: 20)")
    parser.add_argument("--robometer_prompt",
                        default="Pick up the red cube and place it in the bin.",
                        help="Task prompt for Robometer scoring")
    parser.add_argument("--output_dir", default="./eval_results/policy_ranking_robomimic",
                        help="Directory for JSON and plots.")
    parser.add_argument("--run_gym", action="store_true",
                        help="Also run GT gym rollouts (RobomimicWrapper) and overlay.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    policy_ranking_main(args)
