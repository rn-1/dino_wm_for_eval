#!/usr/bin/env python3
"""
evaluate_policy.py — Evaluate the LeRobot diffusion policy on WM vs. ground truth.

Two experiments:

  Exp 1 — Trajectory MSE:
    Run the policy closed-loop on WM-decoded frames and compare the predicted
    trajectory to the ground truth trajectory with per-step pixel MSE. Measures
    how much the WM trajectory drifts when driven by real policy actions.

  Exp 2 — Action Drift:
    From the same starting condition, run K rollouts with the policy conditioned
    on WM-decoded frames and K rollouts conditioned on GT frames. At each
    timestep, fit diagonal Gaussians to the empirical action distributions and
    compute per-step KLD (primary) and L2 of means (secondary). Shows how the
    policy's decisions diverge between WM and GT as the episode progresses.

Usage:
    python evaluate_policy.py \\
        --ckpt_base_path /project2/jessetho_1732/rl_eval_wm/dino_wm \\
        --model_name pusht \\
        --n_eval 20 \\
        --K 50 \\
        --rollout_length 10
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rollout import _load_model, _load_dataset_with_legacy_target_fallback
from lerobot_utils import PolicyWrapper
from utils import seed


# ---------------------------------------------------------------------------
# Logging / visualisation helpers
# ---------------------------------------------------------------------------

class Tee:
    """Mirrors stdout to a log file."""
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


def _to_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    """(C, H, W) in [-1, 1]  →  (H, W, C) or (H, W) numpy array in [0, 1]."""
    img = frame_chw.clamp(-1, 1).add(1).div(2).permute(1, 2, 0).cpu().numpy()
    return img[:, :, 0] if img.shape[2] == 1 else img


def save_example_frames(pred_frames, gt_frames, mse_val, out_path, label=""):
    """
    Save a two-row PNG grid: GT frames (top) vs predicted frames (bottom).

    Args:
        pred_frames: (T, C, H, W) float in [-1, 1]
        gt_frames:   (T, C, H, W) float in [-1, 1]
        mse_val:     scalar exp1 MSE for the trajectory (used in title)
        out_path:    destination file path (.png)
        label:       optional prefix for the figure title
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = pred_frames.shape[0]
    fig, axes = plt.subplots(2, T, figsize=(2.5 * T, 5))
    if T == 1:
        axes = axes.reshape(2, 1)

    cmap = "gray" if pred_frames.shape[1] == 1 else None
    for t in range(T):
        step_mse = ((pred_frames[t] - gt_frames[t]) ** 2).mean().item()
        axes[0, t].imshow(_to_hwc(gt_frames[t]), cmap=cmap, vmin=0, vmax=1)
        axes[0, t].set_title(f"t={t}", fontsize=8)
        axes[0, t].axis("off")
        axes[1, t].imshow(_to_hwc(pred_frames[t]), cmap=cmap, vmin=0, vmax=1)
        axes[1, t].set_title(f"{step_mse:.4f}", fontsize=7)
        axes[1, t].axis("off")

    axes[0, 0].set_ylabel("GT", fontsize=10)
    axes[1, 0].set_ylabel("Pred", fontsize=10)
    fig.suptitle(f"{label}  exp1 MSE={mse_val:.5f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save_policy_plots(exp1_per_step, exp2_per_step_kld, exp2_per_step_l2,
                       all_exp1_mse, all_exp2_kld, all_exp2_l2,
                       output_dir, model_name, rollout_length):
    """
    Save three PNGs:
      1. policy_exp1_per_step_rl{N}.png  — frame MSE degradation over horizon
      2. policy_exp2_per_step_rl{N}.png  — action KLD and L2 over horizon
      3. policy_traj_dist_rl{N}.png      — per-trajectory MSE / KLD distributions
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_exp1   = list(range(len(exp1_per_step)))
    steps_exp2   = list(range(len(exp2_per_step_kld)))

    # -- Plot 1: Exp1 frame MSE per step --------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps_exp1, exp1_per_step.tolist(), marker="o", linewidth=1.5,
            markersize=4, color="steelblue")
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Mean pixel MSE")
    ax.set_title(f"{model_name} — Exp1: WM frame MSE vs GT over horizon")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    out1 = os.path.join(output_dir, f"policy_exp1_per_step_rl{rollout_length}.png")
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"Saved Exp1 per-step plot     : {out1}")

    # -- Plot 2: Exp2 action KLD and L2 per step ------------------------------
    fig, (ax_kld, ax_l2) = plt.subplots(1, 2, figsize=(12, 4))

    ax_kld.plot(steps_exp2, exp2_per_step_kld.tolist(), marker="o", linewidth=1.5,
                markersize=4, color="darkorange")
    ax_kld.set_xlabel("Prediction step")
    ax_kld.set_ylabel("Mean KLD")
    ax_kld.set_title("Exp2: Action KLD (WM‖GT) vs step")
    ax_kld.grid(True, linestyle="--", alpha=0.5)

    ax_l2.plot(steps_exp2, exp2_per_step_l2.tolist(), marker="s", linewidth=1.5,
               markersize=4, color="mediumseagreen")
    ax_l2.set_xlabel("Prediction step")
    ax_l2.set_ylabel("L2 of action means")
    ax_l2.set_title("Exp2: Action mean L2 (WM vs GT) vs step")
    ax_l2.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"{model_name} — Exp2: action drift over horizon (rl={rollout_length})", fontsize=11)
    fig.tight_layout()
    out2 = os.path.join(output_dir, f"policy_exp2_per_step_rl{rollout_length}.png")
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"Saved Exp2 per-step plot     : {out2}")

    # -- Plot 3: per-trajectory distributions (MSE, KLD, L2) -----------------
    traj_mse  = sorted([t.mean().item() for t in all_exp1_mse])
    traj_kld  = sorted([t.mean().item() for t in all_exp2_kld])
    traj_l2   = sorted([t.mean().item() for t in all_exp2_l2])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, vals, label, color in zip(
        axes,
        [traj_mse, traj_kld, traj_l2],
        ["Exp1 frame MSE", "Exp2 KLD", "Exp2 L2"],
        ["steelblue", "darkorange", "mediumseagreen"],
    ):
        mean_val = sum(vals) / len(vals)
        ax.plot(vals, linewidth=1.2, color=color)
        ax.axhline(mean_val, color="red", linestyle="--", linewidth=1,
                   label=f"mean={mean_val:.5f}")
        ax.set_xlabel("Trajectory rank (sorted)")
        ax.set_ylabel(label)
        ax.set_title(f"Per-traj {label}")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle(f"{model_name} — per-trajectory distributions (rl={rollout_length})", fontsize=11)
    fig.tight_layout()
    out3 = os.path.join(output_dir, f"policy_traj_dist_rl{rollout_length}.png")
    fig.savefig(out3, dpi=120)
    plt.close(fig)
    print(f"Saved per-traj dist plot     : {out3}")


# ---------------------------------------------------------------------------
# Action sanity checks
# ---------------------------------------------------------------------------

def _check_actions_not_constant(actions: torch.Tensor, label: str = ""):
    """
    Warn if all actions in a rollout are identical (per-dim std == 0).
    actions: (T, action_dim)
    """
    if actions.shape[0] < 2:
        return
    std_per_dim = actions.std(dim=0)
    if (std_per_dim == 0).all():
        print(f"  WARNING [{label}]: all {actions.shape[0]} actions are identical — "
              f"policy may be returning a cached/dummy action. "
              f"actions[0]={actions[0].numpy()}")
    elif (std_per_dim < 1e-8).any():
        zero_dims = (std_per_dim < 1e-8).nonzero(as_tuple=True)[0].tolist()
        print(f"  WARNING [{label}]: action dims {zero_dims} are constant across all steps.")


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------

def reset_policy(policy: PolicyWrapper):
    """Reset the policy's internal observation queue between rollouts."""
    try:
        policy.policy.reset()
    except AttributeError:
        pass  # DiffusionPolicy has no reset(); state evolves naturally per call


def _img_for_policy(frame_chw: torch.Tensor, policy_img_size: int) -> torch.Tensor:
    """
    Convert a (C, H, W) frame in [-1, 1] to a (1, C, policy_img_size, policy_img_size)
    tensor in [0, 1] suitable for PolicyWrapper.predict.
    """
    img_01 = (frame_chw.clamp(-1.0, 1.0) + 1.0) / 2.0  # [0, 1]
    return F.interpolate(
        img_01.unsqueeze(0),
        size=(policy_img_size, policy_img_size),
        mode="bilinear",
        align_corners=False,
    )


# ---------------------------------------------------------------------------
# Single-rollout functions
# ---------------------------------------------------------------------------

def run_single_wm_rollout(wm, policy, obs, frameskip, n_past, rollout_length,
                          policy_img_size, device, verbose=False):
    """
    Run one closed-loop rollout: policy conditioned on WM-decoded frames.

    Mirrors the logic in RolloutWorkspace.perform_closed_loop_policy_rollout
    (rollout.py:467-509), minus Robometer scoring.

    Returns:
        pred_frames : (rollout_length+1, C, H, W) decoded WM frames in [-1, 1]
        actions     : (rollout_length, action_dim) policy actions at each step
    """
    wm_action_dim = wm.action_encoder.in_chans

    # Initial context: (1, n_past, ...)
    obs_0 = {k: v[:n_past].unsqueeze(0).to(device) for k, v in obs.items()}
    act_0 = torch.zeros(1, n_past, wm_action_dim, device=device)

    decoded_frames = []
    actions = []

    with torch.no_grad():
        z = wm.encode(obs_0, act_0)

        for t in range(rollout_length):
            # Decode the current WM latent
            z_obs_cur, _ = wm.separate_emb(z[:, -1:])
            decoded, _ = wm.decode_obs(z_obs_cur)
            img = decoded["visual"][0, 0]          # (C, H, W) in [-1, 1]
            decoded_frames.append(img.cpu())

            img_policy = _img_for_policy(img, policy_img_size)

            # GT proprio (WM has no proprio decoder)
            state_idx = min(t + n_past - 1, obs["proprio"].shape[0] - 1)
            state_input = obs["proprio"][state_idx].unsqueeze(0).to(device)

            action = policy.predict(observation=state_input, image=img_policy)
            action_cpu = action.squeeze(0).cpu()   # (action_dim,)
            actions.append(action_cpu)
            if verbose:
                print(f"      [WM t={t}] action={action_cpu.numpy()}")

            # Step the WM with the policy action
            action_grouped = action.repeat(1, frameskip).unsqueeze(1)  # (1,1,wm_action_dim)
            z_pred = wm.predict(z[:, -wm.num_hist:])
            z_new = z_pred[:, -1:, ...]
            z_new = wm.replace_actions_from_z(z_new, action_grouped)
            z = torch.cat([z, z_new], dim=1)

        # Decode the final frame
        z_obs_last, _ = wm.separate_emb(z[:, -1:])
        decoded_last, _ = wm.decode_obs(z_obs_last)
        decoded_frames.append(decoded_last["visual"][0, 0].cpu())

    pred_frames = torch.stack(decoded_frames, dim=0)   # (rollout_length+1, C, H, W)
    actions_t = torch.stack(actions, dim=0)             # (rollout_length, action_dim)
    _check_actions_not_constant(actions_t, label="WM")
    return pred_frames, actions_t


def run_single_gt_rollout(policy, obs, n_past, rollout_length, policy_img_size, device, verbose=False):
    """
    Run one rollout with the policy conditioned on GT frames (no WM involvement).

    At each step t the policy sees the ground-truth frame at position (t + n_past - 1)
    and the ground-truth proprio at the same index.

    Returns:
        actions : (rollout_length, action_dim) policy actions at each step
    """
    actions = []

    with torch.no_grad():
        for t in range(rollout_length):
            frame_idx = min(t + n_past - 1, obs["visual"].shape[0] - 1)
            img = obs["visual"][frame_idx].to(device)       # (C, H, W) in [-1, 1]
            img_policy = _img_for_policy(img, policy_img_size)

            state_idx = min(t + n_past - 1, obs["proprio"].shape[0] - 1)
            state_input = obs["proprio"][state_idx].unsqueeze(0).to(device)

            action = policy.predict(observation=state_input, image=img_policy)
            action_cpu = action.squeeze(0).cpu()
            actions.append(action_cpu)
            if verbose:
                print(f"      [GT t={t}] action={action_cpu.numpy()}")

    actions_t = torch.stack(actions, dim=0)   # (rollout_length, action_dim)
    _check_actions_not_constant(actions_t, label="GT")
    return actions_t


# ---------------------------------------------------------------------------
# Per-trajectory evaluation
# ---------------------------------------------------------------------------

def evaluate_trajectory(wm, policy, obs, frameskip, n_past, rollout_length,
                        K, policy_img_size, device, verbose=False):
    """
    Run both experiments for one trajectory starting condition.

    Returns a dict with:
        exp1_per_step_mse : (rollout_length+1,) tensor
        exp2_per_step_kld : (rollout_length,) tensor  — KLD(WM || GT) per step
        exp2_per_step_l2  : (rollout_length,) tensor  — L2(mean_WM, mean_GT) per step
    """
    # ── Exp 1: single WM rollout ──────────────────────────────────────────
    reset_policy(policy)
    pred_frames, exp1_actions_wm = run_single_wm_rollout(
        wm, policy, obs, frameskip, n_past, rollout_length, policy_img_size, device,
        verbose=verbose,
    )
    T_out = pred_frames.shape[0]   # rollout_length + 1
    gt_start = n_past - 1
    gt_frames = obs["visual"][gt_start : gt_start + T_out]   # (T_out, C, H, W)
    # Align length in case trajectory is shorter than expected
    min_T = min(pred_frames.shape[0], gt_frames.shape[0])
    exp1_pred_frames = pred_frames[:min_T].cpu()
    exp1_gt_frames   = gt_frames[:min_T].cpu()
    exp1_mse = ((exp1_pred_frames - exp1_gt_frames) ** 2).mean(dim=[1, 2, 3])

    # ── Exp 2: K-sample drift analysis ───────────────────────────────────
    all_actions_wm = []   # K × (rollout_length, action_dim)
    all_actions_gt = []

    for k in range(K):
        # Only print per-step actions for the first sample if verbose
        v = verbose and (k == 0)
        reset_policy(policy)
        _, act_wm = run_single_wm_rollout(
            wm, policy, obs, frameskip, n_past, rollout_length, policy_img_size, device,
            verbose=v,
        )
        all_actions_wm.append(act_wm)

        reset_policy(policy)
        act_gt = run_single_gt_rollout(
            policy, obs, n_past, rollout_length, policy_img_size, device, verbose=v,
        )
        all_actions_gt.append(act_gt)

    A_wm = torch.stack(all_actions_wm)   # (K, T, action_dim)
    A_gt = torch.stack(all_actions_gt)   # (K, T, action_dim)

    mu_wm  = A_wm.mean(dim=0)                    # (T, action_dim)
    mu_gt  = A_gt.mean(dim=0)
    var_wm = A_wm.var(dim=0).clamp(min=1e-8)     # (T, action_dim)
    var_gt = A_gt.var(dim=0).clamp(min=1e-8)

    # KLD( N(mu_wm, var_wm) || N(mu_gt, var_gt) ) — diagonal Gaussian, summed over dims
    # = sum_d [ 0.5 * ( log(var_gt_d / var_wm_d) + var_wm_d/var_gt_d
    #                   + (mu_wm_d - mu_gt_d)^2 / var_gt_d - 1 ) ]
    per_step_kld = 0.5 * (
        torch.log(var_gt / var_wm)
        + var_wm / var_gt
        + (mu_wm - mu_gt) ** 2 / var_gt
        - 1.0
    ).sum(dim=-1)   # (T,)

    per_step_l2 = (mu_wm - mu_gt).norm(dim=-1)   # (T,)

    return {
        "exp1_per_step_mse": exp1_mse,           # (T_out,)
        "exp2_per_step_kld": per_step_kld,       # (rollout_length,)
        "exp2_per_step_l2":  per_step_l2,        # (rollout_length,)
        "exp1_pred_frames":  exp1_pred_frames,   # (T_out, C, H, W) CPU
        "exp1_gt_frames":    exp1_gt_frames,     # (T_out, C, H, W) CPU
        "all_actions_wm":    all_actions_wm,     # list of K × (rollout_length, action_dim)
        "all_actions_gt":    all_actions_gt,     # list of K × (rollout_length, action_dim)
    }


# ---------------------------------------------------------------------------
# Trajectory sampling
# ---------------------------------------------------------------------------

def sample_valid_trajectory(dset, rollout_length, frameskip, n_past, rng):
    """
    Return a raw trajectory dict (obs, act, state) with enough frames for the
    requested rollout. Uses the full raw dataset (no pre-subsampling).
    """
    min_frames = (n_past + rollout_length) * frameskip
    indices = list(range(len(dset)))
    rng.shuffle(indices)
    for idx in indices:
        obs, act, state, *_ = dset[idx]
        if obs["visual"].shape[0] >= min_frames:
            return obs
    raise RuntimeError(
        f"No trajectory in dataset has >= {min_frames} raw frames "
        f"(n_past={n_past}, rollout_length={rollout_length}, frameskip={frameskip})."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_policy_main(cfg):
    device = torch.device(cfg.device)
    seed(cfg.seed)
    rng = np.random.RandomState(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    tee = Tee(os.path.join(cfg.output_dir, "policy_eval.log"))

    # ── Load model ────────────────────────────────────────────────────────
    model_path = Path(cfg.ckpt_base_path) / "outputs" / cfg.model_name
    with open(model_path / "hydra.yaml") as f:
        model_cfg = OmegaConf.load(f)

    _, dset = _load_dataset_with_legacy_target_fallback(model_cfg)
    dset = dset["valid"]
    print(f"Validation dataset: {len(dset)} trajectories")

    model_ckpt = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    wm = _load_model(model_ckpt, model_cfg, model_cfg.num_action_repeat, device=device)
    wm.eval()

    if wm.decoder is None:
        raise RuntimeError("World model has no decoder — pixel decoding required.")

    # ── Load policy ───────────────────────────────────────────────────────
    n_action_steps = getattr(cfg, "n_action_steps", None)
    policy = PolicyWrapper(model_name=cfg.policy_model_name, n_action_steps=n_action_steps)
    print(f"  policy n_action_steps={policy.policy.config.n_action_steps} "
          f"(actions served per diffusion inference pass)")

    n_past        = model_cfg.num_hist
    frameskip     = model_cfg.frameskip
    rollout_length = cfg.rollout_length
    K             = cfg.K
    policy_img_size = cfg.policy_img_size

    print(f"\nEvaluating '{cfg.model_name}' (epoch={cfg.model_epoch}) "
          f"with policy '{cfg.policy_model_name}'")
    print(f"  n_eval={cfg.n_eval}, K={K}, rollout_length={rollout_length}, "
          f"frameskip={frameskip}, n_past={n_past}")

    # ── Evaluation loop ───────────────────────────────────────────────────
    all_exp1_mse = []    # list of (T_out,) tensors
    all_exp2_kld = []    # list of (rollout_length,) tensors
    all_exp2_l2  = []
    # For saving high-loss examples: list of (exp1_mse_float, pred_frames, gt_frames)
    all_examples = []

    verbose_first = getattr(cfg, "verbose_actions", False)
    for traj_idx in range(cfg.n_eval):
        obs = sample_valid_trajectory(dset, rollout_length, frameskip, n_past, rng)
        verbose_this = verbose_first or (traj_idx == 0)

        metrics = evaluate_trajectory(
            wm, policy, obs, frameskip, n_past, rollout_length,
            K, policy_img_size, device, verbose=verbose_this,
        )

        all_exp1_mse.append(metrics["exp1_per_step_mse"])
        all_exp2_kld.append(metrics["exp2_per_step_kld"])
        all_exp2_l2.append(metrics["exp2_per_step_l2"])
        all_examples.append((
            metrics["exp1_per_step_mse"].mean().item(),
            metrics["exp1_pred_frames"],
            metrics["exp1_gt_frames"],
        ))

        # action stats from Exp 2 (across K rollouts)
        A_wm = torch.stack(metrics["all_actions_wm"])   # (K, T, action_dim)
        A_gt = torch.stack(metrics["all_actions_gt"])
        print(f"  [{traj_idx+1:3d}/{cfg.n_eval}]  "
              f"exp1 mean MSE: {metrics['exp1_per_step_mse'].mean():.5f}  "
              f"exp2 mean KLD: {metrics['exp2_per_step_kld'].mean():.5f}  "
              f"exp2 mean L2: {metrics['exp2_per_step_l2'].mean():.5f}  "
              f"act_wm std={A_wm.std():.4f}  act_gt std={A_gt.std():.4f}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    exp1_per_step = torch.stack(all_exp1_mse).mean(dim=0)   # (T_out,)
    exp2_per_step_kld = torch.stack(all_exp2_kld).mean(dim=0)   # (rollout_length,)
    exp2_per_step_l2  = torch.stack(all_exp2_l2).mean(dim=0)

    exp1_overall_mse = exp1_per_step.mean().item()
    exp2_mean_kld    = exp2_per_step_kld.mean().item()
    exp2_mean_l2     = exp2_per_step_l2.mean().item()

    print(f"\n=== Policy Evaluation Results ===")
    print(f"  Exp 1 overall MSE   : {exp1_overall_mse:.6f}")
    print(f"  Exp 1 per-step MSE  : {[f'{v:.5f}' for v in exp1_per_step.tolist()]}")
    print(f"  Exp 2 mean KLD      : {exp2_mean_kld:.6f}")
    print(f"  Exp 2 per-step KLD  : {[f'{v:.5f}' for v in exp2_per_step_kld.tolist()]}")
    print(f"  Exp 2 mean L2       : {exp2_mean_l2:.6f}")
    print(f"  Exp 2 per-step L2   : {[f'{v:.5f}' for v in exp2_per_step_l2.tolist()]}")

    # ── Save plots ────────────────────────────────────────────────────────
    _save_policy_plots(
        exp1_per_step, exp2_per_step_kld, exp2_per_step_l2,
        all_exp1_mse, all_exp2_kld, all_exp2_l2,
        cfg.output_dir, cfg.model_name, rollout_length,
    )

    # ── Save high-loss example images ─────────────────────────────────────
    if cfg.n_save_examples > 0:
        examples_dir = os.path.join(cfg.output_dir, "high_loss_examples")
        os.makedirs(examples_dir, exist_ok=True)
        top_examples = sorted(all_examples, key=lambda x: x[0], reverse=True)
        for rank, (mse_val, pred_f, gt_f) in enumerate(top_examples[: cfg.n_save_examples]):
            out_img = os.path.join(examples_dir, f"rank{rank + 1:02d}_mse{mse_val:.5f}.png")
            save_example_frames(pred_f, gt_f, mse_val, out_img,
                                label=f"{cfg.model_name} rank#{rank + 1}")
        print(f"\nSaved {min(cfg.n_save_examples, len(top_examples))} "
              f"high-loss example images to: {examples_dir}/")

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "model_name":          cfg.model_name,
        "model_epoch":         cfg.model_epoch,
        "policy_model_name":   cfg.policy_model_name,
        "n_eval":              cfg.n_eval,
        "K":                   K,
        "rollout_length":      rollout_length,
        "frameskip":           frameskip,
        "n_past":              n_past,
        # Exp 1
        "exp1_overall_mse":    exp1_overall_mse,
        "exp1_per_step_mse":   exp1_per_step.tolist(),
        "per_traj_exp1_mse":   [t.mean().item() for t in all_exp1_mse],
        # Exp 2
        "exp2_mean_kld":       exp2_mean_kld,
        "exp2_per_step_kld":   exp2_per_step_kld.tolist(),
        "exp2_mean_l2":        exp2_mean_l2,
        "exp2_per_step_l2":    exp2_per_step_l2.tolist(),
        "per_traj_exp2_kld":   [t.mean().item() for t in all_exp2_kld],
        "per_traj_exp2_l2":    [t.mean().item() for t in all_exp2_l2],
    }
    out_path = os.path.join(cfg.output_dir, f"policy_eval_results_rl{rollout_length}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print(f"Log saved to:     {os.path.join(cfg.output_dir, 'policy_eval.log')}")
    tee.close()

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate LeRobot diffusion policy on WM vs. ground truth."
    )
    parser.add_argument("--ckpt_base_path", required=True,
                        help="Base directory containing outputs/<model_name>/")
    parser.add_argument("--model_name", required=True,
                        help="Model name (subdir under outputs/)")
    parser.add_argument("--model_epoch", default="latest",
                        help="Checkpoint epoch tag (default: latest)")
    parser.add_argument("--policy_model_name", default="lerobot/diffusion_pusht",
                        help="LeRobot policy model name or local path")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of trajectories to evaluate (default: 20)")
    parser.add_argument("--K", type=int, default=50,
                        help="Policy samples per trajectory for drift estimation (default: 50)")
    parser.add_argument("--rollout_length", type=int, default=10,
                        help="Steps to roll out per trajectory (default: 10)")
    parser.add_argument("--policy_img_size", type=int, default=96,
                        help="Image size for policy input in pixels (default: 96)")
    parser.add_argument("--output_dir", default="./eval_results",
                        help="Directory to save results JSON (default: ./eval_results)")
    parser.add_argument("--n_save_examples", type=int, default=5,
                        help="Number of highest-exp1-MSE trajectories to save as PNG grids (0=none)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--n_action_steps", type=int, default=None,
                        help="Override policy n_action_steps (actions per diffusion inference pass). "
                             "Use 1 for fresh inference at every step. Default: use policy's trained value.")
    parser.add_argument("--verbose_actions", action="store_true",
                        help="Print per-step action values for every trajectory (default: first only)")
    parser.add_argument("--device", default="cuda",
                        help="Torch device to use (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_policy_main(args)
