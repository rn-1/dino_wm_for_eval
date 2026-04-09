#!/usr/bin/env python3
"""
gym_rollout.py — Run policy rollouts in the PushT gym environment.

Evaluates the LeRobot diffusion policy by executing it closed-loop in the
actual PushT physics simulator to collect ground-truth success metrics.

Usage:
    python gym_rollout.py \\
        --model_name pusht \\
        --n_eval 50 \\
        --rollout_length 10 \\
        --output_dir ./eval_results/gym_rollout \\
        --seed 42
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lerobot_utils import PolicyWrapper
from direct_success_metric import count_green_pixels
from utils import seed


def _to_hwc_uint8(frame_chw):
    """Convert (C, H, W) tensor in [-1, 1] to (H, W, C) uint8 numpy."""
    if isinstance(frame_chw, torch.Tensor):
        frame_chw = frame_chw.cpu().float().numpy()
    img = np.clip(frame_chw.transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def run_gym_rollout(
    model_name: str,
    n_eval: int,
    rollout_length: int,
    output_dir: str,
    seed_val: int = 42,
    video_fps: int = 4,
    n_save_examples: int = 10,
):
    """
    Run policy rollouts in the PushT gym environment.

    Args:
        model_name: Policy model name (e.g., 'lerobot/diffusion_pusht')
        n_eval: Number of episodes to evaluate
        rollout_length: Length of each rollout in steps
        output_dir: Directory to save results
        seed_val: Random seed
        video_fps: FPS for saved videos
        n_save_examples: Number of example rollouts to save visualizations for
    """
    seed(seed_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_dir, exist_ok=True)

    # Load the PushT gym environment
    try:
        import gym
        env = gym.make("PushT-v0")
    except Exception as e:
        print(f"Error loading PushT environment: {e}")
        print("Trying alternative import...")
        from env.pusht.pusht_wrapper import PushTWrapper
        env = PushTWrapper()

    # Load policy
    print(f"Loading policy: {model_name}")
    policy = PolicyWrapper(model_name=model_name)
    policy_img_size = 96

    # Collect rollouts
    results = {
        "rollout_length": rollout_length,
        "n_eval": n_eval,
        "rollouts": [],
        "success_metrics": [],
    }

    print(f"Running {n_eval} rollouts with length {rollout_length}...")

    for episode_idx in range(n_eval):
        obs, info = env.reset()
        frames = []
        actions = []
        success = None

        # Run episode
        for step in range(rollout_length):
            # Get image and state from observation
            if isinstance(obs, dict):
                if "image" in obs:
                    img = obs["image"]
                elif "visual" in obs:
                    img = obs["visual"]
                else:
                    img = next(iter(obs.values()))

                if "state" in obs:
                    state = obs["state"]
                elif "proprio" in obs:
                    state = obs["proprio"]
                else:
                    state = torch.zeros(2, device=device)  # Default PushT action space
            else:
                img = obs
                state = torch.zeros(2, device=device)

            # Convert image to tensor if needed
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img).float()
            if not isinstance(state, torch.Tensor):
                state = torch.from_numpy(state).float() if isinstance(state, np.ndarray) else torch.tensor(state, dtype=torch.float32)

            # Ensure correct formats
            if img.ndim == 2:
                img = img.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            elif img.ndim == 3:
                if img.shape[0] in (1, 3):  # Already (C, H, W)
                    img = img.unsqueeze(0)  # (1, C, H, W)
                else:  # (H, W, C)
                    img = img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

            # Normalize to [-1, 1] if needed
            if img.max() > 1.5:
                img = img / 127.5 - 1.0

            # Resize to policy's expected size
            img_policy = F.interpolate(
                img.to(device),
                size=(policy_img_size, policy_img_size),
                mode="bilinear",
                align_corners=False,
            )

            state = state.unsqueeze(0).to(device)

            # Get action from policy
            with torch.no_grad():
                action = policy.predict(observation=state, image=img_policy)  # (1, action_dim)

            action_np = action.squeeze(0).cpu().numpy()
            actions.append(action_np)

            # Store frame
            if isinstance(obs, dict) and "image" in obs:
                frames.append(obs["image"])
            else:
                frames.append(img.squeeze(0).cpu().numpy())

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            if done:
                break

        # Final frame
        if isinstance(obs, dict):
            if "image" in obs:
                frames.append(obs["image"])
            elif isinstance(obs.get("image"), np.ndarray):
                frames.append(obs["image"])

        # Convert frames to tensors for success metric computation
        frames_tensor = []
        for frame in frames:
            if isinstance(frame, np.ndarray):
                if frame.ndim == 3 and frame.shape[0] in (1, 3):  # (C, H, W)
                    frames_tensor.append(torch.from_numpy(frame).float())
                elif frame.ndim == 3:  # (H, W, C)
                    f = torch.from_numpy(frame).float()
                    frames_tensor.append(f.permute(2, 0, 1))
                else:  # (H, W)
                    frames_tensor.append(torch.from_numpy(frame).float().unsqueeze(0))
            else:
                frames_tensor.append(frame.float())

        if len(frames_tensor) >= 2:
            frames_tensor = torch.stack(frames_tensor)

            # Compute success metric: 1 - (covered pixel ratio)
            first_np = _to_hwc_uint8(frames_tensor[0])
            final_np = _to_hwc_uint8(frames_tensor[-1])

            success = max(0.0, 1.0 - (count_green_pixels(final_np) / max(count_green_pixels(first_np), 1)))
            results["success_metrics"].append(success)

            # Save example rollouts
            if episode_idx < n_save_examples:
                example_dir = os.path.join(output_dir, f"example_{episode_idx:03d}")
                os.makedirs(example_dir, exist_ok=True)

                # Save frames as numpy
                np.save(os.path.join(example_dir, "frames.npy"), frames_tensor.numpy())
                np.save(os.path.join(example_dir, "actions.npy"), np.array(actions))

        results["rollouts"].append({
            "episode": episode_idx,
            "length": len(frames) - 1,
            "success": success,
        })

        if (episode_idx + 1) % max(1, n_eval // 10) == 0:
            print(f"  [{episode_idx + 1}/{n_eval}] Success so far: {np.mean(results['success_metrics'][-10:]):.3f}")

    # Compute statistics
    if results["success_metrics"]:
        success_arr = np.array(results["success_metrics"])
        stats = {
            "mean_success": float(np.mean(success_arr)),
            "std_success": float(np.std(success_arr)),
            "min_success": float(np.min(success_arr)),
            "max_success": float(np.max(success_arr)),
            "median_success": float(np.median(success_arr)),
        }
        results["statistics"] = stats
        print(f"\nResults for rollout_length={rollout_length}:")
        print(f"  Mean success: {stats['mean_success']:.3f} ± {stats['std_success']:.3f}")
        print(f"  Range: [{stats['min_success']:.3f}, {stats['max_success']:.3f}]")

    # Save results
    results_path = os.path.join(output_dir, f"gym_rollout_rl{rollout_length}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Run policy rollouts in PushT gym environment")
    parser.add_argument("--model_name", default="lerobot/diffusion_pusht", help="Policy model name")
    parser.add_argument("--n_eval", type=int, default=50, help="Number of episodes to evaluate")
    parser.add_argument("--rollout_length", type=int, default=10, help="Rollout length in steps")
    parser.add_argument("--output_dir", default="./eval_results/gym_rollout", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--video_fps", type=int, default=4, help="FPS for saved videos")
    parser.add_argument("--n_save_examples", type=int, default=10, help="Number of example rollouts to save")

    args = parser.parse_args()

    run_gym_rollout(
        model_name=args.model_name,
        n_eval=args.n_eval,
        rollout_length=args.rollout_length,
        output_dir=args.output_dir,
        seed_val=args.seed,
        video_fps=args.video_fps,
        n_save_examples=args.n_save_examples,
    )


if __name__ == "__main__":
    main()
