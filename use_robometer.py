import os
import sys
import numpy as np
import torch

# Add the robometer library to the path (sibling repo)
_ROBOMETER_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robometer")
if os.path.isdir(_ROBOMETER_LIB) and _ROBOMETER_LIB not in sys.path:
    sys.path.insert(0, _ROBOMETER_LIB)

# Local HF-cache snapshot — weights were downloaded but processor files were not,
# so we load via the robometer library which reads config.yaml directly.
_ROBOMETER_SNAPSHOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".hf_cache", "hub", "models--robometer--Robometer-4B",
    "snapshots", "beef63bc914c5c189329d49c6d712d96d632aa34",
)


def load_robometer(device=None):
    """
    Load the Robometer-4B model from the local cached snapshot using the
    robometer library (bypasses the missing AutoProcessor config files).

    Returns (exp_config, tokenizer, processor, model), or
            (None, None, None, None) on failure.
    """

    print(f"[Robometer] snapshot exists: {os.path.isdir(_ROBOMETER_SNAPSHOT)} → {_ROBOMETER_SNAPSHOT}")
    print(f"[Robometer] sibling lib exists: {os.path.isdir(_ROBOMETER_LIB)} → {_ROBOMETER_LIB}")
    try:
        from robometer.utils.save import load_model_from_hf

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        exp_config, tokenizer, processor, model = load_model_from_hf(
            model_path=_ROBOMETER_SNAPSHOT,
            device=device,
        )
        model.eval()
        print(f"[Robometer] Loaded model from {_ROBOMETER_SNAPSHOT}")
        return exp_config, tokenizer, processor, model
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[Warning] Robometer model not available: {e}.")
        print(f"[Warning] Robometer model not available: {e}. Skipping Robometer scoring.")
        return None, None, None, None


def infer_robometer(exp_config, tokenizer, processor, model, frames_uint8, task_prompt, device=None):
    """
    Run Robometer inference on a full rollout video.

    Robometer is a video model — pass all frames at once, not one at a time.

    Args:
        exp_config:    ExperimentConfig returned by load_robometer
        tokenizer:     tokenizer returned by load_robometer
        processor:     processor returned by load_robometer
        model:         RBM model returned by load_robometer
        frames_uint8:  (T, H, W, C) uint8 numpy array, all frames of the rollout
        task_prompt:   text description of the task
        device:        torch device (inferred from model if None)

    Returns:
        progress_array : (T,) float32  — per-frame progress in [0, 1]
        success_array  : (T,) float32  — per-frame success probability in [0, 1]
    """
    from robometer.data.dataset_types import ProgressSample, Trajectory
    from robometer.evals.eval_server import compute_batch_outputs
    from robometer.utils.setup_utils import setup_batch_collator

    if device is None:
        device = next(model.parameters()).device

    T = int(frames_uint8.shape[0])

    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)

    traj = Trajectory(
        frames=frames_uint8,
        frames_shape=tuple(frames_uint8.shape),
        task=task_prompt,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    progress_inputs = batch["progress_inputs"]
    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(device)

    loss_config = getattr(exp_config, "loss", None)
    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config else False
    )
    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    with torch.no_grad():
        results = compute_batch_outputs(
            model,
            tokenizer,
            progress_inputs,
            sample_type="progress",
            is_discrete_mode=is_discrete,
            num_bins=num_bins,
        )

    progress_pred = results.get("progress_pred", [])
    progress_array = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.zeros(T, dtype=np.float32)
    )

    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    success_array = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.zeros(T, dtype=np.float32)
    )

    return progress_array, success_array


def main():
    exp_config, tokenizer, processor, model = load_robometer()
    if model is None:
        return

    # Dummy 10-frame (96×96 RGB) rollout
    frames = np.zeros((10, 96, 96, 3), dtype=np.uint8)
    task = "Push the T block to the goal position."
    progress, success = infer_robometer(exp_config, tokenizer, processor, model, frames, task)
    print(f"progress shape: {progress.shape}, values: {progress}")
    print(f"success shape:  {success.shape}, values:  {success}")


if __name__ == "__main__":
    main()
