#!/usr/bin/env python3
"""
vlm_judge.py — Zero-shot success judge using Qwen3-VL-4B-Instruct.

Given a language task description and a final frame from a robot rollout,
asks the VLM: "Did the robot succeed at this task?"

Functions
---------
load_vlm_judge(model_name, device, hf_cache_dir, local_files_only)
    Load processor + model.  Returns (processor, model).

judge_frame(processor, model, frame, task_prompt, device)
    Score a single final frame.  Returns {"success": bool, "raw_response": str}.

judge_batch(processor, model, frames, task_prompts, device)
    Score a list of frames.  Returns list of judge_frame dicts.

CLI usage
---------
    python vlm_judge.py \\
        --frame  path/to/final_frame.png \\
        --task   "Push the T block to the goal position." \\
        [--model_name Qwen/Qwen3-VL-4B-Instruct] \\
        [--device cuda]
"""

import os
import sys
import argparse

import numpy as np
import torch
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Model name / HF-cache defaults
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"
_DEFAULT_HF_CACHE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".hf_cache"
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are evaluating whether a robot successfully completed a manipulation task. "
    "Answer with ONLY a number between 0.0 and 1.0, where 0.0 means complete failure and 1.0 means perfect success."
    "This may include numbers such as 0.453 or 0.8, but do not include any text or explanation. "
    "It is CRITICAL that you follow these instructions and ONLY respond with a number between 0.0 and 1.0, as I will parse your response programmatically."
)

_USER_PROMPT_TEMPLATE = (
    "Task: {task}\n\n"
    "The image shows the final state of the robot's execution. "
    "Give a score between 0.0 and 1.0 of how successful the robot was at completing the task."
)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_vlm_judge(
    model_name: str = _DEFAULT_MODEL,
    device=None,
    hf_cache_dir: str = _DEFAULT_HF_CACHE,
    local_files_only: bool = True,
):
    """
    Load the Qwen3-VL-4B-Instruct processor and model.

    Args:
        model_name:        HuggingFace model ID or local path.
        device:            torch.device (default: cuda if available).
        hf_cache_dir:      HF cache directory (default: .hf_cache beside this file).
        local_files_only:  If True, never contact HuggingFace (offline mode).

    Returns:
        (processor, model), or (None, None) on failure.
    """
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        os.makedirs(hf_cache_dir, exist_ok=True)

        # Resolve the model ID to its local snapshot directory so that
        # AutoProcessor / AutoModel work even with TRANSFORMERS_OFFLINE=1.
        load_path = model_name
        try:
            from huggingface_hub import snapshot_download
            load_path = snapshot_download(
                repo_id=model_name,
                cache_dir=hf_cache_dir,
                local_files_only=local_files_only,
            )
            print(f"[VLMJudge] Resolved snapshot: {load_path}")
        except Exception:
            # If snapshot_download fails (e.g. model_name is already a path),
            # fall back to passing the name directly.
            pass

        kwargs = dict(
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        print(f"[VLMJudge] Loading processor from '{load_path}' ...")
        processor = AutoProcessor.from_pretrained(load_path, **kwargs)

        print(f"[VLMJudge] Loading model from '{load_path}' ...")
        model = AutoModelForImageTextToText.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map=str(device),
            **kwargs,
        )
        model.eval()
        print(f"[VLMJudge] Model loaded on {device}.")
        return processor, model

    except Exception as e:
        print(f"[Warning] VLMJudge model not available: {e}. Skipping VLM judging.")
        return None, None


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _to_pil(frame) -> PILImage.Image:
    """
    Accept any of:
      - PIL.Image
      - (H, W, C) uint8 numpy array
      - (C, H, W) float tensor in [-1, 1] or [0, 1]
    Returns a PIL RGB image.
    """
    if isinstance(frame, PILImage.Image):
        return frame.convert("RGB")
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().float()
        if frame.ndim == 3 and frame.shape[0] in (1, 3):       # CHW
            frame = frame.permute(1, 2, 0)
        frame = frame.numpy()
        if frame.max() <= 1.01:                                  # [-1,1] or [0,1]
            frame = np.clip(frame * 0.5 + 0.5, 0.0, 1.0) if frame.min() < -0.01 \
                    else np.clip(frame, 0.0, 1.0)
            frame = (frame * 255).astype(np.uint8)
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    return PILImage.fromarray(frame.astype(np.uint8)).convert("RGB")


def _parse_response(text: str) -> bool:
    """Return True if the response starts with or contains YES."""
    upper = text.strip().upper()
    if upper.startswith("YES"):
        return True
    if upper.startswith("NO"):
        return False
    # Fallback: search first word
    first_word = upper.split()[0] if upper.split() else ""
    return first_word == "YES"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def judge_frame(processor, model, frame, task_prompt: str, device=None) -> dict:
    """
    Ask the VLM whether the task succeeded in a single final frame.

    Args:
        processor:    processor returned by load_vlm_judge
        model:        model returned by load_vlm_judge
        frame:        final frame — PIL image, (H,W,C) uint8 numpy, or (C,H,W) tensor
        task_prompt:  text description of the task
        device:       torch device (inferred from model if None)

    Returns:
        {
            "success":      bool,
            "raw_response": str,
        }
    """
    if model is None:
        return {"success": False, "raw_response": ""}

    if device is None:
        device = next(model.parameters()).device

    pil_img = _to_pil(frame)
    user_text = _USER_PROMPT_TEMPLATE.format(task=task_prompt)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": user_text},
            ],
        },
    ]

    # Build input using the chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Some Qwen VL processors expose process_vision_info; use it when available
    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
    except ImportError:
        inputs = processor(
            text=[text],
            images=[pil_img],
            return_tensors="pt",
            padding=True,
        )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()

    return {
        "success":      _parse_response(response),
        "raw_response": response,
    }


def judge_batch(processor, model, frames: list, task_prompts, device=None) -> list:
    """
    Score a list of final frames.

    Args:
        processor, model: from load_vlm_judge
        frames:           list of frames (PIL / numpy / tensor)
        task_prompts:     single string applied to all, or list of strings
        device:           torch device

    Returns:
        list of dicts, each with keys "success" and "raw_response"
    """
    if isinstance(task_prompts, str):
        task_prompts = [task_prompts] * len(frames)

    return [
        judge_frame(processor, model, frame, prompt, device=device)
        for frame, prompt in zip(frames, task_prompts)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Zero-shot VLM success judge using Qwen3-VL-4B."
    )
    parser.add_argument("--frame",  required=True,
                        help="Path to final frame image (PNG/JPG).")
    parser.add_argument("--task",   required=True,
                        help="Language description of the task.")
    parser.add_argument("--model_name", default=_DEFAULT_MODEL,
                        help=f"HuggingFace model ID (default: {_DEFAULT_MODEL})")
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. 'cuda' or 'cpu'. Auto-detected if omitted.")
    parser.add_argument("--hf_cache_dir", default=_DEFAULT_HF_CACHE,
                        help="HuggingFace cache directory.")
    parser.add_argument("--no_local_files_only", action="store_true",
                        help="Allow contacting HuggingFace (default: local files only).")
    return parser.parse_args()


def main():
    args = _parse_args()

    device = torch.device(args.device) if args.device else None

    processor, model = load_vlm_judge(
        model_name=args.model_name,
        device=device,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=not args.no_local_files_only,
    )
    if model is None:
        sys.exit(1)

    frame = PILImage.open(args.frame).convert("RGB")
    result = judge_frame(processor, model, frame, args.task, device=device)

    print(f"Task:         {args.task}")
    print(f"Success:      {result['success']}")
    print(f"Raw response: {result['raw_response']}")


if __name__ == "__main__":
    main()
