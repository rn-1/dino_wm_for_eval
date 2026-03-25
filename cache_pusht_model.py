import os
import sys

REPO_ID = "lerobot/diffusion_pusht"
DEFAULT_HF_HOME = "/project2/jessetho_1732/rl_eval_wm/dino_wm/.hf_cache"


def main() -> int:
    hf_home = os.environ.get("HF_HOME", DEFAULT_HF_HOME)
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", os.path.join(hf_home, "hub"))

    os.makedirs(hub_cache, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        print("huggingface_hub is not installed in this environment.", file=sys.stderr)
        print("Install it with: pip install huggingface_hub", file=sys.stderr)
        return 1

    local_path = snapshot_download(repo_id=REPO_ID, cache_dir=hub_cache)
    print(f"Cached {REPO_ID} at: {local_path}")
    print(f"HF_HOME={hf_home}")
    print(f"HUGGINGFACE_HUB_CACHE={hub_cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
