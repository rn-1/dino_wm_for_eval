import numpy as np
from PIL import Image


def count_green_pixels(image, threshold=100, blur_radius=5):
    """Count pixels where the green channel dominates above a threshold.

    Args:
        image: file path (str), PIL Image, or numpy array (H, W, 3) in RGB.
        threshold: minimum value the green channel must exceed (0-255).
        blur_radius: Gaussian blur kernel size applied before thresholding.
            Suppresses high-frequency decoder checkerboard artifacts without
            affecting large solid regions like the pusht goal area.
            Set to 0 to disable blurring.

    Returns:
        Integer count of qualifying green pixels.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.array(image)

    if blur_radius > 0:
        from PIL import ImageFilter
        image = np.array(
            Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=blur_radius))
        )

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    mask = (g.astype(int) > threshold) & (g.astype(int) > r) & (g.astype(int) > b)
    return int(np.count_nonzero(mask))


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Count green pixels per frame in a rollout .npy file (T, H, W, C) uint8."
    )
    parser.add_argument("npy_path", type=str, help="Path to a rollout .npy file.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=100,
        help="Green channel threshold (0-255, default: 100).",
    )
    parser.add_argument(
        "--blur_radius",
        type=int,
        default=5,
        help="Gaussian blur radius before thresholding (default: 5, 0 to disable).",
    )
    args = parser.parse_args()

    frames = np.load(args.npy_path)  # (T, H, W, C) uint8
    T = frames.shape[0]
    print(f"Loaded {args.npy_path}: shape={frames.shape}, dtype={frames.dtype}")

    # --- cv_debug output dir ---
    stem = os.path.splitext(os.path.basename(args.npy_path))[0]
    debug_dir = os.path.join("cv_debug", stem)
    os.makedirs(debug_dir, exist_ok=True)

    print(f"{'Frame':>6}  {'Green pixels':>12}")
    print("-" * 22)
    counts = []
    for t in range(T):
        frame = frames[t]  # (H, W, C) uint8
        if args.blur_radius > 0:
            from PIL import ImageFilter
            blurred = np.array(
                Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
            )
            Image.fromarray(blurred).save(os.path.join(debug_dir, f"frame{t:02d}_blurred.png"))
        else:
            blurred = frame

        r, g, b = blurred[..., 0].astype(int), blurred[..., 1].astype(int), blurred[..., 2].astype(int)

        cond_threshold = g > args.threshold
        cond_dominates = (g > r) & (g > b)
        mask = cond_threshold & cond_dominates

        c = int(np.count_nonzero(mask))
        counts.append(c)

        marker = " <-- first" if t == 0 else (" <-- last" if t == T - 1 else "")
        print(f"{t:>6}  {c:>12}{marker}")

        # Save original frame
        Image.fromarray(frame).save(os.path.join(debug_dir, f"frame{t:02d}_original.png"))

        # Save green channel as grayscale
        Image.fromarray(g.astype(np.uint8)).save(os.path.join(debug_dir, f"frame{t:02d}_green_ch.png"))

        # Save threshold mask (g > threshold) as white-on-black
        Image.fromarray((cond_threshold * 255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"frame{t:02d}_mask_threshold.png")
        )

        # Save dominance mask (g > r and g > b) as white-on-black
        Image.fromarray((cond_dominates * 255).astype(np.uint8)).save(
            os.path.join(debug_dir, f"frame{t:02d}_mask_dominates.png")
        )

        # Save final combined mask overlaid in green on the original frame
        overlay = frame.copy()
        overlay[mask] = [0, 255, 0]
        Image.fromarray(overlay).save(os.path.join(debug_dir, f"frame{t:02d}_overlay.png"))

    print("-" * 22)
    first, last = counts[0], counts[-1]
    score = max(0.0, 1.0 - last / max(first, 1))
    print(f"first={first}  last={last}  score={score:.4f}")
    print(f"Debug images saved to: {debug_dir}/")