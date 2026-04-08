import numpy as np
from PIL import Image


def count_green_pixels(image, threshold=100):
    """Count pixels where the green channel dominates above a threshold.

    Args:
        image: file path (str), PIL Image, or numpy array (H, W, 3) in RGB.
        threshold: minimum value the green channel must exceed (0-255).

    Returns:
        Integer count of qualifying green pixels.
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        image = np.array(image)

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    mask = (g.astype(int) > threshold) & (g.astype(int) > r) & (g.astype(int) > b)
    return int(np.count_nonzero(mask))
