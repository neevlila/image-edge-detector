"""
This is my implementation of the Sobel edge detection algorithm
"""

from typing import Tuple
import numpy as np
from PIL import Image
from .helper_blur import gaussian_blur, median_blur
from .helper_greyscale import greyscale


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.abs(arr.astype(np.float64))
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    max_val = np.max(arr)
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr / max_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def sobel_edge_detect(img_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform Sobel edge detection and return derivative images.

    Returns:
        final_edge_rgb: RGB image of normalized gradient magnitude
        gx_vis: normalized gradient X visualization (uint8)
        gy_vis: normalized gradient Y visualization (uint8)
        magnitude_vis: normalized gradient magnitude (uint8)
    """

    height, width, _ = img_arr.shape

    # Preprocessing
    greyscaled_img_arr = greyscale(img_arr)
    r = 0 if width < 500 else 1
    blurred = median_blur(greyscaled_img_arr, radius=r)
    blurred = gaussian_blur(blurred, sigma=2)

    intensity_arr = blurred[:, :, 0]

    gx = np.zeros((height, width), dtype=np.float64)
    gy = np.zeros((height, width), dtype=np.float64)

    X_KERNEL = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    Y_KERNEL = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            patch = intensity_arr[y - 1 : y + 2, x - 1 : x + 2]
            gx[y, x] = np.sum(X_KERNEL * patch)
            gy[y, x] = np.sum(Y_KERNEL * patch)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    gx_vis = _normalize_to_uint8(gx)
    gy_vis = _normalize_to_uint8(gy)
    magnitude_vis = _normalize_to_uint8(magnitude)

    final_edge_rgb = np.stack([magnitude_vis, magnitude_vis, magnitude_vis], axis=-1)

    return final_edge_rgb, gx_vis, gy_vis, magnitude_vis
