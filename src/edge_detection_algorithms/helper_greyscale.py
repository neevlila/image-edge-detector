"""
Vectorized greyscale conversion using NumPy — replaces the slow pixel-by-pixel loop.
"""

import numpy as np


def greyscale(img_arr: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image array to greyscale using vectorized NumPy operations.

    Args:
        img_arr (np.ndarray): 3-D array representation of image (H x W x 3)

    Returns:
        np.ndarray: Array representation of the greyscaled image (H x W x 3),
                    where all three channels hold the same grey value.
    """
    # Average across colour channels in one vectorised operation — no Python loops
    grey = np.mean(img_arr, axis=2, keepdims=True).astype(np.uint8)   # (H, W, 1)
    return np.repeat(grey, 3, axis=2)                                  # (H, W, 3)