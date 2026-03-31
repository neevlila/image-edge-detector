"""
Blur helpers rewritten to use PIL's built-in C-accelerated filters instead of
pure-Python pixel loops.  Drop-in replacements for the original gaussian_blur
and median_blur functions.
"""

import numpy as np
from PIL import Image, ImageFilter


def gaussian_blur(img_arr: np.ndarray, sigma: int) -> np.ndarray:
    """
    Applies a Gaussian blur using PIL's ImageFilter.GaussianBlur (C-accelerated).

    Args:
        img_arr (np.ndarray): H x W x 3 uint8 image array
        sigma   (int):        Standard deviation / blur strength

    Returns:
        np.ndarray: Blurred image array (same shape and dtype as input)
    """
    if sigma <= 0:
        return img_arr.copy()

    img = Image.fromarray(img_arr.astype(np.uint8))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.array(blurred)


def median_blur(img_arr: np.ndarray, radius: int) -> np.ndarray:
    """
    Applies a Median blur using PIL's ImageFilter.MedianFilter (C-accelerated).

    Args:
        img_arr (np.ndarray): H x W x 3 uint8 image array
        radius  (int):        Kernel radius; kernel size = 2*radius + 1

    Returns:
        np.ndarray: Blurred image array (same shape and dtype as input)
    """
    size = 2 * radius + 1
    if size < 3:
        return img_arr.copy()

    img = Image.fromarray(img_arr.astype(np.uint8))
    blurred = img.filter(ImageFilter.MedianFilter(size=size))
    return np.array(blurred)