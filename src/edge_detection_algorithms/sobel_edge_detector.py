"""
Vectorized Sobel edge detection — replaces the slow inner pixel loop with
NumPy stride-trick patch extraction and einsum contraction.
No extra dependencies beyond NumPy and Pillow.
"""

from typing import Tuple
import numpy as np
from PIL import Image

from .helper_blur import gaussian_blur, median_blur
from .helper_greyscale import greyscale


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise any float array to the 0-255 uint8 range."""
    arr = np.abs(arr.astype(np.float64))
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    max_val = np.max(arr)
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr / max_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Fast 2-D convolution via NumPy stride tricks + einsum.
    Works on a single-channel (H x W) float64 array.
    No SciPy required.
    """
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    # Edge-pad so output has the same shape as input
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')

    # Build a view of shape (H, W, kh, kw) using stride tricks — zero copies
    from numpy.lib.stride_tricks import as_strided
    h, w = image.shape
    out_shape  = (h, w, kh, kw)
    out_strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
    patches = as_strided(padded, shape=out_shape, strides=out_strides)

    # Dot each patch against the kernel in one vectorised step
    return np.einsum('ijkl,kl->ij', patches, kernel)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sobel_edge_detect(
    img_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Sobel edge detection and return gradient visualisation images.

    Args:
        img_arr (np.ndarray): H x W x 3 uint8 RGB image array

    Returns:
        final_edge_rgb  : RGB uint8 image of normalised gradient magnitude
        gx_vis          : normalised X-gradient visualisation (uint8, H x W)
        gy_vis          : normalised Y-gradient visualisation (uint8, H x W)
        magnitude_vis   : normalised gradient magnitude       (uint8, H x W)
    """
    _,  width, _ = img_arr.shape

    # ------------------------------------------------------------------
    # 1. Pre-processing
    # ------------------------------------------------------------------
    greyscaled = greyscale(img_arr)

    # Only apply median blur on larger images (radius 0 = skip)
    radius = 0 if width < 500 else 1
    blurred = median_blur(greyscaled, radius=radius)
    blurred = gaussian_blur(blurred, sigma=2)

    # Single intensity channel as float64
    intensity = blurred[:, :, 0].astype(np.float64)

    # ------------------------------------------------------------------
    # 2. Sobel kernels
    # ------------------------------------------------------------------
    X_KERNEL = np.array([[1,  0, -1],
                         [2,  0, -2],
                         [1,  0, -1]], dtype=np.float64)

    Y_KERNEL = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]], dtype=np.float64)

    # ------------------------------------------------------------------
    # 3. Vectorised convolution — replaces the Python pixel loop entirely
    # ------------------------------------------------------------------
    gx = _convolve2d(intensity, X_KERNEL)
    gy = _convolve2d(intensity, Y_KERNEL)

    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # ------------------------------------------------------------------
    # 4. Normalise for visualisation
    # ------------------------------------------------------------------
    gx_vis        = _normalize_to_uint8(gx)
    gy_vis        = _normalize_to_uint8(gy)
    magnitude_vis = _normalize_to_uint8(magnitude)

    final_edge_rgb = np.stack([magnitude_vis, magnitude_vis, magnitude_vis], axis=-1)

    return final_edge_rgb, gx_vis, gy_vis, magnitude_vis