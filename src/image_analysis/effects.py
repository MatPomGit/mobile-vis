"""Artistic and visual effects for image processing.

Provides five distinct visual effects that transform an image beyond standard
filtering:

* :func:`apply_invert`   - colour inversion (bitwise NOT)
* :func:`apply_sepia`    - warm sepia-tone look
* :func:`apply_emboss`   - emboss / relief surface texture
* :func:`apply_pixelate` - blocky pixel-art pixelation
* :func:`apply_cartoon`  - cartoon / comic-book rendering

All functions expect **BGR** ``uint8`` NumPy arrays with shape ``(H, W, 3)``
and return an array of the same shape and dtype unless noted otherwise.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from numpy.typing import NDArray

from .utils import validate_image

logger = logging.getLogger(__name__)

# Named constant for the pixel block size used by :func:`apply_pixelate`.
PIXELATE_BLOCK_SIZE: int = 16


def apply_invert(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Invert all colour channels using a bitwise NOT operation.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.

    Returns:
        Inverted BGR image with the same shape and dtype.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unexpected shape or dtype.
    """
    validate_image(image)
    result: NDArray[np.uint8] = cv2.bitwise_not(image)
    logger.debug("apply_invert: shape=%s", image.shape)
    return result


def apply_sepia(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Apply a warm sepia-tone effect to the image.

    The effect is produced by projecting the BGR pixel values through a
    fixed 3x3 colour matrix that emphasises red/green and suppresses blue,
    mimicking the appearance of aged photographic prints.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.

    Returns:
        Sepia-toned BGR image with the same shape and dtype.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unexpected shape or dtype.
    """
    validate_image(image)
    # Sepia kernel: each output channel is a weighted sum of input BGR channels.
    # Coefficients taken from the classic Adobe Photoshop sepia preset.
    kernel = np.array(
        [
            [0.272, 0.534, 0.131],  # output B
            [0.349, 0.686, 0.168],  # output G
            [0.393, 0.769, 0.189],  # output R
        ],
        dtype=np.float32,
    )
    sepia_unclamped = cv2.transform(image.astype(np.float32), kernel)
    result: NDArray[np.uint8] = np.clip(sepia_unclamped, 0, 255).astype(np.uint8)
    logger.debug("apply_sepia: shape=%s", image.shape)
    return result


def apply_emboss(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Produce an emboss / relief texture effect.

    A directional 3x3 convolution kernel highlights intensity gradients as
    raised surface relief.  The result is shifted by 128 so that flat areas
    appear as mid-grey.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.

    Returns:
        Embossed grayscale image returned as a 3-channel BGR array with the
        same ``(H, W, 3)`` shape and ``uint8`` dtype.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unexpected shape or dtype.
    """
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array(
        [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
        dtype=np.float32,
    )
    embossed = cv2.filter2D(gray, -1, kernel).astype(np.int16) + 128
    embossed_u8 = np.clip(embossed, 0, 255).astype(np.uint8)
    result: NDArray[np.uint8] = cv2.cvtColor(embossed_u8, cv2.COLOR_GRAY2BGR)
    logger.debug("apply_emboss: shape=%s", image.shape)
    return result


def apply_pixelate(
    image: NDArray[np.uint8],
    block_size: int = PIXELATE_BLOCK_SIZE,
) -> NDArray[np.uint8]:
    """Pixelate the image by downsampling and upsampling with nearest-neighbour.

    The image is first scaled down by *block_size* using area-averaging and
    then scaled back to the original resolution with nearest-neighbour
    interpolation, producing a blocky pixel-art appearance.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.
        block_size: Side length of each pixel block in the output.  Must be
            a positive integer.  Defaults to :data:`PIXELATE_BLOCK_SIZE`
            (16 pixels).

    Returns:
        Pixelated BGR image with the same shape and dtype as *image*.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unexpected shape or dtype, or if
            *block_size* is not a positive integer.
    """
    validate_image(image)
    if block_size <= 0:
        raise ValueError(f"block_size must be a positive integer, got {block_size}")
    h, w = image.shape[:2]
    small_w = max(1, w // block_size)
    small_h = max(1, h // block_size)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    result: NDArray[np.uint8] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    logger.debug("apply_pixelate: shape=%s block_size=%d", image.shape, block_size)
    return result


def apply_cartoon(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Render the image with a cartoon / comic-book look.

    The effect is achieved in three stages:

    1. **Colour smoothing** - the image is smoothed with a bilateral filter to
       flatten colour regions while preserving edge sharpness.
    2. **Edge extraction** - Canny edges are detected on the median-blurred
       grayscale version of the image.
    3. **Compositing** - the smoothed colour is masked so that detected edges
       appear as black outlines.

    Args:
        image: BGR image array with shape ``(H, W, 3)``, dtype ``uint8``.

    Returns:
        Cartoon-rendered BGR image with the same shape and dtype.

    Raises:
        TypeError: If *image* is not a ``np.ndarray``.
        ValueError: If *image* has an unexpected shape or dtype.
    """
    validate_image(image)

    # Stage 1: smooth colour regions
    smoothed = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

    # Stage 2: extract edges on grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(blurred, threshold1=80, threshold2=200)

    # Invert: edge pixels are 0 (black), non-edge pixels are 255 (white)
    edge_mask = cv2.bitwise_not(edges)

    # Dilate to make lines slightly thicker
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge_mask_dilated = cv2.dilate(edge_mask, kernel)

    # Stage 3: apply mask - zero-out smoothed pixels where edges are detected
    edge_mask_3ch = cv2.cvtColor(edge_mask_dilated, cv2.COLOR_GRAY2BGR)
    result: NDArray[np.uint8] = cv2.bitwise_and(smoothed, edge_mask_3ch)

    logger.debug("apply_cartoon: shape=%s", image.shape)
    return result
