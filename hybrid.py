#!/usr/bin/env python3
"""Image filtering and hybrid-image construction implemented from scratch.

The filtering functions in this file intentionally do not call ready-made
filtering routines from NumPy, SciPy, OpenCV, or Pillow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def _as_float_image(img: np.ndarray) -> np.ndarray:
    """Validate an image and return a float64 copy without changing its scale."""
    array = np.asarray(img)
    if array.ndim not in (2, 3):
        raise ValueError("img must be a 2-D grayscale or 3-D color array")
    if array.ndim == 3 and array.shape[2] == 0:
        raise ValueError("img must contain at least one channel")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("img dimensions must be non-zero")
    return array.astype(np.float64, copy=True)


def cross_correlation_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Cross-correlate a grayscale or color image with a 2-D kernel.

    Zero padding is used and the returned image has the same height, width,
    and (for color input) channel count as ``img``.
    """
    image = _as_float_image(img)
    kernel_array = np.asarray(kernel, dtype=np.float64)

    if kernel_array.ndim != 2:
        raise ValueError("kernel must be a 2-D array")
    if kernel_array.shape[0] == 0 or kernel_array.shape[1] == 0:
        raise ValueError("kernel dimensions must be non-zero")

    kernel_height, kernel_width = kernel_array.shape
    pad_top = kernel_height // 2
    pad_bottom = kernel_height - 1 - pad_top
    pad_left = kernel_width // 2
    pad_right = kernel_width - 1 - pad_left

    if image.ndim == 2:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
        )
    else:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
        )

    output = np.zeros(image.shape, dtype=np.float64)
    height, width = image.shape[:2]

    # Apply every kernel coefficient to every pixel. The loops walk over the
    # kernel while NumPy vectorizes the pixel operations; no filtering routine
    # from NumPy, SciPy, OpenCV, or Pillow is used.
    for kernel_row in range(kernel_height):
        for kernel_column in range(kernel_width):
            source = padded[
                kernel_row : kernel_row + height,
                kernel_column : kernel_column + width,
                ...,
            ]
            output += kernel_array[kernel_row, kernel_column] * source

    return output


def convolve_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve ``img`` with ``kernel`` using cross-correlation."""
    kernel_array = np.asarray(kernel)
    if kernel_array.ndim != 2:
        raise ValueError("kernel must be a 2-D array")
    flipped_kernel = np.flip(kernel_array, axis=(0, 1))
    return cross_correlation_2d(img, flipped_kernel)


def gaussian_blur_kernel_2d(
    sigma: float, height: int, width: int
) -> np.ndarray:
    """Return a normalized Gaussian kernel of the requested dimensions."""
    if sigma <= 0:
        raise ValueError("sigma must be greater than zero")
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise TypeError("height and width must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be greater than zero")

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    y_coordinates = np.arange(height, dtype=np.float64) - center_y
    x_coordinates = np.arange(width, dtype=np.float64) - center_x
    squared_distance = (
        y_coordinates[:, np.newaxis] ** 2
        + x_coordinates[np.newaxis, :] ** 2
    )
    kernel = np.exp(-squared_distance / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel


def _kernel_dimensions(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, (int, np.integer)):
        height = width = int(size)
    else:
        values = tuple(size)
        if len(values) != 2:
            raise ValueError("size must be an integer or a (height, width) pair")
        height, width = values
    if not isinstance(height, (int, np.integer)) or not isinstance(
        width, (int, np.integer)
    ):
        raise TypeError("kernel dimensions must be integers")
    if height <= 0 or width <= 0:
        raise ValueError("kernel dimensions must be greater than zero")
    return int(height), int(width)


def low_pass(
    img: np.ndarray, sigma: float, size: int | Sequence[int]
) -> np.ndarray:
    """Keep the low-frequency component using this project's Gaussian blur."""
    height, width = _kernel_dimensions(size)
    kernel = gaussian_blur_kernel_2d(sigma, height, width)
    return convolve_2d(img, kernel)


def high_pass(
    img: np.ndarray, sigma: float, size: int | Sequence[int]
) -> np.ndarray:
    """Keep the high-frequency component by subtracting the low-pass image."""
    image = _as_float_image(img)
    return image - low_pass(image, sigma, size)


def create_hybrid_image(
    low_image: np.ndarray,
    high_image: np.ndarray,
    low_sigma: float,
    low_size: int | Sequence[int],
    high_sigma: float,
    high_size: int | Sequence[int],
    low_weight: float = 1.0,
    high_weight: float = 1.0,
) -> np.ndarray:
    """Combine the low frequencies of one image with highs of another."""
    low_source = _as_float_image(low_image)
    high_source = _as_float_image(high_image)
    if low_source.shape != high_source.shape:
        raise ValueError("the two aligned input images must have identical shapes")
    low_component = low_pass(low_source, low_sigma, low_size)
    high_component = high_pass(high_source, high_sigma, high_size)
    return low_weight * low_component + high_weight * high_component


def _load_rgb_unit(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def _save_rgb_unit(image: np.ndarray, path: Path) -> None:
    display_image = np.clip(image, 0.0, 1.0)
    output = np.rint(display_image * 255.0).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="RGB").save(path)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a hybrid image from two already aligned images. The first "
            "provides low frequencies and the second provides high frequencies."
        )
    )
    parser.add_argument("left", type=Path, help="aligned low-frequency image")
    parser.add_argument("right", type=Path, help="aligned high-frequency image")
    parser.add_argument("-o", "--output", type=Path, default=Path("hybrid.png"))
    parser.add_argument("--low-sigma", type=_positive_float, default=6.0)
    parser.add_argument("--low-kernel", type=_positive_int, default=37)
    parser.add_argument("--high-sigma", type=_positive_float, default=2.5)
    parser.add_argument("--high-kernel", type=_positive_int, default=17)
    parser.add_argument("--low-weight", type=float, default=0.95)
    parser.add_argument("--high-weight", type=float, default=1.25)
    args = parser.parse_args()

    left_image = _load_rgb_unit(args.left)
    right_image = _load_rgb_unit(args.right)
    hybrid = create_hybrid_image(
        left_image,
        right_image,
        args.low_sigma,
        args.low_kernel,
        args.high_sigma,
        args.high_kernel,
        args.low_weight,
        args.high_weight,
    )
    _save_rgb_unit(hybrid, args.output)
    print(f"Saved hybrid image: {args.output}")


if __name__ == "__main__":
    main()
