from typing import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike


def single_scale_retinex(img: MatLike, sigma: float = 80) -> MatLike:
    """Single Scale Retinex (SSR) algorithm.

    Args:
        img (MatLike): input image
        sigma (float, optional): Gaussian blur sigma. Defaults to 80.

    Returns:
        MatLike: Retinex processed image in log scale and float32 format.
    """
    log_img = np.log1p(img.astype(np.float32))

    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    log_blur = np.log1p(blurred.astype(np.float32))

    retinex = log_img - log_blur
    return retinex


def multi_scale_retinex(
    img: MatLike,
    sigmas: Sequence[int] = [15, 80, 250],
    weights: Sequence[float] = [1 / 3, 1 / 3, 1 / 3],
) -> MatLike:
    """Multi Scale Retinex (MSR) algorithm.

    Args:
        img (MatLike): input image
        sigmas (Sequence[int], optional): Gaussian blur sigmas for different scales. Defaults to [15, 80, 250].
        weights (Sequence[float], optional): Weights for each scale. Defaults to [1 / 3, 1 / 3, 1 / 3].

    Returns:
        MatLike: Retinex processed image in log scale and float32 format.
    """
    assert sum(weights) - 1.0 < 1e-6, "Weights must sum to 1."

    msr = np.zeros_like(img, dtype=np.float32)

    for sigma, weight in zip(sigmas, weights):
        ssr = single_scale_retinex(img, sigma)
        msr += weight * ssr

    return msr


def color_restoration(img: MatLike, alpha: float = 125, beta: float = 46) -> MatLike:
    channels = list(cv2.split(img))

    for i in range(3):
        channels[i] = beta * (
            np.log1p(channels[i]) - np.log1p(alpha * np.mean(channels[i]))  # type: ignore
        )

    return cv2.merge(channels)


def adaptive_gaussian(img: MatLike, sigma: float) -> MatLike:
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    adapt = 1 + 0.5 * (grad_x**2 + grad_y**2)

    blurred = cv2.GaussianBlur(img * adapt, (0, 0), sigma)  # type: ignore
    return blurred / (adapt + 1e-6)


def gain_compensation(img: MatLike) -> MatLike:
    mean = img.mean()
    std = img.std()

    k = 2.5
    min_val = mean - k * std
    max_val = mean + k * std

    result = (img - min_val) / (max_val - min_val) * 255
    result = np.clip(result, 0, 255)

    return result


def retinex_test(img: MatLike) -> MatLike:
    # retinex_img = multi_scale_retinex(img, sigmas=[15, 80, 180])
    retinex_img = single_scale_retinex(img, sigma=80)

    retinex_img = np.expm1(retinex_img)
    final_img = cv2.normalize(retinex_img, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    # final_img = gain_compensation(retinex_img)
    return final_img.astype(np.uint8)
