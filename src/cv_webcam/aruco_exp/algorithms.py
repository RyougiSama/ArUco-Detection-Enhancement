"""Image preprocessing algorithms for ArUco detection experiments."""

from typing import Protocol

import cv2
import numpy as np
from cv2.typing import MatLike

from cv_webcam.core import img_prep


def apply_filter(
    img: MatLike, filter_type: str | None, params: dict | None = None
) -> MatLike:
    """Apply specified filter to image.

    Args:
        img: Input image
        filter_type: Filter type ("gaussian" | "median" | "bilateral" | None)
        params: Filter-specific parameters

    Returns:
        Filtered image

    Raises:
        ValueError: If filter_type is unknown
    """
    if filter_type is None:
        return img

    params = params or {}

    match filter_type:
        case "gaussian":
            ksize = params.get("ksize", (3, 3))
            sigma = params.get("sigma", 0)
            return cv2.GaussianBlur(img, ksize, sigma)

        case "median":
            ksize = params.get("ksize", 5)
            return cv2.medianBlur(img, ksize)

        case "bilateral":
            d = params.get("d", 9)
            sigma_color = params.get("sigma_color", 75)
            sigma_space = params.get("sigma_space", 75)
            return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

        case _:
            raise ValueError(f"Unknown filter type: {filter_type}")


class PreprocessAlgorithm(Protocol):
    """Protocol for preprocessing algorithms."""

    name: str

    def __call__(self, img: MatLike) -> MatLike:
        """Process grayscale image and return processed result."""
        ...


class NoPreprocess:
    """No preprocessing, only applies standard filtering."""

    name = "none"

    def __init__(
        self,
        prefilter: str | None = None,
        postfilter: str | None = None,
        prefilter_params: dict | None = None,
        postfilter_params: dict | None = None,
        # Backward compatibility
        use_gaussian_prefilter: bool = False,
        use_median_postfilter: bool = False,
    ):
        # Handle backward compatibility
        if use_gaussian_prefilter and prefilter is None:
            prefilter = "gaussian"
        if use_median_postfilter and postfilter is None:
            postfilter = "median"

        self.prefilter = prefilter
        self.postfilter = postfilter
        self.prefilter_params = prefilter_params or {}
        self.postfilter_params = postfilter_params or {}

    def __call__(self, img: MatLike) -> MatLike:
        result = apply_filter(img, self.prefilter, self.prefilter_params)
        result = apply_filter(result, self.postfilter, self.postfilter_params)
        return result


class CLAHEPreprocess:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing."""

    name = "clahe"

    def __init__(
        self,
        clip_limit: float = 2.0,
        prefilter: str | None = None,
        postfilter: str | None = None,
        prefilter_params: dict | None = None,
        postfilter_params: dict | None = None,
        # Backward compatibility
        use_gaussian_prefilter: bool = False,
        use_median_postfilter: bool = False,
    ):
        # Handle backward compatibility
        if use_gaussian_prefilter and prefilter is None:
            prefilter = "gaussian"
        if use_median_postfilter and postfilter is None:
            postfilter = "median"

        self.clip_limit = clip_limit
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.prefilter_params = prefilter_params or {}
        self.postfilter_params = postfilter_params or {}
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit)

    def __call__(self, img: MatLike) -> MatLike:
        result = apply_filter(img, self.prefilter, self.prefilter_params)
        result = self.clahe.apply(result)
        result = apply_filter(result, self.postfilter, self.postfilter_params)
        return result


class RetinexPreprocess:
    """Retinex-based preprocessing for low-light enhancement."""

    name = "retinex"

    def __init__(
        self,
        sigma: float = 80,
        use_log_scale: bool = True,
        prefilter: str | None = None,
        postfilter: str | None = None,
        prefilter_params: dict | None = None,
        postfilter_params: dict | None = None,
        # Backward compatibility
        use_gaussian_prefilter: bool = False,
        use_median_postfilter: bool = False,
    ):
        # Handle backward compatibility
        if use_gaussian_prefilter and prefilter is None:
            prefilter = "gaussian"
        if use_median_postfilter and postfilter is None:
            postfilter = "median"

        self.sigma = sigma
        self.use_log_scale = use_log_scale
        self.prefilter = prefilter
        self.postfilter = postfilter
        self.prefilter_params = prefilter_params or {}
        self.postfilter_params = postfilter_params or {}

    def __call__(self, img: MatLike) -> MatLike:
        result = apply_filter(img, self.prefilter, self.prefilter_params)

        result = img_prep.single_scale_retinex(result, self.sigma)

        if not self.use_log_scale:
            result = np.expm1(result)

        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(  # type: ignore
            "uint8"
        )

        result = apply_filter(result, self.postfilter, self.postfilter_params)

        return result


# Algorithm registry
ALGORITHMS: dict[str, type[PreprocessAlgorithm]] = {
    "none": NoPreprocess,
    "clahe": CLAHEPreprocess,
    "retinex": RetinexPreprocess,
}


def get_algorithm(
    name: str,
    **kwargs,
) -> PreprocessAlgorithm:
    """Get algorithm instance by name.

    Args:
        name: Algorithm name
        **kwargs: Additional algorithm-specific parameters (e.g., clip_limit, sigma)

    Returns:
        Algorithm instance

    Raises:
        KeyError: If algorithm name not found
    """
    if name not in ALGORITHMS:
        raise KeyError(
            f"Algorithm '{name}' not found. Available: {list(ALGORITHMS.keys())}"
        )

    algo_class = ALGORITHMS[name]
    return algo_class(
        **kwargs,
    )


def list_algorithms() -> list[str]:
    """List all available algorithm names."""
    return list(ALGORITHMS.keys())
