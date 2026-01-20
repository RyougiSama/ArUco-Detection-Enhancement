"""Image preprocessing algorithms for ArUco detection experiments."""

from typing import Protocol

import cv2
from cv2.typing import MatLike

from cv_webcam.core import img_prep


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
        use_gaussian_prefilter: bool = True,
        use_median_postfilter: bool = True,
    ):
        self.use_gaussian_prefilter = use_gaussian_prefilter
        self.use_median_postfilter = use_median_postfilter

    def __call__(self, img: MatLike) -> MatLike:
        result = img.copy()

        if self.use_gaussian_prefilter:
            result = cv2.GaussianBlur(result, (3, 3), 0)

        if self.use_median_postfilter:
            result = cv2.medianBlur(result, 5)

        return result


class CLAHEPreprocess:
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) preprocessing."""

    name = "clahe"

    def __init__(
        self,
        use_gaussian_prefilter: bool = True,
        use_median_postfilter: bool = True,
        clip_limit: float = 2.0,
    ):
        self.clip_limit = clip_limit
        self.use_gaussian_prefilter = use_gaussian_prefilter
        self.use_median_postfilter = use_median_postfilter
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit)

    def __call__(self, img: MatLike) -> MatLike:
        result = img.copy()

        if self.use_gaussian_prefilter:
            result = cv2.GaussianBlur(result, (3, 3), 0)

        result = self.clahe.apply(result)

        if self.use_median_postfilter:
            result = cv2.medianBlur(result, 5)

        return result


class RetinexPreprocess:
    """Retinex-based preprocessing for low-light enhancement."""

    name = "retinex"

    def __init__(
        self,
        use_gaussian_prefilter: bool = True,
        use_median_postfilter: bool = True,
        sigma: float = 80,
    ):
        self.sigma = sigma
        self.use_gaussian_prefilter = use_gaussian_prefilter
        self.use_median_postfilter = use_median_postfilter

    def __call__(self, img: MatLike) -> MatLike:
        result = img.copy()

        if self.use_gaussian_prefilter:
            result = cv2.GaussianBlur(result, (3, 3), 0)

        result = img_prep.retinex_test(result)

        if self.use_median_postfilter:
            result = cv2.medianBlur(result, 5)

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
