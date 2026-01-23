"""Analysis and visualization tools for image processing algorithms."""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from matplotlib.axes import Axes

from cv_webcam.core.img_prep import gain_compensation, single_scale_retinex


def _add_distribution_plot(
    ax: Axes,
    data: np.ndarray,
    title: str,
    color: str,
    x_label: str,
    show_stats: bool = True,
) -> None:
    """Add distribution histogram with statistics to a subplot.

    Args:
        ax: Matplotlib axis to plot on
        data: Flattened data array
        title: Subplot title
        color: Histogram color
        x_label: X-axis label
        show_stats: Whether to show statistics annotation
    """
    # Plot histogram
    ax.hist(
        data,
        bins=100,
        alpha=0.7,
        color=color,
        edgecolor="black",
        linewidth=0.5,
        density=True,
    )

    if show_stats:
        # Calculate statistics
        data_min, data_max = data.min(), data.max()
        data_mean, data_std = data.mean(), data.std()
        data_median = np.median(data)  # type: ignore

        # Add statistics text box
        stats_text = (
            f"Min: {data_min:.2f}\n"
            f"Max: {data_max:.2f}\n"
            f"Mean: {data_mean:.2f}\n"
            f"Median: {data_median:.2f}\n"
            f"Std: {data_std:.2f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
        )

        # Add reference lines
        ax.axvline(data_min, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(data_max, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(data_mean, color="green", linestyle="-", linewidth=1.5, alpha=0.7)

    # Set labels and styling
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Density (Proportion)", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def show_retinex_distribution(img: MatLike) -> None:
    """Analyze and visualize Retinex algorithm data distributions.

    Shows three figures comparing different normalization strategies:
    - Figure 1: Raw distributions (before normalization)
    - Figure 2: cv2.normalize distributions
    - Figure 3: gain_compensation distributions

    Args:
        img: Input grayscale image
    """
    # Compute all data variants
    retinex_log = single_scale_retinex(img, sigma=80)
    normalized_retinex_log = cv2.normalize(retinex_log, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    gain_retinex_log = gain_compensation(retinex_log)

    retinex_exp = np.expm1(retinex_log)
    normalized_retinex_exp = cv2.normalize(retinex_exp, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    gain_retinex_exp = gain_compensation(retinex_exp)

    # Define plot configurations
    plot_groups = [
        {
            "title": "Raw Retinex Distribution (Before Normalization)",
            "plots": [
                {
                    "data": retinex_log.flatten(),
                    "title": "Log-Scale Retinex",
                    "color": "#2E86AB",
                    "x_label": "Pixel Value",
                },
                {
                    "data": retinex_exp.flatten(),
                    "title": "Exponential Retinex",
                    "color": "#D81159",
                    "x_label": "Pixel Value",
                },
            ],
        },
        {
            "title": "Normalized Retinex Distribution (After Normalization to 0-255)",
            "plots": [
                {
                    "data": normalized_retinex_log.flatten(),
                    "title": "Normalized Log-Scale Retinex",
                    "color": "#06A77D",
                    "x_label": "Pixel Value (0-255)",
                },
                {
                    "data": normalized_retinex_exp.flatten(),
                    "title": "Normalized Exponential Retinex",
                    "color": "#8F2D56",
                    "x_label": "Pixel Value (0-255)",
                },
            ],
        },
        {
            "title": "Gain Compensation Distribution (Using gain_compensation instead of normalize)",
            "plots": [
                {
                    "data": gain_retinex_log.flatten(),
                    "title": "Gain Compensated Log-Scale Retinex",
                    "color": "#6A4C93",
                    "x_label": "Pixel Value (0-255)",
                },
                {
                    "data": gain_retinex_exp.flatten(),
                    "title": "Gain Compensated Exponential Retinex",
                    "color": "#E63946",
                    "x_label": "Pixel Value (0-255)",
                },
            ],
        },
    ]

    # Generate figures
    for group in plot_groups:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(group["title"], fontsize=14, fontweight="bold", y=0.98)

        for i, plot_spec in enumerate(group["plots"]):
            _add_distribution_plot(
                axes[i],
                plot_spec["data"],
                plot_spec["title"],
                plot_spec["color"],
                plot_spec["x_label"],
            )

        plt.tight_layout()

    plt.show()
