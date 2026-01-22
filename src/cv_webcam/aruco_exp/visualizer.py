"""Visualization tools for experiment results."""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike

from cv_webcam import IMAGES_DIR


def draw_low_light_imgs(save: bool = False) -> None:
    """Display low light dataset samples at different levels."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), constrained_layout=True)

    scale = 0.4
    for lv in range(4):
        img_path = IMAGES_DIR / "experiment" / "low_light" / f"img_0_dark_lv{lv}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        assert img is not None

        img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )

        axes[lv // 2, lv % 2].imshow(img)
        axes[lv // 2, lv % 2].set_title(f"Low Light Level {lv}")
        axes[lv // 2, lv % 2].axis("off")

    if save:
        save_path = IMAGES_DIR / "experiment" / "display_data" / "low_light_imgs.pdf"
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
            format="pdf",
        )
    else:
        plt.show()


def draw_non_uniform_imgs(save: bool = False) -> None:
    """Display non-uniform illumination dataset samples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 3), constrained_layout=True)

    scale = 0.4
    for i in range(3):
        s = 300 + i * 200
        img_path = IMAGES_DIR / "experiment" / "non_uniform" / f"img_0_sigma{s}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        assert img is not None

        img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )

        axes[i].imshow(img)
        axes[i].set_title(f"Non-uniform Image (Sigma={s})")
        axes[i].axis("off")

    if save:
        save_path = IMAGES_DIR / "experiment" / "display_data" / "non_uniform_imgs.pdf"
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
            format="pdf",
        )
    else:
        plt.show()


def draw_he_compare(img: MatLike) -> None:
    """Compare original image with histogram equalized version."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[1, 0].hist(
        img.ravel(),
        bins=256,
        range=(0, 256),
        density=True,
        cumulative=True,
        color="blue",
        alpha=0.85,
    )
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Original Image Histogram", fontsize=10)
    axes[1, 0].set_xlabel("Gray value")
    axes[1, 0].set_ylabel("Count")

    img_he = cv2.equalizeHist(img)

    axes[0, 1].imshow(img_he, cmap="gray")
    axes[0, 1].set_title("Histogram Equalized Image")
    axes[0, 1].axis("off")

    axes[1, 1].hist(
        img_he.ravel(),
        bins=256,
        range=(0, 256),
        density=True,
        cumulative=True,
        color="blue",
        alpha=0.85,
    )
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Equalized Image Histogram", fontsize=10)
    axes[1, 1].set_xlabel("Gray value")
    axes[1, 1].set_ylabel("Count")

    plt.show()


def plot_comparison_simple(results: list[dict], save_path: Path | None = None) -> None:
    """Plot simple comparison of algorithm performance across datasets.

    Args:
        results: List of result dictionaries with keys: dataset, algorithm, success_rate
        save_path: Path to save figure, or None to display
    """
    low_light = [r for r in results if r["dataset"] == "low_light"]
    non_uniform = [r for r in results if r["dataset"] == "non_uniform"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if low_light:
        algos = [r["algorithm"] for r in low_light]
        rates = [r["success_rate"] for r in low_light]
        bars = ax1.bar(
            algos, rates, color=["#3498db", "#e74c3c", "#2ecc71"][: len(algos)]
        )
        ax1.set_ylabel("Success Rate (%)")
        ax1.set_title("Low Light Dataset")
        ax1.set_ylim(0, 110)

        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    if non_uniform:
        algos = [r["algorithm"] for r in non_uniform]
        rates = [r["success_rate"] for r in non_uniform]
        bars = ax2.bar(
            algos, rates, color=["#3498db", "#e74c3c", "#2ecc71"][: len(algos)]
        )
        ax2.set_ylabel("Success Rate (%)")
        ax2.set_title("Non-uniform Dataset")
        ax2.set_ylim(0, 110)

        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()


def plot_evaluation_results(
    results: dict[str, dict[str, dict[str, float]]],
    save: bool = False,
    algorithm_order: list[str] | None = None,
    algorithm_labels: dict[str, str] | None = None,
) -> None:
    """Plot comprehensive evaluation results as grouped bar charts.

    Args:
        results: Nested dict {filter_config: {preprocessing: {metric: value}}}
        save: If True, save to PDF. If False, display interactively.
        algorithm_order: Custom order of algorithms to display. If None, use default.
        algorithm_labels: Custom display labels for algorithms. If None, use default.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.5)

    filter_configs = list(results.keys())

    # Use custom algorithm order or default
    if algorithm_order is None:
        preprocessing_methods = ["none", "clahe", "retinex"]
    else:
        preprocessing_methods = algorithm_order

    # Use custom labels or generate from algorithm names
    if algorithm_labels is None:
        default_labels = {"none": "None", "clahe": "CLAHE", "retinex": "Retinex"}
        preprocessing_labels = [
            default_labels.get(m, m.upper()) for m in preprocessing_methods
        ]
    else:
        preprocessing_labels = [
            algorithm_labels.get(m, m.upper()) for m in preprocessing_methods
        ]

    x = np.arange(len(preprocessing_methods))
    width = 0.35
    colors = {"low_light": "#3498db", "non_uniform": "#e74c3c"}

    axes_flat = axes.flatten()

    for idx, (ax, filter_name) in enumerate(zip(axes_flat, filter_configs)):
        low_light_rates = [
            results[filter_name].get(prep, {}).get("low_light", 0)
            for prep in preprocessing_methods
        ]
        non_uniform_rates = [
            results[filter_name].get(prep, {}).get("non_uniform", 0)
            for prep in preprocessing_methods
        ]

        bars1 = ax.bar(
            x - width / 2,
            low_light_rates,
            width,
            label="Low Light",
            color=colors["low_light"],
            alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2,
            non_uniform_rates,
            width,
            label="Non-uniform",
            color=colors["non_uniform"],
            alpha=0.8,
        )

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        ax.set_xlabel("Preprocessing Method", fontsize=11, fontweight="bold")
        ax.set_ylabel("Detection Success Rate (%)", fontsize=11, fontweight="bold")
        ax.set_title(filter_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(preprocessing_labels)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=10)

    fig.suptitle(
        "ArUco Detection Success Rate: Filter Configurations and Preprocessing Methods",
        fontsize=14,
        fontweight="bold",
    )

    if save:
        save_path = (
            IMAGES_DIR / "experiment" / "display_data" / "evaluation_results.pdf"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300, format="pdf")
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()


def plot_performance_comparison(
    results: dict[str, dict[str, dict[str, float]]],
    save: bool = False,
    algorithm_order: list[str] | None = None,
    algorithm_labels: dict[str, str] | None = None,
) -> None:
    """Plot processing time comparison for different methods.

    Args:
        results: Nested dict {filter_config: {preprocessing: {metric: value}}}
        save: If True, save to PDF. If False, display interactively.
        algorithm_order: Custom order of algorithms to display. If None, use default.
        algorithm_labels: Custom display labels for algorithms. If None, use default.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.5)

    filter_configs = list(results.keys())

    # Use custom algorithm order or default
    if algorithm_order is None:
        preprocessing_methods = ["none", "clahe", "retinex"]
    else:
        preprocessing_methods = algorithm_order

    # Use custom labels or generate from algorithm names
    if algorithm_labels is None:
        default_labels = {"none": "None", "clahe": "CLAHE", "retinex": "Retinex"}
        preprocessing_labels = [
            default_labels.get(m, m.upper()) for m in preprocessing_methods
        ]
    else:
        preprocessing_labels = [
            algorithm_labels.get(m, m.upper()) for m in preprocessing_methods
        ]

    x = np.arange(len(preprocessing_methods))
    width = 0.28
    colors = {"low_light": "#9b59b6", "non_uniform": "#f39c12", "total": "#2ecc71"}

    axes_flat = axes.flatten()

    for idx, (ax, filter_name) in enumerate(zip(axes_flat, filter_configs)):
        time_low_light = [
            results[filter_name].get(prep, {}).get("time_low_light", 0)
            for prep in preprocessing_methods
        ]
        time_non_uniform = [
            results[filter_name].get(prep, {}).get("time_non_uniform", 0)
            for prep in preprocessing_methods
        ]
        time_total = [
            results[filter_name].get(prep, {}).get("time_total", 0)
            for prep in preprocessing_methods
        ]

        bars1 = ax.bar(
            x - width,
            time_low_light,
            width,
            label="Low Light",
            color=colors["low_light"],
            alpha=0.8,
        )
        bars2 = ax.bar(
            x,
            time_non_uniform,
            width,
            label="Non-uniform",
            color=colors["non_uniform"],
            alpha=0.8,
        )
        bars3 = ax.bar(
            x + width,
            time_total,
            width,
            label="Total",
            color=colors["total"],
            alpha=0.8,
        )

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}s",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Preprocessing Method", fontsize=11, fontweight="bold")
        ax.set_ylabel("Processing Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_title(filter_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(preprocessing_labels)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "ArUco Detection Processing Time: Filter Configurations and Preprocessing Methods",
        fontsize=14,
        fontweight="bold",
    )

    if save:
        save_path = (
            IMAGES_DIR / "experiment" / "display_data" / "performance_comparison.pdf"
        )
        plt.savefig(save_path, bbox_inches="tight", dpi=300, format="pdf")
        print(f"✓ Performance plot saved to: {save_path}")
    else:
        plt.show()
