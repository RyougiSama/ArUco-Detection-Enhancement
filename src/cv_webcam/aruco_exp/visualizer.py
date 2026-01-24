"""Visualization tools for experiment results."""

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike

from cv_webcam import DATA_DIR, IMAGES_DIR


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


def visualize_sigma_comparison(
    filename: str = "sigma_comparison.json",
    save: bool = False,
) -> None:
    """Visualize sigma parameter sweep results with dual-axis line plot.

    Args:
        filename: JSON file containing sigma comparison results
        save: If True, save plot to PDF. If False, display interactively.
    """
    from pathlib import Path

    # Load results
    load_path = Path(filename)
    if not load_path.is_absolute():
        load_path = DATA_DIR / "experiment" / filename

    with open(load_path) as f:
        results = json.load(f)

    # Organize data by dataset and sigma
    data = {}
    for r in results:
        dataset = r["dataset"]
        sigma = r["algorithm_params"]["sigma"]

        if dataset not in data:
            data[dataset] = {"sigmas": [], "success_rates": [], "times": []}

        data[dataset]["sigmas"].append(sigma)
        data[dataset]["success_rates"].append(r["success_rate"])
        data[dataset]["times"].append(r["processing_time"])

    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Define colors for datasets
    colors = {
        "low_light": {"success": "#2E86AB", "time": "#D81159"},
        "non_uniform": {"success": "#06A77D", "time": "#8F2D56"},
    }

    markers = {"low_light": "o", "non_uniform": "s"}

    # Plot success rates on left y-axis
    for dataset, color_set in colors.items():
        if dataset in data:
            ax1.plot(
                data[dataset]["sigmas"],
                data[dataset]["success_rates"],
                color=color_set["success"],
                marker=markers[dataset],
                markersize=8,
                linewidth=2.5,
                label=f"{dataset.replace('_', ' ').title()} - Success Rate",
                alpha=0.9,
            )

            # Add value labels with staggered positioning
            # Use different vertical offsets for different datasets to avoid overlap
            y_offset = 15 if dataset == "low_light" else 25
            for x, y in zip(data[dataset]["sigmas"], data[dataset]["success_rates"]):
                ax1.annotate(
                    f"{y:.1f}%",
                    xy=(x, y),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color_set["success"],
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )

    ax1.set_xlabel("Sigma (σ)", fontsize=13, fontweight="bold")
    ax1.set_ylabel(
        "Detection Success Rate (%)", fontsize=13, fontweight="bold", color="#2E86AB"
    )
    ax1.tick_params(axis="y", labelcolor="#2E86AB", labelsize=11)
    ax1.tick_params(axis="x", labelsize=11)
    ax1.set_ylim(0, 110)
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # Create second y-axis for processing time
    ax2 = ax1.twinx()

    for dataset, color_set in colors.items():
        if dataset in data:
            ax2.plot(
                data[dataset]["sigmas"],
                data[dataset]["times"],
                color=color_set["time"],
                marker=markers[dataset],
                markersize=8,
                linewidth=2.5,
                linestyle="--",
                label=f"{dataset.replace('_', ' ').title()} - Processing Time",
                alpha=0.9,
            )

            # Add value labels with staggered positioning
            # Use different vertical offsets for different datasets to avoid overlap
            y_offset = -20 if dataset == "low_light" else -30
            for x, y in zip(data[dataset]["sigmas"], data[dataset]["times"]):
                ax2.annotate(
                    f"{y:.2f}s",
                    xy=(x, y),
                    xytext=(0, y_offset),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=color_set["time"],
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor="none",
                    ),
                )

    ax2.set_ylabel(
        "Processing Time (seconds)", fontsize=13, fontweight="bold", color="#D81159"
    )
    ax2.tick_params(axis="y", labelcolor="#D81159", labelsize=11)

    # Set title
    fig.suptitle(
        "Retinex Sigma Parameter Analysis: Success Rate vs Processing Time",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        edgecolor="gray",
    )

    plt.tight_layout()

    if save:
        save_path = IMAGES_DIR / "experiment" / "display_data" / "sigma_comparison.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Sigma comparison plot saved to: {save_path}")
    else:
        plt.show()


def visualize_log_scale_comparison(
    filename: str = "comparison_log_or_exp.json", save: bool = False
) -> None:
    """Visualize comparison between log scale and linear scale Retinex.

    Args:
        filename: JSON file containing comparison results
        save: If True, save plot to PDF. If False, display interactively.
    """
    from pathlib import Path

    # Load results
    load_path = Path(filename)
    if not load_path.is_absolute():
        load_path = DATA_DIR / filename

    with open(load_path, encoding="utf-8") as f:
        results = json.load(f)

    # Organize data by dataset and log_scale parameter
    data = {}
    for r in results:
        dataset = r["dataset"]
        use_log = r["algorithm_params"].get("use_log_scale", False)
        scale_type = "Log Scale" if use_log else "Linear Scale"

        if dataset not in data:
            data[dataset] = {}
        data[dataset][scale_type] = {
            "success_rate": r["success_rate"],
            "processing_time": r["processing_time"],
        }

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    datasets = list(data.keys())
    scale_types = ["Linear Scale", "Log Scale"]

    # Success rate comparison
    x = range(len(datasets))
    width = 0.35

    linear_rates = [
        data[ds].get("Linear Scale", {}).get("success_rate", 0) for ds in datasets
    ]
    log_rates = [
        data[ds].get("Log Scale", {}).get("success_rate", 0) for ds in datasets
    ]

    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        linear_rates,
        width,
        label="Linear Scale",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax1.bar(
        [i + width / 2 for i in x],
        log_rates,
        width,
        label="Log Scale",
        color="#e74c3c",
        alpha=0.8,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax1.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Detection Success Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Retinex: Log Scale vs Linear Scale", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax1.set_ylim(0, 110)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Processing time comparison
    linear_times = [
        data[ds].get("Linear Scale", {}).get("processing_time", 0) for ds in datasets
    ]
    log_times = [
        data[ds].get("Log Scale", {}).get("processing_time", 0) for ds in datasets
    ]

    bars1 = ax2.bar(
        [i - width / 2 for i in x],
        linear_times,
        width,
        label="Linear Scale",
        color="#3498db",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        log_times,
        width,
        label="Log Scale",
        color="#e74c3c",
        alpha=0.8,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax2.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Processing Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title("Processing Time Comparison", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    # Print summary
    print("\n" + "=" * 60)
    print("Retinex Algorithm: Log Scale vs Linear Scale Comparison")
    print("=" * 60)
    for dataset in datasets:
        print(f"\n{dataset.replace('_', ' ').title()}:")
        for scale_type in scale_types:
            if scale_type in data[dataset]:
                info = data[dataset][scale_type]
                print(f"  {scale_type}:")
                print(f"    Success Rate: {info['success_rate']:.2f}%")
                print(f"    Processing Time: {info['processing_time']:.3f}s")
    print("=" * 60 + "\n")

    if save:
        save_path = (
            IMAGES_DIR
            / "experiment"
            / "display_data"
            / "retinex_log_vs_linear_comparison.pdf"
        )
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")
    else:
        plt.show()


def visualize_prefilter_comparison(
    filename: str = "prefilter_comparison.json",
    save: bool = False,
) -> None:
    """Visualize comparison of different prefilters.

    Args:
        filename: JSON file containing comparison results
        save: If True, save plot to PDF. If False, display interactively.
    """
    from pathlib import Path

    # Load results
    load_path = Path(filename)
    if not load_path.is_absolute():
        load_path = DATA_DIR / "experiment" / filename

    with open(load_path, encoding="utf-8") as f:
        results = json.load(f)

    # Organize data by prefilter and dataset
    data = {}
    for r in results:
        prefilter = r["algorithm_params"].get("prefilter", None)
        prefilter_label = prefilter if prefilter else "None"
        dataset = r["dataset"]

        if prefilter_label not in data:
            data[prefilter_label] = {}

        data[prefilter_label][dataset] = {
            "success_rate": r["success_rate"],
            "processing_time": r["processing_time"],
        }

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    prefilters = list(data.keys())
    datasets = ["low_light", "non_uniform"]

    x = range(len(datasets))
    width = 0.25  # Adjusted for 3 bars

    # Color scheme for prefilters
    prefilter_colors = {
        "None": "#2E86AB",  # Deep blue
        "gaussian": "#06A77D",  # Teal
        "median": "#A23B72",  # Purple
    }

    # Success rate comparison
    for idx, prefilter in enumerate(prefilters):
        rates = [data[prefilter].get(ds, {}).get("success_rate", 0) for ds in datasets]
        offset = (idx - len(prefilters) / 2 + 0.5) * width

        # Get color based on prefilter name
        color = prefilter_colors.get(
            prefilter, prefilter_colors.get(prefilter.lower(), "#6A4C93")
        )

        bars = ax1.bar(
            [i + offset for i in x],
            rates,
            width,
            label=prefilter.title() if prefilter != "None" else "No Prefilter",
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax1.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Detection Success Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title("Prefilter Comparison: Success Rate", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax1.set_ylim(
        0,
        max(
            [
                data[pf].get(ds, {}).get("success_rate", 0)
                for pf in prefilters
                for ds in datasets
            ]
        )
        * 1.15,
    )
    ax1.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Processing time comparison
    for idx, prefilter in enumerate(prefilters):
        times = [
            data[prefilter].get(ds, {}).get("processing_time", 0) for ds in datasets
        ]
        offset = (idx - len(prefilters) / 2 + 0.5) * width

        # Get color based on prefilter name
        color = prefilter_colors.get(
            prefilter, prefilter_colors.get(prefilter.lower(), "#6A4C93")
        )

        bars = ax2.bar(
            [i + offset for i in x],
            times,
            width,
            label=prefilter.title() if prefilter != "None" else "No Prefilter",
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.2,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax2.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Processing Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Prefilter Comparison: Processing Time", fontsize=14, fontweight="bold"
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax2.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add overall title
    fig.suptitle(
        "Retinex Prefilter Comparison: None vs Gaussian vs Median",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()

    if save:
        save_path = (
            IMAGES_DIR
            / "experiment"
            / "display_data"
            / filename.replace(".json", ".pdf")
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plot saved to: {save_path}")
    else:
        plt.show()


def visualize_smart_normalize_comparison(
    filename: str = "smart_normalize_comparison.json",
    save: bool = False,
) -> None:
    """Visualize comparison between cv2.normalize and gain_compensation methods.

    Args:
        filename: JSON file containing comparison results
        save: If True, save plot to PDF. If False, display interactively.
    """

    # Load results
    load_path = Path(filename)
    if not load_path.is_absolute():
        load_path = DATA_DIR / "experiment" / filename

    with open(load_path, encoding="utf-8") as f:
        results = json.load(f)

    # Organize data by method and dataset
    data = {}
    for r in results:
        dataset = r["dataset"]
        use_smart = r["algorithm_params"].get("smart_normalize", False)
        method = "gain_compensation" if use_smart else "cv2.normalize"

        if dataset not in data:
            data[dataset] = {}
        data[dataset][method] = {
            "success_rate": r["success_rate"],
            "processing_time": r["processing_time"],
        }

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    datasets = list(data.keys())

    # Success rate comparison
    x = range(len(datasets))
    width = 0.35

    cv2_rates = [
        data[ds].get("cv2.normalize", {}).get("success_rate", 0) for ds in datasets
    ]
    smart_rates = [
        data[ds].get("gain_compensation", {}).get("success_rate", 0) for ds in datasets
    ]

    # Use cool colors (blue/green) for success rate
    bars1 = ax1.bar(
        [i - width / 2 for i in x],
        cv2_rates,
        width,
        label="cv2.normalize",
        color="#2E86AB",  # Deep blue
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = ax1.bar(
        [i + width / 2 for i in x],
        smart_rates,
        width,
        label="gain_compensation",
        color="#06A77D",  # Teal green
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax1.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Detection Success Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Normalization Method Comparison: Success Rate",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax1.set_ylim(0, 110)
    ax1.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Processing time comparison
    cv2_times = [
        data[ds].get("cv2.normalize", {}).get("processing_time", 0) for ds in datasets
    ]
    smart_times = [
        data[ds].get("gain_compensation", {}).get("processing_time", 0)
        for ds in datasets
    ]

    # Use warm colors (red/purple) for processing time
    bars1 = ax2.bar(
        [i - width / 2 for i in x],
        cv2_times,
        width,
        label="cv2.normalize",
        color="#D81159",  # Crimson red
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        smart_times,
        width,
        label="gain_compensation",
        color="#8F2D56",  # Deep purple
        alpha=0.85,
        edgecolor="white",
        linewidth=1.5,
    )

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax2.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Processing Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Normalization Method Comparison: Processing Time",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax2.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add overall title
    fig.suptitle(
        "Retinex Smart Normalize Comparison: cv2.normalize vs gain_compensation",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()

    if save:
        save_path = (
            IMAGES_DIR
            / "experiment"
            / "display_data"
            / "smart_normalize_comparison.pdf"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plot saved to: {save_path}")
    else:
        plt.show()


def visualize_postfilter_comparison(
    filename: str = "postfilter_comparison.json",
    save: bool = False,
) -> None:
    """Visualize comparison between different postfilter options.

    Args:
        filename: JSON file containing comparison results
        save: If True, save plot to PDF. If False, display interactively.
    """

    # Load results
    load_path = Path(filename)
    if not load_path.is_absolute():
        load_path = DATA_DIR / "experiment" / filename

    with open(load_path, encoding="utf-8") as f:
        results = json.load(f)

    # Organize data by dataset and postfilter
    data = {}
    for r in results:
        dataset = r["dataset"]
        postfilter = r["algorithm_params"].get("postfilter", None)
        postfilter_label = postfilter if postfilter else "None"

        if dataset not in data:
            data[dataset] = {}
        data[dataset][postfilter_label] = {
            "success_rate": r["success_rate"],
            "processing_time": r["processing_time"],
        }

    # Plot comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    datasets = list(data.keys())
    postfilters = ["None", "gaussian", "median"]
    postfilter_labels = ["No Filter", "Gaussian", "Median"]

    # Success rate comparison
    x = range(len(datasets))
    width = 0.25

    # Color scheme: blue gradient for different filters
    colors = {
        "None": "#2E86AB",  # Deep blue
        "gaussian": "#06A77D",  # Teal
        "median": "#A23B72",  # Purple
    }

    bars_list = []
    for i, (postfilter, label) in enumerate(zip(postfilters, postfilter_labels)):
        rates = [data[ds].get(postfilter, {}).get("success_rate", 0) for ds in datasets]

        bars = ax1.bar(
            [pos + (i - 1) * width for pos in x],
            rates,
            width,
            label=label,
            color=colors[postfilter],
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )
        bars_list.append(bars)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax1.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Detection Success Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Postfilter Comparison: Success Rate",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax1.set_ylim(
        0,
        max(
            [
                max(data[ds].get(pf, {}).get("success_rate", 0) for pf in postfilters)
                for ds in datasets
            ]
        )
        * 1.15,
    )
    ax1.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Processing time comparison
    bars_list = []
    for i, (postfilter, label) in enumerate(zip(postfilters, postfilter_labels)):
        times = [
            data[ds].get(postfilter, {}).get("processing_time", 0) for ds in datasets
        ]

        bars = ax2.bar(
            [pos + (i - 1) * width for pos in x],
            times,
            width,
            label=label,
            color=colors[postfilter],
            alpha=0.85,
            edgecolor="white",
            linewidth=1.5,
        )
        bars_list.append(bars)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}s",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax2.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Processing Time (seconds)", fontsize=12, fontweight="bold")
    ax2.set_title(
        "Postfilter Comparison: Processing Time",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels([ds.replace("_", " ").title() for ds in datasets])
    ax2.legend(framealpha=0.9, edgecolor="gray", fontsize=10)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    # Add overall title
    fig.suptitle(
        "Retinex Postfilter Comparison: None vs Gaussian vs Median",
        fontsize=15,
        fontweight="bold",
        y=1.00,
    )

    plt.tight_layout()

    if save:
        save_path = (
            IMAGES_DIR / "experiment" / "display_data" / "postfilter_comparison.pdf"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plot saved to: {save_path}")
    else:
        plt.show()


def visualize_retinex_parameter_space(
    filenames: dict[str, str] | None = None,
    save: bool = False,
) -> None:
    """Visualize Retinex parameter space: log/linear × prefilter × normalize.

    Shows comprehensive comparison of 8 parameter combinations:
    - use_log_scale: Log vs Linear
    - prefilter: Gaussian vs None
    - smart_normalize: gain_compensation vs cv2.normalize

    Args:
        filenames: Dict mapping parameter combinations to JSON files:
            - "linear_no_filter": smart_normalize_comparison.json
            - "linear_with_filter": smart_normalize_linear_prefilter.json
            - "log_no_filter": smart_normalize_log_scale_comparison.json
            - "log_with_filter": smart_normalize_log_prefilter.json
        save: If True, save plot to PDF. If False, display interactively.
    """
    from pathlib import Path

    # Default filenames
    if filenames is None:
        filenames = {
            "linear_no_filter": "smart_normalize_comparison.json",
            "linear_with_filter": "smart_normalize_linear_prefilter.json",
            "log_no_filter": "smart_normalize_log_scale_comparison.json",
            "log_with_filter": "smart_normalize_log_prefilter.json",
        }

    # Load all data
    all_data = {}
    for key, filename in filenames.items():
        load_path = Path(filename)
        if not load_path.is_absolute():
            load_path = DATA_DIR / "experiment" / filename

        with open(load_path, encoding="utf-8") as f:
            all_data[key] = json.load(f)

    # Organize data: data[dataset][scale][prefilter][normalize] = {rate, time}
    organized = {}

    for config_key, results in all_data.items():
        # Parse config key
        if "linear" in config_key:
            scale = "Linear"
        else:
            scale = "Log"

        has_filter = "with_filter" in config_key

        for result in results:
            dataset = result["dataset"]
            use_smart = result["algorithm_params"].get("smart_normalize", False)
            normalize = "gain_compensation" if use_smart else "cv2.normalize"
            prefilter = "Gaussian" if has_filter else "None"

            if dataset not in organized:
                organized[dataset] = {}
            if scale not in organized[dataset]:
                organized[dataset][scale] = {}
            if prefilter not in organized[dataset][scale]:
                organized[dataset][scale][prefilter] = {}

            organized[dataset][scale][prefilter][normalize] = {
                "success_rate": result["success_rate"],
                "processing_time": result["processing_time"],
            }

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        "Retinex Parameter Space Analysis: Log/Linear × Prefilter × Normalization",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    datasets = ["low_light", "non_uniform"]
    dataset_labels = ["Low Light", "Non-Uniform"]
    scales = ["Linear", "Log"]

    # Color scheme: prefilter (shape) × normalize (color)
    colors = {
        ("None", "cv2.normalize"): "#2E86AB",  # Deep blue
        ("None", "gain_compensation"): "#06A77D",  # Teal
        ("Gaussian", "cv2.normalize"): "#D81159",  # Red
        ("Gaussian", "gain_compensation"): "#8F2D56",  # Purple
    }

    patterns = {
        "None": "",  # Solid
        "Gaussian": "///",  # Diagonal lines
    }

    # Plot success rates (top row)
    for col, dataset in enumerate(datasets):
        ax = axes[0, col]
        x_positions = range(len(scales))
        width = 0.18
        offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

        # Plot 4 bars for each scale
        for i, (prefilter, normalize) in enumerate(
            [
                ("None", "cv2.normalize"),
                ("None", "gain_compensation"),
                ("Gaussian", "cv2.normalize"),
                ("Gaussian", "gain_compensation"),
            ]
        ):
            rates = []
            for scale in scales:
                try:
                    rate = organized[dataset][scale][prefilter][normalize][
                        "success_rate"
                    ]
                except KeyError:
                    rate = 0
                rates.append(rate)

            # Create descriptive label
            if prefilter == "None":
                prefilter_label = "No Prefilter"
            else:
                prefilter_label = f"{prefilter} Prefilter"

            if normalize == "cv2.normalize":
                normalize_label = "cv2.normalize"
            else:
                normalize_label = "Smart Normalize"

            label = f"{prefilter_label} + {normalize_label}"

            bars = ax.bar(
                [x + offsets[i] for x in x_positions],
                rates,
                width,
                label=label,
                color=colors[(prefilter, normalize)],
                alpha=0.85,
                edgecolor="white",
                linewidth=1.5,
                hatch=patterns[prefilter],
            )

            # Add value labels
            for bar, rate in zip(bars, rates):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1,
                    f"{rate:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        ax.set_xlabel("Retinex Scale", fontsize=11, fontweight="bold")
        ax.set_ylabel("Success Rate (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"{dataset_labels[col]} Dataset", fontsize=12, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(scales)
        ax.set_ylim(0, 85)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # Add legend to the first subplot (Low Light) where there's plenty of space
        if col == 0:
            ax.legend(
                fontsize=9,
                frameon=True,
                framealpha=0.95,
                edgecolor="gray",
                loc="upper left",
                title="Parameter Combinations",
                title_fontsize=10,
            )

    # Plot processing times (bottom row)
    for col, dataset in enumerate(datasets):
        ax = axes[1, col]
        x_positions = range(len(scales))

        for i, (prefilter, normalize) in enumerate(
            [
                ("None", "cv2.normalize"),
                ("None", "gain_compensation"),
                ("Gaussian", "cv2.normalize"),
                ("Gaussian", "gain_compensation"),
            ]
        ):
            times = []
            for scale in scales:
                try:
                    time = organized[dataset][scale][prefilter][normalize][
                        "processing_time"
                    ]
                except KeyError:
                    time = 0
                times.append(time)

            # Create descriptive label
            if prefilter == "None":
                prefilter_label = "No Prefilter"
            else:
                prefilter_label = f"{prefilter} Prefilter"

            if normalize == "cv2.normalize":
                normalize_label = "cv2.normalize"
            else:
                normalize_label = "Smart Normalize"

            label = f"{prefilter_label} + {normalize_label}"

            bars = ax.bar(
                [x + offsets[i] for x in x_positions],
                times,
                width,
                label=label,
                color=colors[(prefilter, normalize)],
                alpha=0.85,
                edgecolor="white",
                linewidth=1.5,
                hatch=patterns[prefilter],
            )

            # Add value labels
            for bar, time in zip(bars, times):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.2,
                    f"{time:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Retinex Scale", fontsize=11, fontweight="bold")
        ax.set_ylabel("Processing Time (s)", fontsize=11, fontweight="bold")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(scales)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()

    if save:
        save_path = (
            IMAGES_DIR / "experiment" / "display_data" / "retinex_parameter_space.pdf"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Parameter space plot saved to: {save_path}")
    else:
        plt.show()
