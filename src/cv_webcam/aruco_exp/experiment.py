import time
from pathlib import Path
from typing import Callable, Literal, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike

from cv_webcam import IMAGES_DIR
from cv_webcam.core import ArucoDetector, create_aruco_detector, img_prep


def draw_low_light_imgs(save: bool = False) -> None:
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


def count_detected_markers(
    detector: ArucoDetector,
    img_names: Sequence[Path],
    prep_fn: Callable[[MatLike], MatLike] | None = None,
    use_prev_filter: bool = True,
    use_post_filter: bool = True,
    print_msgs: bool = True,
) -> tuple[int, float]:
    """Count detected markers and measure processing time.

    Returns:
        tuple: (success_count, elapsed_time_in_seconds)
    """
    if len(img_names) == 0:
        raise FileNotFoundError("No images found")

    start_time = time.perf_counter()
    success_cnt = 0

    for img_name in img_names:
        img = cv2.imread(str(img_name), cv2.IMREAD_GRAYSCALE)
        assert img is not None

        if use_prev_filter:
            img = cv2.GaussianBlur(img, (3, 3), 0)

        if prep_fn is not None:
            img = prep_fn(img)

        if use_post_filter:
            img = cv2.medianBlur(img, 5)

        if not detector.can_be_detected(img, 0):
            if print_msgs:
                print(f"No marker detected in image: {img_name.name}")
        else:
            success_cnt += 1

    elapsed_time = time.perf_counter() - start_time
    return success_cnt, elapsed_time


def test_raw_dataset() -> None:
    detector = create_aruco_detector(marker_length=41)
    img_names = [
        img_name
        for img_name in (IMAGES_DIR / "experiment" / "raw").rglob("*.png")
        if img_name.is_file()
    ]
    success_cnt, elapsed_time = count_detected_markers(
        detector, img_names, prep_fn=None, use_post_filter=False
    )
    total_cnt = len(img_names)
    print(
        f"Detection success rate on raw images: {success_cnt}/{total_cnt} ({success_cnt / total_cnt * 100:.2f}%)"
    )
    print(f"Processing time: {elapsed_time:.3f}s")


def run_detector_evaluation(
    dataset: Literal["all", "low_light", "non_uniform"] = "all",
    preprocessing: Literal["none", "clahe", "retinex"] = "none",
    use_prev_filter: bool = True,
    use_post_filter: bool = True,
    print_msgs: bool = True,
) -> dict[str, int | float]:
    """Run detector evaluation and measure performance.

    Returns:
        Dictionary with success counts, totals, and elapsed times for each dataset.
    """
    match preprocessing:
        case "none":
            prep_fn = None
        case "clahe":
            prep_fn = cv2.createCLAHE(clipLimit=2).apply
        case "retinex":
            prep_fn = img_prep.retinex_test

    detector = create_aruco_detector(marker_length=41)

    result_dict: dict[str, int | float] = {
        "success_low_light": 0,
        "total_low_light": 0,
        "time_low_light": 0.0,
        "success_non_uniform": 0,
        "total_non_uniform": 0,
        "time_non_uniform": 0.0,
    }

    if dataset == "all" or dataset == "low_light":
        img_names = [
            img_name
            for img_name in (IMAGES_DIR / "experiment" / "low_light").rglob("*.png")
            if img_name.is_file()
        ]
        success_cnt, elapsed_time = count_detected_markers(
            detector,
            img_names,
            prep_fn=prep_fn,
            use_prev_filter=use_prev_filter,
            use_post_filter=use_post_filter,
            print_msgs=print_msgs,
        )
        total_cnt = len(img_names)
        print(
            f"Detection success rate on low light images: {success_cnt}/{total_cnt} ({success_cnt / total_cnt * 100:.2f}%)"
        )
        result_dict["success_low_light"] = success_cnt
        result_dict["total_low_light"] = total_cnt
        result_dict["time_low_light"] = elapsed_time

    if dataset == "all" or dataset == "non_uniform":
        img_names = [
            img_name
            for img_name in (IMAGES_DIR / "experiment" / "non_uniform").rglob("*.png")
            if img_name.is_file()
        ]
        success_cnt, elapsed_time = count_detected_markers(
            detector,
            img_names,
            prep_fn=prep_fn,
            use_prev_filter=use_prev_filter,
            use_post_filter=use_post_filter,
            print_msgs=print_msgs,
        )
        total_cnt = len(img_names)
        print(
            f"Detection success rate on non-uniform images: {success_cnt}/{total_cnt} ({success_cnt / total_cnt * 100:.2f}%)"
        )
        result_dict["success_non_uniform"] = success_cnt
        result_dict["total_non_uniform"] = total_cnt
        result_dict["time_non_uniform"] = elapsed_time

    return result_dict


def display_evaluation_results() -> dict[str, dict[str, dict[str, float]]]:
    """Collect evaluation results for all filter configurations and preprocessing methods.

    Returns:
        Dictionary with structure: {filter_config: {preprocessing: {dataset: success_rate}}}
    """
    results = {}

    filter_configs = [
        ("No Filter", False, False),
        ("Gaussian Pre-filter", True, False),
        ("Median Post-filter", False, True),
        ("Both Filters", True, True),
    ]

    preprocessing_methods: list[Literal["none", "clahe", "retinex"]] = [
        "none",
        "clahe",
        "retinex",
    ]

    for filter_name, use_prev, use_post in filter_configs:
        results[filter_name] = {}
        print(f"{filter_name}:")

        for prep in preprocessing_methods:
            prep_display = prep.upper() if prep != "none" else "None"
            print(f"  Evaluation with {prep_display} preprocessing:")

            result_dict = run_detector_evaluation(
                dataset="all",
                preprocessing=prep,
                use_prev_filter=use_prev,
                use_post_filter=use_post,
                print_msgs=False,
            )

            # Calculate success rates
            low_light_rate = (
                result_dict["success_low_light"] / result_dict["total_low_light"] * 100
                if result_dict["total_low_light"] > 0
                else 0
            )
            non_uniform_rate = (
                result_dict["success_non_uniform"]
                / result_dict["total_non_uniform"]
                * 100
                if result_dict["total_non_uniform"] > 0
                else 0
            )

            results[filter_name][prep] = {
                "low_light": low_light_rate,
                "non_uniform": non_uniform_rate,
                "time_low_light": float(result_dict["time_low_light"]),
                "time_non_uniform": float(result_dict["time_non_uniform"]),
                "time_total": float(result_dict["time_low_light"])
                + float(result_dict["time_non_uniform"]),
            }

            print(f"    Time: {results[filter_name][prep]['time_total']:.3f}s")

        print()

    return results


def plot_evaluation_results(save: bool = False) -> None:
    """Plot evaluation results as grouped bar charts.

    Creates 4 subplots in 2x2 layout, one for each filter configuration, showing
    detection success rates for different preprocessing methods.

    Args:
        save: If True, save to PDF file. If False, display interactively.
    """

    # Collect data
    results = display_evaluation_results()

    # Prepare figure with 2x2 layout
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10),
    )
    # Use tight_layout with rect to reserve space for suptitle
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.5)

    filter_configs = [
        "No Filter",
        "Gaussian Pre-filter",
        "Median Post-filter",
        "Both Filters",
    ]
    preprocessing_methods: list[Literal["none", "clahe", "retinex"]] = [
        "none",
        "clahe",
        "retinex",
    ]
    preprocessing_labels = ["None", "CLAHE", "Retinex"]

    # Bar settings
    x = np.arange(len(preprocessing_methods))
    width = 0.35

    # Colors
    colors = {"low_light": "#3498db", "non_uniform": "#e74c3c"}

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    for idx, (ax, filter_name) in enumerate(zip(axes_flat, filter_configs)):
        low_light_rates = [
            results[filter_name][prep]["low_light"] for prep in preprocessing_methods
        ]
        non_uniform_rates = [
            results[filter_name][prep]["non_uniform"] for prep in preprocessing_methods
        ]

        # Create bars
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

        # Add value labels on bars
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

        # Customize subplot
        ax.set_xlabel("Preprocessing Method", fontsize=11, fontweight="bold")
        ax.set_ylabel("Detection Success Rate (%)", fontsize=11, fontweight="bold")
        ax.set_title(filter_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(preprocessing_labels)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=10)

    # Add overall title
    fig.suptitle(
        "ArUco Detection Success Rate: Filter Configurations and Preprocessing Methods",
        fontsize=14,
        fontweight="bold",
    )

    if save:
        save_path = (
            IMAGES_DIR / "experiment" / "display_data" / "evaluation_results.pdf"
        )
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
            format="pdf",
        )
        print(f"\n✓ Plot saved to: {save_path}")
    else:
        plt.show()


def plot_performance_comparison(save: bool = False) -> None:
    """Plot processing time comparison for different methods.

    Creates 4 subplots in 2x2 layout for each filter configuration.

    Args:
        save: If True, save to PDF file. If False, display interactively.
    """
    # Collect data
    results = display_evaluation_results()

    # Prepare figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=3.0, w_pad=2.5)

    filter_configs = [
        "No Filter",
        "Gaussian Pre-filter",
        "Median Post-filter",
        "Both Filters",
    ]
    preprocessing_methods: list[Literal["none", "clahe", "retinex"]] = [
        "none",
        "clahe",
        "retinex",
    ]
    preprocessing_labels = ["None", "CLAHE", "Retinex"]

    # Bar settings
    x = np.arange(len(preprocessing_methods))
    width = 0.28

    # Colors for different datasets
    colors = {"low_light": "#9b59b6", "non_uniform": "#f39c12", "total": "#2ecc71"}

    # Flatten axes array for easier iteration
    axes_flat = axes.flatten()

    for idx, (ax, filter_name) in enumerate(zip(axes_flat, filter_configs)):
        time_low_light = [
            results[filter_name][prep]["time_low_light"]
            for prep in preprocessing_methods
        ]
        time_non_uniform = [
            results[filter_name][prep]["time_non_uniform"]
            for prep in preprocessing_methods
        ]
        time_total = [
            results[filter_name][prep]["time_total"] for prep in preprocessing_methods
        ]

        # Create bars
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

        # Add value labels on bars
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

        # Customize subplot
        ax.set_xlabel("Preprocessing Method", fontsize=11, fontweight="bold")
        ax.set_ylabel("Processing Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_title(filter_name, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(preprocessing_labels)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(loc="upper left", fontsize=9)

    # Add overall title
    fig.suptitle(
        "ArUco Detection Processing Time: Filter Configurations and Preprocessing Methods",
        fontsize=14,
        fontweight="bold",
    )

    if save:
        save_path = (
            IMAGES_DIR / "experiment" / "display_data" / "performance_comparison.pdf"
        )
        plt.savefig(
            save_path,
            bbox_inches="tight",
            dpi=300,
            format="pdf",
        )
        print(f"\n✓ Performance plot saved to: {save_path}")
    else:
        plt.show()


def visulize_test() -> None:
    img = cv2.imread(
        # str(IMAGES_DIR / "experiment" / "non_uniform" / "img_0_sigma300.png"),
        str(IMAGES_DIR / "experiment" / "low_light" / "img_0_dark_lv0.png"),
        cv2.IMREAD_GRAYSCALE,
    )
    assert img is not None

    retinex_img = img_prep.retinex_test(img)
    retinex_img = cv2.medianBlur(retinex_img, 5)
    cv2.imshow("Original Image", img)
    cv2.imshow("Retinex Image", retinex_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_experiment() -> None:
    # draw_low_light_imgs(save=True)
    # draw_non_uniform_imgs(save=True)
    # test_raw_dataset()
    # run_detector_evaluation(dataset="all", preprocessing="retinex")
    plot_evaluation_results(save=True)
    plot_performance_comparison(save=True)
    # visulize_test()
