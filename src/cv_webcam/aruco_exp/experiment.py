"""Experiment execution and evaluation for ArUco detection."""

import json
import time
from dataclasses import dataclass, field
from typing import Literal

import cv2

from cv_webcam import DATA_DIR, IMAGES_DIR
from cv_webcam.core import create_aruco_detector

from . import algorithms, visualizer


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    dataset: Literal["low_light", "non_uniform"] = "low_light"
    algorithm: str = "none"
    algorithm_params: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig
    success_count: int
    total_count: int
    success_rate: float
    processing_time: float
    failed_images: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": self.config.dataset,
            "algorithm": self.config.algorithm,
            "algorithm_params": self.config.algorithm_params,
            "success_count": self.success_count,
            "total_count": self.total_count,
            "success_rate": self.success_rate,
            "processing_time": self.processing_time,
            "failed_images": self.failed_images,
        }


def run_single_experiment(
    config: ExperimentConfig, print_failures: bool = False
) -> ExperimentResult:
    """Run a single experiment with specified configuration.

    Args:
        config: Experiment configuration
        print_failures: If True, print names of images where detection failed

    Returns:
        Experiment result with metrics
    """
    detector = create_aruco_detector(marker_length=40)

    prep_algo = algorithms.get_algorithm(
        config.algorithm,
        **config.algorithm_params,
    )

    dataset_path = IMAGES_DIR / "experiment" / config.dataset
    img_paths = sorted(dataset_path.glob("*.png"))

    if not img_paths:
        raise FileNotFoundError(f"No images found in {dataset_path}")

    start_time = time.perf_counter()
    success_count = 0
    failed_images = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        assert img is not None, f"Failed to load image: {img_path}"

        processed_img = prep_algo(img)

        if detector.can_be_detected(processed_img, 0):
            success_count += 1
        else:
            failed_images.append(img_path.name)
            if print_failures:
                print(f"  Failed: {img_path.name}")

    elapsed = time.perf_counter() - start_time

    return ExperimentResult(
        config=config,
        success_count=success_count,
        total_count=len(img_paths),
        success_rate=success_count / len(img_paths) * 100,
        processing_time=elapsed,
        failed_images=failed_images,
    )


def run_batch_experiments(
    configs: list[ExperimentConfig], print_failures: bool = False
) -> list[ExperimentResult]:
    """Run multiple experiments in sequence.

    Args:
        configs: List of experiment configurations
        print_failures: If True, print failed image names

    Returns:
        List of experiment results
    """
    results = []
    for i, cfg in enumerate(configs, 1):
        params_str = f" {cfg.algorithm_params}" if cfg.algorithm_params else ""
        print(
            f"[{i}/{len(configs)}] Running: {cfg.algorithm} on {cfg.dataset}{params_str}"
        )
        result = run_single_experiment(cfg, print_failures=print_failures)
        results.append(result)
        print(
            f"  Success: {result.success_rate:.1f}% ({result.success_count}/{result.total_count}), "
            f"Time: {result.processing_time:.2f}s"
        )

    return results


def save_results(
    results: list[ExperimentResult], filename: str = "results.json"
) -> None:
    """Save experiment results to JSON file.

    Args:
        results: List of experiment results
        filename: Output filename (saved in experiment directory)
    """
    output = [r.to_dict() for r in results]
    save_path = DATA_DIR / "experiment" / filename
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to: {save_path}")


def load_results(filename: str = "results.json") -> list[dict]:
    """Load experiment results from JSON file.

    Args:
        filename: Input filename (loaded from experiment directory)

    Returns:
        List of result dictionaries
    """
    load_path = DATA_DIR / "experiment" / filename
    with open(load_path) as f:
        return json.load(f)


def test_raw_dataset() -> None:
    """Test detection on raw images without preprocessing."""
    detector = create_aruco_detector(marker_length=40)
    img_names = [
        img_name
        for img_name in (IMAGES_DIR / "experiment" / "raw").rglob("*.png")
        if img_name.is_file()
    ]

    start_time = time.perf_counter()
    success_cnt = 0

    for img_name in img_names:
        img = cv2.imread(str(img_name), cv2.IMREAD_GRAYSCALE)
        assert img is not None

        if detector.can_be_detected(img, 0):
            success_cnt += 1

    elapsed_time = time.perf_counter() - start_time
    total_cnt = len(img_names)

    print(
        f"Detection success rate on raw images: {success_cnt}/{total_cnt} "
        f"({success_cnt / total_cnt * 100:.2f}%)"
    )
    print(f"Processing time: {elapsed_time:.3f}s")


def display_evaluation_results() -> dict[str, dict[str, dict[str, float]]]:
    """Run comprehensive evaluation with all filter configurations and preprocessing methods.

    Returns:
        Nested dict: {filter_config: {preprocessing: {dataset: success_rate}}}
    """
    results = {}

    filter_configs = [
        ("No Filter", False, False),
        ("Gaussian Pre-filter", True, False),
        ("Median Post-filter", False, True),
        ("Both Filters", True, True),
    ]

    preprocessing_methods = ["none", "clahe", "retinex"]

    for filter_name, use_pre, use_post in filter_configs:
        results[filter_name] = {}
        print(f"\n{filter_name}:")

        for prep in preprocessing_methods:
            prep_display = prep.upper() if prep != "none" else "None"
            print(f"  {prep_display} preprocessing:")

            configs = [
                ExperimentConfig(
                    dataset="low_light",
                    algorithm=prep,
                    algorithm_params={
                        "use_gaussian_prefilter": use_pre,
                        "use_median_postfilter": use_post,
                    },
                ),
                ExperimentConfig(
                    dataset="non_uniform",
                    algorithm=prep,
                    algorithm_params={
                        "use_gaussian_prefilter": use_pre,
                        "use_median_postfilter": use_post,
                    },
                ),
            ]

            batch_results = run_batch_experiments(configs, print_failures=False)

            low_light_result = batch_results[0]
            non_uniform_result = batch_results[1]

            results[filter_name][prep] = {
                "low_light": low_light_result.success_rate,
                "non_uniform": non_uniform_result.success_rate,
                "time_low_light": low_light_result.processing_time,
                "time_non_uniform": non_uniform_result.processing_time,
                "time_total": low_light_result.processing_time
                + non_uniform_result.processing_time,
            }

            print(f"    Time: {results[filter_name][prep]['time_total']:.3f}s")

    return results


def plot_results(save: bool = False) -> None:
    """Run evaluation and plot results."""
    results = display_evaluation_results()
    visualizer.plot_evaluation_results(results, save=save)
    visualizer.plot_performance_comparison(results, save=save)


def visulize_test() -> None:
    """Visualize algorithm comparison on a single image."""
    img_path = IMAGES_DIR / "experiment" / "low_light" / "img_0_dark_lv3.png"
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None

    retinex_algo_log = algorithms.get_algorithm(
        "retinex",
        use_gaussian_prefilter=True,
        use_median_postfilter=True,
        use_log_scale=True,
    )
    retinex_img_log = retinex_algo_log(img)

    retinex_algo_linear = algorithms.get_algorithm(
        "retinex",
        use_gaussian_prefilter=True,
        use_median_postfilter=True,
        use_log_scale=False,
    )
    retinex_img_linear = retinex_algo_linear(img)

    cv2.imshow("Original Image", img)
    cv2.imshow("Retinex Image (Log Scale)", retinex_img_log)
    cv2.imshow("Retinex Image (Linear Scale)", retinex_img_linear)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def visualize_log_scale_comparison(
    filename: str = "comparison_log_or_exp.json", save: bool = False
) -> None:
    """Visualize comparison between log scale and linear scale Retinex.

    Args:
        filename: JSON file containing comparison results
    """
    import matplotlib.pyplot as plt

    results = load_results(filename)

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


def run_experiment() -> None:
    """Main experiment runner with various demo options."""
    # visualizer.draw_low_light_imgs(save=True)
    # visualizer.draw_non_uniform_imgs(save=True)
    # test_raw_dataset()

    # Comprehensive evaluation with plots
    # plot_results(save=False)

    # visulize_test()

    # Visualize log scale comparison from saved results
    visualize_log_scale_comparison("comparison_log_or_exp.json", save=True)
