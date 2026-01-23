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

    if False:
        retinex_1 = algorithms.get_algorithm(
            "retinex",
            sigma=80,
            prefilter="gaussian",
            prefilter_params={"ksize": (5, 5)},
        )
        retinex_img_1 = retinex_1(img)

        retinex_2 = algorithms.get_algorithm(
            "retinex",
            sigma=80,
            prefilter="gaussian",
            prefilter_params={"ksize": (5, 5)},
            postfilter="bilateral",
            postfilter_params={"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
        )
        retinex_img_2 = retinex_2(img)

        retinex_img_3 = cv2.Canny(img, 150, 200)

        # cv2.imshow("Original Image", img)
        cv2.imshow("Retinex Image 1", retinex_img_1)
        cv2.imshow("Retinex Image 2", retinex_img_2)
        cv2.imshow("Retinex Image 3", retinex_img_3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if True:
        from .analysis import show_retinex_distribution

        show_retinex_distribution(img)


def compare_prefilters(
    prefilters: dict[str, dict],
    algorithm: str = "retinex",
    postfilter: str = "median",
    algorithm_params: dict | None = None,
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    filename: str = "prefilter_comparison.json",
) -> list[ExperimentResult]:
    """Compare different prefilter types for an algorithm.

    Args:
        prefilters: Dict of {filter_name: filter_params}
            e.g., {"Gaussian": {"filter": "gaussian"}, "Bilateral": {"filter": "bilateral", "params": {...}}}
        algorithm: Algorithm to test (default: "retinex")
        postfilter: Postfilter to use for all tests (default: "median")
        algorithm_params: Additional algorithm parameters (e.g., sigma, use_log_scale)
        datasets: List of datasets to test (default: ["low_light", "non_uniform"])
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    if algorithm_params is None:
        algorithm_params = {}

    configs = []

    for filter_name, filter_config in prefilters.items():
        for dataset in datasets:
            params = {
                "postfilter": postfilter,
                **algorithm_params,
            }

            # Handle prefilter configuration
            if "filter" in filter_config:
                params["prefilter"] = filter_config["filter"]

            if "params" in filter_config:
                params["prefilter_params"] = filter_config["params"]

            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    algorithm=algorithm,
                    algorithm_params=params,
                )
            )

    print("\n" + "=" * 70)
    print(f"Comparing Prefilters: {', '.join(prefilters.keys())}")
    print(f"Algorithm: {algorithm.upper()}, Postfilter: {postfilter}")
    print("=" * 70)

    results = run_batch_experiments(configs)
    save_results(results, filename)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)

    num_datasets = len(datasets)
    filter_names = list(prefilters.keys())

    for idx, filter_name in enumerate(filter_names):
        print(f"\n{filter_name} Prefilter:")
        for ds_idx, dataset in enumerate(datasets):
            result = results[idx * num_datasets + ds_idx]
            dataset_display = dataset.replace("_", " ").title()
            print(f"  {dataset_display} Dataset:")
            print(f"    Success Rate: {result.success_rate:.2f}%")
            print(f"    Processing Time: {result.processing_time:.3f}s")

    print("=" * 70 + "\n")

    return results


def compare_sigma_values(
    sigmas: list[float] | None = None,
    algorithm: str = "retinex",
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    use_filters: bool = False,
    filename: str = "sigma_comparison.json",
) -> list[ExperimentResult]:
    """Compare different sigma values for Retinex algorithm.

    Args:
        sigmas: List of sigma values to test (default: [15, 50, 80, 120, 180, 250])
        algorithm: Algorithm to test (default: "retinex")
        datasets: Datasets to test (default: ["low_light", "non_uniform"])
        use_filters: Whether to use pre/post filters (default: False)
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if sigmas is None:
        sigmas = [15, 50, 80, 120, 180, 250]

    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    configs = []

    for sigma in sigmas:
        for dataset in datasets:
            params = {
                "sigma": sigma,
                "use_log_scale": False,  # Linear scale performs better
            }

            # Add filters if requested
            if use_filters:
                params["prefilter"] = "gaussian"
                params["postfilter"] = "median"
            else:
                params["prefilter"] = None
                params["postfilter"] = None

            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    algorithm=algorithm,
                    algorithm_params=params,
                )
            )

    print("\n" + "=" * 70)
    print(f"Sigma Parameter Sweep: {sigmas}")
    print(
        f"Algorithm: {algorithm.upper()}, Filters: {'Enabled' if use_filters else 'Disabled'}"
    )
    print(f"Datasets: {', '.join(d.replace('_', ' ').title() for d in datasets)}")
    print("=" * 70)

    results = run_batch_experiments(configs)
    save_results(results, filename)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table:")
    print("=" * 70)
    print(f"{'Sigma':<10} {'Dataset':<15} {'Success Rate':<15} {'Time (s)':<10}")
    print("-" * 70)

    for result in results:
        sigma = result.config.algorithm_params["sigma"]
        dataset = result.config.dataset.replace("_", " ").title()
        print(
            f"{sigma:<10.0f} {dataset:<15} {result.success_rate:>6.2f}%         {result.processing_time:>6.3f}"
        )

    print("=" * 70)

    # Find best sigma for each dataset
    print("\nBest Sigma Values:")
    print("-" * 70)
    for dataset in datasets:
        dataset_results = [r for r in results if r.config.dataset == dataset]
        best_result = max(dataset_results, key=lambda r: r.success_rate)
        best_sigma = best_result.config.algorithm_params["sigma"]
        print(
            f"{dataset.replace('_', ' ').title():<15}: σ={best_sigma:<6.0f} "
            f"(Success: {best_result.success_rate:.2f}%, Time: {best_result.processing_time:.3f}s)"
        )

    print("=" * 70 + "\n")

    return results


def run_experiment() -> None:
    """Main experiment runner with various demo options."""
    # visualizer.draw_low_light_imgs(save=True)
    # visualizer.draw_non_uniform_imgs(save=True)
    # test_raw_dataset()

    # Comprehensive evaluation with plots
    # plot_results(save=False)

    visulize_test()
