"""Experiment execution and evaluation for ArUco detection."""

import json
import time
from dataclasses import dataclass, field
from typing import Literal
import numpy as np
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


def visualize_test() -> None:
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
    if False:
        from .analysis import show_retinex_distribution

        show_retinex_distribution(img)
    if True:
        import numpy as np

        from cv_webcam.core import img_prep

        retinex_log = img_prep.single_scale_retinex(img, sigma=80)
        normalized_retinex_log = cv2.normalize(
            retinex_log,
            None,  # type: ignore
            0,
            255,
            cv2.NORM_MINMAX,
        )  # type: ignore
        gain_retinex_log = img_prep.gain_compensation(retinex_log)

        retinex_exp = np.expm1(retinex_log)
        normalized_retinex_exp = cv2.normalize(
            retinex_exp,
            None,  # type: ignore
            0,
            255,
            cv2.NORM_MINMAX,
        )  # type: ignore
        gain_retinex_exp = img_prep.gain_compensation(retinex_exp)

        cv2.imshow("Original Image", img)
        cv2.imshow("Retinex Log", retinex_log.astype(np.uint8))
        cv2.imshow("Normalized Retinex Log", normalized_retinex_log.astype(np.uint8))
        cv2.imshow("Gain Compensated Retinex Log", gain_retinex_log.astype(np.uint8))
        cv2.imshow("Retinex Exp", retinex_exp.astype(np.uint8))
        cv2.imshow("Normalized Retinex Exp", normalized_retinex_exp.astype(np.uint8))
        cv2.imshow("Gain Compensated Retinex Exp", gain_retinex_exp.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def compare_prefilters(
    prefilters: dict[str, dict] | None = None,
    algorithm: str = "retinex",
    postfilter: str | None = None,
    algorithm_params: dict | None = None,
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    filename: str = "prefilter_comparison.json",
) -> list[ExperimentResult]:
    """Compare different prefilter types for an algorithm.

    Args:
        prefilters: Dict of {filter_name: filter_params}
            e.g., {"None": {}, "Gaussian": {"filter": "gaussian"}, "Median": {"filter": "median"}}
            Default: None, Gaussian, and Median prefilters
        algorithm: Algorithm to test (default: "retinex")
        postfilter: Postfilter to use for all tests (default: None)
        algorithm_params: Additional algorithm parameters
            Default: use_log_scale=True, smart_normalize=True, sigma=80
        datasets: List of datasets to test (default: ["low_light", "non_uniform"])
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    if prefilters is None:
        prefilters = {
            "None": {},
            "Gaussian": {"filter": "gaussian"},
            "Median": {"filter": "median"},
        }

    if algorithm_params is None:
        algorithm_params = {
            "sigma": 80,
            "use_log_scale": True,
            "smart_normalize": True,
        }

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
    print(f"Algorithm: {algorithm.upper()} (Log Scale, Smart Normalize)")
    print(
        f"Postfilter: {postfilter if postfilter else 'None'}, Sigma: {algorithm_params.get('sigma', 80)}"
    )
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


def compare_smart_normalize(
    algorithm: str = "retinex",
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    use_log_scale: bool = False,
    prefilter: str | None = None,
    postfilter: str | None = None,
    filename: str = "smart_normalize_comparison.json",
) -> list[ExperimentResult]:
    """Compare Retinex with smart_normalize (gain_compensation) vs cv2.normalize.

    Args:
        algorithm: Algorithm to test (default: "retinex")
        datasets: Datasets to test (default: ["low_light", "non_uniform"])
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    configs = []

    # Test both normalization methods
    normalize_methods = [
        ("cv2.normalize", False),
        ("gain_compensation", True),
    ]

    for method_name, use_smart in normalize_methods:
        for dataset in datasets:
            params = {
                "sigma": 80,  # Default sigma
                "use_log_scale": use_log_scale,
                "smart_normalize": use_smart,
                "prefilter": prefilter,
                "postfilter": postfilter,
            }

            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    algorithm=algorithm,
                    algorithm_params=params,
                )
            )

    print("\n" + "=" * 70)
    print("Smart Normalize Comparison: cv2.normalize vs gain_compensation")
    print(f"Algorithm: {algorithm.upper()}, Sigma: 80, No Filters")
    print(f"Datasets: {', '.join(d.replace('_', ' ').title() for d in datasets)}")
    print("=" * 70)

    results = run_batch_experiments(configs)
    save_results(results, filename)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table:")
    print("=" * 70)
    print(f"{'Method':<20} {'Dataset':<15} {'Success Rate':<15} {'Time (s)':<10}")
    print("-" * 70)

    for result in results:
        use_smart = result.config.algorithm_params.get("smart_normalize", False)
        method = "gain_compensation" if use_smart else "cv2.normalize"
        dataset = result.config.dataset.replace("_", " ").title()
        print(
            f"{method:<20} {dataset:<15} {result.success_rate:>6.2f}%         {result.processing_time:>6.3f}"
        )

    print("=" * 70)

    # Compare methods for each dataset
    print("\nComparison by Dataset:")
    print("-" * 70)
    for dataset in datasets:
        dataset_results = [r for r in results if r.config.dataset == dataset]
        cv2_result = next(
            r
            for r in dataset_results
            if not r.config.algorithm_params.get("smart_normalize", False)
        )
        smart_result = next(
            r
            for r in dataset_results
            if r.config.algorithm_params.get("smart_normalize", False)
        )

        rate_diff = smart_result.success_rate - cv2_result.success_rate
        time_diff = smart_result.processing_time - cv2_result.processing_time

        print(f"\n{dataset.replace('_', ' ').title()}:")
        print(
            f"  cv2.normalize:      {cv2_result.success_rate:>6.2f}%  ({cv2_result.processing_time:.3f}s)"
        )
        print(
            f"  gain_compensation:  {smart_result.success_rate:>6.2f}%  ({smart_result.processing_time:.3f}s)"
        )
        print(f"  Difference:         {rate_diff:>+6.2f}%  ({time_diff:+.3f}s)")

    print("=" * 70 + "\n")

    return results


def compare_postfilter(
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    filename: str = "postfilter_comparison.json",
) -> list[ExperimentResult]:
    """Compare Retinex with different postfilter options.

    Uses:
    - Retinex algorithm with log scale
    - Default sigma (80)
    - Gaussian prefilter (default params)
    - Smart normalize (gain_compensation)
    - Postfilter options: None, Gaussian, Median

    Args:
        datasets: Datasets to test (default: ["low_light", "non_uniform"])
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    configs = []

    # Test three postfilter configurations
    postfilter_configs = [
        ("None", None),
        ("Gaussian", "gaussian"),
        ("Median", "median"),
    ]

    for postfilter_name, postfilter_type in postfilter_configs:
        for dataset in datasets:
            params = {
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "prefilter": "gaussian",
                "postfilter": postfilter_type,
            }

            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    algorithm="retinex",
                    algorithm_params=params,
                )
            )

    print("\n" + "=" * 70)
    print("Postfilter Comparison: None vs Gaussian vs Median")
    print("Algorithm: RETINEX (Log Scale, Smart Normalize)")
    print("Prefilter: Gaussian (default), Sigma: 80")
    print(f"Datasets: {', '.join(d.replace('_', ' ').title() for d in datasets)}")
    print("=" * 70)

    results = run_batch_experiments(configs)
    save_results(results, filename)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table:")
    print("=" * 70)
    print(f"{'Postfilter':<15} {'Dataset':<15} {'Success Rate':<15} {'Time (s)':<10}")
    print("-" * 70)

    for result in results:
        postfilter = result.config.algorithm_params.get("postfilter", "None")
        postfilter_label = postfilter if postfilter else "None"
        dataset = result.config.dataset.replace("_", " ").title()
        print(
            f"{postfilter_label:<15} {dataset:<15} {result.success_rate:>6.2f}%         {result.processing_time:>6.3f}"
        )

    print("=" * 70)

    # Compare postfilters for each dataset
    print("\nComparison by Dataset:")
    print("-" * 70)
    for dataset in datasets:
        dataset_results = [r for r in results if r.config.dataset == dataset]

        print(f"\n{dataset.replace('_', ' ').title()}:")
        for result in dataset_results:
            postfilter = result.config.algorithm_params.get("postfilter", None)
            postfilter_label = postfilter if postfilter else "None"
            print(
                f"  {postfilter_label:<12}: {result.success_rate:>6.2f}%  ({result.processing_time:.3f}s)"
            )

    print("=" * 70 + "\n")

    return results


def compare_retinex_decomposition(
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    filename: str = "retinex_decomposition_comparison.json",
) -> list[ExperimentResult]:
    """Compare different Retinex decomposition methods.

    Compares:
    - SSR (Standard Single Scale Retinex)
    - SSR with Downsampling
    - MSR with Downsampling

    All use:
    - Log scale (use_log_scale=True)
    - Smart normalize (gain_compensation)
    - Gaussian prefilter
    - No postfilter
    - Sigma=80 for SSR methods

    Args:
        datasets: Datasets to test (default: ["low_light", "non_uniform"])
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    configs = []

    # Test three decomposition methods
    decomposition_methods = [
        ("SSR", "ssr"),
        ("SSR_Downsample", "ssr_downsample"),
        ("MSR_Downsample", "msr_downsample"),
    ]

    for method_name, method_type in decomposition_methods:
        for dataset in datasets:
            params = {
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "decomposition_method": method_type,
                "scale_factor": 0.5,
                "prefilter": "gaussian",
                "postfilter": "median",
                "prefilter_params": {"ksize": (5, 5), "sigma": 0},
                "postfilter_params": {"ksize": 5},
            }

            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    algorithm="retinex",
                    algorithm_params=params,
                )
            )

    print("\n" + "=" * 70)
    print("Retinex Decomposition Method Comparison")
    print("Methods: SSR, SSR_Downsample, MSR_Downsample")
    print("Config: Log Scale, Smart Normalize, Gaussian Prefilter, Sigma=80")
    print(f"Datasets: {', '.join(d.replace('_', ' ').title() for d in datasets)}")
    print("=" * 70)

    results = run_batch_experiments(configs)
    save_results(results, filename)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table:")
    print("=" * 70)
    print(f"{'Method':<20} {'Dataset':<15} {'Success Rate':<15} {'Time (s)':<10}")
    print("-" * 70)

    for result in results:
        method = result.config.algorithm_params.get("decomposition_method", "ssr")
        method_label = method.replace("_", " ").upper()
        dataset = result.config.dataset.replace("_", " ").title()
        print(
            f"{method_label:<20} {dataset:<15} {result.success_rate:>6.2f}%         {result.processing_time:>6.3f}"
        )

    print("=" * 70)

    # Compare methods for each dataset
    print("\nComparison by Dataset:")
    print("-" * 70)
    for dataset in datasets:
        dataset_results = [r for r in results if r.config.dataset == dataset]

        print(f"\n{dataset.replace('_', ' ').title()}:")
        for result in dataset_results:
            method = result.config.algorithm_params.get("decomposition_method", "ssr")
            method_label = method.replace("_", " ").upper()
            print(
                f"  {method_label:<18}: {result.success_rate:>6.2f}%  ({result.processing_time:.3f}s)"
            )

    print("=" * 70 + "\n")

    return results


def compare_scale_factor(
    scale_factors: list[float] | None = None,
    datasets: list[Literal["low_light", "non_uniform"]] | None = None,
    filename: str = "scale_factor_comparison.json",
) -> list[ExperimentResult]:
    """Compare different scale factors for SSR with downsampling.

    Uses:
    - SSR with downsampling
    - Gaussian prefilter: ksize=(5,5)
    - Median postfilter: ksize=5
    - Log scale (use_log_scale=True)
    - Smart normalize (gain_compensation)
    - Sigma=80

    Args:
        scale_factors: List of scale factors to test (default: 0.2 to 0.5, step 0.05)
        datasets: Datasets to test (default: ["low_light", "non_uniform"])
        filename: JSON filename to save results

    Returns:
        List of experiment results
    """
    if datasets is None:
        datasets = ["low_light", "non_uniform"]

    if scale_factors is None:
        # Generate scale factors from 0.05 to 0.5 with step 0.05
        scale_factors = np.arange(0.05, 0.55, 0.05).round(2).tolist()

    configs = []

    for scale_factor in scale_factors:
        for dataset in datasets:
            params = {
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "decomposition_method": "ssr_downsample",
                "scale_factor": scale_factor,
                "prefilter": "gaussian",
                "prefilter_params": {"ksize": (5, 5)},
                "postfilter": "median",
                "postfilter_params": {"ksize": 5},
            }

            configs.append(
                ExperimentConfig(
                    dataset=dataset,
                    algorithm="retinex",
                    algorithm_params=params,
                )
            )

    print("\n" + "=" * 70)
    print("Scale Factor Comparison for SSR with Downsampling")
    print(f"Scale Factors: {', '.join(f'{sf:.2f}' for sf in scale_factors)}")
    print("Config: Log Scale, Smart Normalize")
    print("Prefilter: Gaussian (5x5), Postfilter: Median (5), Sigma=80")
    print(f"Datasets: {', '.join(d.replace('_', ' ').title() for d in datasets)}")
    print("=" * 70)

    results = run_batch_experiments(configs)
    save_results(results, filename)

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table:")
    print("=" * 70)
    print(f"{'Scale Factor':<15} {'Dataset':<15} {'Success Rate':<15} {'Time (s)':<10}")
    print("-" * 70)

    for result in results:
        scale_factor = result.config.algorithm_params.get("scale_factor", 0.5)
        dataset = result.config.dataset.replace("_", " ").title()
        print(
            f"{scale_factor:<15.2f} {dataset:<15} {result.success_rate:>6.2f}%         {result.processing_time:>6.3f}"
        )

    print("=" * 70)

    # Compare scale factors for each dataset
    print("\nComparison by Dataset:")
    print("-" * 70)
    for dataset in datasets:
        dataset_results = [r for r in results if r.config.dataset == dataset]

        print(f"\n{dataset.replace('_', ' ').title()}:")
        for result in dataset_results:
            scale_factor = result.config.algorithm_params.get("scale_factor", 0.5)
            print(
                f"  SF={scale_factor:.2f}: {result.success_rate:>6.2f}%  ({result.processing_time:.3f}s)"
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

    if False:
        compare_smart_normalize(
            use_log_scale=False,
            prefilter="gaussian",
            filename="smart_normalize_linear_prefilter.json",
        )
        visualizer.visualize_smart_normalize_comparison(
            filename="smart_normalize_linear_prefilter.json", save=True
        )

        compare_smart_normalize(
            use_log_scale=True,
            prefilter="gaussian",
            filename="smart_normalize_log_prefilter.json",
        )

        visualizer.visualize_smart_normalize_comparison(
            filename="smart_normalize_log_prefilter.json", save=True
        )

    if False:
        # visualize_test()
        visualizer.visualize_retinex_parameter_space(save=True)

    if False:
        # Compare prefilter options
        compare_prefilters(
            postfilter="median", filename="prefilter_use_postmedian_comparison.json"
        )
        visualizer.visualize_prefilter_comparison(
            filename="prefilter_use_postmedian_comparison.json", save=True
        )

    if False:
        # Compare postfilter options
        compare_postfilter()
        visualizer.visualize_postfilter_comparison(save=True)

    if False:
        compare_retinex_decomposition(
            filename="retinex_decomposition_comparison_change_order.json"
        )
        visualizer.visualize_retinex_decomposition_comparison(
            filename="retinex_decomposition_comparison_change_order.json", save=True
        )

    if True:
        compare_scale_factor(filename="scale_factor_comparison_ssr_downsample.json")
        visualizer.visualize_scale_factor_comparison(
            filename="scale_factor_comparison_ssr_downsample.json", save=True
        )

    if False:
        img = cv2.imread(
            # str(IMAGES_DIR / "experiment" / "low_light" / "img_0_dark_lv1.png"),
            str(IMAGES_DIR / "experiment" / "non_uniform" / "img_0_sigma500.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        assert img is not None

        # print("original noise: ", analysis.identify_noise(img))

        algo = algorithms.get_algorithm(
            "retinex",
            sigma=80,
            use_log_scale=True,
            smart_normalize=True,
            prefilter="gaussian",
            postfilter="median",
        )
        algo(img)
        # print("processed noise: ", analysis.identify_noise(processed))

        # cv2.imshow("Processed Image", processed)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
