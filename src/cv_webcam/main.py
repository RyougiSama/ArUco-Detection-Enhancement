import cv_webcam
from cv_webcam import core
from cv_webcam.aruco_exp import dataset_generator, experiment
from cv_webcam.core import utils


def make_baseline_imgs(url: str) -> None:
    """Capture baseline images for dataset generation."""
    save_path = cv_webcam.IMAGES_DIR / "experiment" / "raw"
    utils.img_capture(url, save_path, check_detection=True)


def test_detector(url: str) -> None:
    """Test ArUco detector on webcam feed."""
    detector = core.create_aruco_detector(marker_length=40)
    detector.test_detector(url)


def test_multiple_algorithm() -> None:
    configs = [
        experiment.ExperimentConfig(
            dataset="low_light",
            algorithm="retinex",
            algorithm_params={
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "prefilter": "gaussian",
                "postfilter": "median",
                "prefilter_params": {"ksize": (5, 5), "sigma": 0},
                "postfilter_params": {"ksize": 3},
            },
        ),
        experiment.ExperimentConfig(
            dataset="non_uniform",
            algorithm="retinex",
            algorithm_params={
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "prefilter": "gaussian",
                "postfilter": "median",
                "prefilter_params": {"ksize": (5, 5), "sigma": 0},
                "postfilter_params": {"ksize": 3},
            },
        ),
        experiment.ExperimentConfig(
            dataset="low_light",
            algorithm="retinex",
            algorithm_params={
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "prefilter": "gaussian",
                "postfilter": "median",
                "prefilter_params": {"ksize": (5, 5), "sigma": 0},
                "postfilter_params": {"ksize": 7},
            },
        ),
        experiment.ExperimentConfig(
            dataset="non_uniform",
            algorithm="retinex",
            algorithm_params={
                "sigma": 80,
                "use_log_scale": True,
                "smart_normalize": True,
                "prefilter": "gaussian",
                "postfilter": "median",
                "prefilter_params": {"ksize": (5, 5), "sigma": 0},
                "postfilter_params": {"ksize": 7},
            },
        ),
    ]
    results = experiment.run_batch_experiments(configs)
    experiment.save_results(results, "gaussian_k_size.json")


def main() -> None:
    """Main entry point."""
    cv_webcam.init_app()

    # Option 1: Run comprehensive evaluation with plots
    if False:
        experiment.run_experiment()

    # Option 2: Test single algorithm configuration
    if False:
        config = experiment.ExperimentConfig(
            dataset="low_light",
            algorithm="clahe",
            algorithm_params={
                "use_gaussian_prefilter": True,
                "use_median_postfilter": True,
                "clip_limit": 2.0,
            },
        )
        result = experiment.run_single_experiment(config, print_failures=False)
        print(f"\n✓ Success rate: {result.success_rate:.1f}%")

    # Option 3: Compare multiple algorithms
    if False:
        test_multiple_algorithm()

    # Option 4: Visualize datasets
    if False:
        from cv_webcam.aruco_exp import visualizer

        visualizer.draw_low_light_imgs(save=True)
        visualizer.draw_non_uniform_imgs(save=True)

    # Option 5: Capture baseline images from webcam
    if False:
        make_baseline_imgs("https://192.168.0.103:8080/video")

    # Option 6: Generate degraded datasets
    if True:
        dataset_generator.generate_dataset()

    # Option 7: Compare Gaussian vs Bilateral prefilter with Retinex
    if False:
        prefilters = {
            "Gaussian": {"filter": "gaussian"},
            "Bilateral": {
                "filter": "bilateral",
                "params": {"d": 9, "sigma_color": 75, "sigma_space": 75},
            },
        }

        experiment.compare_prefilters(
            prefilters=prefilters,
            algorithm="retinex",
            postfilter="median",
            algorithm_params={"sigma": 80, "use_log_scale": False},
            filename="gaussian_vs_bilateral_prefilter.json",
        )

        # Visualize results
        visualizer.visualize_prefilter_comparison(
            filename="gaussian_vs_bilateral_prefilter.json",
            save=True,
        )

    # Option 8: Compare different sigma values for Retinex
    if False:
        # experiment.compare_sigma_values(
        #     sigmas=[15, 50, 80, 120, 180, 250],
        #     algorithm="retinex",
        #     use_filters=False,  # No pre/post filters
        #     filename="sigma_comparison.json",
        # )

        # Visualize results
        from cv_webcam.aruco_exp import visualizer

        visualizer.visualize_sigma_comparison(
            filename="sigma_comparison.json",
            save=True,
        )


if __name__ == "__main__":
    main()
