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


def main() -> None:
    """Main entry point."""
    cv_webcam.init_app()

    # Option 1: Run comprehensive evaluation with plots
    if True:
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
        # Default filter settings for all algorithms
        default_filters = {
            "use_gaussian_prefilter": True,
            "use_median_postfilter": True,
        }
        configs = [
            experiment.ExperimentConfig(
                dataset="low_light",
                algorithm="retinex",
                algorithm_params={
                    **default_filters,
                    "sigma": 80,
                    "use_log_scale": False,
                },
            ),
            experiment.ExperimentConfig(
                dataset="low_light",
                algorithm="retinex",
                algorithm_params={
                    **default_filters,
                    "sigma": 80,
                    "use_log_scale": True,
                },
            ),
            experiment.ExperimentConfig(
                dataset="non_uniform",
                algorithm="retinex",
                algorithm_params={
                    **default_filters,
                    "sigma": 80,
                    "use_log_scale": False,
                },
            ),
            experiment.ExperimentConfig(
                dataset="non_uniform",
                algorithm="retinex",
                algorithm_params={
                    **default_filters,
                    "sigma": 80,
                    "use_log_scale": True,
                },
            ),
        ]
        results = experiment.run_batch_experiments(configs)
        experiment.save_results(results, "comparison_log_or_exp.json")

    # Option 4: Visualize datasets
    if False:
        from cv_webcam.aruco_exp import visualizer

        visualizer.draw_low_light_imgs(save=True)
        visualizer.draw_non_uniform_imgs(save=True)

    # Option 5: Capture baseline images from webcam
    if False:
        make_baseline_imgs("https://192.168.0.103:8080/video")

    # Option 6: Generate degraded datasets
    if False:
        dataset_generator.generate_dataset()


if __name__ == "__main__":
    main()
