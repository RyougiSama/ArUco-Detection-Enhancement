import cv_webcam
from cv_webcam import core
from cv_webcam.aruco_exp import experiment, dataset_generator
from cv_webcam.core import utils


def make_baseline_imgs(url: str) -> None:
    save_path = cv_webcam.IMAGES_DIR / "experiment" / "raw"
    utils.img_capture(url, save_path)


def test_detector(url: str) -> None:
    detector = core.create_aruco_detector(marker_length=41)
    detector.test_detector(url)


def main() -> None:
    cv_webcam.init_app()
    experiment.run_experiment()
    if False:
        dataset_generator.generate_dataset()


if __name__ == "__main__":
    main()
