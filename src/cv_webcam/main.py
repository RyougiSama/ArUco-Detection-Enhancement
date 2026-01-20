import cv_webcam
from cv_webcam import core
from cv_webcam.aruco_exp import dataset_generator, experiment
from cv_webcam.core import utils


def make_baseline_imgs(url: str) -> None:
    save_path = cv_webcam.IMAGES_DIR / "experiment" / "raw"
    utils.img_capture(url, save_path, check_detection=True)


def test_detector(url: str) -> None:
    detector = core.create_aruco_detector(marker_length=40)
    detector.test_detector(url)


def main() -> None:
    cv_webcam.init_app()
    if True:
        experiment.run_experiment()
    if False:
        make_baseline_imgs("https://192.168.0.103:8080/video")
    if False:
        dataset_generator.generate_dataset()


if __name__ == "__main__":
    main()
