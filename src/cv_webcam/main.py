import cv_webcam
from cv_webcam import core
from cv_webcam.aruco_exp import experiment
from cv_webcam.core import utils


def make_baseline_imgs(url: str) -> None:
    save_path = cv_webcam.IMAGES_DIR / "experiment" / "raw"
    utils.img_capture(url, save_path)


def test_detector(url: str) -> None:
    detector = core.create_aruco_detector(marker_length=41)
    detector.test_detector(url)


def main() -> None:
    cv_webcam.init_app()
    # url = "http://10.54.204.186:8080/video"
    experiment.test_exp()


if __name__ == "__main__":
    main()
