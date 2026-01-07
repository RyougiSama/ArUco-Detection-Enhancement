import cv_webcam
from cv_webcam import core


def main() -> None:
    cv_webcam.init_app()
    calibrator = core.CameraCalibrator()
    calibrator.load_calibration_params()
    cfg = core.ArucoConfig(marker_length=41)
    # gen = core.ArucoGenerator(cfg)
    # gen.test_generator()
    detector = core.ArucoDetector(cfg, calibrator.calib_params)
    url = "http://10.54.204.186:8080/video"
    detector.test_detector(url)


if __name__ == "__main__":
    main()
