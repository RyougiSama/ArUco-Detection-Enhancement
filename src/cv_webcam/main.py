from cv_webcam import core


def main() -> None:
    calibrator = core.CameraCalibrator()
    calibrator.load_calibration_params()
    cfg = core.ArucoConfig(marker_length=29)
    # gen = core.ArucoGenerator(cfg)
    # gen.test_generator()
    detector = core.ArucoDetector(cfg, calibrator.calib_params)
    url = "http://10.70.81.237:8080/video"
    detector.test_detector(url)


if __name__ == "__main__":
    main()
