from cv_webcam import core


def main() -> None:
    calibrator = core.CameraCalibrator()
    calibrator.test_calibration()


if __name__ == "__main__":
    main()
