from cv_webcam import core


def main():
    calibrator = core.CameraCalibrator()
    calibrator.camera_calibrate()


if __name__ == "__main__":
    main()
