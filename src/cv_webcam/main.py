from cv_webcam import core


def main() -> None:
    cfg = core.ArucoConfig()
    # gen = core.ArucoGenerator(cfg)
    # gen.test_generator()
    detector = core.ArucoDetector(cfg)
    url = "http://10.70.33.195:8080/video"
    detector.test_detector(url)


if __name__ == "__main__":
    main()
