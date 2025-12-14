# from cv_webcam.core.cv_proc import test_camera_calibrate
import cv_webcam


def main():
    # test_camera_calibrate("http://10.70.82.82:8080/video")
    print("Project Root:", cv_webcam.PROJECT_ROOT)
    print("Images Directory:", cv_webcam.IMAGES_DIR)


if __name__ == "__main__":
    main()
