from datetime import datetime

import cv2

from cv_webcam import IMAGES_DIR


def img_capture(url: str, focus_dist: str) -> None:
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video source: {url}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from video source.")
            cv2.imshow("Webcam Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = (
                    IMAGES_DIR / "calibration" / f"calib_f{focus_dist}_{timestamp}.png"
                )
                cv2.imwrite(str(save_path), frame)
                print(f"已保存: {save_path.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
