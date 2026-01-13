import time
from pathlib import Path

import cv2


def img_capture(url: str, save_path: Path) -> None:
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
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = save_path / f"{timestamp}.png"
                cv2.imwrite(str(filename), frame)
                print(f"已保存: {filename.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
