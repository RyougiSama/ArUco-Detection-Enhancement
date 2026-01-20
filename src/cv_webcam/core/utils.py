import time
from pathlib import Path

import cv2

from .aruco import create_aruco_detector


def img_capture(url: str, save_path: Path, check_detection: bool = False) -> None:
    if check_detection:
        detector = create_aruco_detector(marker_length=40)

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
                if check_detection:
                    if not detector.can_be_detected(frame, 0):
                        print("Marker not detected, skipping save.")
                        continue

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = save_path / f"{timestamp}.png"
                cv2.imwrite(str(filename), frame)
                print(f"Saved: {filename.name}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
