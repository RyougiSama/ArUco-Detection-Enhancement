from typing import Sequence
import cv_webcam

import cv2
from cv2.typing import MatLike


class ArucoConfig:
    def __init__(
        self,
        marker_size: int = 800,
        marker_border: int = 2,
        dict: int = cv2.aruco.DICT_6X6_250,
    ) -> None:
        self.marker_size: int = marker_size
        self.marker_border: int = marker_border
        self.predefined_dict: cv2.aruco.Dictionary = cv2.aruco.getPredefinedDictionary(
            dict
        )


class ArucoGenerator:
    def __init__(self, config: ArucoConfig) -> None:
        self.config: ArucoConfig = config

    def generate_marker(self, marker_id: int, save: bool = False) -> MatLike:
        marker_img = self.config.predefined_dict.generateImageMarker(
            marker_id,
            sidePixels=self.config.marker_size,
            borderBits=self.config.marker_border,
        )
        if save:
            filepath = cv_webcam.IMAGES_DIR / "aruco" / f"aruco_marker_{marker_id}.png"
            cv2.imwrite(str(filepath), marker_img)
        return marker_img

    def test_generator(self) -> None:
        marker_id = 0
        marker_img = self.generate_marker(marker_id, save=False)
        cv2.imshow("Aruco Marker", marker_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class ArucoDetector:
    def __init__(self, config: ArucoConfig) -> None:
        self.config = config
        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(
            self.config.predefined_dict, self._detector_params
        )

    def detect_markers(self, img: MatLike) -> tuple[Sequence[MatLike], MatLike]:
        corners, ids, _ = self._detector.detectMarkers(img)
        return corners, ids

    def draw_detected_markers(
        self, img: MatLike, corners: Sequence[MatLike], ids: MatLike
    ) -> MatLike:
        return cv2.aruco.drawDetectedMarkers(img, corners, ids)

    def test_detector(self, url: str) -> None:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Could not open video stream")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            corners, ids = self.detect_markers(frame)
            if ids is not None:
                print("Detected IDs:", ids.flatten())
                frame = self.draw_detected_markers(frame, corners, ids)
            cv2.imshow("Aruco Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
