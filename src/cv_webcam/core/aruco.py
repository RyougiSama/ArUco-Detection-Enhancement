from typing import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

from cv_webcam import IMAGES_DIR

from .calibration import CalibrationParams


class ArucoConfig:
    def __init__(
        self,
        marker_length: int,
        side_pixels: int = 800,
        marker_border: int = 2,
        dict: int = cv2.aruco.DICT_6X6_250,
    ) -> None:
        self.marker_length: int = marker_length
        self.side_pixels: int = side_pixels
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
            sidePixels=self.config.side_pixels,
            borderBits=self.config.marker_border,
        )
        if save:
            filepath = IMAGES_DIR / "aruco" / f"aruco_marker_{marker_id}.png"
            cv2.imwrite(str(filepath), marker_img)
        return marker_img

    def test_generator(self) -> None:
        marker_id = 0
        marker_img = self.generate_marker(marker_id, save=False)
        cv2.imshow("Aruco Marker", marker_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


class ArucoDetector:
    def __init__(self, config: ArucoConfig, calib_params: CalibrationParams) -> None:
        self.config = config
        self._detector_params = cv2.aruco.DetectorParameters()
        self._calib_params = calib_params
        self._detector = cv2.aruco.ArucoDetector(
            self.config.predefined_dict, self._detector_params
        )
        self._obj_points: MatLike = np.array(
            [
                [-self.config.marker_length / 2, self.config.marker_length / 2, 0],
                [self.config.marker_length / 2, self.config.marker_length / 2, 0],
                [self.config.marker_length / 2, -self.config.marker_length / 2, 0],
                [-self.config.marker_length / 2, -self.config.marker_length / 2, 0],
            ],
            dtype=np.float32,
        )

    def detect_markers(self, img: MatLike) -> tuple[Sequence[MatLike], MatLike]:
        corners, ids, _ = self._detector.detectMarkers(img)
        return corners, ids

    def single_estimate_pose(
        self, corners_per_marker: MatLike
    ) -> tuple[MatLike, MatLike]:
        if (
            self._calib_params.camera_matrix is None
            or self._calib_params.dist_coeffs is None
        ):
            raise ValueError("Calibration parameters are not set.")
        ret, rvec, tvec = cv2.solvePnP(
            self._obj_points,
            corners_per_marker,
            self._calib_params.camera_matrix,
            self._calib_params.dist_coeffs,
        )
        if not ret:
            raise RuntimeError("Pose estimation failed.")
        return rvec, tvec

    def is_rotation_matrix(self, R: MatLike) -> bool:
        Rt = np.transpose(R)
        should_be_identity = np.dot(Rt, R)
        _I = np.identity(3, dtype=R.dtype)
        _n = np.linalg.norm(_I - should_be_identity)
        return bool(_n < 1e-6)

    def rotation_matrix_to_euler_angles(self, R: MatLike) -> MatLike:
        assert self.is_rotation_matrix(R)

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    def test_detector(self, url: str) -> None:
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise RuntimeError("Could not open video stream")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                corners, ids = self.detect_markers(frame)
                if ids is not None:
                    for corners_per_marker, _ in zip(corners, ids):
                        rvec, tvec = self.single_estimate_pose(corners_per_marker)
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        assert self.is_rotation_matrix(rotation_matrix)
                        cv2.drawFrameAxes(
                            frame,
                            self._calib_params.camera_matrix,
                            self._calib_params.dist_coeffs,
                            rvec,
                            tvec,
                            1.2 * self.config.marker_length,
                        )
                        dist = np.linalg.norm(tvec)
                        euler_angles = self.rotation_matrix_to_euler_angles(
                            rotation_matrix
                        )
                        roll, pitch, yaw = np.degrees(euler_angles)
                        text_str = f"Dist: {dist:.2f} mm Roll: {roll:.2f}, Pitch: {pitch:.2f}, Yaw: {yaw:.2f}"
                        cv2.putText(
                            frame,
                            text_str,
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )

                cv2.imshow("Aruco Detection", frame)
                key_val = cv2.waitKey(1) & 0xFF
                if key_val == ord("q") or key_val == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
