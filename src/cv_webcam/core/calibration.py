from typing import Sequence

import cv2
from cv2.typing import MatLike

import cv_webcam

logger = cv_webcam.get_logger("core.calibration")


class CameraCalibrator:
    def __init__(self, board_size: tuple[int, int] = (9, 6)):
        self.board_size = board_size

    def camera_calibrate(self):
        imgs = self._get_calib_imgs()
        imgs_points = self._find_calib_points(imgs)
        logger.info(f"Found corners in {len(imgs_points)} images.")

    def _get_calib_imgs(self) -> Sequence[MatLike]:
        imgs = []
        calib_dir = cv_webcam.IMAGES_DIR / "calibration"

        for item in calib_dir.glob("*.png"):
            img = cv2.imread(str(item))
            if img is None:
                raise FileNotFoundError(f"Image not found: {item}")
            imgs.append(img)

        return imgs

    def _find_calib_points(self, imgs: Sequence[MatLike]) -> Sequence[MatLike]:
        imgs_points = []

        for i, img in enumerate(imgs):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, points = cv2.findChessboardCorners(gray, self.board_size)
            if not ret:
                raise ValueError(f"Chessboard corners not found in image {i}")

            ret, points = cv2.find4QuadCornerSubpix(gray, points, (5, 5))
            if not ret:
                raise ValueError(f"Corner refinement failed in image {i}")

            imgs_points.append(points)
        return imgs_points
