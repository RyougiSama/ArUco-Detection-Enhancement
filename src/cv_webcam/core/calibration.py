import logging
from typing import Sequence

import cv2
import numpy as np
from cv2.typing import MatLike

import cv_webcam

logger = logging.getLogger(__name__)


class CalibrationParams:
    def __init__(self) -> None:
        self._rms: float | None = None
        self._camera_matrix: MatLike | None = None
        self._dist_coeffs: MatLike | None = None
        self._rvecs: list[MatLike] | None = None
        self._tvecs: list[MatLike] | None = None

    @property
    def rms(self) -> float | None:
        return self._rms

    @property
    def camera_matrix(self) -> MatLike | None:
        return self._camera_matrix

    @property
    def dist_coeffs(self) -> MatLike | None:
        return self._dist_coeffs

    @property
    def rvecs(self) -> list[MatLike] | None:
        return self._rvecs

    @property
    def tvecs(self) -> list[MatLike] | None:
        return self._tvecs


class CameraCalibrator:
    def __init__(
        self, board_size: tuple[int, int] = (9, 6), square_size: int = 20
    ) -> None:
        self._board_size: tuple[int, int] = board_size
        self._square_size: int = square_size
        self._img_size: tuple[int, int] | None = None
        self._obj_points: list[MatLike] = []
        self._img_points: list[MatLike] = []
        self._obj_p: MatLike = self._calc_obj_p()
        self._calib_params = CalibrationParams()

    def camera_calibrate(self) -> None:
        imgs = self._get_calib_imgs()
        self._set_arg_points(imgs)
        logger.info(f"Found corners in {len(self._img_points)} images.")
        if not self._img_size:
            raise ValueError("No images found for calibration.")

        (
            self._calib_params._rms,
            self._calib_params._camera_matrix,
            self._calib_params._dist_coeffs,
            self._calib_params._rvecs,
            self._calib_params._tvecs,
        ) = cv2.calibrateCamera(
            self._obj_points,
            self._img_points,
            self._img_size,
            None,  # type: ignore
            None,  # type: ignore
        )

        logger.info(f"Calibration RMS error: {self._calib_params.rms}")
        logger.debug(f"Camera matrix:\n{self._calib_params.camera_matrix}")
        logger.debug(f"Distortion coefficients:\n{self._calib_params.dist_coeffs}")

    def undistort_image(self, img: MatLike) -> MatLike:
        if (
            self._calib_params.camera_matrix is None
            or self._calib_params.dist_coeffs is None
        ):
            raise ValueError("Calibration parameters are not set.")

        undistorted_img = cv2.undistort(
            img,
            self._calib_params.camera_matrix,
            self._calib_params.dist_coeffs,
        )
        return undistorted_img

    def test_calibration(self) -> None:
        self.camera_calibrate()
        test_img_path = cv_webcam.IMAGES_DIR / "calib_test.png"
        img = cv2.imread(str(test_img_path))
        if img is None:
            raise FileNotFoundError(f"Test image not found: {test_img_path}")

        undistorted_img = self.undistort_image(img)

        cv2.imshow("Original Image", img)
        cv2.imshow("Undistorted Image", undistorted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _get_calib_imgs(self) -> Sequence[MatLike]:
        imgs = []
        calib_dir = cv_webcam.IMAGES_DIR / "calibration"

        for item in calib_dir.glob("*.png"):
            img = cv2.imread(str(item))
            if img is None:
                raise FileNotFoundError(f"Image not found: {item}")
            if self._img_size is None:
                self._img_size = img.shape[1], img.shape[0]
            imgs.append(img)

        return imgs

    def _set_arg_points(self, imgs: Sequence[MatLike]) -> None:
        for img in imgs:
            self._calc_img_points(img)
            self._obj_points.append(self._obj_p)

    def _calc_img_points(self, img: MatLike) -> None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, points = cv2.findChessboardCorners(gray, self._board_size)
        if not ret:
            raise ValueError("Chessboard corners not found in image")

        ret, points = cv2.find4QuadCornerSubpix(gray, points, (5, 5))
        if not ret:
            raise ValueError("Corner refinement failed in image")

        self._img_points.append(points)

    def _calc_obj_p(self) -> MatLike:
        obj_p = np.zeros((self._board_size[0] * self._board_size[1], 3), np.float32)
        obj_p[:, :2] = np.mgrid[
            0 : self._board_size[0], 0 : self._board_size[1]
        ].T.reshape(-1, 2)
        obj_p *= self._square_size
        return obj_p
