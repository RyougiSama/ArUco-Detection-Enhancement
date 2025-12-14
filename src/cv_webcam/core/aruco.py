from typing import Sequence

import cv2
from cv2.typing import MatLike


def generate_aruco():
    marker_size = 400  # marker影像大小
    marker_id = 20  # marker的ID
    marker_boarder = 1  # marker边界

    # step1 加载预定义的字典
    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # step2 创建marker影像
    marker_img = dict.generateImageMarker(
        marker_id, sidePixels=marker_size, borderBits=marker_boarder
    )

    # step3 显示marker影像
    cv2.imshow("marker", marker_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_aruco(img: MatLike) -> tuple[Sequence[MatLike], MatLike]:
    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dict, detector_params)
    corners, ids, _ = detector.detectMarkers(img)
    return corners, ids


def test_aruco(url: str):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError("Could not open video stream")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        corners, ids = detect_aruco(frame)
        if ids is not None:
            print("Detected IDs:", ids.flatten())
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.imshow("Aruco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
