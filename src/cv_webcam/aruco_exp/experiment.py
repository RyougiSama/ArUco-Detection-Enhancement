import matplotlib

matplotlib.use("TkAgg")

import cv2
import matplotlib.pyplot as plt
from cv2.typing import MatLike

from cv_webcam import IMAGES_DIR
from cv_webcam.core import create_aruco_detector


def draw_he_compare(img: MatLike) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), constrained_layout=True)

    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    axes[1, 0].hist(
        img.ravel(),
        bins=256,
        range=(0, 256),
        density=True,
        cumulative=True,
        color="blue",
        alpha=0.85,
    )
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].set_title("Original Image Histogram", fontsize=10)
    axes[1, 0].set_xlabel("Gray value")
    axes[1, 0].set_ylabel("Count")

    img_he = cv2.equalizeHist(img)

    axes[0, 1].imshow(img_he, cmap="gray")
    axes[0, 1].set_title("Histogram Equalized Image")
    axes[0, 1].axis("off")

    axes[1, 1].hist(
        img_he.ravel(),
        bins=256,
        range=(0, 256),
        density=True,
        cumulative=True,
        color="blue",
        alpha=0.85,
    )
    axes[1, 1].set_xlim(0, 255)
    axes[1, 1].set_title("Equalized Image Histogram", fontsize=10)
    axes[1, 1].set_xlabel("Gray value")
    axes[1, 1].set_ylabel("Count")

    plt.show()


def test_raw_imgs_detection():
    raw_imgs_path = IMAGES_DIR / "experiment" / "raw"
    raw_img_names = [
        img_name for img_name in raw_imgs_path.rglob("*.png") if img_name.is_file()
    ]
    detector = create_aruco_detector(marker_length=41)

    for img_name in raw_img_names:
        img = cv2.imread(str(img_name))
        assert img is not None
        if not detector.can_be_detected(img, 0):
            print(f"No marker detected in image: {img_name.name}")


def test_exp():
    img_path = IMAGES_DIR / "experiment" / "dark_lv0" / "img_0.png"
    detector = create_aruco_detector(marker_length=41)

    img = cv2.imread(str(img_path))
    assert img is not None

    img, data = detector.single_aruco_detection(img)
    if not data:
        print("No marker detected.")
    cv2.imshow("Aruco Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
