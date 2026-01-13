import cv2
import numpy as np
from cv2.typing import MatLike

from cv_webcam import IMAGES_DIR


def gamma_correction(img: MatLike, gamma: float) -> MatLike:
    table = np.array([255 * (i / 255) ** gamma for i in np.arange(0, 256)]).astype(
        "uint8"
    )
    img_corrected = cv2.LUT(img, table)
    return img_corrected


def add_gussian_noise(img: MatLike, sigma: float) -> MatLike:
    img_float = img.astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img_float.shape)
    img_noisy_float = img_float + noise
    img_noisy_float = np.clip(img_noisy_float, 0, 1)
    img_noisy = (img_noisy_float * 255).astype(np.uint8)
    return img_noisy


dark_level_dict = {
    0: [2, 0.05],
    1: [4, 0.08],
    2: [6, 0.12],
    3: [8, 0.16],
}


def generate_dark_img(img: MatLike, level: int):
    img = gamma_correction(img, dark_level_dict[level][0])
    img = add_gussian_noise(img, dark_level_dict[level][1])
    return img


def generate_dark_dataset():
    raw_imgs_path = IMAGES_DIR / "experiment" / "raw"
    raw_img_names = [
        img_name for img_name in raw_imgs_path.rglob("*.png") if img_name.is_file()
    ]

    for lv in range(4):
        save_path = IMAGES_DIR / "experiment" / f"dark_lv{lv}"
        save_path.mkdir(parents=True, exist_ok=True)
        for img_name in raw_img_names:
            img = cv2.imread(str(img_name))
            assert img is not None
            dark_img = generate_dark_img(img, lv)
            save_filepath = save_path / img_name.name
            cv2.imwrite(str(save_filepath), dark_img)
