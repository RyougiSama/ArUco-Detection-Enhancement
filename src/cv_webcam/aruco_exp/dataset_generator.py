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
    0: [2, 0.04],
    1: [3, 0.06],
    2: [4, 0.08],
    3: [6, 0.10],
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
    save_path = IMAGES_DIR / "experiment" / "low_light"
    save_path.mkdir(parents=True, exist_ok=True)

    for lv in range(4):
        for img_name in raw_img_names:
            img = cv2.imread(str(img_name))
            assert img is not None
            dark_img = generate_dark_img(img, lv)
            save_filepath = save_path / f"{img_name.stem}_dark_lv{lv}.png"
            cv2.imwrite(str(save_filepath), dark_img)


def create_gaussian_illumination_map(
    shape: tuple[int, int] | tuple[int, int, int],
    center: tuple[int, int],
    sigma: float,
    center_intensity: float = 1.0,
) -> np.ndarray:
    if sigma <= 0 or center_intensity <= 0:
        raise ValueError("sigma and center_intensity must be greater than 0")
    height, width = shape[:2] if len(shape) == 3 else shape

    x = np.arange(0, width, dtype=np.float32)
    y = np.arange(0, height, dtype=np.float32)
    x, y = np.meshgrid(x, y)

    dx = x - center[0]
    dy = y - center[1]
    distance_sq = dx**2 + dy**2

    illumination_map = center_intensity * np.exp(-distance_sq / (2.0 * sigma**2))
    return illumination_map


def apply_illumination(
    image: MatLike, illumination_map: MatLike, gamma_correction: float = 1.0
) -> MatLike:
    if gamma_correction <= 0:
        raise ValueError("gamma_correction must be greater than 0")
    if illumination_map.shape != image.shape[:2]:
        raise ValueError("illumination_map shape must match image spatial dimensions")

    img_float = image.astype(np.float32) / 255.0

    if len(image.shape) == 3 and len(illumination_map.shape) == 2:
        illumination_map = np.stack([illumination_map] * 3, axis=-1)

    result_float = img_float * illumination_map
    result_float = np.clip(result_float, 0, 1)

    if gamma_correction != 1.0:
        result_float = result_float**gamma_correction

    result = (result_float * 255).astype(np.uint8)
    return result


def generate_non_uniform_illumination_dataset():
    raw_imgs_path = IMAGES_DIR / "experiment" / "raw"
    raw_img_names = [
        img_name for img_name in raw_imgs_path.rglob("*.png") if img_name.is_file()
    ]

    save_path = IMAGES_DIR / "experiment" / "non_uniform"
    save_path.mkdir(parents=True, exist_ok=True)

    for img_name in raw_img_names:
        img = cv2.imread(str(img_name))
        assert img is not None

        center = (
            np.random.randint(0, img.shape[1]),
            np.random.randint(0, img.shape[0]),
        )
        sigma_list = [300, 500, 700]

        for sigma in sigma_list:
            illumination_map = create_gaussian_illumination_map(
                img.shape, center, sigma, center_intensity=1.2
            )

            illum_img = apply_illumination(img, illumination_map, 0.8)

            save_filepath = save_path / f"{img_name.stem}_sigma{sigma}.png"
            cv2.imwrite(str(save_filepath), illum_img)


def generate_dataset():
    """Generate datasets with different lighting conditions."""
    generate_dark_dataset()
    generate_non_uniform_illumination_dataset()
