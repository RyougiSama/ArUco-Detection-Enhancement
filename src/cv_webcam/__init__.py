import logging
from pathlib import Path


def _get_project_root() -> Path:
    curr_dir = Path(__file__).parent
    for _ in range(5):
        if (curr_dir / "pyproject.toml").exists():
            return curr_dir
        curr_dir = curr_dir.parent
    raise FileNotFoundError("Project root with pyproject.toml not found")


def _setup_logging() -> None:
    log_file = LOGS_DIR / "app.log"

    app_logger = logging.getLogger("cv_webcam")
    app_logger.setLevel(logging.DEBUG)
    app_logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if app_logger.hasHandlers():
        return

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)


PROJECT_ROOT = _get_project_root()
IMAGES_DIR = PROJECT_ROOT / "images"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


def init_app() -> None:
    """Initializes runtime directories and logging."""
    for dir_path in [IMAGES_DIR, LOGS_DIR, DATA_DIR]:
        dir_path.mkdir(exist_ok=True)
    _setup_logging()


__all__ = ["PROJECT_ROOT", "IMAGES_DIR", "LOGS_DIR", "DATA_DIR", "init_app"]
