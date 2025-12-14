from pathlib import Path


def _get_project_root() -> Path:
    curr_dir = Path(__file__).parent
    for _ in range(5):
        if (curr_dir / "pyproject.toml").exists():
            return curr_dir
        curr_dir = curr_dir.parent
    raise FileNotFoundError("Project root with pyproject.toml not found")


PROJECT_ROOT = _get_project_root()
IMAGES_DIR = PROJECT_ROOT / "images"

for dir_path in [IMAGES_DIR]:
    dir_path.mkdir(exist_ok=True)

__all__ = ["PROJECT_ROOT", "IMAGES_DIR"]
