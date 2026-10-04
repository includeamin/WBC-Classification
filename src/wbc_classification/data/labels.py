"""Class discovery from the dataset folder layout."""

from pathlib import Path

from wbc_classification.errors import DataError

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def require_split_dir(path: Path) -> None:
    if not path.is_dir():
        raise DataError(
            f"Dataset directory not found: {path}. "
            "Run `wbc download-data` or point data.root at your dataset."
        )


def discover_classes(split_dir: Path) -> list[str]:
    """Class names are the sorted sub-folder names of a split directory."""
    require_split_dir(split_dir)
    class_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_dirs:
        raise DataError(f"No class folders found in {split_dir}")
    for class_dir in class_dirs:
        if not any(is_image_file(f) for f in class_dir.iterdir()):
            raise DataError(f"Class folder {class_dir} contains no images")
    return [p.name for p in class_dirs]
