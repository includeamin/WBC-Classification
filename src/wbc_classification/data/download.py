"""Download and install the Kaggle blood-cells dataset."""

import shutil
from pathlib import Path

from wbc_classification.errors import DataError

DATASET_HANDLE = "paultimothymooney/blood-cells"
SPLITS = ("TRAIN", "TEST")


def find_dataset_root(base: Path) -> Path:
    """Find the folder that contains both TRAIN/ and TEST/ below ``base``."""
    for train_dir in sorted(base.rglob("TRAIN")):
        if train_dir.is_dir() and (train_dir.parent / "TEST").is_dir():
            return train_dir.parent
    raise DataError(f"Could not find TRAIN/ and TEST/ folders under {base}")


def install_dataset(source: Path, dest: Path, force: bool = False) -> Path:
    root = find_dataset_root(source)
    for split in SPLITS:
        target = dest / split
        if target.exists():
            if not force:
                raise DataError(f"{target} already exists; pass --force to overwrite")
            shutil.rmtree(target)
        shutil.copytree(root / split, target)
    return dest


def download_dataset(dest: Path, force: bool = False) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise DataError(
            "kagglehub is not installed. Run `poetry install --extras kaggle` "
            "(or `pip install 'wbc-classification[kaggle]'`)."
        ) from exc
    try:
        source = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as exc:  # network / auth errors from kagglehub vary widely
        raise DataError(
            f"Download failed: {exc}. If Kaggle requires authentication see data/README.md."
        ) from exc
    return install_dataset(source, dest, force)
