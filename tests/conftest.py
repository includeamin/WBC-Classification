from pathlib import Path

import pytest
from PIL import Image

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def data_root() -> Path:
    return FIXTURES / "data"


@pytest.fixture(scope="session")
def cell_image() -> Path:
    return FIXTURES / "cell.jpeg"


def _write_image(path: Path, mode: str = "RGB", size: tuple[int, int] = (64, 48)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size).save(path)
    return path


@pytest.fixture
def write_image():
    return _write_image
