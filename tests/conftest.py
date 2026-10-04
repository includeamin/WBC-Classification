from pathlib import Path

import pytest
from PIL import Image

from wbc_classification.config import Config, DataConfig, ModelConfig, TrainConfig

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


def make_tiny_config(output_dir: Path, data_root: Path) -> Config:
    return Config(
        data=DataConfig(root=data_root, image_size=32, batch_size=4, val_fraction=0.25),
        model=ModelConfig(name="baseline", pretrained=False),
        train=TrainConfig(name="test", epochs=1, amp=False, device="cpu", output_dir=output_dir),
    )


@pytest.fixture
def tiny_config(tmp_path: Path, data_root: Path) -> Config:
    return make_tiny_config(tmp_path / "runs", data_root)


@pytest.fixture(scope="session")
def trained(tmp_path_factory: pytest.TempPathFactory, data_root: Path):
    from wbc_classification.engine.train import train

    return train(make_tiny_config(tmp_path_factory.mktemp("runs"), data_root))
