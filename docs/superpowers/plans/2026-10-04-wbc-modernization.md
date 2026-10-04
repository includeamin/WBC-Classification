# WBC-Classification Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy Keras scripts with a Poetry-managed, PyTorch-based, documented, tested and auto-released open-source package (`wbc-classification`) with a `wbc` CLI.

**Architecture:** A `src/wbc_classification` package with small single-purpose modules (config, labels/data, optional OpenCV segmentation, models + registry, engine for train/evaluate/predict, Typer CLI). Everything is config-driven (pydantic + YAML); a checkpoint file carries weights, class names and config so inference needs nothing else. Errors are raised as `WBCError` subclasses at boundaries and rendered by the CLI.

**Tech Stack:** Python 3.14 (floor 3.12), Poetry, PyTorch + torchvision, OpenCV (headless), pydantic, Typer, scikit-learn (metrics/split), pytest, ruff, mypy, MkDocs Material + mkdocstrings, python-semantic-release, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-10-04-wbc-modernization-design.md`

## Global Constraints

- Python: `requires-python = ">=3.12,<3.15"`; develop on 3.14; CI matrix 3.12 / 3.13 / 3.14.
- Poetry with a committed `poetry.lock`; `src/` layout; console script `wbc = "wbc_classification.cli:app"`.
- Dependencies are added with `poetry add` (latest at install time, caret ranges); never hand-write version numbers from memory.
- Dependency groups: main (torch, torchvision, numpy, opencv-python-headless, pillow, scikit-learn, matplotlib, typer, pydantic, pyyaml); dev (pytest, ruff, mypy, types-pyyaml); docs (mkdocs, mkdocs-material, mkdocstrings[python]); release (python-semantic-release); optional extra `kaggle` (kagglehub).
- Nothing runs at import time (no argparse / `test()` / file IO at module level).
- Errors: raise `WBCError` subclasses (`ConfigError`, `DataError`, `SegmentationError`, `CheckpointError`) with actionable messages; no silent `print` + `return None`.
- Data layout is the Kaggle layout: `<root>/TRAIN/<CLASS>/*.jpeg`, `<root>/TEST/<CLASS>/*.jpeg`; TEST is used only by `evaluate`.
- Cell crop is optional via `preprocessing.crop` (default `false`).
- Conventional Commits for every commit (`feat:`, `fix:`, `docs:`, `chore:`, `ci:`, `test:`, `refactor:`), each commit message ending with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- Before every commit run: `poetry run ruff check --fix . && poetry run ruff format . && poetry run mypy src && poetry run pytest -q` (once those tools exist, i.e. from Task 2 on; mypy/pytest as applicable).
- Work happens on branch `refactor/modernization`; never push or rewrite git history. The `master` → `main` rename and GitHub settings are owner steps (see final task).
- No pretrained checkpoint or metrics are published; docs ship with a results template.

## Review Focus

Inputs/conditions the spec implies but whose behaviour a user would expect, each pinned by a test in the owning task:

1. Non-RGB images (grayscale, RGBA) are loaded as RGB, and an unreadable/corrupt image raises `DataError` naming the file — Task 5.
2. Stray non-image files (`.DS_Store`, `notes.txt`) in class folders are ignored; an empty class folder or missing data dir raises `DataError` with a `wbc download-data` hint — Task 5.
3. A dataset too small for a stratified validation split raises `DataError`, not a raw sklearn `ValueError` — Task 5.
4. `predict` on a directory with no images, on a missing path, or on a corrupt image raises `DataError`, never a traceback from deep inside PIL/torch — Task 9.
5. A corrupt, non-checkpoint, or class-count-mismatched checkpoint file raises `CheckpointError`; a typo'd config key raises `ConfigError` naming the key — Tasks 3 and 7.

---

### Task 1: Remove dataset images from the tree, keep test fixtures

**Files:**
- Create: `tests/fixtures/data/{TRAIN,TEST}/<CLASS>/*.jpeg` (moved), `tests/fixtures/cell.jpeg` (moved), `data/README.md`
- Modify: `.gitignore`
- Delete: `CNN/datasets/TRAIN`, `CNN/datasets/TEST`, `CNN/datasets/TEST_SIMPLE`

**Interfaces:**
- Produces: fixture dataset `tests/fixtures/data` with 6 train + 2 test images per class (EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL), 320×240 RGB JPEG; `tests/fixtures/cell.jpeg` (a real cell image the HSV segmentation is known to work on).

- [ ] **Step 1: Move fixture images with git**

```bash
set -e
src=CNN/datasets; dst=tests/fixtures/data
for cls in EOSINOPHIL LYMPHOCYTE MONOCYTE NEUTROPHIL; do
  mkdir -p $dst/TRAIN/$cls $dst/TEST/$cls
  git ls-files $src/TRAIN/$cls | sort | head -6 | while read -r f; do git mv "$f" $dst/TRAIN/$cls/; done
  git ls-files $src/TEST/$cls  | sort | head -2 | while read -r f; do git mv "$f" $dst/TEST/$cls/; done
done
git mv WBC-Detection/4.jpeg tests/fixtures/cell.jpeg
```

- [ ] **Step 2: Verify counts**

Run: `find tests/fixtures/data/TRAIN -name '*.jpeg' | wc -l; find tests/fixtures/data/TEST -name '*.jpeg' | wc -l`
Expected: `24` then `8`

- [ ] **Step 3: Remove the rest of the dataset from the tree**

```bash
git rm -r -q CNN/datasets/TRAIN CNN/datasets/TEST CNN/datasets/TEST_SIMPLE
git ls-files | grep -ciE '\.(jpe?g|png)$'
```
Expected: a small number (fixtures + `WBC-Detection/*.png` + `image.png`, about 39); the old dataset is gone.

- [ ] **Step 4: Write `data/README.md`**

````markdown
# Data

The dataset is **not** stored in git. It is the public Kaggle
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset
(four classes: EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL; 320×240 JPEG).

## Download

```bash
poetry install --extras kaggle
poetry run wbc download-data --dest data
```

If the download fails because Kaggle requires authentication, create an API token at
<https://www.kaggle.com/settings> and expose it as `KAGGLE_USERNAME` / `KAGGLE_KEY`
(or `~/.kaggle/kaggle.json`), then retry. You can also download and unzip the dataset
manually and copy its `TRAIN/` and `TEST/` folders here.

## Expected layout

```
data/
├── TRAIN/
│   ├── EOSINOPHIL/*.jpeg
│   ├── LYMPHOCYTE/*.jpeg
│   ├── MONOCYTE/*.jpeg
│   └── NEUTROPHIL/*.jpeg
└── TEST/            # same four class folders; used only by `wbc evaluate`
```

Class names are taken from the folder names. Files that are not `.jpg/.jpeg/.png` are ignored.
Everything in this folder except this README is git-ignored.
````

- [ ] **Step 5: Update `.gitignore`**

Append:

```gitignore

# Project
/data/*
!/data/README.md
runs/
site/
docs/reference/cli.md
.venv/
.ruff_cache/
.mypy_cache/
.pytest_cache/
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: remove dataset images from tree, keep test fixtures

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Poetry scaffold, tooling, package skeleton

**Files:**
- Create: `pyproject.toml`, `src/wbc_classification/__init__.py`, `src/wbc_classification/py.typed`, `tests/__init__.py` (empty), `tests/test_package.py`, `CHANGELOG.md`
- Generated: `poetry.lock`

**Interfaces:**
- Produces: importable package `wbc_classification` with `__version__: str`; a working Poetry environment on Python 3.14 with all groups installed.

- [ ] **Step 1: Write the failing test** — `tests/test_package.py`

```python
import wbc_classification


def test_version_is_a_string():
    assert isinstance(wbc_classification.__version__, str)
    assert wbc_classification.__version__.count(".") == 2
```

- [ ] **Step 2: Write `pyproject.toml` skeleton (no dependencies yet)**

```toml
[project]
name = "wbc-classification"
version = "0.0.0"
description = "Classify white blood cell images (eosinophil, lymphocyte, monocyte, neutrophil) with PyTorch."
readme = "README.md"
requires-python = ">=3.12,<3.15"
license = { text = "BSD-3-Clause" }
authors = [{ name = "Amin Jamal" }]
keywords = ["white blood cells", "image classification", "pytorch", "medical imaging"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Image Recognition",
]
dependencies = []

[project.urls]
Homepage = "https://github.com/includeamin/WBC-Classification"
Documentation = "https://includeamin.github.io/WBC-Classification/"
Changelog = "https://github.com/includeamin/WBC-Classification/blob/main/CHANGELOG.md"

[project.scripts]
wbc = "wbc_classification.cli:app"

[project.optional-dependencies]
kaggle = []

[tool.poetry]
packages = [{ include = "wbc_classification", from = "src" }]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
check_untyped_defs = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow'"
markers = ["slow: long-running tests (full training, network); not run in CI"]
```

- [ ] **Step 3: Create the environment and add dependencies (latest versions)**

```bash
poetry env use python3.14
poetry add torch torchvision numpy opencv-python-headless pillow scikit-learn matplotlib typer pydantic pyyaml
poetry add --group dev pytest ruff mypy types-pyyaml
poetry add --group docs mkdocs mkdocs-material "mkdocstrings[python]"
poetry add --group release python-semantic-release
```
Expected: each command resolves and installs. If one fails because a package has no Python 3.14 support, STOP and report which package (the spec's fallback is dev on 3.13); do not pin old versions silently.

- [ ] **Step 4: Add the `kaggle` extra**

Edit `pyproject.toml`: set `kaggle = ["kagglehub"]` under `[project.optional-dependencies]`, then:

```bash
poetry lock
poetry install --with dev,docs,release --extras kaggle
```
Then replace `"kagglehub"` with `"kagglehub>=<version in poetry.lock>"` (read it with `poetry show kagglehub | head -3`) and run `poetry lock` again.

- [ ] **Step 5: Create the package**

`src/wbc_classification/__init__.py`:

```python
"""Classification of white blood cell images with PyTorch."""

from importlib.metadata import version

__version__ = version("wbc-classification")

__all__ = ["__version__"]
```

`src/wbc_classification/py.typed`: empty file.
`CHANGELOG.md`:

```markdown
# CHANGELOG

<!-- version list -->
```
`README.md` currently exists (legacy); leave it for Task 14.

- [ ] **Step 6: Run the test**

Run: `poetry run pytest tests/test_package.py -v`
Expected: PASS. Also run `poetry run ruff check . && poetry run mypy src`; expected clean.

- [ ] **Step 7: Verify OpenCV 5 imports**

Run: `poetry run python -c "import cv2, torch, torchvision; print(cv2.__version__, torch.__version__, torchvision.__version__)"`
Expected: prints three versions. If OpenCV 5 fails to import, run `poetry add "opencv-python-headless<5"` and note it in the commit message.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml poetry.lock src tests CHANGELOG.md
git commit -m "build: scaffold Poetry project with src layout and tooling

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Errors and configuration

**Files:**
- Create: `src/wbc_classification/errors.py`, `src/wbc_classification/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `errors.WBCError(Exception)`, `ConfigError`, `DataError`, `SegmentationError`, `CheckpointError` (all subclass `WBCError`).
  - `config.Config` with sections `preprocessing: PreprocessingConfig(crop: bool=False)`, `data: DataConfig(root: Path="data", image_size: int=224, batch_size: int=32, val_fraction: float=0.2, num_workers: int=0)`, `model: ModelConfig(name: str="resnet18", pretrained: bool=True, freeze_backbone_epochs: int=0)`, `train: TrainConfig(name: str="run", epochs: int=30, lr: float=1e-3, weight_decay: float=1e-4, patience: int=7, amp: bool=True, seed: int=42, device: str="auto", output_dir: Path="runs")`.
  - `load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config` — override keys are `"section.field"`, `None` values are skipped.
  - `dump_config(config: Config) -> str` (YAML).

- [ ] **Step 1: Write the failing tests** — `tests/test_config.py`

```python
from pathlib import Path

import pytest

from wbc_classification.config import Config, dump_config, load_config
from wbc_classification.errors import ConfigError


def test_defaults():
    cfg = Config()
    assert cfg.model.name == "resnet18"
    assert cfg.preprocessing.crop is False
    assert cfg.data.root == Path("data")


def test_load_yaml_with_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("model:\n  name: baseline\ntrain:\n  epochs: 3\n")
    cfg = load_config(path, {"train.epochs": 5, "data.image_size": None, "data.batch_size": 8})
    assert cfg.model.name == "baseline"
    assert cfg.train.epochs == 5
    assert cfg.data.image_size == 224  # None override ignored
    assert cfg.data.batch_size == 8


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_typo_in_section_is_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("trian:\n  epochs: 3\n")
    with pytest.raises(ConfigError, match="trian"):
        load_config(path)


def test_typo_in_field_names_the_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  epohcs: 3\n")
    with pytest.raises(ConfigError, match="epohcs"):
        load_config(path)


def test_out_of_range_value():
    with pytest.raises(ConfigError, match="val_fraction"):
        load_config(None, {"data.val_fraction": 0.9})


def test_bad_override_key():
    with pytest.raises(ConfigError, match="section.field"):
        load_config(None, {"epochs": 3})


def test_invalid_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train: [unclosed")
    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(path)


def test_dump_roundtrip(tmp_path):
    cfg = load_config(None, {"train.epochs": 7, "data.root": "somewhere"})
    path = tmp_path / "out.yaml"
    path.write_text(dump_config(cfg))
    assert load_config(path) == cfg
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: wbc_classification.config`)

- [ ] **Step 3: Implement** — `src/wbc_classification/errors.py`

```python
"""Exception types raised for user-facing problems."""


class WBCError(Exception):
    """Base class for errors the CLI reports without a traceback."""


class ConfigError(WBCError):
    """Invalid or unreadable configuration."""


class DataError(WBCError):
    """Missing, empty or unreadable dataset / image."""


class SegmentationError(WBCError):
    """No cell could be located in an image."""


class CheckpointError(WBCError):
    """Checkpoint is missing, corrupt or inconsistent."""
```

`src/wbc_classification/config.py`:

```python
"""Experiment configuration: pydantic models, YAML loading and CLI overrides."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from wbc_classification.errors import ConfigError


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PreprocessingConfig(_Section):
    crop: bool = False


class DataConfig(_Section):
    root: Path = Path("data")
    image_size: int = Field(224, ge=16)
    batch_size: int = Field(32, ge=1)
    val_fraction: float = Field(0.2, gt=0, lt=0.5)
    num_workers: int = Field(0, ge=0)


class ModelConfig(_Section):
    name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone_epochs: int = Field(0, ge=0)


class TrainConfig(_Section):
    name: str = "run"
    epochs: int = Field(30, ge=1)
    lr: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-4, ge=0)
    patience: int = Field(7, ge=1)
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("runs")


class Config(_Section):
    preprocessing: PreprocessingConfig = PreprocessingConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()


def load_config(path: Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (optional) and apply ``{"section.field": value}`` overrides."""
    raw: dict[str, Any] = {}
    if path is not None:
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        try:
            loaded = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"Config {path} must be a mapping of sections")
        raw = loaded

    for key, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, field = key.partition(".")
        if not field:
            raise ConfigError(f"Override key must look like 'section.field', got {key!r}")
        target = raw.setdefault(section, {})
        if not isinstance(target, dict):
            raise ConfigError(f"Config section {section!r} must be a mapping")
        target[field] = value

    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration: {details}") from exc


def dump_config(config: Config) -> str:
    """Serialise a config to YAML (paths become strings)."""
    return yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False)
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_config.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add error types and validated YAML configuration

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Optional cell segmentation / crop

**Files:**
- Create: `src/wbc_classification/preprocessing/__init__.py` (empty docstring), `src/wbc_classification/preprocessing/segmentation.py`, `tests/test_segmentation.py`

**Interfaces:**
- Consumes: `errors.SegmentationError`.
- Produces (all take an **RGB** `np.ndarray` of shape `(H, W, 3)`, dtype `uint8`):
  - `CellSegmentation(mask: np.ndarray, contour: np.ndarray, center: tuple[int, int], radius: int)` (frozen dataclass)
  - `segment_cell(rgb) -> CellSegmentation` (raises `SegmentationError`)
  - `crop_cell(rgb) -> np.ndarray` (square-ish crop around the minimum enclosing circle, clamped to the image)
  - `draw_overlay(rgb, segmentation) -> np.ndarray`

- [ ] **Step 1: Write the failing tests** — `tests/test_segmentation.py`

```python
import cv2
import numpy as np
import pytest
from PIL import Image

from wbc_classification.errors import SegmentationError
from wbc_classification.preprocessing.segmentation import crop_cell, draw_overlay, segment_cell


def _synthetic_cell() -> np.ndarray:
    image = np.full((200, 300, 3), 255, dtype=np.uint8)
    cv2.circle(image, (150, 100), 40, (120, 60, 160), -1)  # purple disc on white
    return image


def test_segments_synthetic_cell():
    seg = segment_cell(_synthetic_cell())
    assert abs(seg.center[0] - 150) <= 3 and abs(seg.center[1] - 100) <= 3
    assert abs(seg.radius - 40) <= 4
    assert seg.mask.shape == (200, 300)


def test_crop_is_around_the_cell():
    crop = crop_cell(_synthetic_cell())
    assert 70 <= crop.shape[0] <= 90 and 70 <= crop.shape[1] <= 90


def test_crop_is_clamped_at_image_border():
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.circle(image, (5, 5), 30, (120, 60, 160), -1)  # cell hanging off the corner
    crop = crop_cell(image)
    assert crop.size > 0 and crop.shape[0] <= 100 and crop.shape[1] <= 100


def test_no_cell_raises():
    with pytest.raises(SegmentationError):
        segment_cell(np.zeros((240, 320, 3), dtype=np.uint8))


def test_bad_shape_raises():
    with pytest.raises(SegmentationError):
        segment_cell(np.zeros((10, 10), dtype=np.uint8))


def test_real_image_fixture(cell_image):
    rgb = np.asarray(Image.open(cell_image).convert("RGB"))
    crop = crop_cell(rgb)
    assert crop.ndim == 3 and crop.size > 0
    assert crop.shape[0] <= rgb.shape[0] and crop.shape[1] <= rgb.shape[1]


def test_overlay_keeps_shape():
    image = _synthetic_cell()
    assert draw_overlay(image, segment_cell(image)).shape == image.shape
```

- [ ] **Step 2: Create `tests/conftest.py`** (shared by later tasks)

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `poetry run pytest tests/test_segmentation.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement** — `src/wbc_classification/preprocessing/segmentation.py`

```python
"""Classical HSV-threshold segmentation used for the optional cell crop."""

from dataclasses import dataclass

import cv2
import numpy as np

from wbc_classification.errors import SegmentationError

HSV_LOWER = np.array([80, 60, 140], dtype=np.uint8)
HSV_UPPER = np.array([255, 255, 255], dtype=np.uint8)
MIN_AREA_FRACTION = 0.01


@dataclass(frozen=True)
class CellSegmentation:
    mask: np.ndarray
    contour: np.ndarray
    center: tuple[int, int]
    radius: int


def segment_cell(rgb: np.ndarray) -> CellSegmentation:
    """Locate the largest cell-coloured region in an RGB image."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise SegmentationError(f"expected an (H, W, 3) RGB image, got shape {rgb.shape}")
    blurred = cv2.GaussianBlur(rgb, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    threshold = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise SegmentationError("no cell-coloured region found")
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < MIN_AREA_FRACTION * rgb.shape[0] * rgb.shape[1]:
        raise SegmentationError("detected region is too small to be a cell")
    mask = np.zeros(threshold.shape, dtype=np.uint8)
    cv2.drawContours(mask, [biggest], -1, 255, -1)
    (cx, cy), radius = cv2.minEnclosingCircle(biggest)
    return CellSegmentation(mask, biggest, (int(cx), int(cy)), int(radius))


def crop_cell(rgb: np.ndarray) -> np.ndarray:
    """Crop the image to the cell's enclosing square, clamped to the image bounds."""
    segmentation = segment_cell(rgb)
    height, width = rgb.shape[:2]
    cx, cy = segmentation.center
    radius = max(segmentation.radius, 1)
    return rgb[max(cy - radius, 0) : min(cy + radius, height), max(cx - radius, 0) : min(cx + radius, width)]


def draw_overlay(rgb: np.ndarray, segmentation: CellSegmentation) -> np.ndarray:
    """Return a copy of the image with the cell contour and enclosing circle drawn."""
    overlay = rgb.copy()
    cv2.drawContours(overlay, [segmentation.contour], -1, (255, 0, 0), 2)
    cv2.circle(overlay, segmentation.center, segmentation.radius, (0, 255, 0), 2)
    return overlay
```

`src/wbc_classification/preprocessing/__init__.py`: `"""Image preprocessing."""`

- [ ] **Step 5: Run tests**

Run: `poetry run pytest tests/test_segmentation.py -v`
Expected: all PASS. If `test_real_image_fixture` raises "too small", lower `MIN_AREA_FRACTION` until the fixture passes and keep `test_no_cell_raises` green; record the new value in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "feat: add optional HSV cell segmentation and crop

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Labels, datasets, transforms, loaders, dataset download helper

**Files:**
- Create: `src/wbc_classification/data/__init__.py`, `labels.py`, `datasets.py`, `transforms.py`, `download.py`; `tests/test_data.py`

**Interfaces:**
- Consumes: `Config`, `errors.*`, `segmentation.crop_cell`.
- Produces:
  - `labels.IMAGE_EXTENSIONS: frozenset[str]`, `is_image_file(path: Path) -> bool`, `discover_classes(split_dir: Path) -> list[str]` (sorted, raises `DataError`).
  - `transforms.build_transforms(image_size: int, train: bool) -> torchvision.transforms.v2.Compose`.
  - `datasets.list_samples(split_dir: Path, class_names: Sequence[str]) -> list[tuple[Path, int]]`; `stratified_split(samples, val_fraction: float, seed: int) -> tuple[list, list]`; `load_image(path: Path, crop: bool = False) -> PIL.Image.Image` (always RGB); `WBCDataset(samples, transform, crop=False)` yielding `(Tensor, int)`; `Loaders(train: DataLoader, val: DataLoader, class_names: list[str])`; `build_train_val_loaders(cfg: Config) -> Loaders`; `build_test_loader(cfg: Config, class_names: Sequence[str]) -> DataLoader`.
  - `download.find_dataset_root(base: Path) -> Path`, `install_dataset(source: Path, dest: Path, force: bool = False) -> Path`, `download_dataset(dest: Path, force: bool = False) -> Path`.

- [ ] **Step 1: Write the failing tests** — `tests/test_data.py`

```python
import logging

import pytest
import torch
from PIL import Image

from wbc_classification.config import Config, DataConfig
from wbc_classification.data.datasets import (
    WBCDataset,
    build_test_loader,
    build_train_val_loaders,
    list_samples,
    load_image,
    stratified_split,
)
from wbc_classification.data.download import find_dataset_root, install_dataset
from wbc_classification.data.labels import discover_classes
from wbc_classification.data.transforms import build_transforms
from wbc_classification.errors import DataError

CLASSES = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]


def _cfg(data_root):
    return Config(data=DataConfig(root=data_root, image_size=32, batch_size=4, val_fraction=0.25))


def test_discover_classes_sorted(data_root):
    assert discover_classes(data_root / "TRAIN") == CLASSES


def test_missing_dir_hints_download(tmp_path):
    with pytest.raises(DataError, match="wbc download-data"):
        discover_classes(tmp_path / "nope")


def test_empty_class_folder_raises(tmp_path, write_image):
    write_image(tmp_path / "A" / "1.png")
    (tmp_path / "B").mkdir()
    with pytest.raises(DataError, match="B"):
        discover_classes(tmp_path)


def test_hidden_dirs_ignored(tmp_path, write_image):
    write_image(tmp_path / "A" / "1.png")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    assert discover_classes(tmp_path) == ["A"]


def test_list_samples_ignores_non_images(tmp_path, write_image):
    write_image(tmp_path / "A" / "1.png")
    (tmp_path / "A" / ".DS_Store").write_bytes(b"x")
    (tmp_path / "A" / "notes.txt").write_text("hi")
    assert list_samples(tmp_path, ["A"]) == [(tmp_path / "A" / "1.png", 0)]


def test_list_samples_missing_class(tmp_path, write_image):
    write_image(tmp_path / "A" / "1.png")
    with pytest.raises(DataError, match="B"):
        list_samples(tmp_path, ["A", "B"])


def test_stratified_split_keeps_all_classes(data_root):
    samples = list_samples(data_root / "TRAIN", CLASSES)
    train, val = stratified_split(samples, 0.25, seed=0)
    assert len(train) + len(val) == len(samples) == 24
    assert {label for _, label in val} == {0, 1, 2, 3}


def test_stratified_split_is_seeded(data_root):
    samples = list_samples(data_root / "TRAIN", CLASSES)
    assert stratified_split(samples, 0.25, 1) == stratified_split(samples, 0.25, 1)


def test_stratified_split_too_small_raises(tmp_path, write_image):
    for cls in "AB":
        write_image(tmp_path / cls / "1.png")  # one image per class
    with pytest.raises(DataError, match="validation split"):
        stratified_split(list_samples(tmp_path, ["A", "B"]), 0.25, 0)


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB"])
def test_load_image_always_rgb(tmp_path, write_image, mode):
    path = write_image(tmp_path / "x.png", mode=mode)
    assert load_image(path).mode == "RGB"


def test_load_image_corrupt_names_file(tmp_path):
    bad = tmp_path / "broken.jpeg"
    bad.write_bytes(b"not an image")
    with pytest.raises(DataError, match="broken.jpeg"):
        load_image(bad)


def test_crop_falls_back_to_full_image(tmp_path, caplog):
    path = tmp_path / "black.png"
    Image.new("RGB", (64, 48)).save(path)  # black: no cell
    with caplog.at_level(logging.WARNING):
        image = load_image(path, crop=True)
    assert image.size == (64, 48)
    assert "No cell found" in caplog.text


def test_transforms_shape_and_dtype(data_root):
    sample = list_samples(data_root / "TRAIN", CLASSES)[0][0]
    for train in (True, False):
        out = build_transforms(32, train)(load_image(sample))
        assert out.shape == (3, 32, 32) and out.dtype == torch.float32


def test_dataset_item(data_root):
    samples = list_samples(data_root / "TRAIN", CLASSES)
    tensor, label = WBCDataset(samples, build_transforms(32, False))[0]
    assert tensor.shape == (3, 32, 32) and isinstance(label, int)


def test_build_loaders(data_root):
    loaders = build_train_val_loaders(_cfg(data_root))
    images, labels = next(iter(loaders.train))
    assert images.shape == (4, 3, 32, 32) and labels.shape == (4,)
    assert loaders.class_names == CLASSES
    test_loader = build_test_loader(_cfg(data_root), loaders.class_names)
    assert len(test_loader.dataset) == 8


def test_install_dataset_from_nested_kaggle_layout(tmp_path, write_image):
    source = tmp_path / "cache" / "dataset2-master" / "images"
    for split in ("TRAIN", "TEST", "TEST_SIMPLE"):
        write_image(source / split / "A" / "1.png")
    assert find_dataset_root(tmp_path / "cache") == source
    dest = tmp_path / "data"
    install_dataset(tmp_path / "cache", dest)
    assert (dest / "TRAIN" / "A" / "1.png").is_file() and (dest / "TEST").is_dir()
    assert not (dest / "TEST_SIMPLE").exists()
    with pytest.raises(DataError, match="--force"):
        install_dataset(tmp_path / "cache", dest)
    install_dataset(tmp_path / "cache", dest, force=True)


def test_find_dataset_root_missing(tmp_path):
    with pytest.raises(DataError, match="TRAIN"):
        find_dataset_root(tmp_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_data.py -v`
Expected: FAIL (modules not found)

- [ ] **Step 3: Implement** — `src/wbc_classification/data/__init__.py`: `"""Datasets, transforms and data download."""`

`labels.py`:

```python
"""Class discovery from the dataset folder layout."""

from pathlib import Path

from wbc_classification.errors import DataError

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def discover_classes(split_dir: Path) -> list[str]:
    """Class names are the sorted sub-folder names of a split directory."""
    if not split_dir.is_dir():
        raise DataError(
            f"Dataset directory not found: {split_dir}. "
            "Run `wbc download-data` or point data.root at your dataset."
        )
    class_dirs = sorted(
        p for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not class_dirs:
        raise DataError(f"No class folders found in {split_dir}")
    for class_dir in class_dirs:
        if not any(is_image_file(f) for f in class_dir.iterdir()):
            raise DataError(f"Class folder {class_dir} contains no images")
    return [p.name for p in class_dirs]
```

`transforms.py`:

```python
"""Train / eval image transforms."""

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> v2.Compose:
    steps: list[v2.Transform] = [v2.Resize((image_size, image_size))]
    if train:
        steps += [
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(180),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(steps)
```

`datasets.py`:

```python
"""Dataset listing, splitting, loading and DataLoader construction."""

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from wbc_classification.config import Config
from wbc_classification.data.labels import discover_classes, is_image_file
from wbc_classification.data.transforms import build_transforms
from wbc_classification.errors import DataError, SegmentationError
from wbc_classification.preprocessing.segmentation import crop_cell

logger = logging.getLogger(__name__)

Sample = tuple[Path, int]


def list_samples(split_dir: Path, class_names: Sequence[str]) -> list[Sample]:
    samples: list[Sample] = []
    for index, name in enumerate(class_names):
        class_dir = split_dir / name
        if not class_dir.is_dir():
            raise DataError(f"Missing class folder {class_dir}")
        files = sorted(p for p in class_dir.iterdir() if is_image_file(p))
        if not files:
            raise DataError(f"Class folder {class_dir} contains no images")
        samples.extend((p, index) for p in files)
    return samples


def stratified_split(
    samples: list[Sample], val_fraction: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    labels = [label for _, label in samples]
    counts = Counter(labels)
    n_classes = len(counts)
    n_val = math.ceil(val_fraction * len(samples))
    n_train = len(samples) - n_val
    if min(counts.values()) < 2 or n_val < n_classes or n_train < n_classes:
        raise DataError(
            f"Not enough images to make a stratified validation split "
            f"({len(samples)} images, {n_classes} classes, val_fraction={val_fraction}); "
            "add more images or raise data.val_fraction."
        )
    train, val = train_test_split(
        samples, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return train, val


def load_image(path: Path, crop: bool = False) -> Image.Image:
    """Load an image as RGB, optionally cropped to the cell (full image if none found)."""
    try:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
    except OSError as exc:
        raise DataError(f"Cannot read image {path}: {exc}") from exc
    if crop:
        try:
            image = Image.fromarray(crop_cell(np.asarray(image)))
        except SegmentationError as exc:
            logger.warning("No cell found in %s (%s); using the full image", path, exc)
    return image


class WBCDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, samples: list[Sample], transform: Callable[[Image.Image], torch.Tensor], crop: bool = False
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.crop = crop

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return self.transform(load_image(path, self.crop)), label


@dataclass(frozen=True)
class Loaders:
    train: DataLoader
    val: DataLoader
    class_names: list[str]


def build_train_val_loaders(cfg: Config) -> Loaders:
    train_dir = cfg.data.root / "TRAIN"
    class_names = discover_classes(train_dir)
    train_samples, val_samples = stratified_split(
        list_samples(train_dir, class_names), cfg.data.val_fraction, cfg.train.seed
    )
    crop = cfg.preprocessing.crop
    train_ds = WBCDataset(train_samples, build_transforms(cfg.data.image_size, True), crop)
    val_ds = WBCDataset(val_samples, build_transforms(cfg.data.image_size, False), crop)
    common = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(cfg.train.seed)
    return Loaders(
        train=DataLoader(train_ds, shuffle=True, generator=generator, **common),
        val=DataLoader(val_ds, shuffle=False, **common),
        class_names=class_names,
    )


def build_test_loader(cfg: Config, class_names: Sequence[str]) -> DataLoader:
    samples = list_samples(cfg.data.root / "TEST", class_names)
    dataset = WBCDataset(
        samples, build_transforms(cfg.data.image_size, False), cfg.preprocessing.crop
    )
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
```

`download.py`:

```python
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
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_data.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add dataset discovery, splits, transforms, loaders and download helper

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Models and registry

**Files:**
- Create: `src/wbc_classification/models/__init__.py`, `baseline.py`, `backbone.py`, `registry.py`; `tests/test_models.py`

**Interfaces:**
- Consumes: `errors.ConfigError`.
- Produces:
  - `baseline.IncludeNet(num_classes: int, width: int = 32, dropout: float = 0.5)` (`nn.Module`; any input size ≥16).
  - `backbone.SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")`, `build_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module`, `set_backbone_frozen(model: nn.Module, frozen: bool) -> None` (only the classification head stays trainable when frozen).
  - `registry.MODEL_NAMES = ("baseline", *SUPPORTED_BACKBONES)`, `build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module` (raises `ConfigError` for unknown), `supports_freezing(name: str) -> bool`.

- [ ] **Step 1: Write the failing tests** — `tests/test_models.py`

```python
import pytest
import torch

from wbc_classification.errors import ConfigError
from wbc_classification.models.backbone import set_backbone_frozen
from wbc_classification.models.registry import MODEL_NAMES, build_model, supports_freezing


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_shape(name):
    model = build_model(name, num_classes=4).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 4)


def test_baseline_accepts_tiny_and_odd_sizes():
    model = build_model("baseline", num_classes=4).eval()
    with torch.no_grad():
        assert model(torch.zeros(1, 3, 16, 16)).shape == (1, 4)
        assert model(torch.zeros(1, 3, 50, 70)).shape == (1, 4)


def test_unknown_model_lists_choices():
    with pytest.raises(ConfigError, match="resnet18"):
        build_model("vgg", num_classes=4)


@pytest.mark.parametrize("name", ["resnet18", "efficientnet_b0"])
def test_freeze_leaves_only_head_trainable(name):
    model = build_model(name, num_classes=4)
    set_backbone_frozen(model, True)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all(n.startswith(("fc", "classifier")) for n in trainable)
    set_backbone_frozen(model, False)
    assert all(p.requires_grad for p in model.parameters())


def test_supports_freezing():
    assert supports_freezing("resnet18") and not supports_freezing("baseline")
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_models.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement** — `models/__init__.py`: `"""Model definitions and registry."""`

`baseline.py`:

```python
"""IncludeNet: the project's original small CNN, rewritten in PyTorch."""

import torch
from torch import nn


def _block(in_channels: int, out_channels: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class IncludeNet(nn.Module):
    """Four conv blocks, global average pooling and a linear head; input-size independent."""

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(3, width, pool=True),
            _block(width, 2 * width, pool=True),
            _block(2 * width, 4 * width, pool=True),
            _block(4 * width, 4 * width, pool=False),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(4 * width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.classifier(self.dropout(x))
```

`backbone.py`:

```python
"""Pretrained torchvision backbones with a replaced classification head."""

from typing import Any, cast

from torch import nn
from torchvision import models as tv_models

SUPPORTED_BACKBONES = ("resnet18", "resnet50", "efficientnet_b0")


def build_backbone(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    model = cast(Any, tv_models.get_model(name, weights="DEFAULT" if pretrained else None))
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return cast(nn.Module, model)


def set_backbone_frozen(model: nn.Module, frozen: bool) -> None:
    """Freeze everything except the classification head (or unfreeze everything)."""
    m = cast(Any, model)
    head = m.fc if hasattr(m, "fc") else m.classifier
    head_ids = {id(p) for p in head.parameters()}
    for param in model.parameters():
        param.requires_grad = (not frozen) or id(param) in head_ids
```

`registry.py`:

```python
"""Name -> model factory."""

from torch import nn

from wbc_classification.errors import ConfigError
from wbc_classification.models.backbone import SUPPORTED_BACKBONES, build_backbone
from wbc_classification.models.baseline import IncludeNet

MODEL_NAMES = ("baseline", *SUPPORTED_BACKBONES)


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    if name == "baseline":
        return IncludeNet(num_classes)  # no pretrained weights exist for the baseline
    if name in SUPPORTED_BACKBONES:
        return build_backbone(name, num_classes, pretrained)
    raise ConfigError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")


def supports_freezing(name: str) -> bool:
    return name in SUPPORTED_BACKBONES
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_models.py -v`
Expected: all PASS (no network needed; `pretrained=False`)

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add PyTorch baseline model, backbones and registry

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Runtime helpers, metrics, checkpoints

**Files:**
- Create: `src/wbc_classification/engine/__init__.py`, `runtime.py`, `metrics.py`, `checkpoint.py`; `tests/test_engine_basics.py`

**Interfaces:**
- Consumes: `Config`, `build_model`, `errors.*`.
- Produces:
  - `runtime.set_seed(seed: int) -> None`; `runtime.resolve_device(name: str) -> torch.device` (`"auto"` → cuda, then mps, then cpu; invalid → `ConfigError`).
  - `metrics.compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> dict` with keys `accuracy: float`, `report: dict`, `confusion_matrix: list[list[int]]`, `class_names: list[str]` (JSON-serialisable); `metrics.save_confusion_matrix(matrix, class_names, path: Path) -> None`; `metrics.save_curves(history: list[dict[str, float]], path: Path) -> None` (history rows have keys `epoch, train_loss, train_acc, val_loss, val_acc, lr`); `metrics.format_report(metrics: dict) -> str`.
  - `checkpoint.save_checkpoint(path: Path, model: nn.Module, class_names: Sequence[str], config: Config, epoch: int, val_accuracy: float) -> None`; `checkpoint.load_checkpoint(path: Path) -> tuple[nn.Module, list[str], Config]` (CPU, eval-mode-agnostic; raises `CheckpointError`).

- [ ] **Step 1: Write the failing tests** — `tests/test_engine_basics.py`

```python
import json

import pytest
import torch

from wbc_classification.config import Config, ModelConfig
from wbc_classification.engine.checkpoint import load_checkpoint, save_checkpoint
from wbc_classification.engine.metrics import (
    compute_metrics,
    format_report,
    save_confusion_matrix,
    save_curves,
)
from wbc_classification.engine.runtime import resolve_device, set_seed
from wbc_classification.errors import CheckpointError, ConfigError
from wbc_classification.models.registry import build_model

NAMES = ["A", "B", "C", "D"]


def test_resolve_device():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    with pytest.raises(ConfigError, match="bogus"):
        resolve_device("bogus")


def test_set_seed_makes_torch_deterministic():
    set_seed(3)
    first = torch.rand(3)
    set_seed(3)
    assert torch.equal(first, torch.rand(3))


def test_compute_metrics_perfect():
    m = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3], NAMES)
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    json.dumps(m)  # must be JSON-serialisable
    assert "A" in format_report(m)


def test_compute_metrics_handles_missing_class_in_predictions():
    m = compute_metrics([0, 0, 1, 1], [0, 0, 0, 0], NAMES)
    assert m["accuracy"] == 0.5 and len(m["confusion_matrix"]) == 4
    json.dumps(m)


def test_plots_are_written(tmp_path):
    save_confusion_matrix([[1, 0], [0, 1]], ["A", "B"], tmp_path / "cm.png")
    history = [
        {"epoch": 1, "train_loss": 1.0, "train_acc": 0.5, "val_loss": 1.1, "val_acc": 0.4, "lr": 0.1},
        {"epoch": 2, "train_loss": 0.8, "train_acc": 0.6, "val_loss": 0.9, "val_acc": 0.5, "lr": 0.05},
    ]
    save_curves(history, tmp_path / "curves.png")
    assert (tmp_path / "cm.png").stat().st_size > 0 and (tmp_path / "curves.png").stat().st_size > 0


def _baseline():
    return build_model("baseline", num_classes=4)


def test_checkpoint_roundtrip(tmp_path):
    model = _baseline().eval()
    cfg = Config(model=ModelConfig(name="baseline", pretrained=False))
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, NAMES, cfg, epoch=3, val_accuracy=0.9)
    loaded, names, loaded_cfg = load_checkpoint(path)
    assert names == NAMES and loaded_cfg == cfg
    x = torch.rand(1, 3, 32, 32)
    assert torch.allclose(model(x), loaded.eval()(x))


def test_checkpoint_missing(tmp_path):
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")


def test_checkpoint_corrupt_file(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"junk")
    with pytest.raises(CheckpointError, match="bad.pt"):
        load_checkpoint(bad)


def test_checkpoint_not_ours(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(CheckpointError, match="not a wbc-classification checkpoint"):
        load_checkpoint(path)


def test_checkpoint_class_count_mismatch(tmp_path):
    cfg = Config(model=ModelConfig(name="baseline", pretrained=False))
    path = tmp_path / "m.pt"
    save_checkpoint(path, _baseline(), ["A", "B", "C"], cfg, epoch=1, val_accuracy=0.1)  # 4 outputs vs 3 names
    with pytest.raises(CheckpointError, match="do not match"):
        load_checkpoint(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_engine_basics.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement** — `engine/__init__.py`: `"""Training, evaluation and inference."""`

`runtime.py`:

```python
"""Seeding and device selection."""

import random

import numpy as np
import torch

from wbc_classification.errors import ConfigError


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    try:
        return torch.device(name)
    except RuntimeError as exc:
        raise ConfigError(f"Unknown device {name!r} (use auto, cpu, cuda, cuda:0, mps)") from exc
```

`metrics.py`:

```python
"""Classification metrics and plots."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict[str, Any]:
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": list(class_names),
    }
    return json.loads(json.dumps(result, default=float))  # plain, JSON-safe types


def format_report(metrics: dict[str, Any]) -> str:
    lines = [f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"]
    for name in metrics["class_names"]:
        row = metrics["report"][name]
        lines.append(
            f"{name:<14}{row['precision']:>10.3f}{row['recall']:>10.3f}"
            f"{row['f1-score']:>10.3f}{int(row['support']):>10}"
        )
    lines.append(f"{'accuracy':<14}{metrics['accuracy']:>40.3f}")
    return "\n".join(lines)


def save_confusion_matrix(matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path) -> None:
    fig = Figure(figsize=(6, 5))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), labels=class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, str(value), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def save_curves(history: Sequence[dict[str, float]], path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig = Figure(figsize=(10, 4))
    loss_ax, acc_ax = fig.subplots(1, 2)
    loss_ax.plot(epochs, [r["train_loss"] for r in history], label="train")
    loss_ax.plot(epochs, [r["val_loss"] for r in history], label="val")
    loss_ax.set(title="Loss", xlabel="Epoch")
    acc_ax.plot(epochs, [r["train_acc"] for r in history], label="train")
    acc_ax.plot(epochs, [r["val_acc"] for r in history], label="val")
    acc_ax.set(title="Accuracy", xlabel="Epoch")
    loss_ax.legend()
    acc_ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
```

`checkpoint.py`:

```python
"""Single-file checkpoints: weights + class names + config."""

import pickle
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from wbc_classification.config import Config
from wbc_classification.errors import CheckpointError, ConfigError
from wbc_classification.models.registry import build_model

_REQUIRED_KEYS = {"state_dict", "class_names", "config"}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: Sequence[str],
    config: Config,
    epoch: int,
    val_accuracy: float,
) -> None:
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "class_names": list(class_names),
            "config": config.model_dump(mode="json"),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
        },
        path,
    )


def load_checkpoint(path: Path) -> tuple[nn.Module, list[str], Config]:
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError, EOFError, OSError) as exc:
        raise CheckpointError(f"Cannot read checkpoint {path}: {exc}") from exc
    if not isinstance(payload, dict) or not _REQUIRED_KEYS <= payload.keys():
        raise CheckpointError(f"{path} is not a wbc-classification checkpoint")
    try:
        config = Config.model_validate(payload["config"])
        class_names = list(payload["class_names"])
        model = build_model(config.model.name, num_classes=len(class_names), pretrained=False)
        model.load_state_dict(payload["state_dict"])
    except ConfigError as exc:
        raise CheckpointError(f"{path} stores an unusable model config: {exc}") from exc
    except ValueError as exc:  # pydantic.ValidationError subclasses ValueError
        raise CheckpointError(f"{path} stores an invalid config: {exc}") from exc
    except RuntimeError as exc:
        raise CheckpointError(
            f"Weights in {path} do not match its stored class names / model ({exc})"
        ) from exc
    return model, class_names, config
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_engine_basics.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add runtime helpers, metrics and checkpoint IO

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Training engine

**Files:**
- Create: `src/wbc_classification/engine/train.py`; `tests/test_train.py`
- Modify: `tests/conftest.py` (add shared tiny-config helpers and a session-scoped trained run)

**Interfaces:**
- Consumes: `Config`, `build_train_val_loaders`, `build_model`, `set_backbone_frozen`, `supports_freezing`, `save_checkpoint`, `save_curves`, `dump_config`, `set_seed`, `resolve_device`.
- Produces: `train.TrainResult(run_dir: Path, best_checkpoint: Path, last_checkpoint: Path, best_val_accuracy: float, history: list[dict[str, float]])` (frozen dataclass); `train.train(cfg: Config) -> TrainResult`. Run dir layout: `<output_dir>/<name>-<YYYYmmdd-HHMMSS>/{config.yaml, metrics.csv, curves.png, best.pt, last.pt}`.
- conftest produces: `tiny_config(out_dir: Path, data_root: Path) -> Config` helper (module-level function `make_tiny_config`), fixture `tiny_config` (function scope), fixture `trained` (session scope → `TrainResult`).

- [ ] **Step 1: Extend `tests/conftest.py`**

Add below the existing code:

```python
from wbc_classification.config import Config, DataConfig, ModelConfig, TrainConfig


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
```

- [ ] **Step 2: Write the failing tests** — `tests/test_train.py`

```python
import csv

import pytest

from wbc_classification.engine.checkpoint import load_checkpoint
from wbc_classification.engine.train import train
from wbc_classification.errors import ConfigError


def test_train_writes_all_artifacts(trained):
    run = trained.run_dir
    for name in ("config.yaml", "metrics.csv", "curves.png", "best.pt", "last.pt"):
        assert (run / name).is_file(), name
    assert len(trained.history) == 1
    assert 0.0 <= trained.best_val_accuracy <= 1.0
    with (run / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["epoch"] == "1" and "val_acc" in rows[0]


def test_trained_checkpoint_loads(trained):
    model, class_names, cfg = load_checkpoint(trained.best_checkpoint)
    assert len(class_names) == 4 and cfg.model.name == "baseline"


def test_freeze_on_baseline_is_rejected(tiny_config):
    tiny_config.model.freeze_backbone_epochs = 1
    with pytest.raises(ConfigError, match="baseline"):
        train(tiny_config)


def test_early_stopping(tiny_config):
    tiny_config.train.epochs = 6
    tiny_config.train.patience = 1
    result = train(tiny_config)
    assert 1 <= len(result.history) <= 6


def test_same_seed_same_first_epoch(tiny_config, tmp_path):
    first = train(tiny_config).history[0]["train_loss"]
    tiny_config.train.output_dir = tmp_path / "again"
    assert train(tiny_config).history[0]["train_loss"] == pytest.approx(first, rel=1e-4)
```

- [ ] **Step 3: Run to verify failure**

Run: `poetry run pytest tests/test_train.py -v`
Expected: FAIL (module not found)

- [ ] **Step 4: Implement** — `engine/train.py`

```python
"""Training loop."""

import csv
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wbc_classification.config import Config, dump_config
from wbc_classification.data.datasets import build_train_val_loaders
from wbc_classification.engine.checkpoint import save_checkpoint
from wbc_classification.engine.metrics import save_curves
from wbc_classification.engine.runtime import resolve_device, set_seed
from wbc_classification.errors import ConfigError
from wbc_classification.models.backbone import set_backbone_frozen
from wbc_classification.models.registry import build_model, supports_freezing

logger = logging.getLogger(__name__)

HISTORY_FIELDS = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]


@dataclass(frozen=True)
class TrainResult:
    run_dir: Path
    best_checkpoint: Path
    last_checkpoint: Path
    best_val_accuracy: float
    history: list[dict[str, float]]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if optimizer is not None and scaler is not None:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def train(cfg: Config) -> TrainResult:
    if cfg.model.freeze_backbone_epochs and not supports_freezing(cfg.model.name):
        raise ConfigError(
            f"model.freeze_backbone_epochs is not supported for {cfg.model.name!r} "
            "(it has no pretrained backbone)"
        )
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    loaders = build_train_val_loaders(cfg)
    model = build_model(cfg.model.name, len(loaders.class_names), cfg.model.pretrained).to(device)

    run_dir = cfg.train.output_dir / f"{cfg.train.name}-{datetime.now():%Y%m%d-%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(dump_config(cfg))

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    best_acc, bad_epochs = -1.0, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.train.epochs + 1):
        if cfg.model.freeze_backbone_epochs:
            set_backbone_frozen(model, epoch <= cfg.model.freeze_backbone_epochs)
        train_loss, train_acc = _run_epoch(
            model, loaders.train, device, criterion, optimizer, scaler, amp
        )
        val_loss, val_acc = _run_epoch(model, loaders.val, device, criterion, amp=amp)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        scheduler.step()
        logger.info(
            "epoch %d/%d train_loss=%.4f train_acc=%.3f val_loss=%.4f val_acc=%.3f",
            epoch, cfg.train.epochs, train_loss, train_acc, val_loss, val_acc,
        )
        save_checkpoint(last_path, model, loaders.class_names, cfg, epoch, val_acc)
        if val_acc > best_acc:
            best_acc, bad_epochs = val_acc, 0
            save_checkpoint(best_path, model, loaders.class_names, cfg, epoch, val_acc)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    with (run_dir / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    save_curves(history, run_dir / "curves.png")
    return TrainResult(run_dir, best_path, last_path, best_acc, history)
```

- [ ] **Step 5: Run tests**

Run: `poetry run pytest tests/test_train.py -v`
Expected: all PASS (each training run takes a few seconds on CPU). If `test_same_seed_same_first_epoch` is flaky due to nondeterministic CPU ops, relax `rel` to `1e-2` and say so in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src tests
git commit -m "feat: add training loop with run artifacts and early stopping

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Evaluation and prediction

**Files:**
- Create: `src/wbc_classification/engine/evaluate.py`, `src/wbc_classification/engine/predict.py`; `tests/test_evaluate_predict.py`

**Interfaces:**
- Consumes: `load_checkpoint`, `build_test_loader`, `compute_metrics`, `save_confusion_matrix`, `resolve_device`, `build_transforms`, `load_image`, `is_image_file`, `DataError`.
- Produces:
  - `evaluate.evaluate_checkpoint(checkpoint: Path, data_root: Path | None = None, device: str = "auto", output: Path | None = None) -> dict` (the `compute_metrics` dict; if `output` is given writes `<output>` JSON and `<output stem>_confusion.png` next to it).
  - `predict.Prediction(path: Path, label: str, probabilities: dict[str, float])` (frozen dataclass); `predict.collect_images(inputs: Sequence[Path]) -> list[Path]`; `predict.predict_images(checkpoint: Path, inputs: Sequence[Path], device: str = "auto") -> list[Prediction]`; `predict.annotate_image(prediction: Prediction, output_dir: Path) -> Path`.

- [ ] **Step 1: Write the failing tests** — `tests/test_evaluate_predict.py`

```python
import json

import pytest

from wbc_classification.engine.evaluate import evaluate_checkpoint
from wbc_classification.engine.predict import annotate_image, collect_images, predict_images
from wbc_classification.errors import CheckpointError, DataError


def test_evaluate_reports_per_class_and_writes_outputs(trained, data_root, tmp_path):
    out = tmp_path / "metrics.json"
    metrics = evaluate_checkpoint(trained.best_checkpoint, data_root=data_root, device="cpu", output=out)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert set(metrics["class_names"]) == {"EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"}
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 8  # the 8 TEST fixtures
    assert json.loads(out.read_text())["accuracy"] == metrics["accuracy"]
    assert (tmp_path / "metrics_confusion.png").is_file()


def test_evaluate_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointError):
        evaluate_checkpoint(tmp_path / "nope.pt")


def test_evaluate_missing_data_dir(trained, tmp_path):
    with pytest.raises(DataError, match="wbc download-data"):
        evaluate_checkpoint(trained.best_checkpoint, data_root=tmp_path, device="cpu")


def test_predict_directory(trained, data_root):
    preds = predict_images(trained.best_checkpoint, [data_root / "TEST"], device="cpu")
    assert len(preds) == 8
    for pred in preds:
        assert pred.label in pred.probabilities
        assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)


def test_predict_single_file(trained, cell_image):
    (pred,) = predict_images(trained.best_checkpoint, [cell_image], device="cpu")
    assert pred.path == cell_image


def test_predict_empty_dir_raises(trained, tmp_path):
    with pytest.raises(DataError, match="No images"):
        predict_images(trained.best_checkpoint, [tmp_path], device="cpu")


def test_predict_missing_path_raises(trained, tmp_path):
    with pytest.raises(DataError, match="not found"):
        predict_images(trained.best_checkpoint, [tmp_path / "ghost.jpg"], device="cpu")


def test_predict_corrupt_image_raises(trained, tmp_path):
    bad = tmp_path / "bad.jpeg"
    bad.write_bytes(b"nope")
    with pytest.raises(DataError, match="bad.jpeg"):
        predict_images(trained.best_checkpoint, [bad], device="cpu")


def test_collect_images_ignores_non_images(tmp_path, write_image):
    write_image(tmp_path / "a.png")
    (tmp_path / "notes.txt").write_text("x")
    assert collect_images([tmp_path]) == [tmp_path / "a.png"]


def test_annotate_writes_file(trained, cell_image, tmp_path):
    (pred,) = predict_images(trained.best_checkpoint, [cell_image], device="cpu")
    assert annotate_image(pred, tmp_path).is_file()
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_evaluate_predict.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement** — `engine/evaluate.py`

```python
"""Evaluate a checkpoint on the TEST split."""

import json
from pathlib import Path
from typing import Any

import torch

from wbc_classification.data.datasets import build_test_loader
from wbc_classification.engine.checkpoint import load_checkpoint
from wbc_classification.engine.metrics import compute_metrics, save_confusion_matrix
from wbc_classification.engine.runtime import resolve_device


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path | None = None,
    device: str = "auto",
    output: Path | None = None,
) -> dict[str, Any]:
    model, class_names, cfg = load_checkpoint(checkpoint)
    if data_root is not None:
        cfg.data.root = data_root
    torch_device = resolve_device(device)
    model.to(torch_device).eval()
    loader = build_test_loader(cfg, class_names)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(torch_device))
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_true.extend(labels.tolist())

    metrics = compute_metrics(y_true, y_pred, class_names)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2))
        save_confusion_matrix(
            metrics["confusion_matrix"], class_names, output.with_name(f"{output.stem}_confusion.png")
        )
    return metrics
```

`engine/predict.py`:

```python
"""Predict classes for images or folders of images."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import ImageDraw

from wbc_classification.data.datasets import load_image
from wbc_classification.data.labels import is_image_file
from wbc_classification.data.transforms import build_transforms
from wbc_classification.engine.checkpoint import load_checkpoint
from wbc_classification.engine.runtime import resolve_device
from wbc_classification.errors import DataError


@dataclass(frozen=True)
class Prediction:
    path: Path
    label: str
    probabilities: dict[str, float]


def collect_images(inputs: Sequence[Path]) -> list[Path]:
    images: list[Path] = []
    for item in inputs:
        if item.is_dir():
            images.extend(sorted(p for p in item.rglob("*") if is_image_file(p)))
        elif item.is_file():
            images.append(item)
        else:
            raise DataError(f"Path not found: {item}")
    if not images:
        raise DataError("No images found in the given inputs")
    return images


def predict_images(
    checkpoint: Path, inputs: Sequence[Path], device: str = "auto"
) -> list[Prediction]:
    model, class_names, cfg = load_checkpoint(checkpoint)
    torch_device = resolve_device(device)
    model.to(torch_device).eval()
    transform = build_transforms(cfg.data.image_size, train=False)

    predictions: list[Prediction] = []
    with torch.no_grad():
        for path in collect_images(inputs):
            image = load_image(path, crop=cfg.preprocessing.crop)
            batch = transform(image).unsqueeze(0).to(torch_device)
            probs = torch.softmax(model(batch), dim=1)[0].cpu().tolist()
            best = max(range(len(class_names)), key=probs.__getitem__)
            predictions.append(
                Prediction(path, class_names[best], dict(zip(class_names, probs, strict=True)))
            )
    return predictions


def annotate_image(prediction: Prediction, output_dir: Path) -> Path:
    """Write a copy of the image with the predicted label drawn on it."""
    image = load_image(prediction.path)
    confidence = prediction.probabilities[prediction.label]
    ImageDraw.Draw(image).text((8, 8), f"{prediction.label} {confidence:.0%}", fill=(0, 255, 0))
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{prediction.path.stem}_pred.png"
    image.save(out)
    return out
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_evaluate_predict.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add checkpoint evaluation and prediction

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 10: CLI

**Files:**
- Create: `src/wbc_classification/cli.py`, `src/wbc_classification/__main__.py`; `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces: Typer `app` with commands `train`, `evaluate`, `predict`, `segment`, `download-data`; any `WBCError` → `Error: <message>` on stderr, exit code 1. `python -m wbc_classification` runs the app.
  - `wbc train [-c CONFIG] [--model M] [--epochs N] [--batch-size N] [--image-size N] [--lr X] [--data-root P] [--output-dir P] [--device D] [--name S] [--pretrained/--no-pretrained] [--crop/--no-crop]`
  - `wbc evaluate CHECKPOINT [--data-root P] [--device D] [--output FILE.json]`
  - `wbc predict INPUT... -c CHECKPOINT [--device D] [--annotate-dir DIR]`
  - `wbc segment IMAGE [--output-dir DIR]` writes `<stem>_crop.png`, `<stem>_overlay.png`, `<stem>_mask.png`
  - `wbc download-data [--dest DIR] [--force]`

- [ ] **Step 1: Write the failing tests** — `tests/test_cli.py`

```python
from typer.testing import CliRunner

from wbc_classification.cli import app

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "evaluate", "predict", "segment", "download-data"):
        assert command in result.output


def test_train_command(data_root, tmp_path):
    result = runner.invoke(
        app,
        [
            "train", "--data-root", str(data_root), "--output-dir", str(tmp_path),
            "--model", "baseline", "--no-pretrained", "--epochs", "1", "--batch-size", "4",
            "--image-size", "32", "--device", "cpu", "--name", "cli",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "best validation accuracy" in result.output.lower()
    assert list(tmp_path.glob("cli-*/best.pt"))


def test_evaluate_command(trained, data_root, tmp_path):
    out = tmp_path / "m.json"
    result = runner.invoke(
        app, ["evaluate", str(trained.best_checkpoint), "--data-root", str(data_root),
              "--device", "cpu", "--output", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "accuracy" in result.output and out.is_file()


def test_predict_command(trained, cell_image, tmp_path):
    result = runner.invoke(
        app, ["predict", str(cell_image), "-c", str(trained.best_checkpoint),
              "--device", "cpu", "--annotate-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "cell.jpeg" in result.output
    assert (tmp_path / "cell_pred.png").is_file()


def test_segment_command(cell_image, tmp_path):
    result = runner.invoke(app, ["segment", str(cell_image), "--output-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    for suffix in ("crop", "overlay", "mask"):
        assert (tmp_path / f"cell_{suffix}.png").is_file()


def test_segment_no_cell_is_a_clean_error(tmp_path, write_image):
    black = write_image(tmp_path / "black.png")
    result = runner.invoke(app, ["segment", str(black), "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error:" in result.output and "Traceback" not in result.output


def test_missing_checkpoint_is_a_clean_error(tmp_path):
    result = runner.invoke(app, ["evaluate", str(tmp_path / "nope.pt")])
    assert result.exit_code == 1 and "Checkpoint not found" in result.output


def test_bad_config_is_a_clean_error(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("train:\n  epohcs: 2\n")
    result = runner.invoke(app, ["train", "-c", str(cfg)])
    assert result.exit_code == 1 and "epohcs" in result.output
```

(Typer's `CliRunner` mixes stderr into `result.output` by default; if the installed Typer separates them, use `result.output` + `result.stderr` or construct `CliRunner(mix_stderr=...)` per its current API.)

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_cli.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement** — `src/wbc_classification/cli.py`

```python
"""`wbc` command-line interface."""

import functools
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from PIL import Image

from wbc_classification import __version__
from wbc_classification.config import load_config
from wbc_classification.data.datasets import load_image
from wbc_classification.data.download import download_dataset
from wbc_classification.engine.evaluate import evaluate_checkpoint
from wbc_classification.engine.metrics import format_report
from wbc_classification.engine.predict import annotate_image, predict_images
from wbc_classification.engine.train import train as run_training
from wbc_classification.errors import WBCError
from wbc_classification.preprocessing.segmentation import crop_cell, draw_overlay, segment_cell

app = typer.Typer(
    help="Classify white blood cell images (eosinophil, lymphocyte, monocyte, neutrophil).",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log progress per epoch.")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(message)s")


def _handle_errors(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except WBCError as exc:
            typer.secho(f"Error: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1) from exc

    return wrapper


@app.command()
@_handle_errors
def train(
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file.")] = None,
    model: Annotated[str | None, typer.Option(help="baseline, resnet18, resnet50, efficientnet_b0.")] = None,
    epochs: Annotated[int | None, typer.Option(help="Number of epochs.")] = None,
    batch_size: Annotated[int | None, typer.Option(help="Batch size.")] = None,
    image_size: Annotated[int | None, typer.Option(help="Square input size in pixels.")] = None,
    lr: Annotated[float | None, typer.Option(help="Learning rate.")] = None,
    data_root: Annotated[Path | None, typer.Option(help="Folder containing TRAIN/ and TEST/.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Where run folders are written.")] = None,
    device: Annotated[str | None, typer.Option(help="auto, cpu, cuda, mps.")] = None,
    name: Annotated[str | None, typer.Option(help="Run name prefix.")] = None,
    pretrained: Annotated[bool | None, typer.Option("--pretrained/--no-pretrained")] = None,
    crop: Annotated[bool | None, typer.Option("--crop/--no-crop", help="Crop to the cell first.")] = None,
) -> None:
    """Train a model and write a run folder with checkpoints, metrics and plots."""
    cfg = load_config(
        config,
        {
            "model.name": model,
            "train.epochs": epochs,
            "data.batch_size": batch_size,
            "data.image_size": image_size,
            "train.lr": lr,
            "data.root": data_root,
            "train.output_dir": output_dir,
            "train.device": device,
            "train.name": name,
            "model.pretrained": pretrained,
            "preprocessing.crop": crop,
        },
    )
    result = run_training(cfg)
    typer.echo(f"Best validation accuracy: {result.best_val_accuracy:.4f}")
    typer.echo(f"Run folder: {result.run_dir}")
    typer.echo(f"Best checkpoint: {result.best_checkpoint}")


@app.command()
@_handle_errors
def evaluate(
    checkpoint: Annotated[Path, typer.Argument(help="Checkpoint (.pt) written by `wbc train`.")],
    data_root: Annotated[Path | None, typer.Option(help="Folder containing TEST/.")] = None,
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, mps.")] = "auto",
    output: Annotated[Path | None, typer.Option(help="Write metrics JSON (+ confusion PNG) here.")] = None,
) -> None:
    """Evaluate a checkpoint on the TEST split."""
    metrics = evaluate_checkpoint(checkpoint, data_root, device, output)
    typer.echo(format_report(metrics))


@app.command()
@_handle_errors
def predict(
    inputs: Annotated[list[Path], typer.Argument(help="Image files and/or folders of images.")],
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-c", help="Checkpoint (.pt).")],
    device: Annotated[str, typer.Option(help="auto, cpu, cuda, mps.")] = "auto",
    annotate_dir: Annotated[Path | None, typer.Option(help="Save labelled copies of the images here.")] = None,
) -> None:
    """Predict the cell type of one or more images."""
    for prediction in predict_images(checkpoint, inputs, device):
        confidence = prediction.probabilities[prediction.label]
        typer.echo(f"{prediction.path}: {prediction.label} ({confidence:.1%})")
        if annotate_dir is not None:
            annotate_image(prediction, annotate_dir)


@app.command()
@_handle_errors
def segment(
    image: Annotated[Path, typer.Argument(help="Image to segment.")],
    output_dir: Annotated[Path, typer.Option(help="Where to write crop/overlay/mask PNGs.")] = Path("segmentation"),
) -> None:
    """Locate the cell with HSV segmentation and save the crop, overlay and mask."""
    rgb = np.asarray(load_image(image))
    segmentation = segment_cell(rgb)
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(crop_cell(rgb)).save(output_dir / f"{image.stem}_crop.png")
    Image.fromarray(draw_overlay(rgb, segmentation)).save(output_dir / f"{image.stem}_overlay.png")
    Image.fromarray(segmentation.mask).save(output_dir / f"{image.stem}_mask.png")
    typer.echo(f"Saved crop, overlay and mask to {output_dir}")


@app.command("download-data")
@_handle_errors
def download_data(
    dest: Annotated[Path, typer.Option(help="Destination folder.")] = Path("data"),
    force: Annotated[bool, typer.Option(help="Overwrite existing TRAIN/TEST folders.")] = False,
) -> None:
    """Download the Kaggle blood-cells dataset into DEST (needs the `kaggle` extra)."""
    path = download_dataset(dest, force)
    typer.echo(f"Dataset installed in {path}")
```

`src/wbc_classification/__main__.py`:

```python
from wbc_classification.cli import app

app()
```

- [ ] **Step 4: Run tests**

Run: `poetry run pytest tests/test_cli.py -v && poetry run wbc --version && poetry run wbc --help`
Expected: tests PASS; version prints; help lists the five commands.

- [ ] **Step 5: Commit**

```bash
git add src tests
git commit -m "feat: add wbc command-line interface

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Experiment configs and end-to-end smoke test

**Files:**
- Create: `configs/baseline.yaml`, `configs/resnet18.yaml`, `configs/efficientnet_b0.yaml`, `tests/test_configs_and_smoke.py`

**Interfaces:**
- Consumes: `load_config`, `app`, `trained` fixture.
- Produces: three experiment YAMLs that load cleanly; one end-to-end smoke test (train → evaluate → predict via the CLI, through a reloaded checkpoint).

- [ ] **Step 1: Write the configs**

`configs/baseline.yaml`:

```yaml
# Lightweight baseline: the original IncludeNet idea, rewritten with BatchNorm + GAP.
preprocessing:
  crop: false
data:
  root: data
  image_size: 64
  batch_size: 64
  val_fraction: 0.2
model:
  name: baseline
  pretrained: false
train:
  name: baseline
  epochs: 60
  lr: 0.001
  patience: 10
```

`configs/resnet18.yaml`:

```yaml
# Fine-tune an ImageNet-pretrained ResNet-18.
preprocessing:
  crop: false
data:
  root: data
  image_size: 128
  batch_size: 32
model:
  name: resnet18
  pretrained: true
  freeze_backbone_epochs: 2
train:
  name: resnet18
  epochs: 25
  lr: 0.0005
  patience: 6
```

`configs/efficientnet_b0.yaml`:

```yaml
# Fine-tune an ImageNet-pretrained EfficientNet-B0.
preprocessing:
  crop: false
data:
  root: data
  image_size: 160
  batch_size: 32
model:
  name: efficientnet_b0
  pretrained: true
  freeze_backbone_epochs: 2
train:
  name: efficientnet_b0
  epochs: 25
  lr: 0.0005
  patience: 6
```

- [ ] **Step 2: Write the tests** — `tests/test_configs_and_smoke.py`

```python
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wbc_classification.cli import app
from wbc_classification.config import load_config
from wbc_classification.models.registry import MODEL_NAMES

CONFIGS = sorted((Path(__file__).parent.parent / "configs").glob("*.yaml"))


def test_configs_exist():
    assert len(CONFIGS) >= 3


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_shipped_configs_are_valid(path):
    cfg = load_config(path)
    assert cfg.model.name in MODEL_NAMES


def test_end_to_end_smoke(data_root, tmp_path):
    """train -> evaluate -> predict through the CLI, reloading the saved checkpoint."""
    runner = CliRunner()
    train = runner.invoke(
        app,
        ["train", "-c", str(CONFIGS[0].parent / "baseline.yaml"), "--data-root", str(data_root),
         "--output-dir", str(tmp_path), "--epochs", "1", "--batch-size", "4",
         "--image-size", "32", "--device", "cpu"],
    )
    assert train.exit_code == 0, train.output
    (checkpoint,) = tmp_path.glob("baseline-*/best.pt")
    evaluate = runner.invoke(
        app, ["evaluate", str(checkpoint), "--data-root", str(data_root), "--device", "cpu"]
    )
    assert evaluate.exit_code == 0, evaluate.output
    predict = runner.invoke(
        app, ["predict", str(data_root / "TEST"), "-c", str(checkpoint), "--device", "cpu"]
    )
    assert predict.exit_code == 0, predict.output
    assert predict.output.count("(") == 8  # one confidence per TEST image
```

- [ ] **Step 3: Run**

Run: `poetry run pytest -q`
Expected: the whole suite passes. Also time it: `poetry run pytest -q --durations=5` — total should be well under two minutes on CPU.

- [ ] **Step 4: Commit**

```bash
git add configs tests
git commit -m "feat: add experiment configs and end-to-end smoke test

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 12: CI workflows

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/workflows/pr-title.yml`

**Interfaces:**
- Produces: CI jobs `lint`, `test` (matrix 3.12/3.13/3.14), `docs` (builds the docs once Task 13 exists) triggered on PRs and pushes to `main`; PR-title check.

- [ ] **Step 1: Check current action major versions**

Run:

```bash
for r in actions/checkout actions/setup-python actions/upload-pages-artifact actions/deploy-pages amannn/action-semantic-pull-request python-semantic-release/python-semantic-release python-semantic-release/publish-action pypa/gh-action-pypi-publish; do printf '%s ' $r; gh api repos/$r/releases/latest --jq .tag_name; done
```
Use the major tag of each result below instead of the versions written here if they differ. (`pypa/gh-action-pypi-publish` is conventionally pinned to `release/v1`.)

- [ ] **Step 2: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - run: pipx install poetry
      - uses: actions/setup-python@v6
        with:
          python-version: "3.14"
          cache: poetry
      - run: poetry install --with dev
      - run: poetry run ruff check .
      - run: poetry run ruff format --check .
      - run: poetry run mypy src

  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.12", "3.13", "3.14"]
    steps:
      - uses: actions/checkout@v5
      - run: pipx install poetry
      - uses: actions/setup-python@v6
        with:
          python-version: ${{ matrix.python-version }}
          cache: poetry
      - run: poetry install --with dev
      - run: poetry run pytest -q

  docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - run: pipx install poetry
      - uses: actions/setup-python@v6
        with:
          python-version: "3.14"
          cache: poetry
      - run: poetry install --with docs
      - run: poetry run typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md
      - run: poetry run mkdocs build --strict
```

- [ ] **Step 3: Write `.github/workflows/pr-title.yml`**

```yaml
name: PR title

on:
  pull_request:
    types: [opened, edited, synchronize, reopened]

permissions:
  pull-requests: read

jobs:
  conventional-title:
    runs-on: ubuntu-latest
    steps:
      - uses: amannn/action-semantic-pull-request@v6
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 4: Validate the YAML locally**

Run: `poetry run python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('ok')"`
Expected: `ok`. (The workflows themselves can only be exercised on GitHub after the branch is pushed; the `docs` job will pass once Task 13 is done.)

- [ ] **Step 5: Commit**

```bash
git add .github
git commit -m "ci: add lint, test matrix, docs build and PR-title workflows

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 13: MkDocs documentation site

**Files:**
- Create: `mkdocs.yml`, `docs/index.md`, `docs/getting-started.md`, `docs/data.md`, `docs/training.md`, `docs/configuration.md`, `docs/results.md`, `docs/architecture.md`, `docs/wbc-detection.md`, `docs/contributing.md`, `docs/changelog.md`, `docs/reference/api.md`, `docs/assets/detection/{blur,hsv,color-filtering,mask,final}.png` (moved)
- Create: `.github/workflows/docs.yml`

**Interfaces:**
- Produces: a site that builds with `poetry run mkdocs build --strict` (after generating `docs/reference/cli.md`), and a Pages deploy workflow.

- [ ] **Step 1: Move the detection screenshots**

```bash
mkdir -p docs/assets/detection docs/reference
git mv "WBC-Detection/blur.png" docs/assets/detection/blur.png
git mv "WBC-Detection/hsv.png" docs/assets/detection/hsv.png
git mv "WBC-Detection/color filtering.png" docs/assets/detection/color-filtering.png
git mv "WBC-Detection/mask.png" docs/assets/detection/mask.png
git mv "WBC-Detection/final.png" docs/assets/detection/final.png
```

- [ ] **Step 2: Write `mkdocs.yml`**

```yaml
site_name: WBC Classification
site_description: Classify white blood cell images with PyTorch.
site_url: https://includeamin.github.io/WBC-Classification/
repo_url: https://github.com/includeamin/WBC-Classification
repo_name: includeamin/WBC-Classification

theme:
  name: material
  features:
    - navigation.sections
    - content.code.copy
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      toggle: {icon: material/weather-night, name: Switch to dark mode}
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      toggle: {icon: material/weather-sunny, name: Switch to light mode}

nav:
  - Home: index.md
  - Getting started: getting-started.md
  - Guides:
      - Data: data.md
      - Training: training.md
      - Configuration: configuration.md
      - WBC detection walkthrough: wbc-detection.md
  - Results: results.md
  - Reference:
      - CLI: reference/cli.md
      - Python API: reference/api.md
      - Architecture: architecture.md
  - Contributing: contributing.md
  - Changelog: changelog.md

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [src]

markdown_extensions:
  - admonition
  - toc:
      permalink: true
  - pymdownx.highlight
  - pymdownx.superfences
  - pymdownx.snippets:
      base_path: ["."]
```

If `pymdownx` is not importable (it ships with mkdocs-material), run `poetry add --group docs pymdown-extensions`.

- [ ] **Step 3: Write the pages**

`docs/index.md`:

```markdown
# WBC Classification

Classify white blood cell images into four types — **eosinophil, lymphocyte, monocyte, neutrophil** — with PyTorch.

- Fine-tune a pretrained backbone (ResNet, EfficientNet) or train the small `baseline` CNN (IncludeNet).
- Config-driven, reproducible training: every run saves its config, metrics, plots and checkpoints.
- An optional OpenCV step crops the cell before classification.
- One command-line tool: `wbc train | evaluate | predict | segment | download-data`.

Start with [Getting started](getting-started.md).
```

`docs/getting-started.md`:

````markdown
# Getting started

## Install

Requires Python 3.12+ and [Poetry](https://python-poetry.org/).

```bash
git clone https://github.com/includeamin/WBC-Classification.git
cd WBC-Classification
poetry install --extras kaggle
```

## Get the data

```bash
poetry run wbc download-data --dest data
```

See [Data](data.md) for the expected layout and Kaggle credentials.

## Train, evaluate, predict

```bash
poetry run wbc train -c configs/resnet18.yaml
poetry run wbc evaluate runs/resnet18-<timestamp>/best.pt --output metrics.json
poetry run wbc predict path/to/cell.jpeg -c runs/resnet18-<timestamp>/best.pt
```

Use `poetry run wbc --help` or the [CLI reference](reference/cli.md) for all options.
Training on a CPU is slow for the pretrained backbones; see [Training](training.md) for GPU and Colab.
````

`docs/data.md`:

```markdown
# Data

The project uses the public Kaggle
[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset
(320×240 JPEG, four classes). It is not stored in git.

```bash
poetry run wbc download-data --dest data          # add --force to overwrite
```

Expected layout:

    data/
    ├── TRAIN/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg
    └── TEST/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/*.jpeg

- Class names come from the folder names; non-image files are ignored.
- A stratified, seeded validation split is carved from `TRAIN`. `TEST` is only used by `wbc evaluate`.
- If Kaggle asks for authentication, create an API token and set `KAGGLE_USERNAME` / `KAGGLE_KEY`
  (or use `~/.kaggle/kaggle.json`). You can also download the dataset manually and copy `TRAIN/` and `TEST/` into `data/`.
```

`docs/training.md`:

````markdown
# Training

```bash
poetry run wbc train -c configs/resnet18.yaml            # pretrained ResNet-18
poetry run wbc train -c configs/baseline.yaml            # small baseline CNN
poetry run wbc train -c configs/resnet18.yaml --epochs 5 --device cpu
poetry run wbc -v train -c configs/resnet18.yaml         # log every epoch
```

Each run writes `runs/<name>-<timestamp>/`:

| File | Content |
|---|---|
| `config.yaml` | the exact configuration used |
| `metrics.csv` | per-epoch loss / accuracy / learning rate |
| `curves.png` | training curves |
| `best.pt` / `last.pt` | checkpoints (weights + class names + config) |

Training uses AdamW, a cosine learning-rate schedule, mixed precision on CUDA and early stopping on validation accuracy.
Set `model.freeze_backbone_epochs` to train only the new classification head for the first N epochs.

## GPU

`poetry install` installs the default PyTorch build. For a specific CUDA build follow the
[PyTorch install guide](https://pytorch.org/get-started/locally/) and install that wheel into the Poetry environment.
On Google Colab use `notebooks/train_colab.ipynb`.
Training on CPU is fine for the `baseline` model and for smoke tests, but slow for pretrained backbones.
````

`docs/configuration.md`:

````markdown
# Configuration

Experiments are described by a YAML file (see `configs/`). Unknown keys are rejected so typos fail fast.
Any value can be overridden on the command line (`wbc train -c cfg.yaml --epochs 5 --lr 0.0003`).

```yaml
preprocessing:
  crop: false            # crop to the detected cell first (HSV segmentation)
data:
  root: data             # folder with TRAIN/ and TEST/
  image_size: 224        # square input size
  batch_size: 32
  val_fraction: 0.2      # stratified split from TRAIN, 0 < x < 0.5
  num_workers: 0
model:
  name: resnet18         # baseline | resnet18 | resnet50 | efficientnet_b0
  pretrained: true       # ImageNet weights (backbones only)
  freeze_backbone_epochs: 0
train:
  name: run              # run folder prefix
  epochs: 30
  lr: 0.001
  weight_decay: 0.0001
  patience: 7            # early-stopping patience (epochs)
  amp: true              # mixed precision (CUDA only)
  seed: 42
  device: auto           # auto | cpu | cuda | mps
  output_dir: runs
```
````

`docs/results.md`:

```markdown
# Results

!!! note
    No pretrained checkpoints or benchmark numbers are published yet. Train a model with
    `wbc train` and `wbc evaluate`, then fill in the table below.

| Model | Image size | Crop | Epochs | TEST accuracy | Macro F1 | Notes |
|---|---|---|---|---|---|---|
| baseline | 64 | no | – | – | – | – |
| resnet18 | 128 | no | – | – | – | – |
| resnet18 | 128 | yes | – | – | – | crop vs no-crop comparison |
| efficientnet_b0 | 160 | no | – | – | – | – |
```

`docs/architecture.md`:

```markdown
# Architecture

```
wbc_classification/
├── config.py          pydantic config + YAML loading + overrides
├── data/              labels, datasets/loaders, transforms, Kaggle download
├── preprocessing/     optional HSV cell segmentation and crop
├── models/            baseline (IncludeNet), torchvision backbones, registry
├── engine/            train, evaluate, predict, metrics, checkpoints, runtime
└── cli.py             `wbc` Typer application
```

Design notes:

- A checkpoint is one `.pt` file holding weights, class names and the full config, so `predict` and `evaluate` need nothing else.
- Errors at the boundaries (config, data, checkpoint) are `WBCError` subclasses; the CLI prints them without a traceback.
- Adding a model means adding an entry in `models/registry.py`.
- The cell crop is optional; when no cell is found the full image is used and a warning is logged.
```

`docs/wbc-detection.md`:

```markdown
# WBC detection walkthrough

The optional crop step locates the white blood cell with classical computer vision:

1. Blur the RGB image (Gaussian, 7×7) — ![blur](assets/detection/blur.png)
2. Convert to HSV — ![hsv](assets/detection/hsv.png)
3. Keep pixels inside the cell colour range (H 80–255, S 60–255, V 140–255) — ![color filtering](assets/detection/color-filtering.png)
4. Take the largest contour as the cell mask — ![mask](assets/detection/mask.png)
5. Crop around the minimum enclosing circle — ![result](assets/detection/final.png)

Try it on your own image:

```bash
poetry run wbc segment path/to/cell.jpeg --output-dir segmentation
```

or use the code directly: `wbc_classification.preprocessing.segmentation.segment_cell`.
```

`docs/contributing.md`: contents `--8<-- "CONTRIBUTING.md"` (created in Task 14; build the docs after Task 14 or create a stub now).
`docs/changelog.md`: contents `--8<-- "CHANGELOG.md"`.

`docs/reference/api.md`:

```markdown
# Python API

::: wbc_classification.config
::: wbc_classification.data.datasets
::: wbc_classification.preprocessing.segmentation
::: wbc_classification.models.registry
::: wbc_classification.engine.train
::: wbc_classification.engine.evaluate
::: wbc_classification.engine.predict
```

- [ ] **Step 4: Write `.github/workflows/docs.yml`**

```yaml
name: Docs

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  deploy:
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/checkout@v5
      - run: pipx install poetry
      - uses: actions/setup-python@v6
        with:
          python-version: "3.14"
          cache: poetry
      - run: poetry install --with docs
      - run: poetry run typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md
      - run: poetry run mkdocs build --strict
      - uses: actions/upload-pages-artifact@v4
        with:
          path: site
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 5: Stub CONTRIBUTING so the build can run, then build**

Create a one-line `CONTRIBUTING.md` (`# Contributing`) if Task 14 hasn't run yet.

Run:

```bash
poetry run typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md
poetry run mkdocs build --strict
```
Expected: build succeeds with no warnings. Fix any broken link / mkdocstrings error it reports (strict mode turns warnings into failures).

- [ ] **Step 6: Commit**

```bash
git add mkdocs.yml docs .github CONTRIBUTING.md pyproject.toml poetry.lock
git commit -m "docs: add MkDocs Material site and Pages deploy workflow

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 14: README, CONTRIBUTING, notebooks

**Files:**
- Modify: `README.md` (full rewrite), `CONTRIBUTING.md` (replace the stub)
- Create: `notebooks/train_colab.ipynb`, `notebooks/wbc_detection.ipynb`

**Interfaces:**
- Produces: a short README that points to the docs; contributor guide describing Poetry workflow and Conventional Commits; two runnable notebooks.

- [ ] **Step 1: Rewrite `README.md`**

````markdown
# WBC Classification

[![CI](https://github.com/includeamin/WBC-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/includeamin/WBC-Classification/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/includeamin/WBC-Classification)](https://github.com/includeamin/WBC-Classification/releases)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/github/license/includeamin/WBC-Classification)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-informational)](https://includeamin.github.io/WBC-Classification/)

Classify white blood cell images — **eosinophil, lymphocyte, monocyte, neutrophil** — with PyTorch.
Fine-tune a pretrained ResNet / EfficientNet or train the small `baseline` CNN, all from one CLI.

## Quick start

```bash
git clone https://github.com/includeamin/WBC-Classification.git
cd WBC-Classification
poetry install --extras kaggle

poetry run wbc download-data --dest data
poetry run wbc train -c configs/resnet18.yaml
poetry run wbc evaluate runs/<run>/best.pt --output metrics.json
poetry run wbc predict path/to/cell.jpeg -c runs/<run>/best.pt
```

Requires Python 3.12+ and [Poetry](https://python-poetry.org/). Training pretrained models is best done on a GPU
(see the [training guide](https://includeamin.github.io/WBC-Classification/training/) and `notebooks/train_colab.ipynb`).

## Documentation

Full docs: <https://includeamin.github.io/WBC-Classification/> — data setup, configuration reference,
CLI and API reference, architecture, results.

## Dataset

[Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) on Kaggle (not stored in this repository).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Releases and the changelog are automated from
[Conventional Commits](https://www.conventionalcommits.org/); see [CHANGELOG.md](CHANGELOG.md).

## License

BSD 3-Clause — see [LICENSE](LICENSE).
````

- [ ] **Step 2: Write `CONTRIBUTING.md`**

````markdown
# Contributing

Thanks for helping! This project uses [Poetry](https://python-poetry.org/) and Python 3.12+.

## Setup

```bash
poetry install --with dev,docs --extras kaggle
```

## Checks (all run in CI)

```bash
poetry run ruff check . && poetry run ruff format --check .
poetry run mypy src
poetry run pytest
poetry run typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md
poetry run mkdocs build --strict
```

Tests use the small fixture dataset in `tests/fixtures/`; they never download data or pretrained weights.
Mark long-running or network tests with `@pytest.mark.slow` (they are skipped by default).

## Commit messages and PR titles

Releases are automated from [Conventional Commits](https://www.conventionalcommits.org/). PRs are
squash-merged, so the **PR title** must follow the format:

| Prefix | Effect |
|---|---|
| `feat:` | minor release |
| `fix:`, `perf:` | patch release |
| `feat!:` or a `BREAKING CHANGE:` footer | major release (minor while version is 0.x) |
| `docs:`, `chore:`, `ci:`, `test:`, `refactor:` | no release |

Do not edit `CHANGELOG.md` or the version in `pyproject.toml` by hand; the release workflow does it on merge to `main`.

## Adding a model

Add an entry to `models/registry.py`, a config in `configs/`, and a shape test in `tests/test_models.py`.
````

- [ ] **Step 3: Generate the notebooks**

Run this script (do not commit the script itself):

```bash
poetry run python - <<'EOF'
import json
from pathlib import Path

def nb(cells):
    return {
        "cells": [
            {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if kind == "code" else {})}
            for kind, src in cells
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }

colab = nb([
    ("markdown", "# Train WBC Classification on a GPU (Colab / Kaggle)\n\nRuntime → Change runtime type → GPU."),
    ("code", "!git clone https://github.com/includeamin/WBC-Classification.git\n%cd WBC-Classification\n!pip install -q -e '.[kaggle]'"),
    ("code", "!wbc download-data --dest data"),
    ("code", "!wbc -v train -c configs/resnet18.yaml"),
    ("code", "!wbc evaluate $(ls -d runs/*/ | tail -1)best.pt --output metrics.json\n!cat metrics.json | head -40"),
])
detection = nb([
    ("markdown", "# WBC detection walkthrough\n\nHow the optional cell crop works."),
    ("code", "import numpy as np\nfrom PIL import Image\nfrom wbc_classification.preprocessing.segmentation import crop_cell, draw_overlay, segment_cell\n\nrgb = np.asarray(Image.open('../tests/fixtures/cell.jpeg').convert('RGB'))\nseg = segment_cell(rgb)\nImage.fromarray(draw_overlay(rgb, seg))"),
    ("code", "Image.fromarray(crop_cell(rgb))"),
])
Path("notebooks").mkdir(exist_ok=True)
Path("notebooks/train_colab.ipynb").write_text(json.dumps(colab, indent=1))
Path("notebooks/wbc_detection.ipynb").write_text(json.dumps(detection, indent=1))
EOF
poetry run python -c "import json; [json.load(open(f)) for f in ('notebooks/train_colab.ipynb','notebooks/wbc_detection.ipynb')]; print('ok')"
```
Expected: `ok`. The Colab notebook's `pip install -e '.[kaggle]'` relies on the project's PEP 621 `kaggle` extra; verify `pip install -e .` works in a fresh venv (`python -m venv /tmp/v && /tmp/v/bin/pip install -e .`) — if the Poetry build backend or the Python 3.12+ requirement blocks it on Colab, note that in `docs/training.md`.

- [ ] **Step 4: Rebuild docs and run checks**

Run: `poetry run mkdocs build --strict` — expected: success (the `contributing.md` snippet now includes the real file).

- [ ] **Step 5: Commit**

```bash
git add README.md CONTRIBUTING.md notebooks
git commit -m "docs: rewrite README, add contributing guide and notebooks

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 15: Release automation

**Files:**
- Modify: `pyproject.toml` (add `[tool.semantic_release]` config)
- Create: `.github/workflows/release.yml`

**Interfaces:**
- Produces: on every push to `main`, a workflow that runs python-semantic-release: next version from Conventional Commits, updates `pyproject.toml` (`project.version`) and `CHANGELOG.md`, tags `vX.Y.Z`, creates a GitHub Release with the built wheel/sdist; optional PyPI publish gated by repo variable `PUBLISH_PYPI == 'true'`.

- [ ] **Step 1: Check the configuration keys against the installed version**

Run: `poetry run semantic-release --version && poetry run semantic-release generate-config -f toml | head -60`
Use that output as the source of truth for key names; adjust the block below if any key differs in the installed major version.

- [ ] **Step 2: Append to `pyproject.toml`**

```toml
[tool.semantic_release]
version_toml = ["pyproject.toml:project.version"]
commit_parser = "conventional"
allow_zero_version = true
major_on_zero = false
tag_format = "v{version}"
build_command = "pip install -q poetry && poetry build"

[tool.semantic_release.branches.main]
match = "main"

[tool.semantic_release.changelog]
mode = "update"

[tool.semantic_release.commit_parser_options]
allowed_tags = ["feat", "fix", "perf", "docs", "chore", "ci", "test", "refactor", "build", "style"]
minor_tags = ["feat"]
patch_tags = ["fix", "perf"]

[tool.semantic_release.publish]
dist_glob_patterns = ["dist/*"]
upload_to_vcs_release = true
```

- [ ] **Step 3: Dry-run locally (no tags, no push)**

Run: `poetry run semantic-release --noop version --print`
Expected: prints the next version (`0.1.0`, because there are `feat:` commits and the current version is `0.0.0`). Branch-name errors ("not on a release branch") are expected here because the branch is `refactor/modernization`, not `main`; in that case confirm the config parses by running `poetry run semantic-release --noop -vv version --print 2>&1 | head -30` and checking that it reaches the branch check without a config error.

- [ ] **Step 4: Write `.github/workflows/release.yml`**

```yaml
name: Release

on:
  push:
    branches: [main]

permissions:
  contents: write
  id-token: write   # needed only for PyPI trusted publishing

concurrency:
  group: release
  cancel-in-progress: false

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
        with:
          fetch-depth: 0
          ref: ${{ github.ref_name }}

      - name: Semantic release
        id: release
        uses: python-semantic-release/python-semantic-release@v10
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}

      - name: Attach distributions to the GitHub release
        if: steps.release.outputs.released == 'true'
        uses: python-semantic-release/publish-action@v10
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          tag: ${{ steps.release.outputs.tag }}

      - name: Publish to PyPI (enable with repo variable PUBLISH_PYPI=true)
        if: steps.release.outputs.released == 'true' && vars.PUBLISH_PYPI == 'true'
        uses: pypa/gh-action-pypi-publish@release/v1
```

Notes to record in `CONTRIBUTING.md` under a new "Releases" heading: the workflow pushes the version-bump commit and tag to `main`, so branch protection must allow `github-actions[bot]` to push (or use a PAT/GitHub App token); PyPI publishing needs the project registered on PyPI with a trusted publisher configured, then set the repository variable `PUBLISH_PYPI` to `true`.

- [ ] **Step 5: Validate YAML and commit**

Run: `poetry run python -c "import yaml; yaml.safe_load(open('.github/workflows/release.yml')); print('ok')"`
Expected: `ok`

```bash
git add pyproject.toml .github CONTRIBUTING.md
git commit -m "ci: automate releases and changelog with python-semantic-release

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 16: Remove legacy code and final verification

**Files:**
- Delete: `CNN/` (remaining: `README.md`, `datasets/DatasetLoader.py`, `nn/`, `preprocessing/`), `learning.py`, `load_test_model.py`, `requirements.txt`, `_config.yml`, `WBC-Detection/` (remaining: `README.md`, `WBC-Detection.py`), `TrainedModel/`, `image.png`

- [ ] **Step 1: Remove the legacy files**

```bash
git rm -r -q CNN learning.py load_test_model.py requirements.txt _config.yml WBC-Detection TrainedModel image.png
git status --short | head -20
```
Expected: only deletions; `ls` shows no `CNN/`, `WBC-Detection/`, `TrainedModel/`.

- [ ] **Step 2: Verify nothing references the removed files**

Run: `grep -rnE "learning\.py|load_test_model|requirements\.txt|CNN/|TrainedModel|WBC-Detection/|IncludeNet.py" --include='*.md' --include='*.py' --include='*.yml' --include='*.yaml' --include='*.toml' . | grep -v '^./docs/superpowers/' | grep -v '^./.git/'`
Expected: no output (the spec/plan documents are excluded on purpose). Fix any hit.

- [ ] **Step 3: Full verification**

Run:

```bash
poetry run ruff check . && poetry run ruff format --check . && poetry run mypy src && poetry run pytest -q
poetry run typer wbc_classification.cli utils docs --name wbc --output docs/reference/cli.md && poetry run mkdocs build --strict
poetry build && ls dist
```
Expected: every command succeeds; `dist/` holds a wheel and an sdist named `wbc_classification-0.0.0-*`. Then `git status --short` must be clean apart from ignored files (`dist/`, `site/`, `docs/reference/cli.md` are ignored — add `dist/` to `.gitignore` if it shows up).

- [ ] **Step 4: Smoke-test the built wheel in a clean environment**

```bash
python3.14 -m venv /tmp/wbc-wheel-test && /tmp/wbc-wheel-test/bin/pip install -q dist/*.whl && /tmp/wbc-wheel-test/bin/wbc --version && /tmp/wbc-wheel-test/bin/wbc --help | head -20
rm -rf /tmp/wbc-wheel-test
```
Expected: version prints and the five commands are listed.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove legacy Keras code and obsolete files

BREAKING CHANGE: the Keras scripts, requirements.txt and the HDF5 model are removed; use the new `wbc` CLI.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```
(Note: the `BREAKING CHANGE:` footer on a 0.x project bumps the minor version only, because `major_on_zero = false`. The commit message should keep the footer separate from the `Co-Authored-By` trailer by a blank line, as shown.)

- [ ] **Step 6: Owner steps (do not perform; report them to the project owner)**

1. Rename the default branch on GitHub: Settings → Branches → rename `master` to `main` (then `git branch -m master main && git fetch && git branch -u origin/main main` locally).
2. Open a PR from `refactor/modernization` into `main` with a Conventional Commit title (e.g. `feat: modernize project with PyTorch, Poetry, docs and releases`) and squash-merge it; this triggers the first release (`0.1.0`).
3. Enable GitHub Pages (Settings → Pages → Source: GitHub Actions).
4. Allow `github-actions[bot]` to push to `main` (branch protection), or switch the release workflow to a PAT / GitHub App token.
5. Optional: register the project on PyPI with a trusted publisher and set the repo variable `PUBLISH_PYPI=true`.
6. Run full GPU training (`configs/resnet18.yaml`, `configs/efficientnet_b0.yaml`, `configs/baseline.yaml`, plus a `--crop` variant) and fill in `docs/results.md`.
7. Decide separately whether to purge the old dataset images from git history (destructive; needs a force-push).
