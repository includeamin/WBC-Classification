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


def test_build_test_loader_missing_test_dir(tmp_path):
    with pytest.raises(DataError, match="wbc download-data"):
        build_test_loader(_cfg(tmp_path), CLASSES)


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
