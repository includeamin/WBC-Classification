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
