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
    return rgb[
        max(cy - radius, 0) : min(cy + radius, height),
        max(cx - radius, 0) : min(cx + radius, width),
    ]


def draw_overlay(rgb: np.ndarray, segmentation: CellSegmentation) -> np.ndarray:
    """Return a copy of the image with the cell contour and enclosing circle drawn."""
    overlay = rgb.copy()
    cv2.drawContours(overlay, [segmentation.contour], -1, (255, 0, 0), 2)
    cv2.circle(overlay, segmentation.center, segmentation.radius, (0, 255, 0), 2)
    return overlay
