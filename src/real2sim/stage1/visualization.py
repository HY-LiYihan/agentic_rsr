"""Depth and segmentation previews for 32-frame inference exports."""

from pathlib import Path

import cv2
import numpy as np


def depth_preview(depth: np.ndarray, target: Path, *, metric: bool) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    pixels = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        lower, upper = np.percentile(depth[valid], [2, 98])
        if upper > lower:
            pixels[valid] = np.clip((depth[valid] - lower) / (upper - lower) * 255, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(pixels, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    label = "depth: meters" if metric else "depth: VGGT units (NOT metric)"
    cv2.putText(colored, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if not cv2.imwrite(str(target), colored):
        raise OSError(target)


def arm_overlay(frame: Path, mask: np.ndarray, target: Path) -> None:
    image = cv2.imread(str(frame))
    if image is None:
        raise FileNotFoundError(frame)
    output = image.copy()
    output[mask] = (0.55 * output[mask] + 0.45 * np.array([0, 210, 80])).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0, 210, 80), 1)
    if not cv2.imwrite(str(target), output):
        raise OSError(target)
