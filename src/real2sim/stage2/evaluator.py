"""Four-view RGB/Lab and metric-depth evaluator for Stage 2 scenes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_REFERENCE_INDICES = (0, 10, 21, 31)
DEPTH_MAE_THRESHOLD_M = 0.15
LAB_DE76_THRESHOLD = 15.0
VIEW_DEPTH_LIMIT_M = 0.20
VIEW_LAB_LIMIT = 30.0
MIN_DEPTH_COVERAGE = 0.50
MISSING_DEPTH_PENALTY_M = 1.0
MAX_VALID_RENDER_DEPTH_M = 1.0e6
REFERENCE_MAX_DEPTH_M = 2.0
SSIM_WINDOW_SIZE = 11
SSIM_SIGMA = 1.5


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read RGB image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def compute_ssim(reference_rgb: np.ndarray, rendered_rgb: np.ndarray,
                 window_size: int = SSIM_WINDOW_SIZE,
                 sigma: float = SSIM_SIGMA) -> float:
    """Compute grayscale SSIM for two same-sized RGB images.

    The score is the mean SSIM over a Gaussian window, using the standard
    normalized RGB-to-grayscale values and constants C1=(0.01)^2, C2=(0.03)^2.
    """
    if reference_rgb.shape != rendered_rgb.shape:
        raise ValueError(f"SSIM image shape mismatch: {reference_rgb.shape} != "
                         f"{rendered_rgb.shape}")
    if reference_rgb.ndim != 3 or reference_rgb.shape[2] != 3:
        raise ValueError(f"SSIM expects HxWx3 RGB images, got {reference_rgb.shape}")
    height, width = reference_rgb.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("SSIM cannot evaluate an empty image")

    reference = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rendered = cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    kernel_size = min(window_size, height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size < 3:
        reference_mean = float(reference.mean())
        rendered_mean = float(rendered.mean())
        reference_variance = float(reference.var())
        rendered_variance = float(rendered.var())
        covariance = float(((reference - reference_mean) *
                            (rendered - rendered_mean)).mean())
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        score = ((2 * reference_mean * rendered_mean + c1) *
                 (2 * covariance + c2) /
                 ((reference_mean ** 2 + rendered_mean ** 2 + c1) *
                  (reference_variance + rendered_variance + c2)))
        return float(score)

    kernel = (kernel_size, kernel_size)
    reference_mean = cv2.GaussianBlur(reference, kernel, sigma)
    rendered_mean = cv2.GaussianBlur(rendered, kernel, sigma)
    reference_squared_mean = cv2.GaussianBlur(reference * reference, kernel, sigma)
    rendered_squared_mean = cv2.GaussianBlur(rendered * rendered, kernel, sigma)
    cross_mean = cv2.GaussianBlur(reference * rendered, kernel, sigma)
    reference_variance = np.maximum(reference_squared_mean - reference_mean ** 2, 0.0)
    rendered_variance = np.maximum(rendered_squared_mean - rendered_mean ** 2, 0.0)
    covariance = cross_mean - reference_mean * rendered_mean
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    numerator = ((2 * reference_mean * rendered_mean + c1) * (2 * covariance + c2))
    denominator = ((reference_mean ** 2 + rendered_mean ** 2 + c1) *
                   (reference_variance + rendered_variance + c2))
    return float(np.mean(numerator / denominator))


def evaluate_ssim_bundle(reference: Path, render: Path,
                         output: Path | None = None,
                         reference_indices: tuple[int, ...] = DEFAULT_REFERENCE_INDICES
                         ) -> dict[str, Any]:
    """Evaluate SSIM for the four rendered RGB views of a Stage 2 scene."""
    reference = reference.expanduser().resolve(strict=True)
    render = render.expanduser().resolve(strict=True)
    if len(reference_indices) != 4:
        raise ValueError("Stage 2 requires exactly four reference views")

    views: list[dict[str, Any]] = []
    for ordinal, index in enumerate(reference_indices):
        reference_rgb_path = _reference_rgb_path(reference, ordinal)
        render_rgb_path = render / f"camera_{ordinal}_rgb.png"
        if not render_rgb_path.is_file():
            raise FileNotFoundError(f"missing rendered RGB for camera {ordinal} in {render}")
        reference_rgb = _read_rgb(reference_rgb_path)
        rendered_rgb = _read_rgb(render_rgb_path)
        views.append({
            "camera": f"camera_{ordinal}",
            "reference_index": index,
            "reference_rgb": str(reference_rgb_path),
            "render_rgb": str(render_rgb_path),
            "ssim": compute_ssim(reference_rgb, rendered_rgb),
        })

    report = {
        "schema": "stage2-ssim-evaluation-v1",
        "reference": str(reference),
        "render": str(render),
        "ssim_mean": float(np.mean([view["ssim"] for view in views])),
        "views": views,
        "method": {
            "color_space": "OpenCV RGB2GRAY on sRGB",
            "window_size": SSIM_WINDOW_SIZE,
            "sigma": SSIM_SIGMA,
            "higher_is_better": True,
        },
    }
    if output is not None:
        output = output.expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92,
                      ((values + 0.055) / 1.055) ** 2.4)
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]], dtype=np.float32)
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(np.maximum(xyz, 0.0)),
                 xyz / (3.0 * delta * delta) + 4.0 / 29.0)
    return np.stack((116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])), axis=-1)


def _error_preview(error: np.ndarray, output: Path, scale: float) -> None:
    normalized = np.clip(error / max(scale, 1e-8), 0.0, 1.0)
    preview = (normalized * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(output), colored)


def _reference_rgb_path(root: Path, ordinal: int) -> Path:
    rgb = root / "reference" / f"frame_{ordinal:02d}.jpg"
    if not rgb.is_file():
        raise FileNotFoundError(f"missing reference RGB: {rgb}")
    return rgb


def _reference_paths(root: Path, index: int, ordinal: int) -> tuple[Path, Path]:
    rgb = _reference_rgb_path(root, ordinal)
    depth = root / "depth_metric" / f"depth_{index:02d}.npy"
    if not depth.is_file():
        depth = root / "depth" / f"depth_{index:02d}.npy"
    if not depth.is_file():
        raise FileNotFoundError(f"missing metric depth: {depth}")
    return rgb, depth


def evaluate_bundle(reference: Path, render: Path, output: Path,
                    reference_indices: tuple[int, ...] = DEFAULT_REFERENCE_INDICES,
                    depth_threshold_m: float = DEPTH_MAE_THRESHOLD_M,
                    lab_threshold: float = LAB_DE76_THRESHOLD,
                    missing_depth_penalty_m: float = MISSING_DEPTH_PENALTY_M) -> dict[str, Any]:
    """Evaluate four rendered views against a Stage 1 delivery directory."""
    reference = reference.expanduser().resolve(strict=True)
    render = render.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if len(reference_indices) != 4:
        raise ValueError("Stage 2 requires exactly four reference views")

    views: list[dict[str, Any]] = []
    for ordinal, index in enumerate(reference_indices):
        reference_rgb_path, reference_depth_path = _reference_paths(reference, index, ordinal)
        render_rgb_path = render / f"camera_{ordinal}_rgb.png"
        render_depth_path = render / f"camera_{ordinal}_depth.npy"
        if not render_rgb_path.is_file() or not render_depth_path.is_file():
            raise FileNotFoundError(f"missing rendered view {ordinal} in {render}")

        reference_rgb = _read_rgb(reference_rgb_path)
        rendered_rgb = _read_rgb(render_rgb_path)
        reference_depth = np.asarray(np.load(reference_depth_path), dtype=np.float32)
        rendered_depth = np.asarray(np.load(render_depth_path), dtype=np.float32)
        if reference_rgb.shape != rendered_rgb.shape:
            raise ValueError(f"RGB shape mismatch for camera {ordinal}: "
                             f"{reference_rgb.shape} != {rendered_rgb.shape}")
        if reference_depth.shape != rendered_depth.shape[:2] or reference_depth.shape != reference_rgb.shape[:2]:
            raise ValueError(f"depth shape mismatch for camera {ordinal}")

        valid_reference = np.isfinite(reference_depth) & (reference_depth > 0)
        valid_rendered = (np.isfinite(rendered_depth) & (rendered_depth > 0) &
                          (rendered_depth < MAX_VALID_RENDER_DEPTH_M))
        valid_count = int(valid_reference.sum())
        if valid_count == 0:
            raise ValueError(f"reference depth has no valid pixels for camera {ordinal}")
        depth_error = np.where(valid_rendered,
                               np.abs(rendered_depth - reference_depth),
                               missing_depth_penalty_m)
        depth_mae = float(depth_error[valid_reference].mean())
        coverage = float((valid_rendered & valid_reference).sum() / valid_count)

        reference_lab = _srgb_to_lab(reference_rgb)
        rendered_lab = _srgb_to_lab(rendered_rgb)
        lab_error = np.linalg.norm(rendered_lab - reference_lab, axis=-1)
        lab_de76 = float(lab_error[valid_reference].mean())
        ssim = compute_ssim(reference_rgb, rendered_rgb)
        depth_preview = output / f"camera_{ordinal}_depth_error.png"
        lab_preview = output / f"camera_{ordinal}_lab_error.png"
        _error_preview(np.where(valid_reference, depth_error, 0.0), depth_preview,
                       max(depth_threshold_m * 2.0, 1e-6))
        _error_preview(np.where(valid_reference, lab_error, 0.0), lab_preview,
                       max(lab_threshold * 2.0, 1e-6))
        views.append({
            "camera": f"camera_{ordinal}",
            "reference_index": index,
            "reference_rgb": str(reference_rgb_path),
            "reference_depth": str(reference_depth_path),
            "render_rgb": str(render_rgb_path),
            "render_depth": str(render_depth_path),
            "depth_mae_m": depth_mae,
            "lab_delta_e76": lab_de76,
            "ssim": ssim,
            "reference_valid_pixels": valid_count,
            "rendered_valid_pixels": int(valid_rendered.sum()),
            "depth_coverage": coverage,
            "depth_passed": depth_mae < VIEW_DEPTH_LIMIT_M and coverage >= MIN_DEPTH_COVERAGE,
            "lab_passed": lab_de76 < VIEW_LAB_LIMIT,
            "depth_error_preview": str(depth_preview),
            "lab_error_preview": str(lab_preview),
        })

    depth_mean = float(np.mean([item["depth_mae_m"] for item in views]))
    lab_mean = float(np.mean([item["lab_delta_e76"] for item in views]))
    ssim_mean = float(np.mean([item["ssim"] for item in views]))
    passed = bool(depth_mean < depth_threshold_m and lab_mean < lab_threshold and
                  all(item["depth_passed"] and item["lab_passed"] for item in views))
    report = {
        "schema": "stage2-evaluation-v1",
        "passed": passed,
        "reference": str(reference),
        "render": str(render),
        "depth_mae_mean_m": depth_mean,
        "lab_delta_e76_mean": lab_mean,
        "ssim_mean": ssim_mean,
        "normalized_loss": 0.5 * depth_mean / depth_threshold_m +
                           0.5 * lab_mean / lab_threshold,
        "views": views,
            "thresholds": {
            "depth_mae_mean_m": depth_threshold_m,
            "lab_delta_e76_mean": lab_threshold,
            "single_view_depth_mae_m": VIEW_DEPTH_LIMIT_M,
            "single_view_lab_delta_e76": VIEW_LAB_LIMIT,
            "minimum_depth_coverage": MIN_DEPTH_COVERAGE,
            "missing_depth_penalty_m": missing_depth_penalty_m,
            "maximum_valid_render_depth_m": MAX_VALID_RENDER_DEPTH_M,
            "ssim": {
            "color_space": "OpenCV RGB2GRAY on sRGB",
                "window_size": SSIM_WINDOW_SIZE,
                "sigma": SSIM_SIGMA,
                "higher_is_better": True,
            },
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def evaluate_depth_bundle(reference: Path, render: Path, output: Path,
                          reference_indices: tuple[int, ...] = DEFAULT_REFERENCE_INDICES,
                          threshold_m: float = DEPTH_MAE_THRESHOLD_M,
                          missing_depth_penalty_m: float = REFERENCE_MAX_DEPTH_M) -> dict[str, Any]:
    """Evaluate only the four metric-depth views for the geometry phase."""
    reference = reference.expanduser().resolve(strict=True)
    render = render.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    views = []
    for ordinal, index in enumerate(reference_indices):
        _, reference_depth_path = _reference_paths(reference, index, ordinal)
        render_depth_path = render / f"camera_{ordinal}_depth.npy"
        if not render_depth_path.is_file():
            raise FileNotFoundError(f"missing rendered depth for camera {ordinal}")
        reference_depth = np.asarray(np.load(reference_depth_path), dtype=np.float32)
        rendered_depth = np.asarray(np.load(render_depth_path), dtype=np.float32)
        if reference_depth.shape != rendered_depth.shape:
            raise ValueError(f"depth shape mismatch for camera {ordinal}")
        valid_reference = np.isfinite(reference_depth) & (reference_depth > 0) & \
            (reference_depth <= REFERENCE_MAX_DEPTH_M)
        valid_rendered = (np.isfinite(rendered_depth) & (rendered_depth > 0) &
                          (rendered_depth < MAX_VALID_RENDER_DEPTH_M))
        count = int(valid_reference.sum())
        if count == 0:
            raise ValueError(f"reference depth has no valid pixels for camera {ordinal}")
        error = np.where(valid_rendered,
                         np.abs(rendered_depth - reference_depth),
                         missing_depth_penalty_m)
        mae = float(error[valid_reference].mean())
        coverage = float((valid_reference & valid_rendered).sum() / count)
        preview = output / f"camera_{ordinal}_depth_error.png"
        _error_preview(np.where(valid_reference, error, 0.0), preview,
                       max(threshold_m * 2.0, 1e-6))
        views.append({"camera": f"camera_{ordinal}", "reference_index": index,
                      "reference_depth": str(reference_depth_path),
                      "render_depth": str(render_depth_path), "mae_m": mae,
                      "reference_valid_pixels": count,
                      "rendered_valid_pixels": int(valid_rendered.sum()),
                      "coverage": coverage, "passed": mae < threshold_m,
                      "error_preview": str(preview)})
    mean_mae = float(np.mean([view["mae_m"] for view in views]))
    report = {"schema": "stage2-depth-evaluation-v1", "passed": bool(all(view["passed"] for view in views)),
              "mean_mae_m": mean_mae, "max_view_mae_m": float(max(view["mae_m"] for view in views)),
              "views": views, "thresholds": {"per_view_mae_m": threshold_m,
              "reference_max_depth_m": REFERENCE_MAX_DEPTH_M,
              "missing_depth_penalty_m": missing_depth_penalty_m}}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_bundle(args.reference, args.render, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
