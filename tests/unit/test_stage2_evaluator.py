from pathlib import Path

import cv2
import numpy as np

from real2sim.stage2.evaluator import compute_ssim, evaluate_bundle, evaluate_ssim_bundle
from real2sim.stage2.phase_runner import _current_threshold_report


def _make_bundle(root: Path, rendered_rgb: np.ndarray, rendered_depth: np.ndarray) -> None:
    (root / "reference").mkdir(parents=True)
    (root / "depth_metric").mkdir()
    for ordinal, index in enumerate((0, 10, 21, 31)):
        rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(root / "reference" / f"frame_{ordinal:02d}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(root / "depth_metric" / f"depth_{index:02d}.npy", np.full((4, 6), 1.0, dtype=np.float32))


def test_evaluator_passes_identical_four_views(tmp_path):
    reference = tmp_path / "reference"
    render = tmp_path / "render"
    render.mkdir()
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    depth = np.full((4, 6), 1.0, dtype=np.float32)
    _make_bundle(reference, rgb, depth)
    for ordinal in range(4):
        cv2.imwrite(str(render / f"camera_{ordinal}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(render / f"camera_{ordinal}_depth.npy", depth)
    report = evaluate_bundle(reference, render, tmp_path / "evaluation")
    assert report["passed"]
    assert report["depth_mae_mean_m"] == 0.0
    assert report["lab_delta_e76_mean"] < 0.01
    assert report["ssim_mean"] == 1.0
    assert all(view["ssim"] == 1.0 for view in report["views"])


def test_compute_ssim_decreases_for_structural_change():
    reference = np.zeros((32, 32, 3), dtype=np.uint8)
    reference[8:24, 8:24] = 255
    rendered = np.zeros_like(reference)
    rendered[10:26, 8:24] = 255
    assert compute_ssim(reference, reference) == 1.0
    assert 0.0 <= compute_ssim(reference, rendered) < 1.0


def test_ssim_bundle_reports_four_views_without_depth_render(tmp_path):
    reference = tmp_path / "reference"
    render = tmp_path / "render"
    (reference / "reference").mkdir(parents=True)
    render.mkdir()
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    for ordinal in range(4):
        cv2.imwrite(str(reference / "reference" / f"frame_{ordinal:02d}.jpg"), rgb)
        cv2.imwrite(str(render / f"camera_{ordinal}_rgb.png"), rgb)
    result = evaluate_ssim_bundle(reference, render, tmp_path / "ssim")
    assert result["ssim_mean"] == 1.0
    assert [view["camera"] for view in result["views"]] == [f"camera_{i}" for i in range(4)]
    assert (tmp_path / "ssim" / "report.json").is_file()


def test_evaluator_penalizes_missing_rendered_depth(tmp_path):
    reference = tmp_path / "reference"
    render = tmp_path / "render"
    render.mkdir()
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    depth = np.full((4, 6), 1.0, dtype=np.float32)
    _make_bundle(reference, rgb, depth)
    for ordinal in range(4):
        cv2.imwrite(str(render / f"camera_{ordinal}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(render / f"camera_{ordinal}_depth.npy", np.full((4, 6), np.nan, dtype=np.float32))
    report = evaluate_bundle(reference, render, tmp_path / "evaluation")
    assert report["passed"] is False
    assert report["depth_mae_mean_m"] == 1.0
    assert all(view["depth_coverage"] == 0.0 for view in report["views"])


def test_evaluator_accepts_depth_below_new_threshold(tmp_path):
    reference = tmp_path / "reference"
    render = tmp_path / "render"
    render.mkdir()
    rgb = np.full((4, 6, 3), 128, dtype=np.uint8)
    _make_bundle(reference, rgb, np.full((4, 6), 1.14, dtype=np.float32))
    for ordinal in range(4):
        cv2.imwrite(str(render / f"camera_{ordinal}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        np.save(render / f"camera_{ordinal}_depth.npy", np.full((4, 6), 1.14, dtype=np.float32))
    report = evaluate_bundle(reference, render, tmp_path / "evaluation")
    assert report["passed"]
    assert report["thresholds"]["depth_mae_mean_m"] == 0.15
    for ordinal in range(4):
        np.save(render / f"camera_{ordinal}_depth.npy", np.full((4, 6), 1.16, dtype=np.float32))
    assert not evaluate_bundle(reference, render, tmp_path / "evaluation_failed")["passed"]


def test_old_report_uses_current_threshold_without_mutation():
    old_report = {
        "thresholds": {"depth_mae_mean_m": 0.1, "lab_delta_e76_mean": 15.0},
        "depth_mae_mean_m": 0.12, "lab_delta_e76_mean": 12.0,
        "passed": False, "views": [{"depth_passed": True, "lab_passed": True}],
    }
    updated = _current_threshold_report(old_report)
    assert updated["passed"]
    assert updated["thresholds"]["depth_mae_mean_m"] == 0.15
    assert old_report["thresholds"]["depth_mae_mean_m"] == 0.1
