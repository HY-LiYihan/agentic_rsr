import json
import sys

import cv2
import numpy as np
import pytest

from real2sim.stage1 import initial_pose_batch
from real2sim.stage1.sim3_registration import apply_pose, search_rotations


def test_piper_initial_cli_defaults_to_separate_128_review(tmp_path, monkeypatch):
    monkeypatch.setattr(initial_pose_batch, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(initial_pose_batch, "batch", lambda processed, output, rotations, **kwargs:
                        calls.append((output, rotations, kwargs["robot"])) or [])
    monkeypatch.setattr(sys, "argv", ["initial", "--robot", "piper"])
    initial_pose_batch.main()
    assert calls == [(tmp_path / "runs/piper_initial_128_review", 128, "piper")]
    monkeypatch.setattr(sys, "argv", ["initial", "--robot", "franka"])
    initial_pose_batch.main()
    assert calls[-1] == (tmp_path / "runs/initial_poses", 48, "franka")


@pytest.mark.parametrize("count", [48, 128])
def test_rotation_set_has_distinct_proper_rotations(count):
    rotations = search_rotations(count, include_identity=False)
    assert len(rotations) == count
    assert len({tuple(np.round(rotation.ravel(), 7)) for rotation in rotations}) == count
    for rotation in rotations:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(rotation), 1., atol=1e-12)


def test_center_and_scale_use_only_cloud_distribution():
    rng = np.random.default_rng(17)
    model = rng.uniform(-.1, .1, (100, 3))
    observed = 1.3 * model + [1., -.3, 2.]
    model_center, observed_center, scale = initial_pose_batch.center_and_scale(model, observed)
    np.testing.assert_allclose(scale, 1.3, atol=1e-10)
    for rotation in search_rotations(3, include_identity=False):
        translation = observed_center - scale * rotation @ model_center
        np.testing.assert_allclose(apply_pose(model, rotation, translation, scale).mean(0),
                                   observed_center, atol=1e-10)


def test_initial_visualization_does_not_run_icp(tmp_path, monkeypatch):
    rng = np.random.default_rng(2026)
    model = rng.uniform(-.1, .1, (100, 3)).astype(np.float32)
    assets = tmp_path / "asset/robot_surfaces/piper"
    assets.mkdir(parents=True)
    for size in (5000, 40000):
        np.savez_compressed(assets / f"surface_{size}.npz", points=model)
    source = tmp_path / "processed/piper/stage1/task"
    frames = source / "frames"
    frames.mkdir(parents=True)
    for index in range(32):
        cv2.imwrite(str(frames / f"{index:02}.jpg"), np.zeros((80, 80, 3), np.uint8))
    camera = {"intrinsics": np.tile([[100., 0., 40.], [0., 100., 40.], [0., 0., 1.]], (32, 1, 1)),
              "extrinsics": np.tile(np.column_stack((np.eye(3), [0, 0, 0])), (32, 1, 1))}
    masks = {index: np.zeros((80, 80), bool) for index in range(32)}
    for mask in masks.values():
        mask[30:55, 30:55] = True
    monkeypatch.setattr(initial_pose_batch, "ROOT", tmp_path)
    monkeypatch.setattr(initial_pose_batch, "load_observations",
                        lambda _, *, held_out: ({"robot": "piper", "video_sha256": "video"}, camera,
                                   {index: np.full((80, 80), 2., np.float32) for index in range(32)}, masks,
                                   1.5 * model + [0., 0., 2.], []))
    output = tmp_path / "output"
    report = initial_pose_batch.visualize_scene(source, output, rotations=3)
    assert report["rotation_count"] == len(report["candidates"]) == 3
    assert report["status"] == "initialization_only" and not report["accepted"]
    assert json.loads((output / "initialization.json").read_text())["candidates"] == report["candidates"]
    assert all(len(candidate["metrics"]["per_frame"]) == 32 for candidate in report["candidates"])
    assert report["scoring_frames"] == list(range(32))
    assert report["top_four_indices"] == [candidate["index"] for candidate in sorted(
        report["candidates"], key=lambda candidate: (candidate["loss"], candidate["index"]))]
    assert cv2.imread(str(output / "top4.png")).shape == (1695, 1600, 3)
    assert not (output / "aligned_model.npz").exists()
