import json
from types import SimpleNamespace

import numpy as np

from real2sim.stage1.pipeline import pointcloud, run, validate
from real2sim.stage1.registration import REFERENCE, camera_world, register, rigid_fit, score_view


def test_transforms_and_rejection():
    extrinsic = np.column_stack((np.eye(3), [-1, 0, 0]))
    assert np.allclose(camera_world(extrinsic)[:3, 3], [1, 0, 0])
    model = np.array([[0., 0., 2.], [0.1, 0., 2.], [0., 0.1, 2.], [0., 0., 2.1]])
    target = model + [1, 2, 3]
    assert np.allclose(rigid_fit(model, target)[:3, 3], [1, 2, 3])
    intrinsic = np.array([[100., 0, 100], [0, 100, 100], [0, 0, 1]])
    depth = np.full((200, 200), 2., np.float32)
    mask = np.zeros_like(depth, dtype=bool)
    mask[98:108, 98:108] = True
    score = score_view(model, np.eye(4), intrinsic, np.column_stack((np.eye(3), [0, 0, 0])), depth, mask)
    assert score["median_depth_residual_m"] < 0.05
    mask[:, :] = False
    mask[50:60, 50:60] = True
    rejected = score_view(model, np.eye(4), intrinsic,
                          np.column_stack((np.eye(3), [0, 0, 0])), depth, mask)
    assert rejected["mask_iou"] < 0.35
    assert pointcloud(depth, intrinsic, extrinsic).shape[1] == 3


def test_invalid_camera_and_failed_runs_do_not_affect_prior(tmp_path):
    arrays = {"depth": np.ones((32, 288, 512), np.float32),
              "depth_conf": np.ones((32, 288, 512), np.float32),
              "intrinsics": np.tile(np.eye(3), (32, 1, 1)),
              "extrinsics": np.zeros((32, 3, 4))}
    try:
        validate(arrays, 1)
    except ValueError as exc:
        assert "invalid camera" in str(exc)
    else:
        raise AssertionError("invalid extrinsics must fail")
    output = tmp_path / "runs"
    previous = output / "previous"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text('{"status":"passed"}')
    first, passed = run(tmp_path / "missing.mp4", "task", output)
    second, passed_again = run(tmp_path / "missing.mp4", "task", output)
    assert not passed and not passed_again and first != second
    assert json.loads((first / "manifest.json").read_text())["status"] == "failed"
    assert json.loads((previous / "manifest.json").read_text())["status"] == "passed"


def test_registration_does_not_fit_on_held_out_frames(monkeypatch):
    from real2sim.stage1 import registration

    def identity_fit(model, observed):
        assert len(observed) > 30
        return np.eye(4), 0.0

    monkeypatch.setattr(registration, "fit_pose", identity_fit)
    depths = np.full((32, 288, 512), 2., np.float32)
    intrinsics = np.tile(np.array([[400., 0, 256], [0, 400, 144], [0, 0, 1]]), (32, 1, 1))
    extrinsics = np.tile(np.column_stack((np.eye(3), [0, 0, 0])), (32, 1, 1))
    masks = [np.ones((288, 512), bool) if index not in REFERENCE else np.zeros((288, 512), bool)
             for index in range(32)]
    points = np.array([[0, 0, 2.], [.1, .1, 2.], [-.1, -.1, 2.]])
    pose, report, projections = register(points, depths, intrinsics, extrinsics, masks)
    assert not report["accepted"] and report["held_out_frames"] == list(REFERENCE)
    assert len(projections) == 4 and np.allclose(pose, np.eye(4))


def test_uncalibrated_pipeline_does_not_treat_vggt_units_as_meters(tmp_path, monkeypatch):
    from real2sim.stage1 import pipeline

    arrays = {"depth": np.full((32, 288, 512), 10, np.float32),
              "depth_conf": np.ones((32, 288, 512), np.float32),
              "intrinsics": np.tile([[400, 0, 256], [0, 400, 144], [0, 0, 1]], (32, 1, 1)),
              "extrinsics": np.tile(np.column_stack((np.eye(3), [0, 0, 0])), (32, 1, 1))}
    monkeypatch.setattr(pipeline, "sha256", lambda path: "test-hash")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="revision"))
    monkeypatch.setattr(pipeline, "extract", lambda *args: ([tmp_path / "frame.jpg"] * 32, list(range(32))))

    def stop_after_reconstruction(image):
        raise RuntimeError("reached SAM3 after uncalibrated depth validation")

    vggt = SimpleNamespace(infer=lambda frames: arrays)
    sam3 = SimpleNamespace(segment=stop_after_reconstruction)
    folder, passed = run(tmp_path / "video.mp4", "test", tmp_path / "output", vggt=vggt, sam3=sam3)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert not passed
    assert manifest["error"]["message"] == "reached SAM3 after uncalibrated depth validation"
    assert (folder / "reconstruction.npz").is_file()

    folder, passed = run(tmp_path / "video.mp4", "test", tmp_path / "output", metric_scale=1.0,
                         vggt=vggt, sam3=sam3)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert not passed
    assert "plausible positive depth" in manifest["error"]["message"]
