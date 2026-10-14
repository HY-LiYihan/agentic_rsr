import json

import pytest

from real2sim.stage1.franka_delivery import publish, registration_summary


def report():
    return {"robot": "franka", "method": "franka_48_top1_sim3_rotation_v1",
            "status": "candidate_unverified", "accepted": False, "scene": "close_drawer",
            "initialization_sha256": "seed", "selected_seed_rank": 1, "index": 6,
            "rotation_count": 48, "model_stages": [], "rotation_steps_degrees": [15, 8, 4, 2],
            "minimum_loss_improvement": 0.0001, "pointcloud_frames": list(range(32)),
            "scoring_frames": list(range(32)), "loss": 0.11, "baseline_loss": 0.2,
            "metrics": {"per_frame": [{"frame": frame} for frame in range(32)]},
            "metric_calibration": {"unit": "m"}, "search_history": ["large internal history"]}


def test_summary_is_small_but_preserves_all_frame_metrics():
    result = registration_summary(report(), {"source_video": "close_drawer.mp4",
                                             "video_sha256": "sha", "models": {}})
    assert result["accepted"] is False
    assert result["status"] == "candidate_unverified"
    assert result["reference_indices"] == [0, 10, 21, 31]
    assert len(result["metrics"]["per_frame"]) == 32
    assert "search_history" not in result
    assert result["metric_calibration"]["unit"] == "m"


@pytest.mark.parametrize("field,value", [("status", "passed"), ("accepted", True),
                                           ("scoring_frames", list(range(31)))])
def test_delivery_rejects_unverified_or_incomplete_registration(field, value):
    result = report()
    result[field] = value
    with pytest.raises(ValueError):
        registration_summary(result, {"source_video": "close_drawer.mp4", "video_sha256": "sha"})


def test_changed_video_rejected_before_removing_previous_delivery(tmp_path):
    video = tmp_path / "close_drawer.mp4"
    video.write_bytes(b"changed")
    source = tmp_path / "close_drawer"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps({"robot": "franka", "status": "inference_complete",
                                                       "source_video": str(video), "video_sha256": "old"}))
    existing = source / "initial.blend"
    existing.write_bytes(b"previous good candidate")
    raw = source / "raw_vggt.npz"
    raw.write_bytes(b"cached inference")
    result = report()
    result["source"] = str(source)
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="original video is missing or changed"):
        publish(source, result_file, "blender", model=None, prune=True)
    assert existing.read_bytes() == b"previous good candidate"
    assert raw.read_bytes() == b"cached inference"


@pytest.mark.parametrize("robot", ["franka", "piper"])
@pytest.mark.parametrize("matching_seed", [True, False])
def test_delivery_checks_custom_initialization_hash(tmp_path, monkeypatch, robot, matching_seed):
    from real2sim.stage1 import franka_delivery
    from real2sim.stage1.metric_calibration import metric_method
    from real2sim.stage1.pipeline import sha256

    monkeypatch.setattr(franka_delivery, "ROOT", tmp_path)
    urdf = tmp_path / "robot.urdf"
    urdf.write_bytes(b"robot description")
    monkeypatch.setattr(franka_delivery, "FRANKA_URDF", urdf)
    monkeypatch.setattr(franka_delivery, "MODEL", urdf)
    monkeypatch.setattr(franka_delivery, "assembly_sha256", lambda: "assembly")
    monkeypatch.setattr(franka_delivery, "mask_digest", lambda _: "masks")
    asset = tmp_path / "asset/robot_surfaces" / robot
    asset.mkdir(parents=True)
    surface = asset / "surface_40000.npz"
    surface.write_bytes(b"surface")
    (asset / "metadata.json").write_text(json.dumps({
        "urdf_sha256": sha256(urdf), "joint_positions": franka_delivery.POSES[robot],
        "assembly_sha256": "assembly"}))
    video = tmp_path / "close_drawer.mp4"
    video.write_bytes(b"original video")
    source = tmp_path / "fresh-processed" / robot / "stage1/close_drawer"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({
        "robot": robot, "status": "inference_complete", "source_video": str(video),
        "video_sha256": sha256(video), "sam3_masks_sha256": "masks"}))
    initializations = tmp_path / "fresh-initializations" / robot
    initialization = initializations / "close_drawer/initialization.json"
    initialization.parent.mkdir(parents=True)
    initialization.write_text(json.dumps({"robot_surface_sha256": sha256(surface)}))
    result = {**report(), "robot": robot, "method": metric_method(robot), "source": str(source),
              "sam3_masks_sha256": "masks", "initialization_sha256":
              sha256(initialization) if matching_seed else "stale seed"}
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps(result))

    def stop_after_provenance(*args):
        raise RuntimeError("scene preparation reached")

    monkeypatch.setattr(franka_delivery, "prepare_scene", stop_after_provenance)
    if matching_seed:
        with pytest.raises(RuntimeError, match="scene preparation reached"):
            publish(source, result_file, "blender", model=None, initializations=initializations)
    else:
        with pytest.raises(ValueError, match="seed no longer matches"):
            publish(source, result_file, "blender", model=None, initializations=initializations)
