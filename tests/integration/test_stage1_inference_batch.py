import json

import cv2
import numpy as np

from real2sim.stage1.inference_batch import batch, complete


class FakeVGGT:
    def __init__(self):
        self.calls = 0

    def infer(self, frames):
        assert len(frames) == 32
        self.calls += 1
        return {"depth": np.full((32, 288, 512), 1.5, np.float32),
                "depth_conf": np.ones((32, 288, 512), np.float32),
                "intrinsics": np.tile(np.array([[400, 0, 256], [0, 400, 144], [0, 0, 1]], np.float32), (32, 1, 1)),
                "extrinsics": np.tile(np.column_stack((np.eye(3), [0, 0, 0])), (32, 1, 1))}


class FakeSAM3:
    def segment(self, frame):
        mask = np.zeros((288, 512), bool)
        mask[50:70, 40:80] = True
        return mask


def make_video(path):
    path.parent.mkdir(parents=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (160, 90))
    assert writer.isOpened()
    for frame_index in range(32):
        writer.write(np.full((90, 160, 3), frame_index, np.uint8))
    writer.release()


def test_serial_two_robots_and_no_registration(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    make_video(raw / "frank" / "task_a.mp4")
    make_video(raw / "piper" / "task_b.mp4")

    def forbidden_registration(*arguments, **options):
        raise AssertionError("registration must not run during inference")

    monkeypatch.setattr("real2sim.stage1.pipeline.register", forbidden_registration)
    vggt = FakeVGGT()
    results = batch(raw, processed, vggt=vggt, sam3=FakeSAM3(), progress=lambda message: None)
    assert [result["status"] for result in results] == ["inference_complete"] * 2
    assert vggt.calls == 2
    for robot, video in (("franka", "task_a"), ("piper", "task_b")):
        output = processed / robot / "stage1" / video
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["registration"] == "not_started"
        assert manifest["reference_indices"] == [0, 10, 21, 31]
        assert manifest["sam3_prompt"] == "robotic arm with base and camera"
        if robot == "piper":
            assert manifest["sam3_masks_sha256"]
        assert complete(output, manifest["video_sha256"])
        manifest["sam3_prompt"] = "robotic arm"
        (output / "manifest.json").write_text(json.dumps(manifest))
        assert not complete(output, manifest["video_sha256"])
        manifest["sam3_prompt"] = "robotic arm with base and camera"
        (output / "manifest.json").write_text(json.dumps(manifest))
        assert not (output / "initial.blend").exists()
        assert not (output / "registration.json").exists()
        assert np.load(output / "depth" / "depth_31.npy").shape == (288, 512)
        assert np.load(output / "sam3_masks" / "mask_31.npy").dtype == bool
        assert not (output / "depth_vis").exists()
        assert not (output / "sam3_masks_vis").exists()
        assert not (output / "raw_vggt.npz").exists()
    assert len(batch(raw, processed, vggt=vggt, sam3=FakeSAM3(), progress=lambda message: None)) == 2
    assert vggt.calls == 2
