"""Exercise the public example and its artifacts without loading model weights."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from perception import example


@pytest.fixture
def local_services(monkeypatch):
    events = []

    class VGGT:
        metadata = {"backend": "local", "checkpoint_sha256": "fake-vggt"}

        def infer(self, frames):
            events.append("vggt")
            assert len(frames) == 32
            assert cv2.imread(str(frames[0])).shape == (288, 512, 3)
            return {"depth": np.ones((32, 288, 512), dtype=np.float32),
                    "depth_conf": np.ones((32, 288, 512), dtype=np.float32),
                    "intrinsics": np.tile([[400, 0, 256], [0, 400, 144], [0, 0, 1]], (32, 1, 1)),
                    "extrinsics": np.tile(np.column_stack([np.eye(3), [0, 0, 0]]), (32, 1, 1))}

        def close(self):
            events.append("release-vggt")

    class SAM3:
        metadata = {"backend": "local", "checkpoint_sha256": "fake-sam3"}

        def segment(self, image):
            assert "release-vggt" in events
            events.append("sam3")
            return np.ones((288, 512), dtype=bool)

        def close(self):
            events.append("release-sam3")

    monkeypatch.setattr(example, "backend_settings", lambda name: {"backend": "local", "device": "cpu"})
    monkeypatch.setattr(example, "create_vggt", VGGT)
    monkeypatch.setattr(example, "create_sam3", SAM3)
    monkeypatch.setattr(example, "require_device", lambda *args: None)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None)))
    return events


def video(path, count):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    try:
        for index in range(count):
            writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_example_records_manifest_previews_and_serial_model_release(tmp_path, local_services):
    output = tmp_path / "result"
    report = example.run_example(video(tmp_path / "input.avi", 32), output)
    assert report["status"] == "inference_complete"
    assert report["metric_scale_verified"] is False
    assert "numpy" in report["dependency_versions"]
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["models"]["sam3"]["checkpoint_sha256"] == "fake-sam3"
    assert len(list((output / "previews").glob("*.png"))) == 8
    assert local_services.count("sam3") == 32
    assert local_services[-2:] == ["release-vggt", "release-sam3"]


def test_short_video_fails_with_diagnostics_and_without_model_call(tmp_path, local_services):
    output = tmp_path / "result"
    report = example.run_example(video(tmp_path / "short.avi", 8), output)
    assert report["status"] == "failed"
    assert "at least 32" in report["error"]["message"]
    assert (output / "error.log").is_file()
    assert "vggt" not in local_services and "sam3" not in local_services


def test_example_rejects_existing_output_without_overwrite(tmp_path):
    source = tmp_path / "input.mp4"
    source.touch()
    output = tmp_path / "result"
    output.mkdir()
    marker = output / "previous.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        example.run_example(source, output)
    assert marker.read_text() == "keep"


def test_example_rejects_remote_configuration(tmp_path, monkeypatch):
    source = tmp_path / "input.mp4"
    source.touch()
    monkeypatch.setattr(example, "backend_settings", lambda name: {"backend": "remote"})
    with pytest.raises(ValueError, match="requires"):
        example.run_example(source, tmp_path / "result")
    assert not (tmp_path / "result").exists()
