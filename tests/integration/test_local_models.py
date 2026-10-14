"""Opt-in real model checks; never fetch weights or silently use remote services."""

import json
import os
from pathlib import Path
import time

import pytest

pytestmark = pytest.mark.models


def cuda_model_environment():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("A CUDA GPU is required for real model verification")
    if os.getenv("VGGT_OMEGA_BACKEND", "local") != "local" or os.getenv("SAM3_BACKEND", "local") != "local":
        pytest.fail("Model tests require explicitly local backends")
    return torch


def test_real_video_inference_records_models_and_memory(tmp_path):
    video = os.getenv("RSR_MODEL_TEST_VIDEO")
    if not video:
        pytest.skip("Set RSR_MODEL_TEST_VIDEO to a supplied video with a visible robot and 32+ frames")
    cuda_model_environment()
    from perception.example import run_example

    report = run_example(Path(video), tmp_path / "example")
    assert report["status"] == "inference_complete"
    assert report["cuda_memory"]
    assert all(model["checkpoint_sha256"] for model in report["models"].values())
    manifest = json.loads((tmp_path / "example/manifest.json").read_text())
    assert len(manifest["empty_mask_frames"]) < 32
    assert len(list((tmp_path / "example/previews").glob("*.png"))) == 8


def test_real_sim_local_sam3_observation(tmp_path):
    scene_path = os.getenv("RSR_MODEL_TEST_SCENE")
    if not scene_path:
        pytest.skip("Set RSR_MODEL_TEST_SCENE to a prepared Stage 3 runtime XML")
    torch = cuda_model_environment()
    pytest.importorskip("robot_control")
    from perception.config import backend_settings, dependency_versions
    from sim.stage3_scene import Stage3Scene
    from sim.stage3_observation import Stage3RGBDObservation

    device = backend_settings("sam3")["device"]
    assert device.startswith("cuda"), "SIM model validation requires a CUDA SAM3 device"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    scene = Stage3Scene(Path(scene_path), os.getenv("RSR_MODEL_TEST_ROBOT", "piper"))
    observation = None
    try:
        names = tuple(name for name in scene.object_bodies if name != "world")
        assert names
        observation = Stage3RGBDObservation(scene, names[:1], tmp_path / "images")
        result = observation.observe()
        assert Path(result["image"]["path"]).is_file()
        assert observation.metadata["backend"] == "local"
        assert observation.metadata["checkpoint_sha256"]
        report = {"elapsed_seconds": time.monotonic() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "model": observation.metadata, "dependency_versions": dependency_versions(),
                  "detected_objects": list(result["objects"])}
        (tmp_path / "sim_model_report.json").write_text(json.dumps(report, indent=2) + "\n")
    finally:
        if observation is not None:
            observation.close()
        scene.close()
