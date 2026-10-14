import json

import numpy as np
import pytest

from real2sim.stage1.pipeline import sha256
from real2sim.stage1.promote_sam3_masks import mask_digest, promote_scene


def test_promote_valid_trial_preserves_depth_and_invalidates_old_delivery(tmp_path):
    source = tmp_path / "processed/task"
    trial = tmp_path / "trials/task"
    video = tmp_path / "task.mp4"
    video.write_bytes(b"video")
    for folder in (source / "frames", source / "depth", source / "sam3_masks",
                   source / "reference", source / "depth_metric", trial / "masks"):
        folder.mkdir(parents=True)
    mask = np.zeros((288, 512), dtype=bool)
    mask[30:90, 20:80] = True
    for index in range(32):
        (source / "frames" / f"{index:02d}.jpg").write_bytes(b"rgb")
        np.save(source / "depth" / f"depth_{index:02d}.npy", np.ones((288, 512)))
        np.save(source / "sam3_masks" / f"mask_{index:02d}.npy", np.zeros_like(mask))
        np.save(trial / "masks" / f"mask_{index:02d}.npy", mask)
    manifest = {"status": "inference_complete", "robot": "piper",
                "source_video": str(video), "video_sha256": sha256(video),
                "sam3_prompt": "robotic arm", "stage1_delivery": "stage1_delivery.json",
                "models": {"sam3": {"backend": "remote", "response": {"prompts": ["robotic arm"]}}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    report = {"status": "complete", "prompt": "robotic arm with base and camera",
              "video_sha256": sha256(video),
              "per_frame": {f"{index:02d}": {"instance_count": 1} for index in range(32)}}
    (trial / "report.json").write_text(json.dumps(report))
    (source / "initial.blend").write_bytes(b"old blend")
    (source / "depth_metric" / "depth_00.npy").write_bytes(b"old depth")
    digest = mask_digest(trial / "masks")
    assert promote_scene(source, trial)["sam3_masks_sha256"] == digest
    assert not (source / "initial.blend").exists()
    assert not (source / "depth_metric").exists()
    assert (source / "depth" / "depth_00.npy").exists()
    assert mask_digest(source / "sam3_masks") == digest
    assert promote_scene(source, trial)["sam3_masks_sha256"] == digest

    report["prompt"] = "bad prompt"
    (trial / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="stale"):
        promote_scene(source, trial)
    assert mask_digest(source / "sam3_masks") == digest
