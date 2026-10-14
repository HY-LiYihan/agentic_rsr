import hashlib
import json

import cv2
import numpy as np
import pytest

from real2sim.stage1.sam3_prompt_trial import PROMPT, run_all


class FakeSAM3:
    def __init__(self):
        self.calls = 0
        self.metadata = {}

    def segment(self, image, prompt):
        assert prompt == PROMPT
        self.calls += 1
        self.metadata = {"k": 1, "conf_thresh": .25}
        result = np.zeros((288, 512), dtype=bool)
        result[50:80, 100:150] = True
        return result


def test_trial_keeps_original_masks_and_creates_one_image_per_scene(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    video = raw / "task.mp4"
    video.write_bytes(b"recording")
    source = tmp_path / "processed/task"
    (source / "frames").mkdir(parents=True)
    (source / "sam3_masks").mkdir()
    (source / "manifest.json").write_text(json.dumps({
        "robot": "piper", "status": "inference_complete", "source_video": str(video),
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "sam3_prompt": "robotic arm"}))
    original = np.zeros((288, 512), dtype=bool)
    original[10:20, 10:20] = True
    for index in range(32):
        cv2.imwrite(str(source / "frames" / f"{index:02d}.jpg"),
                    np.full((288, 512, 3), 100, dtype=np.uint8))
        np.save(source / "sam3_masks" / f"mask_{index:02d}.npy", original)
    output = tmp_path / "trial"
    service = FakeSAM3()
    results = run_all(tmp_path / "processed", raw, output, service)
    assert len(results) == 1 and service.calls == 32
    assert len(list((output / "task").glob("*.jpg"))) == 1
    assert len(list((output / "task/masks").glob("*.npy"))) == 32
    assert np.load(source / "sam3_masks/mask_00.npy").sum() == 100
    assert np.load(output / "task/masks/mask_00.npy").sum() == 1500
    assert json.loads((output / "task/report.json").read_text())["per_frame"]["31"]["instance_count"] == 1
    assert json.loads((output / "summary.json").read_text())["actual_prompt"] == PROMPT
    run_all(tmp_path / "processed", raw, output, service)
    assert service.calls == 32
    video.write_bytes(b"new footage")
    with pytest.raises(ValueError, match="stale"):
        run_all(tmp_path / "processed", raw, output, service)
