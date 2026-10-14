import hashlib
import json

import cv2
import numpy as np
import pytest

from real2sim.stage1.sam3_review import review_all


def test_review_cached_masks_without_inference(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    video = raw / "task.mp4"
    video.write_bytes(b"video contents")
    source = tmp_path / "processed/task"
    (source / "frames").mkdir(parents=True)
    (source / "sam3_masks").mkdir()
    manifest = {"robot": "piper", "status": "inference_complete",
                "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
                "source_video": str(video), "source_frame_indices": list(range(32)),
                "models": {"sam3": {"response": {"k": 2}}}}
    (source / "manifest.json").write_text(json.dumps(manifest))
    for index in range(32):
        assert cv2.imwrite(str(source / "frames" / f"{index:02d}.jpg"),
                           np.full((16, 24, 3), 150, dtype=np.uint8))
        mask = np.zeros((16, 24), dtype=bool)
        mask[2:6, 3:9] = True
        np.save(source / "sam3_masks" / f"mask_{index:02d}.npy", mask)
    output = tmp_path / "review"
    summaries = review_all(tmp_path / "processed", output, raw)
    assert len(summaries) == 1
    assert summaries[0]["last_response_k"] == 2
    assert summaries[0]["mask_mean_percent"] == pytest.approx(6.25)
    assert len(list((output / "task").glob("overlay_*.png"))) == 32
    assert len(list((output / "task").glob("mask_*.png"))) == 32
    assert cv2.imread(str(output / "task/mask_00.png"), 0).max() == 255
    assert cv2.imread(str(output / "task/mask_00.png"), 0).min() == 0
    assert "Piper：1 段视频" in (output / "index.html").read_text()
    video.write_bytes(b"changed video")
    with pytest.raises(ValueError, match="source video changed"):
        review_all(tmp_path / "processed", output, raw)
