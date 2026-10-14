import csv
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "export_stage2_ssim.py"
SPEC = spec_from_file_location("stage2_ssim_export", SCRIPT)
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_export_only_accepted_scenes_and_selected_render(tmp_path):
    data_root = tmp_path / "data" / "processed"
    scene = data_root / "piper" / "stage2" / "selected"
    geometry = scene / "geometry"
    reference = data_root / "piper" / "stage1" / "selected"
    render = geometry / "round_002" / "render"
    render.mkdir(parents=True)
    (reference / "reference").mkdir(parents=True)
    (reference / "depth_metric").mkdir()
    rgb = np.full((16, 16, 3), 100, dtype=np.uint8)
    for ordinal, index in enumerate((0, 10, 21, 31)):
        cv2.imwrite(str(reference / "reference" / f"frame_{ordinal:02d}.jpg"), rgb)
        np.save(reference / "depth_metric" / f"depth_{index:02d}.npy",
                np.ones((16, 16), dtype=np.float32))
        cv2.imwrite(str(render / f"camera_{ordinal}_rgb.png"), rgb)
    (geometry / "stage2_manifest.json").write_text(json.dumps({
        "accepted": True, "evaluation": {"reference": str(reference), "render": str(render)},
    }), encoding="utf-8")
    rejected = data_root / "piper" / "stage2" / "rejected" / "geometry"
    rejected.mkdir(parents=True)
    (rejected / "stage2_manifest.json").write_text(json.dumps({"accepted": False}), encoding="utf-8")

    output = tmp_path / "runs" / "stage2" / "metadata" / "ssim"
    rows = MODULE.export(data_root, output)
    assert len(rows) == 1
    assert rows[0]["task"] == "selected"
    assert rows[0]["ssim_mean"] == 1.0
    assert rows[0]["render"] == str(render)
    with (output / "scene_ssim.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 1
    assert float(csv_rows[0]["camera_3_ssim"]) == 1.0
    assert (output / "piper" / "selected" / "report.json").is_file()
    assert "selected" in (output / "SSIM.md").read_text(encoding="utf-8")
