#!/usr/bin/env python3
"""Export four-view SSIM metrics for accepted Stage 2 deliveries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from real2sim.stage2.evaluator import evaluate_ssim_bundle


CSV_COLUMNS = (
    "robot", "task", "camera_0_ssim", "camera_1_ssim", "camera_2_ssim",
    "camera_3_ssim", "ssim_mean", "reference", "render",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_scenes(data_root: Path):
    for robot in ("franka", "piper"):
        stage2_root = data_root / robot / "stage2"
        if not stage2_root.is_dir():
            continue
        for task_dir in sorted(path for path in stage2_root.iterdir() if path.is_dir()):
            manifest_path = task_dir / "geometry" / "stage2_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = _load(manifest_path)
            if manifest.get("accepted") is not True:
                continue
            evaluation = manifest.get("evaluation", {})
            reference = Path(evaluation["reference"])
            render = Path(evaluation["render"])
            yield robot, task_dir.name, reference, render


def export(data_root: Path, output: Path) -> list[dict[str, Any]]:
    data_root = data_root.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for robot, task, reference, render in _accepted_scenes(data_root):
        scene_output = output / robot / task
        report = evaluate_ssim_bundle(reference, render, scene_output)
        row = {
            "robot": robot,
            "task": task,
            "camera_0_ssim": report["views"][0]["ssim"],
            "camera_1_ssim": report["views"][1]["ssim"],
            "camera_2_ssim": report["views"][2]["ssim"],
            "camera_3_ssim": report["views"][3]["ssim"],
            "ssim_mean": report["ssim_mean"],
            "reference": report["reference"],
            "render": report["render"],
        }
        rows.append(row)

    with (output / "scene_ssim.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "schema": "stage2-ssim-summary-v1",
        "scene_count": len(rows),
        "camera_count": 4,
        "metric": "grayscale SSIM",
        "higher_is_better": True,
        "scenes": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Stage 2 四视角 SSIM",
        "",
        "SSIM 在 sRGB 图像经 OpenCV RGB2GRAY 转换的灰度图上计算，窗口为 11×11、高斯 σ=1.5；数值越高表示结构相似度越高。",
        "该指标是附加诊断指标，不改变当前 Depth MAE/Lab 的 Stage 2 通过条件。",
        "",
        f"场景数：{len(rows)}；每个场景包含 4 个相机视角。",
        "",
        "| 机械臂 | 场景 | Camera 0 | Camera 1 | Camera 2 | Camera 3 | 四视角均值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['robot']} | {row['task']} | "
            f"{row['camera_0_ssim']:.6f} | {row['camera_1_ssim']:.6f} | "
            f"{row['camera_2_ssim']:.6f} | {row['camera_3_ssim']:.6f} | "
            f"{row['ssim_mean']:.6f} |"
        )
    (output / "SSIM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/stage2/metadata/ssim")
    args = parser.parse_args()
    rows = export(args.data_root, args.output)
    print(json.dumps({"scene_count": len(rows), "output": str(args.output.resolve())},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
