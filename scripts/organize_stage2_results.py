#!/usr/bin/env python3
"""Index Stage 2 deliveries and remove redundant Blender previews after completion."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _running_commands() -> str:
    return subprocess.run(["ps", "-axo", "command"], capture_output=True,
                          text=True, check=True).stdout


def _cleanup_candidates(geometry: Path, passed: bool) -> list[Path]:
    candidates = []
    for round_dir in sorted(geometry.glob("round_[0-9][0-9][0-9]")):
        if not (round_dir / "candidate.blend").is_file():
            continue
        if not (round_dir / "evaluation" / "report.json").is_file():
            continue
        for path in round_dir.iterdir():
            if not path.is_file():
                continue
            if path.suffix in (".blend1", ".blend2", ".blend3"):
                candidates.append(path)
            elif passed and path.name in ("agent_preview.blend", "self_candidate.blend",
                                               "preview.blend"):
                candidates.append(path)
    return candidates


def organize(data_root: Path, prune: bool = False,
             commands: str | None = None,
             metadata_root: Path | None = None) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve(strict=True)
    metadata_root = (metadata_root or data_root.parent.parent / "runs" / "stage2" / "metadata").expanduser().resolve()
    commands = _running_commands() if commands is None else commands
    batch_running = any("run_stage2_batch.py" in line for line in commands.splitlines())
    entries = []
    removed_count = 0
    reclaimed_bytes = 0
    for robot in ("franka", "piper"):
        stage2 = data_root / robot / "stage2"
        if not stage2.is_dir():
            continue
        for task_dir in sorted(path for path in stage2.iterdir() if path.is_dir()):
            geometry = task_dir / "geometry"
            delivery = _read_json(geometry / "stage2_manifest.json")
            session = _read_json(geometry / "session.json")
            final_scene = geometry / "final.blend"
            passed = delivery.get("accepted") is True and final_scene.is_file()
            exhausted = session.get("status") == "max_rounds_exceeded"
            active = any("real2sim.stage2.agent_runner" in line and
                         f"--output {task_dir.resolve()}" in line
                         for line in commands.splitlines())
            status = ("passed" if passed else "max_rounds_exceeded" if exhausted else
                      "running" if active else "queued" if batch_running else "interrupted")
            reports = sorted(geometry.glob("round_[0-9][0-9][0-9]/evaluation/report.json"))
            last_report = _read_json(reports[-1]) if reports else {}
            evaluation = delivery.get("evaluation", last_report) if passed else last_report
            row = {
                "robot": robot,
                "task": task_dir.name,
                "status": status,
                "rounds_evaluated": len(reports),
                "latest_evaluated_round": reports[-1].parent.parent.name if reports else None,
                "accepted_round": session.get("accepted_round"),
                "depth_mae_mean_m": evaluation.get("depth_mae_mean_m"),
                "lab_delta_e76_mean": evaluation.get("lab_delta_e76_mean"),
                "scene": str(final_scene) if passed else None,
                "plan": str(task_dir / "plan" / "plan.md"),
                "delivery_manifest": str(geometry / "stage2_manifest.json") if passed else None,
                "evaluation_report": (str(geometry / "resume_evaluation" / "report.json")
                                      if passed and session.get("accepted_round") else
                                      str(reports[-1]) if reports else None),
            }
            if prune and (passed or exhausted) and not active:
                for path in _cleanup_candidates(geometry, passed):
                    reclaimed_bytes += path.stat().st_size
                    path.unlink()
                    removed_count += 1
            entries.append(row)

    counts = Counter(row["status"] for row in entries)
    output_dir = metadata_root
    previous = _read_json(output_dir / "results_index.json")
    result = {
        "schema": "stage2-results-index-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "depth_mae_limit_m": 0.15,
        "lab_delta_e76_limit": 15.0,
        "max_refinement_rounds": 10,
        "counts": dict(counts),
        "pruned_files": removed_count,
        "reclaimed_bytes": reclaimed_bytes,
        "total_pruned_files": previous.get("total_pruned_files", previous.get("pruned_files", 0)) +
                              removed_count,
        "total_reclaimed_bytes": previous.get("total_reclaimed_bytes", previous.get("reclaimed_bytes", 0)) +
                                 reclaimed_bytes,
        "scenes": entries,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results_index.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = ["# Stage 2 结果", "", "以 `results_index.json` 为机器可读索引；各任务的",
             "`geometry/final.blend` 是已通过验收的交付场景。旧批次清单保留作历史记录，",
             "以本索引和 `resume_10_rounds_manifest.json` 查看本轮状态。", "",
             "验收要求：平均 Depth MAE < 0.15 m、平均 Lab ΔE76 < 15；",
             "单视角门槛仍按 evaluator 执行，最多 10 轮。", "",
             f"状态：{dict(counts)}", "",
             "| 机械臂 | 场景 | 状态 | 已评估轮次 | MAE (m) | Lab ΔE76 |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in entries:
        depth = row["depth_mae_mean_m"]
        lab = row["lab_delta_e76_mean"]
        lines.append(f"| {row['robot']} | {row['task']} | {row['status']} | "
                     f"{row['rounds_evaluated']} | {depth:.4f} | {lab:.2f} |"
                     if depth is not None and lab is not None else
                     f"| {row['robot']} | {row['task']} | {row['status']} | "
                     f"{row['rounds_evaluated']} | — | — |")
    lines.extend(["", "已保留每轮 `candidate.blend`、RGB/depth 渲染、误差图、报告、",
                  "Plan 与会话记录。仅在已结束的任务中清理已评估轮次的 Blender",
                  "备份；通过的任务另外清理可重建的临时预览 Blend。运行中的任务不清理。", ""])
    (output_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--metadata-root", type=Path, default=Path("runs/stage2/metadata"))
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()
    result = organize(args.data_root, args.prune, metadata_root=args.metadata_root)
    print(json.dumps({key: result[key] for key in ("counts", "pruned_files", "reclaimed_bytes")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
