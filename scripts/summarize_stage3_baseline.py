"""Summarize the immutable Stage 3 scene and perception baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam3-host", help="service host recorded in the source experiment, if known")
    parser.add_argument("--sam3-port", type=int, help="service port recorded in the source experiment, if known")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    results = []
    for robot in ("piper", "franka"):
        report = json.loads((root / "perception_clean" / f"{robot}_batch_report.json").read_text())
        for item in report["results"]:
            task = item["task"]
            scene = root / "manifest_scenes_clean" / robot / task / "scene_for_sim.xml"
            observation = item.get("observation") or {}
            objects = observation.get("objects") or {}
            results.append({
                "robot": robot,
                "task": task,
                "scene_status": "loadable" if scene.is_file() else "missing",
                "scene_sha256": sha256(scene) if scene.is_file() else None,
                "perception_status": item["status"],
                "detected_objects": len(objects),
                "detected_names": sorted(objects),
                "report": str(root / "perception_clean" / robot / task / "perception_report.json"),
            })
    summary = {
        "protocol": "Stage 3 baseline; no policy repair or heuristic action generation",
        "scene_count": len(results),
        "scene_preparation": {
            "loadable": sum(item["scene_status"] == "loadable" for item in results),
            "missing": sum(item["scene_status"] != "loadable" for item in results),
            "source": "final_manifest.json scene field",
            "robot_assets": "robot_control-compatible Piper and FR3 mounts",
        },
        "perception": {
            "service": {"host": args.sam3_host, "port": args.sam3_port,
                        "protocol": "length-prefixed TCP"},
            "calls": len(results),
            "successful_calls": sum(item["perception_status"] == "observed" for item in results),
            "failed_calls": sum(item["perception_status"] != "observed" for item in results),
            "scenes_with_detections": sum(item["detected_objects"] > 0 for item in results),
        },
        "task_execution": {
            "status": "not_measured",
            "reason": "No executable policy bundle is stored in the 18 final manifests; model authoring timed out before producing a valid baseline policy.",
        },
        "results": results,
    }
    args.output.expanduser().resolve().write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "scene_count": summary["scene_count"],
        "loadable": summary["scene_preparation"]["loadable"],
        "perception_success": summary["perception"]["successful_calls"],
        "task_execution": summary["task_execution"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
