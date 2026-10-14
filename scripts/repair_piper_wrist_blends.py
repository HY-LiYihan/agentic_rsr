"""Correct the pinned Piper stand orientation in accepted Stage 1 and Stage 2 Blends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from real2sim.blender import resolve_blender
from real2sim.stage1.geometry import MODEL
from real2sim.stage1.piper_wrist import STAND_MESH, piper_description, stand_mesh_correction, stand_mesh_parts
from real2sim.stage1.robot_surface import BASE_LINKS, POSES, link_poses


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src/real2sim/stage1/repair_piper_wrist_blend.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _bounds(mesh: trimesh.Trimesh, frame: np.ndarray) -> list[list[float]]:
    transformed = mesh.copy()
    transformed.apply_transform(frame)
    return transformed.bounds.tolist()


def repair_spec() -> dict:
    scene = trimesh.load(STAND_MESH, force="scene")
    original = scene.dump(concatenate=False)[0]
    corrected = stand_mesh_parts()[0]
    root = piper_description()
    frames, _ = link_poses(root, BASE_LINKS["piper"], POSES["piper"])
    stand_frame = frames["camera_stand_link"]
    correction = stand_frame @ stand_mesh_correction() @ np.linalg.inv(stand_frame)
    return {"correction": correction.tolist(), "before_bounds": _bounds(original, stand_frame),
            "after_bounds": _bounds(corrected, stand_frame),
            "vertex_count": len(original.vertices), "face_count": len(original.faces)}


def _update_stage2_metadata(task: Path, old_hash: str, new_hash: str, backup: Path,
                            stand_dimensions: list[float]) -> None:
    geometry = task / "geometry"
    manifest_path = geometry / "stage2_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["scene_sha256"] != old_hash:
        raise ValueError(f"stale Stage 2 manifest: {manifest_path}")
    shutil.copy2(manifest_path, backup / manifest_path.name)
    manifest["scene_sha256"] = new_hash
    _write_json(manifest_path, manifest)
    metrics_path = task.parent.parent / "stage1" / task.name / "object_metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if Path(metrics["source_blend"]).resolve() == (geometry / "final.blend").resolve():
            if metrics["blend_sha256"] != old_hash:
                raise ValueError(f"stale Stage 1 object metrics: {metrics_path}")
            shutil.copy2(metrics_path, backup / metrics_path.name)
            metrics["blend_sha256"] = new_hash
            _write_json(metrics_path, metrics)
    for name in ("frozen_signature.json", "inspect.json"):
        path = geometry / name
        if not path.is_file():
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        signature = record if name == "frozen_signature.json" else record["frozen_signature"]
        stand = next((row for row in signature["robot"] if row["name"] == "_г_--__3_001-mesh"), None)
        if stand is None:
            raise ValueError(f"stand is absent in {path}")
        shutil.copy2(path, backup / name)
        stand["dimensions"] = stand_dimensions
        signature["robot_hash"] = hashlib.sha256(json.dumps(
            signature["robot"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
        _write_json(path, record)


def finalize_validation(output: Path) -> int:
    repair = json.loads((output / "report.json").read_text(encoding="utf-8"))
    stage2_rows = [row for row in repair["scenes"] if row["stage"] == "stage2"]
    if len(stage2_rows) != 9:
        raise ValueError("expected nine repaired Stage 2 deliveries")
    validation = output / "validation"
    for row in stage2_rows:
        task = row["task"]
        scene = Path(row["scene"])
        manifest_path = scene.parent / "stage2_manifest.json"
        if _sha256(scene) != row["new_sha256"]:
            raise ValueError(f"Stage 2 scene changed after repair: {scene}")
        source = validation / task
        report = json.loads((source / "evaluation/report.json").read_text(encoding="utf-8"))
        if (report.get("passed") is not True or len(report["views"]) != 4 or
                not all(view["depth_passed"] and view["lab_passed"] for view in report["views"])):
            raise ValueError(f"corrected scene did not pass Stage 2 evaluation: {task}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["scene_sha256"] != row["new_sha256"]:
            raise ValueError(f"Stage 2 manifest changed since repair: {task}")
    for row in stage2_rows:
        task = row["task"]
        scene = Path(row["scene"])
        manifest_path = scene.parent / "stage2_manifest.json"
        source = validation / task
        destination = scene.parent
        for kind in ("render", "evaluation"):
            folder = destination / f"wrist_repair_{kind}"
            if folder.exists():
                raise FileExistsError(folder)
            shutil.copytree(source / kind, folder)
        serialized = (source / "evaluation/report.json").read_text(encoding="utf-8")
        serialized = serialized.replace(str(source / "render"), str(destination / "wrist_repair_render"))
        serialized = serialized.replace(str(source / "evaluation"), str(destination / "wrist_repair_evaluation"))
        report = json.loads(serialized)
        _write_json(destination / "wrist_repair_evaluation/report.json", report)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["evaluation"] = report
        manifest["wrist_mount_repair"] = {"original_scene_sha256": row["old_sha256"],
                                          "backup": row["backup"]}
        _write_json(manifest_path, manifest)
        print(f"stage2 {task}: validated and published", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory under runs/ for backups and report")
    parser.add_argument("--blender", help="Blender executable path or command; defaults to BLENDER_BIN or PATH")
    parser.add_argument("--finalize-validation", action="store_true",
                        help="publish previously validated four-view render and metrics alongside repaired Blends")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.finalize_validation:
        return finalize_validation(output)
    blender = resolve_blender(args.blender)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    spec = repair_spec()
    targets = []
    for stage in ("stage1", "stage2"):
        pattern = "*/initial.blend" if stage == "stage1" else "*/geometry/final.blend"
        targets += [(stage, path) for path in sorted((ROOT / "data/processed/piper" / stage).glob(pattern))]
    if len(targets) != 18:
        raise ValueError(f"expected nine Stage 1 and nine Stage 2 Piper scenes: found {len(targets)}")
    records = []
    for stage, path in targets:
        task = path.parent if stage == "stage1" else path.parent.parent
        backup = output / stage / task.name
        backup.mkdir(parents=True)
        old_hash = _sha256(path)
        if stage == "stage2":
            manifest = json.loads((task / "geometry/stage2_manifest.json").read_text(encoding="utf-8"))
            if manifest["scene_sha256"] != old_hash:
                raise ValueError(f"Stage 2 manifest does not match: {path}")
        shutil.copy2(path, backup / path.name)
        candidate = path.with_name(f"{path.stem}.wrist-repair.blend")
        if candidate.exists():
            raise FileExistsError(candidate)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False) as handle:
            config = Path(handle.name)
            _write_json(config, {**spec, "output": str(candidate)})
        log = backup / "blender.log"
        try:
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run([str(blender), "--background", str(path), "--python-exit-code", "1",
                                         "--python", str(WORKER), "--", str(config)],
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=180)
            if result.returncode or not candidate.is_file():
                raise RuntimeError(f"Blender failed repairing {path}; see {log}")
            line = next((value.split("REPAIRED_STAND ", 1)[1] for value in log.read_text().splitlines()
                         if "REPAIRED_STAND " in value), None)
            if line is None:
                raise RuntimeError(f"Blender did not confirm {path}; see {log}")
            dimensions = json.loads(line)["dimensions"]
            os.replace(candidate, path)
            new_hash = _sha256(path)
            if stage == "stage2":
                _update_stage2_metadata(task, old_hash, new_hash, backup, dimensions)
            else:
                metrics = task / "object_metrics.json"
                if metrics.is_file():
                    record = json.loads(metrics.read_text(encoding="utf-8"))
                    if record.get("blend_sha256") == old_hash:
                        shutil.copy2(metrics, backup / metrics.name)
                        record["blend_sha256"] = new_hash
                        _write_json(metrics, record)
            records.append({"stage": stage, "task": task.name, "scene": str(path),
                            "old_sha256": old_hash, "new_sha256": new_hash,
                            "backup": str(backup / path.name)})
            _write_json(output / "report.json", {"repair": "piper_stand_raw_geometry_v1", "scenes": records})
            print(f"{stage} {task.name}: repaired", flush=True)
        finally:
            config.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
