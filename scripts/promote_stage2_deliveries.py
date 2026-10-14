#!/usr/bin/env python3
"""Keep only accepted Stage 2 deliveries in data/processed and archive the rest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RERUN_ID = "20260925_rerun_01"
RERUN_SCENES = {
    ("franka", "stack_snacks"),
    ("franka", "store_three_tools"),
    ("piper", "put_brown_block_in_white_and_green_storage_box"),
    ("piper", "stack_cones"),
    ("piper", "take_out_tissue"),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _accepted_manifest(path: Path) -> dict[str, Any]:
    manifest = _json(path)
    if manifest.get("accepted") is not True:
        raise ValueError(f"not an accepted Stage 2 delivery: {path}")
    scene = Path(manifest.get("scene", ""))
    if not scene.is_file():
        raise FileNotFoundError(f"missing accepted final scene: {scene}")
    evaluation = manifest.get("evaluation", {})
    if evaluation.get("passed") is not True:
        raise ValueError(f"accepted manifest has a failing evaluation: {path}")
    return manifest


def _rewrite_tree(root: Path, old: str, new: str) -> int:
    changed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rewritten = text.replace(old, new)
        if rewritten != text:
            path.write_text(rewritten, encoding="utf-8")
            changed += 1
    return changed


def _move_with_backup(source: Path, destination: Path, archive: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    if destination.exists():
        if archive.exists():
            raise FileExistsError(f"archive already exists: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), str(archive))
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))


def promote(data_root: Path, runs_root: Path, dry_run: bool = False,
            metadata_root: Path | None = None) -> dict[str, Any]:
    data_root = data_root.expanduser().resolve(strict=True)
    runs_root = runs_root.expanduser().resolve()
    metadata_root = (metadata_root or runs_root / "metadata").expanduser().resolve()
    archive_root = runs_root / "archive"
    actions: list[dict[str, str]] = []

    for robot, task in sorted(RERUN_SCENES):
        source = runs_root / task / RERUN_ID
        destination = data_root / robot / "stage2" / task
        archive = archive_root / "pre_rerun_processed" / robot / task
        source_manifest = source / "geometry/stage2_manifest.json"
        manifest = _accepted_manifest(source_manifest)
        actions.append({"action": "promote", "robot": robot, "task": task,
                        "source": str(source), "destination": str(destination),
                        "archive": str(archive) if destination.exists() else ""})
        if dry_run:
            continue
        _move_with_backup(source, destination, archive)
        old = str(source)
        new = str(destination)
        _rewrite_tree(destination, old, new)
        updated = _json(destination / "geometry/stage2_manifest.json")
        final = Path(updated["scene"])
        if final != destination / "geometry/final.blend" or not final.is_file():
            raise ValueError(f"promoted manifest does not point to local final: {updated}")

    if not dry_run:
        for robot, task in sorted(RERUN_SCENES):
            destination = data_root / robot / "stage2" / task
            manifest_path = destination / "geometry/stage2_manifest.json"
            if manifest_path.exists():
                continue
            raise FileNotFoundError(f"missing promoted delivery: {manifest_path}")

    non_delivery: list[tuple[str, str]] = []
    for robot in ("franka", "piper"):
        stage2 = data_root / robot / "stage2"
        for task_dir in sorted(path for path in stage2.iterdir() if path.is_dir()):
            manifest = task_dir / "geometry/stage2_manifest.json"
            final = task_dir / "geometry/final.blend"
            if manifest.is_file() and final.is_file() and _json(manifest).get("accepted") is True:
                continue
            non_delivery.append((robot, task_dir.name))
            archive = archive_root / "non_deliveries" / robot / task_dir.name
            actions.append({"action": "archive_non_delivery", "robot": robot,
                            "task": task_dir.name, "source": str(task_dir),
                            "destination": str(archive), "archive": ""})
            if dry_run:
                continue
            _move_with_backup(task_dir, archive, archive.with_name(archive.name + ".previous"))
            _rewrite_tree(archive, str(task_dir), str(archive))

    if not dry_run:
        index = metadata_root / "delivery_layout.json"
        index.parent.mkdir(parents=True, exist_ok=True)
        accepted = []
        for robot in ("franka", "piper"):
            for task_dir in sorted((data_root / robot / "stage2").iterdir()):
                if not task_dir.is_dir():
                    continue
                manifest_path = task_dir / "geometry/stage2_manifest.json"
                if not manifest_path.is_file():
                    continue
                manifest = _json(manifest_path)
                if manifest.get("accepted") is True:
                    accepted.append({"robot": robot, "task": task_dir.name,
                                     "scene": str(task_dir / "geometry/final.blend"),
                                     "manifest": str(manifest_path),
                                     "evaluation": manifest["evaluation"]})
        index.write_text(json.dumps({
            "schema": "stage2-delivery-layout-v1",
            "data_root": str(data_root),
            "accepted_scene_count": len(accepted),
            "accepted_scenes": accepted,
            "non_delivery_archive": str(archive_root / "non_deliveries"),
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"dry_run": dry_run, "actions": actions,
            "promoted_count": sum(action["action"] == "promote" for action in actions),
            "archived_non_delivery_count": sum(action["action"] == "archive_non_delivery" for action in actions),
            "non_delivery": non_delivery}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/stage2")
    parser.add_argument("--metadata-root", type=Path,
                        help="directory for delivery metadata; defaults to runs-root/metadata")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = promote(args.data_root, args.runs_root, args.dry_run, args.metadata_root)
    print(json.dumps({key: value for key, value in result.items() if key != "actions"},
                     ensure_ascii=False, indent=2))
    for action in result["actions"]:
        print(f"{action['action']}: {action['robot']}/{action['task']} -> {action['destination']}")


if __name__ == "__main__":
    main()
