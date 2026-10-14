#!/usr/bin/env python3
"""Move unfinalized Stage 3 work out of processed deliveries into runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".log", ".txt", ".xml", ".py"}


def archive(root: Path, dry_run: bool = False) -> dict:
    root = root.resolve()
    runs = root / "runs/stage3"
    moves = []
    for robot in ("piper", "franka"):
        source_root = root / "data/processed" / robot / "stage3"
        if not source_root.is_dir():
            continue
        for source in sorted(path for path in source_root.iterdir() if path.is_dir()):
            destination = runs / "workspace" / robot / source.name
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing Stage 3 run: {destination}")
            moves.append((source, destination))
    result = {"moved": [{"from": str(source), "to": str(destination)}
                        for source, destination in moves], "dry_run": dry_run}
    if dry_run:
        return result
    replacements = [(str(source), str(destination)) for source, destination in moves]
    replacements += [(str(source.relative_to(root)), str(destination.relative_to(root)))
                     for source, destination in moves]
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
    for robot in ("piper", "franka"):
        source_root = root / "data/processed" / robot / "stage3"
        if source_root.is_dir() and not any(source_root.iterdir()):
            source_root.rmdir()
    updated = 0
    for path in runs.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rewritten = content
        for old, new in replacements:
            rewritten = rewritten.replace(old, new)
        if rewritten != content:
            path.write_text(rewritten, encoding="utf-8")
            updated += 1
    result["rewritten_text_files"] = updated
    (runs / "archive_manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(archive(args.root, args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
