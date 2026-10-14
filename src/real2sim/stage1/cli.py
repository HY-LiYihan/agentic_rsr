"""CLI entry point for Piper Stage 1."""

import argparse
from pathlib import Path

from real2sim.stage1.pipeline import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and validate a Piper Real2Sim Stage 1 candidate")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blender", help="Blender executable path or command; defaults to BLENDER_BIN or PATH")
    parser.add_argument("--metric-scale", type=float, default=None,
                        help="Externally calibrated meters per VGGT unit; omit to force rejection")
    args = parser.parse_args()
    folder, passed = run(args.video, args.task, args.output,
                         metric_scale=args.metric_scale, blender=args.blender)
    print(f"Stage 1 {'passed' if passed else 'failed'}: {folder} (see manifest.json)")
    if not passed:
        parser.exit(1)


if __name__ == "__main__":
    main()
