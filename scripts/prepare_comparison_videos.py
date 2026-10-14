"""Prepare forward/reverse white-drawer loops at the original playback speed."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--sim", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    args = parser.parse_args()
    validation = json.loads(args.validation.read_text())
    real_hash = digest(args.real)
    if real_hash != validation["source_video_sha256"]:
        raise ValueError("Render validation does not match the original video")
    frame_count = validation["frame_count"]
    if frame_count < 3:
        raise ValueError("Forward/reverse loops require at least three frames")
    output = Path(__file__).resolve().parents[1] / "docs/assets/comparison"
    output.mkdir(parents=True, exist_ok=True)
    sources = {"real": args.real.resolve(), "sim": args.sim.resolve()}
    records = {}
    for label, source in sources.items():
        target = output / f"drawer-{label}-loop.mp4"
        # Skip repeated endpoints so both direction changes remain continuous.
        filters = (
            f"[0:v]scale=1280:720,trim=end_frame={frame_count},"
            "setpts=N*1001/(30000*TB),split[forward][back];"
            f"[back]reverse,trim=start_frame=1:end_frame={frame_count - 1},"
            "setpts=N*1001/(30000*TB)[reverse];"
            "[forward][reverse]concat=n=2:v=1:a=0[out]"
        )
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-filter_complex", filters, "-map", "[out]", "-an", "-map_metadata", "-1",
            "-r", "30000/1001", "-fps_mode", "cfr", "-c:v", "libx264",
            "-crf", "20", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(target),
        ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(target),
            "-frames:v", "1", "-q:v", "2", str(output / f"drawer-{label}.jpg"),
        ], check=True)
        records[label] = {"source": source.name, "source_sha256": digest(source),
                          "file": target.name, "sha256": digest(target)}
    manifest = {
        "task": "close_drawer", "robot": "franka", "model": "gpt-6-astra",
        "backend": "dsh", "source_frame_count": frame_count,
        "loop_frame_count": 2 * frame_count - 2,
        "source_max_timestamp_error_seconds": validation["max_timestamp_error_seconds"],
        "preprocessing": "1280x720 H.264, silent, constant 30000/1001 fps, original speed, forward then reverse without repeated endpoints",
        "note": validation["note"], "videos": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
