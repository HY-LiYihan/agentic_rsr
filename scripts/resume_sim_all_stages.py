"""Serially attempt every SIM stage, retaining independent verdicts and prior work."""

import json
import subprocess
import sys
import time
from pathlib import Path

from scripts import run_stage3_codex_batch as batch


RUN_NAME = "codex_all_stages_20260926_resume"


def main():
    summary = batch.SCENE_ROOT / f"{RUN_NAME}_serial_report.json"
    results = json.loads(summary.read_text()) if summary.exists() else []
    done = {(item["robot"], item["task"], item["stage"]) for item in results}
    for robot in ("piper", "franka"):
        for task_dir in sorted((batch.DATA_ROOT / robot / "stage3").iterdir()):
            if not task_dir.is_dir():
                continue
            task = task_dir.name
            previous_policy = None
            experience = []
            for stage in batch.STAGES:
                key = (robot, task, stage[0])
                base = batch.SCENE_ROOT / robot / task
                old_run = ("codex_three_stage_piper_serial_20260925" if robot == "piper"
                           else "codex_three_stage_all18_continuation_20260925")
                old = base / old_run / stage[0] / "report.json"
                existing = next((item for item in results if
                                 (item["robot"], item["task"], item["stage"]) == key), None)
                if existing:
                    report_path = Path(existing["report"])
                elif old.exists() and json.loads(old.read_text()).get("success"):
                    report_path = old
                else:
                    for attempt in range(4):
                        run_name = f"{RUN_NAME}_attempt{attempt + 1}"
                        output = base / run_name / stage[0]
                        report_path = output / "report.json"
                        if not report_path.exists():
                            print(f"START {robot}/{task}/{stage[0]} attempt={attempt + 1}", flush=True)
                            batch.run_one(robot, task, "gpt-6-astra", 240, "medium",
                                          run_name, total_timeout=1800, stage=stage,
                                          previous_policy=previous_policy,
                                          prior_experience="\n\n".join(experience) or None)
                        report = json.loads(report_path.read_text())
                        error = report.get("error", "")
                        if not any(marker in error for marker in
                                   ("429", "no response", "502", "503")):
                            break
                        if attempt < 3:
                            print(f"SERVICE RETRY {key}: {error}", flush=True)
                            time.sleep(300 * (attempt + 1))
                report = json.loads(report_path.read_text())
                output = report_path.parent
                policy = output / "bundle/policy.py"
                if policy.exists():
                    previous_policy = policy.read_text()
                notes = output / "experience.md"
                if notes.exists():
                    experience.append(notes.read_text())
                else:
                    experience.append(f"{stage[0]}: {report.get('verification')}; "
                                      f"{report.get('error', '')}. Not verified successful.")
                frames = output / "frames"
                video = output / "simulation.mp4"
                if not video.exists() and (frames / "frame_00000.png").exists():
                    subprocess.run(["ffmpeg", "-n", "-loglevel", "error", "-framerate", "30",
                                    "-i", str(frames / "frame_%05d.png"), "-c:v", "libx264",
                                    "-pix_fmt", "yuv420p", str(video)], check=False)
                if key not in done:
                    results.append({"robot": robot, "task": task, "stage": stage[0],
                                    "report": str(report_path), "success": report.get("success", False),
                                    "verification": report.get("verification"),
                                    "error": report.get("error")})
                    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2))
                    done.add(key)
                print(f"END {key}: {report.get('verification')} ({len(results)}/54)", flush=True)


if __name__ == "__main__":
    main()
