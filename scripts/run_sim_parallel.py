"""Run independent scenes concurrently and retain every stage's evidence."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/sim_processed_stage3_all"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def worker(args):
    from scripts import run_stage3_codex_batch as batch

    directory = OUTPUT / args.run_name / f"{args.robot}__{args.task}"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    previous_policy = None
    experience = []
    prior_records = {}
    if args.retry_from:
        prior_directory = OUTPUT / args.retry_from
        jobs = json.loads((prior_directory / "jobs.json").read_text())
        prior_job = next(job for job in jobs if job["robot"] == args.robot and job["task"] == args.task)
        while True:
            process = subprocess.run(["ps", "-p", str(prior_job["pid"]), "-o", "command="],
                                     capture_output=True, text=True)
            if "scripts.run_sim_parallel" not in process.stdout:
                break
            write_json(directory / "progress.json", {"status": "waiting_for_original"})
            time.sleep(10)
        prior_file = prior_directory / f"{args.robot}__{args.task}" / "stages.json"
        prior_records = {record["stage"]: record for record in json.loads(prior_file.read_text())}
    for stage in batch.STAGES:
        prior = prior_records.get(stage[0])
        if prior and "额度不足" not in (prior.get("error") or ""):
            records.append({**prior, "reused": True})
            prior_output = Path(prior["report"]).parent
            policy = prior_output / "bundle/policy.py"
            if policy.exists():
                previous_policy = policy.read_text()
            notes = prior_output / "experience.md"
            if notes.exists():
                experience.append(notes.read_text())
            write_json(directory / "stages.json", records)
            continue
        started = time.time()
        write_json(directory / "progress.json", {"stage": stage[0], "status": "running"})
        stage_output = OUTPUT / args.robot / args.task / args.run_name / stage[0]
        try:
            report = batch.run_one(
                args.robot, args.task, "gpt-6-astra", 240, "medium", args.run_name,
                total_timeout=1800, stage=stage, previous_policy=previous_policy,
                prior_experience="\n\n".join(experience) or None)
        except Exception as exc:
            report = {"error": f"{type(exc).__name__}: {exc}", "status": "infrastructure_error"}
        policy = stage_output / "bundle/policy.py"
        if policy.exists():
            previous_policy = policy.read_text()
        notes = stage_output / "experience.md"
        experience.append(notes.read_text() if notes.exists() else
                          f"{stage[0]} ended: {report.get('error', report.get('status'))}")
        frames = sorted((stage_output / "frames").glob("frame_*.png"))
        observations = sorted(stage_output.glob("**/*third_person.jpg"),
                              key=lambda path: path.stat().st_mtime)
        tail = frames[-1] if frames else (observations[-1] if observations else None)
        saved = None
        if tail:
            saved = directory / f"{stage[0]}_last{tail.suffix}"
            shutil.copy2(tail, saved)
        record = {"robot": args.robot, "task": args.task, "stage": stage[0],
                  "elapsed_s": round(time.time() - started, 1),
                  "report": str(stage_output / "report.json"),
                  "error": report.get("error"), "status": report.get("status"),
                  "last_frame": str(saved) if saved else None,
                  "video": report.get("video")}
        records.append(record)
        write_json(directory / "stages.json", records)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    write_json(directory / "progress.json", {"status": "finished", "stages": len(records)})


def launch(args):
    directory = OUTPUT / args.run_name
    directory.mkdir(parents=True, exist_ok=False)
    jobs = []
    for robot in ("piper", "franka"):
        for task in sorted((ROOT / "data/processed" / robot / "stage3").iterdir()):
            if not task.is_dir():
                continue
            log = directory / f"{robot}__{task.name}.log"
            if args.retry_from:
                prior = OUTPUT / args.retry_from / f"{robot}__{task.name}"
                stages = json.loads((prior / "stages.json").read_text())
                progress = json.loads((prior / "progress.json").read_text())
                if progress.get("status") == "finished" and not any(
                        "额度不足" in (stage.get("error") or "") for stage in stages):
                    continue
            with log.open("w") as stream:
                process = subprocess.Popen(
                    [sys.executable, "-u", "-m", "scripts.run_sim_parallel",
                     "--run-name", args.run_name, "--robot", robot, "--task", task.name]
                    + (["--retry-from", args.retry_from] if args.retry_from else []),
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
            jobs.append({"robot": robot, "task": task.name, "pid": process.pid, "log": str(log)})
    write_json(directory / "jobs.json", jobs)
    print(json.dumps(jobs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--robot")
    parser.add_argument("--task")
    parser.add_argument("--retry-from")
    arguments = parser.parse_args()
    if arguments.robot:
        worker(arguments)
    else:
        launch(arguments)
