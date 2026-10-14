"""Prepare isolated uncalibrated inputs and launch the normal Stage2 batch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from real2sim.blender import resolve_blender

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--piper-asset", type=Path, required=True)
    parser.add_argument("--franka-asset", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--blender", help="Blender executable path or command; defaults to BLENDER_BIN or PATH")
    args = parser.parse_args()
    blender = resolve_blender(args.blender)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    prompt = (ROOT / "src/real2sim/stage2/refinement_prompt.md").read_text()
    start = prompt.index("## Fixed scene convention")
    end = prompt.index("## RGB inputs and final metrics")
    prompt = prompt[:start] + """## Uncalibrated initialization

This ablation omits initial robot-to-camera registration. The robot asset is in
metres at its native base origin. Preserve the robot, attached parts, geometry,
materials, dimensions and pose. There are four GENERIC, UNCALIBRATED cameras.
Estimate each camera's position, rotation, focal length and shift from the four
reference RGBs, matching the robot silhouette first, then reconstruct the scene.
You MAY change camera transforms, lens and shifts. Keep four camera scenes,
their names, reference ordering, resolution, sensor dimensions and clipping fixed.
Never read calibrated scenes, camera JSON, registration files, previous experiment
outputs or depth arrays. Use only the inputs listed here and scalar feedback.

Initial scene: {initial_scene}
Current scene: {current_scene}
Never overwrite either file; write the requested edit script instead.
The working copy has replaceable Stage2_Lighting lights, which you may adjust.
Plan document: {plan_path}
{plan_text}
Current (not calibrated) camera status:
{camera_status}

""" + prompt[end:]
    prompt = prompt.replace("freeze-check cameras and {robot_name}", "freeze-check camera identities and {robot_name}")
    prompt_path = output / "refinement_prompt.md"
    prompt_path.write_text(prompt)
    assets = {"piper": args.piper_asset.resolve(strict=True), "franka": args.franka_asset.resolve(strict=True)}
    provenance = {"model": "gpt-6-astra", "reasoning_effort": "high", "max_rounds": 10, "workers": 18, "ablation": "no initial robot-camera registration", "assets": {}}
    for robot, asset in assets.items():
        seed = output / "assets" / f"{robot}.blend"
        subprocess.run([str(blender), "-b", "--python-exit-code", "1", "--python", str(ROOT / "scripts/prepare_uncalibrated_blend.py"), "--", str(asset), str(seed), robot], check=True)
        provenance["assets"][robot] = {"path": str(asset), "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
        for source in sorted((ROOT / "data/processed" / robot / "stage1").iterdir()):
            if not source.is_dir() or not (source / "stage1_delivery.json").is_file():
                continue
            target = output / "inputs" / robot / "stage1" / source.name
            target.mkdir(parents=True)
            shutil.copy2(seed, target / "initial.blend")
            manifest = json.loads((source / "stage1_delivery.json").read_text())
            (target / "stage1_delivery.json").write_text(json.dumps({"task_description": manifest.get("task_description", source.name.replace("_", " ")), "reference_indices": manifest.get("reference_indices", [0, 10, 21, 31]), "ablation": provenance["ablation"]}, indent=2))
            for name in ("reference", "depth_metric"):
                shutil.copytree(source / name, target / name)
            for extra in (target / "reference").iterdir():
                if not extra.name.startswith("frame_"):
                    extra.unlink()
    (output / "experiment.json").write_text(json.dumps(provenance, indent=2))
    if args.prepare_only:
        return
    environment = os.environ.copy()
    environment["BLENDER_BIN"] = str(blender)
    environment.update(REAL2SIM_UNCALIBRATED="1", REAL2SIM_REFINEMENT_PROMPT=str(prompt_path), REAL2SIM_CODEX_MODEL="gpt-6-astra")
    subprocess.run([sys.executable, str(ROOT / "scripts/run_stage2_batch.py"), "--data-root", str(output / "inputs"), "--output-root", str(output / "results"), "--workers", "18", "--max-rounds", "10", "--model", "gpt-6-astra", "--manifest", str(output / "batch_manifest.json")], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
