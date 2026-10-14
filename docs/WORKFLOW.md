# Public Workflow

This runbook maps the paper's video-to-scene-to-policy workflow to the commands
available in this checkout. Run commands from the repository root. Use a new run
name for each experiment; the examples below use `runs/public_example` and do not
replace the historical deliveries.

The paper's main reconstruction experiments use DeepSeek Harness (DSH) and Claude
Opus 5.5. This repository currently exposes the Codex implementation described in
the paper's historical reconstruction appendix. The paper's aggregate numbers
are research results, not a guarantee that a fresh installation reproduces them.

## Prerequisites

| Component | Required for |
| --- | --- |
| Linux x86_64, compatible NVIDIA driver, Python 3.11 | First supported complete local perception environment |
| VGGT-Omega and SAM3 checkpoint access | Fresh video depth/cameras and segmentation |
| Blender executable | Stage 1 scene delivery; Stage 2 inspect/edit/render; Stage 3 mesh export |
| `external/robot_control` assets and Python package | URDF metric anchor, robot MJCF and SIM runtime |
| MuJoCo and rendering backend | Stage 3 physics checks and SIM |
| Authenticated Codex CLI/app-server and model access | Stage 2 planning/refinement, Stage 3 semantics/physics agents, SIM policy generation |
| Input video, task description and matching fixed robot pose | Scene-specific reconstruction |

```bash
git clone --recurse-submodules https://github.com/HY-LiYihan/agentic_rsr.git
cd agentic_rsr
git submodule update --init --recursive
uv sync --locked --python 3.11 --extra local --extra sim --extra agent --extra dev
# For Blender installed outside PATH, set its actual executable:
# export BLENDER_BIN=/path/to/blender
export VGGT_OMEGA_BACKEND=local VGGT_OMEGA_DEVICE=cuda
export SAM3_BACKEND=local SAM3_DEVICE=cuda
```

Follow [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md) to obtain the two official
checkpoints and run `uv run --no-sync real2sim-models check`. No inference command
downloads missing weights or switches to a remote perception service. Agent
model calls remain online even when the perception backends are local.

All Blender stages use an explicit `--blender` override first, then `BLENDER_BIN`,
then `blender` on PATH, with the macOS application as a final fallback only on
macOS. Invalid explicit settings fail rather than selecting a different install.
The commands below inherit that choice and do not require a service address.

On headless Linux, configure MuJoCo rendering with `MUJOCO_GL=egl` before SIM and
rendering checks. macOS development and CPU/cache checks are separate from CUDA
acceptance; see [STAGE1_CPU_VALIDATION.md](STAGE1_CPU_VALIDATION.md).

## 1. Video, Perception And Robot Registration

Use at least 32 decodable frames of the captured workspace with the robot static
in the supported reference pose. Review `POSES` in
`src/real2sim/stage1/robot_surface.py` before recording: the robot dimensions and
joint pose form the metric anchor. A changing robot pose cannot be explained by
one rigid Sim(3) transform.

The current registration batch commands discover and hash videos in
`data/raw/piper/<task>.mp4` or `data/raw/frank/<task>.mp4` (the raw Franka directory
is spelled `frank`). Put your own video in that layout. The following Piper
example assumes `data/raw/piper/take_out_tissue.mp4` and a new output root:

```bash
uv run --no-sync real2sim-robot-surfaces --robot piper
uv run --no-sync real2sim-stage1-infer-batch \
  --only take_out_tissue --processed-root runs/public_example/processed
uv run --no-sync real2sim-stage1-init-batch --robot piper --rotations 128 \
  --processed runs/public_example/processed \
  --output runs/public_example/initializations
uv run --no-sync real2sim-stage1-piper-top3 --only take_out_tissue --workers 3 \
  --processed runs/public_example/processed \
  --initializations runs/public_example/initializations/piper \
  --top1 runs/public_example/unused_top1 \
  --output runs/public_example/registration/piper
uv run --no-sync real2sim-stage1-piper-deliver --only take_out_tissue \
  --processed runs/public_example/processed/piper/stage1 \
  --results runs/public_example/registration/piper \
  --initializations runs/public_example/initializations/piper
```

The template command writes generated surface caches to `asset/robot_surfaces`;
do not regenerate them during an in-progress experiment. Registration/delivery
checks validate their source assets and provenance. Delivery adds metric depth,
robot-base cameras and `initial.blend` to the new perception scene directory.

For Franka, infer `close_drawer`, generate `--robot franka --rotations 48`
initializations, then run:

```bash
uv run --no-sync real2sim-stage1-franka-final --only close_drawer \
  --processed runs/public_example/processed \
  --initializations runs/public_example/initializations/franka \
  --output runs/public_example/registration/franka
uv run --no-sync real2sim-stage1-franka-deliver --only close_drawer \
  --processed runs/public_example/processed/franka/stage1 \
  --results runs/public_example/registration/franka \
  --initializations runs/public_example/initializations/franka
```

The current implementation uses Piper 128 rotations/Top-3 and historical Franka
48 rotations/Top-1; the paper's summarized 128/Top-3 setting does not describe
both current entry points. These production paths fit and score all 32 frames.
Their output remains `candidate_unverified`, `accepted=false`: metric anchoring
and a readable Blender file do not establish independent registration accuracy.
Inspect `quality_check.png`, source hashes, depth and camera alignment before
continuing with exploratory reconstruction. The phased Stage 2 implementation
can consume an unverified Stage 1 delivery; it does not independently validate
registration or make that flag a proof of acceptance.

For perception only, `real2sim-local-example --video /path/to/task.mp4 --robot
piper --output runs/new_perception_example` accepts an arbitrary video path and
produces previews and a runtime report. It does not run robot registration.

## 2. Feedback-Guided Blender Reconstruction

Codex must already be authenticated in the environment used by the SDK. Check
`codex login status` before starting. The default model is configured in
`src/real2sim/stage2/agent_runner.py`; `REAL2SIM_CODEX_MODEL` can select a model
available to your account. Record the actual model in the run output.

```bash
uv run --no-sync real2sim-stage2 --phase all \
  --input runs/public_example/processed/piper/stage1/take_out_tissue \
  --output runs/public_example/processed/piper/stage2/take_out_tissue \
  --task "take out a tissue" --max-rounds 10
```

Planning writes `plan/plan.md`; refinement freezes the robot and four reference
cameras, edits a private scene copy, renders and evaluates each candidate. Inspect
`run_manifest.json` and `geometry/stage2_manifest.json`, rather than relying on
the CLI exit code: exhausting the round budget is a recorded research outcome
and may return exit code zero. Only a delivery with `accepted=true`, passed
joint evaluation and matching `scene_sha256` is eligible for Stage 3.

The joint evaluator requires mean depth MAE below 0.15 m and mean Lab Delta E76
below 15, plus depth MAE below 0.20 m, Lab below 30 and depth coverage at least
0.50 in every reference view. Depth is compared to the Stage 1 reconstructed
metric reference, not an independently measured ground truth.

To inspect the baseline without a new model turn, use `--phase geometry
--max-rounds 0 --plan /path/to/reviewed_plan.md` with a new output directory.
`--phase all --max-rounds 0` still invokes the planning agent. Detailed phase,
resume and batch behavior is in [the Stage 2 guide](../src/real2sim/stage2/README.md).

Reusing a nonempty `geometry/` requires explicit `--phase geometry --resume`.
Resume verifies the saved input, prepared scene and render hashes before
recomputing acceptance metrics. A changed reference, plan, scene or evaluation
implementation requires a new output directory. Historical deliveries remain
readable, but runs without `geometry/input_provenance.json` cannot be resumed.

## 3. MuJoCo Export And Physical Checks

```bash
uv run --no-sync real2sim-stage3-export \
  --stage2 runs/public_example/processed/piper/stage2/take_out_tissue \
  --output runs/public_example/stage3/piper/take_out_tissue/mujoco
uv run --no-sync real2sim-stage3-physics inspect \
  --scene runs/public_example/stage3/piper/take_out_tissue/mujoco/scene.xml
uv run --no-sync real2sim-stage3-physics auto \
  --scene runs/public_example/stage3/piper/take_out_tissue/mujoco/scene.xml \
  --output runs/public_example/stage3/piper/take_out_tissue/probes
```

The accepted Stage 2 entry calls a semantics agent unless you supply reviewed
`--semantics /path/to/semantics.json` matching the source Blend SHA256. Raw
`--blend` export is a separate debugging path using inferred names, not equivalent
to accepted Stage 2 export. Keep `scene.xml` with its entire `assets/` directory.

For agent-led physical review, prepare a hypothesis JSON that distinguishes
visual estimates from measured values and invoke `real2sim-stage3-agent --scene
... --hypotheses ... --output runs/new_physics_run`. The format is illustrated in
`tests/fixtures/stage3/code_as_policy_hypotheses.json`. See
[the Stage 3 guide](../src/real2sim/stage3/README.md) for settling and grasp probes.
Simulation checks may establish numerical and interaction plausibility, not
real-world mass/friction calibration or successful tissue extraction.

## 4. SIM Visual Observation And Policy Development

Derive a runtime scene that allows robot_control to insert its controlled robot:

```bash
uv run --no-sync python scripts/prepare_sim_scene.py \
  --source runs/public_example/stage3/piper/take_out_tissue/mujoco/scene.xml \
  --output runs/public_example/sim/piper/take_out_tissue/scene_for_sim.xml
MUJOCO_GL=egl uv run --no-sync python scripts/stage3_cli.py \
  --scene runs/public_example/sim/piper/take_out_tissue/scene_for_sim.xml \
  --robot piper --task take_out_tissue \
  --image-dir runs/public_example/sim/observations
```

Enter `state`, `observe`, then `quit`. `observe` performs SAM3 inference using
the configured local backend and returns estimated object locations plus wrist
and third-person image paths. An explicit name-to-prompt mapping must use the
scene's public object names. A missed detection omits that object. This check
tests visual observation, not policy success, and needs SAM3 weights and a
supported inference device.

The existing three-pass batch helper is intended for the registered task catalog
and uses fixed discovery/output roots. It expects delivered Stage 3 tasks under
`data/processed/<robot>/stage3/<task>` and derived scenes under
`runs/sim_processed_stage3_all/<robot>/<task>/scene_for_sim.xml`. Prepare each
catalog runtime scene with `scripts/prepare_sim_scene.py`, using the reviewed
final XML from its Stage 3 manifest. The historical `prepare_all_sim_scenes.py`
helper assumes every source is `mujoco_v8/scene.xml` and has no argument parser;
it must not be used to discover a repaired final candidate. After preparing and
reviewing those inputs, a catalog run is:

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/run_stage3_codex_batch.py \
  --robot piper --task take_out_tissue --run-name public_example \
  --total-timeout 1800 --random-seed 20260926
```

It develops a policy through `privileged_exact`, `visual_exact` and
`visual_randomized`, preserving policy/experience files and execution evidence.
These are development passes. Held-out evaluation and video review are separate;
the policy claiming completion is not an independent task-success proof. The
catalog verifier is task-specific; arbitrary new videos do not automatically
gain success criteria or a general benchmark.

## Current Release Boundary

No test CI is enabled in this repository; GitHub Pages publishes the static project site. Run offline tests
locally using `uv run --no-sync python -m pytest tests/unit tests/integration -m
'not models' -ra`; missing local scenes and real model inputs must be reported as
skips. Actual model acceptance uses the explicitly marked tests and inputs in
[LOCAL_INFERENCE.md](LOCAL_INFERENCE.md).

Public weights and recorded task videos are separate inputs, not checked-in
dependencies. The fresh Linux CUDA checkpoint run and full paper reproduction
are still unverified. Stage 1 registration has no independent accuracy acceptance;
Stage 3 physical properties are provisional; a complete public Sim2Real safety
and hardware execution recipe is not implemented. These limitations must remain
visible in release reports. Original project code uses Apache-2.0; assets and
derived artifacts will be distributed through Hugging Face as described in
[ASSET_DISTRIBUTION.md](ASSET_DISTRIBUTION.md). See [THIRD_PARTY.md](THIRD_PARTY.md) before redistributing dependencies,
weights or derived robot/media assets.
