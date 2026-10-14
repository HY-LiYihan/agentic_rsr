# Scripts

Run scripts from the repository root. Install the project in editable mode, or
set `PYTHONPATH=src` when using a separate Python environment. Stage 1 command
entries are declared in `pyproject.toml`; Stage 3 export and physics commands are
documented in `src/real2sim/stage3/README.md`.

Blender-dependent scripts use the same executable discovery as the stages:
`--blender`, then `BLENDER_BIN`, then PATH, then the macOS application on macOS.
The uncalibrated batch propagates its selected executable to the Stage 2 workers.

## Stage 2

| Script | Purpose |
| --- | --- |
| `run_stage2_batch.py` | Run or resume reconstruction jobs; select a model and an independent output root. |
| `run_stage2_uncalibrated.py` | Prepare and run an isolated camera-initialization ablation from supplied robot assets. |
| `prepare_uncalibrated_blend.py` | Blender worker used by the uncalibrated experiment. |
| `export_stage2_experiment.py` | Export experiment tables and supporting artifacts. |
| `export_stage2_ssim.py` | Export four-view SSIM for accepted deliveries. |
| `organize_stage2_results.py` | Organize run metadata; `--prune` also removes intermediate files. |
| `promote_stage2_deliveries.py` | Promote accepted scenes to processed data; inspect `--dry-run` first. |

## Stage 3 And Sim

| Script | Purpose |
| --- | --- |
| `prepare_sim_scene.py` / `prepare_all_sim_scenes.py` | Derive runtime scenes from Stage 3 exports. |
| `run_stage3_codex_batch.py` | Develop policies through privileged, visual, and randomized stages, retaining verification evidence. |
| `stage3_cli.py` | Exercise a persistent simulation through JSON commands and a Unix socket. |
| `run_sim_parallel.py` | Launch independent scene jobs concurrently. |
| `resume_sim_all_stages.py` | Resume the experiment's stage attempts serially. |
| `run_sim_codex_probe.py` | Develop and evaluate a policy bundle on a selected scene. |
| `evaluate_sim_exact_suite.py` | Evaluate a supplied bundle against the external exact-suite reference. |
| `summarize_stage3_baseline.py` | Summarize saved scene and perception reports. |
| `archive_stage3_runs.py` | Move unfinished work out of processed deliveries; inspect `--dry-run` first. |

SIM execution needs the `sim` dependencies, the `external/robot_control` Python
package and its assets. Agent runs also need the `agent` dependencies and model
authentication. RGB-D perception shares the Stage 1 SAM3 configuration: local is
the default and requires the local dependencies, weights and a supported device;
only explicit `SAM3_BACKEND=remote` uses `SAM3_HOST` / `SAM3_PORT`. These scripts
can start model calls or update run directories; they are not installation checks.

For an end-to-end command sequence and required scene inputs, see
[the public workflow](../docs/WORKFLOW.md) and
[data contracts](../docs/DATA_CONTRACT.md). The three-pass batch helper discovers
the registered tasks in `data/processed`; `stage3_cli.py --scene ... --robot ...`
is the entry point for inspecting a separately prepared runtime scene. Local
perception does not make Codex agent calls offline.

## Visualization And Repairs

| Script | Purpose |
| --- | --- |
| `visualize_stage1_arm_cloud.py` | Render metric robot point clouds, including a view without axes. |
| `visualize_stage3_grasp.py` | Produce scene-specific grasp renders and workflow illustrations. |
| `visualize_sim_profiles.py` | Illustrate evaluation profiles. |
| `preview_sim_randomization.py` | Preview tabletop randomization using the committed primitive fixture or a supplied compatible scene. |
| `repair_piper_wrist_blends.py` | Repair saved wrist-camera geometry, retaining backups. |

Visualization scripts can require `matplotlib` in addition to project extras.
Both tabletop visualization scripts default to
`tests/fixtures/stage3/code_as_policy_tabletop.xml`; they do not require a
separate `code_as_policy` checkout. These fixtures exercise simulation behavior,
not the paper's reconstructed task scenes.
Some experiment helpers still assume specific local datasets or external
checkouts. See `docs/OPEN_SOURCE_PREPARATION.md` before treating them as portable
public entry points.
