# Data Contracts

This document describes the current public implementation, including the
historical formats still supported by the readers. Paths in JSON may be absolute
and tied to the checkout that produced the run. Moving an artifact requires
preserving its associated files and resolving recorded paths; copying a single
manifest or XML does not make it self-contained.

## Capture And Pixel Grid

The input is a video containing at least 32 decodable frames and a task
description. Frame indices are `linspace(0, frame_count - 1, 32).astype(int)`;
`source_frame_indices` records the actual video indices. Each frame is resized
with preserved aspect ratio and centered black padding onto a 512 by 288 grid,
then saved as JPEG. Thus the model pixel grid is the saved frame grid, not
necessarily the original video's grid. RGB, depth, confidence, masks and camera
intrinsics must agree on this grid.

References use sampled frame indices `[0, 10, 21, 31]`, saved as
`reference/frame_00.jpg` through `frame_03.jpg`. Reference ordinals 0..3 and sampled
frame indices 0/10/21/31 are different identifiers. In the production 48/128
registration paths the references are also included in fitting/scoring; they are
not held-out measurements.

## Stage 1 Perception Cache

```text
<processed>/<robot>/stage1/<task>/
  manifest.json                    # schema: stage1-inference-v1
  frames/00.jpg ... 31.jpg
  depth/depth_00.npy ... depth_31.npy
  sam3_masks/mask_00.npy ... mask_31.npy
  depth_conf.npz                  # depth_conf key
  camera.json
  reference/frame_00.jpg ... frame_03.jpg
```

| Field/array | Shape and semantics |
| --- | --- |
| Depth (per frame) | `(288, 512)`, normally float32, camera-axis Z depth in unverified VGGT units |
| `depth_conf` | `(32, 288, 512)`, finite model confidence values; not calibrated probabilities |
| Robot mask (per frame) | `(288, 512)`, boolean union of SAM3 instances |
| `camera.json: intrinsics` | `(32, 3, 3)`, pixel intrinsics `K` on the saved frame grid |
| `camera.json: extrinsics` | `(32, 3, 4)`, `[R|t]`, world-to-camera in VGGT units |
| Camera axes | OpenCV: +X right, +Y down, +Z forward |

`manifest.json` records `robot`, `source_video`, `video_sha256`, actual frame
indices, robot prompt and `models` metadata. Backend metadata identifies local or
explicit remote use; local runs record source revision, checkpoint SHA256,
device and preprocessing version. Failure writes `status=failed` and diagnostics;
`inference_complete` only confirms perception output, not metric scale or robot
registration. Empty masks are recorded in `empty_mask_frames` and can block
subsequent registration even when inference completed.

Current batch cache eligibility also checks the robot prompt. A historical cache
with `robotic arm` can contain valid arrays while being ineligible for reuse under
the current `robotic arm with base and camera` prompt. Preserve its real prompt
and source provenance; do not relabel historical outputs as fresh inference.

SAM3's internal instance interface returns `(K,H,W)` boolean masks and `(K,)`
scores on the requested image grid. No detection is `K=0`. Stage 1 discards
instance identity by unioning the masks; old union-only caches cannot reconstruct
individual scores. SIM uses instance selection before depth backprojection.

## Coordinates And Metric Anchoring

For VGGT-world point `p_G`, the camera point is `p_C = R p_G + t`. A valid depth
pixel `(u,v,d)` backprojects as:

```text
p_C = [(u - cx) * d / fx, (v - cy) * d / fy, d]
p_G = R.T @ (p_C - t)
```

Robot registration fits `p_G = s R_BG p_B + t_BG`, where `p_B` is a robot-base
point in meters and `s` is `scale_vggt_units_per_meter`. Its inverse is:

```text
p_B = R_BG.T @ (p_G - t_BG) / s
depth_m = depth_vggt / s
meters_per_vggt_unit = 1 / s
```

The official delivered camera grid keeps the original intrinsics. Camera
translations are scaled and the world is rebased to the robot root; rotations
change with that rebase. Do not scale the robot mesh, change focal lengths, or
stretch depth to hide a grid mismatch. Intermediate registration camera caches
can be in the VGGT-oriented metric world; only the delivered
`camera_metric.json` declares the robot-base delivery frame. Inspect
`metric_calibration` and its coordinate version before comparing poses.

## Stage 1 Registration And Delivery

Initialization outputs contain `initialization.json`, ranked SO(3) candidates,
surface/video/mask provenance and visual checks. Refinement outputs contain
`result.json`, selected seed, multiscale and rotation history, metric depth and
camera caches. Generated surface templates depend on the fixed joint pose and
the robot-control assets; changing either invalidates their provenance.

Delivery adds these files to the perception cache:

| Artifact | Meaning |
| --- | --- |
| `depth_metric/depth_XX.npy` | All 32 depths in meters, unchanged pixel grid |
| `camera_metric.json` | All 32 cameras in the meter-scale robot-base world |
| `registration.json` | Compact fitting scores, scale, source hashes and coordinate provenance |
| `stage1_delivery.json` | Cross-stage input manifest and delivery status |
| `initial.blend` | Fixed robot visuals and four reference cameras; reference images packed |
| `reference/overlay_*.png`, `quality_check.png` | Robot-mask and projected-robot review images |

The root robot has identity world transform in official delivered scenes.
`stage1_delivery.json` and registration remain `candidate_unverified`,
`accepted=false`, because production fitting and scoring share all 32 frames.
Do not infer acceptance from `initial.blend` existing.

The legacy single-video `real2sim-stage1` path uses `schema=stage1-v1`,
`reconstruction.npz`, `robot_masks.npz`, `view_XX.*` and `scene_spec.json` in a
unique run directory. It has separate metric-scale and held-out checks. Its
`manifest.json` is not interchangeable with the production
`stage1_delivery.json` layout consumed by phased Stage 2.

## Stage 2 Blender Delivery

One run contains `plan/plan.md`, phase/session logs, a frozen robot/camera
signature and `geometry/round_NNN/` candidate/edit/render/evaluation artifacts.
`run_manifest.json` indexes the run. The delivery reader for Stage 3 consumes
`geometry/stage2_manifest.json` (or its enclosing task directory), requiring
`accepted=true`, `evaluation.passed=true` and a matching `scene_sha256` for
`geometry/final.blend`.

The final manifest references the scene, plan/input provenance and four evaluated
RGB views. The Stage 3 reader currently derives the robot from the directory
layout `<root>/<robot>/stage2/<task>/geometry/stage2_manifest.json`; preserve this
layout even inside a new `runs/` root. Each evaluation records depth MAE, Lab Delta E76 and coverage per
reference view, with the exact thresholds. A mean-depth threshold alone does not
prove joint acceptance. The reference depth is a reconstructed pseudo-reference,
not independently measured ground truth. SSIM and object point-cloud metrics
are additional analysis and do not change the delivery gate.

`geometry/input_provenance.json` binds a resumable run to the Stage 1 Blend and
manifest, four reference RGB/depth files, plan, evaluation settings and
renderer/evaluator source hashes. It also records the prepared scene, inspection
and baseline render hashes. Each completed `round_report.json` records the
candidate/render hashes in `artifact_sha256`. Resume verifies those hashes and
recomputes metrics; stale report flags cannot establish acceptance. Missing or
changed provenance requires a new run directory. Existing delivery readers
still accept historical manifests; this requirement applies to continuation.

## Stage 3 And SIM Scene Bundle

Stage 3 preserves the accepted Blend source and exports `scene.xml`,
`assets/meshes/task/`, robot assets, extracted image resources,
`object_manifest.json`, `camera_manifest.json` and `export_manifest.json`.
`semantics.json` ties visual/static/dynamic/operable hypotheses to the source
Blend hash. Four cameras become `stage3_camera_0` through `stage3_camera_3`.
MJCF lengths are meters; action/state quaternions use `[w,x,y,z]`.

An agent physics run preserves candidate XMLs, probe reports, task context and
session evidence separately. A final scene can be a repaired candidate rather
than the first export; read `final_manifest.json` or the batch's `final_scene`
instead of guessing from a `mujoco_vN/` directory. A physical status of
`provisional` is a simulation conclusion, not measured calibration.

`scripts/prepare_sim_scene.py` removes the embedded robot while preserving its
empty mount, task geometry and reference cameras. It creates
`scene_for_sim.xml`, `scene_derivation.json` with hashes and an `assets` symlink
to the export bundle. robot_control inserts the controlled robot at runtime.
The derived XML alone is not portable; keep/copy the source asset bundle and
repair the symlink when relocating it.

Derivation removes actuator, equality and contact entries that reference named
elements of the embedded robot. Unnamed robot elements cannot match omitted
attributes on task entries; task motors, welds and contact rules are preserved.

## SIM Observation And Policy Evidence

`Stage3RGBDObservation.observe()` returns `objects`, `images.wrist`,
`images.third_person` and camera labels. Each detected object has estimated
`xyz_m`, `xy_m` and `bbox_pixels=[x0,y0,x1,y1]` (exclusive right/bottom). A name
must belong to the scene's public object set; absent/insufficient segmentation
omits the name. The position is a robust statistic of visible depth points in
the estimated world frame, not the full object's center of mass or simulator
truth. Image payloads include absolute paths; the practice CLI removes base64
from printed JSON.

Three-pass policy runs keep per-pass `policy.py`, `experience.md`, reports,
actions, rendered evidence and audit logs. `experience_memory.md` carries
development notes forward. Privileged object poses are allowed only in the first
pass. A policy saying it completed, or a probe passing, is distinct from external
task verification and held-out policy evaluation. Save video review evidence and
the locked policy bundle before making success claims.

Weights, credentials, cached model files, raw task recordings and local run
artifacts are ignored and must not be added merely to make recorded absolute
paths resolve for another user. See [WORKFLOW.md](WORKFLOW.md) for preparation
commands and [THIRD_PARTY.md](THIRD_PARTY.md) for redistribution requirements.
