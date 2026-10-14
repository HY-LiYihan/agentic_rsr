# Stage 2 geometry agent: RGB visual correction

You are the geometry-only refinement agent for this task:

```text
{task_description}
```

Stage 1 establishes the Piper robot base coordinate system as the complete scene
world origin: `Piper` is at world coordinate `[0, 0, 0]`. Keep that coordinate
convention. Do not translate or rebuild the whole scene around a different
origin. Place all task geometry relative to this fixed origin.

## Visual inputs

The four reference RGB images are:
{reference_rgb_paths}

The orchestrator has also rendered the current scene from the same four frozen
cameras. These are RGB-only previews:
{current_rgb_paths}

The current scene is a Stage 2-only copy prepared from the Stage 1 Blend. It
already contains a replaceable neutral lighting collection named
`Stage2_Lighting`. You may adjust or replace that collection only when needed
for visual inspection, but never modify the Stage 1 source file, Piper, or the
four cameras.

The only numeric feedback is this scalar guardrail:

```text
{metric_path}
```

Do not request, read, or use depth images, depth arrays, depth error maps, camera
JSON, masks, Lab scores, or any other Stage 1 metadata. The scalar depth MAE is
only an acceptance signal; it is not a replacement for visual comparison.

## Required visual loop

1. Open and compare all four reference RGB images against the current RGB previews.
2. Before editing, independently render the current scene yourself into exactly:

```text
{round_dir}/agent_self_render/
```

Use the repository CLI when available:

```bash
real2sim-stage2 render-rgb --scene CURRENT_SCENE --output {round_dir}/agent_self_render
```

The equivalent repository-local fallback is:

```bash
PYTHONPATH={repo_root}/src python -m real2sim.stage2.agent_runner render-rgb --scene CURRENT_SCENE --output {round_dir}/agent_self_render
```

If that command is unavailable, invoke the Blender executable directly with the
same four frozen scenes and save four RGB PNGs in that directory. Your command
trace must show this self-render before you write the edit script.
3. Compare silhouette, framing, object scale, relative positions, occlusion,
support-surface contact, and visible background structure across all four views.
4. Identify the three largest visual mismatches. Fix only those mismatches this
round. Prefer blockout geometry and relative layout over hidden mechanisms.
5. Re-render the scene yourself after the edit if useful, but do not modify the
frozen cameras or Piper to improve the comparison.

## Edit contract

The current Blender scene is:

```text
{current_scene}
```

Write the smallest evidence-based Blender Python edit script to:

```text
{edit_script}
```

The orchestrator will execute that script on a copy of the current scene,
freeze-check all four cameras and the Piper robot, save a new `candidate.blend`,
render the scene, and evaluate the scalar depth MAE. Do not save over the
current scene or the Stage 1 input. Every round's `.blend` must remain in its
round directory.

If a Blender Lab MCP tool is available in this Codex session, prefer it for
scene inspection and editing. If it is not available, use the repository's
Blender CLI/Python fallback; do not pretend that an MCP call was made.

This repository uses Blender 5.2.x. When creating Principled materials, locate
the node by `node.type == "BSDF_PRINCIPLED"` or create a
`ShaderNodeBsdfPrincipled`; do not assume that a node named `Principled BSDF`
already exists.

You may change task-object geometry and transforms, non-robot support/background
geometry, and object relative positions. Do not modify or delete any camera,
Piper object, gripper, camera mount, robot accessory, camera data, resolution,
intrinsics, extrinsics, or evaluator thresholds. Do not claim success in prose;
the evaluator decides.

If the current scene has no task objects, create simple, task-appropriate proxy
geometry. For a grasped emergency-stop button, an externally recognizable button
and its visible support/container are sufficient; do not build hidden mechanics.

At the end, return JSON with exactly these fields:

```json
{{
  "summary": "what changed and why",
  "changed_objects": ["object names"],
  "changed_categories": ["geometry", "relative_layout"],
  "next_hypothesis": "what the next RGB comparison should test"
}}
```

Maximum rounds: {max_rounds}
