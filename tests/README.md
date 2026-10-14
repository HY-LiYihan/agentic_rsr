# Tests

Run tests from the repository root. The pytest configuration adds `src` to the
import path and the checkout root so script tests work with either pytest entry
point.

## Dependencies

```bash
git submodule update --init --recursive
uv sync --locked --python 3.11 --extra dev --extra sim
uv run --no-sync python -m pytest tests/unit tests/integration -m 'not models' -ra
```

The current unit suite imports SIM batch scripts during collection, so `dev`
alone is insufficient: those scripts import MuJoCo and Pillow. Collider tests
also import MuJoCo directly. Tests which load
robot scenes also need an installed `robot_control` package, initialized nested
assets and, in some cases, generated scenes under ignored `runs/` directories.
Optional scene tests may skip when these inputs are absent. Such skips do not
establish that a full simulation passes.

GitHub Actions CI was removed at the author's request; there is currently no
hosted test workflow. Run the offline command above before submitting changes.
Optional scene and Blender tests report skips where inputs are absent; no
download of model weights is allowed. See
[verification records](../docs/OPEN_SOURCE_PREPARATION.md).
The [fresh public-source check](../docs/FRESH_SOURCE_VALIDATION.md) separately
records installation without author scenes or environment files and identifies
the remaining derived-scene skips.

## Real Model Tests

Follow [local inference installation](../docs/LOCAL_INFERENCE.md) to initialize
the pinned model sources and install `local`, `sim` and `dev` on Linux CUDA.
Tests marked `models` additionally need explicit inputs:

```bash
export RSR_MODEL_TEST_VIDEO=/path/to/task.mp4
export RSR_MODEL_TEST_SCENE=/path/to/scene_for_sim.xml
export RSR_MODEL_TEST_ROBOT=piper
uv run --no-sync python -m pytest tests/integration -m models
```

The video must contain at least 32 frames. The scene must be a complete Stage 3
export. Missing inputs or CUDA produce explicit skips, which are not inference
validation. Preserve the example report and test output as the run record.

## Test Layout

| Directory | Scope |
| --- | --- |
| `unit/` | Geometry, evaluation, manifests, policy interfaces, randomization and agent orchestration. Some tests optionally load scenes. |
| `integration/` | Inference with test doubles, Blender scene generation and MuJoCo execution. Inspect dependencies before running. |
| `contract/` | Reserved for backend contract tests; currently contains documentation only. |
| `fixtures/` | Small committed scene and physics fixtures. |

Blender integration tests use `BLENDER_BIN`, then `blender` on PATH, with the
macOS Blender application as a fallback. An invalid explicit `BLENDER_BIN`
fails the tests; an absent Blender installation produces an explained skip.
The tabletop integration tests use the committed primitive scene in
`fixtures/stage3/code_as_policy_tabletop.xml`, without an external home directory.
Rendering MuJoCo on macOS can require `uv run --no-sync mjpython -m pytest`;
headless Linux rendering uses `MUJOCO_GL=egl`. Neither the unit suite nor an
integration pass validates real robot execution or all published experiment
results. The preparation checklist tracks remaining portability work.
