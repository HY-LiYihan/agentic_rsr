# Contributing

This repository is being prepared for paper reproduction and reusable tools.
The supported installation is a source checkout with pinned submodules. Read
[local inference](docs/LOCAL_INFERENCE.md), [third-party terms](docs/THIRD_PARTY.md)
and the [release checklist](docs/OPEN_SOURCE_PREPARATION.md) before contributing.
Original project code and documentation use [Apache-2.0](LICENSE).
Third-party components, assets, derived artifacts and media retain their own
terms. Assets and derived artifacts will be published separately through
Hugging Face; see [asset distribution](docs/ASSET_DISTRIBUTION.md).

## Development Environment

Use Python 3.11. Linux with NVIDIA CUDA is the complete local-inference target;
offline contracts can run without weights or a GPU.

```bash
git clone --recurse-submodules https://github.com/HY-LiYihan/agentic_rsr.git
cd agentic_rsr
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest tests/unit tests/integration -m 'not models' -ra
```

The SIM suite requires the pinned robot-control package and nested robot assets.
Tests requiring unprovided local scenes or Blender may skip; include skip
reasons in validation reports. MuJoCo rendering on macOS may require
`mjpython -m pytest`; on headless Linux use an available EGL or OSMesa context.
Real model tests require explicitly prepared inputs and weights as described in
the installation guide. They never establish success when skipped. The GitHub
test workflow has been removed; run the relevant checks locally. GitHub Pages
still deploys the project website.

## Changes And Reports

- Keep shared geometry, model backends and stage implementation in `src/`.
  Put task-specific orchestration in `scripts/` and document its input contract.
- Preserve Stage 1 output contracts and readable historical results. Record
  source revisions, checkpoint hashes, configuration and preprocessing changes
  whenever a result depends on them.
- Write new runs to a new output directory. Keep user videos, model weights,
  generated scenes, caches, `.env` files and credentials outside commits.
- Include the command, platform, dependency versions, passed tests, skipped tests
  and inference limits in a pull request. A CPU cache replay is not local model
  inference; a simulator pass is not a real-robot safety validation.
- For redistributed media or assets, include their source, revision, license
  and required notices. Do not assume a public URL grants redistribution rights.

Before submitting:

```bash
git diff --check
git status --short
```

Review the staged diff for credentials and unintentional binary outputs. Report
bugs with a minimal input or synthetic fixture, the command, full error and
environment details. Redact credentials and private endpoint information.
