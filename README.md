<div align="center">

# Agentic RSR

### Real-to-Sim-to-Real through Scene Reconstruction and Execution-Grounded Robot Policies

**English** · [简体中文](README.zh-CN.md)

Yihan Li<sup>1,2,*,†</sup> · Yating Feng<sup>1,*</sup> · Shengjiu Sun² · Jianing Chen² · Hao Ren²<br>
Bowen Yang¹ · Weisheng Xu¹ · Qiwei Wu¹ · Hui Cheng² · Renjing Xu¹†

¹ HKUST (Guangzhou) · ² Sun Yat-sen University<br>
\* Equal contribution (co-first authors) · † Corresponding authors

[![Project Page](https://img.shields.io/badge/Project-Page-126A50?style=for-the-badge&logo=github&logoColor=white)](https://agentic-rsr.github.io/)
[![Paper (arXiv)](https://img.shields.io/badge/arXiv-2610.10479-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2610.10479)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/HY-LiYihan/Agentic-RSR-experiment-results-20261007)
[![Workflow](https://img.shields.io/badge/Docs-Workflow-2563EB?style=for-the-badge&logo=readthedocs&logoColor=white)](docs/WORKFLOW.md)
[![Installation & Weights](https://img.shields.io/badge/Setup-Installation_%26_Weights-555555?style=for-the-badge&logo=python&logoColor=white)](docs/LOCAL_INFERENCE.md)

<img src="docs/assets/overview.png" width="1000" alt="Agentic RSR framework: robot alignment, scene modeling, physics verification, simulation policies, and real-robot feedback">

</div>

Agentic RSR starts from a workspace video, a task description, and a known robot model. It recovers metric scene scale, iteratively models the scene in Blender, verifies interactions in MuJoCo, and develops robot policies through execution feedback. After the three Real2Sim stages, policies progress through privileged, visual, and randomized simulation; the paper also explores experience transfer and adaptation on real robots.

## Visualizations

<img src="docs/assets/real-sim.png" width="1000" alt="Real and reconstructed simulation scenes for Franka FR3 and AgileX Piper">

<img src="docs/assets/robot-execution.png" width="1000" alt="Real-robot bowl stacking from the paper, showing initial state, failed attempts, and successful placements">

## Implementation Status

| Stage | Public implementation and validation scope |
| --- | --- |
| Stage 1: Robot alignment | Local VGGT-Ω / SAM3 by default, 32 frames at 512×288, URDF metric anchoring, and a `.blend` with four reference cameras. CPU cache replay, registration, and delivery are verified; real local CUDA checkpoint inference awaits validation. |
| Stage 2: Visual modeling | Codex planning, modeling, and feedback refinement with frozen robots and cameras, evaluated through depth, color, and coverage. The paper's latest DSH reconstruction configuration is not covered by this checkout's reproduction guarantee. |
| Stage 3: Physics refinement | Blender-to-MuJoCo export, semantic and physical checks, and parameter refinement. Task-level results require inspection of their artifacts and manifests. |
| SIM / Real | Privileged, visual, and randomized policy entry points. Real-robot adaptation and independent safety validation remain incomplete. |

Production Stage 1 registration scores all 32 frames and preserves `candidate_unverified` and `accepted=false`. The four reference views come from one video. Full experiment inputs and local CUDA inference validation are still pending; this is not a complete paper reproduction release.

## Installation And Minimal Example

Target environment: **Linux x86_64, NVIDIA CUDA, Python 3.11**. Install [uv](https://docs.astral.sh/uv/getting-started/installation/); Blender is required for scene generation and evaluation. Model sources are pinned HTTPS submodules. Prepare weights, videos, and credentials separately.

```bash
git clone --recurse-submodules https://github.com/HY-LiYihan/agentic_rsr.git
cd agentic_rsr
uv sync --locked --python 3.11 --extra local --extra sim --extra agent --extra dev
cp -n .env.example .env
```

Follow the [installation guide](docs/LOCAL_INFERENCE.md) to request the official weights and place them at `model/vggt_omega/vggt_omega_1b_512.pt` and `model/sam3/sam3.pt`. Keep `VGGT_OMEGA_BACKEND=local` and `SAM3_BACKEND=local`. Missing weights or devices cause an explicit failure; inference does not download weights or switch to remote services automatically.

```bash
uv run --no-sync real2sim-models check
uv run --no-sync real2sim-local-example \
  --video /path/to/task.mp4 --robot piper --output runs/my-first-example
```

The video must contain at least 32 frames, and the output directory must not exist. The example generates RGB, depth, cameras, masks, previews, and a model provenance manifest. Registration and scene acceptance are separate steps. See the [full workflow](docs/WORKFLOW.md) for Stage 1 → 2 → 3 → SIM commands and delivery checks.

Historical experiment artifacts are archived separately from the code at [Hugging Face](https://huggingface.co/datasets/HY-LiYihan/Agentic-RSR-experiment-results-20261007), currently a private dataset requiring authorized access. See its dataset card and [asset distribution](docs/ASSET_DISTRIBUTION.md) for the layout and pinned download command. Migration of the assets currently tracked in Git remains pending. Model checkpoints remain available from their official upstream sources.

Offline tests without model weights:

```bash
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest tests/unit tests/integration -m 'not models' -ra
```

The GitHub test CI has been removed. Offline tests and real model acceptance run separately; skipped tests do not establish success. See the [Stage 1 CPU validation record](docs/STAGE1_CPU_VALIDATION.md).

## Documentation And Contributions

- [Workflow](docs/WORKFLOW.md) · [Data contracts](docs/DATA_CONTRACT.md) · [Documentation index](docs/README.md)
- [Local inference](docs/LOCAL_INFERENCE.md) · [Stage 1](src/real2sim/stage1/README.md) · [Stage 2](src/real2sim/stage2/README.md) · [Stage 3](src/real2sim/stage3/README.md) · [SIM](src/sim/README.md)
- [Contributing](CONTRIBUTING.md) · [Tests](tests/README.md) · [Release preparation](docs/OPEN_SOURCE_PREPARATION.md)
- [Asset distribution](docs/ASSET_DISTRIBUTION.md) · [Third-party terms](docs/THIRD_PARTY.md) · [Paper media and institutional marks](docs/ASSET_SOURCES.md)

## License

Project code is licensed under [Apache-2.0](LICENSE). Third-party source, model weights, robot assets, derived artifacts, paper media, and institutional marks retain their own terms. VGGT-Ω is restricted to noncommercial research; Ultralytics / CLIP use AGPL; SAM3 weights use the Meta SAM License. Publishing assets on Hugging Face does not change their underlying licenses.

## Citation

Please cite the [arXiv paper](https://arxiv.org/abs/2610.10479). Machine-readable metadata is available in [CITATION.cff](CITATION.cff).

```bibtex
@misc{li2026agenticrsr,
  title={Agentic RSR: Real-to-Sim-to-Real through Scene Reconstruction and Execution-Grounded Robot Policies},
  author={Li, Yihan and Feng, Yating and Sun, Shengjiu and Chen, Jianing and Ren, Hao and Yang, Bowen and Xu, Weisheng and Wu, Qiwei and Cheng, Hui and Xu, Renjing},
  year={2026},
  eprint={2610.10479},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  doi={10.48550/arXiv.2610.10479},
  url={https://arxiv.org/abs/2610.10479}
}
```
