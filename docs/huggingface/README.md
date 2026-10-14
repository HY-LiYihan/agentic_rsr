---
language:
  - en
tags:
  - robotics
  - real-to-sim
  - sim-to-real
  - scene-reconstruction
  - robot-manipulation
  - mujoco
  - blender
  - experiment-artifacts
  - arxiv:2610.10479
---

# Agentic RSR Experiment Results

Companion experiment artifacts for **Agentic RSR: Real-to-Sim-to-Real through
Scene Reconstruction and Execution-Grounded Robot Policies**.

- [Paper (arXiv)](https://arxiv.org/abs/2610.10479)
- [Source code](https://github.com/HY-LiYihan/agentic_rsr)
- [Project page](https://hy-liyihan.github.io/agentic_rsr/)
- [Workflow and installation](https://github.com/HY-LiYihan/agentic_rsr/blob/master/docs/WORKFLOW.md)
- [Artifact formats](https://github.com/HY-LiYihan/agentic_rsr/blob/master/docs/DATA_CONTRACT.md)

## Purpose And Scope

This dataset repository is the distribution location for experiment results,
assets and derived artifacts associated with Agentic RSR. The project connects
workspace-video reconstruction, Blender scene modeling, MuJoCo physics checks,
and execution-guided robot policy development for Franka FR3 and AgileX Piper.

The `20261007` suffix identifies this dataset repository. It does not imply that
every file was generated on that date or with the latest public code. Use the
source revisions, timestamps and manifests accompanying each artifact to
identify its provenance. The repository is an artifact collection rather than
a standardized training dataset with predefined train/test splits.

## Repository Layout

The existing archive was inspected at commit
`4b1970c79b82407d2eceb54896b94720137e6509` on 2026-10-14. It contains 49,884
files before the addition of this card, organized under:

| Path | Contents |
| --- | --- |
| `summary/` | Aggregate report, scene details and report-generation script |
| `models/<model>/` | Historical model-specific experiment bundles; inspected Claude Opus and GPT-6.1 bundles include a `dsh/` directory |
| `migration_checks/` | Checks from the archive migration |
| `migration_manifest.json` | Migration status and job records |

Start with [summary/REPORT.md](summary/REPORT.md) and the matching experiment's
report/configuration. Model directory names identify historical experiment
configurations, not downloadable perception-model checkpoints. The aggregate
archive spans multiple configurations; it should not be treated as a single
paper result or as outputs from the current public checkout.

## Artifact Formats

The following are formats used by the project, not a complete inventory of the
files published in this dataset. Consult **Files and versions** for availability.

| Artifact type | Associated files to preserve |
| --- | --- |
| Perception outputs | Sampled RGB frames, depth/confidence arrays, masks, camera parameters and inference manifests |
| Registered scenes | Metric depth/cameras, registration and delivery manifests, robot references and Blender scenes |
| Scene reconstruction | Blender files, rendered views, evaluations and source/input provenance |
| Physics and simulation | MuJoCo XML, referenced meshes/textures, export/derivation manifests and physics reports |
| Policy execution | Policies, run manifests, videos and reviews |
| Derived robot surfaces | Point clouds, metadata and upstream source/license records |

Keep bundles together and preserve relative paths. Manifest paths may refer to
the original checkout, and exported scenes can reference external files or
symlinks. Relocation may require resolving those paths. A standalone `.blend`,
XML or manifest is not necessarily a self-contained runnable scene.

## Download And Use

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
This repository is currently **private**. Access requires an authorized Hugging
Face account and token; run `hf auth login` locally or configure `HF_TOKEN`.
The following command downloads only the summary and migration manifest from
the inspected archive revision:

```bash
hf download HY-LiYihan/Agentic-RSR-experiment-results-20261007 \
  summary/REPORT.md summary/scene_details.json migration_manifest.json \
  --repo-type dataset --revision 4b1970c79b82407d2eceb54896b94720137e6509 \
  --local-dir /path/to/agentic-rsr-artifacts
```

Select experiment files through **Files and versions** or `--include` filters,
pinning an immutable commit. Follow the source repository's workflow for the matching artifact layout and
code revision. Verify accompanying SHA256 records where provided before use.
Model checkpoints are obtained separately from their official upstream sources;
see the [local inference guide](https://github.com/HY-LiYihan/agentic_rsr/blob/master/docs/LOCAL_INFERENCE.md).

## Interpretation And Limitations

Historical experiment outputs do not establish that the current public local
CUDA backend has been reproduced. Fresh local CUDA checkpoint inference remains
unverified in the public code's validation record.

Reconstruction depth references are estimated depth, not independently measured
physical ground truth. Registration or scene acceptance must be assessed using
the relevant manifests and reports; file existence alone does not establish
acceptance. Simulated physics results do not establish measured calibration or
real-robot readiness. Hardware execution requires separate validation.

## License And Provenance

Original project software and documentation are licensed under Apache-2.0.
This does **not** assign Apache-2.0 to all files in this dataset. Robot meshes,
derived point clouds, model outputs, task media and other third-party materials
retain their respective rights and applicable terms. Check the notices and
provenance supplied with each artifact before reuse or redistribution.

See the source repository's [third-party records](https://github.com/HY-LiYihan/agentic_rsr/blob/master/docs/THIRD_PARTY.md)
and [asset distribution notes](https://github.com/HY-LiYihan/agentic_rsr/blob/master/docs/ASSET_DISTRIBUTION.md).
Hugging Face hosting does not resolve missing upstream permissions or change
the underlying licenses. A dataset-wide license is not asserted by this card.

## Citation

Please cite the paper:

```bibtex
@misc{li2026agenticrsr,
  title={Agentic RSR: Real-to-Sim-to-Real through Scene Reconstruction and Execution-Grounded Robot Policies},
  author={Li, Yihan and Feng, Yating and Sun, Shengjiu and Chen, Jianing and Ren, Hao and Yang, Bowen and Xu, Weisheng and Wu, Qiwei and Cheng, Hui and Xu, Renjing},
  year={2026},
  eprint={2610.10479},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2610.10479}
}
```

For artifact questions or corrections, open an issue in the
[source repository](https://github.com/HY-LiYihan/agentic_rsr/issues).
