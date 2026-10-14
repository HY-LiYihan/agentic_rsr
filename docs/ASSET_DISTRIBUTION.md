# Asset Distribution

The author selected Hugging Face for publishing assets and derived artifacts
on 2026-10-14. GitHub hosts source code, documentation, provenance records and
the project website. Model checkpoints are obtained from the official model
repositories listed in [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md).

## Publication Status

The author specified
[`HY-LiYihan/Agentic-RSR-experiment-results-20261007`](https://huggingface.co/datasets/HY-LiYihan/Agentic-RSR-experiment-results-20261007)
on 2026-10-14. Authenticated inspection confirmed that it is private and contains
49,884 files at `4b1970c79b82407d2eceb54896b94720137e6509`, with `models/`,
`summary/`, `migration_checks/` and `migration_manifest.json`. The repository
archives historical experiments, not a verified replacement for every asset
currently tracked in Git. A basic English [dataset card](huggingface/README.md)
documents the layout, authenticated download, provenance, citation and terms.
The card was uploaded and its exact content and metadata verified at commit
[`dfba1e3c91d3d7e2c8988ff9711e78a1f3bc3b98`](https://huggingface.co/datasets/HY-LiYihan/Agentic-RSR-experiment-results-20261007/commit/dfba1e3c91d3d7e2c8988ff9711e78a1f3bc3b98).
The dataset remains private; adding the card does not make the archive public.
No new asset upload or migration is claimed. Existing tracked robot surface caches and the historical
SIM demonstration remain in the current Git checkout until their replacement
download is published and verified. Local files and experiment results must be
preserved during migration; published Git history is not rewritten.

## Layout

Keep the repository-relative layout in the dataset so users can download or
mount files at the paths already consumed by the workflow:

| Artifact | Destination in a source checkout |
| --- | --- |
| Robot point clouds and metadata | `asset/robot_surfaces/{piper,franka}/` |
| Task videos | `data/raw/{piper,frank}/` |
| Stage 1 perception and registered scenes | `data/processed/` |
| Blender scenes, MuJoCo exports, meshes and textures | The corresponding stage output directory, preserving relative paths |
| Selected policy results and visualizations | `runs/`, with each run's manifest and report |

Publish selected reproducibility artifacts rather than entire local output
directories. Credentials, personal environment files, agent authentication,
unreviewed conversation logs, caches and upstream model weights are excluded.

## Versioning And Terms

Each published bundle needs a dataset card, source revisions, per-file SHA256
hashes, original license texts and required notices, transformation provenance,
and the compatible code commit. Downloads should pin a Hugging Face commit and
verify hashes before the workflow uses the files. No inference entry point
should implicitly download assets.

Apache-2.0 covers original project software and documentation, not third-party
robot assets or derived files. Hugging Face hosting does not change upstream
terms. The unresolved Piper stand and FR3 mount/conversion provenance recorded
in [THIRD_PARTY.md](THIRD_PARTY.md) remains part of the asset publication work.

Once the cleared file list and replacement asset mapping are available, publish and verify
the bundle, add concrete versioned download instructions to both READMEs, and
remove migrated binaries from Git tracking while preserving local copies.
