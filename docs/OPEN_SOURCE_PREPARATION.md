# Open Source Preparation

Audit date: 2026-10-14. This checklist separates completed cleanup from remaining
release gates. It is a preparation record, not a release announcement.

## Baseline

- The published paper page and workflow baseline is commit `6ae6caf`; the
  later entries below document verification and cleanup after that baseline.
- The original 2026-10-13 cleanup included 21 changed tracked files and 15 new
  source/test files. Those changes are now committed; they are historical audit
  context rather than a description of the present working tree.
- GitHub reports both `HY-LiYihan/agentic_rsr` and
  `HY-LiYihan/robot_control` as public. The author selected Apache-2.0 for
  original project code and documentation on 2026-10-14; the control repository
  declares MIT. Third-party assets and media retain their own terms.
- Raw videos, processed scenes, checkpoints and most run evidence are local.
  One emergency-stop policy, video and review are already tracked under `runs/`.
  Ignoring a directory does not remove previously tracked artifacts or history.

## Initial Cleanup

- [x] Inventory tracked code, local additions, dependencies and public repository status.
- [x] Replace placeholder script/test indexes with current entry points.
- [x] Align the root README's Stage 2 mean depth threshold with the evaluator.
- [x] Ignore temporary renders and build output without deleting local files.
- [x] Use HTTPS for the main robot-control submodule URL.
- [x] Remove personal absolute paths and the private-network host default from identified experiment helpers.
- [x] Record the local unit-test baseline and correct the obsolete default SDK-home assertion.

## Validation Record

The records below describe the initial audit before the local-model implementation.

- The repository's base Python 3.13 environment cannot collect the full suite
  without the SIM dependencies. Excluding the two Stage 3 script test modules
  produced 166 passes, 32 skips and three failures: two required MuJoCo, and one
  test still expected the old default private Codex home.
- With the existing robot-control Python 3.11 runtime, `mjpython` and the pinned
  submodule source on `PYTHONPATH`, the complete unit suite passes: **219 passed**,
  with one upstream `hppfcl` deprecation warning. The SDK test was updated to
  assert the current default configuration; explicit private-home coverage remains.
- This is a local baseline using available assets and generated scenes. A fresh
  clone, a newly provisioned environment, integration tests and full model runs
  have not been validated.
- `git diff --check` passes; temporary and build output are ignored.

## Release Blockers

| Finding | Evidence | Required outcome |
| --- | --- | --- |
| Asset redistribution evidence is incomplete. | `asset/robot_surfaces/*/metadata.json` records meshes from multiple robot repositories; the current `piper_isaac_sim` checkout has no discovered license file. | Map source URLs, revisions and license terms to each distributed asset and derivative; include required notices. |
| Complete installation validation remains pending. | The locked environment uses fixed editable submodules and a NumPy 1.26 compatible SIM stack. | Verify the locked installation and full model inference on Linux CUDA from a fresh checkout. |
| Standalone wheel execution assumes repository files. | Stage 1/3 derive top-level `asset/` and `external/` paths from `__file__`. | Initially state that a source checkout is required; later introduce explicit asset roots if standalone packaging is needed. |
| Real local model inference has not been validated. | Local factories, aligned VGGT input, SAM3 instances and weight prerequisites are implemented; no CUDA checkpoints were run in this cleanup environment. | Save real video and SIM reports, dependency versions, runtime and peak CUDA memory. |
| Some experiments require external local inputs. | The exact-suite evaluator requires separately supplied `paper_suite`, `task_scoring` and archived trial scenes. The tabletop preview now uses the committed fixture and shared joint pose. | Publish or explicitly version the reference experiment inputs; fixture verification does not reproduce the task suite. |
| Some historical tests depend on local scenes. | The offline test environment uses `sim,dev`; tests without required ignored scenes skip with reasons. Real model tests are marked `models`. | Keep offline coverage portable and provide publishable model-validation inputs. |

## Phase 1: Local Perception

The intended audience combines paper reproduction and reusable tools. The first
complete inference target is Linux x86_64, NVIDIA CUDA and Python 3.11.

- [x] Share local-by-default backend configuration across Stage 1, object metrics,
  prompt trials, SIM batches and practice CLI; preserve explicit remote modes.
- [x] Use aligned 32-frame 512×288 VGGT input and original-grid SAM3 instance
  masks, with model release between stages and explicit prerequisite failures.
- [x] Add model checks, a new-directory video example, model provenance in
  manifests, offline contracts and opt-in GPU tests.
- [x] Document pinned upstream sources, official weight access and third-party
  licenses; the author has now selected Apache-2.0 for original project code.
- [ ] Complete real CUDA video and local SAM3 SIM validation. No GPU inference
  success is claimed by the implementation or offline tests.

The existing robot-control Python 3.11 runtime passed **277 unit/integration
tests** with two real-model tests deselected. A fresh Python 3.11 dependency
smoke check also initialized a real Piper MuJoCo backend and IK with NumPy 1.26
and Pinocchio 2.7. These checks are distinct from the full Linux CUDA acceptance
run. See [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md) for installation and validation.

Final phase-1 offline verification: a fresh Python **3.11.15** environment in
`/tmp/agentic-rsr-local-deps-check` successfully installed the complete locked
`local,sim,dev` dependency set from this checkout. The unit/integration suite
passed **282 tests**, with **2 real-model tests deselected**, in **54.44 seconds**
using macOS `mjpython`. Actual imports of VGGT, Ultralytics SAM3 and timm, plus
initialization of CLIP's packaged tokenizer, also passed without weights.
The environment used NumPy 1.26.4, OpenCV 4.11, PyTorch 2.6, torchvision 0.21
and Pinocchio 2.7. This verifies a new environment from the existing checkout;
anonymous fresh-clone installation and Linux CUDA checkpoint inference remain
unverified. The repository currently has no GitHub test workflow, so these
offline checks are contributor-run records rather than hosted CI results.

The local VGGT submodule is now a clean full checkout at the exact pinned commit.
It was previously sparse because GitHub transport timeouts prevented expansion
of the training, evaluation and example directories. The official complete
archive was verified against the existing authoritative Git tree before its
missing blobs were added to the local object database and sparse checkout was
disabled. All 113 checkout files and executable modes match the pinned tree;
the commit and official HTTPS remote are unchanged. This local source hydration
does not establish the anonymous recursive clone path. See the scoped record in
[FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md#local-vggt-full-checkout).

## Refactoring Order

1. Establish provenance and regression evidence. Preserve current manifests,
   policies and source changes; record passing tests, failures and skips before
   changing interfaces.
2. Define the public workflow. Decide whether the primary audience is paper
   reproduction, a reusable tool library, or both. Select supported platforms,
   input data and the first reproducible example.
3. Separate reusable implementation from experiment orchestration. Keep core
   geometry, perception, evaluation, export and execution in `src/`; group
   dataset-specific ablations and historical experiments separately after
   checking imports, command references and data paths.
4. Unify configuration and entry points. Resolve Blender, robot assets, service
   endpoints, model options, data roots and output roots consistently. Replace
   personal paths in public examples with the supported checkout workflow.
5. Consolidate documentation. Publish installation, a minimal run, stage
   contracts, evaluation definitions, dependency versions and troubleshooting.
   Keep historical experiment results clearly dated and tied to their evidence.
6. Verify a fresh clone. Test anonymous recursive clone, editable installation,
   offline tests and the example without the author's home directories. Record
   services or model calls needed for full reconstruction separately.
7. Prepare the release. Complete license/attribution review, inspect tracked
   content and history for credentials, add contribution instructions and record
   manual validation, then publish a reviewed version and its reproducibility
   artifacts. CI remains removed at the author's request.

## 2026-10-14 Release Audit

- [x] Add machine-readable citation metadata in
  [`CITATION.cff`](../CITATION.cff), linked from the root README and the
  project page. The software declares Apache-2.0 and prefers citation of the
  paper at arXiv:2610.10479, verified through the official arXiv API.
- [x] Add [CONTRIBUTING.md](../CONTRIBUTING.md) with checkout setup, offline
  testing, result provenance and contribution expectations.
- [x] Remove stale documentation claims of active CI. Commit `afc030e` removed
  the workflow; no hosted test workflow is enabled in this repository.
- [x] Reproduce the installed `pytest` entry point's `scripts` import failure
  and include the checkout root in pytest's import path. Both previously failing
  Stage 3 modules now collect all 18 tests through the installed entry point.
- [x] Run the full offline unit/integration suite through the installed `pytest`
  entry point on macOS Python 3.11.15: **288 passed, 2 real-model tests
  deselected**, in **55.60 seconds**. CUDA was disabled, model downloads were
  disabled and no remote perception endpoint was used. This run used the
  author's available local scenes/assets; it is not a clean-clone claim.
- [x] Review tracked credential/weight filenames and history paths without
  reading private environment files. Only `.env.example` and model README files
  occur in the inspected filename patterns; no actual checkpoints or private
  `.env` paths are tracked. This is a filename audit, not a complete content or
  historical credential scan.
- [x] Scan current tracked text for recognizable OpenAI, GitHub and Hugging Face
  token prefixes and private-key headers using filename-only output; no matches
  were reported. Private `.env` files were not read. This narrow signature check
  does not clear all credential formats or historical content.
- [x] Record the Stage 1 CPU verification separately in
  [STAGE1_CPU_VALIDATION.md](STAGE1_CPU_VALIDATION.md): 132 tests passed,
  two real-model tests skipped, and Piper/Franka cache replay, CPU registration
  and Blender delivery passed. CUDA inference and independent registration
  accuracy remain unverified.
- [x] Add the author-selected Apache-2.0 LICENSE, scope and copyright NOTICE,
  Python license metadata and CFF license field.
- [x] Make README.md English by default and add README.zh-CN.md. Remove the
  requested numeric results and explanation; retain paper visualizations.
- [x] Use the official arXiv paper in both READMEs, the website's BibTeX and
  CITATION.cff preferred citation.
- [x] Record the author's Hugging Face distribution decision for assets and
  derived artifacts in [ASSET_DISTRIBUTION.md](ASSET_DISTRIBUTION.md).
- [ ] Publish and verify the Hugging Face dataset, then migrate tracked asset
  binaries while preserving local files. The dataset address is recorded;
  authenticated inspection confirmed a private historical archive. Migration
  of tracked assets and verification of their replacement downloads are pending.
- [ ] Resolve the redistribution rights of Piper camera/stand meshes and their
  derived point clouds. The pinned `piper_isaac_sim` has no license file and its
  `piper_description/package.xml` says `TODO: License declaration`.
- [ ] Verify per-file FR3, Franka Hand, D435i and custom mount provenance,
  revisions, copyrights and notices. Existing model-scope license files cover specified
  components and are not a blanket clearance for the assembled asset bundle.
- [ ] Confirm rights to distribute paper figures, school branding, task video
  excerpts and tracked emergency-stop output. Source provenance alone is not a
  redistribution grant.
- [x] Verify anonymous fresh clone: the current public repository and all fixed
  recursive submodules initialize successfully over HTTPS. See the structured
  clone record in [FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md).
- [ ] Verify the full local CUDA workflow. Record exact dependencies,
  model/checkpoint hashes, runtime and peak GPU memory.

The core distribution uses editable checkout installation. A standalone wheel
with independently resolved robot assets is outside the declared first-release
installation path. Complete paper experiments and real robot execution still
require separately documented inputs, validation and hardware.

## Paper And Public Workflow Preparation

- [x] Replace the old illustrative README image with representative figures
  from the supplied preprint; add the exact title, author list and university marks.
- [x] Create a static GitHub Pages site with paper PDF, result visualizations,
  method figures, citation and public implementation status. Media provenance is
  recorded in [ASSET_SOURCES.md](ASSET_SOURCES.md).
- [x] Add [WORKFLOW.md](WORKFLOW.md) and [DATA_CONTRACT.md](DATA_CONTRACT.md),
  including current data roots, delivery flags and the Codex/DSH distinction.
- [x] Fix passing Stage 2 baselines to emit `final.blend` and an accepted
  delivery manifest usable by Stage 3. Five new regressions cover accepted and
  rejected baseline cases, zero rounds and resume provenance.
- [x] Repeat the full offline suite after that fix: **293 passed, 2 real-model
  tests deselected**, in **51.39 seconds**, using the existing macOS checkout.
- [x] Verify the site with Playwright at 1440×1000, 768×1024 and 390×844:
  all media rendered, no document overflow, tab keyboard navigation, figure
  enlargement, citation copying and no-JavaScript fallback passed.
- [x] Publish the site using `master` and `/docs`. GitHub Pages reports `built`
  for `6ae6caf`; deployment `38436840592` succeeded. Repeat the browser checks
  against the published URL, including asset-link HEAD checks. The live PDF
  transfer was not completed for checksum verification; the local PDF matches
  the supplied original. See [PAGES.md](PAGES.md).
- [x] Inspect all 29 commits reachable at `6ae6caf` for selected recognizable
  token/private-key signatures and sensitive file paths: zero matches. This
  narrow history scan is additional evidence, not proof that all secrets or
  private data are absent; private working-tree environment files were not read.
- [x] Replace the external-home tabletop integration input with the committed
  fixture. Enable Blender tests through `BLENDER_BIN` or PATH as well as the
  macOS application. All four affected integration tests passed.
- [x] Repeat the complete current-checkout offline suite after the portability
  edits: **293 passed, 2 real-model tests deselected**, in **54.11 seconds**.
  The local JUnit record is `tmp/open-source-portable-offline.xml`, SHA256
  `cdb8d3bcf2d94c15bfea3af2936d26f47f3639fd4687fd8daeae524ff454fa67`.
- [x] Remove the tabletop preview's external `piper_demo` import and use the
  existing identical six-joint pose. Both preview tools now use the committed
  scene by default; their render/control reports were generated without model
  weights, remote perception or Codex calls. Reports are retained locally in
  `runs/public_randomization_preview_20261014/report.json` and
  `runs/public_sim_profiles_20261014/report.json`. macOS MuJoCo warned that
  `ARB_clip_control` was unavailable; this render check does not validate depth
  precision or real SAM3 inference.
- [x] Expand [the robot asset evidence](THIRD_PARTY.md) with exact nested source
  revisions, file groups and unresolved notices. Fix the nine FR3 collision
  paths to existing, provenance-checked vendor meshes and regenerate the bound
  Franka surface metadata; custom mount and conversion rights remain open.
- [x] Verify anonymous public-source archive retrieval and locked `sim,dev`
  installation in a new Python 3.11.15 environment with an empty HOME. Original
  source: **288 passed, 5 skipped, 2 models deselected**; after the two test
  portability edits: **290 passed, 3 skipped, 2 models deselected**. The remaining
  skips require missing derived scenes. Piper Isaac used verified sparse
  resources; full VGGT source and `local,agent` installation were not verified.
  See [FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md).
- [x] Verify a complete anonymous recursive Git checkout. A clean HTTPS
  `--recurse-submodules --depth 1` clone retrieved the current main commit,
  all fixed submodule commits, the complete VGGT tree and 10/10 FR3 collision
  paths. See [anonymous clone evidence](FRESH_SOURCE_VALIDATION.md#fr3-urdf-collision-path-repair)
  and the structured provenance record. This does not verify dependency
  installation or Linux CUDA inference.
- [x] Install `sim,dev` from that fresh clone with a new Python 3.11.15
  environment and run the offline suite: **333 passed, 3 expected derived-scene
  skips, 2 model tests deselected**. Record:
  [fresh-clone-offline-20261014.json](provenance/fresh-clone-offline-20261014.json).
- [ ] Complete Linux CUDA checkpoint inference and benchmark publication.

## Additional Portability And Provenance Checks

- [x] Unify Blender discovery across the Stage 1/2/3 entry points and related
  scripts: explicit argument, environment, PATH and macOS-only fallback.
  Invalid configured executables fail explicitly. The current macOS checkout
  passed 310 offline tests; the isolated archive plus current patches passed
  307 with 3 scene-dependent skips. Both excluded 2 real-model tests.
- [x] Verify Linux CPU installation and offline contracts in a new Debian 13,
  Python 3.11.17, `linux/amd64` Docker container on macOS arm64. Locked `sim,dev`
  installation and four CLI startup checks succeeded; 305 tests passed, 5
  skipped and 2 real-model tests were deselected. No weights, author environment
  or derived scenes were mounted. See [FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md)
  for exact source limitations; this does not establish native Linux CUDA or
  complete model installation.
- [x] Compare all 64 fixed FR3 MJCF asset files to pinned upstream Git trees:
  9 byte-identical matches and 50 CRLF/LF-only matches. Preserve the full FR3
  Apache/BSD license and Intel statement. Per-file hashes and source paths are
  committed in [fr3-assets.json](provenance/fr3-assets.json).
- [ ] Resolve the 5 unmatched FR3 files and the remaining URDF-side assets,
  generated surfaces and Piper camera/stand rights. Matching contents do not
  prove original import history or grant rights for other files.
- [x] Match both original D435 DAE files to the fixed official RealSense source
  and preserve its byte-identical LICENSE and NOTICE in [notices/](notices/README.md).
  The official description package declares Apache License 2.0. Custom mounts,
  stand geometry, converted meshes and mixed-source point clouds remain unresolved.
- [x] Download complete pinned VGGT (113 files) and Piper Isaac (252 files)
  archives and verify every file's official Git blob hash. A new main archive
  at `08cd909`, the complete source set and a new macOS environment successfully
  installed locked `local,sim,agent,dev` extras. Model/SAM3/SDK imports and CLIP
  tokenization passed; 307 offline tests passed, 3 scene tests skipped and 2
  model tests were deselected. Wheel and Git caches were reused. Details and
  the separate anonymous Git transport failure are in [FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md).
- [x] Download the complete public paper PDF and verify its SHA256 against the
  committed and supplied originals. The `08cd909` Pages deployment succeeded.
- [x] Fix Stage 2 resume acceptance: bind source/reference/plan/evaluation inputs
  and cached scene/render files to SHA256, refuse changed or unbound caches, and
  recompute baseline and candidate metrics. Require explicit resume for existing
  geometry before starting a new planning agent. Historical delivery readers
  remain compatible. Fix the empty Required capabilities section validation.
  The current macOS checkout passed 334 offline tests, with 2 real-model tests
  deselected, in 55.07 s; 24 added regressions use synthetic inputs and no agent.
  Details and the remaining Linux/CUDA scope are in [FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md).
- [x] Preserve task constraints when deriving SIM scenes containing unnamed
  robot elements. Piper/Franka synthetic XML regressions cover task actuators,
  welds and contact rules; the final current macOS checkout passed 336 offline
  tests, with 2 real-model tests deselected, in 54.59 s. This does not establish
  scene dynamics or local SAM3 checkpoint observation.
- [x] Verify the `e693451` Pages deployment and byte-identical published macOS
  archive report, URDF asset evidence and RealSense LICENSE / NOTICE. A selected
  secret-signature/path scan of all 33 main-repository revisions reported no
  matches; submodule histories and arbitrary credential formats were not covered.
- [x] Attempt locked Linux `local,sim,agent,dev` installation with complete
  sources and an empty HOME/new wheel cache. The attempt failed downloading
  `nvidia-cusparselt-cu12` after TLS errors; model imports and tests did not run.
  Save the failure and cleanup in [FRESH_SOURCE_VALIDATION.md](FRESH_SOURCE_VALIDATION.md)
  and its structured report. Linux all-extra/CUDA acceptance remains unchecked.
- [x] Expand the existing sparse VGGT submodule from the verified official
  archive without changing its commit, Git tree or remote. All 113 files match
  the pinned Git blobs and modes; no skip-worktree entries remain. Import
  `VGGTOmega` from this checkout using the existing macOS Python 3.11 environment
  without constructing a model. Publish the scoped evidence in
  [vggt-local-materialization-20261014.json](provenance/vggt-local-materialization-20261014.json).

## Decisions To Record

- Main audience is paper reproduction and reusable tools; the initial scope is
  local Stage 1 perception and SIM visual observation on Linux CUDA/Python 3.11.
- Main-project code and documentation use Apache-2.0, as selected by the author.
- Assets and derived artifacts are distributed through Hugging Face; the
  private archive address and authenticated access are verified in
  ASSET_DISTRIBUTION.md; the public asset file list and migration remain pending.
- Which videos, scenes, derived robot assets and experiment evidence may ship.
- Whether to preserve current history or publish a curated release repository.
- Supported platforms, model dependencies and external perception protocol.

Do not infer these decisions from the fact that a GitHub repository is public.
