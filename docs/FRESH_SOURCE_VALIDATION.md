# Fresh Public Source Validation

Date: 2026-10-14. This record verifies an offline installation from anonymously
retrieved public sources on macOS arm64. It does not establish a complete
recursive Git checkout, Linux CUDA inference or paper-result reproduction.

## Sources And Isolation

The main source was downloaded anonymously from
`https://codeload.github.com/HY-LiYihan/agentic_rsr/tar.gz/6ae6cafc22126be45fbd82e95c254bb6a2dde159`.
Its complete archive SHA256 is
`c705560985b25eda7fa438bad110f06558d6188941d18ccd930d44b165504fb1`.
The sources were extracted into a new temporary tree, with a new `.venv` and
Python 3.11.15. Tests used an empty temporary `HOME`, no `.env`, no author videos
or scenes, no weights and no remote perception. The `data/` directory contained
only the four committed README files. Existing uv wheel caches were reused;
editable project and robot-control imports resolved to the new temporary tree.

Complete anonymous fixed-revision archives were also retrieved for:

| Repository | Revision |
| --- | --- |
| `HY-LiYihan/robot_control` | `6016dd5f20d2146a28e425ce77dfb9430f1173` |
| `agilexrobotics/agx_arm_urdf` | `f6642ce0d7872c686f29c99e9e10cd23d1d49313` |
| `agilexrobotics/piper_sdk` | `c9e8a28174e71eeaac448593cb65f8ab258a92fe` |
| `ultralytics/ultralytics` | `f171c4b7aea56062d77061577e6b878b62e520fb` |

The complete Piper Isaac archive transfer was stopped after slow downloads.
The following required resources were downloaded from the official
`raw.githubusercontent.com/agilexrobotics/piper_isaac_sim/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2/`
source and each checked against its pinned Git blob hash:

| Resource | Git Blob SHA1 |
| --- | --- |
| `piper_description/urdf/piper_description_v100_realsense_camera_v2.urdf` | `d8933637a1535f132271938d6d768830a7b7bd30` |
| `piper_description/meshes/dae/realsense_mid_stand.dae` | `567e99a80f656265d86adf3869c1b195a1c75e24` |
| `realsense2_description/meshes/d435.dae` | `0ed32368fc5ba6912be530395a72553de9f21d9e` |

Their SHA256 values also match the committed Piper surface metadata. At this
historical archive attempt, these resources sufficed for the tested paths; the
complete upstream submodule was not retrieved. The VGGT archive timed out; only
its official pinned `pyproject.toml` was retrieved for dependency resolution,
not its complete inference source. A later clean anonymous recursive clone
retrieved the complete sources; see the follow-up record below.

## Installation And Checks

```bash
env -u PYTHONPATH -u VIRTUAL_ENV uv sync --locked --python 3.11 --extra sim --extra dev
```

Installation exited zero, resolving 124 lock packages and installing 43 selected
packages. Installed versions include NumPy 1.26.4, OpenCV 4.11.0.86, MuJoCo
3.14.0, Pinocchio 2.7.0, Boost 1.83.0, trimesh 5.1.0 and pytest 9.1.1.
Neither `local` nor `agent` extras were installed in this environment.

```bash
python -m pytest -m 'not models' -q -ra
```

| Source Under Test | Passed | Skipped | Real Model Tests Deselected | JUnit Duration |
| --- | --- | --- | --- | --- |
| Unmodified public source at `6ae6caf` | 288 | 5 | 2 | 42.537 s |
| Same source plus the two test portability edits | 290 | 3 | 2 | 45.725 s |

Both runs exited zero, with no failures or errors. The second run changed only
`tests/integration/test_sim_tabletop.py` to use the committed primitive fixture
and `test_stage1_blender.py` to discover `BLENDER_BIN` or PATH before the macOS
fallback. Production source remained at the archive revision. Blender used the
installed host application; this was not a newly provisioned Linux system.

The three remaining skips require absent derived Stage 3 emergency-stop scenes:
`test_derived_stage3_scene_exact_episode`,
`test_physical_gripper_automatically_attaches_only_on_bilateral_contact` and
`test_randomized_stage_resets_and_exact_restores_nominal`. Their behavior is not
cleared by this test run.

Stage 1, Stage 2, Stage 3 export and local-example CLI `--help` checks exited
zero. `real2sim-models check` exited one and reported both backends as `local`,
with missing checkpoints, inference dependencies and Git-initialized model
sources. That expected prerequisite failure is not a model-inference pass.

## Evidence And Remaining Gates

Local raw evidence is retained under `/tmp/agentic-rsr-anonymous-fresh.SbjMHb/`:

| Record | SHA256 |
| --- | --- |
| `report.json` | `c616dcd2731e6d52b653f99bcd34c523d82a5680ead1a1d60a6edb410566e204` |
| `original-tests.xml` | `825ac3219aafa36fe05544971f36d37d9e3c4494406636d3fb61d445cc6442e4` |
| `patched-tests.xml` | `ec859cccdba259969a9651ff1b8ba1a15c0e3796a7156a84410c71e31312b97f` |

This temporary evidence is not bundled with the repository and may be cleaned
by the host. The table records the inspected results and their exact scope.
All validation download, installation and test processes finished or were
explicitly stopped before this record was prepared.

Still required: full recursive checkout and `local,sim,agent,dev` installation
on Linux CUDA, actual video and SIM checkpoint inference, publishable task
inputs, main-project licensing and unresolved robot/media redistribution
evidence. See [OPEN_SOURCE_PREPARATION.md](OPEN_SOURCE_PREPARATION.md).

## Blender Portability Follow-Up

On 2026-10-14, the shared Blender resolver and its Stage 1/2/3 integrations
were checked on macOS without CUDA or model weights. The current checkout
passed 310 offline tests (2 real-model tests deselected; JUnit duration
61.805 s). The isolated anonymous-source environment above, with the current
production and test patches applied, passed 307 tests with 3 scene-dependent
skips (2 real-model tests deselected; JUnit duration 52.680 s). This second
check reused the earlier archive and environment; it was not a newly downloaded
public commit or a Linux installation.

The 17 resolver tests also passed in a focused rerun before publication.
The host Blender was 5.2.2 LTS. Linux executable discovery is covered by
controlled tests; Linux CUDA inference remains unverified.

| Record | SHA256 |
| --- | --- |
| `tmp/blender-portability-final.xml` | `7ff7a2e3f0fe6b9d0ee77379bfe83cbf52e564296cea2ff0c51a7b4184e12b88` |
| `/tmp/agentic-rsr-anonymous-fresh.SbjMHb/blender-discovery-tests.xml` | `2bd165d7d1fdf8eb183c87e5b1af339d0be66d0f7de44eb1a7e29ec11822268a` |

These temporary reports are not committed. The earlier immutable reports and
their results remain unchanged. GitHub test CI remains removed at the author's
request; GitHub Pages retains its website deployment workflow.

## Linux CPU Container Follow-Up

On 2026-10-14, the tracked main source at `af27d589a2d53409d22a9d1a69a581fe252ebbef`
was exported with `git archive` and combined with the previously anonymously
retrieved fixed external sources. A new Debian 13 `python:3.11-slim` container
ran as `linux/amd64` through Docker Desktop on macOS arm64. This is an emulated
Linux CPU check, not a native NVIDIA machine or a new anonymous recursive clone.
The image digest was
`sha256:e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce`.

The container used Python 3.11.17, uv 0.11.32, an empty temporary HOME and a new
virtual environment/cache. No author `.env`, weights, videos or derived scenes
were mounted. CUDA was disabled, Hugging Face offline flags were enabled and
MuJoCo used `MUJOCO_GL=egl`. Debian GL, GLib, EGL and Mesa libraries were
installed. The default Debian mirror failed with HTTP 502 and HTTPS TLS errors;
the Tsinghua Debian mirror completed installation with normal package signature
verification. Earlier failed attempts are retained separately from the final run.

```bash
uv sync --locked --python /usr/local/bin/python --extra sim --extra dev
.venv/bin/python -m pytest -m 'not models' -q -ra --junitxml=/evidence/tests.xml
```

Locked installation succeeded in 81.304 s. Tests exited zero: **305 passed,
5 skipped, 2 real-model tests deselected**, in 96.21 s (JUnit: 96.189 s).
Two skips require Blender, which was not installed; three require the same
absent derived Stage 3 scenes described above. Stage 1, Stage 2, Stage 3 export
and perception-example CLI `--help` checks all exited zero. The installed versions
were NumPy 1.26.4, OpenCV 4.11.0.86, MuJoCo 3.14.0, Pinocchio 2.7.0, trimesh
5.1.0 and pytest 9.1.1.

Commands, timings, dependency versions and exact skipped tests are committed in
[linux-offline-20261014.json](provenance/linux-offline-20261014.json).
Temporary raw evidence is under `/tmp/agentic-rsr-linux-check.OnUJvh/`:

| Record | SHA256 |
| --- | --- |
| `main.tar` | `69bb1d75e222bc2b955fe3871f4c898741f9c781bc0a561312d7c59c54d6fc08` |
| `evidence/report.json` | `6b1eb813c4a836264c6617f13bdccafe1cbb2abc0745be81fbd4010579de3521` |
| `evidence/tests.xml` | `1122895974d1ff6fd6be129a2dcd3287fc64edf08069235ce8d35de2184881c9` |
| `check.py` | `62a1e9a94670162c112b8e06f5a39bc473fba98f8fdd90ccc13d1116ba5e9e80` |

VGGT remained metadata-only and Piper Isaac used the previously verified sparse
resources. Linux `local,agent` installation, Blender execution, complete recursive
checkout and actual model inference remain unverified. Passing CPU contracts
does not clear any of those gates.

## Complete Archive And All-Extras Follow-Up

On 2026-10-14, anonymous HTTPS Git clone failed before fetching the main
repository, with a low-speed timeout after 45.661 s. An HTTP/1.1 metadata
request also failed to connect to GitHub port 443. No complete recursive clone
is claimed. Official codeload downloads subsequently succeeded; they provide
additional source and installation evidence without replacing that Git gate.

The main archive was newly downloaded at
`08cd9095a722d0a57952c1f44f89ad3c528cbbb5`, SHA256
`3e3f228b2c201e69f087ace55a66eafb53d41f0c1e1af2fb57992130492d6285`.
Complete VGGT and Piper Isaac archives were newly downloaded at their pinned
commits. Every file was checked against the complete official Git tree's blob
SHA, including the files omitted from the earlier sparse retrievals:

| Source | Commit | Verified Files | Archive SHA256 |
| --- | --- | ---: | --- |
| VGGT-Omega | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` | 113 | `7ee9d87cb0ab0efc42d9092332d3e0809ef379bafe2c7c293354378a5657793c` |
| Piper Isaac | `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` | 252 | `0766ca86dacf2e3632bc2052cb10569c9a8dbd9765600019e03c8c87694c21c2` |

The actual commit trees are `d3a0000eb234137ab80217c7686f3ce91fe5b035`
and `82109f1511cb0f3812fc0ce1ce1ebdde859811b3`, respectively. The earlier
complete anonymous robot-control, Agx, Piper SDK and Ultralytics archives were
reused. All seven archives were extracted into a new source tree; the earlier
partial archives were not used. Full VGGT and Piper source retrieval is now
verified for this macOS check, but the previous Linux run remains as scoped above.

With a new `.venv`, empty HOME and existing uv wheel/Git caches reused:

```bash
uv --no-config sync --locked --python 3.11 \
  --extra local --extra sim --extra agent --extra dev
.venv/bin/mjpython -m pytest -m 'not models' -q -ra
```

All-extra installation exited zero in 35.085 s on macOS arm64/Python 3.11.15.
Actual imports of VGGT's model class, Ultralytics SAM3, torch/torchvision,
timm, robot-control and `openai_codex` passed, and CLIP's packaged tokenizer
encoded a short robot prompt. Editable imports resolved into the new source
tree. No checkpoint model or paid agent was run. The complete offline suite
passed **307 tests**, with **3 derived-scene skips** and **2 real-model tests
deselected**, in 56.60 s (JUnit: 56.259 s). The two Blender tests used the
already installed host application; this check did not provision Blender.

The first result collector failed to parse Ultralytics startup messages as JSON
after a successful import command. The collector was repaired to write a
dedicated JSON file, with Ultralytics/Matplotlib configuration isolated under the
temporary HOME. Only post-install checks were rerun; installation was not repeated.
Original logs and the earlier collector report remain retained separately.

`real2sim-models check` exited one as expected: both backends remained local,
dependencies resolved, checkpoints were missing, archive sources lacked `.git`
metadata and the requested CUDA device was unavailable. It did not report
inference readiness. Archive file matching does not manufacture Git metadata
or relax the production source/version check.

The committed [macos-full-extras-20261014.json](provenance/macos-full-extras-20261014.json)
records all archive hashes, dependency versions, commands, timings and verified
source-file hashes. Temporary evidence is in `/tmp/agentic-rsr-public-clone.mSpgD0/`:

| Record | SHA256 |
| --- | --- |
| `install-report.json` | `63d1ddaaf8c0d94b945a9d70698956e3fd9312fc80f2a2bca7365d9f5f3a34b3` |
| `all-extras-tests.xml` | `7ac5eea4c7d648a94ac057faaf9c48d8ee1f97711be471cc79b77876f1c23208` |
| `vggt-archive-report.json` | `58825d3908c42d63deded64fa36a40e7452c628ebc3d2c91e4c5eb54ad3311ef` |
| `piper-archive-report.json` | `d673cd59a6798535899c209e343fac2d48be4ca47b9d5d75e004e99deb256d25` |

Still unverified: Linux all-extra installation, real CUDA checkpoint inference
and the complete paper experiment inputs. Source retrieval does not resolve the
remaining third-party rights.

## Stage 2 Resume Contract Follow-Up

The current checkout after `e693451` was tested using the macOS Python 3.11.15
all-extras environment described above. This check used the current checkout's
available derived scenes and host Blender, not a new anonymous checkout:

```bash
env -u PYTHONPATH -u VIRTUAL_ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES='' \
  /tmp/agentic-rsr-public-clone.mSpgD0/archive-source/.venv/bin/mjpython \
  -m pytest -m 'not models' -q -ra --junitxml=tmp/stage2-provenance-offline.xml
```

The suite passed **334 tests**, with **2 real-model tests deselected**, in
55.07 s (JUnit: 54.982 s). The new 24 regressions cover stale input/cache
rejection, recomputation of baseline and candidate metrics, valid unchanged
resume, historical missing-provenance rejection, refusal before a planning
agent is called, and task-object capability validation. The focused Stage 2
suite passed 29 tests. All these added cases use synthetic files and controlled
Blender calls; no checkpoint or paid agent was executed.

The new rejection cases and invalid robot-capability plans failed against the
previous implementation before the fix. Stage 2 previously reused a passing
baseline report after reference depths changed; the input/render provenance
checks now prevent that reuse. See [DATA_CONTRACT.md](DATA_CONTRACT.md) for the
new continuation metadata. Historical delivery reading remains unchanged;
missing provenance is not silently fabricated for older runs.

`tmp/stage2-provenance-offline.xml` SHA256 is
`f4699940b6c669134cbb5753d5d4cf883dda3375bbb5f39241fbd3be246f7d33`.
This local report is ignored and may be removed by the host.

A subsequent SIM derivation regression showed that unnamed robot bodies,
joints or geoms inserted `None` into the removal sets; missing attributes on
task constraints then matched those sets. The fix filters unnamed elements
before checking actuator, equality and contact references. Two synthetic
Piper/Franka cases verify that task motors, box/lid welds and contact rules
survive, while named robot constraints are removed. Both cases failed before
the fix and passed afterwards; no physics outcome or local SAM3 inference is
claimed by this XML contract test.

After that fix, the complete current-checkout suite passed **336 tests**, with
**2 real-model tests deselected**, in **54.59 s** (JUnit: 54.146 s). The combined
Stage 2/SIM focused suite passed 31 tests. The same command above was used with
`--junitxml=tmp/workflow-contracts-final.xml`, SHA256
`33e97c26719e2a8a65403253fd8dc1e88c692ff7e61d375ed80fe093585f1d5c`.

An earlier anonymous recursive HTTPS retry failed before main-source retrieval
because of a transient `github.com:443` connection timeout. A subsequent clean
retry passed: `git clone --recurse-submodules --depth 1` retrieved main commit
`a88d732`, all six fixed submodule commits, the complete 113-file VGGT tree and
the repaired FR3 bundle. It used an empty temporary HOME, disabled global/system
Git configuration, no credential helper and no credentials. The fresh checkout
was clean; all 10 FR3 mesh collision references resolved to files inside the
vendor bundle. This proves the public source checkout path, not installation,
checkpoint inference or paper reproduction. Structured evidence is in
[anonymous-recursive-clone-20261014.json](provenance/anonymous-recursive-clone-20261014.json);
raw clone output is in `/tmp/agentic-rsr-fresh-https.CoJl1T/`.

The same earlier retry directory retains a selected signature/path history scan:
33 main-repository revisions, zero matches, excluding submodule histories and
arbitrary credential formats.

## Fresh Clone Offline Installation

Using that clean public checkout, a new Python **3.11.15** environment and a
new uv cache, the locked `sim,dev` installation passed:

```bash
uv --no-config sync --locked --python 3.11 --extra sim --extra dev
```

It installed 43 packages, including NumPy 1.26.4, OpenCV 4.11.0.86, MuJoCo
3.14.0, Pinocchio 2.7.0 and the fixed robot-control source. With weights,
CUDA and remote perception disabled, the complete unit/integration command
passed **333 tests**, skipped **3** expected derived-scene cases and deselected
**2** model tests in 82.71 seconds. The fresh checkout itself remained clean.
The skips are the optional Stage 3 scenes and emergency-stop fixture that are
not distributed in the public source. This is fresh-checkout installation and
offline contract evidence; it does not establish local checkpoint inference,
Linux CUDA or paper results. See the structured
[record](provenance/fresh-clone-offline-20261014.json), JUnit SHA256
`f2a3426fb21598da4b3b6b9728e78a31782491201a98936ea4ecf520d846064f` and raw
checkout at `/tmp/agentic-rsr-fresh-https.CoJl1T/`.

The `e693451` Pages deployment succeeded (run `38452053734`). Complete downloads
of the published macOS archive report, URDF asset report and retained RealSense
LICENSE / NOTICE were SHA256-identical to the committed files. These checks
establish publication of those records, not model inference or rights to the
remaining unmatched assets.

## FR3 URDF Collision Path Repair

The main checkout now pins `robot_control` commit
`6016dd5f20d2146a28e425ce77dfb9430f1173` from the public
`codex/fr3-urdf-paths` branch. This is a focused descendant of the previous
`29fbcc3ad0b74bd2a42349ce107d13b827bc26ff` pin; it does not include the later
camera-extrinsics changes on `robot_control` `main`.

The official URDF previously referenced nine absent paths under
`../meshes/franka_collision/`. The submodule commit changes those references to
the existing `mujoco_fr3/assets/link0.stl` through `link7.stl` and
`fr3_hand_collision.stl`. Their local Git blobs already matched the fixed
Menagerie/Panda sources recorded in [fr3-assets.json](provenance/fr3-assets.json);
no new mesh bytes were introduced. The remaining custom mount collision mesh is
also present, so an XML inspection found **10/10** mesh collision references
resolvable inside the vendor bundle. The submodule adds a unit regression for
this path check; it passed **1 test**.

The old URDF SHA256 was
`8909d0a8fd65bbc9ccfabbec594502f21efd25552ee4210640bc539f75f4c1c1`; the new
SHA256 is `dafb5275ac6f8f86393500e1810366ff7342413652359b99aef85f98a366dea5`.
Because Stage 1 metadata binds this hash, the Franka 40k/10k/5k surface pyramid
was regenerated from the current checkout. Visual mesh count stayed 47 and
surface bounds stayed within floating-point sampling noise; the new metadata
records the new hash. Existing registration/delivery outputs bound to the old
hash must be regenerated rather than relabeled.

This repair establishes path completeness and source compatibility. It does not
grant redistribution rights for the custom mount, converted meshes, generated
point clouds or other unresolved assets.

After updating the submodule and regenerating the Franka surface pyramid, the
current checkout passed **336 offline tests**, with **2 real-model tests
deselected**, in **52.27 seconds** using the existing macOS Python 3.11.15
`mjpython` environment. CUDA, checkpoints, remote perception and paid agents
were disabled. The JUnit record is `tmp/open-source-current.xml`, SHA256
`deeb3c293ad574b889c27d821f25521ca31fa72ddfe9c93b4ce1281cbb14a8b2`.

## Linux Complete-Source All-Extras Attempt

That historical Linux attempt used tracked main source at `e693451` and all six
complete pinned external archives; it predates the focused FR3 submodule repair
documented above.
Unlike the earlier Linux CPU check, neither VGGT nor Piper source was sparse.
The image digest remained `e88e9763f943ec1834f992a4b51e0f24500486803e8bc534e5767af9ea65f6ce`.
HOME and the wheel cache were new; only the fixed CLIP Git cache was reused.
The documented system libraries and uv 0.11.32 installed successfully.

```bash
uv --no-config sync --locked --python /usr/local/bin/python \
  --extra local --extra sim --extra agent --extra dev
```

The all-extras installation **failed**, exit 1, after a recorded wall duration
of 1300.346 s. Official `nvidia-cusparselt-cu12==0.6.2` download requests failed
after four retries with `TLS handshake EOF`. Lock sources were not replaced.
The original 900 s collector deadline was extended to 1800 s while the installer
continued; the exit came from the download failure, not a timeout termination.
The recorded duration includes exit collection, not a model runtime benchmark.

No real imports, tokenizer check, complete pytest run, model check or focused
patch tests followed the failed installation. Five patch files at `31d43d5`
were prepared and their source hashes verified, but **not run on Linux**.
The macOS 336-test result and the older Linux `sim,dev` CPU result retain their
original scopes. This attempt does not upgrade either to Linux all-extra or
CUDA acceptance.

The `--rm` container was automatically removed after failure, taking its new
wheel cache and environment with it. No persistent cache volume was created,
so a same-container retry was unavailable; no second installation was started.
Docker was restored to its original stopped state and all processes finished.
User caches were mounted read-only and were not modified. The structured
[linux-full-extras-attempt-20261014.json](provenance/linux-full-extras-attempt-20261014.json)
retains commands, source/image versions, failure details, cleanup and raw report
hashes. Raw evidence remains in `/tmp/agentic-rsr-linux-full.djZB6Q/evidence/`.

## Local VGGT Full Checkout

The existing VGGT submodule at
`48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` was expanded locally from sparse
checkout to full checkout. Its authoritative local Git tree was already complete:
`d3a0000eb234137ab80217c7686f3ce91fe5b035`. The previously downloaded official
44,143,006-byte archive has SHA256
`7ee9d87cb0ab0efc42d9092332d3e0809ef379bafe2c7c293354378a5657793c`.

Before mutation, every archive member was checked against `git ls-tree -r -z
HEAD`, including its Git blob hash and executable mode. Exact path coverage,
safe archive paths, unique members and the 33 existing tracked files were also
verified. The verifier added 76 missing unique blobs using `git hash-object -w
--stdin`, then ran `git sparse-checkout disable`. It used
`GIT_NO_LAZY_FETCH=1`; no new network retrieval or direct source extraction was
performed. The 80 previously absent paths are now materialized.

All **113 files / 47,401,823 bytes** in the resulting checkout match the pinned
Git tree, including modes. No skip-worktree entries remain, the submodule is
clean, and its HEAD/tree and official HTTPS remote are unchanged. All six fixed
submodule revisions remain unchanged; the main worktree was clean after source
hydration, before recording this documentation.

An offline import probe used the existing macOS Python 3.11.15 all-extras
environment, explicitly selecting `external/vggt_omega` with `PYTHONPATH`.
It imported `vggt_omega.models.vggt_omega.VGGTOmega` from this checkout and
verified the actual module path and SHA256. PyTorch 2.6.0, torchvision 0.21.0
and timm 1.0.30 were present. It did not construct a model, load weights or
perform inference. Production source bytes did not change; the earlier full
suite was not rerun solely for this local Git-object hydration.

The structured [record](provenance/vggt-local-materialization-20261014.json)
contains all file blob hashes/modes, the import result and temporary verifier
hashes. Its SHA256 is
`d63594891fc9896ebd2a4082408f5ec315c51f1bdf25aa45551151ef7e607b03`.
Temporary verification scripts and raw evidence are in
`/tmp/agentic-rsr-source-materialization.frvbA3/`.

This check establishes complete source availability in the existing local
submodule. It does not establish anonymous recursive checkout, a fresh Linux
all-extras installation, local checkpoint inference or paper reproduction.
The separate failed Linux installation record above retains its original scope.
