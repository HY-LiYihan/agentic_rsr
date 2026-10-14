# Sim Architecture

## 1. Design Position

Sim does not reconstruct the scene. Real2Sim Stage 3 produces a nominal MuJoCo
scene that is as close as possible to the accepted real-scene reconstruction.
Sim creates runtime episode instances from that nominal scene, adds controlled
variation, exposes only deployment-compatible observations to Codex-generated
programs, and evaluates the result with a separate truth-aware evaluator.

```text
Stage 3 accepted scene.xml
          │
          ▼
Sim episode reset
  ├─ exact sensor profile
  └─ randomized sensor/scene profile
          │
          ▼
Codex Agent harness
  ├─ required policy.py + optional steps/*.py
  ├─ policy execution loop selector
  └─ public step results returned to the same conversation
          │
          ▼
RobotSession + perception interfaces
          │
          ▼
MuJoCo runtime instance
          │
          ▼
truth-aware evaluator and manifest
```

The Stage 3 source scene is immutable during this process. Every episode gets
an isolated runtime instance or a verified reset to the nominal state.

### Stage 3 agent development curriculum

The Stage 3 Codex runner (`scripts/run_stage3_codex_batch.py`) first develops
and checks a policy in three **ordered** stages:

1. `privileged_exact`: nominal scene; the policy may call `object_poses()`.
2. `visual_exact`: same nominal scene; remove the object-pose oracle and
   require camera observations. Start a fresh author conversation, while
   supplying the previous `policy.py` and its written experience as editable
   development memory. The new stage must re-observe and cannot call the oracle.
3. `visual_randomized`: keep the oracle disabled, and enable bounded object,
   camera/calibration and commanded-pose variation. The visual-exact policy can
   be used as the initial candidate, but must re-observe at runtime. Each stage
   writes an experience document for the next fresh author session.

Advance only after an independent physical goal check or human video review;
`completed` alone is not evidence. This **development** curriculum may revise
the policy between stages. The locked, held-out exact/randomized evaluation
described below applies after development and must not be confused with these
three authoring steps. Each stage retains its own transcript, policy, action
history, and video; randomized resets resample from the nominal Stage 3 XML.

## 2. Two Required Evaluation Profiles

Every task has two profiles. Passing only one profile never marks a task as
complete.

### 2.1 Exact profile

The exact profile is the first bring-up environment:

- Stage 3 nominal geometry and object state;
- nominal camera pose and calibration;
- no injected control error;
- deterministic physics and reset;
- no perception noise.

“Exact” means the public sensor/perception interface is noiseless and aligned
with the nominal scene. It must not expose MuJoCo `model`, `data`, object truth
poses, contacts, or evaluator-only state to the policy. A temporary oracle
debugger may exist outside the policy interface, but oracle success is not a
deployment result.

The exact profile answers: **can the generated program solve the task at all?**

### 2.2 Randomized profile

The randomized profile starts from the same accepted Stage 3 scene and applies
bounded episode variation:

- object position/orientation and initial-state variation;
- physical camera installation variation;
- camera calibration/extrinsic error supplied to perception;
- RGB-D, segmentation, and observation error when enabled;
- control position/orientation error, delay, or bounded execution noise;
- later, only evidence-backed physics variation.

The randomized profile answers: **does the same program remain usable when the
scene, observation, and execution are not perfectly aligned?**

The physical camera pose is used for rendering. The estimated camera pose is
used by the perception/coordinate-conversion path. These two values are never
silently replaced with one another.

## 3. Completion Gates

For a task/run to be accepted:

```text
exact profile       PASS
randomized profile  PASS
both required       task complete
```

The initial single-task bring-up may use one exact episode and a small fixed
randomized set. The randomized set must use seeds that are independent from
the seeds used while repairing the program. For a larger benchmark, the gate
should be expressed as a versioned threshold over a held-out seed set, while
the first vertical slice may require every selected episode to pass.

The evaluator reports at least:

- task success and completion condition;
- failure stage: generation, observation, planning, IK, execution, grasp,
  placement, scoring, or setup;
- collision, workspace, timeout, and safety events;
- commanded versus measured pose error;
- profile, seed, scene version, policy hash, and randomization configuration.

## 4. Runtime Architecture

### 4.1 Canonical scene and runtime instance

```text
Stage 3 scene.xml
  └─ immutable source artifact

SimSceneAdapter
  └─ loads/copies scene into a runtime MuJoCo instance

SimEpisode
  ├─ reset_nominal()
  ├─ apply object randomization
  ├─ apply physical camera pose
  ├─ configure estimated camera calibration
  └─ inject control error at execution boundaries
```

Current implementation locations:

- `src/sim/randomization.py`: distributions, bounds, sampling, manifest data;
- `src/sim/episode.py`: episode reset and the `EpisodeScene` adapter boundary;
- `src/sim/scene_adapter.py`, `observation.py`, `program_runner.py`, and
  `harness.py`: Piper tabletop-only prototype of the runtime, exact task-object
  observation or network SAM3 observation, worker, and dual-profile runner.
  The independent task registry, complete exact scene perception, production
  isolation, and Codex-mediated staged workflow are still future work.

`SimEpisode` must not write back to the Stage 3 `.blend` or source `scene.xml`.

### 4.2 Policy-facing observation boundary

The Agent receives an observation object, not the MuJoCo runtime:

```text
Observation
  ├─ RGB / RGB-D frames
  ├─ camera intrinsics
  ├─ estimated camera extrinsics
  ├─ segmentation or object observations
  ├─ robot state allowed by the real interface
  ├─ task description and public task state
  └─ previous action/result/error summary
```

The following are evaluator/environment-only:

```text
true object pose
true physical camera pose
true contact pairs
true target pose after control noise
MuJoCo model/data handles
unrestricted body/joint access
```

The same policy-facing interface should be usable with the future real backend.

## 5. Codex Agent Harness

The harness is a conversation-driven controller around a generated program. It
does not ask Codex to directly operate MuJoCo or issue raw robot commands.

### 5.1 Conversation inputs

At the beginning of a run, Codex receives:

- task instruction and task contract;
- available object names only when visible through the public observation;
- observation schema and examples;
- action/tool schema;
- workspace, speed, timeout, and safety limits;
- selected profile: exact or randomized;
- current episode observation;
- success and failure conditions;
- whether the run is monolithic or staged.

The harness also tells Codex that final acceptance requires both profiles. A
program that succeeds only because it uses exact-state shortcuts is rejected by
the policy interface or fails the randomized profile.

### 5.2 Execution-grounded programmatic policy and policy execution loop

Codex produces a locked bundle; the mandatory `policy.py` defines:

```python
def run(api):
    observation = api.observe()
    return {"status": "completed", "state": {"phase": "located"}}
```

Optional `steps/<name>.py` files define `run(api, state)` and return JSON
results, optionally with an updated `state` object. A policy may call
`api.observe()` many times without consulting Codex.

The harness then:

1. validates and hashes the entire bundle;
2. starts one persistent SIM episode;
3. executes `policy.py` first in a limited worker process without a Codex turn;
4. if it completes, runs the independent evaluator directly;
5. otherwise forks the authoring conversation for the current episode, sending the
   running command, public intermediate results, and a fresh observation;
6. in that disposable fork, accepts checkpoints, child policies, direct tools,
   retries, completion, or abort;
7. runs the hidden evaluator after completion in both profiles.

For the Piper prototype, the harness owns one persistent `openai-codex` Python
SDK authoring thread per run, using read-only sandbox and denied approvals.
The SDK child process uses a run-local `CODEX_HOME` (`--output/codex_home/`)
by default, or a reusable project-local directory with `--codex-home`,
separate from the user's global Codex CLI/Desktop home. Bundle generation and
reset/refinement share the authoring thread. Supplied bundles first record task
context in that thread. Only interrupted or incomplete episodes natively fork
an ephemeral evaluation thread; its decisions never mutate the authoring thread.
The process saves an audit transcript separately from episode logs and delivers
the accepted `final_policy/` and `real_handoff.json` with the home path and
authoring thread ID. A later REAL process can resume that ID using the same
`CodexConfig(env={"CODEX_HOME": ...})`. The transcript alone cannot recreate
the SDK conversation. Keep the private home out of version control; it may need
separate authentication and local storage does not imply offline inference.
No `codex exec` call or interactive-thread resume occurs.

Execution is Agent-authorized: the authoring session must explicitly request
`mode="execute", command="run_policy"` before the harness runs the current
`policy.py`. After a result, the same session may request another execution,
return a revised policy, reset through the public API, or end at a checkpoint.

This is the default path. A direct tool action passes through the same
public robot/task safety checks as generated code. The whole bundle is
locked across exact and randomized episodes. A normal one-shot `policy.py`
completion is evaluated without a Codex review turn.

The Piper slice uses a locked **Execution-Grounded Programmatic Agent Policy**
bundle. `policy.py` is mandatory and may
be accompanied by `steps/*.py` reusable policies. Each policy may call
`api.observe()` repeatedly without pausing. A checkpoint, failure, or requested
next step sends the result and a fresh public observation to a temporary fork
of the authoring conversation. In that episode-local fork, Codex can
run `policy.py`, run a named child policy again, call a public robot/task tool
directly, finish, or abort. JSON state passes between policy invocations, so
this gives safe semantic goto/retry without source-line jumps or modifying the
locked bundle. The same bundle runs unchanged in exact and randomized
episodes. Exact currently provides task-object coordinates without perception
noise, not every possible scene datum.

### 5.3 Reusable child strategies

For complex tasks, Codex can invoke the optional steps in any order:

```text
observe_and_identify
  → approach_and_grasp
  → transport
  → release
  → verify_and_recover
```

Each stage has:

- input observation schema;
- allowed tools;
- action program or action sequence;
- completion condition;
- timeout and failure condition;
- recovery or escalation policy;
- output state summary for the next Codex turn.

The harness takes a fresh public observation after each invocation and sends
Codex a structured report. Codex can repeat a named step or directly call a
public tool rather than jump to a source line. It cannot silently rewrite a
locked step or access evaluator truth. Arbitrary Stage3 scenes and production
isolation remain future work.

### 5.4 Outer reset-and-refine loop

The policy execution loop is nested inside an outer development loop. Reset is
owned by the harness, not exposed as a policy tool, so a generated program
cannot use reset as an action shortcut. `run_episode()` creates a fresh runtime
instance and applies the same task profile and seed again; this is the SIM
one-click reset boundary.

```text
candidate bundle
  → reset fresh exact episode
  → exact failure report → same Codex session → replacement bundle
  → reset fresh randomized development episode
  → randomized failure report → same Codex session → replacement bundle
  → exact + randomized development pass
  → lock bundle
  → held-out randomized evaluation
```

`src/sim/policy_refinement.py` implements this flow. Every replacement is a
complete validated `policy.py` plus optional `steps/*.py` bundle written under
a new `policy/iteration_NNN/bundle/` directory; previous candidates are never
overwritten. Only public observations, action history, errors, and the
profile-level pass/fail result are sent to the refinement prompt. Evaluator
coordinates and post-noise truth are removed before that report reaches Codex.
The development seed is separate from held-out seeds. Once both development
profiles pass, the candidate is locked and later evaluation cannot modify it.

## 6. Program and Tool Contracts

The first implementation should expose a small, explicit tool set:

```text
observe()
robot.state()
robot.move_pose(pose)
robot.move_joints(joints)
robot.gripper(width)
robot.stop()
task.object_names()
task.check_stage(condition)
task.check_success()
```

Episode reset is intentionally a harness operation rather than a policy
operation. The policy may observe, act, verify, and ask for bounded recovery;
only the outer refinement controller may start a new episode.

The Agent may use NumPy and pure planning helpers, but it must not import
MuJoCo, access backend internals, inspect XML, or call a truth-state helper.
The harness should validate generated code using AST checks and execute it in a
time-limited worker process. The exact allowed imports and resource limits must
be versioned with the task.

The task layer supplies semantics and checks; the robot layer supplies motion;
the perception layer supplies observations; the harness coordinates them. No
single generated program should reimplement scene loading, safety gates, or
truth-aware scoring.

## 7. Artifact and State Model

Every run gets a unique directory:

```text
runs/sim/<run_id>/
├── manifest.json
├── task.json
├── policy/
│   ├── policy.py
│   ├── steps/*.py
│   └── policy_manifest.json
├── exact/
│   └── episodes/<episode_id>/
│       ├── episode_manifest.json
│       ├── observations.jsonl
│       ├── actions.jsonl
│       ├── checks.jsonl
│       └── result.json
├── randomized/
│   └── episodes/<episode_id>/...
├── evaluation.json
└── report.html
```

The run state machine is:

```text
created
  → scene_ready
  → policy_generated
  → exact_running
  → exact_passed
  → randomized_running
  → randomized_passed
  → accepted
```

Failure transitions preserve diagnostics:

```text
any state → failed
any stage → review_required
```

An episode manifest records the Stage 3 scene hash, backend commit, task
version, profile, seed, randomization config, policy hash, observation schema,
tool calls, timing, and failure reason.

## 8. Relationship to `code_as_policy`

`code_as_policy` is a reference for the shape of a tabletop task suite and for
the idea of generating executable Python policies. It is not the architecture
of this project and should not own the Sim runtime.

The first adapter can reuse its semantic ideas:

- deterministic task instances;
- scene/task separation;
- explicit success scoring;
- structured trial records;
- optional RGB-D perception track.

The current project should provide its own task registry, observation contract,
episode lifecycle, Agent harness, safety boundary, and manifest schema. Once a
Piper tabletop task works, the same harness should accept a Stage 3-generated
scene rather than being tied to the reference repository's XML template.

## 9. Implementation Order

1. Implement a concrete MuJoCo `EpisodeScene` adapter for one accepted Piper
   Stage 3 scene.
2. Add exact-profile reset, RGB-D/observation contract, and truth-separated
   evaluator.
3. Register one task and one complete scripted Python program without Codex.
4. Add the Codex monolithic harness and generated-program validation.
5. Add exact-profile Codex execution and artifact recording.
6. Add randomized profile with held-out seeds and the current randomization
   layer.
7. Add staged workflow execution and stage-level checks/recovery.
8. Reuse the same task program and `RobotSession` interface for Sim2Real.

The first vertical slice should therefore be:

```text
one Stage 3 Piper scene
  → one tabletop task
  → exact profile
  → one complete Python program
  → randomized profile
  → structured report
```

## 10. Codex Conversation Protocol

Codex is the policy author and debugger; it is not the simulator and it is
not the final evaluator. The harness owns the conversation state and sends
Codex only public observations plus structured execution reports.

### 10.1 Run phases

```text
task_loaded
  → capabilities_advertised
  → program_or_workflow_received
  → exact_debug_loop
  → exact_candidate_locked
  → randomized_validation
  → accepted | repair_required | failed
```

The first prompt must state all of the following explicitly:

- which task is being solved and what counts as success;
- the available public interfaces and their schemas;
- the active profile (`exact` or `randomized`);
- resource limits, safety restrictions, and reset behavior;
- whether the response must be a complete Python program or a staged
  workflow;
- that both profiles are mandatory for acceptance.

During development, a failed exact run may return a repair report to Codex.
After a candidate is locked, the randomized validation runs the same policy
artifact. A profile result must record the policy hash, and the evaluator must
reject a report that claims both profiles passed with different policy hashes.

### 10.2 Monolithic response contract

The preferred response is one self-contained Python module with a stable
entry point:

```python
def run(api) -> dict:
    observation = api.observe()
    # plan and act through api only
    return {"status": "completed"}
```

The harness parses and validates the module before execution. The returned
dictionary is a self-report only; environment truth and the task evaluator
remain authoritative. The harness stores the exact source, normalized hash,
stdout/stderr, tool trace, and final evaluator result.

### 10.3 Staged response contract

When a single program is too brittle, Codex may return a declarative workflow
whose stages are independently executable:

```json
{
  "mode": "staged",
  "stages": [
    {"name": "identify", "entrypoint": "identify.run"},
    {"name": "grasp", "entrypoint": "grasp.run"},
    {"name": "transport", "entrypoint": "transport.run"},
    {"name": "release", "entrypoint": "release.run"},
    {"name": "verify", "entrypoint": "verify.run"}
  ]
}
```

At every stage boundary the harness:

1. records the previous stage's public result and actions;
2. obtains a fresh observation from the same episode;
3. runs the stage checker and safety checks;
4. sends Codex a `StageReport` containing observations, errors, and the next
   allowed actions;
5. either continues, requests a bounded recovery, or terminates the episode.

The workflow cannot edit history, inject an unrecorded action, or convert a
truth-only evaluator result into a policy observation. A reset starts a new
episode and therefore receives a new episode id and seed.

## 11. Concrete Module Boundaries

The planned module layout is deliberately split by authority:

```text
src/sim/
├── randomization.py       # bounded, seeded variation
├── episode.py             # lifecycle and action-boundary noise
├── scene_adapter.py       # MuJoCo runtime implementation
├── observation.py         # public sensor schema and adapters
├── policy_refinement.py   # reset episodes, refine bundles, lock/evaluate
├── task_registry.py       # task specs, prompts, checks, success rules
├── evaluator.py           # truth-aware scoring and failure taxonomy
├── program_runner.py      # validation, subprocess execution, limits
└── harness.py             # Codex conversation and profile orchestration
```

The dependency direction is one-way:

```text
harness → program_runner → public API facade → observation / episode / task
                                              ↘ scene_adapter → MuJoCo
evaluator ─────────────────────────────────────→ scene_adapter truth
```

Generated code depends only on the public API facade. `evaluator.py` may read
truth from the runtime adapter, but no evaluator dependency may be imported by
the generated program. This separation makes it possible to run the same
program first in Sim and later through a real robot facade.

## 12. Non-Negotiable Invariants

The implementation should enforce these invariants in code and in tests:

1. Stage 3 source artifacts are immutable; every episode resets an isolated
   runtime instance or proves an equivalent clean reset.
2. Exact and randomized runs use the same task, interface version, and policy
   hash; only the profile and episode seed set differ.
3. Exact success is necessary but never sufficient for acceptance.
4. Randomized validation uses held-out seeds that were not used to repair the
   policy.
5. The public observation contains estimated/calibrated data only; physical
   camera pose, object truth, contacts, and actual post-noise poses remain
   evaluator-only.
6. Every action, observation, reset, stage transition, and evaluator result is
   append-only recorded with timestamps and episode identity.
7. Safety limits terminate an episode before the generated program can issue
   an out-of-bounds or otherwise forbidden action.
