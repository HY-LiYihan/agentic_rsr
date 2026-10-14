# Sim

预留任务环境、受约束的随机化、感知链、执行反馈驱动的程序化智能体策略、策略执行闭环及仿真评测。正式视觉策略不能读取仿真物体真值位姿；仅第一关的教学阶段开放真值接口。

Stage 3 Codex 运行入口 `scripts/run_stage3_codex_batch.py` 顺序执行三关：`privileged_exact`（名义场景、开放 `api.object_poses()` 和 CLI `object_poses`）、`visual_exact`（关闭真值、名义场景）、`visual_randomized`（保持关闭真值，开启物体、相机与控制随机化）。每关使用独立场景和全新的 ephemeral 作者会话；上一关的 `policy.py` 和 `experience.md` 会作为下一关的开发记忆传入，但下一关仍必须重新观测并通过当前关卡验证。每关必须通过独立任务检查（目前急停按钮放入托盘可自动检查），其他任务需提供视频人工复核 `video_review.json`，未验证不得晋级。运行目录保存每关 `policy.py`、经验文档、本地对话、动作日志和视频；`--random-seed` 设置随机种子，关内 reset 使用递增种子，不修改 Stage 3 的原始 XML。

完整的 Sim 运行架构、Execution-Grounded Programmatic Agent Policy / Policy Execution Loop 以及准确场景和随机化场景的双重验收见 `docs/SIM_ARCHITECTURE.md`。

## 当前随机化基础

`randomization.py` 提供每个 episode 的可复现随机化采样器；`episode.py` 定义通用运行时适配接口，Stage 3 通过 `stage3_scene.py` 在 MuJoCo 内应用扰动。Stage 3 源文件不会被修改：

- **物体场景随机化**：在名义物体位姿附近扰动位置和姿态，并检查 XY 工作空间与最小物体间距。
- **相机随机化**：分别记录物理相机位姿和提供给感知链的估计位姿。前者用于渲染，后者用于模拟安装/标定误差，避免把误差错误地当成真实场景变化。
- **控制随机化**：对任务程序请求的目标位姿采样位置和旋转到达误差，用于模拟控制不精确；`apply_control_error()` 返回实际到达位姿。

```python
from sim.randomization import EpisodeRandomizer, Pose3D, RandomizationConfig

randomizer = EpisodeRandomizer(
    seed=episode_seed,
    config=RandomizationConfig(
        workspace_min_xy_m=(-0.19, -0.19),
        workspace_max_xy_m=(0.01, 0.19),
        min_object_separation_m=0.08,
    ),
)
objects = randomizer.sample_object_poses(nominal_objects)
camera = randomizer.sample_camera("wrist", nominal_camera)
control_error = randomizer.sample_control_error("move_to_pick")
arrived_pose = randomizer.apply_control_error(commanded_pose, control_error)
episode_manifest = randomizer.to_manifest()
```

采样结果应写入 episode manifest。策略只能看到经过相机估计位姿和感知链处理后的观测；物理物体位姿、真实相机位姿和实际控制误差只供环境/评测器使用。

## Stage 3 到 Sim 的边界

### 正式 Stage 3 场景的 exact 烟测

`scripts/prepare_all_sim_scenes.py` 从 `data/processed/<robot>/stage3/<task>/mujoco_v8/scene.xml` 派生空 mount 场景供 `robot_control` 加载；原始 XML 保持不变。随后 `scripts/run_stage3_codex_batch.py` 让 Codex 为每场编写 `policy.py`，在 SIM 中运行，保存帧、视频与报告。先按 [本地安装说明](../../docs/LOCAL_INFERENCE.md) 安装 `local`、`sim`、`agent` 依赖并放置 SAM3 权重，默认使用本地 CUDA，无需服务地址：

```bash
uv run --no-sync python scripts/prepare_all_sim_scenes.py
uv run --no-sync python scripts/run_stage3_codex_batch.py --effort medium
```

新接口仅提供 `state()`（末端、腕相机世界位姿和夹爪宽度）、`observe(queries=None)`（腕部 RGB-D、固定 `stage3_camera_0` 第三人称 RGB、SAM3 命名物体位置估计和 JPEG 图片路径/内嵌 base64）、`robot.move_pose()`、`robot.gripper()`、`robot.move_trajectory(waypoints)`、`reset()`、`return_control(reason)` 与 `stop()`。轨迹接收少量位姿点和可选夹爪宽度，内部插值；夹爪仅在双指触碰同一个自由物体时建立仿真抓取约束。`reset()` 恢复初始状态并开始有限次新尝试，`return_control()` 结束当前策略段但不重置场景，`stop()` 终止当前尝试；三者不等价。任务规划、抓放分解、重试及成功判断交由策略程序；不提供 `pick_place`、`grasp`、`set_joint` 或成功 oracle。`observe` 的物体位置是 SAM3 深度估计，不能当仿真真值。生成策略时 Codex 会看到初始腕部 RGB 和第三人称 RGB；运行时每次 `observe()` 都返回两张图。

Stage 3 当前不会把 Piper 的夹爪几何先验写入 Agent 提示词；策略必须通过公开状态、视觉观测和实际执行反馈建立抓取判断。Stage 3 作者会话沿用 real2sim 的 SDK `ephemeral=True`，不会保留为可从 CLI/UI 恢复的原生线程；完整 JSONL 审计、每关的 `policy.py` 和 `experience.md` 仍保存在本次 `runs/` 目录。同一 run 内的练习、修订和显式执行请求共用当前 stage 的临时会话；下一 stage 创建新会话，但读取上一 stage 的策略和经验文档。

独立练习 CLI 在同一个常驻场景进程里接收命令（以下以急停开关为例）；返回 JSON，图片经绝对路径提供，CLI 不输出大段 base64。它可供人或外部 Agent 在写策略前练习。批量生成时会另启动练习场景的 CLI socket；SDK 作者 thread 保持只读沙箱，不直接连接 socket，而是输出白名单 `practice` 命令，由 harness 执行、回传 JSON 和两张图片，再在同一 thread 中继续，最后输出 `policy.py`。练习次数只受整体时间预算限制；`practice_actions.jsonl` 留存记录。练习与正式评测使用两份场景实例：

```bash
uv run --no-sync python scripts/stage3_cli.py --robot piper --task put_emergency_stop_button_in_green_storage_box --listen /tmp/stage3_sim.sock
uv run --no-sync python scripts/stage3_cli.py --connect /tmp/stage3_sim.sock --command 'observe {"Stage2_EStopHousing":"red emergency stop button"}'
uv run --no-sync python scripts/stage3_cli.py --connect /tmp/stage3_sim.sock --command reset
uv run --no-sync python scripts/stage3_cli.py --connect /tmp/stage3_sim.sock --command quit
```

输出位于忽略的 `runs/sim_processed_stage3_all/<robot>/<task>/codex_rgbd/`，与旧 `codex_exact/` 隔离。Agent 显式请求 `mode="execute", command="run_policy"` 后才运行当前策略；失败后可以再次执行、修改策略、reset 或返回 checkpoint。`--total-timeout` 是唯一的整体运行预算（非抢占式，正在执行的远程感知请求可能超过截止时刻）。程序返回 `completed` 或调用 `stop` 仅表示策略声称完成；视频仍需人工复核，复核结果写入当前运行目录的 `video_review.json` 后重新运行批量脚本以汇总。

`SimEpisode` 接收 Stage 3 的 `scene.xml` 路径和一个场景适配器。适配器负责加载或复制 MuJoCo runtime instance；episode 层只调用 `reset_nominal()`、`set_object_pose()`、`set_camera_pose()` 和 `move_pose()`，不直接修改 Stage 3 源文件。

每次 `reset()` 的顺序固定为：恢复名义场景、采样物体位姿、采样相机物理/估计位姿、把物理位姿写入场景。Agent 后续只能从观测适配器获得公开传感器数据；控制误差在 `move_pose()` 执行边界注入，真实随机化和控制记录写入 episode manifest。

## Piper tabletop + Codex 垂直切片

`scene_adapter.py` 在 SIM 内封装 robot_control 的 MuJoCo runtime；`observation.py` 使用 Stage 1 同一套 SAM3 后端配置（默认本地，支持显式远端）和逐实例 mask/score 接口，用腕部 RGB-D 反投影给出估计位置。对于碗提示词使用 `<颜色> container`，同色多实例在接近最高分的 mask 中按物体尺寸选择；对部分被夹爪遮挡的方块使用视野侧相关的掩码点云分位数定位。`harness.py` 向程序提供 `observe()`、`robot.survey()`、`robot.pick_place()`、`robot.move_pose()`、`robot.gripper()` 和公开视觉检查；环境在策略进程之外独立计分。相机物理扰动写入挂载位姿，估计标定仅进入反投影；控制扰动仅在动作边界注入。

可以使用仓库已提交的基础场景 `tests/fixtures/stage3/code_as_policy_tabletop.xml`，或准备兼容 Piper tabletop 的 XML。该 fixture 用于开发和回归，不是论文中的完整重建场景。按本地安装说明准备仓库内环境与 SAM3 权重；仅显式 `SAM3_BACKEND=remote` 时才配置 `SAM3_HOST` / `SAM3_PORT`。

SIM 的 Codex 调用使用 `openai-codex` Python SDK：生成和失败后的策略修订使用一个持久化作者 thread，默认调用方式与 `real2sim/stage2`、`stage3` 相同，即直接使用 `Codex()` 的已认证默认 home；这避免了未认证的运行目录导致 websocket 超时。每个 episode 只有需要 Agent 判断时才从作者 thread 原生 fork 临时验证 thread；验证对话不回写作者 thread，只有失败报告和修订请求进入作者 thread。`authoring_conversation.jsonl` 与结果目录仍保存在本地，是审计副本；原生 thread 由 Codex 默认 home 持久化。若确实要使用项目私有 home，必须先在该 home 单独完成 `codex login`，再用 `--codex-home` 指定。没有嵌套 `codex exec` CLI 或当前 UI 对话的续接。运行下面命令前，须在所用 `robot_control` Python 环境中安装本仓库的 `.[agent,sim]` 依赖（包括 `openai-codex`）。thread 使用只读 sandbox 和拒绝权限请求。

```bash
uv sync --locked --python 3.11 --extra local --extra sim --extra agent
```

```bash
uv run --no-sync python scripts/run_sim_codex_probe.py \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml \
  --source 'blue block' --target 'yellow bowl' --output runs/codex_probe/example
```

独立 home 不继承全局 CLI 的登录状态。首次运行前，先在项目本地的**同一个**目录单独登录（不会修改全局 `~/.codex`）：

```bash
CODEX_HOME="$PWD/runs/codex_probe/example/codex_home" codex login
CODEX_HOME="$PWD/runs/codex_probe/example/codex_home" codex login status
```

预先创建 `codex_home/` 不会触发脚本的旧评测保护；请确保目录私有且不提交。默认每个 `--output` 独立；要多次运行共用一次项目本地登录，可先执行 `CODEX_HOME="$PWD/runs/sim_codex_home" codex login`，然后每次运行都加 `--codex-home "$PWD/runs/sim_codex_home"`。交接文件保存实际使用的 home 与 thread ID。显式指定全局 `~/.codex` 会被拒绝。

无需登录 Codex 即可复现当前 exact / randomized 参数下的腕相机图像和末端到达误差对比（需在仿真环境另装 `matplotlib`）：

```bash
uv run --no-sync python scripts/visualize_sim_profiles.py \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml --seed 202 --output runs/sim_profiles
```

生成的 `wrist_image_difference.png`、`control_arrival_difference.png` 和 `report.json` 保存在已忽略的 `runs/sim_profiles/`；此脚本仅比较图像与控制误差，不代表 Codex 已生成或验收了任务策略。

脚本先给 Codex 完整工具和坐标/安全范围提示，让其输出一个策略 bundle：根目录必须有 `policy.py`，还可以有 `steps/*.py` 子策略。`policy.py` 是主入口，子策略定义 `run(api, state)`；harness 锁定整个 bundle，分别在 `exact` 和随机种子上运行。`exact` 不注入扰动，通过相同的公开 `observe()` 接口返回任务物体的准确坐标；`randomized` 通过 RGB-D + SAM3 返回估计坐标。每个子策略内部可以任意多次调用 `api.observe()`，例如移动、观察、再移动、再观察；这些正常观测不暂停程序，也不必与 Codex 对话。

最终执行仍由 Agent 驱动：Agent 生成或修订 `policy.py` 后，必须显式返回 `mode="execute", command="run_policy"`，SIM 才提供受时间预算约束的执行器和公开结果，不绕过 Agent 直接替它做决策。Agent 可以反复运行同一程序、调用 `reset()`，或让程序返回 `checkpoint`/调用 `return_control()` 把中间状态和图像交回外层 Agent；外层 Agent 再决定继续运行、重置、修改策略或停止。一次稳定的 `policy.py` 可以直接完成任务，不稳定的程序则可以多轮尝试。运行日志、经验文档和视觉证据保留用于后续阶段及 REAL 验证。

默认脚本还会在失败时进入外层 `policy refinement loop`：harness 结束当前 episode 并创建新的 runtime 实例，相当于一键 reset；随后把不含真值坐标的结构化失败报告发回当前阶段的 Codex 会话，由 Codex 生成新 bundle、再次请求执行，或返回 checkpoint。修订、执行和 reset 都只受 `--total-timeout` 控制，没有固定的执行次数上限。exact 通过后才运行 randomized 开发测试；三关都通过后 bundle 才能进入后续部署验证。

2026-09-24 的同场景 reset/refinement 实测与 Code as Policies 对比见 `docs/SIM_REFINEMENT_EXPERIMENT.md`：机制已运行，但尚未证明随机化成功率提升；感知误检是当前瓶颈。

每个 episode 保存工具调用、决策日志、随机化 manifest 与真值评测；RGB 帧仅在 randomized 感知时保存。外层流程另外保存 `policy/iteration_NNN/bundle/`、`refinement/iteration_NNN_failure.json` 和 `refinement.json`；作者 thread 的审计副本保存在 `authoring_conversation.jsonl`，原生可恢复会话保存在 `codex_home/`，线程 ID 和 home 路径另存在 `authoring_thread.json`，临时验证分支的日志单独放在 episode 下。最终 `accepted=true` 时将经过验证的主程序与子策略复制到 `final_policy/`，并写入 `real_handoff.json`，包含 `authoring_thread_id`、`codex_home`、`model`、程序和审计日志的绝对路径。后续 REAL 进程可使用相同 home：`Codex(CodexConfig(env={"CODEX_HOME": handoff["codex_home"]})).thread_resume(handoff["authoring_thread_id"])`；`authoring_conversation.jsonl` 只是审计副本，不能代替原生会话。输出推荐放在仓库已忽略的 `runs/` 下；不要提交 `codex_home/`，也不要把其中的凭证加入评测产物。本地保存会话不表示模型推理离线。最终验收需要开发 profile 与 held-out 评测都通过。可用 `--bundle <dir>` 指定含 `policy.py` 的现成目录、`--program <path>` 包装一个现成文件、`--seed <int>` 指定多颗随机种子，或 `--max-iterations 0 --no-refinement` 关闭修订。

当前随机化上界：所有 tabletop 物体 XY ±5 mm / Z 轴 ±3°，腕相机安装 ±2 mm / ±1°，标定 ±3 mm / ±0.5°，位姿控制 ±3 mm / ±1°；每次 episode 均有单独的 seed 和不可变源场景 hash。上述值仅供原型验证，需要从真实误差统计校准后再发布正式 benchmark。

**当前仅为可验证的实验切片**：准确感知目前仅提供任务物体坐标，不提供全部场景真值；视角巡检点目前是 Piper tabletop 标定点，随机化范围保守。子进程 + AST 检查、只读 Codex SDK thread 用于原型，不能当作强资源隔离的安全沙箱。外层修订会替换候选 bundle，但每个候选和失败报告都保留；锁定后的 held-out 评测不会触发改码。Stage3 任意场景注册、资源隔离和正式留出 benchmark 尚未实现；不应把少量通过样本表述为通用任务成功率。下方的 2026-09-24 烟测数据来自改动前的名义（SAM3）模式，不能当作新的 exact 结果。

2026-09-24 本地烟测：参考模板 `blue block → yellow bowl`、参考 `seen_1` 种子 `20260924` 生成的 `yellow block → red bowl`，各由 Codex 生成一份程序。两任务各 1 次名义 episode 和 10 次独立随机化 episode 均通过；对应名义落点偏差分别 4.0 mm、6.5 mm，随机化平均分别 7.7 mm、6.8 mm，使用的判定阈值是 35 mm。这只说明这两个具体任务及种子上的初步可行性，不能外推其他场景、长尾误差或真实机器人。
