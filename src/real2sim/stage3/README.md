# Stage 3：Blender 场景编译与物理调优

Stage 3 首先校验 Stage 2 已验收的 `final.blend` 哈希，再编译 MuJoCo MJCF。
支持 Stage 2 的 `Reference_00..03` 四相机（导出为 `stage3_camera_0..3`），
及 Piper / Franka 两种机械臂。Blender 机械臂网格不导出，只提取基座位姿；
Piper 使用外部 Xacro 挂到 `piper_mount`，Franka 使用外部 FR3 MJCF 挂到
`fr3_mount`。Stage 2 原文件不会被修改。

其余可见 mesh 导出到 `assets/meshes/task/`。在 Stage 2 交付入口，
独立 Codex 会话先看 Plan、物体清单和四张 RGB，判断哪些是仅视觉的墙面/
背景、哪些是需要碰撞的静态支持面、哪些是可移动物体。装饰件可以附着到
主物体，随其运动但不产生独立关节和碰撞。分类是可审阅的物理假设，
不是质量或摩擦的实测值；Blend 中明确的 `real2sim_*` 属性仍优先。

## 命令

推荐直接使用 Stage 2 已通过的交付目录：

```bash
PYTHONPATH=src python -m real2sim.stage3.mujoco_export \
  --stage2 data/processed/piper/stage2/put_emergency_stop_button_in_green_storage_box \
  --output runs/stage3/workspace/piper/put_emergency_stop_button_in_green_storage_box/mujoco
```

`--stage2` 也可指向 `geometry/stage2_manifest.json`；要求已通过验收，
Plan 与四张评估 RGB 可读，输出目录为空。模型使用 `gpt-6-astra`、`high`
以及只读临时 Codex 会话。分类结果和审计材料保存在 `semantics.json`、
`semantics_prompt.md`、`semantics_inventory.json`、`semantics_session.json`。
需要人工修订时，把已审阅的分类 JSON 用 `--semantics` 传给**新**输出目录；
程序仍会验证源 Blend SHA256 和所有物体的分类。没有 SDK/模型时不会假装
已完成模型分类。原有单独 `--blend` 入口仍保留名称推断，不调用分类模型。

批量处理所有已验收的 Piper 和 Franka Stage 2 场景（顺序执行、逐场景记录失败）：

```bash
real2sim-stage3-batch
# 或：PYTHONPATH=src python -m real2sim.stage3.batch --robot piper --task close_cup_lid
```

可加 `--export-only` 只做分类与导出。每个任务的 MuJoCo XML 存在
`runs/stage3/workspace/<robot>/<task>/mujoco/scene.xml`（已有旧版时为当前导出器版本目录，例如 `mujoco_v8/scene.xml`），Plan 与任务描述存于
`task_context.md`，未经真实测量的物理先验存于 `hypotheses.json`；物理 Agent
的独立试验存在 `physics/<run_id>/`。本轮批量进度和每任务错误写在
`runs/stage3/batch_<run_id>/batch_manifest.json`。已完成且源文件哈希仍匹配的
导出可复用；旧版场景如已存在，将新场景写入当前导出器版本目录（目前为
`mujoco_v8/`）而不覆盖历史记录。
若已确认交付到 `data/processed`，可以使用
`--runs-root data/processed --reuse-semantics-from runs/stage3/workspace`；
结果写入 `data/processed/<robot>/stage3/<task>/mujoco/`，批次清单写入
`data/processed/stage3/batch_<run_id>/`。仅复用与 Stage 2 Blend 哈希完全一致的
历史语义分类；无法匹配的任务重新分类。所有任务碰撞体放在 MuJoCo 的
`geom group=3`，默认渲染隐藏，但碰撞和质量仍然有效；可视网格保留在 group 0。
物理 Agent 若通过独立复验，批次清单的 `final_scene` 指向该任务在 `data` 中的
最终 XML；可能是初始 `mujoco_v8/scene.xml`，也可能是 `physics/<run_id>/` 下的
修复候选，不能只按 `mujoco_v8/scene.xml` 推断最终结果。本批交付的每个任务
还包含 `final_manifest.json` 与 `final_views/camera_0_rgb.png` 等四张默认可见性图。
未完成或过期的目录不会自动删除或覆盖。墙面、帘子等无关背景
仅视觉；工作台、托盘底部与可接触容器壁保留碰撞。
导出器不会添加与真实桌面位置不符的全局 `z=0` 碰撞地面；必要的支撑
由任务场景中的桌面、托盘和容器几何提供。场景外侧使用 MuJoCo 的蓝色渐变
天空与内置棋盘纹理背景平面；平面不参与接触或碰撞，不是额外的物理支撑面。

```bash
PYTHONPATH=src python -m real2sim.stage3.mujoco_export \
  --blend path/to/scene_02_blockout.blend \
  --output path/to/stage3/mujoco
```

安装项目后也可以使用：

```bash
real2sim-stage3-export \
  --blend path/to/scene_02_blockout.blend \
  --output path/to/stage3/mujoco
```

首次使用可安装 MuJoCo 依赖：

```bash
python -m pip install -e ".[sim,agent]"
```

输出的 `scene.xml` 可以直接用 `mujoco.MjModel.from_xml_path()` 加载。
此外会生成 `object_manifest.json`、`camera_manifest.json` 和
`export_manifest.json`，用于后续物理调优和视觉回归。导出的机器人包含
`stage2_initial` 初始 keyframe：Piper 使用 Stage 2/CTRL 注册姿态
（`0°, 30°, -45°, 0°, 60°, 0°`），不会回到外部模型的默认零姿态。
关节初值只写入 `qpos` / `ctrl`，不写 MJCF 的 `joint ref`（它会改变关节零点）；
导出时检查 Piper 关键链节的 MuJoCo 世界坐标与 Stage 2 注册 URDF 的正运动学一致。

当前编译策略是：

- Blender 中的机械臂只用于提供挂载点位姿；Piper/FR3 使用外部机器人模型；
- 除机械臂和四相机外的可见 mesh 保留为视觉资源；背景网格可完全不参与碰撞；
- 共面视觉网格导出时添加极小厚度，以避免 MuJoCo 的平面凸包错误；
- `real2sim_role`、`real2sim_joint_type`、`real2sim_collision`、`real2sim_mass_kg`
  等对象属性优先，缺少属性时使用名称回退并写入推断记录；
- 四个相机使用 Blender 的世界位姿和像素内参写入 MuJoCo `focalpixel`、
  `principalpixel`、`resolution` 参数。

Stage 3 预览默认使用 Stage 2 的四个参考相机，而不是自由相机：

```bash
PYTHONPATH=src mjpython -m real2sim.stage3.render_overview \
  --scene runs/stage3/workspace/piper/<task>/mujoco_v8/scene.xml \
  --output runs/stage3/workspace/piper/<task>/mujoco_v8/reference_preview \
  --reference-views
```

这会生成 `camera_0_rgb.png` 至 `camera_3_rgb.png`，对应 Stage 2 的
`Reference_00` 至 `Reference_03`；自由第三人称图只用于 Agent 观察，不作为
Stage 2/Stage 3 视觉对照图。

## Stage 3 流程

1. **编译**：以 Stage 2 已验收的目录为输入运行 `stage3-export`，得到可加载的
   `scene.xml`、任务物体/相机清单和 mesh；机器人使用外部模型替换，四个
   Blender 相机保留为 MJCF 相机。
2. **观测**：给定上述 XML 和有来源的物理假设，`stage3-agent` 查询任务物体
   位姿、质量、摩擦，并把临时第三人称照片交给独立 Codex 会话。照片在会话
   消费完成后立即删除，只保留结构化观测和检查报告。
3. **探测与修复**：有自由物体时，Agent 先只凭第三人称图、任务和 Plan
   估测质量、摩擦和每侧夹力的宽松区间，再做静置检查和双夹块试验；1 N
   失败本身不触发调参。只有排除接触几何问题且证据充分时，才分别试质量和
   摩擦候选，并在相同力档复测；原始 `scene.xml` 不覆盖。
4. **验收与交付**：本地验证 Agent 的命令轨迹、场景哈希和报告，并重算静置/
   双夹块测试结果。仅凭仿真时状态为 `provisional`，不能视为真实物理标定；确认后
   再把可用 XML 交给后续任务仿真。

## 独立 Codex 物理审查 Agent

安装 `python -m pip install -e '.[sim,agent]'`，在 macOS 上确保同一 Python 环境
有 `mjpython`（第三人称离屏渲染需要）。先给定只读假设 JSON，区分推测和实测；
模板见 `tests/fixtures/stage3/code_as_policy_hypotheses.json`。真实世界测量必须附
可比的量、单位及实验条件，否则不应称为真实标定。

```bash
real2sim-stage3-agent \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml \
  --hypotheses tests/fixtures/stage3/code_as_policy_hypotheses.json \
  --output runs/stage3/my_agent_run
```

程序通过 Python `openai-codex` SDK 启动**独立临时 thread**。有自由物体时先只看
第三人称照片、任务与 Stage 2 Plan，保存 `visual_force_prior.json`，该轮不读取
仿真参数；随后再把物体状态和固定的 `agent_prompt.md` 交给它。运行器保存
`prompt.md`、`observe/observation.json`、`session.json`（含 thread ID、每个命令及
退出码、结构化回答和本地二次校验）。物理检查未通过时，同一临时会话最多再
尝试两轮，逐轮记录结果；使用 full-access 沙箱以允许 macOS 离屏渲染，但只允许
在独立输出目录创建候选场景，不覆盖 Stage 2 或原始 XML。导出器会解包 Blender
内嵌图片、保留 UV 及多材质对应的可视网格；MuJoCo 光照与 Blender 不保证逐像素一致。
`overview.png` 仅在 SDK 会话消费图片期间存在，成功或失败都会删除；
验收还会在本地重跑静置/双夹块测试，比对数值；若外层 shell 失败但探针
报告有效，会单独记录警告，不会将 shell 退出码误当作物理失败。
输出目录必须为空；原 XML 不应被覆盖，只有证据充分时才生成候选 XML。
夹块采用滑动关节、侧向力驱动和成对接触摩擦，不包含真实 Piper 夹爪的
几何、控制与软指接触；`--normal-forces` 是每侧驱动力目标，实际接触力需看报告，
并非实测机械爪夹持力。该测试仍只是仿真 sanity check；
当前没有独立真实测量比较器，因此结论最多只能是 `provisional`。若 SDK 身份验证或沙箱不可用，
错误及已有 thread ID 会写到 `session.json`，不伪造运行成功。

双夹块夹取测试可单独运行：

```bash
real2sim-stage3-physics grasp \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml \
  --body blue_block --normal-forces 0.2 0.4 0.6 \
  --jaw-friction 1.0 --seconds 0.4 --output runs/stage3/grasp_blue
```

程序在临时场景中给物体两侧添加带摩擦的方块，以侧向电机力夹紧、再用位置
驱动向上移动夹块。每档数值是**每侧电机驱动力**，而非精确恒定接触力；报告
另列实际峰值接触力、双侧接触保持率、夹块相对滑移、末端抬升量和最终双侧接触。
只有持续接触并抬升才算夹住。`--jaw-friction`
规定夹块与目标物体的成对接触摩擦。用于判断模拟夹持物理是否合理，不能
直接等同真实夹爪的力、柔顺性或成功率。定时力/力矩仍由下文 `run` 单独完成。

## 物理探针 CLI

`real2sim-stage3-physics` 面向已经生成的 MJCF 场景，专门用于让 Codex 或
其他自动化程序做可重复的物理试验。它不修改原场景：`run` 只读取场景并
输出报告，`patch` 将参数写到一个新的 XML 文件。

### Codex 最短流程

无需手写动作 JSON 或参数补丁 JSON。先列出任务物体，再对其中一个物体
试调质量或滑动摩擦；按上文先安装 `.[sim]`，`try` 自动用相同探针分别测试
原场景和候选场景：

```bash
real2sim-stage3-physics objects --scene tests/fixtures/stage3/code_as_policy_tabletop.xml
real2sim-stage3-physics try \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml \
  --body blue_block --mass-kg 0.05 \
  --output runs/stage3/probes/blue_block_trial
```

`try` 会输出**单份 JSON**，同时生成 `report.json` 和 `candidate.xml`；
`--friction 0.45` 可试调滑动摩擦，`--damping` 仅适用于有直接关节的物体。
可通过 `--force-n`、`--torque-nm`、`--duration` 调整标准扰动。
报告对比施力与无施力基线之差，`passed` 只表示数值稳定且有可测响应，
**不代表与真实物理参数吻合**。复杂的定时动作仍使用下文的 `run`。

可以直接给 Codex 这样的提示词：

> 在 `tests/fixtures/stage3/code_as_policy_tabletop.xml` 里找到 `blue_block`，
> 只试着把它的质量从 0.04 kg 改到 0.05 kg；用同样的力对比改动前后的反应，
> 把报告放进 `runs/stage3/probes/codex_first_try`。不要覆盖原场景，
> 也不要把这次试验当作已完成的物理标定。

### 检查场景

```bash
real2sim-stage3-physics inspect \
  --scene /path/to/stage3/mujoco/scene.xml
```

`tests/fixtures/stage3/code_as_policy_tabletop.xml` 原样复制了
`code_as_policy/scenes/piper_tabletop.xml`，用于独立回归。`--attach-piper auto`
（默认）也支持这个场景模板。该模板
只有空的 `piper_mount`，CLI 会复用 `external/robot_control` 的 Piper 组装
逻辑后再枚举物体；直接用 `mujoco.MjModel.from_xml_path()` 读取该模板会因
抓取 weld 所引用的 `gripper_base` 尚未存在而失败。

### 时间动作与反应检查

动作写成 JSON，支持 `force`、`torque`、`wrench`、`set_position`、
`set_linear_velocity`、`set_angular_velocity`、`set_rotation` 和相对
`rotate`。连续动作使用 `start` + `duration`（或 `end`），力和力矩默认
在世界坐标系，也可以设置 `"frame": "body"`。

```json
{
  "duration": 0.6,
  "settle": 0.1,
  "actions": [
    {"body": "blue_block", "type": "force", "start": 0.1,
     "duration": 0.2, "vector": [0.8, 0, 0]},
    {"body": "red_block", "type": "rotate", "at": 0.1,
     "axis_angle": [0, 0, 1, 0.5]}
  ],
  "checks": [
    {"body": "blue_block", "type": "move", "min_m": 0.001, "max_m": 1.0},
    {"body": "red_block", "type": "rotation", "min_rad": 0.1, "max_rad": 1.0}
  ]
}
```

运行后 stdout 是机器可读 JSON；传 `--output runs/stage3/probes/case_01`
还会保存 `report.json` 和 `trajectory.json`。报告包含每个 body 的位移、
旋转、末速度、最大施力/力矩和检查是否通过；检查失败时 CLI 返回码为 `2`。

```bash
real2sim-stage3-physics run \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml \
  --script /path/to/probe.json \
  --output runs/stage3/probes/code_as_policy
```

如果暂时没有明确动作脚本，可以让 CLI 对场景中每个任务侧的自由物体、
滑轨和铰链逐个复位后施加一组小力/小力矩（机械臂本体排除）：

```bash
real2sim-stage3-physics auto \
  --scene tests/fixtures/stage3/code_as_policy_tabletop.xml \
  --output runs/stage3/probes/code_as_policy_auto
```

`auto` 会把固定 body 列入 `skipped_fixed_bodies`，把 Piper 本体列入
`skipped_robot_bodies`，对每个可动物体报告
`responded`、相对于未施力基线的位移/旋转、末速度和数值稳定性。它是物理 sanity check，不是
任务成功率评测；需要验证具体动作时仍应使用 `run` 的显式脚本和 `checks`。

### 更新质量和摩擦

更新使用单独的 patch JSON，质量会写成 body 的显式惯量，摩擦会写到该
body 的所有直接 geom；标量摩擦只改滑动摩擦，保留每个 geom 的扭转/滚动摩擦，
三元数组可同时指定三者。原 XML 保留不变。这样 Codex 可以先看 `report.json`
再决定下一轮参数，而不是让一次不稳定试验隐式覆盖场景。
没有实测物理目标时，不应把通过 sanity check 的质量或摩擦值称为已标定。
`tests/fixtures/stage3/code_as_policy_trial_patch.json` 是验证 CLI 功能的
候选数值，不作为真实物体参数的最终估计。

`patch` 也接受 `position: [x,y,z]` 与 `quaternion: [w,x,y,z]`，对自由关节物体会同步更新
`stage2_initial` 关键帧的初始位姿，使静置和夹取探针真正从候选位置开始。只有原位置与
Stage 2 四视角/任务目标不符时才应改位置；若是碰撞代理形状错误，应修复碰撞体，不要
为了消除穿透而移动视觉上正确的物体。

`patch` 另支持 `center_of_mass: [x,y,z]`（body 局部坐标，显式惯量保留；可与 `mass_kg` 联合指定）、
`collision: {"geom":"<name>_collision","mode":"none"}`（仅静态场景背景；保留 visual geom）和
`collision: {"geom":"<name>_collision","mode":"box","pos":[x,y,z],"size":[hx,hy,hz]}`
（替换该物体的碰撞代理，不改变可视网格）。不要删除桌面、容器支撑或动态物体的碰撞。
修复碰撞体、重心和质量时应逐项对照原始静置／夹取结果；通过仿真不代表真实纸巾柔性或抽取已经验证。

```json
{
  "blue_block": {"mass_kg": 0.05, "friction": [0.45, 0.005, 0.0001]},
  "yellow_bowl": {"friction": 0.8}
}
```

```bash
real2sim-stage3-physics patch \
  --scene /path/to/scene.xml \
  --patch-file /path/to/patch.json \
  --output /path/to/scene_tuned.xml
```
