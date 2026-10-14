# 系统架构（实施基线）

本文说明当前源码的组件、边界和验证限制。Stage 1、Stage 2、Stage 3 和 SIM 已有实现；`sim2real/` 与 `robot_control_adapter/` 仍只有预留说明。已实现不等于完整论文复现或真机验证：公开命令顺序见 [WORKFLOW.md](WORKFLOW.md)，文件、坐标和验收状态见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。论文主实验使用 DSH/Claude，而当前仓库公开的是历史 Codex 路径。

## 一条主线、三种运行形态

```text
真实视频 + 任务描述 + 固定版本机器人资产
  │
  ├─ perception：VGGT-Ω / SAM3（默认本地，显式远端 TCP，同一输出契约）
  ▼
Real2Sim orchestrator
  ├─ Stage 1：相机/深度/点云 → 机械臂配准 → 程序生成四相机与机械臂的 .blend
  ├─ Stage 2：Codex 改物体/外观 → 四视角渲染 → 深度 MAE + Lab ΔE76 → 迭代
  └─ Stage 3：物理参数/有限几何修正 → MuJoCo 验证 → 回归视觉指标
  ▼
Sim：同一机器人资产 + 随机化 + 感知 → 可复用任务程序 → 多轮评测
  ▼
Real（规划）：同一任务程序切换受控的真实后端 → 成功或记录失败
  └─ 失败 → 仿真重置/repair + 保留 memory → 回归/审批 → 下一次真机尝试
```

这里的“三个 Stage”**仅属于 Real2Sim**；Sim 和 Real 是后续运行形态，不再给它们套用参考项目的 Stage 0–6 编号。每次运行应有独立 `run_id`、输入指纹、配置快照、依赖版本和阶段状态；通过状态由校验器而非 Agent 自报。

## 组件与职责

| 组件路径 | 当前职责／状态 | 不应负责 |
| --- | --- | --- |
| `src/perception/` | 远端/本地 VGGT-Ω、SAM3 的后端选择、协议转换和统一结果 | 固定业务场景、硬编码服务地址 |
| `src/real2sim/stage1/` | 抽帧、推理、尺度/坐标校验、机械臂配准、程序化建 `.blend` | 把未验证的配准候选值标记为成功 |
| `src/real2sim/stage2/` | 驱动 Codex、冻结四相机和机械臂、渲染、计算误差和版本选择 | 用模型自评替代图像/深度程序评分 |
| `src/real2sim/stage3/` | 模型导出、物理参数和有限几何调优、MuJoCo 验证、视觉回归 | 擅自修改相机和机器人资产 |
| `src/sim/` | 任务环境、受约束的随机化、单体程序/分段 Agent 和评测 | 给策略访问仿真物体真值位姿的捷径 |
| `src/sim2real/` | 预留真机门控、执行/监测、失败记录、仿真 repair 和版本化 memory | 绕过安全审批或把仿真 reset 当成真机 reset |
| `src/robot_control_adapter/` | 预留统一适配接口；当前 SIM 在 `scene_adapter.py`、`stage3_scene.py` 中复用 `robot_control` | 再实现一套机器人 URDF 或底层运动控制 |
| `external/robot_control/` | 独立 Git submodule：Piper/FR3 MuJoCo、真机运动和相机/控制底层 | 任务重建、策略或 Agent 编排 |

`external/robot_control` 的当前 Python 包是 `robot_control`。`robot_control.api.Robot` 已提供 `camera(width=..., height=...)` 返回 RGB-D 帧，也支持 `state`、`move_joints`、`move_p`、`gripper`、`stop`；`pose` 从 `state().pose` 读取。当前 SIM 的场景封装与 `Stage3RGBDObservation` 复用其相机与运动接口。上游含 FR3/Piper 后端和轨迹能力，但本仓库尚未提供已验收的端到端真机执行入口；上游能力不构成本项目硬件闭环的验证记录。

## 跨模块契约

### 机器人和观测

- 当前任务程序通过 SIM controller／scene 的公开动作和观测接口执行，场景封装复用 `robot_control.api.Robot`。统一 `RobotSession` 尚为设计目标，不是当前可导入的接口；公开实践接口见 [DATA_CONTRACT.md](DATA_CONTRACT.md)。
- 机器人 URDF、关节顺序、限位、腕部相机安装定义及版本从**同一个 submodule commit**读取。Stage 1 的机械臂可视化、Sim 的 MuJoCo 机器人和真机运动学需要核对来源，不允许悄悄混用参考仓库的另一套 Piper 资产。
- 腕部 RGB-D 带时间戳、深度单位、内参和具名坐标变换；Sim 的第三视角只提供 RGB。如果 Real 没有同等可用的第三视角，部署程序不得依赖它做必要的闭环决策。
- 策略只接收公开的传感器观测与感知估计；仿真真实位姿只允许评测器/随机化器访问，并和策略进程/接口隔离。

### 感知

- 感知 factory 统一创建 `LocalVGGT`／`RemoteVGGT` 和 `LocalSAM3`／`RemoteSAM3`。VGGT 的 `infer(frames)` 返回相机、depth、置信度等数组字典，32 帧采用 512×288 对齐网格；原始深度使用未确认的 VGGT 单位，配准求解米制尺度后另存 metric 输出，不能称为独立测量真值。
- SAM3 的 `segment(image, prompt)` 合并实例，`segment_instances(image, prompt, expected_shape=...)` 返回逐实例 mask／score；mask 恢复到原图网格。SIM 通过实例选择适配器保留既有任务观测语义。
- `src/perception/config.py` 默认选择本地后端，`.env` 或进程环境提供设备／模型目录；只有显式 `remote` 才读取远端地址。两个客户端都是**长度前缀 TCP**，具体请求/响应由适配器封装；不能把 SAM3 假定为 HTTP。缺少依赖、设备、权重时失败，不隐式下载或自动切换后端。配置与离线契约检查见 [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md)。

### 场景与评测

- Franka 与当前 Piper Stage 1 均用已知尺寸的固定姿态 URDF 配准获得米制尺度，交付以机械臂根部为世界原点与轴向的 `camera_metric.json`、`depth_metric/`、精简的 `registration.json` 和四参考相机/固定机械臂的 `initial.blend`；原始 VGGT 相机/深度保留用于复算。四视角来自同一视频；32 帧均参与拟合和评分，尚无独立验收，因此 `stage1_delivery.json` 仍标记 `candidate_unverified`、`accepted=false`。当前 phased Stage 2 可探索性消费此输入，不会把其配准状态变成已验收。交付清单和命令见 `STAGE1_FRANKA_DELIVERY.md` 和 `STAGE1_PIPER_DELIVERY.md`；旧版单视频入口的输出布局与验收逻辑不同。
- Stage 2 每轮保存新的 `.blend`、四视角渲染、逐视角误差和冻结资产检查。汇总 `D` 为四视角米制 depth MAE，`C` 为四视角平均 Lab ΔE76；排序为 `0.5×D/0.15 + 0.5×C/15`。当前 `evaluator.py` 要求 `D < 0.15 m`、`C < 15`，并且每个视角 depth MAE `< 0.20 m`、Lab `< 30`、有效渲染深度覆盖率 `>= 0.50`。缺失渲染深度在有效参考区域按 1 米误差计入。阈值以各次 report 保存值为准；参考深度是重建伪参考。
- Stage 3 导出可复制加载的 MuJoCo 场景和物理参数报告。允许任务物体有限几何修正，但变动后必须重新检验 Stage 2 的两项门槛；物理常识测试不等于真实参数标定。
- 所有阶段报告均记录输入文件、校验和、变换的方向与单位、模型/机器人版本、随机种子、指标、状态及失败原因。失败结果可以留作诊断，不能作为下一阶段的已通过输入。

## Sim 和 Real 的执行边界

Sim 先运行**完整 Python 任务程序**，通过多种物体相对摆放、限定工作空间内的可达性检查、深度/分割误差和相机外参扰动做评测；完整程序不稳定时，再增加分段 VLM/Agent 的观察—决策—执行—检查循环。两条路径都使用相同的机器人适配层和感知结果格式，不让 VLM 直接操作仿真真值或裸硬件命令。

真机首轮使用 Sim 已通过的**同一版本任务程序**（只切换后端和安全配置）。真机未通过时，先记录失败，再在仿真复现/重置到便于观察的初始位姿，用累积 memory 和真实失败信息修复程序；修复通过仿真回归后才能申请下次真机尝试。真机回到可观察安全位姿或重置物体**是独立的、需测量和确认的真实操作**，不能因为仿真已 reset 就自动发送真机归位动作。每次真机动作应有工作空间、速度、力/碰撞和急停限制，异常时停止并转人工复核。

## 后续验证顺序

1. **全新环境**：匿名 recursive checkout、锁定依赖安装、离线测试和 CLI 检查；目前完成的是已有 checkout 上的新环境与 CPU 验证。
2. **真实本地感知**：Linux CUDA 上运行 VGGT-Ω/SAM3 视频示例和本地 SAM3 SIM 观测，记录模型版本、checkpoint SHA256、耗时及峰值显存。
3. **Stage 1 独立验收**：补充未参与拟合的相机／机器人观测，验证米制尺度与坐标，对已有 `candidate_unverified` 保留正确状态。
4. **Stage 2/3 与 SIM**：使用公开输入复跑完整纵向示例，检查冻结资产、联合视觉门槛、物理报告、任务复核与锁定策略在独立场景下的表现。
5. **Real**：实现当前预留的部署接口及硬件门控，再逐次验证传感器、运动限制、失败记录和修复回归。

实现、测试通过、公开输入复现和真实硬件结果应分别记录。发布范围及剩余许可问题见 [OPEN_SOURCE_PREPARATION.md](OPEN_SOURCE_PREPARATION.md)。
