# Project Structure

公开运行顺序见 [WORKFLOW.md](WORKFLOW.md)，阶段文件／坐标／状态契约见
[DATA_CONTRACT.md](DATA_CONTRACT.md)。下述目录描述当前实现，不意味着论文
完整实验或真机闭环已可从零复现。

## 设计原则

本项目将 `Real2Sim`、`Sim` 和 `Sim2Real` 分开组织，但共享统一的场景、感知和任务数据契约。VGGT-Ω 与 SAM3 都支持两种后端：

1. **本地推理**：默认方式，首个完整推理环境为 Linux、NVIDIA CUDA、Python 3.11。推理代码放在 `src/perception/`，固定版本的上游源码放在 `external/`，权重放在 `model/`，权重不提交到 Git。
2. **远端调用**：显式选择的可选方式。服务地址、端口和协议从环境变量或本机 `.env` 读取。两个模型可以分别选择后端，失败不会自动切换。

远端和本地后端必须输出同一份结构化结果，使上层 Stage 不需要知道当前使用哪种后端。

## 目录结构

```text
agentic_rsr/
├── README.md                       # 项目目标、三个 Real2Sim Stage、Sim/Real 闭环
├── .env.example                    # 环境变量模板，不含真实地址
├── .gitignore                      # 忽略本机数据和模型文件
├── .gitmodules                     # 机器人与本地模型的 HTTPS 子模块地址
├── pyproject.toml                  # editable 安装、阶段 CLI 与测试配置
├── uv.lock                         # 固定依赖解析结果
├── CONTRIBUTING.md                 # 开发、离线检查与贡献约定
├── docs/
│   ├── README.md                   # 文档索引
│   ├── PROJECT_STRUCTURE.md        # 当前文档
│   ├── ARCHITECTURE.md             # 当前架构、规划边界和验证限制
│   ├── WORKFLOW.md                 # 从视频到场景到策略的公开命令
│   ├── DATA_CONTRACT.md            # 输出布局、坐标／单位和验收字段
│   ├── index.html                 # GitHub Pages 静态项目页面
│   ├── site.css / site.js         # 页面样式与媒体交互
│   ├── assets/                    # 论文图、论文 PDF 和学校 logo
│   └── ROBOT_CONTROL_INTEGRATION.md # 子模块及控制接口
├── src/
│   ├── perception/
│   │   ├── vggt_omega/            # 远端/本地 VGGT-Ω 推理
│   │   └── sam3/                  # 远端/本地 SAM3 推理
│   ├── real2sim/
│   │   ├── stage1/                # 配准与初版场景
│   │   ├── stage2/                # Codex 建模与视觉验收
│   │   └── stage3/                # 物理调优与回归
│   ├── sim/                        # 随机化、任务程序与评测
│   ├── sim2real/                   # 预留：真机闭环、repair 和 memory
│   └── robot_control_adapter/      # 预留：统一机器人与相机适配层
├── external/robot_control/         # 必选 Git submodule，固定控制仓库版本
├── external/vggt_omega/            # 官方 VGGT-Ω 固定版本子模块
├── external/ultralytics/           # Ultralytics SAM3 固定版本子模块
├── configs/
│   ├── robots/                    # 机器人配置预留目录；当前只有说明
│   └── tasks/                     # 任务参数预留目录；当前只有说明
├── asset/                          # 可提交的固定姿态 Piper/FR3 点云模板及元数据
│   └── robot_surfaces/             # 每台机械臂各 40000/10000/5000 点
├── model/
│   ├── README.md                   # 模型放置和下载说明
│   ├── vggt_omega/                 # VGGT-Ω 权重、配置、缓存（不提交）
│   ├── sam3/                       # SAM3 权重、配置、缓存（不提交）
│   └── cache/                      # 模型缓存（不提交）
├── scripts/                        # 批处理、实验、数据整理、仿真与可视化脚本
├── tests/
│   ├── unit/                       # 单元测试
│   ├── contract/                   # 预留说明；后端契约用例目前在 unit/
│   └── integration/                # 小规模集成测试
├── data/
│   ├── raw/
│   │   ├── piper/                 # Piper 原始视频（不提交）
│   │   └── frank/                 # FR3 原始视频（不提交；沿用现有目录名）
│   └── processed/                 # 可复用的预处理缓存（不提交）
└── runs/                           # 每次实验的产物和日志（不提交）
```

上图展示**已建立的目录**。`src/real2sim/stage1/` 提供独立的 32 帧 VGGT-Ω/SAM3 串行批处理入口（FR3/Piper，输出到 `data/processed/`）以及机器人配准与 Blender 交付入口；推理批处理不调用配准或生成 `.blend`。`asset/robot_surfaces/` 保存刚体点云模板及来源元数据。感知客户端分别位于 `src/perception/vggt_omega/`、`src/perception/sam3/`。Stage 2 已实现 Agent 建模和四视角评估，Stage 3 已实现语义分类、MuJoCo 导出和物理调优，Sim 已实现策略执行与随机化；Sim2Real 和 `robot_control_adapter` 仍为规划边界。`.env`、模型权重和真实视频不提交 Git；运行产物默认忽略，但已有一个急停演示的策略、视频和复核记录被跟踪在 `runs/` 下。

## 感知后端切换

代码通过进程环境或本机 `.env` 中的 backend 变量选择实现：

```text
VGGT_OMEGA_BACKEND=local
SAM3_BACKEND=local
VGGT_OMEGA_DEVICE=cuda
SAM3_DEVICE=cuda
```

设置任一后端为 `remote` 后才读取对应的 `HOST`、`PORT`。local 模式读取 `VGGT_OMEGA_MODEL_DIR` 和 `SAM3_MODEL_DIR`，并按需加载和释放模型。缺少权重、依赖或指定设备时明确失败。安装与最小示例见 [LOCAL_INFERENCE.md](LOCAL_INFERENCE.md)。

### VGGT-Ω 统一输出

上层需要拿到：

- 输入帧和采样索引；
- 每帧 depth；外部米制标定缺失时仅为未经确认的 VGGT 单位，不能用作已验收的米制深度；
- 相机 intrinsics 和 extrinsics；
- 点云或可重建点云的必要中间结果；
- 推理分辨率、坐标系、单位、backend、模型版本和失败信息。

### SAM3 统一输出

上层需要拿到：

- 输入图像或帧索引；
- 目标类别/文本提示及模型版本；
- mask、置信度和图像坐标信息；
- backend、模型版本、阈值、推理分辨率和失败信息。

Stage 1 使用 VGGT-Ω 的相机/深度/点云结果和 SAM3 的机器人或物体 mask（如果启用）完成配准；后续 Stage、Sim 感知和 Sim2Real 不应直接依赖远端服务的私有返回格式。

## 配置与数据边界

- `.env` 只保存本机服务地址、端口、模型路径和运行时配置；提交 `.env.example`，不提交 `.env`。
- `configs/` 只保存可复现、可分享的实验参数；如果参数包含真实地址、token 或机器路径，应改为环境变量名。
- `model/` 只保存目录结构和说明；权重、下载缓存、编译引擎和本地 tokenizer 不提交。
- `data/` 与 `runs/` 保存运行产物，必须通过 manifest 记录来源、版本、坐标系、单位和生成时间。
- 远端和本地推理的结果都要记录 backend，避免无法区分实验结果来源。
- 机器人控制仓库保持独立，并以 `external/robot_control` Git submodule 接入；当前 SIM 在 `scene_adapter.py`／`stage3_scene.py` 直接封装其 API，`robot_control_adapter` 尚未实现。具体边界见 [ROBOT_CONTROL_INTEGRATION.md](ROBOT_CONTROL_INTEGRATION.md)。
- 安装路径是源码 checkout＋editable，不是携带完整机器人资产的独立 wheel。权重、数据和运行产物默认忽略；追踪文件的例外与第三方资产发布限制见 [THIRD_PARTY.md](THIRD_PARTY.md)。
- GitHub Actions CI 已按作者要求移除；贡献者需手动运行 [tests/README.md](../tests/README.md) 的离线检查。
