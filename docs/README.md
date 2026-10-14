# Documentation

本目录记录项目的架构、数据契约、运行流程和验收规则。Stage 1/2/3 与 Sim 已有实现和实验入口；各阶段的实现情况与验收证据应分别查阅对应文档。Sim2Real 的完整闭环仍在规划中。

## 文档索引

- `WORKFLOW.md`：从源码 checkout 到 Stage 1/2/3 和 SIM 的运行步骤、输入与交付检查。
- `DATA_CONTRACT.md`：像素网格、坐标系、深度单位与阶段 manifest 契约。
- `PAGES.md`：静态 GitHub Pages 项目页和预览方式。
- `ASSET_SOURCES.md`：论文图、预印本 PDF 和学校标识的来源与归属。
- `ASSET_DISTRIBUTION.md`：Hugging Face 资产与派生文件发布布局、版本和迁移状态。
- `PROJECT_STRUCTURE.md`：项目目录、模块职责、数据流和远端/本地推理切换方式。
- `ARCHITECTURE.md`：端到端组件边界、跨模块契约、门控和实施顺序。
- `ROBOT_CONTROL_INTEGRATION.md`：统一机器人控制仓库的职责和接入方式。
- `STAGE1_FRANKA_DELIVERY.md`：Franka 从命名视频到米制交付和四相机 `.blend` 的运行与保留清单。
- `STAGE1_PIPER_DELIVERY.md`：Piper 新姿态 `[0°, 30°, −45°, 0°, 60°, 0°]` 的重采视频和同构 Stage 1 交付流程。
- `STAGE1_RESULTS.md`：Franka 与 Piper 共 18 个任务的最终配准 loss、尺度、掩码和深度残差汇总与解释。
- `SIM_ARCHITECTURE.md`：仿真策略、执行闭环、随机化与独立验收。
- `SIM_REFINEMENT_EXPERIMENT.md`：策略修订实验的设计和记录。
- `OPEN_SOURCE_PREPARATION.md`：开源前审计结果、整理顺序与发布条件。
- `../CITATION.cff`：机器可读的论文与软件引用元数据。
- `FRESH_SOURCE_VALIDATION.md`：匿名公开源码、新虚拟环境的安装与离线测试结果，以及未验证范围。
- `LOCAL_INFERENCE.md`：本地 CUDA 环境、源码版本、权重获取和最小示例。
- `THIRD_PARTY.md`：第三方代码、模型权重和派生资产的来源与许可。
- `provenance/`：机器人网格来源比对、完整源码归档核验，以及 macOS / Linux 无 GPU 安装和测试机器记录。
- `notices/`：已匹配原始 D435 网格的官方 RealSense LICENSE / NOTICE，以及声明适用范围。
- `../scripts/README.md`：现有脚本的用途和运行边界。
- `../tests/README.md`：测试依赖、运行命令和验证范围。
- `../README.md`：项目目标、三个 Real2Sim Stage、Sim 与 Sim2Real 闭环。

后续建议增加：

- `PERCEPTION_BACKENDS.md`：VGGT-Ω 与 SAM3 的远端服务协议、本地推理协议和统一输出格式。
- `SIM_CONTRACT.md`：Sim/Real 的 URDF、相机观测、随机化和任务接口。
- `VALIDATION.md`：深度 MAE、Lab ΔE76、配准、物理常识和真机闭环验收方法。
