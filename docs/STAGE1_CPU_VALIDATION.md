# Stage 1 无 CUDA 验证记录

2026-10-13，在 macOS arm64 上完成 Stage 1 的 CPU 后续流程验证。
基线提交为 `7891da5a3077961ad3fd65655018e34f517783bf`，运行时包含本次验证发现的三个修复。
**真实视频抽帧、历史感知结果回放、初始化、完整配准与 Blender 交付已运行；本地 VGGT-Ω / SAM3 权重推理尚未验证。**

## 环境

- Python 3.11.15，NumPy 1.26.4，OpenCV 4.11.0.86。
- PyTorch 2.6.0，torchvision 0.21.0，CUDA 不可用。
- VGGT-Ω 0.0.1，Ultralytics 8.4.160，Blender 5.2.2 LTS。
- 两份本地模型权重均未准备；未下载权重、未调用远端感知服务。
- `VGGTOmega`、`SAM3SemanticPredictor`、`timm` 实际导入与 CLIP 文本 tokenizer 检查通过，未实例化模型。

## 完成的检查

| 检查 | 结果 |
| --- | --- |
| Stage 1 / 感知 / 几何 / 配准 / Blender 测试回归 | 132 通过，2 个真实模型测试跳过，0 失败 |
| 已安装 Stage 1 CLI 的 `--help` | 15 个入口通过 |
| 本地模型检查命令，CUDA 配置 | 明确报告缺少权重及 CUDA |
| 本地模型检查命令，CPU 配置 | 依赖及设备检查通过，只报告缺少权重 |
| 18 个真实任务的历史感知缓存 | 视频 SHA256、32 帧 RGB / mask / 深度 / 置信度 / 相机契约与 CPU 反投影通过 |
| Piper、Franka 各一段视频重新抽帧 | 每段 32 帧，RGB 与历史抽帧结果逐像素一致 |
| Piper `take_out_tissue` | 128 旋转初始化、三个完整优化分支、选优及 `.blend` 交付通过 |
| Franka `close_drawer` | 48 旋转初始化、完整单分支优化及 `.blend` 交付通过 |
| 两份交付 `.blend` 重新打开 | 根对象单位变换、机器人网格、四相机 / 四 Scene、打包 RGB 检查通过 |

Piper 初始化耗时约 47.5 秒，三个分支使用三个 CPU worker，配准总耗时约 337.6 秒，最终 loss 为 0.10083。
Franka 初始化耗时约 17.7 秒，配准约 375.4 秒，最终 loss 为 0.10624。
两台机器人配准同时运行，因此这些耗时仅用于记录此次运行，不作为性能基准。

## 验证中修复

1. 旧单视频入口不再把未标定的 VGGT 深度单位直接按米制范围检查；提供 `metric_scale` 时仍做米制合理性校验。
2. Piper Top-3 分支校验使用调用方实际传入的感知场景路径，支持独立输出目录。
3. 两台机器人的交付命令新增 `--initializations`，允许使用新初始化目录，并继续校验初始化 SHA256。

对应回归用例覆盖了未标定与已标定深度、自定义场景路径以及匹配 / 过期初始化哈希。

## 证据与限制

本机完整记录保存在独立目录 `runs/stage1_cpu_validation.i0I8HC/`，该目录被 Git 忽略：

- `report.json`：环境、各阶段结果、交付文件 SHA256 和验证边界。
- `final-tests.xml`：最终测试报告；`environment.json`：CLI 和本地模型前置条件检查。
- `cache_audit.json`：18 个缓存的独立结构检查与当前批量入口的缓存复用资格。
- `verified-changes.patch`：本次测试时使用的源码 / 测试修改。
- `processed/<robot>/stage1/<task>/`：新的 `initial.blend`、`quality_check.png` 与配准文件。
- `verify.py`：本次真实视频抽帧、缓存回放及 CPU 后续流程的检查脚本；依赖本机历史视频和缓存，不是无需数据的公共示例。

历史 Franka mask 使用 `robotic arm`，当前默认提示词为 `robotic arm with base and camera`。
九个 Franka 缓存的结构完整，但不满足当前默认提示词的自动复用条件；本次回放 manifest 保留历史提示词与 `cache_replay` 来源。
九个 Piper 缓存满足当前批量入口的复用条件。

两项真实模型测试因未设置输入视频 / 场景环境变量而跳过；此机器同时缺少 CUDA 和权重。
本次验证不能证明当前本地模型的数值输出、显存需求、推理耗时或新提示词分割质量。
还需在 Linux NVIDIA CUDA 环境上准备权重，运行本地模型最小示例与 `models` 标记测试。
配准仍为 `candidate_unverified`、`accepted=false`：生产路径的全部 32 帧参与评分，缺少独立验收帧；本次流程检查不提供物理测量意义的精度验收。
