# 本地感知模型安装与最小复现

首个完整推理验收目标是 **Linux x86_64、NVIDIA CUDA、Python 3.11**。源码以固定提交的子模块放在本仓库，checkpoint 可以直接放入 `model/`，也可以挂载到对应目录。默认使用本地 VGGT-Ω 和 SAM3；仅显式选择 `remote` 时连接远端服务。

当前阶段交付本地接入和离线契约测试；没有在本次 macOS 整理环境执行真实 CUDA checkpoint 推理。安装成功、模型检查成功与真实视频复现通过是三个独立的验收步骤。CPU/MPS 可显式配置用于开发，但不作为完整复现验收环境。

## 1. 源码与环境

```bash
git clone --recurse-submodules https://github.com/HY-LiYihan/agentic_rsr.git
cd agentic_rsr
git submodule update --init --recursive
uv sync --locked --python 3.11 --extra local --extra sim --extra dev
```

当前按源码 checkout + editable 安装使用，不承诺独立 wheel 的机器人资产可用性。`uv.lock` 固定依赖解析结果；`uv sync --locked` 遇到过期锁文件会报错，不更新版本。Linux x86_64 的本地推理使用 PyTorch 2.6.0 / torchvision 0.21.0 的 CUDA 12.4 wheel，宿主需要兼容驱动；其他平台从 PyPI 解析对应 wheel。

精简 Linux 系统还需 OpenCV 的 GL/GLib 动态库，以及无窗口 MuJoCo 渲染使用的 EGL/Mesa。Debian 13 的无 GPU 容器验证使用以下系统包；其他发行版应安装对应包，较旧系统的 GLib 包名通常是 `libglib2.0-0`。这些库不提供 NVIDIA 驱动或 CUDA 验收。

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0t64 libegl1 libgl1-mesa-dri
export MUJOCO_GL=egl
```

Blender 另行安装，用于 Stage 1 场景交付、Stage 2 建模/渲染和 Stage 3 导出；感知最小示例本身不调用 Blender。各阶段和相关实验脚本统一按以下优先级选择可执行文件：显式 `--blender` 参数、`BLENDER_BIN` 环境变量、PATH 中的 `blender`；仅在 macOS 上最后尝试 `/Applications/Blender.app/Contents/MacOS/Blender`。参数和环境变量既可使用路径（支持 `~` 和空格），也可使用 PATH 中的命令名。显式配置缺失或不可执行时直接报错，不改用另一个安装。

```bash
blender --version  # Blender already on PATH
# For a separate installation, replace this path:
export BLENDER_BIN=/path/to/blender
```

| 源码 | 位置 | 固定提交 |
| --- | --- | --- |
| [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) | `external/vggt_omega/` | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` |
| [Ultralytics v8.4.160](https://github.com/ultralytics/ultralytics) | `external/ultralytics/` | `f171c4b7aea56062d77061577e6b878b62e520fb` |
| [机器人控制包](https://github.com/HY-LiYihan/robot_control) | `external/robot_control/` | 以主仓库 gitlink 为准 |

`local` extra 包含 `numpy>=1.26,<2`、`opencv-python<4.12`：VGGT-Ω 要求 NumPy 1.x，而 OpenCV 4.12 开始的相关 wheel 要求 NumPy 2.x。支持的项目 Python 范围为 3.10 至 3.12，完整推理统一选 3.11。

SAM3 文本编码器需要 Ultralytics CLIP 包及其内置 BPE 词表；锁文件固定 [CLIP 提交](https://github.com/ultralytics/CLIP/tree/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40)。不要安装 PyPI 上同名但不兼容的 `clip` 包。安装阶段获取依赖；本地推理检查依赖，不在运行中安装包或下载模型。

`sim` extra 从现有子模块安装 `robot-control` 基础包，并单独安装 MuJoCo/Pillow 和 NumPy 1.26 兼容的 Pinocchio 2.7 / Boost 1.83 / urdfdom 3 / TinyXML2 10。机器人连接时即初始化 IK，视觉观测也需要这些依赖。上游 `robot-control[mujoco]` 固定 Pinocchio 3.9，其 Boost 1.89 wheel 在 Python 3.11 上要求 NumPy 2.3，与 VGGT-Ω 冲突；本项目明确锁定兼容栈，保持机器人源码子模块不变，不应再额外安装上游 `mujoco` extra。

仅运行无权重测试时不安装 `local`：

```bash
uv sync --locked --python 3.11 --extra sim --extra dev
uv run --no-sync python -m pytest tests/unit tests/integration -m 'not models' -ra
```

## 2. 申请并准备权重

先阅读[第三方来源与许可](THIRD_PARTY.md)，分别在官方 Hugging Face 页面申请访问并接受条款：

- [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)：选择 `vggt_omega_1b_512.pt`。416 reproduction checkpoint 不用于当前 512 输入契约。
- [facebook/sam3](https://huggingface.co/facebook/sam3)：选择 `sam3.pt`，不混用 SAM3.1 checkpoint。

批准后在本机登录 Hugging Face，下载到忽略目录：

```bash
uv run --no-sync hf auth login
uv run --no-sync hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir model/vggt_omega
uv run --no-sync hf download facebook/sam3 sam3.pt --local-dir model/sam3
```

也可用自己的下载工具或将已有权重目录挂载到 `model/vggt_omega/`、`model/sam3/`。默认路径：

```text
model/vggt_omega/vggt_omega_1b_512.pt
model/sam3/sam3.pt
```

权重、Hugging Face 下载缓存与凭据不提交 Git。没有 checkpoint 时明确报错，不隐式下载，不自动改用远端。

## 3. 后端与设备

从 `.env.example` 复制本机 `.env`，或设置环境变量。相对模型目录始终相对于仓库根目录解析；绝对路径可用于外部挂载。

```dotenv
VGGT_OMEGA_BACKEND=local
VGGT_OMEGA_MODEL_DIR=model/vggt_omega
VGGT_OMEGA_DEVICE=cuda
SAM3_BACKEND=local
SAM3_MODEL_DIR=model/sam3
SAM3_DEVICE=cuda
```

设备可指定 `cuda:0`；配置请求的 CUDA 设备不存在时会失败。两模型可独立选择后端；服务地址仅在该模型的 `remote` 模式读取。例如：

```dotenv
VGGT_OMEGA_BACKEND=remote
VGGT_OMEGA_HOST=your-vggt-host
VGGT_OMEGA_PORT=9001
VGGT_OMEGA_PROTOCOL=tcp
SAM3_BACKEND=local
```

SAM3 远端对应 `SAM3_BACKEND=remote`、`SAM3_HOST`、`SAM3_PORT`、`SAM3_PROTOCOL=tcp`。旧 CLI 显式传入的 SAM3 host/port 继续表示远端调用。Stage 1、物体评估、提示词试验、SIM batch 与练习 CLI 共享配置。

## 4. 检查与最小示例

检查配置、文件、源码版本和设备；`--load` 进一步串行加载并释放两个本地模型，但不代替真实推理：

```bash
uv run --no-sync real2sim-models check
uv run --no-sync real2sim-models check --load
```

使用自己的至少 32 帧任务视频，选择一个尚不存在的输出目录：

```bash
uv run --no-sync real2sim-local-example \
  --video /path/to/task.mp4 --robot piper \
  --output runs/local-example
```

示例输出保持 Stage 1 的帧、深度、相机、mask 和 manifest 结构，并增加 `previews/` 和 `example_report.json`。报告包含后端、源码提交、checkpoint SHA256、预处理版本、设备、Torch/CUDA 版本、耗时和峰值显存。目录已存在即拒绝执行；失败状态保留诊断，不能当作复现通过。

输入为 32 帧、512×288 RGB；VGGT-Ω 直接使用此像素网格，不调用会裁剪宽图的默认加载器。深度、置信度与内参需与该网格对齐。SAM3 实例 mask 恢复到原图网格后，Stage 1 合并实例，SIM 按现有规则选实例。VGGT 与 SAM3 串行执行，阶段切换释放模型；显存不足会报错，不自动减少帧数或分辨率。

这一步只验收感知输出，不完成 URDF 配准、米制尺度锚定或 Blender 场景交付。后续步骤参考 Stage 1 文档。

已有 Stage 3 场景时，可用练习 CLI 验证一次本地 SAM3 RGB-D 观测。Linux 无显示服务器时设置 `MUJOCO_GL=egl`，scene 必须是已导出的完整场景，不能替换成视频或 `initial.blend`：

```bash
MUJOCO_GL=egl uv run --no-sync python scripts/stage3_cli.py \
  --scene /path/to/scene_for_sim.xml --robot piper \
  --image-dir runs/local-sim-observation
```

在 CLI 输入 `observe`，查看 RGB-D 图像及分割后的物体观测，再输入 `quit`。也可指定物体提示词，例如 `observe {"blue_block":"blue block"}`；键名必须属于该场景公开物体。该命令不需要 `SAM3_HOST/PORT`。

## 5. 验证边界

当前仓库没有 GitHub Actions CI（已按作者要求移除）。无 GPU/无 checkpoint 的离线测试覆盖配置、后端契约、实例合并/选择、mask 网格以及相机投影，同时保留远端协议回归；提交修改前需手动运行。Linux 无 GPU 的 MuJoCo 渲染可使用 OSMesa，真实 Linux GPU 观测使用 EGL。真实模型测试标记为 `models`，需要显式提供权重和测试视频，在 Linux CUDA 环境运行：

```bash
export RSR_MODEL_TEST_VIDEO=/path/to/task.mp4
export RSR_MODEL_TEST_SCENE=/path/to/scene_for_sim.xml
export RSR_MODEL_TEST_ROBOT=piper
uv run --no-sync python -m pytest tests/integration -m models
```

没有提供相应的显式输入或缺少 CUDA 时真实模型测试会跳过；跳过不代表推理通过。视频测试和 SIM 观测测试分别读取这些输入。

必须保存真实运行报告及依赖版本后，才能声明本地完整推理复现通过。推理性能和峰值显存受视频、GPU 与模型设备设置影响，不引用上游不同输入尺寸的显存表作为本流程保证。
