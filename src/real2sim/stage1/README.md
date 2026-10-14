# Stage 1：配准与初版场景

当前生产路径：Franka 使用 48 旋转 Top-1；新姿态 Piper 使用 128 旋转 Top-3，各分支独立优化后以 32 帧最终 loss 选择交付结果。两者均使用多尺度 Sim(3) ICP、增强旋转、URDF 锚定米制和四相机 `.blend`。命令和保留清单见 `docs/STAGE1_FRANKA_DELIVERY.md`、`docs/STAGE1_PIPER_DELIVERY.md`；旧 Piper 256/Top-K 实验不用于新视频。

使用独立输出目录时，交付命令需要同时指定感知场景、配准结果和机器人初始化目录。例如 Franka：

```bash
uv run real2sim-stage1-franka-deliver \
  --processed runs/my_stage1/processed/franka/stage1 \
  --results runs/my_stage1/registration/franka \
  --initializations runs/my_stage1/initializations/franka \
  --only close_drawer --blender /path/to/blender
```

Piper 使用 `real2sim-stage1-piper-deliver` 并传入对应的 Piper 目录。`--initializations` 指向包含 `<任务>/initialization.json` 的目录；交付前仍校验配准所记录的初始化 SHA256。无 CUDA 的流程验证见 [CPU 验证记录](../../../docs/STAGE1_CPU_VALIDATION.md)。

## Stage 2 模型对 Stage 1 物体观测的指标

运行 `uv run real2sim-stage1-object-metrics`，对 18 个场景分别读取
`data/processed/<robot>/stage2/<task>/geometry/final.blend`。每个目标的完整网格组
由任务专属映射明确指定；例如急停开关同时包含壳体、标签、红色按钮、黄色环
等六块网格，另外将绿色收纳盒单独评估；沙包分别匹配红色和蓝绿色实例。Blender 在四台参考相机上以网格
身份作为特权信息，光线投射得到可见表面点云和预测掩码。对应的 Stage 1
四张 RGB（抽帧 0、10、21、31）用任务专属 SAM3 提示词分割，经
`depth_metric/` 与 `camera_metric.json` 反投影，剔除机械臂像素。

每个有效视角上直接比较**Stage 2 模型可见点云**与**Stage 1 物体观测点云**：
对称 Chamfer Distance (m)、双向最近邻 1 cm F1、可见点云的轴对齐包围盒
中心距离 (m)。同视角图像通常以 2 像素步长采样；模型预测不足 30 点的细薄
物体会在该视角自动加密到 1 像素，具体步长记录于 `model/blend_inventory.json`。
点云按 5 mm 体素去重；任一侧仍少于 30 点的视角明确标为无效。结果保存在
`runs/stage1_vs_stage2_objects/<robot>/<task>/report.json` 和场景/物体级 CSV。
逐物体原始 SAM3 掩码（`masks/<object>/mask_*.npy/png`）、掩码叠图、剔除
机器人后的有效掩码，以及 Blender 特权渲染的预测掩码与点云分别保存；
`model/blend_inventory.json` 可复查每个物体包含的全部 Blend 网格。
同色零食盒和棕色积木先以共同提示词分割，保留原始联合掩码在
`masks/_shared_instances/`，从最大两个连通域分离实例并按模型投影匹配身份；
结果标记 `instance_scope=model_assigned_same_appearance_instance`，匹配可能因模型偏差
或实例接触而出错，需要人工复核。纸巾盒的原始 SAM3 掩码会包含露出的纸巾，
其评估掩码另存 `evaluation_mask_*.npy/png`，从原始盒掩码扣除纸巾掩码；
`mask_*.npy/png` 始终保留原始结果。
如果两个实例的 SAM3 原始掩码明显重叠，`review_flags` 会提示人工复核。
`model/<object>/comparison_*.png` 在原始 RGB 上以绿色显示 SAM3-only 像素、
红色显示 Blend-only 像素、黄色显示双方重合；实际逐像素采样分辨率由步长决定。
Stage 1 交付目录保存 `object_metrics.json` 报告副本；`--no-sidecars` 可禁用。

这些是模型相对 SAM3＋单目估计米制深度的**伪参考**误差，而非物理测量的
真实 GT；位置指标是*可见表面点云*的包围盒中心差，不是完整 3D 包围盒位姿
误差。物体在录制过程中的运动、遮挡、实例漏检、深度系统偏差仍可能影响分数，
因此报告标记 `measured_unverified`，不能用作自动放行。

如需复查 Piper SAM3 掩码，运行 `uv run real2sim-stage1-sam3-review` 重新生成 `runs/sam3_review/piper/index.html`。每个当前视频包含 32 帧原图、绿色 SAM3 联合 mask 叠图、黑白 mask 和整段接触表。**不调用远端模型、不重新配准**；缓存已把多个实例合并，不能恢复每帧逐实例的分数与框。

对比新提示词可运行 `uv run real2sim-stage1-sam3-prompt-trial`：只串行重跑 SAM3 的全部 32 帧，不触碰原始 mask、VGGT 或已有配准。用户中文原句“带底座机械臂加相机”在当前九段视频的第 0 帧均返回 `K=0`；实际发送同义英文 `robotic arm with base and camera`（`conf=0.25`）。结果在 `runs/sam3_prompt_trials/piper/base_camera/index.html`，**每场景只保存一张** 32 帧原图/旧 mask/新 mask 对照图，并保留新的 32 帧 mask NPY 和每帧实例数供核查；对照结果不会自动覆盖原 mask 或重新开始配准。

## Franka：固定的 48 初值 + 单分支 Sim(3) 配准流程

运行 `uv run real2sim-stage1-franka-final`，依次读取 `runs/initial_poses/franka/<任务>/initialization.json` 中 **48 个旋转**按全部 32 帧评分的排名，每个任务严格只取第一名作为初值。先以 5000→10000→20000 点模板进行截尾 Sim(3) ICP，并用小步局部搜索优化位置、旋转、scale；再沿世界坐标轴以 ±15°、±8°、±4°、±2° 做增强旋转搜索。每级重复尝试直至无法继续改善；每个旋转候选围绕模型中心转动，并允许同步微调平移、尺度。全部更新按 20000 点模型、全部 32 帧的原有综合 loss 评价，改善不足 0.0001 的候选不会被接受。不运行 Piper 的 256 采样或 Top-K 多分支搜索。

产物放在 `runs/franka_stage1_final_48/`，`index.html` 按任务顺序展示每个任务优化前、后的 32 帧叠图；每个任务的 `result.json` 保存初始分支、多尺度 ICP 的结果、各级旋转搜索历史、最终位姿/scale 和 32 帧评分。`before_after_32.png` 用于可视化对比，`overlay_32.png` 只显示增强旋转后的结果。单任务检查可传 `--only close_drawer --output runs/your_output`。

**Franka 米制尺度与交付坐标**：FR3 固定关节姿态的 URDF 表面点云单位为米。最终 Sim(3) 拟合得到 `s = scale_vggt_units_per_meter`，即每米对应的 VGGT 单位数；以 FR3 几何为米制锚点得到 `meters_per_vggt_unit = 1/s`。优化结果中的 `metric_calibration` 与运行缓存相机先沿用 VGGT 世界轴与原点、将平移除以 `s`；原始 `depth/`、`camera.json` 留作复算。正式交付的 `camera_metric.json`、`registration.json` 和 `initial.blend` 再整体重基到 URDF 机械臂根部：根对象为单位变换，32 帧相机外参同步变换，内参与米制深度不变；原 VGGT 米制基座位姿记录在 `registration.json` 以供追溯。这一步是**基于已知机器人真实尺寸求得的尺度**，不再要求另给一个手工 `--metric-scale`。

**验收边界**：虽然尺度已由 URDF 锚定为米，但其精度取决于真实机械臂的关节角和 URDF 模板一致，以及掩码、深度、相机和静态配准的质量。32 帧同时参与选种子、优化和评分，没有独立验收帧；米制化**不等于**深度 MAE、重投影及任务场景整体已经验收。因此结果仍标记 `candidate_unverified`、`accepted=false`，不能作为 Stage 2 放行依据。旧版 `runs/franka_final_48/` 和单场景实验结果保留用于对照，不会被默认入口覆盖。

**Franka 初版 `.blend` 和精简交付**：已有配准结果无须重新运行长时间 ICP；执行 `uv run real2sim-stage1-franka-deliver --prune`，按任务逐个检查视频、URDF、初值哈希、32 帧深度与米制变换，生成包含 FR3 visual 几何与四个打包 RGB 的参考相机的 `initial.blend`，重新打开验证变换和文件完整性。正式产物放进 `data/processed/franka/stage1/<任务>/`；`registration.json` 保留最终位姿/尺度/逐帧评分但不重复整个搜索历史，`reference/overlay_{00,10,21,31}.png` 和 `quality_check.png` 用于人工审阅。`--prune` 仅在该任务成功后清理可重建的 `raw_vggt.npz`、32 张深度/掩码预览；保留原始深度/相机与置信度以便重跑配准。运行与保留清单详见 `docs/STAGE1_FRANKA_DELIVERY.md`。

## Piper：128 初值、Top-3 分支和正式交付

运行 `uv run real2sim-stage1-init-batch --robot piper --active-only` 对当前九段视频按全部 32 帧评分 128 个旋转初值，输出到 `runs/piper_initial_128_review/piper/`。然后运行 `uv run real2sim-stage1-piper-top3 --workers 4`，对前三名分别进行多尺度 Sim(3) 与旋转搜索，按最终 32 帧 loss 自动选优，结果在 `runs/piper_stage1_top3/`。已有同一输入的 Top-1 结果从 `runs/piper_stage1_final_128/` 复用；保留这三个运行目录，便于复现排名、各分支优化和重新交付。运行 `uv run real2sim-stage1-piper-deliver --prune` 将九份精简交付写到 `data/processed/piper/stage1/`，总览见该目录的 `index.html`。旧 48/256 初值、强制 R42 和过期视频缓存均不参与正式流程。

## 旧版调试：48 旋转初值可视化

运行 `uv run real2sim-stage1-init-batch`，依次对 `data/processed/{franka,piper}/stage1/` 下的全部场景生成初值总览。输出索引是 `runs/initial_poses/index.html`；每个场景只生成一张 `top4.png`，从上到下为评分前四名，每名展示全部 32 帧（绿色是 SAM3 mask，红色是固定姿态机械臂的 20000 点表面投影）。同目录的 `initialization.json` 保存 48 个旋转的矩阵、平移、32 帧评分及前四名，`observation.ply` 是用于中心/尺度及点云评分的观测点云。

观测点来自全部 32 帧 mask 内深度；4 万点模型与观测点分别求中心，使用两云 5%–95% 分位数包围盒对角线之比作为**唯一**粗尺度，20000 点模型从 40000 点模板中确定性抽样。每个旋转的平移只由中心对齐得到。48 个旋转为确定性拟均匀 SO(3) 四元数采样，所有场景用同一组旋转；用 **32 帧** mask IoU/召回率/投影深度及双向点云近邻距离综合计算 loss，**loss 越小排名越高**，只展示前四名。**不运行 ICP、不调优位置或尺度，也不生成 `.blend` 或验收 Stage 1**。此调试输出没有独立留出帧，因此评分只是本轮比较初值的指标，不是配准验收；VGGT 深度单位尚未独立标定为米。

## 缓存结果的探索性 Sim(3) 配准（不自动放行）

两台机械臂均已有 32 帧推理结果时，可分别运行：

```bash
uv run real2sim-stage1-register --input data/processed/piper/stage1/take_out_tissue --output runs/registration/piper/take_out_tissue
uv run real2sim-stage1-register --input data/processed/franka/stage1/close_drawer --output runs/registration/franka/close_drawer
```

该实验用 mask 内深度及相机参数反投影出观测云，假设 32 帧机械臂静止；不使用旧项目的 RANSAC 直线初值。以观测云和模板中心对齐，确定性遍历 SO(3) 与尺度候选，依次使用 `asset/robot_surfaces/<robot>/` 的 5000/10000/40000 点优化平移、旋转和全局尺度。候选根据训练帧的多视角 mask IoU、可见投影深度及双向截尾点云距离排序；索引 0/10/21/31 不参与拟合和候选选择，只作留出诊断。每次输出观测云、对齐模型、四张绿色 mask / 红色模型叠图和 `report.json`，报告仅标为 `candidate_unverified`。报告中的尺度是 **VGGT 单位/URDF 米**，不是独立米制标定；如果视频的实际关节角与固定模板不一致，单一刚体相似变换无法校正。实验不会生成 `.blend` 或把旧 Stage 1 标记为通过。

## 当前批量推理（不做配准）

先运行 `uv run real2sim-stage1-infer-batch`：按顺序处理 `data/raw/frank/` 与 `data/raw/piper/` 的全部 MP4，分别输出到 `data/processed/franka/stage1/<视频名>/` 和 `data/processed/piper/stage1/<视频名>/`。每个目录含 32 帧 `frames/`、32 个未定尺度深度 `.npy`、32 个 SAM3 mask `.npy`、四帧 `reference/`、`camera.json`、`depth_conf.npz` 和 `manifest.json`；默认不再生成重复的 64 张预览图和 `raw_vggt.npz`。失败保留诊断，成功记为 `inference_complete`，**不代表完成 Stage 1 或获得米制深度**。指定单视频可加 `--only <文件名>`；已完成且输入未变化的结果默认跳过，加 `--force` 则归档历史结果后重跑。

## 旧版 Piper 单视频配准入口（非默认生产流程）

第一批仅支持 Piper。先按 [本地安装说明](../../../docs/LOCAL_INFERENCE.md) 初始化子模块、安装依赖并准备两份本地权重；后端默认 `local`、设备默认 CUDA（参考 `.env.example`），运行：

```bash
uv run real2sim-stage1 --video data/raw/piper/take_out_tissue.mp4 --task "抽出纸巾" --output runs/stage1
```

Blender 可用 `--blender /path/to/blender` 指定；否则读取 `BLENDER_BIN`、查找 PATH 中的 `blender`，仅 macOS 再尝试 Blender.app。正式批量交付和物体评估使用相同优先级。配置了不存在或不可执行的路径时明确失败。

每次运行生成唯一目录，不复用上次结果。均匀抽取 32 帧，索引 0、10、21、31 是独立的配准验收帧；其他 28 帧用于拟合。保存重建数据（世界到 OpenCV 相机的外参）、四组 RGB/深度/点云、机械臂 mask 和投影、配准报告、初版 `initial.blend`（四参考相机 + Piper）和四视角 Blender 渲染烟测。每个相机对应一个 Blender Scene，以保留各自像素纵横比。产物目录中的 `manifest.json` 是唯一放行标记：仅 `status=passed` 时 Stage 2 才可消费。失败保留诊断与候选，命令返回非零；以前的运行不会被覆盖。

**尺度前提**：单目 VGGT-Ω 不能单凭视频证明米制尺度。不提供外部测量的 `--metric-scale` 时使用 1.0 仅生成候选，但始终拒绝放行；经独立标定得到“每 VGGT 单位的米数”后，才使用 `--metric-scale <meters-per-unit>` 并重新运行。深度与外参平移一起缩放；这个参数不能为通过阈值而随意调节。输出报告记录阈值版本、四个独立视角逐项掩码 IoU / 覆盖率 / 深度残差 / 有效像素门槛。现有实现假设机器人在采样的 32 帧里具有固定关节姿态；任务视频若机械臂运动，应保留诊断并拒绝当作可信的静态配准，后续接入带时间戳的关节状态或仅静止标定片段。

可显式设置任一后端为 `remote` 并配置对应 TCP 地址，协议见 [VGGT-Ω](../../perception/vggt_omega/README.md) 和 [SAM3](../../perception/sam3/README.md)。本地权重分别放到 `model/vggt_omega/vggt_omega_1b_512.pt` 和 `model/sam3/sam3.pt`。依赖/权重不全直接失败。本地 manifest 记录源码提交、checkpoint SHA256、设备和预处理版本；远端未报告的模型信息保留为空。
