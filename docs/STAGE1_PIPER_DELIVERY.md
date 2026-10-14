# Piper Stage 1：新姿态视频的米制重建与初版 `.blend`

当前 `data/raw/piper/` 的九段视频为正式输入。整机固定姿态采用 `joint1`–`joint6 = [0°, 30°, −45°, 0°, 60°, 0°]`；夹爪保持先前的 0.07 m 总开口。Piper 的 40000/10000/5000 点模板沿用 AGX 腕部/夹爪 URDF，加上 `robot_control` 固定的 Isaac V100 D435 和相机支架 visual，以及其 MuJoCo 中的腕部安装变换，单位为米；来源哈希与组合模型哈希记入 `metadata.json`。原来的 11 网格点云缺相机与支架，不能再用于本次配准。

顺序运行（远端推理地址仅从本地 `.env` 读取）：

```bash
uv run real2sim-robot-surfaces --robot piper
uv run real2sim-stage1-infer-batch
uv run real2sim-stage1-init-batch --robot piper --active-only
uv run real2sim-stage1-piper-top3 --workers 4
uv run real2sim-stage1-piper-deliver --prune
```

SAM3 默认提示词为 `robotic arm with base and camera`（远端/本地一致，`conf=0.25`）；中文原句在当前九段视频的测试中未产生有效实例。推理缓存同时核对视频哈希和提示词，提示词变更后不会误用旧结果。需要试验其他提示词时可运行 `uv run real2sim-stage1-sam3-prompt-trial`；满意后运行 `uv run real2sim-stage1-promote-sam3` 校验并提升试验掩码：仅替换 mask、更新 manifest、作废旧交付，保留 VGGT 的相机与深度；然后依次重跑上面的初值、优化、交付三条命令。

推理命令会按视频哈希和提示词复用未变化的完成结果；视频更换时重新推理并归档旧缓存。`--active-only` 和 Piper 最终优化入口只处理仍存在于 `data/raw/piper/` 且视频哈希匹配的场景。注册沿用 Franka 的优化流程：全部 32 帧计算观测点云、**128 个 SO(3) 初始旋转按综合 loss 自动选前三名**，分别进行 5000→10000→20000 点截尾 Sim(3) ICP、局部调整和 ±15°→8°→4°→2° 旋转搜索，之后以同一 32 帧最终 loss 选出交付分支；不使用手工强制 R42。有效的 Top-1 优化可以复用，Top-2/3 分支的缓存分别校验初始化文件、掩码及序号。

固定姿态的 Piper URDF 表面提供米制锚点，拟合尺度 `s` 的单位为 VGGT 单位/米。深度、外参平移及机器人基座平移先除以 `s`，机械臂网格和旋转不缩放；交付时整体重基到 Piper `base_link` 原点与轴向，32 帧米制相机外参与四视角 `.blend` 同步变换，深度与内参不变。交付目录为 `data/processed/piper/stage1/<视频名>/`，结构与 `STAGE1_FRANKA_DELIVERY.md` 相同：32 帧 RGB/掩码、供重跑配准的原始 VGGT 深度/相机/置信度、32 帧米制深度/相机、紧凑 `registration.json`（含重基前基座位姿）、四帧 RGB 与四帧配准叠图、`quality_check.png`、`stage1_delivery.json` 和打包参考图像的四相机＋固定姿态 Piper `initial.blend`。`--prune` 仅在场景验证成功后移除重复预览缓存。

`initial.blend` 的 Piper 颜色取自 `robot_control` 中 AGX Piper、D435 和支架的 DAE 嵌入材质；各颜色部分分别导入并配置 Blender 材质，不再把所有网格合并成无材质 OBJ。若使用 Blender 实体视图，需选「材质」颜色或切换材质预览；这不改变用于 ICP 的米制点云与配准 loss。

点云拟合与评分共用这 32 帧；四个相机是视频中 0、10、21、31 帧的重建视角，**不是四台实体相机，也不是独立验收帧**。米制标定并不等于配准已独立验证，交付状态仍为 `candidate_unverified`、`accepted=false`，不能据此启动 Stage 2。Piper 总览生成在 `data/processed/piper/stage1/index.html`；原视频及运行产物均在 `.gitignore` 排除范围，点云模板及代码/文档可提交。

## 128 位姿初值复查

`uv run real2sim-stage1-init-batch --robot piper --active-only` 默认使用 **128** 个初始旋转，写入 `runs/piper_initial_128_review/`，每个视频 `top4.png` 按全部 32 帧评分升序展示前四名（均为未经 ICP 优化的姿态）。Franka 默认仍为 48。Piper Top-3 后续结果在 `runs/piper_stage1_top3/`：分支结果在 `branches/rank02/`、`branches/rank03/`（可用则复用 `runs/piper_stage1_final_128/` 中的 Top-1），每个场景的 `result.json` 记录三个分支的初始/最终排名、loss 与选中分支；交付命令生成包含 D435 和支架的 `.blend`。只保留上述三处复现缓存及当前视频的正式交付；过期视频、`.history/` 和旧 48/256/R42 试验结果可清理。由于三个分支仍使用同一组 32 帧评分，所有结果仍为 `candidate_unverified`，不能仅凭最低 loss 放行 Stage 2。
