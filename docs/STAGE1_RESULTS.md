# Stage 1：Franka 与 Piper 配准结果汇总

数据快照：2026-09-24。来源为 `data/processed/{franka,piper}/stage1/<任务>/registration.json`，不是按可视化图片估读。每类机械臂 9 个任务，共 18 个；逐任务检查了 32 帧 RGB、VGGT 深度、米制深度、SAM3 掩码、32 个米制相机外参、4 组参考图/配准叠图以及 `initial.blend`。交付坐标系为机械臂根部，Piper 为 `base_link`、Franka 为 `fr3_link0`；旧 VGGT 坐标的基座变换保存在 `registration.json` 中。

**状态边界：**全部任务均为 `candidate_unverified`、`accepted=false`。同一 32 帧同时参与初始化、配准优化与评分，没有独立验收集。这里的最终 `loss` 是机械臂配准的综合训练分数，不是 Stage 2 场景深度 MAE、RGB Lab 色差，也不是任务成功率；不能把低于 `0.10` 的配准 loss 误当成深度 MAE 达标。Franka 和 Piper 的公式相同，但任务、模板和观测不同，不宜直接用两组均值证明哪种机器人“更准”。

## 指标含义

- **最终 loss ↓**：对优化后的最终姿态、平移和尺度在全部 32 帧计算的无量纲分数；四个交付视角只是其中的可视化抽样。公式见 `src/real2sim/stage1/sim3_registration.py` 的 `ranking()`：`0.50×(1−IoU) + 0.12×(1−Recall) + 0.14×min(5d₁/r,1) + 0.08×min(5d₂/r,1) + 0.08×(1−min(o₁,o₂)) + 0.08×min(d_depth/r,1)`。`d₁,d₂` 是模型→观测与观测→模型的 60% 分位最近邻距离，`r` 是观测点云的尺度，`o₁,o₂` 是双向近邻覆盖比例；没有有效深度时最后一项取满额惩罚。
- **旋转前 loss ↓**：多尺度 Sim(3) ICP 和局部优化之后、最终增强旋转搜索之前的 `baseline_loss`；**不是**未经优化的 48/128 个初始旋转分数。最终 loss 对应 Franka Top-1 或 Piper Top-3 最终获选分支。
- **IoU ↑、Recall ↑**：机械臂投影轮廓与 SAM3 掩码的 32 帧均值；不是所有任务物体的分割得分。
- **深度残差中位数 ↓**：先在每帧模型投影、SAM3 掩码与有效观测深度共同覆盖的像素上取绝对深度残差中位数，再对有残差的帧取中位数；原值为 VGGT 单位，此处除以 `s` 转为 **毫米**。**不是**全部像素的平均深度误差（MAE），亦不包含尚未建模的任务物体。
- **尺度 `s`**：`scale_vggt_units_per_meter`，单位为 VGGT 单位/米；`1/s` 是每 VGGT 单位对应的米数，不是误差。深度残差列的有效观测覆盖为各任务 **32/32 帧**。

## 概览

| 机械臂 | 任务数 | 最终 loss 均值 | 中位数 | 范围 | 平均 IoU | 深度残差中位数的任务中位数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Franka | 9 | 0.11364 | 0.11221 | 0.10624–0.12115 | 0.8438 | 35.3 mm |
| Piper | 9 | 0.10464 | 0.10373 | 0.09997–0.11216 | 0.8405 | 21.3 mm |

下表按视频任务名排序；箭头表示单项指标的优选方向。数值四舍五入仅用于阅读，精确值以各场景 `registration.json` 为准。`Rank` 为胜出初始种子的**评分排名**，不是旋转候选的编号。

### Franka（48 个初始旋转，Top-1 后续优化）

| 任务 | 最终 loss ↓ | 旋转前 loss ↓ | IoU ↑ | Recall ↑ | 深度残差中位数 (mm) ↓ | 尺度 `s` (VGGT/m) | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `close_drawer` | 0.10624 | 0.17494 | 0.8519 | 0.9763 | 35.3 | 0.5800 | 1 |
| `put_corn_in_bucket` | 0.11385 | 0.28598 | 0.8558 | 0.9817 | 39.3 | 0.5355 | 1 |
| `put_cup_in_box_and_close_box` | 0.11465 | 0.29848 | 0.8434 | 0.9624 | 27.7 | 0.5371 | 1 |
| `stack_snacks` | 0.11196 | 0.26301 | 0.8397 | 0.9692 | 38.0 | 0.5312 | 1 |
| `stack_two_bowls_1` | 0.11191 | 0.25840 | 0.8469 | 0.9757 | 33.3 | 0.5643 | 1 |
| `store_fruit_in_bamboo_tray` | 0.11221 | 0.24543 | 0.8433 | 0.9684 | 32.5 | 0.5342 | 1 |
| `store_fruit_on_plate` | 0.12115 | 0.31231 | 0.8328 | 0.9701 | 37.3 | 0.5283 | 1 |
| `store_three_tools` | 0.12021 | 0.26966 | 0.8371 | 0.9761 | 25.3 | 0.5293 | 1 |
| `store_two_beanbags` | 0.11058 | 0.27465 | 0.8431 | 0.9789 | 41.4 | 0.5318 | 1 |

### Piper（128 个初始旋转，Top-3 分支各自优化后选优）

| 任务 | 最终 loss ↓ | 旋转前 loss ↓ | IoU ↑ | Recall ↑ | 深度残差中位数 (mm) ↓ | 尺度 `s` (VGGT/m) | Rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `close_cup_lid` | 0.10251 | 0.13380 | 0.8395 | 0.9476 | 19.9 | 0.8238 | 1 |
| `put_brown_block_in_white_and_green_storage_box` | 0.10042 | 0.14351 | 0.8503 | 0.9567 | 22.1 | 0.8151 | 2 |
| `put_emergency_stop_button_in_green_storage_box` | 0.11216 | 0.18959 | 0.8413 | 0.9482 | 27.9 | 0.7961 | 3 |
| `put_glue_in_clear_cup` | 0.10432 | 0.14583 | 0.8352 | 0.9587 | 20.2 | 0.7793 | 2 |
| `put_glue_in_pen_holder` | 0.10807 | 0.14802 | 0.8392 | 0.9370 | 21.3 | 0.7900 | 1 |
| `stack_brown_blocks` | 0.10373 | 0.15450 | 0.8388 | 0.9491 | 19.1 | 0.8476 | 2 |
| `stack_cones` | 0.10927 | 0.15361 | 0.8308 | 0.9582 | 23.1 | 0.8517 | 2 |
| `stack_storage_boxes` | 0.09997 | 0.12980 | 0.8463 | 0.9523 | 21.3 | 0.8167 | 2 |
| `take_out_tissue` | 0.10130 | 0.14177 | 0.8434 | 0.9561 | 21.0 | 0.8599 | 2 |

## 保留数据与用途

| 每任务路径（`data/processed/<机械臂>/stage1/<任务>/`） | 可用范围 |
| --- | --- |
| `frames/`、`sam3_masks/` | 32 帧 RGB/机械臂掩码，供配准审阅或后续场景建模 |
| `depth/`、`camera.json`、`depth_conf.npz` | VGGT 原始未定尺度数据，供重新拟合与复算；**不能**当米制输入 |
| `depth_metric/`、`camera_metric.json` | 32 帧米制深度与机器人根部坐标系下的相机，供后续建模使用 |
| `registration.json`、`stage1_delivery.json` | 逐帧指标、最终 loss/尺度、配准状态、交付坐标约定；精确数值来源 |
| `reference/`、`quality_check.png` | 0、10、21、31 帧 RGB 与配准叠图，供人工复核 |
| `initial.blend` | 根部位于世界原点的固定姿态机械臂及四个打包 RGB 的参考相机；尚未完成物体的 Stage 2 建模 |

交付数据与 `runs/` 优化缓存受 `.gitignore` 排除，文档中的数字是从当前本机交付快照转录的。论文中的“重投影误差”“深度 MAE”“成功率”须另设独立验证和相应评测，不能直接用上述训练 loss 代替。

## Stage 2 物体模型与 Stage 1 观测对照（2026-09-25）

早期的四视角观测互相比对只能反映多视角一致性，**不是 Stage 2 模型误差**，
因此旧数字不再作为实验表格使用。正确对照使用
`data/processed/<robot>/stage2/<task>/geometry/final.blend` 的特权网格身份，
从同四台参考相机光线投射出逐物体可见点云，与 SAM3 分割＋米制深度生成的
同视角观测点云比较；指标为对称 Chamfer (m)、F1 @ 1 cm 和可见点云 AABB
中心位置差 (m)。任务专属 SAM3 原始掩码、有效掩码和 Blender 预测掩码均逐
物体/逐视角保存至 `runs/stage1_vs_stage2_objects/<robot>/<task>/`，便于核验是否
分割了**完整的任务物体**。每个 Blend 网格组在 `model/blend_inventory.json`。

逐物体/场景数值分别以 `runs/stage1_vs_stage2_objects/objects.csv` 和
`summary.csv` 为准；仅在模型网格及观测点云都有效时给出数字。米制深度是单目
重建，不是真实深度传感器 GT；F1 的 1 cm 阈值对尺度/位置偏差特别敏感，
不能将未通过阈值简单归因于几何形状。

| 场景 | 目标组 | Chamfer (m) ↓ | F1 @ 1 cm ↑ | 可见 BBox 中心差 (m) ↓ |
| --- | ---: | ---: | ---: | ---: |
| `franka/close_drawer` | 1 | 0.0222 | 0.2936 | 0.0556 |
| `franka/put_corn_in_bucket` | 2 | 0.0435 | 0.0351 | 0.0894 |
| `franka/put_cup_in_box_and_close_box` | 2 | 0.0149 | 0.4363 | 0.0711 |
| `franka/stack_snacks` | 2 | 0.0133 | 0.4375 | 0.0589 |
| `franka/stack_two_bowls_1` | 2 | 0.0167 | 0.3212 | 0.0471 |
| `franka/store_fruit_in_bamboo_tray` | 4 | 0.0417 | 0.1913 | 0.1003 |
| `franka/store_fruit_on_plate` | 4 | 0.0145 | 0.3949 | 0.0550 |
| `franka/store_three_tools` | 4 | 0.0153 | 0.3588 | 0.0297 |
| `franka/store_two_beanbags` | 3 | 0.0319 | 0.1037 | 0.0991 |
| `piper/close_cup_lid` | 2 | 0.0082 | 0.6817 | 0.0174 |
| `piper/put_brown_block_in_white_and_green_storage_box` | 2 | 0.0359 | 0.3730 | 0.0864 |
| `piper/put_emergency_stop_button_in_green_storage_box` | 2 | 0.0350 | 0.0809 | 0.0772 |
| `piper/put_glue_in_clear_cup` | 2 | 0.0132 | 0.5571 | 0.0451 |
| `piper/put_glue_in_pen_holder` | 2 | 0.0092 | 0.7296 | 0.0405 |
| `piper/stack_brown_blocks` | 2 | 0.0077 | 0.7146 | 0.0324 |
| `piper/stack_cones` | 2 | 0.0161 | 0.2658 | 0.0371 |
| `piper/stack_storage_boxes` | 2 | 0.0806 | 0.4905 | 0.2856 |
| `piper/take_out_tissue` | 2 | 0.1245 | 0.1131 | 0.3367 |

重新分割后共 18 场景、42 个目标；双零食盒和双棕色积木分别用模型投影对
联合 SAM3 掩码的连通域指认，结果仍需人工复核。纸巾盒评估掩码从其 SAM3
原始掩码扣除了单独分割出的纸巾。模型对细薄目标在低于 30 个采样点时自适应
改用逐像素采样，`franka/store_two_beanbags` 的托盘四视角现均可计算。
`piper/take_out_tissue` 第 21 帧两个目标均未检测到。
其余单视角漏检/重叠可见对应物体的 `invalid_views`，失效视角不纳入均值；
这些数值都是未经人工核验的伪参考，不是已通过验收的结果。旧的观测对观测
实验 `runs/stage1_object_metrics/` 已清理，不再作为实验表格使用。
