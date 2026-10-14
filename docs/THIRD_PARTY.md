# 第三方来源与许可

本清单记录第一阶段本地感知接入及现有机器人资产来源，不代替各上游许可全文。主项目原创代码与文档采用 [Apache-2.0](../LICENSE)；第三方源码、权重、机器人资产及派生文件、论文媒体和学校标识保留原有条款，适用范围见 [NOTICE](../NOTICE)。资产与派生文件通过 Hugging Face 独立分发，发布状态见[资产分发说明](ASSET_DISTRIBUTION.md)。托管平台的选择不改变资产许可。

| 组件 | 来源与固定版本 | 许可与使用边界 |
| --- | --- | --- |
| VGGT-Ω 源码 | [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega/tree/48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6)，`external/vggt_omega/` | [FAIR Noncommercial Research License](../external/vggt_omega/LICENSE)，附 Acceptable Use Policy；仅非商业研究使用，发布研究结果须致谢，再分发须遵守并附上许可。不能称为允许任意商业用途的开源许可证。 |
| VGGT-Ω 512 checkpoint | [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega)，`vggt_omega_1b_512.pt` | 需申请访问并接受模型页面条款；Research Materials 包含权重。权重不包含在仓库中。 |
| Ultralytics SAM3 实现 | [ultralytics/ultralytics v8.4.160](https://github.com/ultralytics/ultralytics/tree/f171c4b7aea56062d77061577e6b878b62e520fb)，`external/ultralytics/` | [AGPL-3.0](../external/ultralytics/LICENSE)；组合、修改、分发及提供网络服务时须评估对应源码与许可义务。商业另有 [Ultralytics Enterprise License](https://www.ultralytics.com/license)，本仓库不提供该授权。 |
| SAM3 checkpoint | [facebook/sam3](https://huggingface.co/facebook/sam3)，`sam3.pt` | [SAM License](https://github.com/facebookresearch/sam3/blob/main/LICENSE)，2025-11-19 版本；是 Meta 自定义条款，不能将其等同于 Ultralytics AGPL 或 Apache-2.0。需申请访问；条款包含再分发、致谢、贸易管制与用途限制。权重不包含在仓库中。 |
| CLIP 文本分词依赖 | [ultralytics/CLIP](https://github.com/ultralytics/CLIP/tree/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40)，锁定 git revision | fork 使用 [AGPL-3.0 LICENSE](https://github.com/ultralytics/CLIP/blob/7ffa84b3bfa40c42ecc1c77147a855e69cb2dd40/LICENSE)，继承 OpenAI CLIP 来源并包含 BPE 词表；保留上游版权与许可。 |
| robot-control | [HY-LiYihan/robot_control](https://github.com/HY-LiYihan/robot_control)，固定 gitlink `6016dd5f20d2146a28e425ce77dfb9430f1173`（公开分支 `codex/fr3-urdf-paths`） | [子模块 LICENSE](../external/robot_control/LICENSE)；vendor 资产分别适用其各自授权，不能由控制包许可证一并覆盖。 |

## 现有机器人资产

机器人控制子模块引用 AgileX 的 [piper_sdk](https://github.com/agilexrobotics/piper_sdk)、[piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim)、[agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf)，还包含 `vendor/fr3_d435i/`。`piper_sdk` 当前 checkout 保留 MIT 声明，`agx_arm_urdf` 保留其 MIT 文件；当前 `piper_isaac_sim` checkout 未发现许可证文件，不能据此假设其中 mesh、URDF 或贴图可以再分发。GitHub API 在 2026-10-14 对该仓库返回 `license: null`，仓库内的 `piper_description/package.xml` 仍写有 `TODO: License declaration`。各供应商的 mesh、URDF、贴图与 SDK 需要独立核对再分发权利，模型实现许可证不覆盖这些文件。

FR3/D435i 已保留独立声明：`external/robot_control/vendor/fr3_d435i/licenses/D435i_Apache-2.0.txt`、`Panda_Hand_Apache-2.0.txt` 和 `mujoco_fr3/LICENSE`。本阶段保留原文件，不重新授权。正式开源发布前仍需完成供应商资产逐项核验，并确认生成并提交的 `asset/robot_surfaces/*` 点云是否属于允许分发的衍生文件；当前 metadata 只记录源路径和 SHA256，不构成许可证明。

### 固定资产版本与待核验范围

2026-10-14 按 `robot_control` 的固定提交 `6016dd5f20d2146a28e425ce77dfb9430f1173` 核验。下表中的 FR3 路径相对于 `external/robot_control/vendor/fr3_d435i/`；版本是源码定位证据，不是再分发授权。

| 资产范围 | 当前来源证据 | 发布前待补 |
| --- | --- | --- |
| Piper 手臂、夹爪 Xacro 和网格 | `agx_arm_urdf` 固定 `f6642ce0d7872c686f29c99e9e10cd23d1d49313`；MIT 文件声明 Copyright (c) 2026 aalicecc | 保留 MIT 全文；核对供应商网格是否有独立条款或署名 |
| Piper / FR3 原始 D435 DAE | 两处 `realsense2_description/meshes/d435.dae` 均逐字节匹配官方 RealSense 固定来源，见下文 | 官方描述包声明 Apache License 2.0；本仓库已保留对应 LICENSE / NOTICE。它们不覆盖改造后的 `d435_official.obj` 或打印支架 |
| Piper 打印支架及其他 Isaac 资产 | `piper_isaac_sim` 固定 `8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2`；来源及哈希见 Piper `metadata.json` | 该仓库缺少许可证，`piper_description/package.xml` 为 `TODO: License declaration`；支架等未匹配资产仍需权利方授权或许可明确的替代资产 |
| Piper SDK | `piper_sdk` 固定 `c9e8a28174e71eeaac448593cb65f8ab258a92fe`；MIT，Copyright (c) 2024 Agilex Robotice Co., Ltd.（沿用原声明拼写） | 保留 MIT 全文和版权声明 |
| FR3 手臂：`mujoco_fr3/assets/link*`、MJCF | 42 个 link 网格与固定 Menagerie `franka_fr3/assets/` 内容匹配，见下表；随附 `mujoco_fr3/LICENSE` 包含 Apache-2.0 和 BSD 附段。`codex/fr3-urdf-paths` 分支修正官方 URDF 的 9 个 collision 相对路径，复用同一组已核对的 STL | 完整保留现有许可，包括 Willow Garage 署名与背书限制；自定义挂载件、转换 OBJ 和其他混合来源资产仍需单独核验 |
| Franka Hand：`mujoco_fr3/assets/fr3_hand*`、`fr3_finger*` | 5 个 hand、2 个 finger OBJ 和 hand collision STL 与固定上游 Panda 内容匹配；对应上游模型许可与 `licenses/Panda_Hand_Apache-2.0.txt` 全文匹配 | 保留许可；两个 `*_official.obj` 的合并/转换来源及权利仍未确认，不能由拆分网格的匹配一并推定 |
| D435i：`realsense2_description/`、`mujoco_fr3/assets/d435*` | 9 个拆分 OBJ 与固定 Menagerie/kyokochi `realsense_d435i/assets/` 内容匹配；对应模型许可与 `licenses/D435i_Apache-2.0.txt` 全文匹配，声明 Copyright 2023 Intel Corporation | 保留 Intel 声明与许可；`d435_official.obj` 和 URDF 侧资产的对应关系、转换和权利仍需补齐 |
| CNC 挂载件：`models/FR3_D435i_mount.*`、`mujoco_fr3/assets/FR3_D435i_mount.obj` | README 说明 CAD 装配尺寸；STL SHA256 为 `185a904aa4c2d679e9caa0d85b54dc07a9bd21d3e2a1b1487d26302afc16c913` | CAD 作者、来源、再分发许可和 STL/OBJ 转换关系 |
| `asset/robot_surfaces/{piper,franka}/*` | 固定姿态采样的 NPZ/PLY；`metadata.json` 记录 URDF 与部分组合来源哈希 | 继承所有参与网格的许可；未知授权不因采样成为点云而消失 |

固定 FR3 描述共有 88 个跟踪文件。主仓库现在固定 `robot_control` 的公开分支提交 `6016dd5f20d2146a28e425ce77dfb9430f1173`；该提交把官方 URDF 中原本指向不存在的 `../meshes/franka_collision/{link0..7,hand}.stl` 改为同一 vendor bundle 中已存在且已核对的 `../mujoco_fr3/assets/link0..7.stl` 与 `fr3_hand_collision.stl`。9 个 replacement 文件的 Git blob 与固定 Menagerie/Panda 来源见下表及 [fr3-assets.json](provenance/fr3-assets.json)，没有添加未知来源网格。独立检查确认 URDF 的 10 个 mesh collision 引用（另含自定义挂载件）均解析到仓库内文件；这修复了路径可用性，但不替代自定义挂载件的授权核验。

Stage 1 表面采样只使用 visual；Stage 3 FR3 导出使用单独的 MJCF。更新 URDF 后重新生成了 Franka surface pyramid，视觉网格数量和点云范围保持一致，新的 metadata 记录当前 URDF SHA256；历史结果若绑定旧 URDF SHA256，必须按交付校验规则重新生成，不能手改哈希。

### FR3 网格内容匹配记录

2026-10-14 核查了固定控制子模块的 `mujoco_fr3/assets/` 全部 64 个文件。完整树响应与 Git blob 哈希用于比较，没有下载上游大网格。固定比较来源如下；这些提交包含匹配内容，不代表已证明原始复制或导入提交。

| 比较来源 | 固定提交 | Git tree SHA |
| --- | --- | --- |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/0059d4335f8156206f63a35662313385f7ad6d74) | `0059d4335f8156206f63a35662313385f7ad6d74` | `693a847fb65ac42afdd9c8c820fd06ebd5e1d33b` |
| [kyokochi/mujoco-model](https://github.com/kyokochi/mujoco-model/tree/cc3868ec21bae324e6dd3b01ea185b6e6f0fa172) | `cc3868ec21bae324e6dd3b01ea185b6e6f0fa172` | `9e5e1583490bb1a613b681cf603cf13c9a27d7a7` |

| 本地网格组 | 完全一致 | 仅 CRLF 转 LF 后匹配 | 上游对应组 |
| --- | ---: | ---: | --- |
| `link*` OBJ/STL，42 个 | 8 | 34 | Menagerie `franka_fr3/assets/` 同名文件 |
| `fr3_hand_{0..4}.obj`，5 个 | 0 | 5 | Panda `hand_{0..4}.obj` |
| `fr3_finger_{0..1}.obj`，2 个 | 0 | 2 | Panda `finger_{0..1}.obj` |
| `fr3_hand_collision.stl`，1 个 | 1 | 0 | Panda `hand.stl` |
| `d435i_{0..8}.obj`，9 个 | 0 | 9 | `realsense_d435i/assets/` 同名文件 |

逐文件本地 blob SHA、换行归一后 SHA、上游路径和许可比对保存在 [fr3-assets.json](provenance/fr3-assets.json)。共 9 个文件逐字节一致，50 个仅 CRLF/LF 不同；没有修改本地网格或重新授权。三个模型范围的许可全文仅归一换行后与现有保留文件相等。FR3 许可含 BSD 附段，不能用通用 Apache 文本替换。已检查的仓库根目录和模型目录没有独立 `NOTICE*` 或 `COPYRIGHT*` 文件，这不排除嵌入式署名或其他通知义务。

仍未匹配的 5 个文件是 `FR3_D435i_mount.obj`、`d435_official.obj`、`fr3_finger_official.obj`、`fr3_hand_official.obj` 和 `material.mtl`。核查未覆盖这些自定义/转换文件、生成点云、学校标志或主项目。内容匹配补充了来源证据，不能作为这些未核查文件的再分发许可。

### 原始 D435 DAE 与独立通知

随后核查了 FR3 的另外三个网格及 Piper 的 D435 DAE。以下两份固定文件逐字节匹配官方 [realsenseai/realsense-ros](https://github.com/realsenseai/realsense-ros/tree/7a87097c499e3b0b69518d54a8e7a9ac1924e87c) 的 `realsense2_description/meshes/d435.dae`，比较提交为 `7a87097c499e3b0b69518d54a8e7a9ac1924e87c`，完整 Git tree 为 `5534b3aa6caba676ced9ec9825e3a3a06e4cf63c`：

- `robot_control@6016dd5f20d2146a28e425ce77dfb9430f1173` 的 `vendor/fr3_d435i/realsense2_description/meshes/d435.dae`。
- `piper_isaac_sim@8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2` 的 `realsense2_description/meshes/d435.dae`。

两者 Git blob 均为 `0ed32368fc5ba6912be530395a72553de9f21d9e`，SHA256 均为 `42f3b66f47a1f8f425a2e4dc07c1d9c283183167d8441f520a15623d98f9bf78`。官方 `realsense2_description/package.xml` 明确声明 Apache License 2.0，该包及其 meshes 目录没有更近的 LICENSE / NOTICE / COPYING / README 条款文件。

本仓库在 [notices/](notices/README.md) 保留了该固定来源的原版 LICENSE 和 NOTICE，文件 Git blob 与官方相同。NOTICE 是上游仓库的软件组件总清单，不能据此推断清单中所有软件都随这一个网格分发。现有 `D435i_Apache-2.0.txt` 对应的是分材质的 D435i 模型，与这里的原始 DAE 官方声明不同，不能互相替代。

[urdf-assets.json](provenance/urdf-assets.json) 保存了此次四个网格的来源比对。两个自定义挂载件 OBJ / STL 未匹配官方 RealSense 或 Franka 树；九个缺失碰撞路径没有本地 blob 可供比较。此次核查不清除支架、自定义安装件、转换 OBJ 或混合来源点云的授权问题。

完成许可核验时，每组材料至少记录：上游 URL 和固定提交、原始/转换后文件对应关系、完整许可及署名、修改说明、权利方授权证据。若选择排除未获授权的网格，需同时处理派生点云和引用这些文件的运行路径，并重新生成、核验配准模板；不能只删除 NOTICE 或对文件换名。历史内容是否清理需由作者单独决定。

## 数据、输出与凭据

用户自行准备任务视频及模型权重，并遵守其访问和使用条款。`data/raw/`、`model/` 中实际权重、推理缓存和运行结果默认忽略；本项目不随仓库发布数据集、checkpoint、Hugging Face token 或远端服务凭据。推理 manifest 中 checkpoint 哈希用于追溯，不授予模型或数据许可。

例外：历史上已提交 `runs/sim_processed_stage3_all/piper/put_emergency_stop_button_in_green_storage_box/codex_exact/` 的 policy、视频和 review；Git 忽略规则不会从当前版本或历史中移除它们。正式发布前须核对视频画面中第三方资产与任务素材的许可，并确认该运行产物可以公开。学校 logo 和论文图片分别遵循学校品牌规则与论文作者/出版方授权，不能由主代码 Apache-2.0 许可证自动覆盖。
