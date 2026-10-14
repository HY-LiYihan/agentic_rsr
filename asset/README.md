# 固定姿态机械臂表面点云

资产与派生文件按作者决定通过 Hugging Face 发布；目标地址及迁移状态见
[资产分发说明](../docs/ASSET_DISTRIBUTION.md)。当前 Git 中已有的点云先保留，
待可下载的发布包验证后迁移；主代码 Apache-2.0 许可证不覆盖这些派生资产。

`robot_surfaces/{piper,franka}/` 各保存 `surface_40000.{npz,ply}`、`surface_10000.{npz,ply}`、`surface_5000.{npz,ply}` 及 `metadata.json`。NPZ 中数组键为 `points`，形状为 `(N, 3)`，单位米；坐标系分别是 Piper 的 `base_link` 和 FR3 的 `fr3_link0`。PLY 用于点云查看器。低密度点云从同一份 40000 点中确定性抽取，5000 点是 10000 点的子集；六份点云资产纳入 Git，不含原始视频或模型权重。

从 `external/robot_control` 中 Piper 的 `piper_with_gripper_description.xacro` 与 FR3 的 `fr3_d435i_official.urdf` 的 **visual 网格**计算前向运动学，把各个 link 固定到以下姿态，再把整机视作刚体，按表面积采样并以 3 mm 网格去重：

- Piper：`joint1`–`joint6` 为 `[0°, 30°, −45°, 0°, 60°, 0°]`；夹爪总开口沿用 0.07 m（两侧各 0.035 m）。从 AGX 手臂/夹爪 Xacro 组合控制仓库固定的 Isaac Piper V100 D435 URDF 中的 D435、打印支架 visual（位于嵌套 `vendor/piper_isaac_sim`），相机/支架的安装变换采用 `robot_control` 的 MuJoCo 后端定义。`metadata.json` 记录所有来源哈希和组合哈希；旧版 11 visual 机械臂点云不可用于带腕部相机的视频。
- FR3：`fr3_joint1`–`fr3_joint7` 为 `[0°, -45°, 0°, -135°, 0°, 90°, 45°]`；手指各开到 0.04 m，包含腕部相机与安装件。

**注意：FR3 的目标姿态不同于控制子模块当前 MuJoCo `home` 的 `[0, 0, 0, -1.57, 0, 1.57, 0.785]` 弧度（第 2、4 关节不同）。**生成点云不会更改子模块的 `home`；配准时须确认使用的是哪一套姿态。点云本身不会自动修改 VGGT 深度单位，完成静态机械臂的 Sim(3) 配准后，才可用拟合的尺度把相机、深度和机器人一起转换为米；精度仍需独立核验。

在项目根目录运行 `uv run real2sim-robot-surfaces` 可以重新生成；可选 `--robot piper`、`--sizes 40000 10000 5000`、`--voxel 0.003`。网格去重并不等于填补或布尔并合网格中的真实缝隙。
