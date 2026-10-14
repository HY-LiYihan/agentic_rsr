# Franka Stage 1：从视频到初版 `.blend`

以 `data/raw/frank/close_drawer.mp4` 为例。文件名是任务标识；manifest 记录原视频路径、SHA-256、真实抽帧索引及远端/本地模型信息。原视频不复制到处理目录。

1. `uv run real2sim-stage1-infer-batch` 串行均匀抽取 32 帧，缩放补边到 512×288，依次调用 VGGT-Ω（32 帧相机、深度和置信度）及 SAM3（32 帧 `robotic arm` 掩码）。此时的 `depth/`、`camera.json` 均为**未定尺度的 VGGT 单位**；原始深度和置信度作为重跑配准的输入缓存保留。
2. `uv run real2sim-stage1-init-batch --rotations 48` 将 32 帧机械臂掩码内的深度反投影到 VGGT 世界；与固定关节姿态、米制 FR3 URDF 的表面点云对齐中心及粗尺度，按 32 帧综合评分 48 个确定性的 SO(3) 初值。四个参考序号为 0、10、21、31，但**没有留出**作当前算法的独立验收。
3. `uv run real2sim-stage1-franka-final` 对每任务的第一名初值依次做 5000、10000、20000 点 Sim(3) ICP，以及 ±15°、8°、4°、2° 的旋转搜索和平移/尺度再调整。结果记于 `runs/franka_stage1_final_48/<任务>/result.json`；它是**可丢弃的优化诊断**，不是 Stage 2 的放行报告。以 URDF 米制几何求得 `s`（VGGT 单位/米），深度、世界到相机外参平移和机械臂基座平移均除以 `s`；旋转及机器人真实尺寸不缩放。
4. `uv run real2sim-stage1-franka-deliver --prune` 检查 32 帧米制数据与原始数据/尺度一致、模板 URDF 及配准结果相符；以 FR3 根部为交付世界原点及坐标轴，同步重基 32 帧相机外参，基于固定关节姿态的 URDF **visual** 网格生成一个 FR3，并以 0、10、21、31 帧生成四个参考相机。将四张 RGB 打包进 `.blend`；保存、重新打开，核查四台相机、机器人变换、画面参数和内嵌图像。成功后只清理可重建的 `raw_vggt.npz`、32 张深度伪彩和 32 张掩码预览。单任务可加 `--only close_drawer`；失败不清理推理输入。设置 `BLENDER_BIN` 可指定 Blender 程序。

正式任务目录为 `data/processed/franka/stage1/close_drawer/`：

| 文件 | 保留理由 |
| --- | --- |
| `manifest.json` | 原视频哈希、抽帧索引、推理版本和推理状态 |
| `frames/`、`sam3_masks/` | 32 帧 RGB、机器人掩码；配准和复查的基础数据 |
| `depth/`、`camera.json`、`depth_conf.npz` | VGGT 单位的输入缓存，用于可重复地重跑配准；不是米制交付 |
| `depth_metric/`、`camera_metric.json` | 32 帧米制深度及相机；相机外参处于 URDF 机械臂根部坐标系，根部即世界原点 |
| `registration.json`、`stage1_delivery.json` | 精简后的配准、尺度、逐帧指标、交付坐标版本与状态；前者保留重基前 VGGT 米制基座位姿 |
| `reference/`、`quality_check.png` | 四张参考 RGB、四张 SAM3/机械臂投影叠图及单张汇总图 |
| `initial.blend` | 四个视频参考相机、固定姿态 FR3 的可独立打开场景；保留 `robot_control` FR3 URDF visual 的白/黑/按钮/腕部相机材质颜色 |

`.blend` 中的四个相机是**同一任务视频的四个重建视角，不是四台实体相机**。根对象位于 `(0, 0, 0)`、旋转为单位阵；`camera_metric.json` 中全部 32 帧均使用相同的机器人根部坐标，不是只改 `.blend` 的四个相机。原始 `camera.json`、`depth/` 与运行目录的相机缓存仍处于 VGGT 世界坐标以便复算。配准和评分使用相同的 32 帧；尽管尺度已经由 URDF 锚定为米，`registration.json` 和 `stage1_delivery.json` 仍标记 `candidate_unverified`、`accepted=false`。当前未提供独立配准验收或场景深度 MAE 门槛结果，不允许据此宣称 Stage 1 `passed` 或启动 Stage 2。

此前机械臂整体合并为一个无材质 OBJ，Blender 显示为灰白。现在按 URDF 材质颜色拆分 visual、各色合并后写入 Blender 材质；几何、根位姿和相机不变。打开 `.blend` 后使用 **材质预览**或实体模式的「材质」颜色；离线渲染还需要 Stage 2 的灯光/世界照明。

`runs/initial_poses/`、`runs/franka_stage1_final_48/` 中的 48 候选、优化过程与总览仅用于调试；正式交付不复制其点云、每步截图及搜索历史。因当前注册入口仍依赖原始 `depth/`、`camera.json` 和初始化缓存，**不要直接删除这些输入或整个 `runs/initial_poses/`**。
