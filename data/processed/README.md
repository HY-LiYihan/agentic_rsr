# Processed Data

固定姿态机器人点云现存于项目的 `asset/robot_surfaces/`，不放在此处；生成方式与约定见 `asset/README.md`。

存放抽帧与推理产物；生成数据不提交 Git。先执行 `uv run real2sim-stage1-infer-batch` 串行处理 `data/raw/frank` 和 `data/raw/piper` 中所有 MP4。FR3 放在 `franka/stage1/<视频名>/`，Piper 放在 `piper/stage1/<视频名>/`。

新推理每个视频保存 `frames/`（32 帧）、`depth/`（32 个 VGGT 单位 `.npy`）、`sam3_masks/`（32 个布尔 `.npy`）、`reference/`（序号 0、10、21、31 的四张参考图）、`camera.json`（VGGT 单位的 32 帧内外参）、`depth_conf.npz` 和 `manifest.json`。不再默认保存可重建的原始 VGGT 数组包或 64 张预览图。`inference_complete` 仅表示模型推理结束。Franka 和新姿态 Piper 配准/米制交付之后，增加 `depth_metric/`、`camera_metric.json`、`registration.json`、`stage1_delivery.json`、`quality_check.png`、四张配准叠图及 `initial.blend`。详见 `docs/STAGE1_FRANKA_DELIVERY.md` 和 `docs/STAGE1_PIPER_DELIVERY.md`。重跑已完成的视频可用 `--force`，重跑时 `.history/` 中归档旧结果；核验新视频交付后可清理这些过期备份。
