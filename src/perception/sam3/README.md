# SAM3 Backend

这里放 SAM3 的远端客户端和本地推理实现。两种实现必须提供相同的上层接口和结果 schema。

- `remote_client.py`：从 `.env` 读取远端服务地址，不硬编码 IP。
- `local_inference.py`：从 `SAM3_MODEL_DIR` 加载本地模型。
- 模型权重放在 `model/sam3/`，不提交 Git。

默认 `SAM3_BACKEND=local`、`SAM3_DEVICE=cuda`，本地实现使用固定版本 Ultralytics。`segment_instances()` 返回原图网格上的逐实例 mask 和分数；`segment()` 合并实例供 Stage 1 使用；`InstanceSegmenter` 按既有分数/面积规则选单实例供 SIM 使用。模型和同一图像的特征按需复用，`close()` 释放。安装及 CLIP 分词器预置见 `docs/LOCAL_INFERENCE.md`。

输出至少包含 mask、置信度、提示信息、图像坐标、backend 和模型版本。
