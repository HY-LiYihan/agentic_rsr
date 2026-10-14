# VGGT-Ω Backend

这里放 VGGT-Ω 的远端客户端和本地推理实现。两种实现必须提供相同的上层接口和结果 schema。

- `remote_client.py`：从 `.env` 读取远端服务地址，不硬编码 IP。
- `local_inference.py`：从 `VGGT_OMEGA_MODEL_DIR` 加载本地模型。
- 模型权重放在 `model/vggt_omega/`，不提交 Git。

默认 `VGGT_OMEGA_BACKEND=local`、`VGGT_OMEGA_DEVICE=cuda`。本地适配器直接读取 32 张 512×288 RGB，不调用上游可能裁剪宽图的预处理器；返回数组必须严格对齐原图，不缩放深度来修补尺寸。模型首次调用加载，`close()` 释放；安装和来源见 `docs/LOCAL_INFERENCE.md`。

输出至少包含深度、内外参、点云/点云重建所需数据、单位、坐标系、backend 和模型版本。
