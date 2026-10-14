# SAM3 Model Directory

把 SAM3 的 checkpoint、配置文件和本地缓存放在这里。真实文件不提交 Git；路径通过 `SAM3_MODEL_DIR` 配置。

默认文件为 `sam3.pt`，可放置或挂载到本目录。需先在 [facebook/sam3](https://huggingface.co/facebook/sam3) 申请访问并接受 SAM License；推理不自动下载。不使用 SAM3.1 checkpoint。

```bash
uv run --no-sync hf download facebook/sam3 sam3.pt --local-dir model/sam3
```

固定的 Ultralytics 实现位于 `external/ultralytics/`，实现与权重适用不同许可。详见[本地安装](../../docs/LOCAL_INFERENCE.md)和[许可说明](../../docs/THIRD_PARTY.md)。
