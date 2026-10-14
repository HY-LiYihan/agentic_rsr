# VGGT-Ω Model Directory

把 VGGT-Ω 的 checkpoint、配置文件和本地缓存放在这里。真实文件不提交 Git；路径通过 `VGGT_OMEGA_MODEL_DIR` 配置。

默认文件为 `vggt_omega_1b_512.pt`，可放置或挂载到本目录。需先在 [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) 申请访问并接受非商业研究条款；推理不自动下载。

```bash
uv run --no-sync hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir model/vggt_omega
```

固定源码位于 `external/vggt_omega/`。详见[本地安装](../../docs/LOCAL_INFERENCE.md)和[许可说明](../../docs/THIRD_PARTY.md)。
