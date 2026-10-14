# Stage 2：分阶段 Codex 迭代建模

公开的阶段顺序见 [WORKFLOW.md](../../../docs/WORKFLOW.md)，输入、交付和验收
字段见 [DATA_CONTRACT.md](../../../docs/DATA_CONTRACT.md)。这是当前 Codex 实现
的说明；论文主实验使用的 DSH/Claude 路径及完整实验输入尚未在本仓库发布。

Stage 2 现在采用一个完整的 `plan → refinement` 闭环：Plan Agent 只看四张
RGB 和任务描述；Refinement Agent 接收原始 `initial.blend`、Plan、四相机状态、
参考 RGB 和当前场景 RGB 预览。它先建立几何，再增加材质/光照，程序最后统一
用 `Depth MAE + Lab ΔE76` 验收。Agent 不接收 depth 图、depth 数组或误差图。
相机、Piper 机械臂、夹爪、相机安装件和其他机器人附属物由程序冻结，不能被
Agent 修改。

Stage 2 Codex turn 默认使用 `gpt-6-astra`、`high` reasoning effort、
`full-access` sandbox 和 `auto_review` approval。Python SDK 通过本机 Codex
app-server 运行，使用与 CLI 相同的 Codex 配置、工作目录和 MCP 配置；运行记录
会写入实际使用的模型和权限参数。
`REAL2SIM_CODEX_MODEL` 可以覆盖模型名称；需要本机 Codex 身份认证及所选模型
访问权限。本地感知模式不会使这些 Agent 调用离线。

## 运行指定场景

Stage 1 生产交付仍为 `candidate_unverified`、`accepted=false`，因为拟合与评分
共用 32 帧。当前分阶段入口读取交付清单但不要求 `accepted=true`，可用于探索性
重建；它不会验证调用者是否人工检查过，也不证明配准精度。先检查源哈希、
四视角、metric 相机与 depth，再使用独立输出目录：

```bash
real2sim-stage2-agent \
  --phase all \
  --input data/processed/piper/stage1/put_emergency_stop_button_in_green_storage_box \
  --output runs/stage2/put_emergency_stop_button_in_green_storage_box/20260924_01 \
  --task "put emergency stop button in green storage box" \
  --max-rounds 10
```

单独运行 Plan：

```bash
real2sim-stage2-agent --phase plan \
  --input data/processed/piper/stage1/put_emergency_stop_button_in_green_storage_box \
  --output runs/stage2/put_emergency_stop_button_in_green_storage_box/20260924_01
```

单独运行 Refinement：

```bash
real2sim-stage2-agent --phase geometry \
  --input data/processed/piper/stage1/put_emergency_stop_button_in_green_storage_box \
  --output runs/stage2/put_emergency_stop_button_in_green_storage_box/20260924_01 \
  --plan runs/stage2/put_emergency_stop_button_in_green_storage_box/20260924_01/plan/plan.md \
  --max-rounds 10
```

`--phase geometry --max-rounds 0` 在已有 Plan 的前提下只执行 Refinement 基线
inspect/render/evaluate，不启动 Refinement Codex。`--phase all --max-rounds 0`
仍会启动 Plan Agent；不能把它当成完全离线检查。

不写 `--phase` 的旧式 `real2sim-stage2-agent --input ...` 通过 `agent` 子命令进入
相同的 `run_phases()`。需要 `--plan` 或 `--resume` 时显式使用
`--phase geometry`；`--resume` 不支持 `--phase all`。

如果手动在 CLI 中复现实验，应显式指定同一模型和 reasoning effort：

```bash
codex -m gpt-6-astra -c model_reasoning_effort=high
```

```bash
real2sim-stage2-agent \
  --phase geometry \
  --input data/processed/piper/stage1/put_emergency_stop_button_in_green_storage_box \
  --output runs/stage2/put_emergency_stop_button_in_green_storage_box/20260924_baseline \
  --plan runs/stage2/put_emergency_stop_button_in_green_storage_box/20260924_01/plan/plan.md \
  --max-rounds 0
```

## 批量运行 Franka/Piper

从 `data/processed/<robot>/stage1/<task>` 自动发现所有 Stage 1 交付，并将通过验收的
最终场景写入对应的 `data/processed/<robot>/stage2/<task>`；运行日志、中间过程和
历史结果保存在 `runs/stage2/`，可以运行：

```bash
PYTHONPATH=src python scripts/run_stage2_batch.py \
  --robots franka piper \
  --workers 2 \
  --max-rounds 10
```

模型消融应使用独立输出根目录，避免覆盖已交付的 Stage 2：

```bash
PYTHONPATH=src python scripts/run_stage2_batch.py \
  --robots franka piper \
  --workers 9 \
  --max-rounds 10 \
  --model gpt-5.6-sol \
  --output-root runs/stage2/ablations/gpt-5.6-sol \
  --manifest runs/stage2/metadata/ablation_gpt-5.6-sol_manifest.json
```

`--model` 通过每个子进程的 `REAL2SIM_CODEX_MODEL` 传入 Plan 和 Geometry 两个
Codex 会话；`--output-root` 只改变实验输出位置，不会修改 Stage 1 或已有的
`data/processed/<robot>/stage2` 交付。

批处理默认拒绝覆盖已有的非空 Stage 2 目录。当前验收要求为四视角平均 Depth MAE
小于 0.15 m、Lab ΔE76 小于 15，且每视角仍需满足原有单视角限制。
若之前的任务中途停止，可用 `--resume --max-rounds 10` 跳过已交付的场景；已有
Plan 的任务会复用每轮候选场景并从最后完成的轮次继续，尚未生成 Plan 的任务会从
Plan 阶段开始。历史候选若已满足新阈值，会重新验收并直接交付，不会重复运行 Plan
或额外消耗轮次。
续跑请用 `--manifest runs/stage2/metadata/resume_manifest.json` 保存独立状态，
每场景的续跑日志写入 `resume_process.log`，原始日志保留。
可运行 `python scripts/organize_stage2_results.py --prune` 刷新
`runs/stage2/metadata/results_index.json` 和 `STATUS.md`，清理已结束场景中
已评估轮次的 Blender 备份及已交付场景的临时预览 Blend；每轮正式候选、
渲染、评估和会话记录均保留，运行中的任务不清理。不带 `--prune` 只刷新索引。
全局批次状态写入
`runs/stage2/metadata/batch_manifest.json`，每个任务自己的 Codex/Blender 输出仍
完整保存在对应任务目录；每个任务的标准输出和错误输出保存在
`<task>/batch_process.log`。正式启动前可以使用 `--dry-run` 只检查任务列表和输出路径。

## 交付目录

历史本机实验的 18 个验收场景使用如下布局（场景文件被忽略，不随公开 checkout 提供）：
`data/processed/<robot>/stage2/<task>/geometry/final.blend`，对应的
`geometry/stage2_manifest.json` 是交付入口。未通过或被新重跑版本替换的历史目录
放在 `runs/stage2/archive/`，不会混入可供 Stage 3 消费的 `data/processed` 目录。
`runs/stage2/metadata/delivery_layout.json` 汇总当前 18 个交付场景。
实验用的基线和逐轮指标在 `runs/stage2/metadata/experiments/`，其中
`round_metrics.csv` 保留原始运行和独立重跑的完整轨迹。
同时启动两批时，用 `--manifest` 为后启动的批次指定单独的清单路径，避免覆盖
正在更新的另一份批次状态。

可对现有已交付场景的最终验收 RGB 渲染补充四视角 SSIM：

```bash
python scripts/export_stage2_ssim.py
```

结果写入 `runs/stage2/metadata/ssim/`：`scene_ssim.csv` 和 `SSIM.md` 提供
每个场景四个相机及均值，`<robot>/<task>/report.json` 保留对应的参考图和渲染图路径。
SSIM 使用整张 RGB 图经 OpenCV RGB2GRAY 转换的灰度图、11×11 高斯窗口（σ=1.5），越高越相似；
它是补充分析指标，不改变 Depth MAE/Lab 的验收条件。

也可以单独调用底层工具：

```bash
real2sim-stage2 inspect --scene path/to/scene.blend --output inspect.json
real2sim-stage2 render --scene path/to/scene.blend --output render/
real2sim-stage2 render-rgb --scene path/to/scene.blend --output rgb_render/
real2sim-stage2 evaluate --reference path/to/stage1 --render render/ --output evaluation/
real2sim-stage2 evaluate-depth --reference path/to/stage1 --render render/ --output evaluation/
```

## 每轮流程

1. Plan Agent 接收四张 `reference/frame_*.jpg`，只写 `plan/plan.md`。
2. Geometry 程序从 `initial.blend` 生成只属于本次 run 的
   `initial_with_default_light.blend`，加入可替换默认灯光后再记录四相机/Piper
   冻结签名并做深度基线渲染。
3. Refinement Agent 每轮收到原始 Blend 路径、Plan、四相机状态、四个方向的参考 RGB、当前场景 RGB 预览，以及只含整体 Depth MAE/Lab ΔE76 的 scalar 报告。
4. Agent 必须先自己执行 `real2sim-stage2 render-rgb` 到 `agent_self_render/`，先修几何再修材质/光照；外层程序执行脚本、保存 `candidate.blend`、冻结检查、渲染和最终双指标。
5. 每轮都保留独立的 `geometry/round_NNN/`，包含 Agent 自己生成的 `agent_self_render/`，并通过前不断使用同一个临时 Codex thread。
6. 每轮按四个视角的整体 Depth MAE 和 Lab ΔE76 联合验收，直到通过或达到轮数上限。

原始 Stage 1 文件永不覆盖。每轮保存 `candidate.blend`、修改脚本、深度渲染、
误差图、`report.json` 和 `round_report.json`。会话内容保存到对应 phase 的
`conversation.jsonl`，并在结束时归档/清理 ephemeral thread。Agent 不能通过
自报结果放行；达到轮数上限、工具失败或没有候选脚本时会保留诊断并返回失败状态。

Stage 2 启动时会在对应 run 目录生成
`initial_with_default_light.blend`。它是从 Stage 1 `initial.blend` 创建的私有副本，
原始文件保持不变；副本包含可替换的 `Stage2_Lighting` 集合，Agent 可以调整、删除或
替换其中的默认中性灯光。每轮候选都从这个副本或上一轮候选继续，不会回写 Stage 1。

已有非空 `geometry/` 必须显式使用 `--phase geometry --resume`，或选择新的输出目录。
恢复前会核对 `geometry/input_provenance.json`：Stage 1 场景和交付清单、四张参考
RGB/深度、Plan、评估设置和渲染/评估实现必须与首次运行相同；已准备场景、检查记录、
基线与候选渲染也需匹配保存的 SHA256。变化或缺失会明确拒绝续跑，原交付不会改写。
恢复时重新计算视觉指标，旧 `report.json` 的通过标记不能直接放行。较早的结果仍可读取，
但没有这些来源记录的旧 run 不能安全续跑，应保留原目录并在新目录重新运行。

## 标准文件布局

`--output` 始终指向一次完整运行的根目录：

```text
runs/stage2/<task>/<run_id>/
├── run_manifest.json                 # 整次运行的索引和最终结果入口
├── plan/
│   ├── prompt.md                     # Plan Agent 实际收到的提示词
│   ├── plan.md                       # 任务计划，含每个物体的尺寸预估
│   ├── session.json                  # Plan 会话、模型、权限和校验状态
│   └── conversation.jsonl            # Plan 对话和命令轨迹
└── geometry/
    ├── initial_with_default_light.blend
    ├── inspect.json
    ├── frozen_signature.json
    ├── input_provenance.json         # 续跑所需输入和基线资产 SHA256
    ├── baseline_render/              # 初始四视角 RGB、depth 和 render manifest
    ├── baseline_evaluation/          # 初始双指标和误差预览
    ├── round_001/                    # 每一轮完全独立保存
    │   ├── prompt.md
    │   ├── current_rgb/
    │   ├── agent_self_render/
    │   ├── edit_scene.py
    │   ├── candidate.blend
    │   ├── render/
    │   ├── evaluation/
    │   └── round_report.json
    ├── ...
    ├── final.blend                   # 通过当前程序视觉门槛的最终场景
    ├── stage2_manifest.json          # 最终场景、哈希和验收指标
    └── session.json                  # Refinement 全过程汇总
```

尺寸预估位于 `plan/plan.md` 的 `Approximate dimensions` 章节。每个任务物体
都要给出四舍五入的长 × 宽 × 高、单位、估计依据和不确定性；它是建模初始尺度
的依据，不会被当作精确测量值。

最终交付以 `geometry/stage2_manifest.json` 为入口：其中的 `scene` 指向
`geometry/final.blend`，`scene_sha256` 用于确认文件没有被替换，`evaluation`
记录四个相机的 Depth MAE、Lab ΔE76、覆盖率和误差图路径。顶层
`run_manifest.json` 只做跨阶段索引，方便程序或后续 Stage 3 找到 Plan、会话、
最终 Blend 和验收清单。
# 无初始机器人校准消融

`scripts/run_stage2_uncalibrated.py --output runs/stage2/ablations/<name> --piper-asset <piper.blend> --franka-asset <franka.blend>`
从两份独立机器人资产生成统一的四视角相机初值，不读取 Stage1 配准或相机参数。
机器人本体、尺寸和姿态保持冻结；Agent 可根据四张 RGB 调整相机位置、旋转、焦距和偏移。
其余采用正常 Plan、RGB 自检、几何和外观联合验收，GPT-6 Astra high、18 并发、10 轮。
米制深度仍来自正常 Stage1，仅作为后台伪参考验收，故本实验消融的是初始化而非评估标尺。
输入、输出、资产哈希、提示词及批次状态均保存在独立 output 下，不修改正式数据。
`--prepare-only` 仅构造输入。`REAL2SIM_UNCALIBRATED=1` 和
`REAL2SIM_REFINEMENT_PROMPT` 必须一起传给续跑进程；不设置时正常流程不变。
