# SIM reset/refinement 实测（2026-09-24）

## 口径

- 外层流程：exact → randomized development；失败后 reset 新的 MuJoCo runtime，在同一 Codex 会话中生成新的完整 bundle，再从 exact 开始。只有开发 profile 全部通过，才锁定代码并进入独立随机 seed 的最终测试。
- Code as Policies 参考：`$HOME/code_as_policy/results/archive/legacy-runs-20260923/20260923-paper-suite-reachable/summary.json`，22 个任务中 10 个通过（45.5%），每类别仅 1 次；这是名义场景 smoke suite，**不是**同一随机化评测。
- 目前本项目的 `TabletopTask` 仅实现指定源/目标物体的 pick-and-place 子集，不能拿它对完整 22 类任务算相同成功率。

## 实际运行

| 场景及方法 | exact | randomized development | 修订次数 | 最终 held-out |
| --- | --- | --- | ---: | --- |
| 归档 `trial_0000.xml`：orange block → yellow block，初始 bundle | 通过 | `20260924` 失败 | 0 | 未运行 |
| 同一归档场景，Codex 重写后的 bundle | 通过 | `20260924` 失败 | 1 | 未运行（开发未通过） |
| 同一归档场景，向 Codex 附加公开 RGB 帧后的重写 bundle | 通过 | `20260924` 失败 | 1 | 未运行（开发未通过） |
| 标准 `piper_tabletop.xml`：blue block → green block，初始 bundle | 通过 | `20260930` 失败 | 0 | 未运行 |
| 同一标准场景，第一次修订 | 通过 | `20260930` 失败 | 1 | 未运行 |
| 同一标准场景，第二次修订 | 通过 | `20260930` 失败 | 2 | 未运行（开发未通过） |

归档 orange 任务是 Code as Policies 在**同一场景**已通过的任务。原始策略的匹配场景基线也为 exact 通过、随机化 `20260924` 失败。本次自动修订未改变结果。blue → green 的简单策略在无 Codex 决策的 10 个随机 seed（`20260930`—`20260939`）中通过 2/10；外层流程在预先选择的失败 seed 上修订两次，仍未通过。另一个较容易的 blue block → yellow bowl 单一 bundle 在 exact、`20260924`、`20260925` 均通过，不需要修订。上述样本量不足以估计整体成功率，也不能声称比 Code as Policies 更优。

## 失败归因

- orange 任务：随机化的 SAM3/腕部 RGB-D 曾将 orange block 估为 `(-0.204, 0.135)` m，仿真物体真实初始 XY 约为 `(-0.184, 0.006)` m；画面中相近颜色的碗占据中心，目标在视野边缘，导致错误检测。首次迭代的 survey 还触发 IK 无解，第一版修订继续使用误检坐标时双指未接触。将公开 RGB 帧直接附加到第二次复测的 Codex 修订输入后，新策略选择验证多视角一致性并安全停止，没有误抓，但也未完成任务。
- blue → green：绿色目标真实初始 XY 约 `(-0.128, 0.042)` m，公开初次定位为 `(-0.112, -0.276)` m，Y 方向误差约 318 mm。两版策略能重试或安全停止，但无法靠策略代码修复大幅感知误检；首版执行遇到 32 次工具调用上限。
- 此次还发现工具调用上限会阻断安全停止，已修复，使 `task.stop` 超额时仍可执行，并补单元测试。重写提示补充“不得使用下划线开头私有标识符”，避免 Codex 返回不符合 AST 规则的 bundle。修订输入现在包含最近两张公开 RGB 帧，而不是仅仅传文件路径。

## 可复核输出

- `/tmp/sim_compare/orange_matched_refinement_20260924/refinement.json`
- `/tmp/sim_compare/orange_matched_visual_refinement_20260924/refinement.json`
- `/tmp/sim_compare/blue_green_refinement_20260924/refinement.json`
- `/tmp/sim_compare/blue_baseline_20260924/evaluation.json`
- `/tmp/sim_scan/20260930/`—`/tmp/sim_scan/20260939/`

结论：reset → 失败反馈 → 同会话 Codex 改码 → 再测试的**机制已实跑**，但本轮没有使两个失败任务在 randomized development 上成功。下一步应先测 SAM3 检测置信度、多视角一致性、视野边缘裁切与 XY 定位误差，再在相同场景/seed、相同任务集合、锁定的 held-out 种子上重新比较；不能通过修改随机化强度或使用真值坐标获取虚假增益。

## 22 场景 exact 快速评测（2026-09-25）

使用 `scripts/evaluate_sim_exact_suite.py` 在归档的同一批 22 个 XML 场景中逐项 reset 并运行一个固定策略 bundle，保存终帧和动作记录；通过条件是程序正常完成、动作完成且原 Code as Policies `task_scoring.score_case` 判定物理目标达成。终帧拼图已人工查看，失败项同时经最终物体坐标和物理目标判定核查，不能仅凭有遮挡的单帧判断堆叠是否成功。

| 策略 | exact 通过 | 说明 |
| --- | ---: | --- |
| 初版通用策略 | 14/22（63.6%） | 第一轮；`/tmp/sim_exact_22_20260924/generic_v1/summary.json` |
| 修复匹配颜色指令解析后 | 15/22（68.2%） | 基于同一测试集失败反馈修订，**不是**独立留出集结果；`/tmp/sim_exact_22_20260924/generic_v2/summary.json` |
| 归档 Code as Policies | 10/22（45.5%） | 相同归档任务，但策略生成与执行协议不同，仅供探索性参考 |

修订版按场景组分别为 4/8、6/8、5/6；失败的 7 项均未达到物理目标。拼图保存在 `/tmp/sim_exact_22_20260924/generic_v2/final_contact_sheet.png`，每个 `trial_*/` 下有独立 `result.json` 和 `final.png`。此次策略为直接编写的 `scripts/policies/exact_suite/policy.py`，使用任务文字及 exact 公开坐标；**没有**逐任务启动 Codex 生成策略，也没有完整 Codex reset/refinement loop 或 randomized 通过率。不能将本轮 15/22 称为自动 agent 的最终胜率或相对基线的严格提升。

针对原先被评测器人为缩窄的 XY 边界拒绝的两项，已移除 `SuiteTools.robot_pick_place` 的额外范围判断，并将公开工作区改成与底层移动工具一致的 `[-0.3, -0.3]` 至 `[0.1, 0.3]`。在相同策略、原始场景和正确的 `robot_control` MuJoCo 环境下单独重跑，红碗任务的 3 次搬运动作均完成，但最终物理目标未达成；粉碗任务不再被人为边界拒绝，而是在第一次搬运中遇到 IK 不收敛。**两项仍为 0/2**；只重跑了这两项，没有重新声称完整套件的新通过率。动作日志及终帧位于 `/tmp/sim_exact_22_20260924/without_narrow_bound_3_11_valid_env/`。
