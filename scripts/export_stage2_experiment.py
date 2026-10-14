#!/usr/bin/env python3
"""Export Stage 2 baseline, per-round metrics, and selected deliveries."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from real2sim.stage2.evaluator import DEPTH_MAE_THRESHOLD_M, LAB_DE76_THRESHOLD


ROUND_COLUMNS = (
    "robot", "task", "run", "round", "depth_mae_m", "lab_delta_e76",
    "depth_change_from_previous_m", "lab_change_from_previous",
    "depth_reduction_from_baseline_pct", "lab_reduction_from_baseline_pct",
    "passed_at_evaluation", "passes_current_criteria", "selected_final", "report_path",
)
SUMMARY_COLUMNS = (
    "robot", "task", "selected_run", "selected_round", "baseline_depth_mae_m",
    "final_depth_mae_m", "depth_reduction_m", "depth_reduction_pct",
    "baseline_lab_delta_e76", "final_lab_delta_e76", "lab_reduction",
    "lab_reduction_pct", "original_evaluated_rounds", "rerun_evaluated_rounds",
    "final_report_path", "final_scene",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_passes_current(report: dict) -> bool:
    return (report["depth_mae_mean_m"] < DEPTH_MAE_THRESHOLD_M and
            report["lab_delta_e76_mean"] < LAB_DE76_THRESHOLD and
            all(view["depth_passed"] and view["lab_passed"] for view in report["views"]))


def _metrics(geometry: Path, robot: str, task: str, run: str) -> tuple[list[dict], dict | None]:
    baseline_path = geometry / "baseline_evaluation/report.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(baseline_path)
    paths = [baseline_path] + sorted(geometry.glob("round_[0-9][0-9][0-9]/evaluation/report.json"))
    delivery_path = geometry / "stage2_manifest.json"
    delivery = _load(delivery_path) if delivery_path.is_file() else None
    selected_render = delivery["evaluation"]["render"] if delivery and delivery.get("accepted") else None
    baseline = _load(baseline_path)
    previous = None
    rows = []
    for path in paths:
        report = _load(path)
        round_number = 0 if path == baseline_path else int(path.parent.parent.name.removeprefix("round_"))
        depth = report["depth_mae_mean_m"]
        lab = report["lab_delta_e76_mean"]
        rows.append({
            "robot": robot, "task": task, "run": run, "round": round_number,
            "depth_mae_m": depth, "lab_delta_e76": lab,
            "depth_change_from_previous_m": (previous["depth_mae_mean_m"] - depth
                                             if previous is not None else None),
            "lab_change_from_previous": (previous["lab_delta_e76_mean"] - lab
                                         if previous is not None else None),
            "depth_reduction_from_baseline_pct":
                100 * (baseline["depth_mae_mean_m"] - depth) / baseline["depth_mae_mean_m"],
            "lab_reduction_from_baseline_pct":
                100 * (baseline["lab_delta_e76_mean"] - lab) / baseline["lab_delta_e76_mean"],
            "passed_at_evaluation": report["passed"],
            "passes_current_criteria": _report_passes_current(report),
            "selected_final": report["render"] == selected_render,
            "report_path": str(path.resolve()),
        })
        previous = report
    if delivery and sum(row["selected_final"] for row in rows) != 1:
        raise ValueError(f"delivery not matched to a single report in {geometry}")
    return rows, delivery


def collect(data_root: Path, runs_root: Path, rerun_id: str = "20260925_rerun_01") -> tuple[list[dict], list[dict]]:
    rounds = []
    summaries = []
    for robot in ("franka", "piper"):
        stage2 = data_root / robot / "stage2"
        for task_dir in sorted(path for path in stage2.iterdir() if path.is_dir()):
            task = task_dir.name
            archived_original = (runs_root / "archive" / "pre_rerun_processed" /
                                 robot / task / "geometry")
            external_rerun = runs_root / task / rerun_id / "geometry"
            original_geometry = archived_original if archived_original.is_dir() else task_dir / "geometry"
            original, original_delivery = _metrics(original_geometry, robot, task, "original")
            rounds.extend(original)
            rerun, rerun_delivery = ([], None)
            rerun_geometry = task_dir / "geometry" if archived_original.is_dir() else external_rerun
            if rerun_geometry.is_dir():
                rerun, rerun_delivery = _metrics(rerun_geometry, robot, task, rerun_id)
                rounds.extend(rerun)
            selected = [row for row in original + rerun if row["selected_final"]]
            delivery = rerun_delivery if rerun_delivery and rerun_delivery.get("accepted") else original_delivery
            if delivery is None or not selected:
                raise ValueError(f"no accepted delivery for {robot}/{task}")
            if len(selected) != (bool(original_delivery) + bool(rerun_delivery)):
                raise ValueError(f"unexpected delivery count for {robot}/{task}")
            final = next(row for row in selected if row["run"] == (rerun_id if rerun_delivery else "original"))
            baseline = next(row for row in (rerun if rerun_delivery else original) if row["round"] == 0)
            if not final["passes_current_criteria"] or not Path(delivery["scene"]).is_file():
                raise ValueError(f"selected delivery is not valid for {robot}/{task}")
            depth_reduction = baseline["depth_mae_m"] - final["depth_mae_m"]
            lab_reduction = baseline["lab_delta_e76"] - final["lab_delta_e76"]
            summaries.append({
                "robot": robot, "task": task, "selected_run": final["run"],
                "selected_round": final["round"],
                "baseline_depth_mae_m": baseline["depth_mae_m"],
                "final_depth_mae_m": final["depth_mae_m"],
                "depth_reduction_m": depth_reduction,
                "depth_reduction_pct": 100 * depth_reduction / baseline["depth_mae_m"],
                "baseline_lab_delta_e76": baseline["lab_delta_e76"],
                "final_lab_delta_e76": final["lab_delta_e76"],
                "lab_reduction": lab_reduction,
                "lab_reduction_pct": 100 * lab_reduction / baseline["lab_delta_e76"],
                "original_evaluated_rounds": len(original) - 1,
                "rerun_evaluated_rounds": len(rerun) - 1 if rerun else 0,
                "final_report_path": final["report_path"],
                "final_scene": delivery["scene"],
            })
    return rounds, summaries


def _csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if row[key] is None else row[key]) for key in columns})


def _trajectory(rows: list[dict], field: str, digits: int) -> str:
    return " → ".join(f"{row[field]:.{digits}f}" for row in rows)


def export(data_root: Path, runs_root: Path, output: Path) -> tuple[int, int]:
    rounds, summaries = collect(data_root.resolve(), runs_root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    _csv(output / "round_metrics.csv", ROUND_COLUMNS, rounds)
    _csv(output / "scene_summary.csv", SUMMARY_COLUMNS, summaries)
    lines = ["# Stage 2 验收指标逐轮变化", "",
             f"场景：{len(summaries)}；运行（包含独立重跑）：{len(set((r['robot'], r['task'], r['run']) for r in rounds))}；"
             f"正式评估轮次：{sum(row['round'] > 0 for row in rounds)}。", "",
             "`round_metrics.csv` 是基线及每轮数据长表（轮次 0 = 基线），正数 `change_from_previous` 表示比上一轮下降。"
             "`scene_summary.csv` 每场景仅取实际交付的版本；对于独立重跑，轮次从 0 重新计算，"
             "不得把旧运行最后一轮与重跑第 1 轮视作连续迭代。", "",
             "当前验收：四视角平均 Depth MAE < 0.15 m，平均 Lab ΔE76 < 15；"
             "每视角深度 < 0.20 m、Lab < 30、覆盖率 ≥ 0.50。"
             "原始早期报告曾按 0.10 m 判定，故 `passed_at_evaluation` 与 `passes_current_criteria` 可能不同。", "",
             "注意：Depth MAE 对参考有效但渲染缺失的深度像素罚 1 m，"
             "基线通常因此较高；逐轮曲线不保证单调下降。"
             "原始记录中个别场景超过现在的 10 轮上限，是调整轮次上限之前留下的历史记录。", "",
             "## 结果解读", "",
             f"18 个交付版本的平均 Depth MAE："
             f"{mean(row['baseline_depth_mae_m'] for row in summaries):.4f} → "
             f"{mean(row['final_depth_mae_m'] for row in summaries):.4f} m；"
             f"平均 Lab ΔE76：{mean(row['baseline_lab_delta_e76'] for row in summaries):.2f} → "
             f"{mean(row['final_lab_delta_e76'] for row in summaries):.2f}。", "",
             f"在 {sum(row['round'] > 0 for row in rounds)} 次正式轮次中，"
             f"Depth MAE 有 {sum(row['depth_change_from_previous_m'] < 0 for row in rounds if row['round'] > 0)} 次回升，"
             f"Lab 有 {sum(row['lab_change_from_previous'] < 0 for row in rounds if row['round'] > 0)} 次回升；"
             "正数差值才代表本轮误差下降，不能假定每轮都变好。", "",
             "`stack_snacks` 原始运行第 5–9 轮 Depth MAE 约 0.4806 m，"
             "第 10 轮降至 0.2778 m 仍不达标；独立重跑第 3 轮降至 0.1445 m 才交付。"
             "`store_three_tools` 原始运行第 6 轮为 0.2502 m，"
             "独立重跑第 4 轮为 0.1144 m。两次运行之间的轮次必须分开统计。", "",
             "`store_fruit_in_bamboo_tray` 保留了原始运行第 11–13 轮的评估，"
             "但正式交付采用的是第 10 轮，不能把最后一次评分当成交付评分。", "",
             "## 各场景最终交付", "",
             "| 机械臂 | 场景 | 采用运行/轮次 | Depth 基线→最终 (m) | Lab 基线→最终 | Depth 下降 | Lab 下降 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for summary in summaries:
        label = "原始" if summary["selected_run"] == "original" else "重跑"
        lines.append(f"| {summary['robot']} | {summary['task']} | {label} {summary['selected_round']} | "
                     f"{summary['baseline_depth_mae_m']:.3f} → {summary['final_depth_mae_m']:.3f} | "
                     f"{summary['baseline_lab_delta_e76']:.2f} → {summary['final_lab_delta_e76']:.2f} | "
                     f"{summary['depth_reduction_pct']:.1f}% | {summary['lab_reduction_pct']:.1f}% |")
    lines.extend(["", "## 逐轮轨迹", "", "下列每行均从基线（第 0 轮）开始，后面按轮次顺序排列；"
                  "各轮精确值、与前轮差值及对应报告路径见 `round_metrics.csv`。", ""])
    for summary in summaries:
        for run in ("original", "20260925_rerun_01"):
            current = [row for row in rounds if row["robot"] == summary["robot"] and
                       row["task"] == summary["task"] and row["run"] == run]
            if not current:
                continue
            chosen = next((row["round"] for row in current if row["selected_final"]), None)
            label = "原始" if run == "original" else "重跑"
            note = f"，采用第 {chosen} 轮" if chosen is not None else "，本次未通过"
            if run == "original" and summary["selected_run"] != "original":
                note = "，后续独立重跑通过"
            lines.extend([f"### {summary['robot']}/{summary['task']} — {label}{note}", "",
                          f"Depth MAE (m)：{_trajectory(current, 'depth_mae_m', 4)}", "",
                          f"Lab ΔE76：{_trajectory(current, 'lab_delta_e76', 2)}", ""])
    (output / "TRAJECTORIES.md").write_text("\n".join(lines), encoding="utf-8")
    return len(rounds), len(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs/stage2")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/stage2/metadata/experiments")
    args = parser.parse_args()
    rows, scenes = export(args.data_root, args.runs_root, args.output)
    print(f"Exported {rows} baseline/round records for {scenes} scenes to {args.output}")


if __name__ == "__main__":
    main()
