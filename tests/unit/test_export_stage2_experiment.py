import csv
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/export_stage2_experiment.py"
SPEC = importlib.util.spec_from_file_location("stage2_experiment_export", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")


def _report(geometry, round_number, depth, lab, limit=0.1):
    render = geometry / ("baseline_render" if round_number == 0 else f"round_{round_number:03d}/render")
    report_path = (geometry / "baseline_evaluation/report.json" if round_number == 0 else
                   geometry / f"round_{round_number:03d}/evaluation/report.json")
    _write_json(report_path, {
        "render": str(render), "depth_mae_mean_m": depth, "lab_delta_e76_mean": lab,
        "passed": depth < limit and lab < 15,
        "views": [{"depth_passed": True, "lab_passed": True}] * 4,
        "thresholds": {"depth_mae_mean_m": limit, "lab_delta_e76_mean": 15},
    })
    return str(render)


def _delivery(geometry, render):
    final = geometry / "final.blend"
    final.write_bytes(b"blend")
    _write_json(geometry / "stage2_manifest.json", {
        "accepted": True, "scene": str(final), "evaluation": {"render": render},
    })


def test_export_tracks_selected_earlier_round_and_separate_rerun(tmp_path):
    data = tmp_path / "data/processed"
    runs = tmp_path / "runs/stage2"
    original = data / "franka/stage2/fruit/geometry"
    original.mkdir(parents=True)
    _report(original, 0, 0.9, 30)
    accepted = _report(original, 1, 0.12, 10)
    _report(original, 2, 0.13, 9)
    _delivery(original, accepted)

    earlier = data / "piper/stage2/box/geometry"
    earlier.mkdir(parents=True)
    _report(earlier, 0, 0.8, 20)
    _report(earlier, 1, 0.4, 17)
    rerun = runs / "box/20260925_rerun_01/geometry"
    rerun.mkdir(parents=True)
    _report(rerun, 0, 0.8, 20, 0.15)
    selected = _report(rerun, 1, 0.14, 12, 0.15)
    _delivery(rerun, selected)

    output = tmp_path / "experiments"
    assert MODULE.export(data, runs, output) == (7, 2)
    with (output / "round_metrics.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    with (output / "scene_summary.csv").open(encoding="utf-8-sig", newline="") as file:
        summary = {row["task"]: row for row in csv.DictReader(file)}
    assert summary["fruit"]["selected_round"] == "1"
    assert summary["box"]["selected_run"] == "20260925_rerun_01"
    assert summary["box"]["rerun_evaluated_rounds"] == "1"
    fruit = [row for row in rows if row["task"] == "fruit"]
    assert fruit[1]["passed_at_evaluation"] == "False"
    assert fruit[1]["passes_current_criteria"] == "True"
    assert fruit[1]["selected_final"] == "True"
    assert fruit[2]["selected_final"] == "False"
    assert fruit[2]["depth_change_from_previous_m"] == str(0.12 - 0.13)
    box = [row for row in rows if row["task"] == "box"]
    assert [(row["run"], row["round"]) for row in box] == [
        ("original", "0"), ("original", "1"),
        ("20260925_rerun_01", "0"), ("20260925_rerun_01", "1"),
    ]
