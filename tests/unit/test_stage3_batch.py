import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from real2sim.stage3 import batch
from real2sim.stage3.batch import accepted_tasks, run_task


def test_accepted_tasks_ignores_unaccepted_deliveries(tmp_path):
    accepted = tmp_path / "piper/stage2/good/geometry/stage2_manifest.json"
    rejected = tmp_path / "franka/stage2/bad/geometry/stage2_manifest.json"
    for path, status in ((accepted, True), (rejected, False)):
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"accepted": status, "evaluation": {"passed": True}}))
    assert accepted_tasks(tmp_path, ["piper", "franka"]) == [accepted.parent.parent]


def test_batch_passes_task_plan_and_description_to_physical_agent(tmp_path, monkeypatch):
    task = tmp_path / "piper/stage2/place_object_on_table"
    geometry = task / "geometry"
    geometry.mkdir(parents=True)
    scene = geometry / "final.blend"
    scene.write_bytes(b"source")
    (task / "plan").mkdir()
    (task / "plan/plan.md").write_text("Tabletop must support the object.")
    images = []
    for index in range(4):
        image = task / f"camera_{index}.png"
        image.write_bytes(b"image")
        images.append({"render_rgb": str(image)})
    (geometry / "stage2_manifest.json").write_text(json.dumps({
        "scene": str(scene), "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "accepted": True, "evaluation": {"passed": True, "views": images},
    }))

    def fake_export(config):
        output = config.output
        output.mkdir(parents=True)
        xml = output / "scene.xml"
        xml.write_text("<mujoco/>")
        (output / "semantics.json").write_text(json.dumps({"objects": [
            {"role": "dynamic"}, {"role": "static"}, {"role": "visual_only"}]}))
        (output / "object_manifest.json").write_text(json.dumps({"objects": [
            {"name": "cup", "semantics": {"joint_type": "free"}}]}))
        return {"scene": str(xml)}

    def fake_agent(xml, hypotheses, output, task_context):
        assert "Task: place object on table" in task_context
        assert "Tabletop must support" in task_context
        assert "cup" in json.loads(hypotheses.read_text())["hypotheses"][0]["objects"]
        return {"verification": {"passed": True, "sim_static_passed": True,
                                 "sim_grasp_passed": True}}

    monkeypatch.setattr("real2sim.stage3.batch.export_blend", fake_export)
    monkeypatch.setattr("real2sim.stage3.batch.run_agent", fake_agent)
    result = run_task(task, tmp_path, "trial", runs_root=tmp_path / "runs/stage3/workspace")
    assert result["status"] == "audited"
    assert result["final_scene"] == result["scene"]
    assert str(result["scene"]).startswith(str(tmp_path / "runs/stage3/workspace/piper"))
    assert result["roles"] == {"visual_only": 1, "static": 1, "dynamic": 1, "operable": 0}


def test_parallel_batch_saves_each_completed_scene(tmp_path, monkeypatch):
    tasks = [tmp_path / "piper/stage2" / name for name in ("first", "second")]
    for task in tasks:
        task.mkdir(parents=True)
    barrier = Barrier(2, timeout=5)

    def fake_task(task, processed, run_id, run_physics, runs_root, semantics_dir,
                  reuse_semantics_from):
        barrier.wait()
        return {"robot": "piper", "task": task.name, "status": "audited"}

    monkeypatch.setattr(batch, "accepted_tasks", lambda *_: tasks)
    monkeypatch.setattr(batch, "run_task", fake_task)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(batch, "REPO_ROOT", tmp_path)
    assert batch.main(["--robot", "piper", "--workers", "2", "--run-id", "parallel_test"]) == 0
    manifest = json.loads((tmp_path / "runs/stage3/batch_parallel_test/batch_manifest.json").read_text())
    assert manifest["workers"] == 2
    assert {entry["task"] for entry in manifest["tasks"]} == {"first", "second"}


def test_batch_writes_to_processed_stage3_without_overwriting_stage2(tmp_path, monkeypatch):
    task = tmp_path / "piper/stage2/place_object_on_table"
    geometry = task / "geometry"
    geometry.mkdir(parents=True)
    scene = geometry / "final.blend"
    scene.write_bytes(b"source")
    (task / "plan").mkdir()
    (task / "plan/plan.md").write_text("Plan")
    images = []
    for index in range(4):
        image = geometry / f"camera_{index}.png"
        image.write_bytes(b"rgb")
        images.append({"render_rgb": str(image)})
    (geometry / "stage2_manifest.json").write_text(json.dumps({
        "scene": str(scene), "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "accepted": True, "evaluation": {"passed": True, "views": images},
    }))
    prior = tmp_path / "old/piper/place_object_on_table/mujoco_v7/semantics.json"
    prior.parent.mkdir(parents=True)
    prior.write_text(json.dumps({"source_sha256": hashlib.sha256(scene.read_bytes()).hexdigest()}))

    def fake_export(config):
        assert config.semantics_path == prior
        config.output.mkdir(parents=True)
        (config.output / "scene.xml").write_text("<mujoco/>")
        (config.output / "semantics.json").write_text(json.dumps({"objects": []}))
        return {"scene": str(config.output / "scene.xml")}

    monkeypatch.setattr(batch, "export_blend", fake_export)
    result = run_task(task, tmp_path, "trial", False, tmp_path, reuse_semantics_from=tmp_path / "old")
    assert result["status"] == "exported"
    assert result["scene"] == str(tmp_path / "piper/stage3/place_object_on_table/mujoco/scene.xml")
    assert scene.read_bytes() == b"source"
