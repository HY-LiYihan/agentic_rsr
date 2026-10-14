import json

import cv2
import numpy as np
import pytest

from real2sim.stage1.object_metrics import (
    OPERABLE_OBJECT_PROMPTS,
    MetricsError,
    bbox_position_error,
    evaluate_scene,
    match_blend_groups,
    point_cloud_metrics,
    _split_same_appearance,
)


class FakeSAM3:
    def segment(self, image, prompt):
        mask = np.zeros((288, 512), dtype=bool)
        mask[100:150, 260:310] = True if prompt == "pink cone" else False
        mask[100:150, 200:250] = True if prompt != "pink cone" else False
        return mask


def _scene(tmp_path, task="stack_cones"):
    scene = tmp_path / "data/processed/piper/stage1" / task
    (scene / "frames").mkdir(parents=True)
    (scene / "depth_metric").mkdir()
    (scene / "sam3_masks").mkdir()
    camera = {"intrinsics": [], "extrinsics": [], "world_frame": "robot_base"}
    for frame in range(32):
        camera["intrinsics"].append([[100., 0., 256.], [0., 100., 144.], [0., 0., 1.]])
        camera["extrinsics"].append([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
        cv2.imwrite(str(scene / "frames" / f"{frame:02d}.jpg"), np.full((288, 512, 3), 120, dtype=np.uint8))
        np.save(scene / "depth_metric" / f"depth_{frame:02d}.npy", np.full((288, 512), 1., dtype=np.float32))
        np.save(scene / "sam3_masks" / f"mask_{frame:02d}.npy", np.zeros((288, 512), dtype=bool))
    (scene / "camera_metric.json").write_text(json.dumps(camera))
    return scene


def test_point_cloud_metrics_and_bbox():
    first = np.array([[0, 0, 0], [0.01, 0, 0]], dtype=np.float32)
    second = first + 0.001
    result = point_cloud_metrics(first, second)
    assert result["chamfer_distance_m"] == pytest.approx(np.sqrt(3) * 0.001)
    assert result["f1_score"] == 1.0
    assert bbox_position_error(first, second) == pytest.approx(np.sqrt(3) * 0.001)


def _mock_blend(monkeypatch, scene):
    blend = scene.parent.parent / "stage2" / scene.name / "geometry" / "final.blend"
    blend.parent.mkdir(parents=True)
    blend.write_bytes(b"fake blend")

    def extract(_blend, output, camera, frames, stride, blender):
        groups = {}
        for object_label, _ in OPERABLE_OBJECT_PROMPTS[scene.name]:
            directory = output / object_label
            directory.mkdir(parents=True, exist_ok=True)
            for frame in frames:
                mask = np.zeros((288, 512), dtype=bool)
                start = 260 if object_label == "pink_cone" else 200
                mask[100:150:stride, start:start + 50:stride] = True
                col, row = np.where(mask)[1], np.where(mask)[0]
                points = np.column_stack(((col - 256.) / 100., (row - 144.) / 100., np.ones(len(row))))
                np.save(directory / f"predicted_mask_{frame:02d}.npy", mask)
                np.save(directory / f"predicted_points_{frame:02d}.npy", points.astype(np.float32))
            groups[object_label] = ["Stage2_EStopHousing" if object_label == "emergency_stop_switch" else "Stage2_Blue_Cone"]
        return groups

    monkeypatch.setattr("real2sim.stage1.object_metrics._extract_model", extract)
    return blend


def test_evaluate_scene_writes_report(tmp_path, monkeypatch):
    scene = _scene(tmp_path)
    _mock_blend(monkeypatch, scene)
    report = evaluate_scene(scene, tmp_path / "runs", segmenter=FakeSAM3(), frames=(0, 10, 21, 31))
    assert report["schema"] == "stage1-vs-stage2-object-metrics-v2"
    assert report["objects"]["blue_cone"]["status"] == "measured_unverified"
    assert (tmp_path / "runs/piper/stack_cones/report.json").is_file()
    assert (tmp_path / "runs/piper/stack_cones/masks/blue_cone/overlay_00.png").is_file()
    assert (tmp_path / "runs/piper/stack_cones/masks/blue_cone/mask_00.png").is_file()
    assert (tmp_path / "runs/piper/stack_cones/model/blue_cone/predicted_mask_00.png").is_file()
    assert (tmp_path / "runs/piper/stack_cones/model/blue_cone/comparison_00.png").is_file()


def test_object_masks_are_cached_and_empty_views_are_not_scored(tmp_path, monkeypatch):
    scene = _scene(tmp_path)
    _mock_blend(monkeypatch, scene)
    output = tmp_path / "runs"
    service = FakeSAM3()
    first = evaluate_scene(scene, output, segmenter=service, frames=(0, 10))
    assert first["objects"]["blue_cone"]["mean"]["f1_score"] == pytest.approx(1.0)

    class OfflineSAM3:
        def segment(self, image, prompt):
            raise AssertionError("cached mask must be reused")

    again = evaluate_scene(scene, output, segmenter=OfflineSAM3(), frames=(0, 10))
    assert again["objects"]["blue_cone"]["mean"] == first["objects"]["blue_cone"]["mean"]
    assert first["objects"]["blue_cone"]["blend_meshes"] == ["Stage2_Blue_Cone"]


def test_bad_frame_and_thresholds_are_rejected(tmp_path):
    scene = _scene(tmp_path)
    with pytest.raises(MetricsError, match="only the four"):
        evaluate_scene(scene, tmp_path / "runs", frames=(1,))
    with pytest.raises(MetricsError, match="positive"):
        evaluate_scene(scene, tmp_path / "runs", voxel_size=0)


def test_emergency_switch_targets_entire_assembly(tmp_path, monkeypatch):
    scene = _scene(tmp_path, "put_emergency_stop_button_in_green_storage_box")
    _mock_blend(monkeypatch, scene)

    class RecordingSAM3(FakeSAM3):
        def __init__(self):
            self.prompts = []

        def segment(self, image, prompt):
            self.prompts.append(prompt)
            if prompt == "green storage box":
                mask = np.zeros((288, 512), dtype=bool)
                mask[100:150, 260:310] = True
                return mask
            return super().segment(image, prompt)

    service = RecordingSAM3()
    report = evaluate_scene(scene, tmp_path / "runs", segmenter=service, frames=(0, 10))
    target = report["objects"]["emergency_stop_switch"]
    assert service.prompts == ["black and red switch", "black and red switch",
                               "green storage box", "green storage box"]
    assert "red button" in target["target_scope"]
    assert "black housing" in target["target_scope"]
    assert target["mean"]["f1_score"] == pytest.approx(1.0)


def test_stage2_emergency_mesh_group_includes_housing_and_button():
    names = ["Stage2_EStopHousing", "Stage2_EStopRedMushroom", "Stage2_EStopYellowCollar",
             "Stage2_TrayFloor", "robot_visual_00"]
    groups = match_blend_groups("put_emergency_stop_button_in_green_storage_box", names)
    assert groups["emergency_stop_switch"] == names[:3]
    assert groups["green_storage_box"] == ["Stage2_TrayFloor"]
    with pytest.raises(MetricsError, match="missing target"):
        match_blend_groups("put_emergency_stop_button_in_green_storage_box", ["robot_visual_00"])


def test_same_appearance_instances_keep_separate_masks():
    union = np.zeros((50, 100), dtype=bool)
    union[10:30, 10:30] = True
    union[10:30, 70:90] = True
    predicted = {"a": union & (np.arange(100)[None, :] < 50),
                 "b": union & (np.arange(100)[None, :] >= 50)}
    masks = _split_same_appearance(union, predicted)
    assert np.array_equal(masks["a"], predicted["a"])
    assert np.array_equal(masks["b"], predicted["b"])


def test_same_appearance_scene_segments_once_per_view(tmp_path, monkeypatch):
    scene = _scene(tmp_path, "stack_brown_blocks")
    _mock_blend(monkeypatch, scene)

    class SharedSAM3:
        calls = 0

        def segment(self, image, prompt):
            self.calls += 1
            assert prompt == "brown blocks"
            mask = np.zeros((288, 512), dtype=bool)
            mask[100:150, 200:250] = True
            mask[100:150, 260:310] = True
            return mask

    service = SharedSAM3()
    report = evaluate_scene(scene, tmp_path / "runs", segmenter=service, frames=(0, 10))
    mask_root = tmp_path / "runs/piper/stack_brown_blocks/masks"
    first = np.load(mask_root / "brown_block_a/mask_00.npy")
    second = np.load(mask_root / "brown_block_b/mask_00.npy")
    assert service.calls == 2
    assert (first & second).sum() == 0
    assert first.sum() == second.sum() == 2500
    assert (mask_root / "_shared_instances/mask_00.npy").is_file()
    assert all(result["instance_scope"] == "model_assigned_same_appearance_instance"
               for result in report["objects"].values())


def test_tissue_box_keeps_raw_mask_and_excludes_the_tissue(tmp_path, monkeypatch):
    scene = _scene(tmp_path, "take_out_tissue")
    _mock_blend(monkeypatch, scene)

    class TissueSAM3:
        def segment(self, image, prompt):
            mask = np.zeros((288, 512), dtype=bool)
            mask[100:150, 200:250] = True
            if prompt == "tissue box":
                mask[150:200, 200:250] = True
            return mask

    report = evaluate_scene(scene, tmp_path / "runs", segmenter=TissueSAM3(), frames=(0,))
    root = tmp_path / "runs/piper/take_out_tissue/masks"
    raw = np.load(root / "tissue_box/mask_00.npy")
    evaluated = np.load(root / "tissue_box/evaluation_mask_00.npy")
    tissue = np.load(root / "tissue/mask_00.npy")
    assert raw.sum() == 5000
    assert evaluated.sum() == tissue.sum() == 2500
    assert not np.any(evaluated & tissue)
    assert report["objects"]["tissue_box"]["raw_mask_pixels"]["0"] == 5000
    assert not any(flag["reason"] == "sam3_instance_masks_overlap" for flag in report["review_flags"])


def test_all_targets_have_distinct_blend_groups():
    for task, targets in OPERABLE_OBJECT_PROMPTS.items():
        assert len({label for label, _ in targets}) == len(targets)


def test_overlapping_sam3_instances_exclude_ambiguous_view(tmp_path, monkeypatch):
    scene = _scene(tmp_path)
    _mock_blend(monkeypatch, scene)

    class OverlapSAM3(FakeSAM3):
        def segment(self, image, prompt):
            mask = super().segment(image, prompt)
            if prompt == "blue cone":
                mask[100:150, 260:310] = True
            return mask

    report = evaluate_scene(scene, tmp_path / "runs", segmenter=OverlapSAM3(), frames=(0, 10))
    assert report["status"] == "partially_measured_unverified"
    assert report["objects"]["blue_cone"]["invalid_views"] == {
        "0": "SAM3 mask overlaps another task object",
        "10": "SAM3 mask overlaps another task object",
    }
    assert report["objects"]["pink_cone"]["valid_views"] == [0, 10]
    assert {tuple(flag["excluded_objects"]) for flag in report["review_flags"]} == {("blue_cone",)}
