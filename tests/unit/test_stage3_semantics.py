import pytest
import json
import sys
from types import SimpleNamespace

from real2sim.stage3.semantics_agent import classify_objects, object_inventory, validate_classification


def test_classification_marks_wall_visual_only_without_collision():
    objects = [{"name": "RoomWall"}, {"name": "TableSurface"}]
    response = {"objects": [
        {"source_name": "RoomWall", "role": "visual_only", "joint_type": "fixed",
         "collision": "none", "attached_to": None, "reason": "background"},
        {"source_name": "TableSurface", "role": "static", "joint_type": "fixed",
         "collision": "box", "attached_to": None, "reason": "support"},
    ]}
    result = validate_classification(response, objects)
    assert result["RoomWall"]["collision"] == "none"
    assert result["TableSurface"]["role"] == "static"


def test_classification_rejects_unphysical_or_incomplete_response():
    objects = [{"name": "RoomWall"}]
    for response in ({"objects": []}, {"objects": [{
        "source_name": "RoomWall", "role": "dynamic", "joint_type": "fixed", "collision": "none",
    }]}):
        with pytest.raises(ValueError):
            validate_classification(response, objects)


def test_visual_fragments_attach_only_to_a_moving_entity():
    objects = [{"name": "button"}, {"name": "label"}]
    response = {"objects": [
        {"source_name": "button", "role": "dynamic", "joint_type": "free",
         "collision": "box", "attached_to": None, "reason": "graspable"},
        {"source_name": "label", "role": "visual_only", "joint_type": "fixed",
         "collision": "none", "attached_to": "button", "reason": "decoration"},
    ]}
    assert validate_classification(response, objects)["label"]["attached_to"] == "button"
    response["objects"][1]["attached_to"] = "label"
    with pytest.raises(ValueError, match="attachment"):
        validate_classification(response, objects)


def test_compound_wall_requires_collidable_moving_owner():
    objects = [{"name": "box_floor"}, {"name": "box_wall"}]
    result = {"objects": [{"source_name": "box_floor", "role": "dynamic", "joint_type": "free",
                           "collision": "box", "attached_to": None, "reason": "container"},
                          {"source_name": "box_wall", "role": "compound", "joint_type": "fixed",
                           "collision": "box", "attached_to": "box_floor", "reason": "rigid side"}]}
    assert validate_classification(result, objects)["box_wall"]["attached_to"] == "box_floor"
    result["objects"][1]["attached_to"] = None
    with pytest.raises(ValueError, match="moving owner"):
        validate_classification(result, objects)


def test_hollow_container_can_be_dynamic_or_static_without_filling_opening():
    objects = [{"name": "cup"}, {"name": "bucket"}]
    decisions = {"objects": [
        {"source_name": "cup", "role": "dynamic", "joint_type": "free",
         "collision": "open_container", "attached_to": None, "reason": "hollow cup"},
        {"source_name": "bucket", "role": "static", "joint_type": "fixed",
         "collision": "open_container", "attached_to": None, "reason": "receiving bin"},
    ]}
    assert set(validate_classification(decisions, objects)) == {"cup", "bucket"}


def test_inventory_dimensions_are_not_scaled_twice():
    inventory = object_inventory([{
        "name": "Part", "parent": None, "vertices": [[0, 0, 0], [0.5, 0.2, 0.1]],
        "location": [1, 2, 3], "scale": [3, 3, 3], "custom_properties": {},
    }])
    assert inventory[0]["size_m"] == [0.5, 0.2, 0.1]


def test_classifier_retries_inconsistent_joint_and_collision(tmp_path, monkeypatch):
    item = {"name": "Drawer", "parent": None, "vertices": [[0, 0, 0], [1, 1, 1]],
            "location": [0, 0, 0], "custom_properties": {}}
    calls = []

    class Thread:
        id = "temporary"

        def run(self, inputs, **kwargs):
            calls.append(inputs)
            return SimpleNamespace(id="turn", status="completed", usage=None,
                                   final_response=json.dumps({"objects": [{
                                       "source_name": "Drawer", "role": "operable",
                                       "joint_type": "slide", "collision": "mesh" if len(calls) == 1 else "box",
                                       "attached_to": None, "reason": "drawer"}]}))

    class Codex:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def thread_start(self, **kwargs):
            return Thread()

        def thread_archive(self, thread_id):
            assert thread_id == "temporary"

    monkeypatch.setitem(sys.modules, "openai_codex", SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"), Codex=Codex,
        LocalImageInput=lambda path: path, Sandbox=SimpleNamespace(read_only="read_only"),
        TextInput=lambda text: text))
    assert classify_objects([item], "open drawer", "drawer plan", [], tmp_path, "hash")["Drawer"]["collision"] == "box"
    assert len(calls) == 2
    assert len(json.loads((tmp_path / "semantics_session.json").read_text())["validation_errors"]) == 1
