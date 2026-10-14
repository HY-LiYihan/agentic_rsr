import pytest

from real2sim.stage2.phase_runner import _validate_plan


def _plan(capabilities):
    return f"""## Object hierarchy
Open the box on the table.
## Approximate dimensions
Box: 0.2 x 0.1 x 0.08 m.
## Required capabilities
{capabilities}
## Relative layout
The box sits on the table beside the Piper gripper.
## Necessary assumptions
The box lid can rotate on a hinge.
"""


@pytest.mark.parametrize("capabilities", ["Piper robot arm gripper", "FR3", "Franka", "robotic arm"])
def test_plan_rejects_robot_as_object_capability(tmp_path, capabilities):
    plan = tmp_path / "plan.md"
    plan.write_text(_plan(capabilities))
    with pytest.raises(ValueError, match="task objects, not the robot"):
        _validate_plan(plan, "open the box")


def test_plan_accepts_task_capabilities_and_robot_mentions_in_layout(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(_plan("The box lid rotates; the box remains supported on the table."))
    assert _validate_plan(plan, "open the box")["sha256"]
