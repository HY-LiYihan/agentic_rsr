# Stage 3 visual physical prior

Before seeing the simulator's masses, friction or grasp trials, estimate a **rough prior** for ONE task-relevant movable object. Use only the attached third-person image, the task description and the Stage 2 visual plan below. Choose an exact body name from the given list. Ignore the robot and background. Consider apparent size, possible material, whether the item is hollow or heavy, and how a 1.0-friction, two-finger proxy might lift it. Record broad plausible ranges rather than pretending to know the real mass or the real gripping threshold. The same apparent object can have widely different mass and friction; explicitly describe that uncertainty. The per-finger actuator limit is 10 N; estimates outside that capability may have a high endpoint above 10 N but may not be tested. Do not inspect the scene XML, mass values, physics reports or workspace files in this turn. This is not a real-world measurement or calibration.

Allowed body names: {body_names}

Task and visual plan:
{task_context}
