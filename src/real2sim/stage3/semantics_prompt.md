# Stage 3 physical-entity classification

You are preparing a Stage 2 Blender scene for a MuJoCo physics simulation. Decide the physical role of **each listed non-robot mesh** before scene compilation. Use the task description, Stage 2 plan, inventory and four rendered views. This is a hypothesis about geometry and physical contact, not a measurement of mass or friction.

For every mesh, choose exactly one role:
- `visual_only`: room walls, curtains, backdrops, distant decoration and labels that cannot support or constrain task objects. They remain visible but **do not need a physical body or collision**. A wall that is part of a reachable container is not a backdrop.
- `static`: reachable tabletops/work surfaces, tray or container floors and walls, and any other surface needed to support or constrain manipulated objects. They MUST retain collision, even if visually similar to a non-colliding room backdrop.
- `dynamic`: a graspable independent rigid object with a free joint and box collision.
- `operable`: a task part with a real hinge/slide degree of freedom, only if the geometry and task make the joint plausible. Decorative pieces of the same object should be `visual_only`, not separate moving joints.
- `compound`: a **collidable fixed part attached to a moving entity** (basket walls/rims, a rigid gripper-relevant handle or structural box side). It has no independent joint; its box collider follows the owner's body. Do NOT use `visual_only` for a wall needed to contain items or support stacked containers merely because it is a child piece. Room walls and curtains remain `visual_only`.

Choose `collision` from `none`, `box`, `mesh`, `open_container`. Use `none` only for `visual_only`, use `box` for `compound`, prefer `box` for physical meshes with flat or thin geometry; never use `mesh` for a freely movable mesh. MuJoCo treats a mesh collider as a convex hull: for a task-relevant hollow cup, bowl, bucket, or pen holder, use `open_container` instead of a solid box or mesh, so a bottom and four wall proxies leave an accessible interior. This is only an approximation: note its limitations in the reason. Choose `joint_type` from `fixed`, `free`, `hinge`, `slide`: it must be `fixed` for `static`/`visual_only`/`compound`, `free` for `dynamic`, and `hinge` or `slide` for `operable`. Be conservative when uncertain: preserve appearance without inventing physics. Do not classify the robot or any of its cameras. Return **every source_name exactly once** and a short reason for each. No physical mass or friction measurement is available.

For visual pieces of a moving entity (e.g. labels, decorative cap), set `attached_to` to the **exact source_name** of that entity's `dynamic` or `operable` mesh. They will move with that entity without additional collision. For structural parts that need contact (for example open basket walls), choose `compound` and set `attached_to` to their owner, so they move with collision but no separate joint. Set `attached_to` to null for backdrops, room walls, standalone static support surfaces, and primary physical entities. Do not leave visual fragments of a movable object fixed in world space.

Task: {task_description}

Stage 2 plan:
{plan}

Meshes (dimensions and positions in metres, dimensions are approximations):
{inventory}
