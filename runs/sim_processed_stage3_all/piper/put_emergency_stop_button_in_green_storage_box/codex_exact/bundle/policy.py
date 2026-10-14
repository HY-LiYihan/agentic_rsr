def run(api):
    source = 'Stage2_EStopHousing'
    target = 'Stage2_TrayFloor'
    required = [source, target, 'Stage2_TrayWallLeft',
                'Stage2_TrayWallRight', 'Stage2_TrayWallFront',
                'Stage2_TrayWallBack']

    def observe():
        observation = api.observe()
        objects = observation.get('objects', observation.get('bodies', {}))
        for name in required:
            if name not in objects:
                raise ValueError('Required task body is not observable')
        return objects

    def contained(objects):
        x, y, z = objects[source]['xyz_m']
        floor = objects[target]['xyz_m']
        left = objects['Stage2_TrayWallLeft']['xyz_m'][0]
        right = objects['Stage2_TrayWallRight']['xyz_m'][0]
        front = objects['Stage2_TrayWallFront']['xyz_m'][1]
        back = objects['Stage2_TrayWallBack']['xyz_m'][1]
        # Keep the button well away from all four walls and near the floor.
        margin = 0.025
        return (left + margin < x < right - margin and
                front + margin < y < back - margin and
                floor[2] < z < floor[2] + 0.075)

    def stable(previous, current):
        p = previous[source]['xyz_m']
        c = current[source]['xyz_m']
        displacement = sum((p[i] - c[i]) ** 2 for i in range(3))
        a = previous[source]['quat_wxyz']
        b = current[source]['quat_wxyz']
        rotation_change = min(
            sum((a[i] - b[i]) ** 2 for i in range(4)),
            sum((a[i] + b[i]) ** 2 for i in range(4)))
        return displacement < 0.000004 and rotation_change < 0.0004

    try:
        objects = observe()
        if (not objects[source]['movable'] or
                objects[source]['joint_type'] != 'free' or
                objects[target]['movable']):
            api.task.stop()
            return {'status': 'failed', 'reason': 'Unexpected task body configuration'}

        for attempt in range(2):
            # The floor is the observable static placement surface inside the bin.
            api.robot.pick_place(source, target)
            objects = observe()
            api.robot.release(source)
            objects = observe()
            api.robot.gripper(0.08)
            objects = observe()

            floor = objects[target]['xyz_m']
            button = objects[source]['xyz_m']
            wall_height = max(objects[name]['xyz_m'][2] for name in required[2:])
            clearance = max(button[2], wall_height, floor[2]) + 0.16
            api.robot.move_pose(floor[0], floor[1], clearance, (0, 0, 1, 0))
            objects = observe()

            consecutive = 0
            for verification in range(5):
                # The summary is not a success flag; inspect fresh public poses.
                api.task.check_success()
                current = observe()
                if contained(current) and stable(objects, current):
                    consecutive += 1
                else:
                    consecutive = 0
                objects = current
                if consecutive >= 3:
                    return {'status': 'completed',
                            'reason': 'Released button remains stable inside the green tray with the gripper withdrawn'}

        api.task.stop()
        return {'status': 'failed', 'reason': 'Could not verify stable placement inside the tray after two attempts'}
    except Exception:
        api.task.stop()
        return {'status': 'failed', 'reason': 'Public robot action or observation failed'}
