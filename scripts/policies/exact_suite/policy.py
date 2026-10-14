def names_by_type(objects, kind):
    return [name for name in objects if name.endswith(kind)]


def distance_squared(first, second):
    delta_x = first[0] - second[0]
    delta_y = first[1] - second[1]
    return delta_x * delta_x + delta_y * delta_y


def side_target(side, index=0, count=1):
    left = "left" in side
    right = "right" in side
    top = "top" in side
    bottom = "bottom" in side
    if left and top:
        return [[-0.165, 0.135], [-0.13, 0.18], [-0.175, 0.185]][index % 3]
    if right and top:
        return [[-0.045, 0.13], [0.0, 0.135], [-0.02, 0.18]][index % 3]
    if left and bottom:
        return [[-0.165, -0.135], [-0.13, -0.18], [-0.175, -0.185]][index % 3]
    if right and bottom:
        return [[-0.045, -0.13], [0.0, -0.135], [-0.02, -0.18]][index % 3]
    if left:
        return [-0.165, -0.15 + 0.15 * index]
    if right:
        return [-0.025, -0.15 + 0.15 * index]
    if bottom:
        return [-0.16 + 0.07 * index, -0.155]
    if top:
        return [-0.16 + 0.07 * index, 0.155]
    return [-0.10, 0.0]


def move_block(api, name, target):
    objects = api.observe()["objects"]
    if name not in objects:
        return False
    if isinstance(target, str):
        if target not in objects:
            return False
        target_xy = objects[target]["xy_m"]
    else:
        target_xy = target
    try:
        result = api.robot.pick_place(name, objects[name]["xy_m"], target_xy)
        return result.get("status") == "completed"
    except Exception:
        return False


def outcome(api, attempted):
    if not attempted:
        return {"status": "failed", "reason": "The requested operation could not be performed"}
    if api.task.check_success():
        return {"status": "completed"}
    return {"status": "failed", "reason": "Public exact check did not confirm the goal"}


def match_color_bowls(api, blocks, bowls, matching):
    if matching:
        for block in blocks:
            color = block.split()[0]
            target = color + " bowl"
            if target not in bowls or not move_block(api, block, target):
                return False
        return True
    for first in range(len(bowls)):
        for second in range(len(bowls)):
            for third in range(len(bowls)):
                indices = [first, second, third]
                if len(blocks) != 3 or len(bowls) != 3:
                    return False
                if first == second or first == third or second == third:
                    continue
                valid = True
                for index in range(3):
                    if blocks[index].split()[0] == bowls[indices[index]].split()[0]:
                        valid = False
                if valid:
                    for index in range(3):
                        if not move_block(api, blocks[index], bowls[indices[index]]):
                            return False
                    return True
    return False


def run(api):
    observation = api.observe()
    objects = observation["objects"]
    instruction = observation["instruction"].lower().rstrip(".")
    blocks = names_by_type(objects, " block")
    bowls = names_by_type(objects, " bowl")
    if not blocks:
        return {"status": "failed", "reason": "No block was observed"}

    if instruction.startswith("stack all the blocks on the "):
        side = instruction.split(" on the ", 1)[1].replace(" corner", "").replace(" ", "-")
        base = blocks[0]
        if not move_block(api, base, side_target(side)):
            return outcome(api, False)
        for index in range(1, len(blocks)):
            if not move_block(api, blocks[index], blocks[index - 1]):
                return outcome(api, False)
        return outcome(api, True)

    if instruction == "stack all the blocks":
        for index in range(1, len(blocks)):
            if not move_block(api, blocks[index], blocks[index - 1]):
                return outcome(api, False)
        return outcome(api, True)

    if instruction.startswith("put all the blocks on the "):
        side = instruction.split(" on the ", 1)[1].replace(" corner", "").replace(" side", "")
        side = side.replace(" ", "-")
        for index in range(len(blocks)):
            if not move_block(api, blocks[index], side_target(side, index, len(blocks))):
                return outcome(api, False)
        return outcome(api, True)

    if instruction == "put all the blocks in different corners":
        corners = [[-0.165, -0.165], [-0.165, 0.165],
                   [-0.015, -0.165], [-0.015, 0.165]]
        for index in range(len(blocks)):
            if not move_block(api, blocks[index], corners[index]):
                return outcome(api, False)
        return outcome(api, True)

    if instruction.startswith("put all the blocks in a ") and instruction.endswith(" line"):
        orientation = instruction.split(" in a ", 1)[1].split(" line")[0]
        for index in range(len(blocks)):
            if orientation == "diagonal":
                target = [-0.16 + 0.07 * index, -0.07 + 0.07 * index]
            elif orientation == "horizontal":
                target = [-0.16 + 0.07 * index, 0.0]
            else:
                target = [-0.1, -0.14 + 0.07 * index]
            if not move_block(api, blocks[index], target):
                return outcome(api, False)
        return outcome(api, True)

    if (instruction.startswith("put all the blocks in the bowls with ")
            or instruction.startswith("put the blocks in the bowls with ")):
        return outcome(api, match_color_bowls(
            api, blocks, bowls, "mismatched" not in instruction))

    if instruction.startswith("put the blocks in the ") and instruction.endswith(" bowl"):
        bowl = instruction.split("in the ", 1)[1]
        if bowl not in bowls:
            return outcome(api, False)
        objects = api.observe()["objects"]
        center = objects[bowl]["xy_m"]
        offsets = [[-0.03, -0.03], [0.03, -0.03], [0.0, 0.03], [-0.03, 0.03]]
        for index in range(len(blocks)):
            delta = offsets[index % len(offsets)]
            target = [center[0] + delta[0], center[1] + delta[1]]
            if not move_block(api, blocks[index], target):
                return outcome(api, False)
        return outcome(api, True)

    if not instruction.startswith("pick up the "):
        return outcome(api, False)

    source_phrase, remainder = instruction.split(" and place it ", 1)
    source_phrase = source_phrase[len("pick up the "):]

    if source_phrase.endswith(" block") and source_phrase in blocks:
        source = source_phrase
    elif source_phrase.startswith("block to the ") and " of the " in source_phrase:
        direction = source_phrase.split("block to the ", 1)[1].split(" of the ", 1)[0]
        bowl = source_phrase.split(" of the ", 1)[1]
        if bowl not in objects:
            return outcome(api, False)
        center = objects[bowl]["xy_m"]
        candidates = []
        for name in blocks:
            position = objects[name]["xy_m"]
            if ((direction == "top" and position[1] > center[1] + 0.02)
                    or (direction == "bottom" and position[1] < center[1] - 0.02)
                    or (direction == "left" and position[0] < center[0] - 0.02)
                    or (direction == "right" and position[0] > center[0] + 0.02)):
                candidates.append(name)
        if len(candidates) != 1:
            return outcome(api, False)
        source = candidates[0]
    elif source_phrase.startswith("block closest to the ") or source_phrase.startswith("block farthest to the "):
        nearest = "closest" in source_phrase
        bowl = source_phrase.split(" to the ", 1)[1]
        if bowl not in objects:
            return outcome(api, False)
        center = objects[bowl]["xy_m"]
        source = blocks[0]
        for name in blocks[1:]:
            candidate_distance = distance_squared(objects[name]["xy_m"], center)
            previous_distance = distance_squared(objects[source]["xy_m"], center)
            if (nearest and candidate_distance < previous_distance) or (not nearest and candidate_distance > previous_distance):
                source = name
    elif " block from the " in source_phrase:
        ordinal = source_phrase.split(" block from", 1)[0]
        direction = source_phrase.split(" from the ", 1)[1]
        rank = {"first": 0, "second": 1, "third": 2, "fourth": 3}.get(ordinal)
        if rank is None or rank >= len(blocks):
            return outcome(api, False)
        ordered = []
        axis = 0 if direction in ("left", "right") else 1
        for name in blocks:
            insertion = 0
            value = objects[name]["xy_m"][axis]
            while insertion < len(ordered):
                other = objects[ordered[insertion]]["xy_m"][axis]
                if (direction in ("top", "right") and value > other) or (direction in ("bottom", "left") and value < other):
                    break
                insertion += 1
            ordered.insert(insertion, name)
        source = ordered[rank]
    else:
        return outcome(api, False)

    if remainder.startswith("on the "):
        target_name = remainder[len("on the "):]
        if target_name in objects:
            return outcome(api, move_block(api, source, target_name))
        side = target_name.replace(" corner", "").replace(" side", "").replace(" ", "-")
        return outcome(api, move_block(api, source, side_target(side)))

    if remainder.startswith("a lot to the ") or remainder.startswith("a little to the "):
        amount = 0.12 if remainder.startswith("a lot") else 0.06
        direction = remainder.split(" to the ", 1)[1].split(" of the ", 1)[0]
        bowl = remainder.split(" of the ", 1)[1]
        if bowl not in objects:
            return outcome(api, False)
        center = objects[bowl]["xy_m"]
        delta = {"left": [-amount, 0.0], "right": [amount, 0.0],
                 "top": [0.0, amount], "bottom": [0.0, -amount]}.get(direction)
        if delta is None:
            return outcome(api, False)
        return outcome(api, move_block(api, source, [center[0] + delta[0], center[1] + delta[1]]))

    if remainder.startswith("in the corner "):
        farthest = "farthest" in remainder
        bowl = remainder.split(" to the ", 1)[1]
        if bowl not in objects:
            return outcome(api, False)
        corners = [[-0.165, -0.165], [-0.165, 0.165],
                   [-0.015, -0.165], [-0.015, 0.165]]
        center = objects[bowl]["xy_m"]
        target = corners[0]
        for corner in corners[1:]:
            distance = distance_squared(corner, center)
            previous = distance_squared(target, center)
            if (farthest and distance > previous) or (not farthest and distance < previous):
                target = corner
        return outcome(api, move_block(api, source, target))

    return outcome(api, False)
