"""Run a policy with a small RPC surface in a separate, time-limited process."""

from __future__ import annotations

import ast
import builtins
import multiprocessing
import time
from types import SimpleNamespace


FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Global, ast.Nonlocal,
                   ast.With, ast.AsyncWith, ast.Lambda)
SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
    "Exception", "RuntimeError", "ValueError",
)}
TOOLS = {"state", "observe", "robot.pick_place", "robot.survey", "robot.move_pose",
         "robot.move_trajectory", "robot.gripper", "robot.grasp", "robot.release",
         "object.set_joint", "task.check_success", "task.stop", "stop"}
TOOLS.update({"reset", "return_control", "object_poses"})


def validate_program(source: str) -> ast.Module:
    tree = ast.parse(source)
    if not any(isinstance(node, ast.FunctionDef) and node.name == "run" for node in tree.body):
        raise ValueError("Program must define run(api)")
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            raise ValueError(f"Generated program cannot use {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Generated program cannot use private attributes")
    return tree


class RemoteFacade:
    def __init__(self, connection, interface="legacy"):
        self._connection = connection
        if interface == "stage3":
            self.robot = SimpleNamespace(
                move_pose=lambda *args: self.call("robot.move_pose", *args),
                move_trajectory=lambda waypoints: self.call("robot.move_trajectory", waypoints),
                gripper=lambda width: self.call("robot.gripper", width),
            )
            self.reset = lambda: self.call("reset")
            self.return_control = lambda reason="": self.call("return_control", reason)
            self.stop = lambda: self.call("stop")
            self.object_poses = lambda: self.call("object_poses")
            return
        self.robot = SimpleNamespace(
            pick_place=lambda *args: self.call("robot.pick_place", *args),
            survey=lambda: self.call("robot.survey"),
            move_pose=lambda *args: self.call("robot.move_pose", *args),
            gripper=lambda *args: self.call("robot.gripper", *args),
            grasp=lambda *args: self.call("robot.grasp", *args),
            release=lambda *args: self.call("robot.release", *args),
        )
        self.object = SimpleNamespace(
            set_joint=lambda *args: self.call("object.set_joint", *args),
        )
        self.task = SimpleNamespace(
            check_success=lambda: self.call("task.check_success"),
            stop=lambda: self.call("task.stop"),
        )
        self.agent = SimpleNamespace(
            ask=lambda label, question, options: self.call("agent.ask", label, question, options),
        )

    def observe(self, queries=None):
        return self.call("observe") if queries is None else self.call("observe", queries)

    def state(self):
        return self.call("state")

    def stop(self):
        return self.call("stop")

    def call(self, name, *args):
        self._connection.send((name, args))
        success, value = self._connection.recv()
        if not success:
            raise RuntimeError(value)
        if name == "return_control":
            raise _ReturnControl(value)
        return value


class _ReturnControl(BaseException):
    def __init__(self, checkpoint):
        super().__init__("Policy returned control to the outer loop")
        self.checkpoint = checkpoint


def _worker(source, connection, entry, state, interface):
    try:
        variables = {"__builtins__": SAFE_BUILTINS}
        exec(compile(validate_program(source), "<sim-policy>", "exec"), variables)
        facade = RemoteFacade(connection, interface=interface)
        if entry == "run":
            result = variables["run"](facade) if state is None else variables["run"](facade, state)
        else:
            if not entry.startswith("step_") or entry not in variables or not callable(variables[entry]):
                raise ValueError("Unknown policy step")
            result = variables[entry](facade, state)
        connection.send(("finished", result))
    except _ReturnControl as exc:
        connection.send(("returned", exc.checkpoint))
    except BaseException as exc:
        connection.send(("failed", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def run_program(source: str, dispatch, timeout: float = 180.0, on_pause=None,
                entry: str = "run", state=None, interface: str = "legacy") -> dict:
    validate_program(source)
    if interface not in ("legacy", "stage3"):
        raise ValueError(f"Unknown program interface: {interface}")
    parent, child = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=_worker, args=(source, child, entry, state, interface))
    process.start()
    child.close()
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if not parent.poll(min(0.1, max(0, deadline - time.monotonic()))):
                if not process.is_alive():
                    raise RuntimeError("Policy worker exited without a result")
                continue
            name, args = parent.recv()
            if name == "finished":
                return {"status": "finished", "return": args}
            if name == "returned":
                return {"status": "returned", "checkpoint": args}
            if name == "failed":
                return {"status": "failed", "error": args}
            if name == "agent.ask":
                try:
                    if on_pause is None:
                        raise RuntimeError("No Codex decision controller configured")
                    started = time.monotonic()
                    try:
                        request = dispatch(name, *args)
                        choice = on_pause(request)
                        if choice not in request["options"]:
                            raise ValueError("Codex selected an unknown branch")
                        if choice == "abort":
                            dispatch("task.stop")
                        parent.send((True, choice))
                    finally:
                        deadline += time.monotonic() - started
                except Exception as exc:
                    parent.send((False, f"{type(exc).__name__}: {exc}"))
                continue
            if name not in TOOLS:
                parent.send((False, f"Unknown tool: {name}"))
                continue
            try:
                parent.send((True, dispatch(name, *args)))
            except Exception as exc:
                parent.send((False, f"{type(exc).__name__}: {exc}"))
        return {"status": "timeout", "error": f"Program exceeded {timeout:g} seconds"}
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        parent.close()
