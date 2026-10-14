"""Interactive JSON-lines CLI for manually exercising a Stage 3 SIM scene."""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
from pathlib import Path

from sim.stage3_harness import Stage3Tools
from sim.stage3_observation import Stage3RGBDObservation
from sim.stage3_scene import Stage3Scene

ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = ROOT / "runs/sim_processed_stage3_all"


def load_endpoint() -> tuple[str | None, int | None]:
    from perception.config import sam3_endpoint
    return sam3_endpoint()


def public_result(value):
    if isinstance(value, dict):
        return {key: public_result(item) for key, item in value.items()
                if key != "rgb_base64"}
    if isinstance(value, list):
        return [public_result(item) for item in value]
    return value


def parse_json(value: str):
    return json.loads(value)


def run_command(tools, line):
    command, _, payload = line.strip().partition(" ")
    try:
        if command in ("quit", "exit"):
            return {"status": "quit"}
        if command == "state":
            result = tools.dispatch("state")
        elif command == "object_poses":
            result = tools.dispatch("object_poses")
        elif command == "observe":
            result = tools.dispatch("observe", parse_json(payload)) if payload else tools.dispatch("observe")
        elif command == "move_pose":
            value = parse_json(payload)
            result = tools.dispatch("robot.move_pose", *value["xyz_m"], value["quaternion_wxyz"])
        elif command == "gripper":
            result = tools.dispatch("robot.gripper", float(payload))
        elif command == "move_trajectory":
            result = tools.dispatch("robot.move_trajectory", parse_json(payload))
        elif command == "reset":
            result = tools.dispatch("reset")
        elif command == "return_control":
            result = tools.dispatch("return_control", payload)
        elif command == "stop":
            result = tools.dispatch("stop")
        else:
            raise ValueError(f"Unknown command: {command}")
        return public_result(result)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def serve(tools, path: Path):
    if path.exists():
        raise FileExistsError(f"Socket already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request = self.rfile.readline(65536).decode("utf-8")
            if request:
                response = run_command(tools, request)
                self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
                if response.get("status") == "quit":
                    self.server.shutdown_requested = True

    try:
        with socketserver.UnixStreamServer(str(path), Handler) as server:
            os.chmod(path, 0o600)
            server.shutdown_requested = False
            print(json.dumps({"status": "ready", "socket": str(path)}), flush=True)
            while not server.shutdown_requested:
                server.handle_request()
    finally:
        path.unlink(missing_ok=True)


def request_command(path: Path, command: str):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.connect(str(path))
        connection.sendall((command + "\n").encode())
        connection.shutdown(socket.SHUT_WR)
        response = connection.makefile().readline()
    if not response:
        raise RuntimeError("SIM CLI service disconnected")
    return json.loads(response)


def send_command(path: Path, command: str):
    print(json.dumps(request_command(path, command), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--robot", choices=("piper", "franka"))
    parser.add_argument("--task")
    parser.add_argument("--image-dir", type=Path, default=ROOT / "runs/stage3_cli/observations")
    parser.add_argument("--listen", type=Path, help="Keep the scene alive on this Unix socket")
    parser.add_argument("--privileged", action="store_true", help="Enable ground-truth poses for stage one only")
    parser.add_argument("--seed", type=int, help="Randomized scene seed (visual stages only)")
    parser.add_argument("--connect", type=Path, help="Send one command to an existing SIM scene")
    parser.add_argument("--command", help="Command to send with --connect")
    args = parser.parse_args()
    if args.connect:
        if not args.command:
            parser.error("--connect requires --command")
        send_command(args.connect, args.command)
        return
    if args.scene is None:
        if not args.robot or not args.task:
            parser.error("provide --scene or both --robot and --task")
        args.scene = SCENE_ROOT / args.robot / args.task / "scene_for_sim.xml"
    if args.robot is None:
        parser.error("--robot is required when --scene is specified")

    scene = Stage3Scene(args.scene, args.robot)
    perception = None
    try:
        scene.set_randomization(args.seed)
        names = tuple(name for name in scene.object_bodies if name != "world")
        host, port = load_endpoint()
        perception = Stage3RGBDObservation(scene, names, args.image_dir, host, port)
        tools = Stage3Tools(scene, args.task or args.scene.stem, "manual CLI",
                            perception=perception, privileged=args.privileged)
        print(json.dumps({"status": "ready", "commands": [
            "state", "observe [json]", "move_pose {json}", "gripper <width_m>",
            "move_trajectory [json]", "reset", "return_control [reason]", "stop", "quit",
        ]}, ensure_ascii=False), flush=True)
        if args.listen:
            serve(tools, args.listen)
            return
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            result = run_command(tools, line)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result.get("status") == "quit":
                break
    finally:
        try:
            if perception is not None:
                perception.close()
        finally:
            scene.close()


if __name__ == "__main__":
    main()
