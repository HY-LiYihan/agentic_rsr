"""Publish the active Piper Stage 1 scenes with the shared Franka-tested exporter."""

from .franka_delivery import main as deliver


def main() -> None:
    deliver(default_robot="piper")


if __name__ == "__main__":
    main()
