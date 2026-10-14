"""Backward-compatible names for the former ReAct runtime module."""

from .policy_loop import run_policy_loop

run_react = run_policy_loop

__all__ = ["run_policy_loop", "run_react"]
