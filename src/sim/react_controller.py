"""Backward-compatible names for the former controller module."""

from .policy_loop_controller import (
    CodexPolicyLoopController,
    generate_bundle,
    policy_loop_prompt_for,
)

CodexReActController = CodexPolicyLoopController
react_prompt_for = policy_loop_prompt_for

__all__ = [
    "CodexPolicyLoopController",
    "CodexReActController",
    "generate_bundle",
    "policy_loop_prompt_for",
    "react_prompt_for",
]
