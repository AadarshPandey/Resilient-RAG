"""Conditional edge routing logic for the pipeline graph."""

from typing import Literal

from rag.graph.state import RagState


def _can_heal(state: RagState) -> bool:
    """Check if healing supervisor is enabled."""
    return bool(state.get("healing_enabled", True))


def route_after_retrieve(state: RagState) -> Literal["grade", "heal", "finalize"]:
    """Route after retrieve node."""
    if state.get("last_failure") is not None:
        if _can_heal(state):
            return "heal"
        return "finalize"
    return "grade"


def route_after_grade(state: RagState) -> Literal["generate", "heal", "finalize"]:
    """Route after grade node."""
    if state.get("last_failure") is not None:
        if _can_heal(state):
            return "heal"
        return "finalize"
    return "generate"


def route_after_generate(state: RagState) -> Literal["verify", "heal", "finalize"]:
    """Route after generate node."""
    if state.get("last_failure") is not None:
        if _can_heal(state):
            return "heal"
        return "finalize"
    return "verify"


def route_after_verify(state: RagState) -> Literal["heal", "finalize"]:
    """Route after verify node."""
    if state.get("last_failure") is not None and _can_heal(state):
        return "heal"
    return "finalize"
