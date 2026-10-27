"""Graph heal node and routing back into the execution flow."""

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from rag.graph.state import RagState
from rag.healing.executor import execute_heal
from rag.healing.failures import FailureType, HealAction
from rag.healing.selector import select_heal_action
from rag.schemas import TraceEvent


def _get_components(config: RunnableConfig | None) -> Any:
    if config and "configurable" in config and "components" in config["configurable"]:
        return config["configurable"]["components"]
    return None


def heal_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Healing supervisor node: detects failure, selects action, and applies patch."""
    components = _get_components(config)
    clock = components.clock if components else None
    t = clock.now() if clock else 0.0

    healing_enabled = (
        config["configurable"].get("healing_enabled", True)
        if config and "configurable" in config
        else True
    )
    max_attempts = (
        components.settings.max_heal_attempts if components and components.settings else 4
    )

    failure = state.get("last_failure") or FailureType.LLM_ERROR
    failure_enum = FailureType(failure) if isinstance(failure, str) else failure

    history = state.get("heal_history", [])
    attempts = state.get("heal_attempts", 0)

    # Pure deterministic action selection
    action = select_heal_action(
        failure=failure_enum,
        history=history,
        attempts=attempts,
        max_attempts=max_attempts,
        healing_enabled=healing_enabled,
    )

    # Pure executor state patch
    updated = execute_heal(dict(state), action=action, failure=failure_enum)

    trace = list(state.get("trace", []))
    trace.append(
        TraceEvent(
            node="heal",
            timestamp=t,
            details={
                "failure": str(failure_enum.value),
                "action": str(action.value),
                "attempt": attempts + 1,
            },
        )
    )
    updated["trace"] = trace
    return updated


def route_after_heal(
    state: RagState,
) -> Literal["retrieve", "grade", "generate", "verify", "finalize"]:
    """Route execution back to the appropriate pipeline node after healing."""
    action = state.get("last_action")
    if action is None:
        return "finalize"

    action_enum = HealAction(action) if isinstance(action, str) else action

    if action_enum in (HealAction.ABSTAIN, HealAction.ESCALATE):
        return "finalize"

    if action_enum in (
        HealAction.BM25_ONLY,
        HealAction.VECTOR_ONLY,
        HealAction.WIDEN_TOPK,
        HealAction.REWRITE_QUERY,
        HealAction.QUARANTINE_CHUNK,
    ):
        return "retrieve"

    if action_enum in (
        HealAction.REGENERATE_STRICT,
        HealAction.MODEL_FALLBACK,
        HealAction.COMPRESS_CONTEXT,
    ):
        return "generate"

    if action_enum == HealAction.RETRY_SAME:
        # Determine from trace which node failed
        trace = state.get("trace", [])
        if len(trace) >= 2:
            last_node = trace[-2].node
            if last_node in ("retrieve", "generate", "verify"):
                return last_node  # type: ignore[return-value]
        return "generate"

    return "finalize"
