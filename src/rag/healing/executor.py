"""Healing executor applying state patches and tracking recovery attempts."""

from typing import Any

from rag.healing.failures import FailureType, HealAction
from rag.healing.patterns import (
    apply_abstain,
    apply_bm25_only,
    apply_compress_context,
    apply_escalate,
    apply_model_fallback,
    apply_quarantine_chunk,
    apply_regenerate_strict,
    apply_retry_same,
    apply_rewrite_query,
    apply_vector_only,
    apply_widen_topk,
)

ACTION_HANDLERS = {
    HealAction.RETRY_SAME: apply_retry_same,
    HealAction.REWRITE_QUERY: apply_rewrite_query,
    HealAction.WIDEN_TOPK: apply_widen_topk,
    HealAction.BM25_ONLY: apply_bm25_only,
    HealAction.VECTOR_ONLY: apply_vector_only,
    HealAction.REGENERATE_STRICT: apply_regenerate_strict,
    HealAction.MODEL_FALLBACK: apply_model_fallback,
    HealAction.COMPRESS_CONTEXT: apply_compress_context,
    HealAction.QUARANTINE_CHUNK: apply_quarantine_chunk,
    HealAction.ABSTAIN: apply_abstain,
    HealAction.ESCALATE: apply_escalate,
}


def execute_heal(
    state: dict[str, Any],
    action: HealAction,
    failure: FailureType,
) -> dict[str, Any]:
    """Execute a healing action by computing and applying a state patch.

    Increments heal_attempts, logs (failure, action) into heal_history,
    clears last_failure for non-terminal actions, and applies pattern patch.

    Args:
        state: Current pipeline state dictionary.
        action: Selected recovery action.
        failure: Failure type being mitigated.

    Returns:
        Updated state dictionary.
    """
    handler = ACTION_HANDLERS.get(action, apply_escalate)
    patch = handler(state)

    updated = dict(state)
    updated.update(patch)

    history = list(state.get("heal_history", []))
    history.append((str(failure.value), str(action.value)))
    updated["heal_history"] = history
    updated["heal_attempts"] = int(state.get("heal_attempts", 0)) + 1
    updated["last_action"] = action

    # Clear last_failure if non-terminal so the next attempt can proceed cleanly
    if action not in (HealAction.ABSTAIN, HealAction.ESCALATE):
        updated["last_failure"] = None

    return updated
