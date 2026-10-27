"""Deterministic healing policy selector adhering strictly to Section 3.4."""

from rag.healing.failures import FailureType, HealAction

# Policy table ordered low-cost-first
POLICY_TABLE: dict[FailureType, list[HealAction]] = {
    FailureType.RETRIEVAL_BACKEND_DOWN: [
        HealAction.RETRY_SAME,
        HealAction.BM25_ONLY,
        HealAction.VECTOR_ONLY,
        HealAction.ESCALATE,
    ],
    FailureType.EMPTY_RETRIEVAL: [
        HealAction.WIDEN_TOPK,
        HealAction.REWRITE_QUERY,
        HealAction.ABSTAIN,
    ],
    FailureType.PROMPT_INJECTION: [
        HealAction.QUARANTINE_CHUNK,
        HealAction.ABSTAIN,
    ],
    FailureType.LOW_RELEVANCE: [
        HealAction.REWRITE_QUERY,
        HealAction.WIDEN_TOPK,
        HealAction.ABSTAIN,
    ],
    FailureType.CONTEXT_OVERFLOW: [
        HealAction.COMPRESS_CONTEXT,
        HealAction.MODEL_FALLBACK,
        HealAction.ABSTAIN,
    ],
    FailureType.LLM_ERROR: [
        HealAction.RETRY_SAME,
        HealAction.MODEL_FALLBACK,
        HealAction.ESCALATE,
    ],
    FailureType.MALFORMED_OUTPUT: [
        HealAction.RETRY_SAME,
        HealAction.REGENERATE_STRICT,
        HealAction.MODEL_FALLBACK,
        HealAction.ABSTAIN,
    ],
    FailureType.INSUFFICIENT_CONTEXT: [
        HealAction.WIDEN_TOPK,
        HealAction.REWRITE_QUERY,
        HealAction.ABSTAIN,
    ],
    FailureType.UNGROUNDED_ANSWER: [
        HealAction.REGENERATE_STRICT,
        HealAction.MODEL_FALLBACK,
        HealAction.ABSTAIN,
    ],
}


def select_heal_action(
    failure: FailureType | str,
    history: list[tuple[str, str]],
    attempts: int,
    max_attempts: int = 4,
    healing_enabled: bool = True,
) -> HealAction:
    """Select the next bounded, deterministic healing action.

    Rules:
    1. If healing_enabled is False, immediately return terminal action.
    2. If attempts >= max_attempts, return terminal action.
    3. Iterate through ordered actions; pick the first (failure, action) pair not yet executed.
    4. If all actions in the list were attempted, return terminal action.

    Args:
        failure: The detected FailureType.
        history: Sequence of executed (failure_type, action) pairs.
        attempts: Current cumulative count of heal attempts.
        max_attempts: Hard cap on healing attempts per query (default: 4).
        healing_enabled: Master switch for recovery behavior.

    Returns:
        The selected HealAction.
    """
    failure_enum = FailureType(failure) if isinstance(failure, str) else failure
    candidate_actions = POLICY_TABLE.get(
        failure_enum,
        [HealAction.ESCALATE],
    )
    terminal_action = candidate_actions[-1]

    if not healing_enabled or attempts >= max_attempts:
        return terminal_action

    norm_history = {(str(f), str(a)) for f, a in history}

    for action in candidate_actions:
        pair = (str(failure_enum.value), str(action.value))
        if pair not in norm_history:
            return action

    return terminal_action
