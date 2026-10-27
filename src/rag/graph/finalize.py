"""Finalization logic resolving final Outcome and recording to audit log."""

from rag.audit import AuditLogger
from rag.graph.state import RagState
from rag.schemas import Outcome, RunRecord


def finalize_run(state: RagState, audit_logger: AuditLogger | None = None) -> RagState:
    """Resolve outcome and write audit log."""
    last_failure = state.get("last_failure")
    answer = state.get("answer", "")
    citations = state.get("citations", [])

    # If already set by supervisor
    current_outcome = state.get("outcome")

    if current_outcome is None:
        if last_failure is not None:
            # In Phase 1 skeleton, unresolved failure routes to FAILED / ESCALATED
            if last_failure in ("llm_error", "retrieval_backend_down"):
                outcome = Outcome.ESCALATED
            elif last_failure in ("empty_retrieval", "low_relevance", "insufficient_context"):
                outcome = Outcome.ABSTAINED
            else:
                outcome = Outcome.FAILED
        elif answer and answer != "INSUFFICIENT CONTEXT":
            outcome = Outcome.ANSWERED
        else:
            outcome = Outcome.ABSTAINED
        state["outcome"] = outcome

    raw_outcome = state.get("outcome") or Outcome.FAILED
    outcome_enum = Outcome(raw_outcome) if isinstance(raw_outcome, str) else raw_outcome

    # Build RunRecord
    record = RunRecord(
        question=state.get("question", ""),
        outcome=outcome_enum,
        answer=answer,
        citations=citations,
        attempts=state.get("heal_attempts", 1),
        detected_faults=[str(state.get("last_failure"))] if state.get("last_failure") else [],
        heal_actions=[a for _, a in state.get("heal_history", [])],
        trace=state.get("trace", []),
        tokens_used=state.get("token_usage", 0),
        llm_calls=state.get("llm_calls", 0),
    )

    if audit_logger:
        audit_logger.log(record)

    return state
