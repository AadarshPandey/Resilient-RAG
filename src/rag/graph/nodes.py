"""Pipeline graph nodes with trace event logging."""

from typing import Any

from langchain_core.runnables import RunnableConfig

from rag.errors import MalformedOutputError
from rag.graph.finalize import finalize_run
from rag.graph.parsing import parse_defensive_json
from rag.graph.prompts import (
    CLAIM_VERIFICATION_PROMPT,
    GROUNDED_GENERATION_PROMPT,
    STRICT_GENERATION_PROMPT,
)
from rag.graph.state import RagState
from rag.healing.detector import detect_retrieval_failure
from rag.schemas import (
    Citation,
    Claim,
    FailureType,
    RetrievalMode,
    RetrievedChunk,
    TraceEvent,
    VerificationResult,
)
from rag.security import check_system_prompt_leakage, sanitize_retrieved_text


def _get_components(config: RunnableConfig | None) -> Any:
    """Retrieve components from RunnableConfig."""
    if config and "configurable" in config and "components" in config["configurable"]:
        return config["configurable"]["components"]
    return None


def query_prep_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Prepare and normalize the search query."""
    components = _get_components(config)
    clock = components.clock if components else None
    t = clock.now() if clock else 0.0

    question = state.get("question", "").strip()
    rewritten = state.get("rewritten_query") or question

    event = TraceEvent(node="query_prep", timestamp=t, details={"query": rewritten})
    trace = list(state.get("trace", [])) + [event]

    return {
        "rewritten_query": rewritten,
        "trace": trace,
        "heal_attempts": state.get("heal_attempts", 0),
        "heal_history": state.get("heal_history", []),
        "quarantined_ids": state.get("quarantined_ids", []),
        "strict_mode": state.get("strict_mode", False),
        "llm_tier": state.get("llm_tier", 0),
        "token_usage": state.get("token_usage", 0),
        "llm_calls": state.get("llm_calls", 0),
    }


def retrieve_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Execute candidate retrieval using configured mode."""
    components = _get_components(config)
    clock = components.clock if components else None
    t = clock.now() if clock else 0.0

    query = state.get("rewritten_query") or state["question"]
    raw_mode = state.get("retrieval_mode", RetrievalMode.HYBRID)
    mode = RetrievalMode(raw_mode) if isinstance(raw_mode, str) else raw_mode
    top_k = state.get("top_k", 5)
    quarantined = set(state.get("quarantined_ids", []))

    trace = list(state.get("trace", []))

    if not components or not components.retriever:
        trace.append(TraceEvent(node="retrieve", timestamp=t, details={"error": "no retriever"}))
        return {
            "last_failure": FailureType.RETRIEVAL_BACKEND_DOWN,
            "trace": trace,
        }

    try:
        retrieved = components.retriever.retrieve(query=query, mode=mode, top_k=top_k)
    except Exception as e:
        trace.append(TraceEvent(node="retrieve", timestamp=t, details={"error": str(e)}))
        return {
            "last_failure": FailureType.RETRIEVAL_BACKEND_DOWN,
            "error_message": str(e),
            "trace": trace,
        }

    # Filter out quarantined chunks
    valid_retrieved: list[RetrievedChunk] = [c for c in retrieved if c.chunk.id not in quarantined]

    if not valid_retrieved:
        trace.append(TraceEvent(node="retrieve", timestamp=t, details={"status": "empty"}))
        return {
            "chunks": [],
            "retrieved_chunks": [],
            "last_failure": FailureType.EMPTY_RETRIEVAL,
            "trace": trace,
        }

    chunks = [r.chunk for r in valid_retrieved]

    # Check for prompt injection in candidate chunks
    fail = detect_retrieval_failure(chunks=chunks)
    if fail == FailureType.PROMPT_INJECTION:
        trace.append(
            TraceEvent(
                node="retrieve",
                timestamp=t,
                details={"failure": "prompt_injection"},
            )
        )
        return {
            "chunks": chunks,
            "retrieved_chunks": valid_retrieved,
            "last_failure": FailureType.PROMPT_INJECTION,
            "trace": trace,
        }

    trace.append(
        TraceEvent(
            node="retrieve",
            timestamp=t,
            details={"count": len(chunks), "mode": str(mode)},
        )
    )

    return {
        "chunks": chunks,
        "retrieved_chunks": valid_retrieved,
        "last_failure": None,
        "trace": trace,
    }


def grade_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Assess relevance of retrieved chunks."""
    components = _get_components(config)
    clock = components.clock if components else None
    t = clock.now() if clock else 0.0

    trace = list(state.get("trace", []))
    chunks = state.get("chunks", [])

    if not chunks:
        trace.append(TraceEvent(node="grade", timestamp=t, details={"failure": "empty_chunks"}))
        return {
            "last_failure": FailureType.EMPTY_RETRIEVAL,
            "trace": trace,
        }

    # Check relevance scores if available
    retrieved = state.get("retrieved_chunks", [])
    if retrieved:
        avg_score = sum(r.score for r in retrieved) / len(retrieved)
        # If all scores are effectively zero, flag low relevance
        if avg_score <= 0.01:
            trace.append(
                TraceEvent(
                    node="grade",
                    timestamp=t,
                    details={"failure": "low_relevance", "avg_score": avg_score},
                )
            )
            return {
                "last_failure": FailureType.LOW_RELEVANCE,
                "trace": trace,
            }

    trace.append(TraceEvent(node="grade", timestamp=t, details={"count": len(chunks)}))
    return {
        "last_failure": None,
        "trace": trace,
    }


def generate_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Generate answer from retrieved chunks using Tier-0 (or fallback) LLM."""
    components = _get_components(config)
    clock = components.clock if components else None
    t = clock.now() if clock else 0.0

    trace = list(state.get("trace", []))
    chunks = state.get("chunks", [])
    question = state["question"]
    strict = state.get("strict_mode", False)
    tier = state.get("llm_tier", 0)

    context_str = "\n\n".join(f"[{c.id}]: {sanitize_retrieved_text(c.text)}" for c in chunks)
    template = STRICT_GENERATION_PROMPT if strict else GROUNDED_GENERATION_PROMPT
    prompt = template.format(context=context_str, question=question)

    messages = [{"role": "user", "content": prompt}]

    try:
        llm_client = components.llm_client if components else None
        if not llm_client:
            raise RuntimeError("LLM client not configured")

        llm_result = llm_client.invoke(messages, tier=tier)
    except Exception as e:
        trace.append(TraceEvent(node="generate", timestamp=t, details={"error": str(e)}))
        return {
            "last_failure": FailureType.LLM_ERROR,
            "error_message": str(e),
            "trace": trace,
        }

    token_usage = (
        state.get("token_usage", 0) + llm_result.prompt_tokens + llm_result.completion_tokens
    )
    llm_calls = state.get("llm_calls", 0) + 1

    try:
        parsed = parse_defensive_json(llm_result.content)
    except MalformedOutputError as e:
        trace.append(
            TraceEvent(node="generate", timestamp=t, details={"error": "malformed_output"})
        )
        return {
            "last_failure": FailureType.MALFORMED_OUTPUT,
            "error_message": str(e),
            "token_usage": token_usage,
            "llm_calls": llm_calls,
            "trace": trace,
        }

    answer = str(parsed.get("answer", ""))
    if check_system_prompt_leakage(answer):
        trace.append(
            TraceEvent(
                node="generate",
                timestamp=t,
                details={"failure": "prompt_injection", "reason": "system_prompt_leakage"},
            )
        )
        return {
            "last_failure": FailureType.PROMPT_INJECTION,
            "error_message": "System prompt leakage detected in generated response",
            "token_usage": token_usage,
            "llm_calls": llm_calls,
            "trace": trace,
        }

    claims = list(parsed.get("claims", []))
    raw_citations = list(parsed.get("citations", []))

    citations: list[Citation] = []
    chunk_lookup = {c.id: c for c in chunks}
    for c_id in raw_citations:
        if c_id in chunk_lookup:
            citations.append(Citation(chunk_id=c_id, text=chunk_lookup[c_id].text))
        else:
            citations.append(Citation(chunk_id=str(c_id), text=""))

    trace.append(TraceEvent(node="generate", timestamp=t, details={"citations": len(citations)}))

    return {
        "answer": answer,
        "claims": claims,
        "citations": citations,
        "last_failure": None,
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "trace": trace,
    }


def verify_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Verify answer groundedness against retrieved context passages."""
    components = _get_components(config)
    clock = components.clock if components else None
    t = clock.now() if clock else 0.0

    trace = list(state.get("trace", []))
    answer = state.get("answer", "")
    claims = state.get("claims", [])
    chunks = state.get("chunks", [])
    tier = state.get("llm_tier", 0)

    if not claims or not answer or answer == "INSUFFICIENT CONTEXT":
        trace.append(TraceEvent(node="verify", timestamp=t, details={"skip": "no_claims"}))
        return {
            "last_failure": None,
            "trace": trace,
        }

    context_str = "\n\n".join(f"[{c.id}]: {c.text}" for c in chunks)
    claims_str = "\n".join(f"- {c}" for c in claims)
    prompt = CLAIM_VERIFICATION_PROMPT.format(
        context=context_str,
        answer=answer,
        claims=claims_str,
    )
    messages = [{"role": "user", "content": prompt}]

    try:
        llm_client = components.llm_client if components else None
        if not llm_client:
            raise RuntimeError("LLM client not configured")
        llm_result = llm_client.invoke(messages, tier=tier)
    except Exception as e:
        trace.append(TraceEvent(node="verify", timestamp=t, details={"error": str(e)}))
        return {
            "last_failure": FailureType.LLM_ERROR,
            "error_message": str(e),
            "trace": trace,
        }

    token_usage = (
        state.get("token_usage", 0) + llm_result.prompt_tokens + llm_result.completion_tokens
    )
    llm_calls = state.get("llm_calls", 0) + 1

    try:
        parsed = parse_defensive_json(llm_result.content)
    except MalformedOutputError:
        trace.append(TraceEvent(node="verify", timestamp=t, details={"error": "malformed_output"}))
        return {
            "last_failure": FailureType.MALFORMED_OUTPUT,
            "token_usage": token_usage,
            "llm_calls": llm_calls,
            "trace": trace,
        }

    is_grounded = bool(parsed.get("is_grounded", True))
    unsupported = list(parsed.get("unsupported_claims", []))
    summary = str(parsed.get("summary", ""))

    claim_objs = [
        Claim(
            claim=c,
            grounded=c not in unsupported,
            reasoning=summary,
        )
        for c in claims
    ]
    verification = VerificationResult(
        is_grounded=is_grounded,
        claims=claim_objs,
        summary=summary,
    )

    last_failure = None if is_grounded else FailureType.UNGROUNDED_ANSWER

    trace.append(
        TraceEvent(
            node="verify",
            timestamp=t,
            details={"is_grounded": is_grounded, "unsupported": len(unsupported)},
        )
    )

    return {
        "verification": verification,
        "last_failure": last_failure,
        "token_usage": token_usage,
        "llm_calls": llm_calls,
        "trace": trace,
    }


def finalize_node(state: RagState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Execute finalization and persistence."""
    components = _get_components(config)
    audit_logger = components.audit_logger if components else None
    updated_state = finalize_run(state, audit_logger=audit_logger)
    return dict(updated_state)
