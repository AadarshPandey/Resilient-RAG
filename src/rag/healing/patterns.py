"""Pure state-patch functions implementing the deterministic recovery actions."""

from typing import Any

from rag.healing.detector import detect_prompt_injection
from rag.schemas import Chunk, Outcome, RetrievalMode


def apply_retry_same(state: dict[str, Any]) -> dict[str, Any]:
    """Retry same configuration without altering parameters."""
    return {}


def apply_rewrite_query(state: dict[str, Any]) -> dict[str, Any]:
    """Expand or rewrite search query with key terminology."""
    current_q = state.get("rewritten_query") or state.get("question", "")
    # Deterministic query normalization and expansion
    rewritten = current_q.strip()
    if not rewritten.endswith("overview details"):
        rewritten = f"{rewritten} overview details"
    return {"rewritten_query": rewritten}


def apply_widen_topk(state: dict[str, Any]) -> dict[str, Any]:
    """Double candidate retrieval budget."""
    current_k = int(state.get("top_k", 5))
    return {"top_k": current_k * 2}


def apply_bm25_only(state: dict[str, Any]) -> dict[str, Any]:
    """Switch retrieval to keyword BM25 only."""
    return {"retrieval_mode": RetrievalMode.BM25_ONLY}


def apply_vector_only(state: dict[str, Any]) -> dict[str, Any]:
    """Switch retrieval to vector search only."""
    return {"retrieval_mode": RetrievalMode.VECTOR_ONLY}


def apply_regenerate_strict(state: dict[str, Any]) -> dict[str, Any]:
    """Enable strict generation mode prohibiting extrapolation."""
    return {"strict_mode": True}


def apply_model_fallback(state: dict[str, Any]) -> dict[str, Any]:
    """Advance to next model tier (e.g. tier-0 -> tier-1) and reset strict mode."""
    current_tier = int(state.get("llm_tier", 0))
    return {
        "llm_tier": current_tier + 1,
        "strict_mode": False,
    }


def apply_compress_context(state: dict[str, Any]) -> dict[str, Any]:
    """Compress context passages by halving character length deterministically."""
    chunks: list[Chunk] = state.get("chunks", [])
    compressed: list[Chunk] = []
    for c in chunks:
        half_len = max(len(c.text) // 2, 50)
        compressed.append(
            Chunk(
                id=c.id,
                doc_id=c.doc_id,
                text=c.text[:half_len],
                index=c.index,
                metadata=c.metadata,
            )
        )
    return {"chunks": compressed}


def apply_quarantine_chunk(state: dict[str, Any]) -> dict[str, Any]:
    """Identify and quarantine poisoned or injecting chunks."""
    chunks: list[Chunk] = state.get("chunks", [])
    quarantined = list(state.get("quarantined_ids", []))

    for c in chunks:
        if detect_prompt_injection(c.text) and c.id not in quarantined:
            quarantined.append(c.id)

    # Filter out quarantined chunks from state
    quarantine_set = set(quarantined)
    clean_chunks = [c for c in chunks if c.id not in quarantine_set]

    return {
        "quarantined_ids": quarantined,
        "chunks": clean_chunks,
    }


def apply_abstain(state: dict[str, Any]) -> dict[str, Any]:
    """Produce safe terminal ABSTAINED outcome."""
    return {
        "outcome": Outcome.ABSTAINED,
        "answer": "I do not have sufficient reliable information to answer this question.",
    }


def apply_escalate(state: dict[str, Any]) -> dict[str, Any]:
    """Produce safe terminal ESCALATED outcome."""
    return {
        "outcome": Outcome.ESCALATED,
        "answer": "This request requires escalation to human engineering support.",
    }
