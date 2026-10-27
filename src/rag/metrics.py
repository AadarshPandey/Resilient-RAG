"""Prometheus observability collectors and recording utilities for Resilient RAG."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Total requests processed by final pipeline outcome
RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Total requests processed by the Resilient RAG pipeline.",
    ["outcome"],
)

# End-to-end request duration histogram
RAG_REQUEST_DURATION = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request latency in seconds.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

# Healing attempts executed partitioned by action and failure cause
RAG_HEAL_ATTEMPTS = Counter(
    "rag_heal_attempts_total",
    "Total self-healing attempts executed by the supervisor.",
    ["action", "failure"],
)

# Circuit breaker status per LLM tier (0=closed, 1=half_open, 2=open)
RAG_BREAKER_STATE = Gauge(
    "rag_circuit_breaker_state",
    "Circuit breaker state per LLM tier (0=closed, 1=half_open, 2=open).",
    ["tier"],
)

# LLM tokens consumed partitioned by tier and direction (prompt vs completion)
RAG_LLM_TOKENS = Counter(
    "rag_llm_tokens_total",
    "Total LLM tokens consumed across calls.",
    ["tier", "direction"],
)


def record_ask_metrics(
    outcome: str,
    duration: float,
    heal_history: list[list[str]] | list[tuple[str, str]] | None = None,
    llm_tier: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> None:
    """Record Prometheus metrics for a completed /ask execution."""
    RAG_REQUESTS.labels(outcome=outcome.lower()).inc()
    RAG_REQUEST_DURATION.observe(duration)

    if heal_history:
        for item in heal_history:
            if len(item) == 2:
                fail_type, act = item[0], item[1]
                RAG_HEAL_ATTEMPTS.labels(
                    action=str(act).lower(),
                    failure=str(fail_type).lower(),
                ).inc()

    if prompt_tokens > 0:
        RAG_LLM_TOKENS.labels(tier=str(llm_tier), direction="prompt").inc(prompt_tokens)
    if completion_tokens > 0:
        RAG_LLM_TOKENS.labels(tier=str(llm_tier), direction="completion").inc(completion_tokens)


def get_prometheus_metrics() -> tuple[bytes, str]:
    """Generate Prometheus exposition text and content-type header."""
    return generate_latest(), CONTENT_TYPE_LATEST
