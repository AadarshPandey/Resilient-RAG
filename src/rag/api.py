"""FastAPI application for Resilient RAG."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.bootstrap import Components, build_components
from rag.config import get_settings
from rag.schemas import Outcome


class AskRequest(BaseModel):
    """Request payload for /ask endpoint."""

    question: str = Field(..., description="The user query to answer")
    thread_id: str | None = Field(default=None, description="Optional conversation/thread ID")
    healing_enabled: bool = Field(default=True, description="Whether self-healing is active")
    retrieval_mode: str = Field(
        default="hybrid", description="Retrieval mode: hybrid, vector, bm25"
    )
    top_k: int = Field(default=5, description="Number of chunks to retrieve")


class CitationItem(BaseModel):
    """Citation item in answer response."""

    chunk_id: str
    text: str
    score: float | None = None


class HealingReport(BaseModel):
    """Summary of healing actions attempted during execution."""

    heal_attempts: int = 0
    heal_history: list[list[str]] = Field(default_factory=list)
    quarantined_ids: list[str] = Field(default_factory=list)


class TraceItem(BaseModel):
    """Trace event captured along the execution pipeline."""

    node: str
    timestamp: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    """Response payload for /ask endpoint."""

    question: str
    answer: str
    outcome: str
    citations: list[CitationItem] = Field(default_factory=list)
    healing_report: HealingReport
    trace: list[TraceItem] = Field(default_factory=list)
    llm_calls: int = 0
    token_usage: int = 0


class HealthResponse(BaseModel):
    """Health check status response."""

    status: str = "ok"
    version: str = "0.1.0"


def create_app(components: Components | None = None) -> FastAPI:
    """Create and configure FastAPI application instance."""
    app = FastAPI(
        title="Resilient RAG API",
        description="Production-grade self-healing RAG API with fault tolerance",
        version="0.1.0",
    )

    if components is not None:
        app.state.components = components
    else:
        app.state.components = build_components(get_settings())

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version="0.1.0")

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        active_components = getattr(app.state, "components", None)
        if not isinstance(active_components, Components):
            raise HTTPException(status_code=500, detail="Components not initialized")

        final_state = active_components.run(
            question=request.question,
            thread_id=request.thread_id,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            healing_enabled=request.healing_enabled,
        )

        # Parse citations
        raw_citations = final_state.get("citations", [])
        citations = [
            CitationItem(
                chunk_id=c.chunk_id,
                text=c.text,
                score=c.score,
            )
            for c in raw_citations
        ]

        # Parse trace
        raw_trace = final_state.get("trace", [])
        trace = [
            TraceItem(
                node=t.node,
                timestamp=round(t.timestamp, 4),
                details=t.details,
            )
            for t in raw_trace
        ]

        # Parse outcome
        raw_outcome = final_state.get("outcome")
        outcome_str = (
            raw_outcome.value if isinstance(raw_outcome, Outcome) else str(raw_outcome or "failed")
        )

        # Parse heal history
        raw_history = final_state.get("heal_history", [])
        history_list = [[str(pair[0]), str(pair[1])] for pair in raw_history]

        return AskResponse(
            question=request.question,
            answer=final_state.get("answer", ""),
            outcome=outcome_str,
            citations=citations,
            healing_report=HealingReport(
                heal_attempts=final_state.get("heal_attempts", 0),
                heal_history=history_list,
                quarantined_ids=final_state.get("quarantined_ids", []),
            ),
            trace=trace,
            llm_calls=final_state.get("llm_calls", 0),
            token_usage=final_state.get("token_usage", 0),
        )

    return app
