"""FastAPI application for Resilient RAG with async endpoints, concurrency bounds, timeouts, and ingestion queue."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from rag.bootstrap import Components, build_components
from rag.config import Settings, get_settings
from rag.ingest.pipeline import IngestionPipeline
from rag.ingest.queue import (
    InMemoryJobQueue,
    JobQueue,
    create_job_queue,
)
from rag.logging_setup import (
    current_request_id,
    get_current_request_id,
    set_current_request_id,
)
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


class IngestRequest(BaseModel):
    """Payload to trigger background ingestion."""

    input_dir: str = Field(default="data/raw", description="Directory of files to ingest")
    force: bool = Field(default=False, description="Force re-indexing ignoring manifest")


class IngestResponse(BaseModel):
    """Response returned upon enqueuing an ingestion job."""

    job_id: str
    status: str = "queued"


class IngestJobStatusResponse(BaseModel):
    """Detailed status report of an ingestion job."""

    job_id: str
    input_dir: str
    status: str
    attempts: int = 0
    max_attempts: int = 3
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float
    updated_at: float


def _get_semaphore(app: FastAPI) -> asyncio.Semaphore:
    """Retrieve or create application-level concurrency semaphore."""
    if not hasattr(app.state, "semaphore") or app.state.semaphore is None:
        limit = getattr(app.state, "settings", get_settings()).api_concurrency_limit
        app.state.semaphore = asyncio.Semaphore(limit)
    return app.state.semaphore  # type: ignore[no-any-return]


def create_app(
    components: Components | None = None,
    settings: Settings | None = None,
    queue: JobQueue | None = None,
    pipeline: IngestionPipeline | None = None,
) -> FastAPI:
    """Create and configure FastAPI application instance with async handlers and ingestion queue."""
    cfg = settings or (components.settings if components else get_settings())

    app = FastAPI(
        title="Resilient RAG API",
        description="Production-grade self-healing RAG API with fault tolerance",
        version="0.1.0",
    )

    app.state.settings = cfg
    if components is not None:
        app.state.components = components
    else:
        app.state.components = build_components(cfg)

    # Ingestion queue and pipeline setup
    if queue is not None:
        app.state.queue = queue
    else:
        app.state.queue = create_job_queue(cfg.redis_url)

    if pipeline is not None:
        app.state.pipeline = pipeline
    else:
        try:
            app.state.pipeline = IngestionPipeline(
                embedder=app.state.components.embedder,
                vector_store=app.state.components.vector_store,
                bm25_index=app.state.components.bm25_index,
            )
        except Exception:
            app.state.pipeline = None

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = set_current_request_id(req_id)
        try:
            response: Response = await call_next(request)
            response.headers["X-Request-ID"] = req_id
            return response
        finally:
            current_request_id.reset(token)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version="0.1.0")

    @app.post("/ask", response_model=AskResponse)
    async def ask(request: AskRequest, raw_req: Request) -> AskResponse:
        active_components = getattr(app.state, "components", None)
        if not isinstance(active_components, Components):
            raise HTTPException(status_code=500, detail="Components not initialized")

        req_id = raw_req.headers.get("X-Request-ID") or get_current_request_id() or uuid.uuid4().hex
        sem = _get_semaphore(app)
        timeout_sec = getattr(app.state, "settings", cfg).api_timeout_seconds

        try:
            async with asyncio.timeout(timeout_sec):
                async with sem:
                    final_state = await asyncio.to_thread(
                        active_components.run,
                        question=request.question,
                        thread_id=request.thread_id,
                        retrieval_mode=request.retrieval_mode,
                        top_k=request.top_k,
                        healing_enabled=request.healing_enabled,
                        request_id=req_id,
                    )
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Request timed out") from None

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

    @app.post("/ingest", response_model=IngestResponse, status_code=202)
    async def ingest(request: IngestRequest) -> IngestResponse:
        q = getattr(app.state, "queue", None)
        if not isinstance(q, (JobQueue, InMemoryJobQueue)):
            raise HTTPException(status_code=500, detail="Ingestion queue not initialized")

        job_id = q.enqueue(input_dir=request.input_dir, force=request.force)
        return IngestResponse(job_id=job_id, status="queued")

    @app.get("/ingest/{job_id}", response_model=IngestJobStatusResponse)
    async def get_ingest_job(job_id: str) -> IngestJobStatusResponse:
        q = getattr(app.state, "queue", None)
        if not isinstance(q, (JobQueue, InMemoryJobQueue)):
            raise HTTPException(status_code=500, detail="Ingestion queue not initialized")

        job = q.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

        status_str = job.status.value if hasattr(job.status, "value") else str(job.status)
        return IngestJobStatusResponse(
            job_id=job.id,
            input_dir=job.input_dir,
            status=status_str,
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            result=job.result,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )

    return app
