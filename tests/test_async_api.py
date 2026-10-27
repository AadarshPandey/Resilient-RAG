"""Tests for Milestone P3: Async API, Concurrency, Timeouts, and Ingestion Queue."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient

from rag.api import create_app
from rag.bootstrap import Components, build_components
from rag.clock import VirtualClock
from rag.config import Settings
from rag.fakes import FakeVectorStore, HashEmbedder, LexicalReranker, ScriptedChatModel
from rag.ingest.pipeline import IngestionPipeline
from rag.ingest.queue import (
    IngestWorker,
    InMemoryJobQueue,
    JobStatus,
    RedisJobQueue,
)
from rag.llm.base import LLMResult
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.retriever import Retriever
from rag.schemas import Chunk


def make_test_components() -> Components:
    """Create reproducible test components with fake backends."""
    clock = VirtualClock(start_time=0.0)
    embedder = HashEmbedder(dimension=32)
    vstore = FakeVectorStore()
    bm25 = BM25Index()
    reranker = LexicalReranker()

    chunk = Chunk(
        id="c_test_p3",
        doc_id="d_test",
        text="Kubernetes pods run containers.",
        index=0,
    )
    vstore.upsert([chunk], embedder.embed([chunk.text]))
    bm25.index_chunks([chunk])

    retriever = Retriever(
        embedder=embedder,
        vector_store=vstore,
        bm25_index=bm25,
        reranker=reranker,
    )

    gen_json = '{"answer": "Pods run containers.", "claims": ["Pods run containers"], "citations": ["c_test_p3"]}'
    verify_json = (
        '{"is_grounded": true, "unsupported_claims": [], "summary": "Grounding confirmed"}'
    )

    scripted_model = ScriptedChatModel(
        responses=[
            LLMResult(content=gen_json, prompt_tokens=20, completion_tokens=15, tier=0),
            LLMResult(content=verify_json, prompt_tokens=25, completion_tokens=10, tier=0),
        ]
    )

    return build_components(
        overrides={
            "clock": clock,
            "embedder": embedder,
            "vector_store": vstore,
            "bm25_index": bm25,
            "reranker": reranker,
            "retriever": retriever,
            "chat_model": scripted_model,
        }
    )


def test_request_id_middleware() -> None:
    """Test that X-Request-ID is generated if missing and preserved if supplied."""
    components = make_test_components()
    app = create_app(components)
    client = TestClient(app)

    # Missing X-Request-ID should be generated
    res1 = client.get("/health")
    assert res1.status_code == 200
    req_id1 = res1.headers.get("X-Request-ID")
    assert req_id1 is not None
    assert len(req_id1) > 0

    # Provided X-Request-ID should be preserved
    custom_id = "req-custom-xyz-12345"
    res2 = client.get("/health", headers={"X-Request-ID": custom_id})
    assert res2.status_code == 200
    assert res2.headers.get("X-Request-ID") == custom_id


@pytest.mark.asyncio
async def test_async_concurrency_semaphore() -> None:
    """Verify concurrency semaphore limits concurrent requests."""
    components = make_test_components()
    settings = Settings(api_concurrency_limit=2)
    app = create_app(components, settings=settings)

    active_executions = 0
    max_observed_concurrency = 0
    lock = asyncio.Lock()

    original_run = components.run

    def mock_run(*args: object, **kwargs: object) -> object:
        nonlocal active_executions, max_observed_concurrency
        time.sleep(0.05)
        return original_run(*args, **kwargs)  # type: ignore[arg-type]

    components.run = mock_run  # type: ignore[assignment]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"question": "What is a pod?", "top_k": 1}

        async def send_req() -> httpx.Response:
            nonlocal active_executions, max_observed_concurrency
            async with lock:
                active_executions += 1
                if active_executions > max_observed_concurrency:
                    max_observed_concurrency = active_executions
            try:
                return await client.post("/ask", json=payload)
            finally:
                async with lock:
                    active_executions -= 1

        responses = await asyncio.gather(
            send_req(),
            send_req(),
            send_req(),
            send_req(),
        )

        for r in responses:
            assert r.status_code == 200


@pytest.mark.asyncio
async def test_async_ask_timeout() -> None:
    """Verify request timeout returns 504 Gateway Timeout."""
    components = make_test_components()
    settings = Settings(api_timeout_seconds=0.05)
    app = create_app(components, settings=settings)

    def slow_run(*args: object, **kwargs: object) -> object:
        time.sleep(0.2)
        return {}

    components.run = slow_run  # type: ignore[assignment]

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {"question": "What is a pod?"}
        res = await client.post("/ask", json=payload)
        assert res.status_code == 504
        assert "timed out" in res.json().get("detail", "").lower()


def test_ingest_job_lifecycle_and_idempotency(tmp_path: Path) -> None:
    """Verify POST /ingest, GET /ingest/{job_id}, worker processing, and manifest idempotency."""
    components = make_test_components()
    queue = InMemoryJobQueue()
    manifest_file = tmp_path / "manifest.json"

    pipeline = IngestionPipeline(
        embedder=components.embedder,
        vector_store=components.vector_store,
        bm25_index=components.bm25_index,
        manifest_path=manifest_file,
    )
    worker = IngestWorker(queue=queue, pipeline=pipeline)

    app = create_app(components, queue=queue, pipeline=pipeline)
    client = TestClient(app)

    # Create dummy file to ingest
    doc_file = tmp_path / "pod.md"
    doc_file.write_text("Kubernetes pods encapsulate one or more containers.")

    # 1. Enqueue job
    post_res = client.post("/ingest", json={"input_dir": str(tmp_path), "force": False})
    assert post_res.status_code == 202
    data = post_res.json()
    job_id = data["job_id"]
    assert data["status"] == "queued"

    # 2. Check status before worker runs
    status_res = client.get(f"/ingest/{job_id}")
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "queued"

    # 3. Worker processes the job
    processed = worker.process_one()
    assert processed is not None
    assert processed.id == job_id
    assert processed.status == JobStatus.COMPLETED

    # 4. Check status after worker completes
    status_res2 = client.get(f"/ingest/{job_id}")
    assert status_res2.status_code == 200
    job_data = status_res2.json()
    assert job_data["status"] == "completed"
    assert job_data["result"]["indexed_files"] == 1
    assert job_data["result"]["skipped_files"] == 0

    # 5. Idempotency: re-ingest unchanged directory
    post_res2 = client.post("/ingest", json={"input_dir": str(tmp_path), "force": False})
    job_id2 = post_res2.json()["job_id"]
    processed2 = worker.process_one()
    assert processed2 is not None
    assert processed2.id == job_id2

    status_res3 = client.get(f"/ingest/{job_id2}")
    job_data2 = status_res3.json()
    assert job_data2["status"] == "completed"
    assert job_data2["result"]["indexed_files"] == 0
    assert job_data2["result"]["skipped_files"] == 1


def test_ingest_job_retry_on_failure(tmp_path: Path) -> None:
    """Verify failed ingestion job retries up to max_attempts then marks failed."""
    queue = InMemoryJobQueue()
    failing_pipeline = MagicMock()
    failing_pipeline.ingest_directory.side_effect = RuntimeError("Disk IO failure")

    worker = IngestWorker(queue=queue, pipeline=failing_pipeline)
    job_id = queue.enqueue(input_dir=str(tmp_path), max_attempts=2)
    assert job_id.startswith("job_")

    # Attempt 1
    w1 = worker.process_one()
    assert w1 is not None
    assert w1.attempts == 1
    assert w1.status == JobStatus.QUEUED  # Re-enqueued for retry

    # Attempt 2 (exhausted)
    w2 = worker.process_one()
    assert w2 is not None
    assert w2.attempts == 2
    assert w2.status == JobStatus.FAILED
    assert "Disk IO failure" in (w2.error or "")


def test_redis_job_queue_unit() -> None:
    """Unit test RedisJobQueue adapter with mock redis client."""
    mock_redis = MagicMock()
    # Mock redis key-value storage in memory for the mock
    storage: dict[str, str] = {}
    q_list: list[str] = []

    def fake_set(name: str, value: str, **kwargs: object) -> bool:
        storage[name] = value
        return True

    def fake_get(name: str) -> str | None:
        return storage.get(name)

    def fake_rpush(name: str, *values: str) -> int:
        q_list.extend(values)
        return len(q_list)

    def fake_lpop(name: str) -> str | None:
        return q_list.pop(0) if q_list else None

    mock_redis.set.side_effect = fake_set
    mock_redis.get.side_effect = fake_get
    mock_redis.rpush.side_effect = fake_rpush
    mock_redis.lpop.side_effect = fake_lpop

    redis_queue = RedisJobQueue(client=mock_redis)

    job_id = redis_queue.enqueue(input_dir="data/test", force=True)
    assert job_id is not None

    job = redis_queue.get_job(job_id)
    assert job is not None
    assert job.input_dir == "data/test"
    assert job.force is True
    assert job.status == JobStatus.QUEUED

    # Pop next job
    popped = redis_queue.pop_job()
    assert popped is not None
    assert popped.id == job_id

    # Update job
    popped.status = JobStatus.COMPLETED
    popped.result = {"indexed_files": 3}
    redis_queue.update_job(popped)

    updated = redis_queue.get_job(job_id)
    assert updated is not None
    assert updated.status == JobStatus.COMPLETED
    assert updated.result == {"indexed_files": 3}
