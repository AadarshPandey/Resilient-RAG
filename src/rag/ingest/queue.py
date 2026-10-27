"""Lightweight background ingestion job queue and worker (Redis-backed with in-memory fallback)."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from rag.ingest.pipeline import IngestionPipeline, IngestionResult

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    """Lifecycle status of an ingestion job."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJob(BaseModel):
    """Data representation of an ingestion job."""

    id: str
    input_dir: str
    force: bool = False
    status: JobStatus = JobStatus.QUEUED
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    attempts: int = 0
    max_attempts: int = 3
    result: dict[str, Any] | None = None
    error: str | None = None


@runtime_checkable
class JobQueue(Protocol):
    """Interface for background ingestion job queues."""

    def enqueue(self, input_dir: str, force: bool = False, max_attempts: int = 3) -> str:
        """Enqueue an ingestion job and return the assigned job_id."""
        ...

    def get_job(self, job_id: str) -> IngestionJob | None:
        """Retrieve job status and details by id."""
        ...

    def pop_job(self) -> IngestionJob | None:
        """Pop the next available job for processing."""
        ...

    def update_job(self, job: IngestionJob) -> None:
        """Update job state in the queue storage."""
        ...


class InMemoryJobQueue:
    """Thread-safe in-memory job queue for hermetic testing and single-node setups."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestionJob] = {}
        self._queue: queue.Queue[str] = queue.Queue()

    def enqueue(self, input_dir: str, force: bool = False, max_attempts: int = 3) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = IngestionJob(
            id=job_id,
            input_dir=input_dir,
            force=force,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            attempts=0,
            max_attempts=max_attempts,
        )
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return job_id

    def get_job(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.model_copy()

    def pop_job(self) -> IngestionJob | None:
        try:
            job_id = self._queue.get_nowait()
        except queue.Empty:
            return None

        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return job.model_copy()

    def update_job(self, job: IngestionJob) -> None:
        with self._lock:
            self._jobs[job.id] = job.model_copy()
            if job.status == JobStatus.QUEUED:
                self._queue.put(job.id)


class RedisJobQueue:
    """Redis-backed job queue storing jobs as serialized JSON and IDs in a FIFO list."""

    def __init__(
        self, client: Any, prefix: str = "rag:job:", queue_key: str = "rag:queue:ingest"
    ) -> None:
        self.client = client
        self.prefix = prefix
        self.queue_key = queue_key

    def _key(self, job_id: str) -> str:
        return f"{self.prefix}{job_id}"

    def enqueue(self, input_dir: str, force: bool = False, max_attempts: int = 3) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = time.time()
        job = IngestionJob(
            id=job_id,
            input_dir=input_dir,
            force=force,
            status=JobStatus.QUEUED,
            created_at=now,
            updated_at=now,
            attempts=0,
            max_attempts=max_attempts,
        )
        self.client.set(self._key(job_id), json.dumps(job.model_dump()))
        self.client.rpush(self.queue_key, job_id)
        return job_id

    def get_job(self, job_id: str) -> IngestionJob | None:
        raw = self.client.get(self._key(job_id))
        if not raw:
            return None
        data = json.loads(raw)
        return IngestionJob(**data)

    def pop_job(self) -> IngestionJob | None:
        job_id = self.client.lpop(self.queue_key)
        if not job_id:
            return None
        if isinstance(job_id, bytes):
            job_id = job_id.decode("utf-8")
        return self.get_job(str(job_id))

    def update_job(self, job: IngestionJob) -> None:
        self.client.set(self._key(job.id), json.dumps(job.model_dump()))
        if job.status == JobStatus.QUEUED:
            self.client.rpush(self.queue_key, job.id)


class IngestWorker:
    """Worker that consumes ingestion jobs from a JobQueue and processes them via IngestionPipeline."""

    def __init__(self, queue: JobQueue, pipeline: IngestionPipeline) -> None:
        self.queue = queue
        self.pipeline = pipeline

    def process_one(self) -> IngestionJob | None:
        """Process one pending job from the queue. Handles retries up to max_attempts."""
        job = self.queue.pop_job()
        if job is None:
            return None

        job.status = JobStatus.RUNNING
        job.attempts += 1
        job.updated_at = time.time()
        self.queue.update_job(job)

        try:
            res: IngestionResult = self.pipeline.ingest_directory(
                job.input_dir,
                force=job.force,
            )
            job.status = JobStatus.COMPLETED
            job.result = res.model_dump()
            job.updated_at = time.time()
            self.queue.update_job(job)
            return job
        except Exception as exc:
            logger.warning(
                "Ingestion job %s attempt %d failed: %s",
                job.id,
                job.attempts,
                exc,
                exc_info=True,
            )
            job.error = str(exc)
            job.updated_at = time.time()

            if job.attempts < job.max_attempts:
                job.status = JobStatus.QUEUED
                self.queue.update_job(job)
            else:
                job.status = JobStatus.FAILED
                self.queue.update_job(job)
            return job


def create_job_queue(redis_url: str | None = None) -> JobQueue:
    """Factory creating RedisJobQueue if redis is reachable, otherwise InMemoryJobQueue."""
    if redis_url and not redis_url.startswith(":memory:"):
        try:
            import redis

            client = redis.from_url(redis_url, decode_responses=True)
            client.ping()
            return RedisJobQueue(client=client)
        except Exception as err:
            logger.info("Redis unreachable (%s); using in-memory job queue.", err)
            return InMemoryJobQueue()
    return InMemoryJobQueue()
