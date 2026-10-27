"""Tests for M1: Foundations (config, errors, logging, schemas, audit, clock, device)."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.audit import AuditLogger
from rag.clock import Clock, SystemClock, VirtualClock
from rag.config import Settings
from rag.device import resolve_device
from rag.errors import (
    BackendError,
    ContextOverflowError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    MalformedOutputError,
    RagError,
)
from rag.schemas import (
    Chunk,
    Citation,
    Document,
    FailureType,
    HealAction,
    Outcome,
    RetrievedChunk,
    RunRecord,
)


def test_settings_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify settings defaults and environment variable overrides."""
    # Test defaults
    settings = Settings(_env_file=None)  # do not read from disk .env
    assert settings.healing_enabled is True
    assert settings.max_heal_attempts == 4
    assert settings.device == "auto"
    assert "google:" in settings.llm_chain

    # Test overrides with RAG_ prefix
    monkeypatch.setenv("RAG_MAX_HEAL_ATTEMPTS", "2")
    monkeypatch.setenv("RAG_HEALING_ENABLED", "false")
    monkeypatch.setenv("RAG_LLM_CHAIN", "google:gemini-1.5-flash,google:gemini-1.5-pro")
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    overridden = Settings(_env_file=None)
    assert overridden.max_heal_attempts == 2
    assert overridden.healing_enabled is False
    assert overridden.llm_chain == "google:gemini-1.5-flash,google:gemini-1.5-pro"
    assert overridden.device == "cpu"
    assert overridden.gemini_api_key == "test-key-123"

    # Helper method to get tiers
    tiers = overridden.get_llm_tiers()
    assert len(tiers) == 2
    assert tiers[0] == ("google", "gemini-1.5-flash")
    assert tiers[1] == ("google", "gemini-1.5-pro")


def test_typed_errors_hierarchy() -> None:
    """Verify typed exception hierarchy and retryable flags."""
    assert issubclass(BackendError, RagError)
    assert issubclass(LLMError, RagError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(MalformedOutputError, RagError)
    assert issubclass(ContextOverflowError, RagError)

    rate_limit = LLMRateLimitError("Too many requests")
    assert rate_limit.retryable is True

    timeout = LLMTimeoutError("Request timed out")
    assert timeout.retryable is True

    generic_llm = LLMError("API failure", retryable=False)
    assert generic_llm.retryable is False

    malformed = MalformedOutputError("Failed to parse JSON")
    assert isinstance(malformed, RagError)


def test_virtual_clock() -> None:
    """Verify VirtualClock is deterministic and advances only on advance()."""
    clock: Clock = VirtualClock(start_time=100.0)
    assert clock.now() == 100.0
    clock.advance(15.5)
    assert clock.now() == 115.5
    clock.advance(0.0)
    assert clock.now() == 115.5

    with pytest.raises(ValueError, match="Cannot move clock backwards"):
        clock.advance(-1.0)


def test_system_clock() -> None:
    """Verify SystemClock advances and meets Clock protocol."""
    clock: Clock = SystemClock()
    t1 = clock.now()
    assert t1 > 0
    # advance does not move system wall clock, or is a no-op
    clock.advance(5.0)
    t2 = clock.now()
    assert t2 >= t1


def test_device_selection_resolution() -> None:
    """Verify resolve_device handles auto, cpu, cuda, and fallbacks."""
    assert resolve_device("cpu") == "cpu"

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=False),
    ):
        assert resolve_device("auto") == "cpu"

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert resolve_device("auto") == "mps"

    with patch("torch.cuda.is_available", return_value=True):
        assert resolve_device("auto") == "cuda"


def test_logging_setup(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify JSON logging configuration formats logs correctly."""
    import logging

    from rag.logging_setup import configure_logging

    configure_logging(level="DEBUG")
    logger = logging.getLogger("test_logger")
    logger.info("Hello structured log")

    captured = capsys.readouterr().out.strip()
    data = json.loads(captured)
    assert data["message"] == "Hello structured log"
    assert data["level"] == "INFO"
    assert data["logger"] == "test_logger"
    assert "timestamp" in data

    # Test exception formatting
    try:
        raise ValueError("test exception")
    except ValueError:
        logger.error("Error occurred", exc_info=True)

    captured_err = capsys.readouterr().out.strip().split("\n")[-1]
    err_data = json.loads(captured_err)
    assert err_data["level"] == "ERROR"
    assert "exception" in err_data
    assert "ValueError: test exception" in err_data["exception"]


def test_settings_methods(tmp_path: Path) -> None:
    """Test helper methods on Settings."""
    settings = Settings(
        _env_file=None,
        gemini_api_key=None,
        corpus_dir=str(tmp_path / "raw"),
        qdrant_path=str(tmp_path / "qdrant"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
        llm_chain="gemini-1.5-flash",
    )
    assert settings.get_effective_gemini_api_key() is None
    tiers = settings.get_llm_tiers()
    assert len(tiers) == 1
    assert tiers[0] == ("google", "gemini-1.5-flash")

    settings.ensure_directories()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "qdrant").is_dir()


def test_audit_logger_thread_safety(tmp_path: Path) -> None:
    """Verify thread-safe append-only logging to JSONL."""
    log_file = tmp_path / "audit.jsonl"
    logger = AuditLogger(path=log_file)

    num_threads = 10
    records_per_thread = 20

    def log_task(thread_id: int) -> None:
        for i in range(records_per_thread):
            logger.log(
                {
                    "thread_id": thread_id,
                    "iteration": i,
                    "event": "test_event",
                }
            )

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(log_task, t) for t in range(num_threads)]
        for f in futures:
            f.result()

    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == num_threads * records_per_thread

    for line in lines:
        data = json.loads(line)
        assert "thread_id" in data
        assert "iteration" in data
        assert data["event"] == "test_event"


def test_schemas_models() -> None:
    """Verify core schemas and serialization."""
    doc = Document(
        id="doc_1", content="Kubernetes architecture overview", metadata={"source": "k8s"}
    )
    assert doc.id == "doc_1"
    chunk = Chunk(
        id="chunk_1",
        doc_id="doc_1",
        text="The kubelet is the primary node agent.",
        index=0,
        metadata={"section": "node"},
    )
    retrieved = RetrievedChunk(chunk=chunk, score=0.88, source="hybrid")
    citation = Citation(chunk_id=chunk.id, text=chunk.text, score=0.88)

    assert retrieved.chunk.id == "chunk_1"
    assert citation.chunk_id == "chunk_1"

    record = RunRecord(
        question="What is kubelet?",
        outcome=Outcome.ANSWERED,
        answer="The kubelet is the primary node agent.",
        citations=[citation],
        attempts=1,
        detected_faults=[],
        heal_actions=[],
        duration_seconds=1.2,
    )
    dumped = record.model_dump_json()
    assert "kubelet" in dumped
    assert Outcome.ANSWERED == "answered"
    assert FailureType.LLM_ERROR == "llm_error"
    assert HealAction.RETRY_SAME == "retry_same"
