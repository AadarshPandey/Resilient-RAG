"""Tests for LangSmith integration utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag.evals.langsmith_utils import (
    build_run_metadata,
    build_run_tags,
    eval_key_fact_coverage,
    eval_safe_outcome,
    is_langsmith_enabled,
    upload_golden_dataset,
)


def test_is_langsmith_enabled() -> None:
    with patch.dict("os.environ", {}, clear=True):
        assert not is_langsmith_enabled()

    with patch.dict(
        "os.environ",
        {"LANGSMITH_TRACING": "true"},
        clear=True,
    ):
        assert not is_langsmith_enabled()

    with patch.dict(
        "os.environ",
        {"LANGSMITH_API_KEY": "lsv2_pt_xxx"},
        clear=True,
    ):
        assert not is_langsmith_enabled()

    with patch.dict(
        "os.environ",
        {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "lsv2_pt_xxx"},
        clear=True,
    ):
        assert is_langsmith_enabled()


def test_build_run_metadata() -> None:
    assert build_run_metadata() == {}

    meta = build_run_metadata(
        condition="C",
        scenario="llm_tier0_outage",
        seed=42,
        heal_actions=["MODEL_FALLBACK"],
        outcome="ANSWERED",
        extra={"notes": "test"},
    )
    assert meta["condition"] == "C"
    assert meta["scenario"] == "llm_tier0_outage"
    assert meta["seed"] == 42
    assert meta["heal_actions"] == ["MODEL_FALLBACK"]
    assert meta["heal_count"] == 1
    assert meta["outcome"] == "ANSWERED"
    assert meta["notes"] == "test"


def test_build_run_tags() -> None:
    tags = build_run_tags(
        condition="C",
        scenario="vector_backend_down",
        outcome="ANSWERED",
        extra_tags=["offline", "v1.0"],
    )
    assert tags == [
        "condition:C",
        "scenario:vector_backend_down",
        "outcome:ANSWERED",
        "offline",
        "v1.0",
    ]


def test_upload_golden_dataset_disabled() -> None:
    with patch.dict("os.environ", {}, clear=True):
        res = upload_golden_dataset(file_path="data/golden/golden_set.jsonl")
        assert res is False


def test_upload_golden_dataset_mock_client(tmp_path: any) -> None:
    mock_client = MagicMock()
    mock_client.has_dataset.return_value = False

    fake_file = tmp_path / "test_set.jsonl"
    fake_file.write_text(
        '{"id":"q1","question":"What is K8s?","reference_answer":"Kubernetes","key_facts":["k8s"],"source_chunk_ids":["c1"]}\n'
    )

    success = upload_golden_dataset(
        dataset_name="test-ds",
        file_path=fake_file,
        client=mock_client,
    )
    assert success is True
    mock_client.has_dataset.assert_called_once_with(dataset_name="test-ds")
    mock_client.create_dataset.assert_called_once()
    assert mock_client.create_example.call_count == 1
    call_kwargs = mock_client.create_example.call_args[1]
    assert call_kwargs["inputs"] == {"question": "What is K8s?"}
    assert call_kwargs["outputs"]["reference_answer"] == "Kubernetes"
    assert call_kwargs["outputs"]["key_facts"] == ["k8s"]


def test_eval_key_fact_coverage() -> None:
    run_obj = MagicMock()
    run_obj.outputs = {"answer": "A Deployment manages stateless pods and replicasets."}

    example_obj = MagicMock()
    example_obj.outputs = {"key_facts": ["stateless pods", "replicasets"]}

    res = eval_key_fact_coverage(run_obj, example_obj)
    assert res["key"] == "key_fact_coverage"
    assert res["score"] == 1.0


def test_eval_safe_outcome() -> None:
    run_answered = MagicMock()
    run_answered.outputs = {"outcome": "ANSWERED"}
    assert eval_safe_outcome(run_answered, MagicMock())["score"] == 1.0

    run_abstain = MagicMock()
    run_abstain.outputs = {"outcome": "ABSTAIN"}
    assert eval_safe_outcome(run_abstain, MagicMock())["score"] == 1.0

    run_fail = MagicMock()
    run_fail.outputs = {"outcome": "FAILED"}
    assert eval_safe_outcome(run_fail, MagicMock())["score"] == 0.0
