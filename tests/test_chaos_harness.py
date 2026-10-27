"""Tests for M8: Chaos harness (fault models, controller determinism, virtual latency, wrappers, and 14 scenarios)."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from rag.chaos.controller import FaultController
from rag.chaos.faults import FaultAction, FaultConfig
from rag.chaos.loader import load_scenarios
from rag.chaos.wrappers import ChaosChatModel, ChaosVectorStore
from rag.clock import VirtualClock
from rag.errors import BackendError
from rag.fakes import FakeVectorStore, ScriptedChatModel
from rag.llm.base import LLMResult


def test_fault_controller_determinism_and_schedule() -> None:
    """Verify FaultController executes schedule deterministically based on call counts and seed."""
    fault = FaultConfig(
        component="llm",
        action=FaultAction.ERROR,
        after_n_calls=1,  # Skip call 1 (index 0), fire on call 2 (index 1)
        times=2,  # Fire for 2 calls (calls 2 and 3)
        error_type="LLMRateLimitError",
        error_message="Rate limit 429",
    )
    controller = FaultController(faults=[fault], seed=42)

    # Call 1 (index 0): should NOT fire
    assert controller.check_fault("llm") is None
    # Call 2 (index 1): should fire
    f2 = controller.check_fault("llm")
    assert f2 is not None
    assert f2.action == FaultAction.ERROR
    # Call 3 (index 2): should fire
    f3 = controller.check_fault("llm")
    assert f3 is not None
    # Call 4 (index 3): times exhausted, should NOT fire
    assert controller.check_fault("llm") is None


def test_virtual_latency_advances_clock_without_sleep() -> None:
    """Delays in chaos wrappers must advance VirtualClock without calling time.sleep."""
    clock = VirtualClock(start_time=10.0)
    fault = FaultConfig(
        component="llm",
        action=FaultAction.DELAY,
        delay_seconds=5.5,
        after_n_calls=0,
        times=1,
    )
    controller = FaultController(faults=[fault])
    base_model = ScriptedChatModel(responses=[LLMResult(content='{"answer": "ok"}', tier=0)])

    chaos_model = ChaosChatModel(
        target=base_model,
        controller=controller,
        clock=clock,
    )

    with patch.object(time, "sleep", side_effect=AssertionError("time.sleep was called!")):
        res = chaos_model.invoke([{"role": "user", "content": "hi"}], tier=0)
        assert res.content == '{"answer": "ok"}'
        # Virtual clock advanced exactly 5.5s
        assert clock.now() == 15.5


def test_scenarios_yaml_loads_all_14(tmp_path: Path) -> None:
    """Verify all 14 scenarios load from configs/scenarios.yaml and pass validation."""
    scenarios_path = Path("configs/scenarios.yaml")
    assert scenarios_path.is_file(), "configs/scenarios.yaml must exist"

    scenarios = load_scenarios(scenarios_path)
    assert len(scenarios) == 14

    expected_names = {
        "llm_rate_limit_transient",
        "llm_timeout_transient",
        "llm_tier0_outage",
        "llm_all_tiers_down",
        "llm_malformed_json_generate",
        "llm_malformed_json_verify",
        "llm_ungrounded_answer",
        "llm_context_overflow",
        "embedder_failure",
        "vector_backend_down",
        "bm25_backend_down",
        "both_retrievers_down",
        "retriever_empty_results",
        "retriever_prompt_injection",
    }
    loaded_names = {s.name for s in scenarios}
    assert loaded_names == expected_names


def test_chaos_wrappers_fire_as_specified() -> None:
    """Verify ChaosVectorStore raises BackendError when configured."""
    vstore = FakeVectorStore()
    fault = FaultConfig(
        component="vector",
        action=FaultAction.ERROR,
        error_type="BackendError",
        error_message="Qdrant connection dropped",
    )
    controller = FaultController(faults=[fault])
    chaos_vstore = ChaosVectorStore(target=vstore, controller=controller)

    with pytest.raises(BackendError, match="Qdrant connection dropped"):
        chaos_vstore.search([0.1, 0.2], top_k=5)
