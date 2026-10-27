"""Tests for M9: Experiment runner and statistical scorecard (Wilson CIs, MTTR, paired overhead, and resumability)."""

from pathlib import Path

import pytest

from rag.chaos.faults import Scenario
from rag.chaos.runner import ExperimentRunner
from rag.chaos.scorecard import compute_scorecard, wilson_score_interval
from rag.schemas import Outcome, RunRecord, TraceEvent


def test_wilson_confidence_interval() -> None:
    """Verify Wilson score confidence interval calculation."""
    # 0 out of 10: lower bound should be 0.0, upper bound > 0
    center_0, (low_0, high_0) = wilson_score_interval(successes=0, total=10)
    assert low_0 == 0.0
    assert 0.2 < high_0 < 0.35

    # 10 out of 10: upper bound should be 1.0, lower bound < 1.0
    center_10, (low_10, high_10) = wilson_score_interval(successes=10, total=10)
    assert high_10 == 1.0
    assert 0.65 < low_10 < 0.8

    # 50 out of 100: centered around 0.5
    center_50, (low_50, high_50) = wilson_score_interval(successes=50, total=100)
    assert pytest.approx(center_50, abs=0.01) == 0.5
    assert 0.39 < low_50 < 0.42
    assert 0.58 < high_50 < 0.61

    # Empty total
    center_empty, (low_e, high_e) = wilson_score_interval(successes=0, total=0)
    assert center_empty == 0.0
    assert low_e == 0.0 and high_e == 0.0


def test_scorecard_metrics_calculation() -> None:
    """Verify scorecard computes task success, safe-outcome rate, MTTR, and paired overhead."""
    # Condition A (ceiling)
    runs_a = [
        RunRecord(
            question="q1",
            outcome=Outcome.ANSWERED,
            condition="A",
            scenario="none",
            seed=1,
            llm_calls=1,
            tokens_used=100,
            duration_seconds=1.0,
        ),
        RunRecord(
            question="q2",
            outcome=Outcome.ANSWERED,
            condition="A",
            scenario="none",
            seed=1,
            llm_calls=1,
            tokens_used=100,
            duration_seconds=1.0,
        ),
    ]

    # Condition B (faults, no healing)
    runs_b = [
        RunRecord(
            question="q1",
            outcome=Outcome.FAILED,
            condition="B",
            scenario="llm_tier0_outage",
            seed=1,
            llm_calls=1,
            tokens_used=50,
            duration_seconds=0.5,
            detected_faults=["llm_error"],
        ),
        RunRecord(
            question="q2",
            outcome=Outcome.FAILED,
            condition="B",
            scenario="llm_tier0_outage",
            seed=1,
            llm_calls=1,
            tokens_used=50,
            duration_seconds=0.5,
            detected_faults=["llm_error"],
        ),
    ]

    # Condition C (faults + healing)
    runs_c = [
        RunRecord(
            question="q1",
            outcome=Outcome.ANSWERED,
            condition="C",
            scenario="llm_tier0_outage",
            seed=1,
            llm_calls=3,
            tokens_used=250,
            duration_seconds=2.5,
            detected_faults=["llm_error"],
            heal_actions=["retry_same", "model_fallback"],
            trace=[
                TraceEvent(node="generate", timestamp=0.5, details={"error": "fault"}),
                TraceEvent(node="finalize", timestamp=2.5, details={}),
            ],
        ),
        RunRecord(
            question="q2",
            outcome=Outcome.ANSWERED,
            condition="C",
            scenario="llm_tier0_outage",
            seed=1,
            llm_calls=3,
            tokens_used=250,
            duration_seconds=2.5,
            detected_faults=["llm_error"],
            heal_actions=["retry_same", "model_fallback"],
            trace=[
                TraceEvent(node="generate", timestamp=0.5, details={"error": "fault"}),
                TraceEvent(node="finalize", timestamp=2.5, details={}),
            ],
        ),
    ]

    all_runs = runs_a + runs_b + runs_c
    scorecard = compute_scorecard(all_runs)

    assert scorecard.condition_metrics["A"].task_success_rate > 0.9
    assert scorecard.condition_metrics["B"].task_success_rate == 0.0
    assert scorecard.condition_metrics["C"].task_success_rate > 0.9

    # Recovery rate for C
    assert scorecard.condition_metrics["C"].recovery_rate == 1.0

    # MTTR for C: (2.5 - 0.5) = 2.0s
    assert pytest.approx(scorecard.condition_metrics["C"].mttr, abs=0.1) == 2.0

    # Paired overhead: C calls (3) - B calls (1) = 2.0 calls added
    assert pytest.approx(scorecard.paired_overhead["mean_added_calls"]) == 2.0

    # Markdown and CSV exports
    md = scorecard.to_markdown()
    assert "| Metric | A | B | C |" in md
    assert "Task success" in md

    csv_text = scorecard.to_csv()
    assert "metric,A,B,C" in csv_text


def test_experiment_runner_dry_run_and_resumability(tmp_path: Path) -> None:
    """Verify dry-run calculates run budget without calls and runner resumes from JSONL."""
    runner = ExperimentRunner()
    scenarios = [
        Scenario(name="s1", description="d1", target_component="llm"),
        Scenario(name="s2", description="d2", target_component="vector"),
    ]
    questions = ["What is kubelet?", "What is etcd?"]
    conditions = ["A", "B", "C"]
    seeds = [1, 2]

    # Dry-run: 2 questions * 2 scenarios * 2 seeds * 3 conditions = 24 runs
    estimate = runner.dry_run(
        questions=questions,
        scenarios=scenarios,
        seeds=seeds,
        conditions=conditions,
    )
    assert estimate["total_runs"] == 24
    assert estimate["estimated_llm_calls"] > 0

    # Test resumability
    output_jsonl = tmp_path / "runs.jsonl"
    # Pre-populate 1 run
    existing_record = RunRecord(
        question=questions[0],
        outcome=Outcome.ANSWERED,
        condition="A",
        scenario="s1",
        seed=1,
    )
    output_jsonl.write_text(existing_record.model_dump_json() + "\n", encoding="utf-8")

    completed_keys = runner.load_completed_keys(output_jsonl)
    assert len(completed_keys) == 1
    assert (questions[0], "s1", 1, "A") in completed_keys
