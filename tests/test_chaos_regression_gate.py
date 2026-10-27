"""Tests for Milestone P7: CI/CD Workflow, Chaos Regression Gate, and Release Checklist."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.chaos_regression_gate import run_chaos_regression


def test_ci_workflow_yaml_structure() -> None:
    """Verify .github/workflows/ci.yml is valid YAML and defines required jobs."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.is_file(), "ci.yml does not exist"

    data = yaml.safe_load(ci_path.read_text(encoding="utf-8"))
    assert "jobs" in data
    jobs = data["jobs"]
    assert "quality" in jobs
    assert "chaos-regression" in jobs
    assert "image" in jobs
    assert "integration" in jobs


def test_baseline_json_exists() -> None:
    """Verify results/baseline.json exists and specifies safe outcome thresholds."""
    baseline_path = Path("results/baseline.json")
    assert baseline_path.is_file(), "results/baseline.json does not exist"

    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert "min_acceptable_safe_outcome_rate" in data
    assert data["min_acceptable_safe_outcome_rate"] >= 0.85


def test_chaos_regression_gate_success() -> None:
    """Verify chaos regression gate completes successfully against baseline."""
    # Run with small sample to keep test fast (<5s)
    success = run_chaos_regression(
        baseline_path="results/baseline.json",
        n_questions=2,
        seeds=[42],
    )
    assert success is True


def test_chaos_regression_gate_detects_drop(tmp_path: Path) -> None:
    """Verify chaos regression gate returns False when threshold exceeds observed rate."""
    impossible_baseline = tmp_path / "baseline.json"
    impossible_baseline.write_text(json.dumps({"min_acceptable_safe_outcome_rate": 1.50}))

    success = run_chaos_regression(
        baseline_path=impossible_baseline,
        n_questions=2,
        seeds=[42],
    )
    assert success is False


def test_release_checklist_exists() -> None:
    """Verify docs/release.md exists and contains operational deployment steps."""
    release_path = Path("docs/release.md")
    assert release_path.is_file()
    content = release_path.read_text(encoding="utf-8")
    assert "Quality Gate" in content
    assert "Chaos Regression Gate" in content
    assert "Rollback Plan" in content
