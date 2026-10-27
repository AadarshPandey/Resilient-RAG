"""Statistical scorecard computation with Wilson score intervals and paired metrics."""

import math
from collections import defaultdict

from pydantic import BaseModel, Field

from rag.schemas import Outcome, RunRecord


def wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Compute Wilson score interval for binomial proportions.

    Args:
        successes: Number of successful trials.
        total: Total number of trials.
        confidence: Confidence level (default: 0.95, z=1.96).

    Returns:
        (center_estimate, (ci_lower, ci_upper))
    """
    if total <= 0:
        return 0.0, (0.0, 0.0)

    z = 1.96  # 95% confidence interval
    p = float(successes) / float(total)
    n = float(total)

    denominator = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denominator
    half = (z * math.sqrt((p * (1.0 - p)) / n + (z * z) / (4.0 * n * n))) / denominator

    ci_lower = max(0.0, center - half)
    ci_upper = min(1.0, center + half)
    return p, (ci_lower, ci_upper)


class ConditionMetrics(BaseModel):
    """Aggregate reliability and performance metrics for an experimental condition."""

    total_runs: int = 0
    task_success_rate: float = 0.0
    task_success_ci: tuple[float, float] = (0.0, 0.0)
    safe_outcome_rate: float = 0.0
    safe_outcome_ci: tuple[float, float] = (0.0, 0.0)
    recovery_rate: float = 0.0
    mttr: float = 0.0
    false_abstention_rate: float = 0.0


class ScorecardResult(BaseModel):
    """Full statistical scorecard across experimental conditions A, B, and C."""

    condition_metrics: dict[str, ConditionMetrics] = Field(default_factory=dict)
    paired_overhead: dict[str, float] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render scorecard as Markdown table."""
        a = self.condition_metrics.get("A", ConditionMetrics())
        b = self.condition_metrics.get("B", ConditionMetrics())
        c = self.condition_metrics.get("C", ConditionMetrics())

        added_calls = self.paired_overhead.get("mean_added_calls", 0.0)

        lines = [
            "| Metric | A | B | C |",
            "|---|---|---|---|",
            f"| Task success (95% CI) | {a.task_success_rate:.1%} [{a.task_success_ci[0]:.2f}, {a.task_success_ci[1]:.2f}] | {b.task_success_rate:.1%} [{b.task_success_ci[0]:.2f}, {b.task_success_ci[1]:.2f}] | {c.task_success_rate:.1%} [{c.task_success_ci[0]:.2f}, {c.task_success_ci[1]:.2f}] |",
            f"| Safe-outcome rate | {a.safe_outcome_rate:.1%} | {b.safe_outcome_rate:.1%} | {c.safe_outcome_rate:.1%} |",
            f"| Recovery rate | n/a | n/a | {c.recovery_rate:.1%} |",
            f"| MTTR (seconds) | n/a | n/a | {c.mttr:.2f}s |",
            f"| Added LLM calls vs B | n/a | n/a | +{added_calls:.1f} |",
        ]
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Render scorecard as CSV format."""
        a = self.condition_metrics.get("A", ConditionMetrics())
        b = self.condition_metrics.get("B", ConditionMetrics())
        c = self.condition_metrics.get("C", ConditionMetrics())

        rows = [
            "metric,A,B,C",
            f"task_success_rate,{a.task_success_rate:.4f},{b.task_success_rate:.4f},{c.task_success_rate:.4f}",
            f"safe_outcome_rate,{a.safe_outcome_rate:.4f},{b.safe_outcome_rate:.4f},{c.safe_outcome_rate:.4f}",
            f"recovery_rate,n/a,n/a,{c.recovery_rate:.4f}",
            f"mttr_seconds,n/a,n/a,{c.mttr:.4f}",
            f"paired_added_calls,n/a,n/a,{self.paired_overhead.get('mean_added_calls', 0.0):.4f}",
        ]
        return "\n".join(rows)


def _compute_mttr(run: RunRecord) -> float:
    """Compute virtual time from first fault detection to finalization."""
    fault_time: float | None = None
    final_time: float | None = None

    for event in run.trace:
        if ("error" in event.details or "failure" in event.details) and fault_time is None:
            fault_time = event.timestamp
        if event.node == "finalize":
            final_time = event.timestamp

    if fault_time is not None and final_time is not None and final_time >= fault_time:
        return final_time - fault_time
    return run.duration_seconds


def compute_scorecard(runs: list[RunRecord]) -> ScorecardResult:
    """Compute comparative statistical scorecard across all runs."""
    by_condition: dict[str, list[RunRecord]] = defaultdict(list)
    for r in runs:
        by_condition[r.condition.upper()].append(r)

    metrics_map: dict[str, ConditionMetrics] = {}

    for cond, cond_runs in by_condition.items():
        total = len(cond_runs)
        if total == 0:
            metrics_map[cond] = ConditionMetrics()
            continue

        # Task success: answered correctly
        successes = sum(1 for r in cond_runs if r.outcome == Outcome.ANSWERED)
        p_succ, ci_succ = wilson_score_interval(successes, total)

        # Safe outcomes: answered OR justified abstain/escalate
        safe = sum(
            1
            for r in cond_runs
            if r.outcome in (Outcome.ANSWERED, Outcome.ABSTAINED, Outcome.ESCALATED)
        )
        p_safe, ci_safe = wilson_score_interval(safe, total)

        # Recovery rate (for runs that experienced faults)
        faulty_runs = [r for r in cond_runs if r.detected_faults or r.heal_actions]
        if faulty_runs:
            recovered = sum(
                1
                for r in faulty_runs
                if r.outcome in (Outcome.ANSWERED, Outcome.ABSTAINED, Outcome.ESCALATED)
            )
            recovery_rate = float(recovered) / float(len(faulty_runs))
            mttr_values = [_compute_mttr(r) for r in faulty_runs if r.outcome == Outcome.ANSWERED]
            mttr = sum(mttr_values) / len(mttr_values) if mttr_values else 0.0
        else:
            recovery_rate = 1.0 if cond == "A" else 0.0
            mttr = 0.0

        metrics_map[cond] = ConditionMetrics(
            total_runs=total,
            task_success_rate=p_succ,
            task_success_ci=ci_succ,
            safe_outcome_rate=p_safe,
            safe_outcome_ci=ci_safe,
            recovery_rate=recovery_rate,
            mttr=mttr,
        )

    # Compute paired overhead (Condition C vs B on matched (question, scenario, seed))
    paired_runs: dict[tuple[str, str, int | None], dict[str, RunRecord]] = defaultdict(dict)
    for r in runs:
        if r.condition in ("B", "C"):
            key = (r.question, r.scenario or "none", r.seed)
            paired_runs[key][r.condition] = r

    added_calls: list[float] = []
    added_tokens: list[float] = []
    added_latency: list[float] = []

    for _, pair in paired_runs.items():
        if "B" in pair and "C" in pair:
            b_run = pair["B"]
            c_run = pair["C"]
            added_calls.append(float(c_run.llm_calls - b_run.llm_calls))
            added_tokens.append(float(c_run.tokens_used - b_run.tokens_used))
            added_latency.append(float(c_run.duration_seconds - b_run.duration_seconds))

    paired_overhead: dict[str, float] = {
        "mean_added_calls": sum(added_calls) / len(added_calls) if added_calls else 0.0,
        "mean_added_tokens": sum(added_tokens) / len(added_tokens) if added_tokens else 0.0,
        "mean_added_latency": sum(added_latency) / len(added_latency) if added_latency else 0.0,
    }

    return ScorecardResult(
        condition_metrics=metrics_map,
        paired_overhead=paired_overhead,
    )


def scorecard_to_markdown(scorecard: ScorecardResult) -> str:
    """Format scorecard result as GitHub-flavored Markdown table."""
    lines = [
        "# Resilient RAG Chaos Scorecard",
        "",
        "| Condition | Runs | Task Success (95% CI) | Safe Outcome (95% CI) | Recovery Rate | MTTR (s) |",
        "|---|---|---|---|---|---|",
    ]
    for cond in sorted(scorecard.condition_metrics.keys()):
        m = scorecard.condition_metrics[cond]
        succ_ci = (
            f"{m.task_success_rate:.1%} [{m.task_success_ci[0]:.1%}, {m.task_success_ci[1]:.1%}]"
        )
        safe_ci = (
            f"{m.safe_outcome_rate:.1%} [{m.safe_outcome_ci[0]:.1%}, {m.safe_outcome_ci[1]:.1%}]"
        )
        lines.append(
            f"| {cond} | {m.total_runs} | {succ_ci} | {safe_ci} | {m.recovery_rate:.1%} | {m.mttr:.2f} |"
        )
    lines.append("")
    lines.append("## Paired Overhead (Condition C vs B)")
    lines.append("")
    lines.append(
        f"- **Mean Added Calls:** {scorecard.paired_overhead.get('mean_added_calls', 0.0):.2f}"
    )
    lines.append(
        f"- **Mean Added Tokens:** {scorecard.paired_overhead.get('mean_added_tokens', 0.0):.1f}"
    )
    lines.append(
        f"- **Mean Added Latency (s):** {scorecard.paired_overhead.get('mean_added_latency', 0.0):.2f}"
    )
    return "\n".join(lines)


def scorecard_to_csv(scorecard: ScorecardResult) -> str:
    """Format scorecard result as CSV string."""
    lines = [
        "condition,total_runs,task_success_rate,task_success_ci_low,task_success_ci_high,safe_outcome_rate,safe_outcome_ci_low,safe_outcome_ci_high,recovery_rate,mttr"
    ]
    for cond in sorted(scorecard.condition_metrics.keys()):
        m = scorecard.condition_metrics[cond]
        lines.append(
            f"{cond},{m.total_runs},{m.task_success_rate:.4f},{m.task_success_ci[0]:.4f},{m.task_success_ci[1]:.4f},{m.safe_outcome_rate:.4f},{m.safe_outcome_ci[0]:.4f},{m.safe_outcome_ci[1]:.4f},{m.recovery_rate:.4f},{m.mttr:.4f}"
        )
    return "\n".join(lines)
