"""Plot benchmark comparison between conditions A, B, and C with Wilson error bars."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# Non-interactive backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_scorecard(csv_path: str | Path, output_path: str | Path) -> None:
    """Generate grouped bar chart comparing conditions A, B, and C with Wilson 95% CIs."""
    df = pd.read_csv(csv_path)

    # Sort conditions A, B, C
    condition_order = ["A", "B", "C"]
    df["order"] = df["condition"].map(
        lambda c: condition_order.index(c) if c in condition_order else 99
    )
    df = df.sort_values("order").reset_index(drop=True)

    conditions = df["condition"].tolist()
    labels = [
        {
            "A": "Cond A\n(No Faults)",
            "B": "Cond B\n(Faults, No Healing)",
            "C": "Cond C\n(Resilient RAG)",
        }.get(c, c)
        for c in conditions
    ]

    task_success = (df["task_success_rate"] * 100.0).tolist()
    task_err_low = ((df["task_success_rate"] - df["task_success_ci_low"]) * 100.0).tolist()
    task_err_high = ((df["task_success_ci_high"] - df["task_success_rate"]) * 100.0).tolist()

    safe_outcome = (df["safe_outcome_rate"] * 100.0).tolist()
    safe_err_low = ((df["safe_outcome_rate"] - df["safe_outcome_ci_low"]) * 100.0).tolist()
    safe_err_high = ((df["safe_outcome_ci_high"] - df["safe_outcome_rate"]) * 100.0).tolist()

    recovery = (df["recovery_rate"] * 100.0).tolist()

    x = range(len(conditions))
    width = 0.25

    plt.figure(figsize=(10, 6))

    # Bar 1: Task Success Rate
    plt.bar(
        [i - width for i in x],
        task_success,
        width=width,
        yerr=[task_err_low, task_err_high],
        capsize=5,
        color="#2b5c8f",
        label="Task Success Rate (%)",
    )

    # Bar 2: Safe Outcome Rate
    plt.bar(
        [i for i in x],
        safe_outcome,
        width=width,
        yerr=[safe_err_low, safe_err_high],
        capsize=5,
        color="#2ca02c",
        label="Safe-Outcome Rate (%)",
    )

    # Bar 3: Recovery Rate
    plt.bar(
        [i + width for i in x],
        recovery,
        width=width,
        color="#ff7f0e",
        label="Recovery Rate (%)",
    )

    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title(
        "Resilient RAG: Fault Tolerance & Recovery Across Experimental Conditions",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, labels, fontsize=11)
    plt.ylim(0, 110)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Scorecard plot saved to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot Resilient RAG benchmark scorecard comparison"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="results/scorecard.csv",
        help="Path to input scorecard CSV file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/scorecard.png",
        help="Path for generated comparison plot PNG",
    )
    args = parser.parse_args()
    plot_scorecard(args.input, args.output)


if __name__ == "__main__":
    main()
