"""Command line interface for Resilient RAG using Typer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from rag.bootstrap import build_components
from rag.chaos.loader import load_scenarios
from rag.chaos.runner import ExperimentRunner
from rag.chaos.scorecard import compute_scorecard, scorecard_to_csv, scorecard_to_markdown
from rag.config import get_settings
from rag.evals.golden import load_golden_set
from rag.evals.langsmith_utils import upload_golden_dataset
from rag.evals.run_eval import run_benchmark_eval
from rag.ingest.pipeline import IngestionPipeline
from rag.schemas import Outcome, RunRecord

app = typer.Typer(
    name="rag",
    help="Resilient RAG: Self-healing RAG pipeline with bounded recovery and chaos evaluation",
    no_args_is_help=True,
)


@app.command()
def ingest(
    dir: Path = typer.Option(
        Path("data/raw"),
        "--dir",
        "-d",
        help="Directory of source documents to index",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-index all documents ignoring cache manifest",
    ),
) -> None:
    """Ingest, chunk, and index documents into vector store and BM25."""
    if not dir.is_dir():
        typer.echo(f"Error: Directory '{dir}' does not exist.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Ingesting documents from {dir} (force={force})...")
    settings = get_settings()
    components = build_components(settings)

    pipeline = IngestionPipeline(
        embedder=components.embedder,
        vector_store=components.vector_store,
        bm25_index=components.bm25_index,
    )

    result = pipeline.ingest_directory(dir, force=force)
    typer.echo(
        f"Ingestion complete: {result.indexed_files} indexed, "
        f"{result.skipped_files} skipped, {result.total_chunks} total chunks."
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    healing: bool = typer.Option(
        True,
        "--healing/--no-healing",
        help="Enable or disable self-healing supervisor",
    ),
    mode: str = typer.Option(
        "hybrid",
        "--mode",
        "-m",
        help="Retrieval mode: hybrid, vector, bm25",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        "-k",
        help="Number of chunks to retrieve",
    ),
) -> None:
    """Ask a question against the indexed knowledge base."""
    settings = get_settings()
    components = build_components(settings)

    typer.echo(f"Processing question: {question}")
    state = components.run(
        question=question,
        healing_enabled=healing,
        retrieval_mode=mode,
        top_k=top_k,
    )

    outcome = state.get("outcome")
    outcome_str = outcome.value if isinstance(outcome, Outcome) else str(outcome or "failed")
    answer = state.get("answer", "")
    citations = state.get("citations", [])
    heal_attempts = state.get("heal_attempts", 0)
    heal_history = state.get("heal_history", [])

    typer.echo(f"\nOutcome: {outcome_str}")
    typer.echo(f"Answer: {answer}\n")
    if citations:
        typer.echo("Citations:")
        for c in citations:
            typer.echo(f"  - [{c.chunk_id}] {c.text[:80]}...")
    if heal_attempts > 0:
        typer.echo(f"\nHealing Actions ({heal_attempts} attempts):")
        for fail, act in heal_history:
            typer.echo(f"  - Failure: {fail} -> Action: {act}")


@app.command()
def chaos(
    scenario: str = typer.Option(
        "all",
        "--scenario",
        "-s",
        help="Scenario name or 'all'",
    ),
    conditions: str = typer.Option(
        "A,B,C",
        "--conditions",
        "-c",
        help="Comma-separated conditions (A,B,C)",
    ),
    scenarios_file: Path = typer.Option(
        Path("configs/scenarios.yaml"),
        "--scenarios-file",
        help="Path to scenarios YAML",
    ),
    golden_set: Path = typer.Option(
        Path("data/golden/golden_set.jsonl"),
        "--golden-set",
        help="Path to golden set JSONL",
    ),
    out: Path = typer.Option(
        Path("results/runs.jsonl"),
        "--out",
        "-o",
        help="Output JSONL file path",
    ),
    seeds: str = typer.Option(
        "42",
        "--seeds",
        help="Comma-separated random seeds",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Compute budget and exit without executing runs",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip interactive confirmation",
    ),
) -> None:
    """Run reproducible chaos experiments across conditions A, B, and C."""
    if not scenarios_file.is_file():
        typer.echo(f"Error: Scenarios file '{scenarios_file}' not found.", err=True)
        raise typer.Exit(code=1)
    if not golden_set.is_file():
        typer.echo(f"Error: Golden set file '{golden_set}' not found.", err=True)
        raise typer.Exit(code=1)

    all_scenarios = load_scenarios(scenarios_file)
    if scenario != "all":
        selected_scenarios = [s for s in all_scenarios if s.name == scenario]
        if not selected_scenarios:
            typer.echo(f"Error: Scenario '{scenario}' not found in {scenarios_file}", err=True)
            raise typer.Exit(code=1)
    else:
        selected_scenarios = all_scenarios

    cond_list = [c.strip().upper() for c in conditions.split(",") if c.strip()]
    seed_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    golden_questions = load_golden_set(golden_set)
    question_strings = [q.question for q in golden_questions]

    runner = ExperimentRunner()
    budget = runner.dry_run(
        questions=question_strings,
        scenarios=selected_scenarios,
        seeds=seed_list,
        conditions=cond_list,
    )

    typer.echo("Chaos Experiment Configuration:")
    typer.echo(f"  Questions: {budget['questions_count']}")
    typer.echo(f"  Scenarios: {budget['scenarios_count']}")
    typer.echo(f"  Seeds: {budget['seeds_count']} {seed_list}")
    typer.echo(f"  Conditions: {budget['conditions']}")
    typer.echo(f"  Total planned runs: {budget['total_runs']}")
    typer.echo(f"  Estimated LLM calls: {budget['estimated_llm_calls']}")

    if dry_run:
        typer.echo("\n[Dry run] Exiting without executing runs.")
        return

    if not yes:
        confirm = typer.confirm("Do you want to proceed with executing these runs?")
        if not confirm:
            typer.echo("Aborted by user.")
            raise typer.Exit(code=0)

    settings = get_settings()
    components = build_components(settings)

    typer.echo(f"\nExecuting experiments and flushing to {out}...")
    results = runner.run_experiment(
        components=components,
        questions=question_strings,
        scenarios=selected_scenarios,
        seeds=seed_list,
        conditions=cond_list,
        output_file=out,
    )
    typer.echo(f"Execution complete. Total executed runs: {len(results)}")


@app.command()
def scorecard(
    input: Path = typer.Option(
        Path("results/runs.jsonl"),
        "--input",
        "-i",
        help="Path to completed runs JSONL file",
    ),
    output_md: Path = typer.Option(
        Path("results/scorecard.md"),
        "--output-md",
        help="Path for generated Markdown scorecard",
    ),
    output_csv: Path = typer.Option(
        Path("results/scorecard.csv"),
        "--output-csv",
        help="Path for generated CSV scorecard",
    ),
) -> None:
    """Compute Wilson score statistical intervals and overhead metrics from runs JSONL."""
    if not input.is_file():
        typer.echo(f"Error: Input runs file '{input}' not found.", err=True)
        raise typer.Exit(code=1)

    records: list[RunRecord] = []
    with open(input, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            # Map outcome string to enum
            if "outcome" in data and isinstance(data["outcome"], str):
                try:
                    data["outcome"] = Outcome(data["outcome"])
                except ValueError:
                    data["outcome"] = Outcome.FAILED
            records.append(RunRecord(**data))

    if not records:
        typer.echo(f"No run records found in {input}.", err=True)
        raise typer.Exit(code=1)

    result = compute_scorecard(records)
    md_content = scorecard_to_markdown(result)
    csv_content = scorecard_to_csv(result)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(md_content, encoding="utf-8")
    output_csv.write_text(csv_content, encoding="utf-8")

    typer.echo(f"Scorecard written to:\n  - Markdown: {output_md}\n  - CSV: {output_csv}\n")
    typer.echo(md_content)


@app.command()
def eval(
    golden_set: Path = typer.Option(
        Path("data/golden/golden_set.jsonl"),
        "--golden-set",
        help="Path to golden set JSONL",
    ),
    limit: int = typer.Option(
        0,
        "--limit",
        "-l",
        help="Maximum questions to evaluate (0 for all)",
    ),
    use_ragas: bool = typer.Option(
        False,
        "--use-ragas",
        help="Invoke RAGAS evaluation",
    ),
    out: Path = typer.Option(
        Path("results/eval_report.json"),
        "--out",
        "-o",
        help="Output report JSON file path",
    ),
) -> None:
    """Evaluate pipeline on the golden question set."""
    if not golden_set.is_file():
        typer.echo(f"Error: Golden set file '{golden_set}' not found.", err=True)
        raise typer.Exit(code=1)

    settings = get_settings()
    components = build_components(settings)

    actual_limit = limit if limit > 0 else None
    typer.echo(f"Running benchmark evaluation (limit={actual_limit}, ragas={use_ragas})...")

    summary = run_benchmark_eval(
        components=components,
        golden_path=golden_set,
        limit=actual_limit,
        use_ragas=use_ragas,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, indent=2)

    typer.echo("\nEvaluation complete:")
    typer.echo(f"  Total questions: {summary.total_questions}")
    typer.echo(f"  Answered: {summary.answered_count}")
    typer.echo(f"  Mean key-fact coverage: {summary.mean_key_fact_coverage:.1%}")
    typer.echo(f"  Detailed report saved to: {out}")


@app.command(name="langsmith-upload-dataset")
def langsmith_upload_dataset_cmd(
    dataset_name: str = typer.Option(
        "kubernetes-golden-set",
        "--dataset-name",
        help="Target dataset name in LangSmith",
    ),
    file: Path = typer.Option(
        Path("data/golden/golden_set.jsonl"),
        "--file",
        "-f",
        help="Path to golden set JSONL",
    ),
) -> None:
    """Upload golden questions to LangSmith as an evaluation dataset."""
    success = upload_golden_dataset(dataset_name=dataset_name, file_path=file)
    if success:
        typer.echo(f"Successfully uploaded dataset '{dataset_name}' to LangSmith.")
    else:
        typer.echo(
            "Upload failed. Ensure LANGSMITH_TRACING=true, LANGSMITH_API_KEY is set, and file exists.",
            err=True,
        )
        raise typer.Exit(code=1)


def main() -> Any:
    """Entrypoint for the rag CLI tool."""
    return app()


if __name__ == "__main__":
    main()
