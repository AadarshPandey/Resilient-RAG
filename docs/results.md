# Resilient RAG: Chaos Evaluation Results

This document records the empirical results of evaluating the Resilient RAG pipeline across three experimental conditions under simulated chaos.

## 1. Experimental Methodology

To evaluate resilience without confounding factors, evaluations are conducted across three controlled conditions:

| Condition | Description | Self-Healing Active? | Fault Injection? | Purpose |
|---|---|:---:|:---:|---|
| **Condition A** | Clean Ceiling | No | No | Benchmark upper-bound performance on normal workloads. |
| **Condition B** | Unprotected Baseline | No | Yes | Measure vulnerability and failure modes under infrastructure and model faults. |
| **Condition C** | Resilient RAG | Yes | Yes | Validate automated detection, recovery actions, and bounded supervisor performance. |

Each evaluation runs across the 14 chaos scenarios defined in `configs/scenarios.yaml` with fixed random seeds for 100% reproducibility.

---

## 2. Statistical Scorecard

The results below were calculated over 420 paired runs (14 scenarios × 10 golden benchmark questions × 3 conditions) using Wilson 95% confidence intervals:

| Condition | Total Runs | Task Success Rate (95% CI) | Safe-Outcome Rate (95% CI) | Recovery Rate | MTTR |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **A** (Ceiling) | 140 | **100.0%** [97.3%, 100.0%] | **100.0%** [97.3%, 100.0%] | 100.0% | 0.00s |
| **B** (Baseline) | 140 | **14.3%** [9.4%, 21.0%] | **92.9%** [87.4%, 96.1%] | 91.7% | 0.00s |
| **C** (Resilient RAG) | 140 | **21.4%** [15.4%, 28.9%] | **100.0%** [97.3%, 100.0%] | **100.0%** | 1.33s |

### Paired Overhead (Condition C vs Condition B)

Recovery actions introduce minimal computational overhead:
- **Mean Added LLM Invocations:** `+0.14` calls / query
- **Mean Added Token Usage:** `+9.6` tokens / query
- **Mean Virtual Latency:** `< 0.01` s

![Scorecard Comparison](scorecard.png)

---

## 3. Scenario Analysis

### A. Infrastructure & Backend Outages
- `vector_backend_down`: Condition B crashes immediately upon retrieval failure. Condition C detects `RETRIEVAL_BACKEND_DOWN`, executes `apply_bm25_only`, and recovers or escalates gracefully if BM25 is also unavailable.
- `bm25_backend_down`: Condition C falls back cleanly to `apply_vector_only`.
- `both_retrievers_down`: When both backends fail simultaneously, Condition C exhausts its retrieval options and safely terminates in `ESCALATE` rather than hanging or returning hallucinations.

### B. LLM Tier Failures & Circuit Breakers
- `llm_rate_limit_transient` & `llm_timeout_transient`: The per-tier circuit breaker and exponential backoff retry on the same tier successfully ride out transient glitches with zero human intervention.
- `llm_tier0_outage`: Condition C supervisor detects persistent tier-0 outage, selects `apply_model_fallback`, switches active tier to `1`, and successfully answers the question.
- `llm_all_tiers_down`: All breakers open; supervisor halts recovery budget and routes to `ESCALATE`.

### C. Context Corruption & Poisoning
- `retriever_prompt_injection`: Heuristic pattern detectors intercept jailbreak and instruction-override tokens in retrieved candidates; offending chunks are quarantined via `apply_quarantine_chunk` or the pipeline terminates safely.
- `llm_ungrounded_answer`: Post-generation claim verification checks candidate claims against context passages; ungrounded statements trigger `apply_regenerate_strict` in strict mode.

---

## 4. Reproducing the Scorecard

To reproduce these exact benchmark numbers and regenerate the scorecard and chart:

```bash
# Execute baseline chaos evaluation across all scenarios
uv run python scripts/generate_baseline_scorecard.py

# Compute scorecard tables
uv run rag scorecard --input results/runs.jsonl --output-md results/scorecard.md --output-csv results/scorecard.csv

# Generate comparative grouped bar chart
uv run python scripts/plot_results.py --input results/scorecard.csv --output results/scorecard.png
```
