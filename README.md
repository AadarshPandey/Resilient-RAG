# Resilient RAG

**A RAG pipeline that detects its own failures, recovers with bounded deterministic policies, and proves it with chaos experiments.**

Built with **PyTorch · LangChain · LangGraph · LangSmith · RAGAS · FastAPI · Streamlit · Qdrant**

> **Status:** Phase 1 (Prototype) in progress, Phase 2 (Production) planned. Track progress in [`PLAN.md`](PLAN.md). Items are marked ✅ done, 🚧 in progress or ⬜ not started; update them as you build.

---

## Contents

1. [What it does](#1-what-it-does)
2. [Architecture](#2-architecture)
3. [Tech stack](#3-tech-stack)
4. [Feature status](#4-feature-status)
5. [Quickstart (prototype)](#5-quickstart-prototype)
6. [CLI, API and UI](#6-cli-api-and-ui)
7. [Chaos experiments and evaluation](#7-chaos-experiments-and-evaluation)
8. [LangSmith observability](#8-langsmith-observability)
9. [Production variant](#9-production-variant)
10. [Configuration](#10-configuration)
11. [Testing](#11-testing)
12. [Limitations](#12-limitations)
13. [Docs and license](#13-docs-and-license)

---

## 1. What it does

Normal RAG works until a dependency breaks. Resilient RAG treats failure as a first-class, testable concern:

- **Detect** typed faults at every graph node (backend down, empty or low-relevance retrieval, prompt injection, LLM error, malformed output, context overflow, ungrounded answer).
- **Decide** with a pure deterministic policy: ordered, low-cost-first actions, never repeating the same action for the same failure.
- **Act** within a hard budget (`max_heal_attempts = 4`), then end safely as **Answer, Abstain or Escalate**.
- **Measure** with a seeded chaos harness comparing three conditions:

| Condition | Faults | Healing | Purpose |
|---|---|---|---|
| **A** | none | off | quality ceiling |
| **B** | injected | off | cost of failure |
| **C** | injected | on | benefit and overhead of healing |

## 2. Architecture

```mermaid
flowchart TD
    Start([Question]) --> QP[query_prep]
    QP --> Ret["retrieve: Qdrant dense + BM25, RRF k=60"]
    Ret --> Grade["grade: cross-encoder rerank"]
    Grade --> Gen["generate: Tier-0 LLM"]
    Gen --> Ver["verify: claim groundedness"]
    Ver --> Fin["finalize: audit record"]
    Fin --> Done([Answer / Abstain / Escalate])

    Ret -- fault --> Heal["heal: detect -> select -> execute"]
    Grade -- fault --> Heal
    Gen -- fault --> Heal
    Ver -- fault --> Heal

    Heal -- "retry / widen_topk / rewrite / bm25_only / vector_only" --> Ret
    Heal -- quarantine_chunk --> Grade
    Heal -- "regenerate_strict / compress_context / model_fallback" --> Gen
    Heal -- "retry_same / model_fallback" --> Ver
    Heal -- "abstain / escalate" --> Fin
```

**Design rules**

| Rule | Why |
|---|---|
| Hybrid retrieval fails loudly (`BackendError`); single-backend failover is a healing action | No silent degradation, honest experiments |
| LLM client retries on its own tier only; tier switching happens via `MODEL_FALLBACK` | Keeps Condition B from healing itself |
| Per-tier circuit breakers (`closed`, `open`, `half_open`) | Isolated blast radius |
| Healing logic is pure Python, independent of LangGraph | Trivially unit-testable |
| Chaos delays use a virtual clock, never `time.sleep()` | Fast, deterministic runs |

## 3. Tech stack

| Concern | Choice |
|---|---|
| Deep learning runtime | **PyTorch** (via `sentence-transformers`): `BAAI/bge-small-en-v1.5` bi-encoder, `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker; CUDA, MPS or CPU auto-selected |
| LLM abstraction | **LangChain** chat-model adapters (Gemini default, OpenAI and Ollama optional) |
| Orchestration | **LangGraph** state machine with checkpointing |
| Tracing and experiments | **LangSmith** |
| Evaluation | **RAGAS** (faithfulness, answer relevancy, context precision and recall) plus custom chaos metrics |
| Vector store | **Qdrant** (local mode in Phase 1, server mode in Phase 2) |
| Sparse search | `rank-bm25` (Phase 1), OpenSearch or Qdrant sparse vectors (Phase 2) |
| API | **FastAPI** |
| Demo UI | **Streamlit** |
| Tooling | `uv`, `ruff`, `mypy`, `pytest` |

Use Python 3.12 or 3.13 unless you have verified that every dependency, especially `torch`, ships wheels for your Python version.

## 4. Feature status

| Feature | Phase | Status |
|---|---|---|
| Settings, typed errors, JSON logging, audit log | 1 | ⬜ |
| Loaders, chunking, deterministic IDs, concurrent ingestion | 1 | ⬜ |
| Dense store (Qdrant local) and BM25 index | 1 | ⬜ |
| RRF hybrid retrieval and cross-encoder grading | 1 | ⬜ |
| Multi-tier LLM client with circuit breakers | 1 | ⬜ |
| LangGraph pipeline with verify step | 1 | ⬜ |
| Detectors, selector, patterns, executor (self-healing) | 1 | ⬜ |
| Chaos wrappers, 14 scenarios, runner | 1 | ⬜ |
| Scorecard (Wilson CI, recovery, MTTR, overhead) | 1 | ⬜ |
| RAGAS evaluation and LangSmith tracing | 1 | ⬜ |
| CLI, FastAPI, Streamlit | 1 | ⬜ |
| Docker, Qdrant server, Postgres checkpointer | 2 | ⬜ |
| Async API, auth, rate limiting, Prometheus metrics | 2 | ⬜ |
| Queue-based ingestion | 2 | ⬜ |
| CI/CD with chaos regression gate | 2 | ⬜ |
| PyTorch-trained groundedness verifier (stretch) | 2 | ⬜ |

## 5. Quickstart (prototype)

```bash
uv sync
cp .env.example .env
```

```env
GEMINI_API_KEY=your-key
RAG_LLM_CHAIN=google:<tier0-model>,google:<tier1-model>
RAG_HEALING_ENABLED=true

# optional tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=resilient-rag
```

```bash
uv run python scripts/populate_corpus.py   # build the reference corpus
uv run rag ingest data/raw                  # index it
uv run rag ask "What is the role of kubelet on worker nodes?"
```

## 6. CLI, API and UI

```bash
# Ask, with and without healing
uv run rag ask "What port does etcd listen on?"
uv run rag ask "What port does etcd listen on?" --no-healing

# REST API: GET /health, POST /ask
uv run uvicorn --factory rag.api:create_app --port 8000

# Demo UI
uv run streamlit run app.py
```

Local Qdrant allows one process at a time. Stop the CLI before starting the API or Streamlit, or use Qdrant server mode (Phase 2).

`POST /ask` returns `outcome` (`answer`, `abstain` or `escalate`), the answer, citations, and a healing report (attempts, actions, detected faults).

## 7. Chaos experiments and evaluation

```bash
uv run rag chaos --dry-run                       # budget estimate, no API calls
uv run rag chaos --scenarios configs/scenarios.yaml \
    --n-questions 5 --seeds 1 --conditions A,B,C -y
uv run rag scorecard --runs results/runs.jsonl \
    --output-csv results/scorecard.csv --output-md results/scorecard.md
uv run python scripts/plot_results.py results/scorecard.csv
uv run rag eval --golden data/golden/golden_set.jsonl --limit 5   # RAGAS + key-fact baseline
```

**Fault scenarios (14)** cover LLM rate limits and timeouts, tier-0 outage, all-tiers outage, malformed JSON, ungrounded answers, context overflow, embedder failure, vector and BM25 outages, both retrievers down, empty retrieval and prompt-injection chunks.

**Metrics:** task success rate and safe-outcome rate (with Wilson 95% CI), recovery rate, MTTR, false-abstention rate, paired call, latency and token overhead (C vs B), plus RAGAS quality scores.

### Results

Fill this table from your own live run. Do not publish numbers you have not measured.

| Metric | A | B | C |
|---|---|---|---|
| Task success (95% CI) | | | |
| Safe-outcome rate | | | |
| Recovery rate | n/a | n/a | |
| MTTR | n/a | n/a | |
| Added LLM calls vs B | n/a | n/a | |
| RAGAS faithfulness | | | |

## 8. LangSmith observability

With `LANGSMITH_TRACING=true`, every graph node is traced. Tag runs with `condition`, `scenario`, `seed` and heal action so you can filter and compare in the UI. Upload `data/golden/golden_set.jsonl` as a LangSmith dataset to run and compare experiments across code versions. Traces complement, and do not replace, the local `data/audit.jsonl` record.

## 9. Production variant

Phase 2 turns the prototype's documented gaps into implemented features:

| Capability | Prototype | Production |
|---|---|---|
| Vector store | Local Qdrant | Qdrant server or cluster |
| Sparse index | In-memory BM25 | OpenSearch or Qdrant sparse vectors |
| Checkpointer | SQLite | Postgres (`langgraph-checkpoint-postgres`) |
| Ingestion | Thread pool | Redis-backed queue workers |
| API | Sync FastAPI | Async FastAPI, API-key auth, rate limits, request IDs |
| Metrics | LangSmith + audit log | plus Prometheus `/metrics` and Grafana dashboards |
| Delivery | `uv run` | Docker images, docker-compose, CI/CD, chaos regression gate |

```bash
docker compose up -d      # api + qdrant + postgres + redis (Phase 2)
```

## 10. Configuration

All settings use the `RAG_` prefix.

| Variable | Default | Purpose |
|---|---|---|
| `RAG_LLM_CHAIN` | none | Ordered LLM tiers, `provider:model` |
| `RAG_HEALING_ENABLED` | `true` | Master healing switch |
| `RAG_MAX_HEAL_ATTEMPTS` | `4` | Recovery budget per query |
| `RAG_QDRANT_URL` | unset | Set for server mode |
| `RAG_CHECKPOINT_DSN` | SQLite path | Postgres DSN in production |
| `RAG_DEVICE` | `auto` | `cuda`, `mps`, `cpu` |

## 11. Testing

```bash
uv run ruff check . && uv run ruff format --check . \
  && uv run mypy src scripts app.py && uv run pytest -q
```

The unit and e2e suite runs offline against deterministic fakes (hash embedder, lexical reranker, scripted chat model, virtual-latency fault wrappers), so it costs nothing and exercises every healing path.

## 12. Limitations

- Prototype corpus is small (about 36 short Kubernetes chunks); real workloads need long documents and harder distractors.
- Injected faults approximate real incidents but are synthetic.
- Prompt-injection detection is heuristic, not a guarantee.
- Scorecard numbers exist only after a live `rag chaos` run.

## 13. Docs and license

- [`PLAN.md`](PLAN.md): build plan for the prototype and production phases
- [`STORY.md`](STORY.md): project narrative
- `docs/metrics-definitions.md`, `docs/architecture-decision-records/` (ADRs 001 to 005)

License: MIT.
