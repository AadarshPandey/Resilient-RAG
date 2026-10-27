# Production Release Checklist

This document details the mandatory verification and deployment steps required before tagging and shipping a release of the Resilient RAG service.

---

## 1. Pre-Release Verification

Every release candidate must pass all automated verification checks:

- [ ] **Quality Gate**: Run local quality gate:
  ```bash
  make gate
  ```
  Must pass with 0 Ruff errors, 0 Mypy strict errors, and >=85% test coverage across all source modules.
- [ ] **Offline Chaos Regression Gate**: Verify that safe-outcome rates have not degraded from the baseline:
  ```bash
  uv run python scripts/chaos_regression_gate.py
  ```
  Condition C safe-outcome rate must be >= 95.0%.
- [ ] **Docker Image Build**:
  ```bash
  docker compose build
  ```
  Image must build cleanly without warnings or root user violations.
- [ ] **Container Integration Verification**:
  ```bash
  docker compose up -d
  curl -f http://localhost:8000/health
  curl -f http://localhost:8000/metrics
  docker compose down
  ```

---

## 2. Security & Compliance Checklist

- [ ] **No Committed Secrets**: Verify no API keys, tokens, or environment files are tracked:
  ```bash
  git status
  git diff --staged
  ```
- [ ] **Dependency Audit**:
  ```bash
  uv run pip-audit || true
  ```
- [ ] **Non-root Container Execution**: Confirm container runs under `appuser` (UID 1000).
- [ ] **PII & Prompt Injection Tests**: Confirm `tests/test_security.py` passes all 20+ injection regression attack strings.

---

## 3. Deployment & Rollout Sequence

1. **Database & Storage Migrations**:
   - Verify Qdrant collection is online: `curl http://qdrant:6333/collections`
   - Verify PostgreSQL checkpointer schema: `pg_isready -h postgres -U rag`
2. **Canary Deployment**:
   - Deploy new container image to a single replica in the cluster.
   - Run warmup sequence: `RAG_WARMUP_ON_STARTUP=true`.
3. **Traffic Health Check**:
   - Inspect Prometheus `/metrics`:
     - `rag_circuit_breaker_state{tier="0"} == 0` (closed)
     - `rag_requests_total{outcome="answered"}` progressing normally
     - `histogram_quantile(0.95, sum(rate(rag_request_duration_seconds_bucket[5m])) by (le)) < 2.5`
4. **Full Fleet Rollout**:
   - Promote to 100% traffic once canary satisfies error and latency budgets.

---

## 4. Rollback Plan

If an alert fires (`LLMCircuitBreakerOpen`, `AbstainRateSpike`, or `HighRequestLatency`):
1. Immediately roll back image to previous stable git SHA:
   ```bash
   git checkout <PREVIOUS_STABLE_SHA>
   docker compose up -d --build api
   ```
2. Verify system recovery using `GET /health` and Grafana dashboard (`deploy/grafana/dashboard.json`).
