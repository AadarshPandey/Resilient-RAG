# Metrics Definitions and Methodology

This document formalizes the statistical metrics and evaluation methodology used in Resilient RAG.

---

## 1. Experimental Conditions

The chaos testing harness evaluates queries across three strictly separated conditions:

| Condition | Injected Faults | Self-Healing Supervisor | Purpose |
|---|---|---|---|
| **A** | None | Off | Quality Ceiling (baseline performance) |
| **B** | Active | Off | Cost of Failure (unmitigated impact) |
| **C** | Active | On | Benefit & Overhead of Self-Healing |

---

## 2. Statistical Metric Definitions

### 2.1 Task Success Rate
The proportion of queries ending in `ANSWERED` where key factual claims match reference facts:
$$\text{Task Success} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{outcome}_i = \text{ANSWERED} \land \text{accurate}_i)}{N}$$

Reported with **Wilson 95% Confidence Intervals** rather than normal approximations to ensure valid bounds for small samples ($N < 100$) or extreme probabilities ($p \approx 0$ or $p \approx 1$).

#### Wilson Score Interval Formula
For observed proportion $p = \frac{x}{n}$ with critical value $z = 1.96$:
$$\text{Center} = \frac{p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}}$$
$$\text{Margin} = \frac{z \cdot \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
$$\text{CI} = [\text{Center} - \text{Margin},\; \text{Center} + \text{Margin}]$$

---

### 2.2 Safe-Outcome Rate
Measures resilience against fabricating ungrounded or compromised answers. A run is defined as "Safe" if it succeeds or terminates in a justified abstention or escalation:
$$\text{Safe-Outcome Rate} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{outcome}_i \in \{\text{ANSWERED (accurate)}, \text{ABSTAINED}, \text{ESCALATED}\})}{N}$$

---

### 2.3 Recovery Rate
Measures the effectiveness of healing policies specifically when failures are triggered:
$$\text{Recovery Rate} = \frac{\sum_{i \in \text{Faulty}} \mathbb{I}(\text{outcome}_i \in \{\text{ANSWERED (accurate)}, \text{ABSTAINED}, \text{ESCALATED}\})}{|\text{Faulty}|}$$

---

### 2.4 Mean Time To Recovery (MTTR)
The mean virtual clock duration from first fault detection to successful completion among recovered runs:
$$\text{MTTR} = \frac{1}{|\text{Recovered}|} \sum_{i \in \text{Recovered}} (t_{\text{finalize}} - t_{\text{first\_fault}})$$

---

### 2.5 Paired Overhead
Evaluates the marginal resource cost of self-healing by computing paired differences on identical $(q, \text{scenario}, \text{seed})$ triplets between Condition C and Condition B:
- **Added LLM Calls**: $\text{calls}_C - \text{calls}_B$
- **Added Token Usage**: $\text{tokens}_C - \text{tokens}_B$
- **Added Virtual Latency**: $\text{latency}_C - \text{latency}_B$
