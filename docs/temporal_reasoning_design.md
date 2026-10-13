# Temporal Reasoning Engine Design

> **Status:** DESIGN / PLANNED ONLY — not the current runtime implementation.
> **Source of truth:** CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION.


This document details the architecture, temporal filtering logic, and verification specifications of the Temporal Reasoning Engine (Phase 3).

---

## 1. Engine Design and Temporal Filtering Logic

The temporal reasoning engine resolves a major RAG challenge: returning laws that were inactive or repealed at the time of the dispute (temporal mismatch).

```
[User Query] ──> [Extract Dispute Year (T)] ──> [Filter Candidate Laws] ──> [Rerank & Output]
                                                        │
                                          Compare T against:
                                          - Law Active Start (T_start)
                                          - Law Active End (T_end)
                                          - Valid if: T_start <= T <= T_end
```

### Filtering Strategy:
1. **Dispute Year Extraction**: Programmatically extracts the year of dispute ($T$) from the metadata of the query. If missing, it defaults to the judgment year, or falls back to the current year.
2. **Statutory Active Window**: Every law article in the corpus has an active window $[T_{start}, T_{end}]$ specified in `laws_config.json` and enriched inside `temporal_metadata`.
3. **Strict Intersection**: Filters out any retrieved article where $T < T_{start}$ or $T > T_{end}$.
4. **Amendment Routing**: If a law was amended in year $T_{amend}$, the engine routes to the specific version of the article that was active at time $T$.

---

## 2. Target Files

* **Modified Files**:
  * `backend/app/services/search_service.py` (integrate strict temporal filtering logic inside vector search and BM25 search calls).
  * `backend/scripts/Evaluation/run_rag_evaluation.py` (extend evaluation comparison matrix to track `mismatch_before` vs `mismatch_after`).

---

## 3. Evaluation Plan

### Expected Metric Impact
* **Temporal Mismatch**: Reduced to **0%** (absolute elimination of temporal drift).
* **Recall_core**: +3% (by filtering out obsolete matching terms that pollute top-k results).
* **nDCG**: +4% (as correct temporal versions get higher positions).
* **Semantic Drift Reduction**: Obsolete laws no longer drift into model context.

---

## 4. Acceptance Gate: P3-G4 Temporal Reasoning

* **PASS Criteria**:
  * All retrieved articles are active for the corresponding dispute year of the query.
  * Temporal mismatch count under `temporal_comparison` is exactly `0`.
* **FAIL Criteria**:
  * Any law article with $T > T_{end}$ or $T < T_{start}$ is retrieved in the final top-k context.
* **Required Evidence**:
  * Output log of the evaluation pipeline showing:
    - `Total Temporal Mismatches (After Filter) = 0`
