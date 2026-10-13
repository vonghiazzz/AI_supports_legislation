# Phase 3 & 4 Comprehensive Implementation Plan

> **Status:** DESIGN / PLANNED ONLY — not the current runtime implementation.
> **Source of truth:** CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION.


This document integrates all specifications, migration steps, testing protocols, and completion definitions for Phase 3 and Phase 4.

---

## 1. Migration Plan

### Backward Compatibility
* Existing Schema v2 files will remain fully functional during development.
* The API services will load fallback values when encountering documents lacking the enriched fields.

### Rollback Strategy
* Prior to executing any database migrations or index rebuilds, a snapshot is captured (`backend/data/snapshots/`).
* Standard backup scripts will revert configuration paths to legacy datasets if failures occur.

---

## 2. Comprehensive Acceptance Gates

| Gate ID | Gate Name | PASS Criteria | FAIL Criteria | Required Evidence |
| :--- | :--- | :--- | :--- | :--- |
| **P3-G1** | Corpus Schema | 100% articles successfully parse into Schema v3 | Any validation exceptions or missing metadata | Validator output logs |
| **P3-G2** | Enrichment Pipeline | Pipeline runs to completion without null fields | LLM errors or empty attributes | Script execution logs |
| **P3-G3** | Leakage Control | Zero ground truth query terms leak into corpus text | Query leakage audit returns positive matches | Leakage audit report |
| **P3-G4** | Temporal Reasoning | Mismatches after filtering = 0 | Temporal mismatches detected in evaluation | RAG evaluation temporal log |
| **P4-G1** | Query Profiling | Profiler accuracy $\ge 90\%$ | Intent classification accuracy $< 90\%$ | Intent test run output |
| **P4-G2** | Adaptive Routing | Correct routing based on query profile matrix | Citation query routed to vector search | Router test output |
| **P4-G3** | Evaluation | Adaptive hybrid routed score exceeds baselines | Underperforms BM25 or FAISS standalone | Evaluation metrics tables |

---

## 3. Phase Completion Definition

Phase 3 and Phase 4 will be considered **APPROVED** if and only if all the following conditions are met:

### Implementation Scope
1. Enriched corpus (Schema v3) populated for all civil, land, and marriage law datasets.
2. Temporal reasoning module integrated inside `search_service.py`.
3. Intent classifier module implemented inside `router_service.py`.

### Experimental Validation
1. Dynamic RAG evaluation pipeline executed on the entire 107 benchmark query set.
2. Output table confirms temporal mismatch rate of 0.00%.

### Required Evidence Files
1. `validate_benchmark_v3.log` showing 100% corpus schema validation success.
2. `leakage_audit_report.json` showing 0 leaks.
3. `final_evaluation_report.md` displaying the final metrics table (proving routed hybrid search exceeds standard baselines).
