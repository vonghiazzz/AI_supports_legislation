# Runtime Query Profiling and Adaptive Routing Design

> **Status:** DESIGN / PLANNED ONLY — not the current runtime implementation.
> **Source of truth:** CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION.


This specification defines the query intent classification and routing architecture (Phase 4).

---

## 1. Intent Classification & Adaptive Routing

Legal queries vary dramatically in intent (e.g., specific statutory citations vs. complex factual narratives). A single retrieval strategy is suboptimal. Adaptive routing profiles queries at runtime and selects the best search execution path.

```
                  ┌───────────────┐
                  │  User Query   │
                  └───────┬───────┘
                          │
            [Extract Intent / Profiler]
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
[Citation Search]    [Semantic Search]   [Hybrid Search]
(Strict BM25)         (Dense Vector)     (Sparse + Dense)
```

### Routing Matrix:
* **`exact_citation`**: Routes to strict BM25 search over titles and articles.
* **`factual_narrative`**: Routes to Dense Vector search using dense embeddings.
* **`mixed_multi_issue`**: Routes to hybrid fusion search (BM25 + FAISS) with reciprocal rank fusion (RRF) or softmax normalization.

---

## 2. Target Files

* **New Files**:
  * `backend/app/services/router_service.py` (query intent classification service)
* **Modified Files**:
  * `backend/app/services/search_service.py` (updated to support routed execution paths)

---

## 3. Evaluation Plan

### Expected Metric Impact
* **WeightedRecall**: +5% (by matching search strategies to query profile).
* **nDCG**: +6% (since routing avoids diluting specific search metrics).
* **Semantic Drift Reduction**: Substantial reduction in semantic drift for exact intent queries.

---

## 4. Acceptance Gates

### P4-G1 Query Profiling
* **PASS Criteria**: Classifier accuracy $\ge 90\%$ on test cases with known profile annotations.
* **FAIL Criteria**: Accuracy $< 90\%$.
* **Required Evidence**: Unit test execution results comparing predictions with query profile ground truth labels.

### P4-G2 Adaptive Routing
* **PASS Criteria**: Queries are successfully routed to the specific index/strategy corresponding to their profile.
* **FAIL Criteria**: A citation query routes to dense semantic search or vice versa.
* **Required Evidence**: Service execution logs showing active routing.

### P4-G3 Evaluation
* **PASS Criteria**: The adaptive routed search achieves higher WeightedRecall than either BM25 or FAISS standalone baselines.
* **FAIL Criteria**: Hybrid adaptive routing underperforms standalone BM25 or FAISS search.
* **Required Evidence**: Tabular results in RAG evaluation logs.
