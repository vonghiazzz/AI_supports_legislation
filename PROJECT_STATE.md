# AI Supports Legislation — Current Project State

Last updated: 2026-10-16
Source-of-truth rule: `CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION`

This document describes the current repository state after repository hygiene, frozen-baseline recovery, deterministic artifact reconstruction, FastAPI runtime retrieval recovery, and completion of EXP-02, EXP-03, and EXP-04.

---

## 1. Executive Summary

The project is a Vietnamese Legal RAG research and application system with three clearly separated concerns:

1. **Runtime application**
   - FastAPI legal search and legal Q&A API.
   - React frontend.
   - Local statutory retrieval using BM25 + FAISS.
   - Gemini is used for answer generation, not for baseline retrieval.

2. **Offline data pipelines**
   - Vietnamese statute extraction and Gemini-assisted enrichment.
   - Judgment OCR and benchmark construction.
   - Runtime retrieval artifact generation.

3. **Experimental evaluation**
   - Judgment-derived statutory retrieval benchmark.
   - BM25, Dense, Hybrid, reranking, temporal and diagnostic experiments.
   - Research code is more extensive than the production runtime and must not be conflated with it.

The previously broken FastAPI retrieval path has now been recovered.

The active runtime retrieval contract is:

```text
runtime_retrieval_v1
  -> baseline protocol: baseline_d0_v2
  -> corpus: 1,811 statutory articles
  -> query: Q0
  -> document representation: D0
  -> BM25: whitespace tokenizer
  -> Dense: keepitreal/vietnamese-sbert
  -> exact revision:
     a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23
  -> embedding dimension: 768
  -> L2 normalization
  -> FAISS: IndexFlatIP
  -> candidate pool: 30
  -> fusion: min-max
  -> weight: BM25 0.5 / Dense 0.5
  -> deterministic tie:
     score DESC, then standard_law_id ASC
  -> reranker: none in baseline runtime

Runtime artifacts have been reproduced against the historical frozen baseline with:
107 queries
3 retrieval configurations/query
321 / 321 top-20 ranking checks passed

FastAPI SearchService also reproduces the historical golden Hybrid top-20 ranking, and /search has been verified through TestClient.
2. Current Research Objective
The active research goal is:
Evaluate retrieval design choices for Vietnamese judgment-derived statutory retrieval under a controlled and reproducible protocol.

The benchmark task is not general facts-only QA.
Each judgment-derived query can require:
- substantive law,
- procedural law,
- core legal authorities,
- supporting legal authorities.
The benchmark therefore preserves two distinct dimensions:
Importance:
core / supporting

Legal function:
substantive / procedural

These dimensions must not be conflated.
The current experiment progression is:
B0  BM25
B1  Dense
E1  Static Hybrid
    ↓
EXP-02  D0 vs D1
    ↓
EXP-03  leakage-controlled static hybrid tuning
    ↓
Per-query oracle/headroom analysis
    ↓
EXP-04  selective Dynamic Hybrid nested CV
    ↓
Temporal retrieval / reranking
    ↓
Statistical analysis

EXP-02, EXP-03, and EXP-04 are complete. EXP-03 did not support promoting its CV-selected static policy. The subsequent oracle/headroom diagnostic found moderate but sparse and concentrated theoretical headroom, but EXP-04's leakage-controlled selective router did not convert that headroom into useful held-out overrides. Runtime and the research baseline remain unchanged at static 0.5/0.5 fusion.
3. Frozen Baseline Protocol
Protocol ID:
baseline_d0_v2

The superseded baseline_d0_v1 had nondeterministic ordering for exact Hybrid score ties.
The deterministic fix is:
combined.sort(    key=lambda item: (-item[1], item[0]))


Therefore ties are ordered by:
fusion score DESC
then standard_law_id ASC

Query representation
Q0 is the benchmark query with whitespace normalization only.
Conceptually:
" ".join(query.strip().split())


There is no:
- query expansion,
- intent routing,
- keyword extraction,
- temporal transformation,
- LLM rewriting.
Document representation
D0 is:
Tiêu đề: {title}. Nội dung: {content}

Both BM25 and Dense use the exact same D0 document text.
D1
D1 adds:
summary
lexical_keywords
keywords

D1 is not the current frozen runtime representation.
It was evaluated in EXP-02 and was not promoted to the frozen baseline.
4. Frozen Corpus
The primary controlled corpus contains exactly five laws.
Corpus key	Articles
bo_luat_dan_su_2015	689
bo_luat_to_tung_dan_su_2015	517
luat_dat_dai_2013	212
luat_dat_dai_2024	260
luat_hon_nhan_gia_dinh_2014	133
Total	1,811


Excluded from the frozen corpus:
test_law
mau_so_52

Combined corpus fingerprint:
66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58

Every runtime metadata record has a canonical:
standard_law_id

Example:
điều 129 bo_luat_dan_su_2015

law_id alone must not be treated as globally unique because article numbering restarts in each law.
5. Benchmark State
The current benchmark contains:
107 judgment-derived queries

Ground-truth labels:
total       1,212
core          755
supporting     457

Corpus coverage:
in frozen corpus      1,018
outside corpus          194
coverage              83.99%

All 107 queries contain at least one in-corpus ground-truth authority.
Ground-truth weighting:
core        = 3
supporting  = 1

Primary metric protocol:
world       closed_world
primary K   10
reported K  5, 10, 20

Primary metrics:
Recall
nDCG
MRR
Weighted Recall

The benchmark remains predominantly AI-generated.
Human annotation is currently only a small experiment:
3 / 107 queries
Cohen's Kappa = 0.3438

This is not sufficient to claim dataset-level human adjudication.
6. Frozen Baseline Results
Primary closed-world @10:
Retriever	Recall	nDCG	MRR	Weighted Recall
BM25	0.0760	0.0333	0.1415	0.0522
FAISS	0.0632	0.0513	0.1450	0.0591
Static Hybrid	0.0883	0.0529	0.1901	0.0725


Static Hybrid wins all four predefined primary metrics at K=10.
This does not by itself establish statistical significance.
No significance claim should be made until an appropriate paired analysis is performed.

6A. EXP-02 — D0 vs D1

Status: **COMPLETE**

Official runs:

- D0 control: `exp02_d0_vs_d1_v1_d0_20261013_085034_383461_4828f45f`
- D1 treatment: `exp02_d0_vs_d1_v1_d1_20261013_090537_436695_379066cd`

Both runs evaluated all 107 queries over the same 1,811-document corpus at commit `bcfd918b89c533a6138e1187cf08ebe0beebe9c6`. Their manifests report clean Git provenance, the frozen corpus fingerprint, the frozen Q0 fingerprint, the exact Vietnamese-SBERT revision, normalized 768-D embeddings, and `IndexFlatIP`.

Observed primary results, closed-world @10:

| Method | Metric | D0 | D1 | Delta |
|---|---:|---:|---:|---:|
| BM25 | Recall | 0.076011 | 0.081298 | +0.005287 |
| BM25 | nDCG | 0.033347 | 0.039831 | +0.006484 |
| BM25 | MRR | 0.141470 | 0.147567 | +0.006097 |
| BM25 | Weighted Recall | 0.052248 | 0.057271 | +0.005023 |
| FAISS | Recall | 0.063228 | 0.063908 | +0.000680 |
| FAISS | nDCG | 0.051339 | 0.049385 | -0.001954 |
| FAISS | MRR | 0.144975 | 0.131872 | -0.013103 |
| FAISS | Weighted Recall | 0.059063 | 0.059895 | +0.000832 |
| Hybrid | Recall | 0.088254 | 0.082146 | -0.006108 |
| Hybrid | nDCG | 0.052870 | 0.053311 | +0.000441 |
| Hybrid | MRR | 0.190090 | 0.211041 | +0.020950 |
| Hybrid | Weighted Recall | 0.072481 | 0.066704 | -0.005777 |

Observed paired uncertainty at K=10:

- All 12 tested method/metric comparisons had paired-bootstrap 95% confidence intervals containing zero.
- All paired sign-flip permutation tests had `p >= 0.05`.
- BM25 nDCG showed a suggestive positive trend (`p = 0.07564`), not a statistically significant result.
- Hybrid queries where MRR improved while Recall decreased on the same query: `0`; the aggregate movements came from different query subsets.

Interpretation:

The current benchmark does not provide sufficient evidence to establish a consistent D1 improvement over D0. BM25 shows suggestive lexical benefits, while Dense and Hybrid effects are mixed. Hybrid D1 was higher on all four reported metrics at K=20, but K=10 remains the predeclared primary endpoint.

Observed D1/truncation audit:

- All 1,811 documents contain `summary`, `lexical_keywords`, and `keywords`; missing enrichment fields did not confound the comparison.
- Median document length increased from 699 to 1,036 characters.
- With the model's 256-token maximum, 619 D0 documents and 967 D1 documents exceed the limit.
- Dense D0 and D1 truncated token sequences are identical for 619 documents; Dense input changed for 1,192 documents.

Hypothesis / limitation:

Dense truncation is a plausible explanation for some mixed Dense behavior, but EXP-02 does not establish it as a causal mechanism.

Decision:

- D1 is not promoted.
- Runtime remains D0 under `runtime_retrieval_v1`.
- The research baseline remains Q0/D0 under `baseline_d0_v2`.
- Runtime artifacts remain unchanged.
- EXP-03 subsequently evaluated static fusion weights without promoting a replacement policy; see Section 6B.

6B. EXP-03 — Leakage-Controlled Static Fusion Weight

Status: **COMPLETE**

Experiment: `exp03_static_hybrid_weight_v1`

Protocol: `exp03_static_hybrid_5fold_cv_v1`

Control: `baseline_d0_v2`

Official run: `exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23`

The run completed at Git commit `e78d9c4e62e7468ca4dc68d6255628b0c6e7baa6` with `git_dirty=false`, `tracked_code_dirty=false`, and `baseline_source_dirty=false`. It evaluated all 107 Q0 queries against the frozen 1,811-document D0 corpus with no temporal filter, whitespace BM25, normalized 768-D Vietnamese-SBERT embeddings at revision `a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23`, `IndexFlatIP`, `pool_k=30`, and `stored_k=20`.

Golden parity passed against `exp02_d0_vs_d1_v1_d0_20261013_085034_383461_4828f45f`. BM25, FAISS, and `Hybrid_Static_Fixed` rankings matched exactly through depth 20; summary metrics used tolerance `1e-12`.

The predeclared protocol used five deterministic folds of `22/22/21/21/21`, seed 42, and `sha256_protocol_seed_query_id_contiguous_v1`. The fold-mapping fingerprint is `94e9f68e7113f21b24e4cfbbccb2faacac1bfb106ac35e5c2167e7f51eee7310`; the candidate-cache fingerprint is `4e94fc54bf2dc1ec09c4fbb605daadb1ba0dfa8c7c0c10a2d40ea34e90c2c805`. Each training split selected from `w_bm25 = 0.0, 0.1, ..., 1.0` using closed-world Weighted Recall@10 with `alpha_core=3` and `alpha_support=1`.

Selected BM25 weights were:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| `w_bm25` | 0.4 | 0.5 | 0.5 | 0.5 | 0.4 |

Observed primary results, closed-world @10:

| Metric | OOF selected policy | Fixed 0.5 control | Delta |
|---|---:|---:|---:|
| Recall | 0.078065498578 | 0.088253830495 | -0.010188332 |
| nDCG | 0.049783452607 | 0.052869718471 | -0.003086266 |
| MRR | 0.174358403798 | 0.190090491025 | -0.015732087 |
| Weighted Recall | 0.065288123567 | 0.072481155004 | -0.007193031 |

The primary Weighted Recall change was approximately `-9.92%` relative to the fixed control. Paired query counts were Recall `1/96/10`, nDCG `6/84/17`, MRR `6/85/16`, and Weighted Recall `1/96/10` for win/tie/loss.

Paired uncertainty used 20,000 bootstrap samples, 100,000 sign-flip samples, and seed 42:

| Metric | Delta | 95% CI | Sign-flip p |
|---|---:|---:|---:|
| Recall | -0.010188332 | [-0.017938938, -0.003733736] | 0.00294 |
| nDCG | -0.003086266 | [-0.008108557, +0.002179487] | 0.25937 |
| MRR | -0.015732087 | [-0.037190884, +0.004064864] | 0.14364 |
| Weighted Recall | -0.007193031 | [-0.013734461, -0.001972219] | 0.00896 |

Observed:

- Three folds selected the frozen weight 0.5; two selected 0.4.
- Both held-out folds using 0.4 performed worse on primary Weighted Recall@10: fold 0 delta `-0.016190430` with `1/17/4` win/tie/loss, and fold 4 delta `-0.019688805` with `0/15/6`.
- The OOF selected policy was below the fixed control on all four aggregate closed-world metrics at K=10.
- Recall and Weighted Recall decreases had 95% intervals excluding zero and sign-flip `p < 0.05`; nDCG and MRR decreases were not statistically established under the same analysis.

Interpretation:

The predeclared static fusion tuning procedure did not generalize better than the frozen 0.5/0.5 policy. The training-fold preference for `w_bm25=0.4` in two folds did not transfer successfully to their held-out queries.

Hypothesis:

Weight preference may be heterogeneous or noisy across query subsets. This can motivate investigation of query-dependent fusion, but EXP-03 does not establish that a dynamic router will work. It also does not establish that 0.5 is globally optimal, that 0.4 is intrinsically bad, or that training overfitting is the proven causal mechanism.

Decision:

- Do not promote the CV-selected static policy.
- Keep frozen Static Hybrid at `w_bm25=0.5`, `w_dense=0.5`.
- Keep `baseline_d0_v2`, runtime, and runtime artifacts unchanged.
- Do not create a new static-weight baseline.
- The subsequent oracle/headroom diagnostic is recorded in Section 6C.

6C. Offline Oracle/Headroom Diagnostic

Status: **COMPLETE — DIAGNOSTIC ONLY**

This was an offline oracle/headroom diagnostic using ground truth, not a deployable experiment result or held-out policy evaluation.

Input was the frozen candidate cache from official EXP-03 run `exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23`, fingerprint `4e94fc54bf2dc1ec09c4fbb605daadb1ba0dfa8c7c0c10a2d40ea34e90c2c805`. It used all 107 queries, complete BM25 and Dense top-30 pools, the existing min-max `fuse_pools()` behavior, and the predeclared grid `w_bm25 = 0.0, 0.1, ..., 1.0`.

For each query, the oracle maximized closed-world Weighted Recall@10. Exact ties selected the representative weight closest to 0.5 and then the smaller BM25 weight; the complete tied-best set was retained for plateau analysis. Fixed-0.5 parity reproduced the EXP-03 control with maximum metric error `0.0`.

Primary closed-world @10:

| Metric | Fixed 0.5 | Per-query WR@10 oracle | Delta |
|---|---:|---:|---:|
| Recall | 0.088253830495 | 0.110923 | +0.022670 |
| nDCG | 0.052869718471 | 0.061993 | +0.009123 |
| MRR | 0.190090491025 | 0.196295 | +0.006205 |
| Weighted Recall | 0.072481155004 | 0.090051 | +0.017569792095 |

The primary WR@10 ceiling was `+24.24%` relative to fixed 0.5. This is intentionally optimistic because each query's ground truth selected its own ranking.

Headroom distribution:

| Diagnostic | Result |
|---|---:|
| Oracle better / tied / worse | 23 / 84 / 0 queries |
| Mean / median delta | 0.017569792095 / 0 |
| p75 / p90 / maximum | 0 / 0.070431893688 / 0.2 |
| Gain >= 0.05 / 0.10 / 0.20 | 18 / 6 / 1 queries |
| Top 1 / 5 / 10 share of headroom | 10.64% / 40.27% / 62.69% |

Plateau analysis:

- 84 best-weight sets contain 0.5.
- 8 lie entirely below 0.5; 14 lie entirely above 0.5.
- 72 span both sides of 0.5.
- 46 queries are flat across all 11 weights; only 2 have a unique optimum.

Representative weights were `0.1:1`, `0.3:2`, `0.4:6`, `0.5:84`, `0.6:5`, `0.7:3`, `0.8:4`, and `1.0:2`, with zero representatives at 0.0, 0.2, and 0.9. Grouped counts were 9 Dense-heavy, 84 balanced, and 14 BM25-heavy. These representatives must not be interpreted as sharp preferences or semantic query categories because most surfaces are broad.

The descriptive all-query WR@10 surface was:

| `w_bm25` | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WR@10 | 0.059063 | 0.059613 | 0.063450 | 0.066244 | 0.070223 | 0.072481 | 0.068713 | 0.065607 | 0.063992 | 0.060863 | 0.052248 |

On this tested grid/data, 0.5 is the descriptive all-query maximum. This is not a claim of global mathematical optimality.

Observed:

- The oracle ceiling is materially above fixed 0.5, but only 23/107 queries improve.
- Median and p75 gain are zero, and the top ten queries contribute 62.69% of total headroom.
- Most weight-response surfaces are broad or completely flat.
- Nevertheless, 22 queries have best sets exclusively on one side of 0.5, showing some genuine directional preference.

Interpretation:

Query-dependent fusion has real but sparse and moderately concentrated theoretical headroom.

Hypothesis:

Some inference-time query or retrieval signals may identify the minority of queries for which departing from 0.5 is useful. This has not been established.

Decision:

- Classify the evidence as **MODERATE HEADROOM**.
- Dynamic Hybrid was sufficiently motivated to proceed to the leakage-controlled EXP-04 experiment, but not for runtime adoption.
- Prefer an initially selective policy that stays at 0.5 unless routing confidence is high.

Methodological caveat:

The oracle used ground truth from all 107 benchmark queries and influenced selection of the next policy family. The subsequent EXP-04 result on these same 107 queries, even with nested CV, is therefore development/exploratory evidence rather than a pristine untouched confirmatory evaluation. Nested CV remained necessary to prevent direct router-training leakage. Strong confirmatory claims require an additional untouched benchmark/test set.

6D. EXP-04 — Selective Dynamic Hybrid Nested CV

Status: **COMPLETE — DO NOT PROMOTE**

Protocol: `exp04_selective_dynamic_hybrid_nested_5fold_cv_v1`

Official run: `exp04_selective_dynamic_hybrid_nested_5fold_cv_v1_20261016_071606_287853_8d0649b7`

Primary endpoint: closed-world Weighted Recall@10.

The leakage-controlled nested evaluation trained a selective directional LogisticRegression router using only the frozen 18-dimensional inference-time feature contract. Fixed 0.5 used the same frozen candidate cache and retained exact parity with the EXP-03 control: maximum metric error `0.0`.

Closed-world results @10:

| Metric | Dynamic OOF | Fixed 0.5 |
|---|---:|---:|
| Recall | 0.08569671730749247 | 0.08825383049544679 |
| nDCG | 0.05436130327518431 | 0.05286971847087022 |
| MRR | 0.1891410769915442 | 0.1900904910250704 |
| Weighted Recall | 0.07053616402199708 | 0.07248115500412532 |

Routing outcomes:

| Outcome | Count |
|---|---:|
| `route_count` | 16 |
| `stay_count` | 91 |
| `dense_override` | 6 |
| `bm25_override` | 10 |
| `beneficial` | 0 |
| `tied` | 13 |
| `harmful` | 3 |

Total positive WR@10 gain was `0`; total negative harm was `0.2081140350877193`. The harmful queries were:

| Query | WR@10 delta |
|---:|---:|
| 6 | -0.0625 |
| 46 | -0.07894736842105263 |
| 70 | -0.06666666666666667 |

Outer-fold mean WR@10 deltas were:

| Fold | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| Mean delta | 0 | 0 | 0 | -0.0037593984962406013 | -0.00615079365079365 |

No outer fold had a positive mean primary delta (`0/5`).

Success contract:

| Condition | Result |
|---|---|
| `dynamic_wr10_above_fixed` | false |
| `bootstrap_lower_above_zero` | false |
| `sign_flip_p_below_0_05` | false |
| `nonzero_routing` | true |
| `benefit_exceeds_harm` | false |
| `positive_after_removing_largest_gain` | false |
| `positive_in_at_least_three_folds` | false |
| `no_serious_secondary_collapse` | true |
| `all_conditions_hold` | false |

Official decision: `DO_NOT_PROMOTE`.

Observed:

- Dynamic Hybrid routed 16/107 queries.
- No override improved primary Weighted Recall@10; 13 tied fixed 0.5 and 3 were harmful.
- Dynamic OOF Weighted Recall@10 was below fixed 0.5.
- No outer fold had a positive mean Weighted Recall@10 delta.
- Fixed-control parity remained exact.

Interpretation:

The selective directional LogisticRegression router did not exploit the oracle headroom identified earlier.

Hypothesis:

The theoretical query-dependent fusion headroom exists, but the current 18 inference-time features and simple selective routing model do not predict useful overrides reliably.

This result does not establish that all Dynamic Hybrid methods cannot work, that the oracle headroom was false, or that LogisticRegression is universally unsuitable.

Decision:

- EXP-04 is complete.
- Do not promote Dynamic Hybrid.
- Keep static Hybrid at `w_bm25=0.5`, `w_dense=0.5`.
- Keep runtime, `baseline_d0_v2`, and runtime artifacts unchanged.
- Do not immediately retune the router on the same 107 queries.

Methodological note:

Repeated router redesign using the same 107-query benchmark after observing EXP-04 outcomes would increase development-set overfitting risk. Any renewed router work should be motivated and evaluated with additional untouched data rather than iterative same-benchmark tuning.

7. Runtime Architecture
Current runtime path:
React frontend
    ↓
FastAPI
    ├── POST /search
    ├── POST /ask
    ├── POST /ocr
    ├── POST /ai/export-docx
    ├── GET /health/live
    ├── GET /health/ready
    └── GET /health
          ↓
SearchService
    ├── manifest validation
    ├── artifact SHA validation
    ├── BM25(D0)
    ├── Dense(D0)
    ├── min-max fusion
    └── deterministic ranking
          ↓
runtime_law_metadata.json
runtime_law_index.faiss
runtime_retrieval_manifest.json

The runtime does not use a relational database or vector server.
Storage remains local:
JSON
FAISS
PDF
process-local state

8. Runtime Retrieval Contract
Runtime artifacts
Active generated artifacts:
backend/data/runtime_law_index.faiss
backend/data/runtime_law_metadata.json
backend/data/runtime_retrieval_manifest.json

These artifacts are generated and intentionally ignored by Git.
The source code required to regenerate and verify them is version-controlled.
Index invariants
Runtime startup verifies:
FAISS type       IndexFlatIP
dimension        768
vectors          1811
metadata rows    1811
unique IDs       1811

It also verifies:
runtime contract ID
baseline protocol ID
corpus fingerprint
embedding model
exact embedding revision
representation
BM25 tokenizer
fusion method
fusion weights
artifact SHA256
artifact filenames

A contract mismatch causes initialization failure instead of silently running an incompatible retriever.
9. Runtime Embedding
Active model:
keepitreal/vietnamese-sbert

Exact revision:
a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23

Dimension:
768

Runtime query encoding no longer truncates the query to 800 characters.
Q0 normalization is performed before retrieval.
The encoded query is converted to float32 and L2-normalized before IndexFlatIP search.
The old runtime configuration:
BAAI/bge-m3
1024-D

is superseded and is not part of the active retrieval path.
10. Runtime BM25
Runtime BM25 uses the same D0 text as Dense retrieval.
Tokenizer:
text.lower().split()


This is the frozen deterministic whitespace tokenizer.
The old runtime BM25 implementation used regex tokenization over:
title + content + keywords

That implementation has been superseded.
11. Runtime Hybrid Retrieval
Runtime retrieval flow:
Q0
 ├── BM25 top-30
 └── Dense top-30
        ↓
union by standard_law_id
        ↓
min-max normalization
        ↓
0.5 * BM25 + 0.5 * Dense
        ↓
score DESC
standard_law_id ASC on exact ties

No CrossEncoder is part of runtime_retrieval_v1.
The previous runtime fallback:
0.6 Dense + 0.4 BM25

and its undefined top_k behavior are no longer active.
12. API State
POST /search
This is the canonical retrieval endpoint for the frozen runtime contract.
Input:
query
top_k

Valid range:
1 <= top_k <= 20

Flow:
raw query
  ↓
Q0 normalization
  ↓
BM25 + Dense
  ↓
static min-max Hybrid
  ↓
SearchResult[]

The endpoint has been verified against the historical golden Hybrid top-20 ranking.
POST /ask
Retrieval uses the same raw Q0 path as /search.
LLM query expansion / Gemini intent extraction has been deliberately removed from runtime_retrieval_v1.
Current structure:
raw user query
   ↓
frozen retrieval
   ↓
retrieved statutes
   ↓
Gemini answer generation

Gemini is therefore downstream of retrieval.
/ask can still make Gemini calls for:
answer generation
answer evaluation

It should not be treated as a zero-cost local endpoint.
Health
Endpoints:
GET /health/live
GET /health/ready
GET /health

/health/live confirms that the process is alive.
/health/ready exposes retrieval readiness and contract information.
/health currently delegates to readiness for compatibility.
The FastAPI lifespan performs fail-fast retrieval initialization.
13. Runtime Verification
Artifact-level verification:
corpus count       1811
metadata count     1811
unique IDs         1811
FAISS d             768
FAISS ntotal       1811
FAISS type         IndexFlatIP

Offline golden parity:
107 queries
BM25 top-20          exact ordering match
FAISS top-20         exact ordering match
Hybrid top-20        exact ordering match

321 / 321 retrieval checks passed

Small floating-point score differences were observed for Dense/Hybrid, approximately 1e-6, while ranking order remained identical.
FastAPI service-level verification:
SearchService
  -> golden Hybrid top-20 exact ranking match

HTTP integration verification:
/health/live    200
/health/ready   200
/health         200
/search         200

/search top-20
  = frozen golden Hybrid top-20

Runtime and baseline targeted test checkpoint:
66 passed

Existing local retrieval smoke:
backend/scripts/Tests/test_search.py
exit = 0

14. Runtime Artifact Recovery Tooling
Version-controlled tooling:
backend/scripts/Runtime/
├── __init__.py
├── build_retrieval_artifacts.py
└── verify_golden_parity.py

Responsibilities:
build_retrieval_artifacts.py
Builds the runtime artifacts from the frozen corpus while reusing evaluator primitives.
It validates:
required corpus files
per-law counts
total 1811
corpus fingerprint
Q0/D0 protocol
exact model revision
embedding dimension
FAISS type/count
unique canonical IDs

verify_golden_parity.py
Loads newly generated runtime artifacts and compares their retrieval rankings against the historical golden run.
It checks:
BM25
FAISS
Hybrid_Static_Fixed

for selected queries or the full 107-query benchmark.
15. Relevant Git Checkpoints
Historical exact golden baseline provenance remains preserved separately.
Recent repository recovery checkpoints:
64db4f0b
feat(runtime): add deterministic retrieval artifact recovery

886a0783
fix(runtime): recover deterministic baseline retrieval

The old historical exact golden commit remains important for scientific provenance, even though repository history was sanitized to remove an oversized installer.
Do not equate a rewritten/sanitized commit hash with the historical exact golden commit.
16. Offline Statute Pipeline
The generic statute pipeline remains:
01_extract_laws.py
    ↓
02_transform_laws.py
    ↓
03_load_vectors.py

Important distinction:
03_load_vectors.py is a legacy/general vector builder and does not define the frozen runtime retrieval contract.
Do not use it to overwrite the recovered runtime artifacts.
Runtime artifacts should instead be generated through:
backend/scripts/Runtime/build_retrieval_artifacts.py

Cost characteristics:
01_extract_laws.py
  no Gemini

02_transform_laws.py
  ⚠ Gemini

03_load_vectors.py
  no Gemini
  but can overwrite vector artifacts

Runtime/build_retrieval_artifacts.py
  no Gemini

17. Judgment and Benchmark Pipeline
Judgment PDFs are benchmark/query sources, not runtime retrieval documents.
The retrieval corpus contains statutory articles.
Two judgment-processing paths exist:
Full structural OCR
ocrFullBanAn.py

Produces page-aware judgment JSON and section segmentation.
Benchmark generation
Evaluation/process_judgments.py

Uses OCR + Gemini for benchmark generation.
The benchmark has already been migrated to GT Schema V2.
The current generator still contains older behavior that can be unsafe for the frozen V2 benchmark.
Therefore:
DO NOT run run_all_batches.sh
DO NOT casually rewrite judgments_evaluation.json
DO NOT run migration/cleanup scripts on the frozen benchmark

without a deliberate benchmark-version change.
18. Human Annotation State
The current human annotation experiment covers only:
3 / 107 queries

Stored Cohen's Kappa:
0.3438

This is evidence of an annotation pilot only.
The benchmark must not be described as fully human-validated.
Future audit categories should distinguish:
TRUE_MISSING_LAW
ALIAS / CANONICALIZATION
BAD_AI_LABEL
OUT_OF_SCOPE

Human adjudication, not the same AI that produced the original labels, should make final label decisions.
19. Runtime vs Experimental Evaluation
This distinction is mandatory.
Runtime
Runtime currently implements only the frozen retrieval contract:
Q0
D0
BM25
Dense
Static Hybrid

Experimental evaluation
Research code also contains or explores:
D1
static tuning
dynamic hybrid
temporal filtering
routing
reranking
error analysis
statistical analysis

Experimental functionality must not be described as production runtime functionality unless explicitly integrated and tested.
20. Known Remaining Issues
Runtime retrieval is no longer the primary blocker.
Remaining issues include:
1. /ask answer evaluation is not yet established as a scientifically valid answer metric.
2. Gemini answer generation remains an external dependency.
3. OCR upload validation/security remains weak.
4. There is no authentication/authorization layer.
5. CORS remains permissive.
6. Local file/FAISS storage has no multi-node consistency model.
7. Benchmark labels remain predominantly AI-generated.
8. Human adjudication coverage is very small.
9. Judgment benchmark generation and frozen Schema V2 are not yet safely unified.
10. Experimental advanced retrieval code remains separate from runtime.
11. Generated runtime artifacts are intentionally ignored and must be regenerated locally.
12. Project documentation is being updated after runtime recovery.
21. Current Priority
The runtime recovery engineering blocker has been resolved.
Current order of work:
1. Record and preserve the completed EXP-04 decision
2. Preserve runtime recovery state, `baseline_d0_v2`, and fixed 0.5/0.5 fusion
3. Do not immediately redesign or retune the router on the same 107 queries
4. Obtain an untouched benchmark/test set before renewed Dynamic Hybrid work or strong confirmatory claims
5. Continue temporal/reranking experiments and benchmark audit only under controlled protocols

No new runtime retrieval feature should be added merely because it exists in the experimental evaluator.
Research modifications should first be evaluated as controlled experiments.
22. Next Research Action — Preserve EXP-04 and Avoid Same-Benchmark Retuning

EXP-04 has answered its predeclared exploratory question: the selective directional router did not improve over the frozen 0.5/0.5 D0 control.

Do not immediately redesign the router using the same 107-query outcomes. Preserve the official run and decision, keep fixed 0.5/0.5 as the runtime and research baseline, and obtain additional untouched evaluation data before renewed Dynamic Hybrid development.

Controlled temporal retrieval, reranking, and benchmark audit may continue as separate research tracks without changing runtime.
23. Source-of-Truth Entry Points
Runtime:
backend/app/config.py
backend/app/main.py
backend/app/models/schema.py
backend/app/services/embedding_service.py
backend/app/services/search_service.py

Runtime artifact construction:
backend/scripts/Runtime/build_retrieval_artifacts.py
backend/scripts/Runtime/verify_golden_parity.py

Frozen evaluation:
backend/scripts/Evaluation/configs/baseline_d0.yaml
backend/scripts/Evaluation/configs/exp02_d0.yaml
backend/scripts/Evaluation/configs/exp02_d1.yaml
backend/scripts/Evaluation/configs/exp03_static_fusion_cv.yaml
backend/scripts/Evaluation/configs/exp04_selective_dynamic_hybrid_nested_cv.yaml
backend/scripts/Evaluation/experiments/baseline.py
backend/scripts/Evaluation/experiments/exp03_static_fusion_cv.py
backend/scripts/Evaluation/experiments/exp04_selective_dynamic_hybrid_nested_cv.py
backend/scripts/Evaluation/core/
backend/scripts/Evaluation/retrieval/

Runtime regression tests:
backend/scripts/Tests/test_runtime_retrieval.py
backend/scripts/Tests/test_baseline_protocol.py
backend/scripts/Tests/test_exp02_protocol.py
backend/scripts/Tests/test_exp03_protocol.py
backend/scripts/Tests/test_exp04_protocol.py

Benchmark:
backend/data/judgments_evaluation.json
backend/data/gt_schema_v2.json

24. Current State Matrix
Area	Status	Current evidence
FastAPI API	IMPLEMENTED / PARTIAL PRODUCT	/search, /ask, OCR, export, health endpoints
Runtime retrieval	RECOVERED	runtime_retrieval_v1
Runtime corpus	FROZEN	1,811 articles
Dense retrieval	RECOVERED	Vietnamese-SBERT, exact revision, 768-D
BM25	RECOVERED	D0 + whitespace
Static Hybrid	RECOVERED	min-max 0.5/0.5
Determinism	VERIFIED	canonical tie-break
Runtime artifacts	VERIFIED	count/dim/SHA/manifest checks
Golden parity	VERIFIED	321/321
HTTP /search	VERIFIED	golden top-20 parity
Reranking	EXPERIMENTAL	not in baseline runtime
Temporal retrieval	EXPERIMENTAL	evaluator only
Dynamic routing	EXPERIMENT COMPLETE / NOT PROMOTED	EXP-04 selective router did not improve primary WR@10
Benchmark	FROZEN FOR CURRENT EXPERIMENTS	107 queries
Human validation	INCOMPLETE	3-query pilot
OCR	IMPLEMENTED / PARTIAL	validation/security gaps
Answer generation	PARTIAL	Gemini dependent
Authentication	NOT IMPLEMENTED	no auth layer
Database/vector server	NOT USED	local JSON + FAISS
EXP-02	COMPLETE	D1 not promoted; baseline/runtime remain D0
EXP-03	COMPLETE	CV-selected static policy not promoted; fixed 0.5/0.5 retained
Oracle/headroom	COMPLETE / DIAGNOSTIC	Moderate, sparse and concentrated upper-bound headroom
EXP-04	COMPLETE / NOT PROMOTED	Dynamic OOF WR@10 below fixed 0.5; 0 beneficial and 3 harmful overrides
Next research step	ADDITIONAL DATA REQUIRED	Avoid immediate router retuning on the same 107 queries
