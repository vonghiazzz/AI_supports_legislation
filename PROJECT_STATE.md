# AI Supports Legislation — Current Project State

Last updated: 2026-10-13  
Source-of-truth rule: `CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION`

This document describes the current repository state after repository hygiene, frozen-baseline recovery, deterministic artifact reconstruction, and FastAPI runtime retrieval recovery.

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
Static hybrid tuning with proper dev/test discipline
    ↓
Dynamic hybrid
    ↓
Temporal retrieval
    ↓
Reranking
    ↓
Statistical analysis

The immediate research experiment is EXP-02: D0 vs D1.
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
It is the next controlled research experiment.
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
1. Documentation checkpoint
2. Preserve runtime recovery state
3. EXP-02: D0 vs D1
4. Establish dev/test discipline before tuning
5. Static hybrid tuning
6. Dynamic / temporal / reranking experiments
7. Statistical testing

No new runtime retrieval feature should be added merely because it exists in the experimental evaluator.
Research modifications should first be evaluated as controlled experiments.
22. Next Research Experiment — EXP-02
Research question:
Does D1 enrichment improve Vietnamese statutory retrieval compared with the controlled D0 baseline?

Control requirements:
same corpus
same 107 benchmark queries
same Q0
same embedding model/revision
same BM25 tokenizer
same FAISS implementation
same fusion
same metric protocol
same deterministic tie policy

Only the document representation changes:
D0
vs
D1

The comparison must report at least:
Recall@5/10/20
nDCG@5/10/20
MRR@5/10/20
Weighted Recall@5/10/20

Primary endpoint remains:
closed-world @10

Do not tune D1 and D0 differently during this experiment.
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
backend/scripts/Evaluation/experiments/baseline.py
backend/scripts/Evaluation/core/
backend/scripts/Evaluation/retrieval/

Runtime regression tests:
backend/scripts/Tests/test_runtime_retrieval.py
backend/scripts/Tests/test_baseline_protocol.py

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
Dynamic routing	EXPERIMENTAL	evaluator only
Benchmark	FROZEN FOR CURRENT EXPERIMENTS	107 queries
Human validation	INCOMPLETE	3-query pilot
OCR	IMPLEMENTED / PARTIAL	validation/security gaps
Answer generation	PARTIAL	Gemini dependent
Authentication	NOT IMPLEMENTED	no auth layer
Database/vector server	NOT USED	local JSON + FAISS
Next research step	READY	EXP-02 D0 vs D1