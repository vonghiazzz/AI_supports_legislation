# AI Supports Legislation — Codebase Map

Last updated: 2026-10-16

This document maps current implementation to files/functions and explicitly separates:

- **RUNTIME**
- **FROZEN BASELINE**
- **EXPERIMENTAL EVALUATION**
- **OFFLINE DATA PIPELINES**
- **PLANNED ONLY**

Source-of-truth rule:

`CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION`

Do not infer runtime behavior merely because similar logic exists in the evaluator.

---

# 1. Runtime API

| Feature | File | Class/function | Responsibility / actual call path | State |
|---|---|---|---|---|
| FastAPI application | `backend/app/main.py` | `app` | Owns middleware, routes and lifespan | IMPLEMENTED |
| Runtime startup | `backend/app/main.py` | `lifespan()` | Calls `search_service.ensure_ready()` before serving requests | IMPLEMENTED / FAIL-FAST |
| Request timing | `backend/app/main.py` | `add_process_time_header()` | Adds `X-Process-Time` and logs request duration | IMPLEMENTED |
| Search API | `backend/app/main.py` | `search_api()` | `SearchRequest` -> `SearchService.search(query, top_k)` | IMPLEMENTED |
| Legal Q&A API | `backend/app/main.py` | `ask_api()` | greeting fast path or Q0 retrieval -> Gemini answer generation -> optional answer evaluation | IMPLEMENTED / GEMINI DEPENDENT |
| OCR API | `backend/app/main.py` | `ocr_api()` | Upload -> temp file -> `OCRService.extract_text()` -> cleanup | IMPLEMENTED / VALIDATION GAPS |
| DOCX export | `backend/app/main.py` | `export_docx()` | Calls `build_legal_analysis_docx()` and streams DOCX bytes directly | IMPLEMENTED |
| Liveness | `backend/app/main.py` | `health_live()` | Reports process alive | IMPLEMENTED |
| Readiness | `backend/app/main.py` | `health_ready()` | Requires runtime retrieval contract to be valid and loaded | IMPLEMENTED |
| Compatibility health | `backend/app/main.py` | `health_check()` | Delegates to readiness | IMPLEMENTED |
| API DTOs | `backend/app/models/schema.py` | Pydantic models | Request/response validation | IMPLEMENTED |

---

# 2. Runtime Request / Response Models

File:

```text
backend/app/models/schema.py

Model	Responsibility	Important contract
SearchRequest	/search input	query, top_k; 1 <= top_k <= 20
SearchResult	Retrieval source/result	Includes standard_law_id, law_id, content fields and retrieval scores
AskRequest	/ask input	query, top_k; 1 <= top_k <= 20
AskResponse	/ask output	Query, answer, sources, LLM status, optional evaluation
OCRResponse	/ocr output	Filename + extracted text


Canonical retrieval identity:
standard_law_id

must be preferred over law_id when uniquely identifying an article across laws.
3. Runtime Retrieval Contract
Active contract:
runtime_retrieval_v1

Research protocol represented by runtime:
baseline_d0_v2

Frozen runtime configuration:
Corpus                  1,811 articles
Query representation    Q0
Document representation D0
BM25 tokenizer          whitespace
Dense model             keepitreal/vietnamese-sbert
Dense revision          a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23
Embedding dimension     768
Normalization           L2
FAISS                   IndexFlatIP
Candidate pool          30
Fusion                  min-max
BM25 weight             0.5
Dense weight            0.5
Tie-break               score DESC -> standard_law_id ASC
Reranker                none
API top_k               1..20

4. Runtime Configuration
File:
backend/app/config.py

Class:
Settings

Active runtime retrieval settings
Setting	Responsibility
RUNTIME_INDEX_PATH	Frozen runtime FAISS artifact
RUNTIME_METADATA_PATH	Frozen runtime metadata artifact
RUNTIME_MANIFEST_PATH	Runtime artifact/contract manifest
RUNTIME_CONTRACT_ID	runtime_retrieval_v1
RUNTIME_PROTOCOL_ID	baseline_d0_v2
EXPECTED_CORPUS_COUNT	1811
EXPECTED_EMBEDDING_DIMENSION	768
EXPECTED_CORPUS_FINGERPRINT	Frozen five-law corpus fingerprint
EMBEDDING_MODEL	keepitreal/vietnamese-sbert
EMBEDDING_REVISION	Exact pinned model revision
EMBEDDING_LOCAL_FILES_ONLY	Prevents implicit model drift/download during runtime recovery
RUNTIME_POOL_K	30
RUNTIME_FUSION_METHOD	minmax
RUNTIME_STATIC_W_BM25	0.5


Legacy settings retained
The following names may still exist for legacy/offline scripts:
DATA_PATH
INDEX_PATH
METADATA_PATH

They are not the active FastAPI retrieval artifact contract.
Do not infer active runtime behavior from those legacy settings.
5. Runtime Embedding Service
File:
backend/app/services/embedding_service.py

Function	Responsibility	State
EmbeddingService._get_model()	Lazy-load exact Vietnamese-SBERT revision on CPU	IMPLEMENTED
EmbeddingService.ensure_loaded()	Explicit initialization used before FAISS load	IMPLEMENTED
EmbeddingService.encode()	Encodes full normalized Q0 query to float32 vector	IMPLEMENTED
resolved_revision	Exposes verified model revision	IMPLEMENTED


Important behavior:
model      keepitreal/vietnamese-sbert
revision   a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23
dimension  768

The service no longer:
uses BAAI/bge-m3
truncates query to 800 characters
implicitly changes model revision

L2 normalization for retrieval is applied in the retrieval path using FAISS.
6. Runtime Search Service
File:
backend/app/services/search_service.py

Class:
SearchService

Initialization / validation
Function	Responsibility
_sha256_file()	Computes generated artifact SHA256
_load_json()	Loads manifest/metadata
_validate_manifest_contract()	Enforces frozen runtime contract
_validate_artifact_hashes()	Checks index and metadata hashes
_validate_metadata()	Checks count, canonical IDs and D0 field consistency
ensure_ready()	Loads and validates manifest, metadata, BM25, embedding model and FAISS
status()	Returns readiness/contract information


ensure_ready() verifies at least:
runtime contract ID
protocol ID
Q0
D0
corpus fingerprint
corpus count
metadata count
embedding model
requested revision
resolved revision
embedding dimension
embedding normalization
FAISS type
FAISS ntotal
BM25 tokenizer
fusion method
fusion weights
tie policy
artifact filenames
artifact SHA256

A mismatch causes initialization failure.
BM25
Runtime reuses modular evaluation primitives:
backend/scripts/Evaluation/retrieval/bm25.py

Imported functions:
build_bm25_index()
search_bm25()

Document field:
metadata["bm25_text"]

Tokenizer:
text.lower().split()


Candidate pool:
30

The old runtime regex-tokenized BM25 over:
title + content + keywords

is superseded.
Query normalization
Runtime reuses:
backend/scripts/Evaluation/core/query.py

Function:
normalize_query_q0()

Q0 behavior:
strip
collapse whitespace
no query expansion
no LLM rewrite
no intent routing

Dense retrieval
Method:
SearchService._search_dense_pool()

Flow:
Q0 query
  -> EmbeddingService.encode()
  -> float32
  -> L2 normalize
  -> IndexFlatIP.search()
  -> standard_law_id + raw dense score

Candidate pool:
30

Static Hybrid fusion
Runtime reuses:
backend/scripts/Evaluation/retrieval/fusion.py

Function:
fuse_pools()

Contract:
BM25 pool
   +
Dense pool
   ↓
min-max normalization
   ↓
0.5 BM25 + 0.5 Dense
   ↓
score DESC
standard_law_id ASC on ties

No CrossEncoder is part of runtime_retrieval_v1.
Canonical search method
Method:
SearchService.search()

Actual runtime flow:
ensure_ready()
    ↓
normalize_query_q0()
    ↓
BM25 top-30
Dense top-30
    ↓
fuse_pools()
    ↓
format SearchResult-compatible objects
    ↓
top_k

This is the canonical retrieval method used by:
POST /search
POST /ask

Compatibility wrapper
Method:
SearchService.search_hybrid()

State:
BACKWARD-COMPATIBILITY WRAPPER

It does not implement the old multi-query/CrossEncoder pipeline.
It selects the first supplied query and forwards to:
SearchService.search()

Expanded query lists and keyword dictionaries do not alter runtime_retrieval_v1.
7. Runtime Retrieval Artifacts
Active generated artifacts:
backend/data/runtime_law_index.faiss
backend/data/runtime_law_metadata.json
backend/data/runtime_retrieval_manifest.json

These files are intentionally ignored by Git.
Artifact	Producer	Consumer	Current contract
runtime_law_index.faiss	Runtime artifact builder	SearchService.ensure_ready()	IndexFlatIP, d=768, ntotal=1811
runtime_law_metadata.json	Runtime artifact builder	BM25 + result mapping	1811 rows, canonical IDs, D0 text
runtime_retrieval_manifest.json	Runtime artifact builder / promotion step	Runtime startup validation	Model/corpus/artifact contract


Old artifacts:
backend/data/law_index.faiss
backend/data/law_metadata.json

are not active runtime artifacts.
Observed historical state:
law_index.faiss     1813 vectors
law_metadata.json   missing

They are retained only as superseded/legacy artifacts.
8. Runtime Artifact Build / Verification Tooling
Directory:
backend/scripts/Runtime/

build_retrieval_artifacts.py
Responsibility:
frozen corpus
  -> D0 metadata
  -> exact Vietnamese-SBERT embeddings
  -> normalized IndexFlatIP
  -> runtime metadata
  -> manifest

Validates:
five required laws
per-law article counts
total 1811
unique standard_law_id
corpus fingerprint
Q0/D0 protocol
exact embedding revision
dimension 768
IndexFlatIP
ntotal 1811

State:
IMPLEMENTED / VERSION-CONTROLLED

verify_golden_parity.py
Responsibility:
Compares generated runtime artifacts against historical frozen baseline retrieval outputs.
Modes:
selected sample queries
full 107-query verification

Retrievers checked:
BM25
FAISS
Hybrid_Static_Fixed

Verified full result:
107 queries
321 retrieval checks
0 failures

State:
IMPLEMENTED / VERSION-CONTROLLED

9. Runtime Retrieval Tests
backend/scripts/Tests/test_runtime_retrieval.py
Covers:
valid runtime manifest
contract drift rejection
wrong model rejection
wrong dimension rejection
wrong corpus/count rejection
wrong representation rejection
wrong FAISS type rejection
wrong BM25 tokenizer rejection
wrong fusion rejection
wrong tie policy rejection
D0 document fields
SearchRequest top_k bounds
AskRequest top_k bounds
canonical standard_law_id response field
search_hybrid compatibility behavior

State:
AUTOMATED PYTEST

backend/scripts/Tests/test_baseline_protocol.py
Covers frozen research protocol invariants.
Recent combined targeted checkpoint:
66 passed

for:
test_runtime_retrieval.py
+
test_baseline_protocol.py

backend/scripts/Tests/test_search.py
Local retrieval smoke script.
Calls:
search_service.search()

No Gemini required.
State:
PRINT-BASED LOCAL SMOKE

10. /search Runtime Path
Route:
POST /search

File:
backend/app/main.py

Call path:
SearchRequest
    ↓
search_api()
    ↓
SearchService.search()
    ↓
Q0
    ↓
BM25(D0) + Dense(D0)
    ↓
min-max 0.5/0.5
    ↓
SearchResult[]

Verified:
HTTP 200
golden Hybrid top-20 exact ranking match

State:
RECOVERED / VERIFIED

11. /ask Runtime Path
Route:
POST /ask

Current retrieval flow:
AskRequest
    ↓
greeting fast path?
    ↓ no
SearchService.search(raw query)
    ↓
retrieved statutes
    ↓
LLMService.ask_with_context()
    ↓
optional LLMService.evaluate_answer()
    ↓
AskResponse

Important:
Gemini intent extraction is NOT part of runtime retrieval.

expanded_queries currently represents the raw query only.
keywords currently remain empty response metadata.
Retrieval therefore stays aligned with Q0 baseline.
State:
RETRIEVAL VERIFIED
ANSWER GENERATION GEMINI-DEPENDENT
ANSWER EVALUATION NOT SCIENTIFICALLY ESTABLISHED

12. Runtime LLM Service
File:
backend/app/services/llm_service.py

This service is separate from the frozen retrieval contract.
Potential responsibilities present in this file include:
Gemini client initialization
answer prompting
answer generation
context handling
caching
legacy intent helpers
heuristic answer-context logic
answer evaluation

Important distinction:
Functions existing in LLMService are not automatically active retrieval components.
For example:
extract_legal_intent()

may still exist as code but is no longer called by main.py for the frozen runtime retrieval flow.
Likewise, any LLM-side document reordering must not be described as part of SearchService's deterministic baseline ranking.
13. Runtime OCR
File:
backend/app/services/ocr_service.py

Class:
OCRService

Responsibilities:
image OCR
PDF rendering
Vietnamese Tesseract OCR
text extraction

Used by:
POST /ocr

State:
IMPLEMENTED / SECURITY AND VALIDATION GAPS

Known concerns remain around:
upload type validation
upload size
filename handling
OCR failure semantics
resource limits

14. DOCX Export
Service:
backend/app/services/docx_export_service.py

Route:
POST /ai/export-docx

Responsibility:
legal analysis text
   ↓
formatted DOCX
   ↓
BytesIO / StreamingResponse

Important current behavior:
backend streams the generated DOCX directly

No runtime retrieval behavior depends on DOCX export.
State:
IMPLEMENTED

15. Frontend
Primary directory:
frontend/src/

Typical responsibilities:
Area	File	Responsibility
Main application	frontend/src/App.*	Conversation/UI state
API client	frontend/src/api.*	Calls backend endpoints
Source display	frontend/src/components/SourceCard.*	Shows retrieved law source
Law modal	frontend/src/components/LawModal.*	Expanded article content
Message rendering	frontend/src/components/Message.*	Assistant/user content
DOCX helper	frontend/src/utils/exportDoc.*	Requests backend DOCX and downloads Blob


Exact frontend extension/path should be verified from the working tree when editing frontend code.
Frontend does not define retrieval correctness.
16. Frozen Baseline Evaluation
Primary protocol implementation:
backend/scripts/Evaluation/experiments/baseline.py

This is the authoritative modular frozen-baseline experiment path.
It imports modular primitives from:
backend/scripts/Evaluation/core/
backend/scripts/Evaluation/retrieval/
backend/scripts/Evaluation/reporting/

Baseline protocol:
baseline_d0_v2

State:
FROZEN / VERIFIED

17. Evaluation Core
Directory:
backend/scripts/Evaluation/core/

Responsibilities include:
corpus loading
representation building
query normalization
canonical IDs
fingerprints

Important runtime-shared primitive:
normalize_query_q0()

The evaluator remains the source of truth for frozen protocol semantics.
18. Evaluation Retrieval Modules
Directory:
backend/scripts/Evaluation/retrieval/

BM25
File:
bm25.py

Responsibilities:
build BM25 index
whitespace tokenization
deterministic BM25 search

Runtime reuses these primitives.
Dense
File:
dense.py

Responsibilities:
SentenceTransformer initialization
exact revision support
corpus embedding
L2 normalization
IndexFlatIP construction
dense retrieval

Used primarily by offline evaluation/artifact construction.
Runtime does not rebuild corpus embeddings on startup.
Fusion
File:
fusion.py

Responsibilities:
pool normalization
static fusion
deterministic ranking

Runtime reuses:
fuse_pools()

19. Legacy Monolithic Evaluator
File:
backend/scripts/Evaluation/run_rag_evaluation.py

This file contains extensive historical/experimental logic.
It may include:
old retrieval implementations
dynamic routing
temporal filtering
fusion tuning
reranking
diagnostics
reporting
statistical code

Important:
DO NOT copy retrieval logic from this file into runtime
without verifying which definitions are currently authoritative.

The frozen baseline path is modular:
experiments/baseline.py

not the historical monolithic implementation.
State:
EXPERIMENTAL / LEGACY-MIXED

20. Experimental Retrieval Features
The research evaluator contains or explores features that are not active runtime behavior.
Feature	State
D1 representation	EXP-02 COMPLETE / NOT PROMOTED
Static fusion tuning	EXP-03 COMPLETE / NOT PROMOTED
Dynamic hybrid weights	EXPERIMENTAL
Intent/profile routing	EXPERIMENTAL
Temporal filtering	EXPERIMENTAL
PhoRanker reranking	EXPERIMENTAL
Failure taxonomy	EXPERIMENTAL
Per-query oracle/headroom	NEXT READ-ONLY ANALYSIS
Statistical tests	EXPERIMENTAL
Paper/LaTeX reporting	EXPERIMENTAL


Do not map these to FastAPI until deliberate integration exists.

EXP-02 ownership:

| File | Responsibility |
|---|---|
| `backend/scripts/Evaluation/configs/exp02_d0.yaml` | Controlled D0 arm |
| `backend/scripts/Evaluation/configs/exp02_d1.yaml` | Controlled D1 arm |
| `backend/scripts/Evaluation/experiments/baseline.py` | Protocol validation, execution, provenance and artifacts |
| `backend/scripts/Tests/test_exp02_protocol.py` | EXP-02 arm/control/provenance contract tests |

The generated EXP-02 run artifacts remain research evidence under `backend/data/runs/`; they are not runtime artifacts.

EXP-03 ownership:

| File | Responsibility |
|---|---|
| `backend/scripts/Evaluation/configs/exp03_static_fusion_cv.yaml` | Frozen EXP-03 protocol, folds, weight grid, objective, and golden reference |
| `backend/scripts/Evaluation/experiments/exp03_static_fusion_cv.py` | Candidate-cache generation, golden parity, deterministic CV, OOF evaluation, provenance, and artifact assembly |
| `backend/scripts/Evaluation/run_experiment.py` | Explicit protocol-aware CLI dispatch |
| `backend/scripts/Evaluation/reporting/serialization.py` | Atomic EXP-03 artifact publication |
| `backend/scripts/Tests/test_exp03_protocol.py` | Protocol, leakage, parity, dispatch, and persistence regression tests |
| `backend/data/runs/exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23/` | Official EXP-03 evidence: candidate cache, folds, OOF results, summary, manifest, and config snapshot |

EXP-03 is complete and did not promote a new static policy. The evaluator and artifacts remain research-only; runtime continues to use frozen 0.5/0.5 Static Hybrid.

21. Statute Extraction Pipeline
Directory:
backend/scripts/Flow/

01_extract_laws.py
Responsibilities:
input discovery
PDF extraction
OCR fallback
DOC/DOCX/TXT extraction
text cleanup
article splitting
raw_<law>.json output

Gemini:
NO

State:
OFFLINE ETL

02_transform_laws.py
Responsibilities:
Gemini enrichment
title/summary/keyword/etc. generation
checkpoint/resume
standardized article output

Gemini:
YES

State:
OFFLINE / COST-BEARING

03_load_vectors.py
Responsibilities:
load standardized files
construct combined representation
embed with Vietnamese-SBERT
write legacy/general FAISS + metadata artifacts

Gemini:
NO

Important:
NOT the authoritative runtime artifact builder

Its document representation differs from frozen D0.
Do not use it to overwrite:
runtime_law_index.faiss
runtime_law_metadata.json

State:
GENERIC / LEGACY VECTOR BUILD PATH

run_pipeline.py
Orchestrates:
01
02
03

Running all steps may call Gemini through Step 02.
State:
OFFLINE ORCHESTRATOR

22. Frozen Statutory Corpus
Runtime/baseline corpus:
Key	Count
bo_luat_dan_su_2015	689
bo_luat_to_tung_dan_su_2015	517
luat_dat_dai_2013	212
luat_dat_dai_2024	260
luat_hon_nhan_gia_dinh_2014	133
Total	1811


Excluded:
test_law
mau_so_52

Combined fingerprint:
66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58

23. Judgment Full-Text OCR
File:
backend/scripts/ocrFullBanAn.py

Responsibilities include:
page extraction
per-page OCR fallback
text cleanup
section detection
header/facts/reasoning/decision segmentation
structured JSON writing

Gemini:
NO

Current artifacts:
107 structured judgment files

State:
OFFLINE / IMPLEMENTED

These structured artifacts are not the runtime retrieval corpus.
24. Benchmark Generation
File:
backend/scripts/Evaluation/process_judgments.py

Responsibilities include:
judgment OCR
Gemini query extraction
Gemini ground-truth generation
dataset generation/update
query cleanup

Gemini:
YES

Important:
The current frozen benchmark is already Schema V2.
The generator contains historical behavior that must not be assumed to reproduce the frozen V2 benchmark safely.
State:
OFFLINE / MUTATING / COST-BEARING

Do not run casually.
25. Benchmark Schema / Validation
Important files:
backend/data/judgments_evaluation.json
backend/data/gt_schema_v2.json
backend/scripts/Evaluation/validate_benchmark.py
backend/scripts/Evaluation/migrate_gt_v2.py

Current benchmark:
107 queries
1212 GT labels
755 core
457 supporting

Frozen-corpus GT coverage:
1018 / 1212
83.99%

All 107 queries contain at least one in-corpus GT label.
State:
FROZEN FOR CURRENT RETRIEVAL EXPERIMENTS

26. Benchmark Mutation Utilities
migrate_gt_v2.py
Purpose:
historical V1 -> V2 migration

Current use:
DO NOT rerun on frozen benchmark
unless intentionally versioning a new benchmark

clean_duplicates.py
Purpose:
historical benchmark cleanup/mutation

Current use:
DO NOT run on frozen benchmark casually

run_all_batches.sh
Purpose:
batch judgment regeneration/update

Gemini:
YES

Current use:
DO NOT run against frozen benchmark

27. Ground-Truth Semantics
Schema V2 preserves independent dimensions.
Importance
core
supporting

Weights:
core        3
supporting  1

Legal function
substantive
procedural

Provenance
explicit
inferred

These dimensions must not be collapsed into each other.
28. Human Annotation
Directory:
backend/scripts/Annotation/

compute_kappa.py
Responsibility:
Cohen's Kappa / agreement calculation

adjudicate.py
Responsibility:
merge reviewer decisions into pilot adjudication output

Current evidence:
3 / 107 queries
Kappa = 0.3438

State:
PILOT / EXPERIMENTAL

Not sufficient for dataset-level human-validation claims.
29. Other Tests
Directory:
backend/scripts/Tests/

Known categories include:
File	Purpose / state
test_runtime_retrieval.py	Automated runtime contract regression
test_baseline_protocol.py	Automated frozen protocol regression
test_benchmark_assertions.py	Benchmark/config assertions
test_weighted_recall.py	Weighted recall math
test_iaa.py	Kappa math
test_search.py	Local retrieval smoke
test_legal_qa.py	Gemini/model smoke; potentially external-cost
test_llm_intent.py	LLM intent/prompt smoke
test_judicial_logic.py	Manual/print-oriented logic smoke


Do not assume every test_*.py file is a proper isolated pytest unit test.
30. Data / Artifact Map
Artifact	Producer	Consumer	State
Frozen standardized five-law corpus	Flow/enrichment history	Evaluator/runtime builder	ACTIVE RESEARCH CORPUS
runtime_law_index.faiss	Runtime builder	FastAPI SearchService	ACTIVE GENERATED RUNTIME
runtime_law_metadata.json	Runtime builder	FastAPI SearchService	ACTIVE GENERATED RUNTIME
runtime_retrieval_manifest.json	Runtime builder/promotion	FastAPI SearchService	ACTIVE GENERATED RUNTIME
law_index.faiss	Legacy vector build	Legacy scripts only	SUPERSEDED
law_metadata.json	Legacy vector build	Legacy runtime path	MISSING / SUPERSEDED
judgments_evaluation.json	Benchmark history/migration	Evaluation	FROZEN BENCHMARK
gt_schema_v2.json	Schema definition	Validator/tests	ACTIVE
structured_judgments/*.json	ocrFullBanAn.py	Audit/manual/offline	OFFLINE
Golden baseline run artifacts	baseline evaluator	parity verifier/research audit	FROZEN HISTORICAL EVIDENCE
Annotation artifacts	annotation scripts/manual work	pilot adjudication	EXPERIMENTAL


31. Golden Baseline Evidence
Protocol:
baseline_d0_v2

Historical run:
baseline_d0_v2_20260825_081357_617763_02bf9e13

Primary closed-world @10:
Retriever	Recall	nDCG	MRR	Weighted Recall
BM25	0.0760	0.0333	0.1415	0.0522
FAISS	0.0632	0.0513	0.1450	0.0591
Hybrid	0.0883	0.0529	0.1901	0.0725


No statistical-significance claim has yet been established.
32. Reproducibility / Git Checkpoints
Runtime artifact tooling:
64db4f0b
feat(runtime): add deterministic retrieval artifact recovery

FastAPI runtime recovery:
886a0783
fix(runtime): recover deterministic baseline retrieval

Historical exact golden provenance is preserved separately and must not be replaced conceptually by rewritten/sanitized repository commit hashes.
33. Cost / Mutation Safety Map
Safe from Gemini cost
pytest
py_compile
grep
git
Runtime/build_retrieval_artifacts.py
Runtime/verify_golden_parity.py
Flow/01_extract_laws.py
Flow/03_load_vectors.py
ocrFullBanAn.py

However:
Flow/03_load_vectors.py

can overwrite vector artifacts and should still not be run casually.
Gemini-consuming
Flow/02_transform_laws.py
Evaluation/process_judgments.py
Evaluation/run_all_batches.sh
run_pipeline.py --all
POST /ask

Mark these explicitly before running.
34. Current Research Boundary
Runtime
Q0
D0
BM25
Dense
Static Hybrid

Research-only / experimental
D1
hybrid tuning
dynamic hybrid
routing
temporal filtering
reranking
significance testing
advanced diagnostics

A research feature becomes runtime functionality only after:
controlled experiment
    ↓
decision
    ↓
explicit implementation
    ↓
runtime tests
    ↓
documentation update

35. Planned-Only Architecture
The following remain planned unless actual implementation is later added:
Planned feature	Documentation / concept	Current implementation status
Corpus Schema V3	corpus design docs	PLANNED
Leakage-controlled enrichment	enrichment design	PLANNED
Dedicated runtime router service	routing design	NOT IMPLEMENTED
Runtime temporal version routing	temporal design	NOT IMPLEMENTED
Production snapshot/rollback	deployment/recovery concept	NOT IMPLEMENTED
Full benchmark human adjudication	research requirement	NOT IMPLEMENTED


Experimental logic with similar names inside the evaluator does not make these runtime features implemented.
36. Highest-Priority Entry Points
Runtime
backend/app/config.py
backend/app/main.py
backend/app/models/schema.py
backend/app/services/embedding_service.py
backend/app/services/search_service.py

Runtime artifact generation
backend/scripts/Runtime/build_retrieval_artifacts.py
backend/scripts/Runtime/verify_golden_parity.py

Controlled evaluation
backend/scripts/Evaluation/configs/baseline_d0.yaml
backend/scripts/Evaluation/configs/exp02_d0.yaml
backend/scripts/Evaluation/configs/exp02_d1.yaml
backend/scripts/Evaluation/configs/exp03_static_fusion_cv.yaml
backend/scripts/Evaluation/experiments/baseline.py
backend/scripts/Evaluation/experiments/exp03_static_fusion_cv.py
backend/scripts/Evaluation/core/
backend/scripts/Evaluation/retrieval/

Benchmark
backend/data/judgments_evaluation.json
backend/data/gt_schema_v2.json
backend/scripts/Evaluation/validate_benchmark.py

Tests
backend/scripts/Tests/test_runtime_retrieval.py
backend/scripts/Tests/test_baseline_protocol.py
backend/scripts/Tests/test_exp02_protocol.py
backend/scripts/Tests/test_exp03_protocol.py

37. Current State Matrix
Area	Status	Authoritative implementation/evidence
FastAPI API	IMPLEMENTED / PARTIAL PRODUCT	backend/app/main.py
Runtime retrieval	RECOVERED	SearchService.search()
Runtime contract	VERIFIED	runtime_retrieval_v1
Query representation	FROZEN	Q0
Document representation	FROZEN	D0
BM25	RECOVERED	whitespace BM25 module
Dense retrieval	RECOVERED	Vietnamese-SBERT exact revision
FAISS	RECOVERED	IndexFlatIP, 768-D, 1811
Static Hybrid	RECOVERED	min-max 0.5/0.5
Determinism	VERIFIED	standard_law_id tie-break
Runtime reranking	NOT USED	none in baseline
Runtime temporal filtering	NOT USED	evaluator only
Runtime dynamic routing	NOT USED	evaluator only
Golden parity	VERIFIED	321/321
/search	VERIFIED	HTTP golden top-20 parity
/ask retrieval	ALIGNED	same raw Q0 retrieval
/ask generation	GEMINI DEPENDENT	LLMService
Benchmark	FROZEN	107 queries
Human annotation	INCOMPLETE	3-query pilot
Advanced evaluator	EXPERIMENTAL	Evaluation modules
EXP-02	COMPLETE	D1 not promoted; runtime/baseline remain D0
EXP-03	COMPLETE	CV-selected static policy not promoted; frozen 0.5/0.5 retained
Next research action	READ-ONLY ANALYSIS	Per-query oracle/headroom on existing EXP-03 candidate cache
