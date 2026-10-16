# AI Supports Legislation

JSON
FAISS
PDF/files

There is currently no relational database or external vector database in the retrieval path.
7. /search
POST /search is the canonical deterministic retrieval endpoint.
Example request:
{
  "query": "giao dịch dân sự vô hiệu do không tuân thủ hình thức",
  "top_k": 10
}

Valid:
1 <= top_k <= 20

Flow:
raw query
    ↓
Q0 whitespace normalization
    ↓
BM25 top-30
Dense top-30
    ↓
min-max normalization
    ↓
0.5 BM25 + 0.5 Dense
    ↓
deterministic ranking
    ↓
SearchResult[]

/search does not require Gemini.
Use this endpoint for deterministic retrieval testing.
8. /ask
POST /ask uses the same frozen Q0 retrieval path as /search.
High-level flow:
user query
    ↓
frozen statutory retrieval
    ↓
retrieved legal articles
    ↓
Gemini answer generation

Gemini intent extraction/query expansion is not part of runtime_retrieval_v1.
/ask may still call Gemini for:
answer generation
answer evaluation

Therefore /ask is not a zero-cost/local-only endpoint.
9. Health Endpoints
Available endpoints:
GET /health/live
GET /health/ready
GET /health

Semantics:
/health/live
    process is alive

/health/ready
    frozen retrieval contract is loaded and valid

/health
    compatibility endpoint delegating to readiness

FastAPI performs fail-fast retrieval initialization during startup.
If runtime artifacts are absent or incompatible, the app should not silently serve an invalid retriever.
10. Runtime Artifacts
Active generated artifacts:
backend/data/runtime_law_index.faiss
backend/data/runtime_law_metadata.json
backend/data/runtime_retrieval_manifest.json

These files are generated and intentionally ignored by Git.
Do not confuse them with the superseded legacy artifacts:
backend/data/law_index.faiss
backend/data/law_metadata.json

The old local law_index.faiss contains 1,813 vectors and is not the active runtime index.
11. Runtime Artifact Tooling
Version-controlled runtime tooling:
backend/scripts/Runtime/
├── __init__.py
├── build_retrieval_artifacts.py
└── verify_golden_parity.py

Build
build_retrieval_artifacts.py constructs the frozen runtime artifact set and validates:
required five-law corpus
per-law counts
total 1811
corpus fingerprint
D0/Q0 protocol
exact embedding revision
dimension 768
IndexFlatIP
ntotal 1811
unique standard_law_id

Inspect supported CLI options before running:
python backend/scripts/Runtime/build_retrieval_artifacts.py --help

Golden parity
verify_golden_parity.py compares generated artifacts with the historical frozen baseline.
Inspect options:
python backend/scripts/Runtime/verify_golden_parity.py --help

Full verified checkpoint:
queries_checked   107
retrieval_checks  321
failures          0

12. Backend Setup
Recommended Python version:
Python 3.10+

Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate

Install backend dependencies:
pip install -r backend/requirements.txt

Additional system dependencies may be required for OCR:
macOS:
brew install poppler
brew install tesseract
brew install tesseract-lang

Runtime Gemini functionality requires:
GEMINI_API_KEY

in the environment or .env.
13. Run FastAPI
From repository root:
PYTHONPATH=backend uvicorn app.main:app --reload

The retrieval artifacts must already exist in:
backend/data/

Startup should validate the runtime contract before the app begins serving requests.
Check readiness:
curl http://127.0.0.1:8000/health/ready

Search example:
curl -X POST \
  http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "giao dịch dân sự vô hiệu do không tuân thủ hình thức",
    "top_k": 5
  }'

14. Runtime Regression Tests
Fast runtime/baseline checkpoint:
PYTHONPATH=backend python -m pytest \
  backend/scripts/Tests/test_runtime_retrieval.py \
  backend/scripts/Tests/test_baseline_protocol.py \
  -q

Recent verified checkpoint:
66 passed

Compile check:
PYTHONPATH=backend python -m py_compile \
  backend/app/config.py \
  backend/app/main.py \
  backend/app/models/schema.py \
  backend/app/services/embedding_service.py \
  backend/app/services/search_service.py \
  backend/scripts/Tests/test_runtime_retrieval.py

Local retrieval smoke:
PYTHONPATH=backend python \
  backend/scripts/Tests/test_search.py

This smoke does not call Gemini.
15. Repository Structure
Important directories:
backend/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── models/
│   └── services/
│
├── data/
│
└── scripts/
    ├── Runtime/
    ├── Flow/
    ├── Evaluation/
    ├── Annotation/
    └── Tests/

frontend/
docs/

More detailed ownership is documented in:
CODEBASE_MAP.md

16. Offline Statute Pipeline
Generic statute processing pipeline:
backend/scripts/Flow/

Conceptually:
law source files
    ↓
01_extract_laws.py
    ↓
raw_<law>.json
    ↓
02_transform_laws.py
    ↓
standardized_<law>.json
    ↓
03_load_vectors.py

Important:
Flow/03_load_vectors.py

is not the authoritative frozen runtime artifact builder.
Do not use it to overwrite:
runtime_law_index.faiss
runtime_law_metadata.json

The frozen runtime artifacts should be built through:
backend/scripts/Runtime/build_retrieval_artifacts.py

17. Judgment / Benchmark Pipeline
Judgments are used to construct the research benchmark.
Relevant files include:
backend/scripts/ocrFullBanAn.py
backend/scripts/Evaluation/process_judgments.py
backend/scripts/Evaluation/validate_benchmark.py
backend/data/judgments_evaluation.json
backend/data/gt_schema_v2.json

The current benchmark is frozen for the active retrieval experiments.
Do not casually mutate it.
In particular, do not run:
run_all_batches.sh
migrate_gt_v2.py
clean_duplicates.py

against the frozen benchmark unless intentionally creating a new benchmark version.
18. Gemini Cost / Mutation Safety
No Gemini call
These operations do not require Gemini:
pytest
py_compile
grep
git
Runtime/build_retrieval_artifacts.py
Runtime/verify_golden_parity.py
Flow/01_extract_laws.py
Flow/03_load_vectors.py
ocrFullBanAn.py
POST /search

However:
Flow/03_load_vectors.py

can overwrite vector artifacts and should still not be run casually.
⚠ Gemini-consuming
These paths can make external Gemini calls:
Flow/02_transform_laws.py
Evaluation/process_judgments.py
Evaluation/run_all_batches.sh
run_pipeline.py --all
POST /ask

Do not use them as routine verification commands.
19. Runtime vs Experimental Research
This distinction is mandatory.
Runtime
Current frozen runtime:
Q0
D0
BM25
Dense
Static Hybrid

Experimental / research-only
The evaluator may additionally contain:
D1
fusion tuning
dynamic hybrid
query routing
temporal filtering
reranking
error taxonomy
statistical testing
paper reporting

A feature existing in backend/scripts/Evaluation/ does not mean it is active in FastAPI.
20. Current Research Roadmap
Completed:
B0  BM25
B1  Dense
E1  Static Hybrid
EXP-02  D0 vs D1
EXP-03  leakage-controlled static fusion-weight CV
offline oracle/headroom diagnostic
EXP-04  selective Dynamic Hybrid nested CV

Next:
preserve EXP-04 and avoid immediate router retuning on the same 107 queries
obtain an untouched benchmark before renewed Dynamic Hybrid work or strong claims
temporal retrieval
reranking
statistical analysis

The completed oracle/headroom analysis uses ground truth and is only an upper-bound diagnostic, not deployable, held-out, or confirmatory performance.
21. EXP-02 — D0 vs D1

Status: **COMPLETE**.

EXP-02 changed only the document representation while preserving Q0, the 1,811-document corpus, all 107 queries, model revision, retrieval parameters, and metric protocol.

At the primary closed-world @10 endpoint, D1 showed suggestive BM25 gains but mixed Dense and Hybrid effects. No tested delta had a paired-bootstrap 95% confidence interval excluding zero or paired sign-flip `p < 0.05`.

Decision: D1 was not promoted. Runtime and the frozen research baseline remain Q0/D0. See `PROJECT_STATE.md` for run IDs, detailed metrics, paired uncertainty, and the Dense truncation limitation.

EXP-03 subsequently completed that leakage-controlled static-weight evaluation; see the next section.

21A. EXP-03 — Static Fusion Weight CV

Status: **COMPLETE**.

Official run: `exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23`.

The five-fold OOF selected policy was below the fixed 0.5/0.5 control on all four aggregate closed-world metrics at K=10. Primary Weighted Recall@10 changed by `-0.007193031`, approximately `-9.92%`; Recall and Weighted Recall decreases had paired intervals excluding zero and sign-flip `p < 0.05`.

Decision: do not promote the CV-selected static policy. Runtime and `baseline_d0_v2` remain at 0.5/0.5. See `PROJECT_STATE.md` for the canonical detailed protocol, results, uncertainty, limitations, and interpretation.

21B. Offline Oracle/Headroom Diagnostic

Status: **COMPLETE — DIAGNOSTIC ONLY**.

Using ground truth and the frozen EXP-03 candidate cache, the per-query WR@10 oracle ceiling was 24.24% above fixed 0.5, but only 23/107 queries improved, median and p75 gain were zero, and broad tied-weight plateaus dominated. The evidence is classified as **MODERATE HEADROOM**.

Decision at that gate: proceed to a leakage-safe selective-policy experiment, without runtime adoption. The resulting same-benchmark nested-CV evidence remained exploratory because the all-query oracle influenced the policy-family choice. EXP-04 subsequently completed that experiment; see the next section and `PROJECT_STATE.md`.

21C. EXP-04 — Selective Dynamic Hybrid

Status: **COMPLETE — DO NOT PROMOTE**.

Official run: `exp04_selective_dynamic_hybrid_nested_5fold_cv_v1_20261016_071606_287853_8d0649b7`.

Dynamic OOF closed-world Weighted Recall@10 was `0.07053616402199708`, below `0.07248115500412532` for fixed 0.5. The router made 16 overrides: none improved primary WR@10, 13 tied, and 3 were harmful. No outer fold had a positive mean primary delta, while fixed-control parity remained exact.

Decision: do not promote Dynamic Hybrid; keep static 0.5/0.5 fusion and runtime unchanged. Do not immediately retune the router on the same 107-query outcomes because repeated redesign would increase development-set overfitting risk. See `PROJECT_STATE.md` for the canonical detailed result and limitations.

22. Human Annotation Status
Current human annotation is only a pilot:
3 / 107 queries
Cohen's Kappa = 0.3438

The benchmark must not be described as fully human validated.
Future audit should distinguish categories such as:
TRUE_MISSING_LAW
ALIAS / CANONICALIZATION
BAD_AI_LABEL
OUT_OF_SCOPE

Final adjudication should be human-reviewed rather than self-validated by the same AI process that generated the labels.
23. Important Project Documents
Use the following files for different purposes:
PROJECT_STATE.md

Current overall system state.
CURRENT_WORK.md

Immediate work and next research steps.
CODEBASE_MAP.md

Feature → file/function ownership.
DECISIONS_AND_HISTORY.md

Architectural/research decisions and why they were made.
README.md

Onboarding and operational overview.
24. Relevant Git Checkpoints
Runtime artifact recovery tooling:
64db4f0b
feat(runtime): add deterministic retrieval artifact recovery

FastAPI runtime recovery:
886a0783
fix(runtime): recover deterministic baseline retrieval

Historical exact experiment provenance is preserved separately and must not be confused with rewritten/sanitized Git commit hashes.
25. Development Rule
When investigating project behavior, use:
CODE
>
TEST
>
CONFIG
>
DOCUMENTATION
>
README
>
COMMENT
>
ASSUMPTION

For research conclusions, distinguish:
Observed
Interpretation
Hypothesis

Do not turn an observed metric difference directly into an unsupported causal explanation.

## Sau README, bộ 5 docs sẽ có vai trò rõ ràng

```text
README.md
    onboarding / cách chạy / project là gì

PROJECT_STATE.md
    hiện tại hệ thống đang ở đâu

CURRENT_WORK.md
    hiện tại làm gì tiếp

CODEBASE_MAP.md
    code nào sở hữu chức năng nào

DECISIONS_AND_HISTORY.md
    tại sao lại thiết kế/chọn như vậy
