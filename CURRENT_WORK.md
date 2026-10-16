# AI Supports Legislation — Current Work

Last updated: 2026-10-16

This document records the work that is currently active or immediately next.

It is intentionally narrower than `PROJECT_STATE.md`.

Use:

- `PROJECT_STATE.md` for the current overall system state.
- `CURRENT_WORK.md` for what is being worked on now and what should happen next.
- `CODEBASE_MAP.md` for feature → file/function ownership.
- `DECISIONS_AND_HISTORY.md` for why important architectural/research decisions were made.

Source-of-truth rule:

`CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION`

---

# 1. Current Work Order

The current priority order is:

```text
1. Record and preserve the completed EXP-03 and oracle/headroom decisions
2. Preserve runtime_retrieval_v1 and baseline_d0_v2 unchanged
3. Design a leakage-safe selective Dynamic Hybrid protocol without implementing it
4. Plan nested exploratory evaluation and an untouched confirmatory benchmark
5. Evaluate temporal retrieval
6. Evaluate reranking
7. Expand human annotation / benchmark audit
```

The previous engineering blocker — broken FastAPI retrieval — has been resolved.
Do not introduce new retrieval features into runtime merely because they exist in the experimental evaluator.
2. Runtime Retrieval Recovery
Status
COMPLETED
The FastAPI retrieval path has been recovered to a deterministic frozen baseline contract.
Runtime contract:
runtime_retrieval_v1

Research protocol:
baseline_d0_v2

Current retrieval configuration:
Corpus          1,811 articles
Query           Q0
Document        D0
BM25 tokenizer  whitespace
Dense model     keepitreal/vietnamese-sbert
Revision        a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23
Dimension       768
Normalization   L2
FAISS           IndexFlatIP
Pool K          30
Fusion          min-max
BM25 weight     0.5
Dense weight    0.5
Tie-break       score DESC -> standard_law_id ASC
Reranker        none
API top_k       1..20

Runtime artifacts
Generated artifacts:
backend/data/runtime_law_index.faiss
backend/data/runtime_law_metadata.json
backend/data/runtime_retrieval_manifest.json

These files are generated artifacts and are intentionally ignored by Git.
They must not be confused with the superseded:
backend/data/law_index.faiss
backend/data/law_metadata.json

The old law_index.faiss contains 1,813 vectors and is not the active runtime index.
Verification completed
Artifact contract validation:
corpus_count       1811
metadata_count     1811
unique IDs         1811
FAISS type         IndexFlatIP
FAISS dimension    768
FAISS ntotal       1811

Golden parity:
107 queries
BM25   top-20 exact ranking match
FAISS  top-20 exact ranking match
Hybrid top-20 exact ranking match

321 / 321 retrieval checks passed

FastAPI SearchService also reproduces the historical Hybrid top-20 ranking.
HTTP verification:
GET  /health/live   -> 200
GET  /health/ready  -> 200
GET  /health        -> 200
POST /search        -> 200

POST /search reproduces the golden Hybrid top-20 ranking for the verified query.
Runtime + baseline targeted tests:
66 passed

Existing retrieval smoke:
backend/scripts/Tests/test_search.py
exit = 0

Relevant commits
64db4f0b
feat(runtime): add deterministic retrieval artifact recovery

886a0783
fix(runtime): recover deterministic baseline retrieval

Runtime retrieval work still allowed
Only maintenance/guardrail work should be performed before the next research experiment.
Examples:
bug fixes
test improvements
artifact validation
documentation
deployment/readiness improvements

Do not add:
dynamic routing
temporal filtering
CrossEncoder reranking
D1
query expansion
adaptive hybrid

to the baseline runtime unless they have first been evaluated as controlled research experiments.
3. EXP-02 — D0 vs D1

Status: **COMPLETE**

Official runs:

- D0: `exp02_d0_vs_d1_v1_d0_20261013_085034_383461_4828f45f`
- D1: `exp02_d0_vs_d1_v1_d1_20261013_090537_436695_379066cd`

Observed:

- Both runs used all 107 queries and the same frozen 1,811-document corpus with clean provenance.
- At closed-world @10, BM25 improved on all four aggregate metrics under D1; FAISS and Hybrid effects were mixed.
- No tested @10 comparison had a paired-bootstrap 95% confidence interval excluding zero or a paired sign-flip `p < 0.05`.
- D1 enrichment fields were populated for all 1,811 documents.
- The 256-token Dense limit leaves 619 D0/D1 document pairs with identical truncated Dense input.

Interpretation:

The benchmark does not provide sufficient evidence for a consistent D1 improvement over D0. BM25 nDCG is a suggestive positive trend, not a significant result.

Hypothesis / limitation:

Dense truncation may contribute to mixed Dense results, but causality was not established.

Decision:

- Do not promote D1.
- Keep runtime and frozen research baseline on Q0/D0.
- Do not rebuild runtime artifacts because of EXP-02.
- Use `PROJECT_STATE.md` as the canonical detailed EXP-02 result record.
3A. EXP-03 — Static Fusion Weight CV

Status: **COMPLETE**

Official run: `exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23`

Observed:

- Five deterministic folds selected BM25 weights `0.4/0.5/0.5/0.5/0.4`.
- The OOF selected policy was below the fixed 0.5 control on all four closed-world metrics at K=10.
- Primary Weighted Recall@10 changed from `0.072481155004` to `0.065288123567`, a delta of `-0.007193031` or approximately `-9.92%`.
- Recall and Weighted Recall decreases had paired 95% intervals excluding zero and sign-flip `p < 0.05`; nDCG and MRR decreases were not statistically established.

Decision:

- Do not promote the CV-selected static policy.
- Keep the frozen 0.5/0.5 policy, runtime, and `baseline_d0_v2` unchanged.
- Use `PROJECT_STATE.md` as the canonical detailed EXP-03 result record.
3B. Offline Oracle/Headroom Diagnostic

Status: **COMPLETE — DIAGNOSTIC ONLY**

Using the existing EXP-03 top-30 candidate cache and per-query ground truth, the WR@10 oracle ceiling was `0.090051` versus `0.072481` for fixed 0.5: `+0.017569792095`, or `+24.24%`. Only 23/107 queries improved, 84 tied, median and p75 gain were zero, and the top ten queries contributed 62.69% of headroom. Forty-six queries were flat over all eleven weights and only two had unique optima.

Decision: **MODERATE HEADROOM**. This supports Dynamic Hybrid protocol design only. It is not held-out, deployable, or confirmatory evidence. `PROJECT_STATE.md` is the canonical detailed oracle record.
4. Frozen Baseline
Status
COMPLETED / FROZEN
Protocol:
baseline_d0_v2

Primary closed-world @10 results:
Retriever	Recall	nDCG	MRR	Weighted Recall
BM25	0.0760	0.0333	0.1415	0.0522
FAISS	0.0632	0.0513	0.1450	0.0591
Hybrid	0.0883	0.0529	0.1901	0.0725


Hybrid wins all four predefined primary metrics at K=10.
No statistical significance claim has been established yet.
Frozen baseline must not be mutated
Do not silently change:
Q0
D0
corpus selection
corpus order
BM25 tokenizer
embedding model
embedding revision
normalization
FAISS type
fusion
weights
tie-break
ground truth

Any such change requires:
new protocol ID
new run
new manifest
new result comparison

5. Benchmark State
Status
FROZEN FOR CURRENT RETRIEVAL EXPERIMENTS
Benchmark:
107 judgment-derived queries

Ground truth:
1,212 total labels

755 core
457 supporting

Frozen-corpus coverage:
1,018 in corpus
194 outside corpus
83.99% coverage

All 107 queries have at least one in-corpus ground-truth authority.
Ground-truth semantics
Keep these dimensions separate:
core / supporting

means:
importance / relevance role

while:
substantive / procedural

means:
legal function

And:
explicit / inferred

means:
source provenance

Do not collapse these dimensions into a single label.
6. Benchmark Mutation Safety
The frozen benchmark must not be casually regenerated.
Potentially destructive or incompatible scripts include:
backend/scripts/Evaluation/process_judgments.py
backend/scripts/Evaluation/run_all_batches.sh
backend/scripts/Evaluation/migrate_gt_v2.py
backend/scripts/Evaluation/clean_duplicates.py

Rules:
DO NOT run run_all_batches.sh on the frozen benchmark.

DO NOT run migrate_gt_v2.py again unless intentionally
creating a new benchmark version.

DO NOT run clean_duplicates.py on the frozen benchmark.

DO NOT replace benchmark records without validation.

process_judgments.py contains older generation behavior and must not be assumed to produce the exact current frozen V2 benchmark contract.
7. Human Annotation and Benchmark Audit
Status
INCOMPLETE
Current annotation pilot:
3 / 107 queries
Cohen's Kappa = 0.3438

This is not sufficient for a dataset-level human-validation claim.
Next annotation work
Before expanding annotation, define a stable audit taxonomy.
Recommended categories:
TRUE_MISSING_LAW
ALIAS / CANONICALIZATION
BAD_AI_LABEL
OUT_OF_SCOPE

Human reviewers should adjudicate final labels.
The same AI that originally produced a label should not be treated as independent validation of that label.
Potential audit scope
A reasonable next manual audit is:
20–30 stratified queries

Sampling should consider retrieval behavior and benchmark characteristics, not only random query IDs.
Possible dimensions:
core vs supporting
substantive vs procedural
explicit vs inferred
in-corpus vs out-of-corpus GT
retrieval success vs retrieval failure
law family

8. Runtime vs Experimental Evaluation
This distinction must remain explicit in all future work.
Runtime
Current runtime implements:
Q0
D0
BM25
Dense
Static Hybrid

and no baseline reranker.
Experimental evaluator
Research code may additionally contain:
D1
dynamic weights
routing
temporal filtering
reranking
fusion tuning
error taxonomy
oracle analysis
statistical tests
LaTeX reporting

These features are EXPERIMENTAL until deliberately integrated into runtime.
Never write:
"The runtime uses temporal routing/reranking"

merely because such functionality exists inside the evaluator.
9. Static Hybrid Tuning
Status
COMPLETE — NOT PROMOTED

EXP-03 completed a leakage-controlled five-fold static fusion-weight evaluation. Its OOF selected policy did not improve on the frozen control.
The existing frozen baseline uses:
min-max
w_bm25 = 0.5
w_dense = 0.5

No new static weight becomes the baseline. See `PROJECT_STATE.md` for the official run, full protocol, metrics, uncertainty, and interpretation.
10. Dynamic Hybrid
Status
PROTOCOL DESIGN JUSTIFIED — NOT IMPLEMENTED

The completed oracle diagnostic found moderate theoretical headroom concentrated in a minority of queries.

The preferred initial hypothesis is selective routing: remain at fixed 0.5 unless an inference-time observable signal supports a high-confidence override. Do not default to unrestricted eleven-class weight prediction.

Any router must be evaluated independently of the retrieval performance it is supposed to improve.
11. Temporal Retrieval
Status
EXPERIMENTAL
Temporal filtering logic exists in research code.
It is not part of runtime_retrieval_v1.
Known limitation:
current temporal metadata is primarily law-level,
not article-amendment-level

Before making claims about legal temporal correctness, verify:
effective dates
replacement relationships
article-level amendments
overlapping validity
judgment/dispute date semantics

Temporal retrieval should not be enabled in runtime before this evidence exists.
12. Reranking
Status
EXPERIMENTAL
No reranker is part of the frozen baseline runtime.
Research reranking should be evaluated only after the candidate retriever is stable.
Questions to answer before runtime integration:
Does reranking improve primary @10 metrics?
Does it improve weighted recall?
Does it harm recall by reordering core authorities?
What is latency?
What is the device requirement?
Is the gain statistically reliable?

Do not assume CrossEncoder reranking is better simply because it is more complex.
13. Runtime Application Work
Retrieval is recovered, but the application is not production-ready.
Remaining runtime/application concerns include:
Gemini answer dependency
answer-evaluation validity
OCR upload validation
authentication
authorization
CORS policy
error semantics
persistent export storage
rate limits
observability
deployment configuration

These are engineering/product concerns and should not be mixed into retrieval experiment conclusions.
14. /ask State
/ask now uses the same raw Q0 retrieval path as /search.
Removed from baseline retrieval:
Gemini intent extraction
LLM query expansion
keyword-driven BM25 routing

Current high-level flow:
user query
   ↓
Q0 retrieval
   ↓
retrieved statutes
   ↓
Gemini answer generation

This preserves retrieval determinism.
However:
/ask still calls Gemini

for answer generation and possibly answer evaluation.
Therefore /ask is not a zero-cost local regression endpoint.
Use /search for deterministic retrieval testing.
15. Runtime Health and Readiness
Current endpoints:
GET /health/live
GET /health/ready
GET /health

Semantics:
/health/live
    process alive

/health/ready
    retrieval contract loaded and valid

/health
    currently delegates to readiness

Runtime initialization is fail-fast.
If required retrieval artifacts are missing or incompatible, startup fails rather than silently serving an invalid retriever.
16. Runtime Artifact Generation
The authoritative runtime artifact builder is:
backend/scripts/Runtime/build_retrieval_artifacts.py

Parity checker:
backend/scripts/Runtime/verify_golden_parity.py

Do not use:
backend/scripts/Flow/03_load_vectors.py

to overwrite the recovered runtime artifacts.
That script belongs to the generic/legacy Flow pipeline and uses a different document representation contract.
17. Gemini-Cost Safety
Before running scripts, distinguish local operations from Gemini-consuming operations.
No Gemini
pytest
py_compile
grep
git
Runtime/build_retrieval_artifacts.py
Runtime/verify_golden_parity.py
01_extract_laws.py
03_load_vectors.py
ocrFullBanAn.py

Note:
03_load_vectors.py does not call Gemini but can overwrite local vector artifacts, so it is still unsafe to run casually.
Gemini-consuming
⚠ Flow/02_transform_laws.py
⚠ Evaluation/process_judgments.py
⚠ Evaluation/run_all_batches.sh
⚠ run_pipeline.py --all
⚠ /ask

Do not run these simply as verification commands.
18. Source-Control State
Runtime artifact tooling and runtime integration are now committed.
Relevant commits:
64db4f0b
886a0783

Generated runtime data remains ignored.
That is intentional.
The version-controlled reproducibility boundary is:
builder
protocol/config
retrieval primitives
tests
parity verifier
documentation

not the large generated FAISS/metadata artifacts themselves.
19. Documentation Work
Status
CURRENT

The project documentation records the completed EXP-02 and EXP-03 decisions. `PROJECT_STATE.md` is the canonical detailed EXP-03 result record; this file tracks only immediate work.

Documentation must remove stale claims that runtime currently uses:
BGE-M3
1024-D embeddings
1813-vector production corpus
regex BM25
0.6/0.4 fallback
CrossEncoder baseline reranking
missing runtime metadata
unhealthy retrieval

Those describe superseded runtime behavior.
20. Do Not Do Yet
Before a Dynamic Hybrid protocol is reviewed and approved, do not:
change frozen baseline weights
add reranking to runtime
add temporal filtering to runtime
implement or add dynamic routing to runtime
regenerate benchmark with Gemini
mutate frozen ground truth
overwrite runtime artifacts via Flow/03
use full-107 oracle labels for router fitting or nested-CV decisions
describe oracle performance as deployable, held-out, or confirmatory performance
claim EXP-03 proves dynamic routing will work
claim benchmark is human validated

21. Immediate Next Action

Design, but do not implement, a selective Dynamic Hybrid experiment:

- Default to fixed 0.5 and override only at high confidence.
- Define plateau-aware labels using training-fold ground truth only.
- Restrict features to inference-time query/retrieval signals.
- Nest all preprocessing, model, threshold, and action selection inside outer training folds.
- Keep fixed 0.5 as the primary control and treat same-benchmark nested-CV results as exploratory.
- Plan an untouched additional benchmark before strong confirmatory claims.

22. EXP-03 Guardrail

EXP-03 satisfied its direct tuning guardrail through deterministic five-fold selection and one held-out prediction per query.

The completed oracle deliberately used all-query ground truth and influenced the next policy-family choice. It cannot be presented as deployable/test performance, and its full-107 labels must not enter nested router training.

23. Current Blocker

There is no runtime retrieval blocker. The active research task is methodological design: define a statistically defensible selective router under severe sample-size and class-imbalance constraints without allowing oracle information into outer held-out decisions.
