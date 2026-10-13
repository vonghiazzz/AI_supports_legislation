# AI Supports Legislation — Current Work

Last updated: 2026-10-13

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
1. Finish documentation checkpoint
2. Preserve runtime_retrieval_v1 recovery state
3. Run EXP-02: D0 vs D1
4. Establish dev/test discipline before tuning
5. Tune static hybrid only after EXP-02
6. Evaluate dynamic hybrid
7. Evaluate temporal retrieval
8. Evaluate reranking
9. Perform statistical analysis
10. Expand human annotation / benchmark audit

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
3. Immediate Research Work — EXP-02 D0 vs D1
Status
NEXT ACTIVE RESEARCH EXPERIMENT
Research question:
Does D1 document enrichment improve Vietnamese judgment-derived statutory retrieval compared with the frozen D0 baseline?

Control
Everything except the document representation must remain fixed.
Corpus                  same 1,811 articles
Benchmark               same 107 queries
Query representation    Q0
Embedding model         keepitreal/vietnamese-sbert
Embedding revision      a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23
Embedding dimension     768
BM25 tokenizer          whitespace
FAISS                   IndexFlatIP
Candidate pool          30
Fusion                  min-max
BM25 weight             0.5
Dense weight            0.5
Tie-break               deterministic
Evaluation protocol     same

Only this changes:
D0
vs
D1

D0
Current baseline representation:
Tiêu đề: {title}. Nội dung: {content}

Used identically for:
BM25
Dense

D1
D1 adds enrichment fields.
Current evaluator definition should remain the source of truth.
Conceptually D1 includes:
title
content
summary
terms / lexical enrichment

Before running EXP-02, verify the exact current build_document_text(..., "d1") implementation rather than relying on documentation wording.
Metrics
Report:
Recall@5
Recall@10
Recall@20

nDCG@5
nDCG@10
nDCG@20

MRR@5
MRR@10
MRR@20

Weighted Recall@5
Weighted Recall@10
Weighted Recall@20

Primary endpoint:
closed-world @10

Secondary reporting:
@5
@20

Do not choose the winner based on whichever K happens to look best.
Experimental constraints
Do not:
tune D0 and D1 separately
change hybrid weights
change tokenizer
change query representation
change model revision
add temporal filtering
add reranking
change benchmark
change corpus

during EXP-02.
Otherwise the experiment would no longer isolate document representation.
Required outputs
EXP-02 should produce:
experiment configuration
corpus fingerprint
benchmark fingerprint
query fingerprint
model revision
per-query retrieval results
aggregate metrics
D0 vs D1 delta table
failure-change analysis
run manifest

If possible, preserve:
query-level paired results

for later significance testing.
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
NOT NEXT
Static tuning comes after EXP-02.
The existing frozen baseline uses:
min-max
w_bm25 = 0.5
w_dense = 0.5

This weight is a baseline configuration, not permission to optimize against all 107 queries.
Required methodological improvement
Before tuning:
define dev/test split
or
define deterministic cross-validation
or
define another defensible resampling protocol

Do not optimize hyperparameters over all 107 queries and then report performance on the same 107 as unbiased test performance.
Query intent distribution is approximately:
104 mixed
3 semantic

Therefore naive intent-stratified splitting is not useful.
10. Dynamic Hybrid
Status
PLANNED RESEARCH
Only consider after:
EXP-02
static tuning protocol
static tuned baseline

Dynamic weighting must demonstrate improvement against:
frozen static baseline
and
properly tuned static baseline

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
IN PROGRESS
Current documentation checkpoint order:
1. PROJECT_STATE.md       done/rewrite prepared
2. CURRENT_WORK.md        this file
3. CODEBASE_MAP.md        next
4. DECISIONS_AND_HISTORY.md
5. README.md

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
Until the documentation checkpoint and EXP-02 design are complete, do not:
change frozen baseline weights
add reranking to runtime
add temporal filtering to runtime
add dynamic routing to runtime
regenerate benchmark with Gemini
mutate frozen ground truth
overwrite runtime artifacts via Flow/03
tune on all 107 queries
claim statistical significance
claim benchmark is human validated

21. Immediate Next Action
After documentation is synchronized:
Design and execute EXP-02: D0 vs D1

The first EXP-02 step should be an audit of the exact current D1 implementation and experimental configuration.
Then create a reproducible experiment definition that changes only:
document representation

from:
D0

to:
D1

Everything else remains frozen.
22. Definition of Done for EXP-02
EXP-02 is complete only when all of the following exist:
reproducible config
protocol ID
corpus fingerprint
benchmark fingerprint
query fingerprint
exact model revision
per-query output
aggregate @5/@10/@20 metrics
D0 vs D1 comparison
paired delta data
failure-change analysis
run manifest
documented conclusion

The conclusion must distinguish:
Observed result
Interpretation
Hypothesis

Do not convert an observed metric difference directly into a causal explanation.
23. Current Blocker
There is currently no runtime retrieval blocker.
The immediate blocker to further research is methodological:
EXP-02 must be defined so that D0 vs D1 is an isolated comparison and does not accidentally change other retrieval variables.

Therefore the next technical research task is not “improve retrieval.”
It is:
freeze EXP-02 controls
verify exact D1 representation
run controlled D0 vs D1 experiment