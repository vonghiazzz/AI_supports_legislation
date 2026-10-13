# AI Supports Legislation — Decisions and History

Last updated: 2026-10-13

This document records:

1. important architectural/research decisions;
2. why those decisions were made;
3. decisions that were later superseded;
4. reproducibility and Git-history checkpoints;
5. questions that remain genuinely unresolved.

It is not a description of the current implementation alone.

Use:

- `PROJECT_STATE.md` for the current system state;
- `CURRENT_WORK.md` for immediate next work;
- `CODEBASE_MAP.md` for implementation ownership;
- this file for historical reasoning and decision provenance.

Source-of-truth rule:

`CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION`

---

# 1. Core System Identity

## Decision

The project is a Vietnamese statutory Legal RAG system.

Judgments are primarily used to create:

```text
retrieval queries
ground truth
research benchmark evidence

The retrieval corpus is statutory law.
Current runtime retrieval therefore answers:
Which statutory articles are relevant to this judgment-derived legal query?

It is not currently a precedent retrieval system.
Consequence
The two major data roles are intentionally different:
Judgments
    -> queries / GT / evaluation

Statutory articles
    -> retrieval corpus

Do not silently mix judgment passages into the statutory corpus without defining a new task and protocol.
2. Retrieval Unit
Decision
The statutory retrieval unit is:
one legal article

not:
paragraph
sentence
fixed token chunk
sliding window
whole law

Reason
Existing law extraction and benchmark ground truth are both naturally article-oriented.
Ground truth typically references:
law
article number

Therefore article-level retrieval gives a direct evaluation unit.
Consequence
Current experiments do not test:
hierarchical retrieval
paragraph-level retrieval
parent-child retrieval
multi-vector article representation

Those would be separate experiments.
3. Canonical Article Identity
Historical problem
law_id alone is not globally unique.
For example:
Điều 1

exists in multiple laws.
Decision
Use:
standard_law_id

as the canonical retrieval identity.
Conceptually:
article identity + canonical law code

Example:
điều 129 bo_luat_dan_su_2015

Consequence
All deterministic retrieval comparison, fusion, parity verification and exact tie-breaking should operate on:
standard_law_id

rather than article number alone.
4. Ground-Truth Semantics
Decision
Ground truth preserves independent dimensions.
Importance
core
supporting

Legal function
substantive
procedural

Label provenance
explicit
inferred

These dimensions must not be conflated.
Weighting decision
Frozen retrieval evaluation uses:
core        = 3
supporting  = 1

This weighting is used by Weighted Recall and related graded analysis.
Consequence
A procedural article can be:
core

and a substantive article can be:
supporting

The labels answer different questions.
5. Closed-World vs Open-World Evaluation
Decision
Both views are useful, but the current primary controlled protocol uses:
closed-world

for the primary endpoint.
Reason
Some benchmark GT references laws outside the frozen five-law corpus.
A retriever cannot retrieve a document that does not exist in its corpus.
Therefore primary retrieval quality must distinguish:
retrieval failure

from:
corpus coverage limitation

Current frozen coverage
total GT          1,212
in corpus         1,018
outside corpus      194
coverage          83.99%

All 107 queries contain at least one in-corpus GT authority.
Primary endpoint
closed-world @10

Secondary reporting:
@5
@20

6. Query Representation Q0
Decision
The frozen baseline query representation is:
Q0

Definition:
" ".join(str(query).strip().split())


That is:
whitespace normalization only

Explicitly excluded from Q0
LLM rewriting
query expansion
intent extraction
keyword extraction
temporal rewriting
query routing
synthetic descriptions

Reason
Q0 creates a clean experimental baseline.
Any improvement from query enrichment should be measured as a separate controlled experiment.
7. Document Representation D0
Decision
The frozen baseline document representation is:
D0

Exact conceptual format:
Tiêu đề: {title}. Nội dung: {content}

Both retrieval modalities consume the same D0 representation:
BM25
Dense

Reason
Using the same representation isolates retriever differences from representation differences.
Important provenance note
The title already exists in the standardized corpus and was created during prior enrichment.
Therefore D0 is not equivalent to raw statute text alone.
D1 represents additional enrichment beyond this shared title/content representation.
8. D1 Is an Experiment, Not Runtime State
Decision
D1 is reserved for controlled experimental comparison.
It is not silently enabled in runtime.
Research question
Does additional enrichment improve statutory retrieval compared with D0?

Consequence
EXP-02 must change only:
D0 -> D1

while holding fixed:
corpus
queries
model
revision
BM25 tokenizer
FAISS
pool size
fusion
weights
tie-break
metrics

9. Dense Embedding Authority
Historical state
The old runtime and offline pipeline diverged:
old runtime
    BAAI/bge-m3
    1024-D

old local FAISS artifact
    keepitreal/vietnamese-sbert
    768-D
    ntotal=1813

This made runtime vector search contractually invalid.
The old project documentation correctly identified this as a serious conflict at that time. That state is now historical, not current.
Current decision
For the frozen baseline runtime:
model
keepitreal/vietnamese-sbert

Exact revision:
a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23

Embedding dimension:
768

Normalization:
L2

FAISS:
IndexFlatIP

Reason
This is the embedding contract used by the frozen historical baseline that has reproducible golden outputs.
Runtime recovery prioritizes:
scientific provenance
determinism
reproducibility

over replacing the baseline model with a newer/larger model.
Consequence
BGE-M3 may be evaluated later as a research experiment.
It is not part of:
runtime_retrieval_v1

or:
baseline_d0_v2

10. Frozen Corpus Selection
Decision
The controlled corpus contains exactly five laws:
bo_luat_dan_su_2015
bo_luat_to_tung_dan_su_2015
luat_dat_dai_2013
luat_dat_dai_2024
luat_hon_nhan_gia_dinh_2014

Counts:
689
517
212
260
133

Total:
1811

Excluded
test_law
mau_so_52

Corpus fingerprint
66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58

Historical note
The superseded local artifact:
law_index.faiss

contains:
1813 vectors

because it includes two non-baseline records.
That artifact is retained as historical/local data but is not an active runtime artifact.
11. BM25 Tokenization
Historical runtime
Old runtime BM25 used regex tokenization over:
title + content + keywords

This differed from the frozen evaluator.
Decision
Frozen baseline/runtime BM25 uses:
text.lower().split()


on D0.
Reason
The goal of runtime recovery was not to design a new tokenizer.
The goal was to reproduce the known baseline exactly.
Consequence
Alternative Vietnamese tokenization should be treated as a future experiment.
Examples:
PyVi
underthesea
word segmentation
field weighting
BM25 variants

should not be inserted into baseline runtime without evaluation.
12. Static Hybrid Fusion
Decision
Frozen Hybrid uses:
min-max normalization

with:
w_bm25  = 0.5
w_dense = 0.5

Candidate pool:
30

Deterministic tie policy
Final sort:
fusion score DESC
standard_law_id ASC

13. Why baseline_d0_v1 Was Superseded
Historical issue
The earlier baseline path created candidate IDs through unordered set behavior and then sorted only by score.
When two candidates had exactly equal fusion scores, their order could vary.
Effect
Approximately 20 queries showed tie-order changes.
The ranking instability did not materially change Recall, but it made exact reproduction impossible.
Decision
Supersede that protocol with:
baseline_d0_v2

using deterministic tie-breaking.
Deterministic implementation
Conceptually:
combined.sort(    key=lambda item: (        -item[1],        item[0],    ))


Consequence
Any future retrieval implementation claiming parity with the frozen baseline must reproduce this ordering policy.
14. Frozen Baseline Results
Primary closed-world @10:
Retriever	Recall	nDCG	MRR	Weighted Recall
BM25	0.0760	0.0333	0.1415	0.0522
FAISS	0.0632	0.0513	0.1450	0.0591
Static Hybrid	0.0883	0.0529	0.1901	0.0725


Interpretation decision
Hybrid is allowed to be described as:
best among the three frozen methods
for the predefined primary K=10 metrics

It must not yet be described as:
statistically significantly better

because significance has not been established.
15. No Reranker in Frozen Runtime
Historical state
Old runtime contained:
BAAI/bge-reranker-v2-m3

Experimental evaluation also contains:
itdainb/PhoRanker

These were two different ranking systems.
Decision
runtime_retrieval_v1 contains:
no CrossEncoder reranker

Reason
A frozen baseline must first reproduce:
BM25
Dense
Static Hybrid

before adding another model.
Consequence
Reranking remains:
EXPERIMENTAL

until evaluated in a later controlled experiment.
16. No Dynamic Routing in Frozen Runtime
Historical/experimental code
The evaluator contains heuristic routing logic for categories such as:
exact citation
procedural
substance
mixed

and can apply different retrieval weights.
Decision
This logic is not part of:
runtime_retrieval_v1

Reason
The benchmark intent distribution is highly imbalanced:
104 mixed
3 semantic

and no independently validated runtime router exists.
Consequence
Do not advertise “intent-aware routing” as current baseline runtime functionality.
17. No Temporal Filtering in Frozen Runtime
Experimental capability
The evaluator contains law-level temporal filtering.
Decision
Temporal filtering is excluded from:
baseline_d0_v2
runtime_retrieval_v1

Reason
Current temporal metadata is not sufficiently article/amendment granular to treat it as established legal temporal reasoning.
Consequence
Temporal retrieval remains research-only.
18. Runtime and Experimental Evaluation Are Separate Layers
Decision
The project explicitly distinguishes:
RUNTIME

from:
EXPERIMENTAL EVALUATION

Runtime currently represents
Q0
D0
BM25
Dense
Static Hybrid

Experimental evaluation may additionally contain
D1
fusion tuning
dynamic hybrid
routing
temporal filtering
reranking
error taxonomy
oracle overlap
statistical tests
paper reporting

Consequence
Presence of code in:
backend/scripts/Evaluation/

does not automatically mean the feature is active in FastAPI.
19. Runtime Artifact Manifest
Historical problem
Previously the runtime could load:
an arbitrary index
an arbitrary embedding model
missing/incompatible metadata

with little or no contract enforcement.
Decision
Runtime artifacts are governed by an explicit manifest.
Contract ID:
runtime_retrieval_v1

Manifest validates
protocol ID
corpus count
corpus fingerprint
query representation
document representation
embedding model
requested revision
resolved revision
dimension
normalization
FAISS type
FAISS ntotal
BM25 tokenizer
fusion method
fusion weights
tie policy
artifact filenames
artifact SHA256

Consequence
Artifact/model mismatch is now:
startup failure

rather than:
silent wrong retrieval

20. Runtime Fail-Fast Startup
Decision
FastAPI lifespan calls:
search_service.ensure_ready()

before serving traffic.
Current semantics
If required retrieval artifacts are missing or incompatible:
application startup fails

Reason
For this research/runtime recovery phase, serving an invalid retriever is worse than failing startup.
Trade-off
Because initialization is fail-fast:
/health/live

is only available once application startup succeeds.
A future production deployment may choose a different liveness/readiness model.
That decision is not yet required.
21. Health Semantics
Decision
Expose:
/health/live
/health/ready
/health

Meaning
/health/live
    process alive

/health/ready
    retrieval artifacts/model contract valid

/health
    compatibility endpoint delegating to readiness

Historical state
The old /health returned:
healthy

even when zero metadata records were loaded.
That behavior is superseded.
22. /search Is the Canonical Retrieval Endpoint
Decision
POST /search is the deterministic retrieval endpoint.
Flow:
raw query
  -> Q0
  -> BM25 + Dense
  -> Static Hybrid
  -> results

Consequence
Use /search for:
retrieval smoke tests
runtime parity checks
deterministic regression tests

It makes no Gemini calls.
23. /ask Uses the Same Retrieval Contract
Historical behavior
/ask previously performed Gemini intent extraction/query expansion before retrieval.
After runtime recovery, expanded query outputs no longer changed the deterministic retrieval path.
The intent Gemini call therefore became unnecessary retrieval overhead.
Decision
Remove Gemini intent extraction from the active /ask retrieval flow.
Current retrieval:
raw user query
    ↓
same Q0 SearchService.search()

Consequence
/search and /ask now retrieve from the same baseline contract.
Gemini is downstream of retrieval for:
answer generation
optional answer evaluation

24. Gemini Is Not Part of Baseline Retrieval
Decision
Frozen retrieval must be reproducible without Gemini.
Gemini remains used by
runtime answer generation
statute enrichment
benchmark generation
some experimental/offline tooling

Consequence
A Gemini outage must not change deterministic /search ranking.
25. Runtime Artifacts Are Generated, Not Versioned
Decision
Do not force-add these generated artifacts:
runtime_law_index.faiss
runtime_law_metadata.json
runtime_retrieval_manifest.json

when Git ignore policy excludes them.
Reason
The reproducibility boundary should be:
source
config
protocol
builder
manifest rules
tests
parity verifier

rather than committing large generated artifacts.
Consequence
A fresh runtime environment must regenerate or securely obtain the artifacts before startup.
26. Runtime Artifact Builder
Decision
The authoritative runtime builder is:
backend/scripts/Runtime/build_retrieval_artifacts.py

not:
backend/scripts/Flow/03_load_vectors.py

Reason
03_load_vectors.py uses a different representation contract and belongs to the generic/legacy Flow pipeline.
The runtime builder reuses frozen evaluator primitives and validates the exact baseline protocol.
27. Golden Parity as an Acceptance Gate
Decision
A newly generated runtime artifact is not accepted merely because:
dimension == 768
count == 1811

It must also reproduce historical retrieval behavior.
Tool
backend/scripts/Runtime/verify_golden_parity.py

Verified result
107 queries
3 retrieval methods/query
321 checks
0 failures

Retrievals checked:
BM25
FAISS
Hybrid

Ranking depth:
top-20

Consequence
Runtime recovery is supported by behavioral parity, not only structural compatibility.
28. Floating-Point Tolerance
Observation
Dense/Hybrid scores can differ by approximately:
1e-6

between environments while ranking remains identical.
Decision
Golden acceptance prioritizes:
exact ranked IDs

while small floating-point score differences may be tolerated.
Reason
The scientific result depends on ranking, not bit-identical floating-point values.
29. Source-Control History Rewrite
Historical problem
An early commit contained a Miniconda installer larger than GitHub's normal file-size limit:
backend/scripts/Miniconda3-latest-MacOSX-x86_64.sh

approximately:
118 MB

This blocked normal push.
Decision
Rewrite/sanitize repository history to remove the oversized installer from the public branch history.
Safety measures
Historical state was preserved through:
Git bundle
local backup ref
working-tree patch snapshots

The historical exact baseline commit was not conceptually discarded.
Important provenance rule
A sanitized rewritten commit hash is:
repository equivalent

but not:
the historical exact experiment commit

Scientific provenance must retain the original historical hash when referring to the golden run.
30. Historical Golden Commit
The exact historical golden baseline provenance was associated with:
ff3978af9773aed0a5f1bda4622874a02e18f347

The sanitized repository contains rewritten equivalent history.
Do not replace the historical exact hash inside old experiment provenance records merely because the public repository was rewritten.
31. Relevant Recovery Commits
Runtime artifact recovery tooling:
64db4f0b
feat(runtime): add deterministic retrieval artifact recovery

FastAPI runtime integration:
886a0783
fix(runtime): recover deterministic baseline retrieval

These commits establish the current runtime recovery checkpoint.
32. Why the Old 1813/BGE Runtime Was Not Simply Rebuilt
One possible response to the old mismatch would have been:
rebuild the 1813 index with BGE-M3

That was deliberately not chosen for recovery v1.
Reason
Doing so would create:
a new retriever

rather than:
a recovered frozen baseline

It would destroy direct comparability with historical experimental evidence.
Decision
Recover known-good baseline first.
Evaluate BGE-M3 later as a separate experiment if desired.
33. Why Advanced Evaluator Logic Was Not Copied Into Runtime
The evaluator contains richer functionality than runtime.
Examples:
routing
temporal filtering
reranking
tuning
diagnostics

Decision
Do not port these wholesale during runtime recovery.
Reason
Runtime recovery had one engineering objective:
restore a valid, reproducible retrieval contract.

Adding multiple research features simultaneously would make it impossible to identify which change affected retrieval behavior.
Consequence
Research advances must follow:
experiment
  -> evidence
  -> decision
  -> runtime integration

not:
experimental code exists
  -> copy into runtime

34. Baseline Before Optimization
Decision
Freeze and verify the baseline before tuning.
Reason
Without a frozen baseline:
metric changes cannot be attributed
reproduction is difficult
regressions are hidden

Consequence
Current experiment order is:
B0 / B1 / E1 frozen
    ↓
EXP-02 D0 vs D1
    ↓
static tuning
    ↓
dynamic hybrid
    ↓
temporal
    ↓
reranking

35. Do Not Tune on All 107 Queries
Historical concern
Existing experimental code contains in-memory fusion tuning over the evaluation set.
Decision
Future hyperparameter selection must introduce:
dev/test discipline

or another defensible resampling protocol.
Reason
Tuning and evaluating on the same 107 queries biases reported performance.
Consequence
Before static hybrid tuning, define:
deterministic split
cross-validation
or
another predeclared paired evaluation design

36. EXP-02 Before Fusion Tuning
Decision
Run:
D0 vs D1

before changing fusion weights.
Reason
This isolates document representation.
If representation and weights change simultaneously, an improvement cannot be attributed to either variable.

36A. EXP-02 Result — Retain D0

EXP-02 `exp02_d0_vs_d1_v1` completed with clean provenance at commit `bcfd918b89c533a6138e1187cf08ebe0beebe9c6`.

Official runs:

- D0 control: `exp02_d0_vs_d1_v1_d0_20261013_085034_383461_4828f45f`
- D1 treatment: `exp02_d0_vs_d1_v1_d1_20261013_090537_436695_379066cd`

Observed:

- Both runs evaluated 107 queries over the same 1,811-document corpus with matching corpus and Q0 fingerprints.
- All D1 enrichment fields were present for all documents.
- At closed-world @10, D1 changed BM25 Recall by `+0.005287`, nDCG by `+0.006484`, MRR by `+0.006097`, and Weighted Recall by `+0.005023`.
- FAISS effects were mixed; D1 nDCG changed by `-0.001954` and MRR by `-0.013103`.
- Hybrid effects were mixed; D1 Recall changed by `-0.006108`, nDCG by `+0.000441`, MRR by `+0.020950`, and Weighted Recall by `-0.005777`.
- No tested @10 delta had a paired-bootstrap 95% interval excluding zero or paired sign-flip `p < 0.05`. BM25 nDCG had a suggestive, non-significant `p = 0.07564`.
- The 256-token Dense window produced identical truncated D0/D1 token sequences for 619/1,811 documents; Dense input changed for 1,192/1,811.

Interpretation:

The current benchmark does not provide sufficient evidence to establish a consistent D1 improvement over D0. The lexical BM25 trend is suggestive; Dense and Hybrid results are mixed.

Hypothesis / limitation:

Dense truncation is a plausible explanatory limitation, not a proven cause of the observed metric pattern.

Decision:

- EXP-02 is complete.
- D1 is not promoted.
- Runtime remains D0.
- The frozen research baseline remains `baseline_d0_v2` with Q0/D0.
- Runtime artifacts are not changed because of EXP-02.
- The next research task is EXP-03 protocol design for leakage-controlled static fusion-weight tuning.

37. Human Annotation Is a Pilot
Historical evidence
Current double-annotation experiment covers:
3 / 107 queries

with stored:
Cohen's Kappa = 0.3438

Decision
Do not describe the benchmark as:
human validated
human adjudicated
expert verified

at dataset level.
Consequence
A larger stratified audit is needed.
Recommended adjudication categories:
TRUE_MISSING_LAW
ALIAS / CANONICALIZATION
BAD_AI_LABEL
OUT_OF_SCOPE

38. AI Must Not Self-Validate AI Labels
Decision
The same AI process that generated benchmark labels cannot be treated as independent validation of those labels.
Reason
That would create correlated error and false confidence.
Consequence
Final benchmark adjudication should use human review or genuinely independent evidence.
39. Benchmark Mutation Safety
Decision
Treat the current 107-query benchmark as frozen for current experiments.
Do not casually run:
run_all_batches.sh
migrate_gt_v2.py
clean_duplicates.py

against it.
Reason
These scripts can:
call Gemini
rewrite records
change schema
change IDs
change GT
invalidate historical fingerprints

Consequence
Any benchmark mutation requires:
new dataset version
new fingerprint
new validation
new experiment baseline

40. Generic Flow Pipeline Is Not the Frozen Runtime Pipeline
Decision
Maintain conceptual separation between:
Flow/

and:
Runtime/

Flow purpose
extract
Gemini-enrich
generic vector build

Runtime purpose
reconstruct frozen corpus
build exact artifacts
validate contract
verify parity

Consequence
Flow/03_load_vectors.py must not overwrite recovered runtime artifacts.
41. Local Files + FAISS Remain the Storage Design
Decision
Current retrieval architecture uses:
local JSON
local FAISS

rather than:
PostgreSQL
Elasticsearch
Milvus
Qdrant
Pinecone
Weaviate

Reason
Current priority is controlled research and local reproducibility.
Consequence
The current system does not provide:
distributed consistency
multi-node vector serving
transactional artifact updates
online index mutation

Those are deployment concerns, not current retrieval research requirements.
42. No NuExtract Dependency
Decision status
NuExtract is not part of the current implementation.
No active:
dependency
model
prompt
service
call site

has established it as part of the project.
Status:
UNUSED

43. DOCX Export Is Independent of Retrieval Research
Decision
DOCX generation is an application feature, not part of retrieval evaluation.
Consequence
Changes to:
DOCX formatting
download behavior
frontend export

must not be interpreted as retrieval improvements or research results.
44. OCR Pipelines Are Distinct
The project contains multiple OCR/extraction flows:
runtime upload OCR
statute extraction OCR
full judgment structural OCR
benchmark judgment OCR

Decision
Do not assume these paths are interchangeable.
Each currently has different:
trigger rules
rendering
page coverage
error behavior
downstream consumer

Consequence
Any future consolidation should be an explicit engineering project.
45. Observed vs Interpretation vs Hypothesis
Decision
Research documentation must distinguish:
Observed
Interpretation
Hypothesis

Example
Allowed:
Observed:
D1 Recall@10 increased by X.

Interpretation:
D1 performed better on the predefined primary endpoint.

Hypothesis:
Additional lexical enrichment may help retrieve articles
whose wording differs from the query.

Not allowed:
D1 improved because semantic enrichment solved vocabulary mismatch.

unless that causal explanation is separately supported.
46. Reproducibility Metadata Is Mandatory
Important controlled experiments should preserve:
protocol ID
corpus fingerprint
benchmark fingerprint
query fingerprint
model name
exact model revision
representation
retrieval parameters
tie policy
Git provenance
run output

Reason
A metric without provenance is not sufficient evidence for a research conclusion.
47. Runtime Recovery Acceptance Evidence
Current recovery evidence includes:
artifact contract validation
1811 metadata records
1811 vectors
768 dimensions
IndexFlatIP
exact model revision
321/321 golden retrieval checks
FastAPI SearchService top-20 golden parity
HTTP /search golden parity
runtime regression tests
baseline regression tests

Recent targeted checkpoint:
66 passed

48. Superseded Runtime Decisions
The following describe historical code but are no longer active baseline runtime decisions:
BAAI/bge-m3 runtime embedding
1024-D runtime query vectors
1813-vector production index
regex BM25
title + content + keywords BM25 representation
multi-query expanded retrieval
Gemini intent extraction before retrieval
BGE CrossEncoder baseline reranking
0.6 Dense + 0.4 BM25 fallback
score-only hybrid ordering
health always returning healthy

Keep these only as historical context when needed.
Do not describe them as current runtime behavior.
49. Current Decisions Still Open
The following remain genuinely unresolved.
Research
1. What deterministic held-out or cross-validation scheme should EXP-03 use?
2. Which predeclared objective should select the static fusion weight?
3. Does leakage-controlled tuned static Hybrid beat the frozen 0.5/0.5 baseline?
4. Does dynamic fusion add value beyond tuned static fusion?
5. Does temporal filtering improve legally valid retrieval without harming recall?
6. Does reranking provide reliable gain relative to latency/cost?
7. What manual annotation coverage is sufficient for benchmark confidence?
Runtime / product
8. Should /ask retain automatic answer evaluation or remove/replace it?
9. What authentication and authorization model is required?
10. What deployment topology is intended?
11. Should generated retrieval artifacts be distributed externally or always regenerated?
12. What artifact/version rollout and rollback mechanism is needed?
13. What OCR upload limits and security policy are required?
14. What retention/privacy policy applies to uploaded documents and generated reports?
Scope
15. Should judgments eventually become a searchable precedent corpus?
16. Should future runtime include a learned router?
17. Should richer temporal article-version modeling be built?
These questions remain open and should not be inferred from old code.
50. Current Decision Matrix
Topic	Decision
Retrieval unit	Statutory article
Runtime corpus	Five-law frozen corpus, 1811 articles
Query baseline	Q0
Document baseline	D0
Dense model	keepitreal/vietnamese-sbert
Model revision	Exact pinned revision
Dense dimension	768
Vector index	FAISS IndexFlatIP
BM25 tokenizer	Whitespace
Hybrid fusion	Min-max
Static weights	0.5 / 0.5
Candidate pool	30
Tie-break	score DESC -> standard_law_id ASC
Runtime reranker	None
Runtime temporal filtering	None
Runtime routing	None
Runtime query expansion	None
Primary evaluation	closed-world @10
Core/support weight	3 / 1
Runtime artifacts	Generated, ignored by Git
Artifact validation	Manifest + SHA + structural checks
Golden acceptance	Ranking parity
EXP-02 decision	Complete; D1 not promoted
Next research experiment	EXP-03 static fusion-weight tuning protocol
Human annotation	Pilot only
Database/vector server	Not used
