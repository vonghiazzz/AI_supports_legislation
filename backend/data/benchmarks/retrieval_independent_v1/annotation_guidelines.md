# Independent Retrieval Benchmark v1 — Annotation Guidelines

Status: Phase 1 scaffold. No query or ground-truth record has been created.

Provenance base: Git commit `da6e8fada6c13fb53de5fe8ae1a128de61981532`, generated 2026-10-24. The benchmark manifest records the authoritative artifact SHA-256 values.

## Roles and blinding

- The query writer must not see retrieval outputs, candidate lists, scores, ranks, or metrics.
- Two legal annotators independently assign ground truth from the source judgment and authoritative law.
- Annotators must not see rankings from any candidate-generation or reranking method.
- A third legal adjudicator resolves disagreements without using retrieval output.
- Gemini or another generative model must not generate, infer, validate, or adjudicate ground truth.
- The benchmark custodian keeps independent query and GT content sealed from the method-development team.

## Query construction

- Use a frozen Q0 writing rubric approved before annotation begins.
- Describe the facts, procedural posture, and legal issue needed by the target task.
- Do not add an article or law reference merely because an annotator later assigns it as GT.
- Record the source ID, writer ID, guideline version, and timestamp.
- Do not create paraphrases of any query in the 107-query development benchmark.

## Schema V2 semantics

Every benchmark record retains `ground_truth.core[]` and `ground_truth.supporting[]`.

Each GT item contains:

- `law_code`
- `article`
- `reasoning`
- `importance`
- `importance_score`
- `source_type`
- `domain`

Core and supporting express importance. Substantive and procedural express legal function. These dimensions are independent and must never be substituted for one another.

- Core: `importance=decisive`, `importance_score=3`.
- Supporting: `importance=supporting`, `importance_score=1`.
- Domain: exactly `substantive` or `procedural`.
- Source type: exactly `explicit` or `inferred`.

Preserve raw khoản/điểm references in `article`. Closed-world corpus matching uses the repository's `canonical_retrieval_id()` behavior to map the raw reference to its article-level retrieval unit.

Keep out-of-corpus GT for audit. Only GT mapping to the frozen 1,811-document corpus enters the closed-world denominator.

## Annotation sequence

1. Freeze the selected source list and its fingerprint.
2. Assign anonymized source IDs to the query writer and annotators.
3. Write Q0 without retrieval access.
4. Annotators A and B independently assign labels.
5. Compute disagreement records without exposing retrieval output.
6. Adjudicator C resolves disagreements.
7. Validate Schema V2 and canonical article mapping.
8. Compute annotation agreement and coverage reports.
9. Freeze query/GT JSON, provenance, and hashes.
10. Seal the benchmark until candidate methods and success criteria are preregistered.

## Required provenance

For each query and annotation decision record:

- source ID and source SHA-256;
- writer/annotator/adjudicator pseudonymous IDs;
- guideline and schema versions;
- creation/update timestamps;
- original and adjudicated values;
- reason for adjudication;
- canonical article mapping and in-corpus status.

## Agreement report

Report at minimum:

- exact article-set agreement;
- set Jaccard and set F1;
- core/support agreement after article alignment;
- substantive/procedural agreement after article alignment;
- adjudication count and rate;
- in-corpus and out-of-corpus label counts and coverage.

The query is the statistical evaluation unit. Repeated labels within a query do not create independent samples.

## Prohibited information

Sampling, query-writing, annotation, and adjudication artifacts must not contain BM25, Dense, Hybrid, reranker, candidate, score, rank, or retrieval-metric output.

Once the manifest reaches `SEALED`, query and GT content must not be used for model, tokenizer, representation, fusion, or depth selection. Opening is permitted only after the confirmatory comparison protocol has been preregistered and fingerprinted.
