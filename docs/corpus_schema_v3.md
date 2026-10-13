# Corpus Schema v3 Design Specification

> **Status:** DESIGN / PLANNED ONLY — not the current runtime implementation.
> **Source of truth:** CODE > TEST > CONFIG > DOCUMENTATION > README > COMMENT > ASSUMPTION.


This document defines the Corpus Schema v3 specification for standardizing and structuring legal documents in the Legislation AI RAG system.

---

## 1. Schema Specifications

Corpus Schema v3 extends the existing document structure by introducing rich temporal anchors, cross-document relations, hierarchical structures, and LLM-enriched conceptual tags.

### JSON Schema Definition (`corpus_schema_v3.json`)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Corpus Schema v3",
  "type": "array",
  "items": {
    "type": "object",
    "required": [
      "standard_law_id",
      "law_name",
      "law_code",
      "title",
      "content",
      "temporal_metadata",
      "relations",
      "hierarchical_metadata"
    ],
    "properties": {
      "standard_law_id": { "type": "string" },
      "law_name": { "type": "string" },
      "law_code": { "type": "string" },
      "title": { "type": "string" },
      "content": { "type": "string" },
      "summary": { "type": "string" },
      "keywords": {
        "type": "array",
        "items": { "type": "string" }
      },
      "tags": {
        "type": "array",
        "items": { "type": "string" }
      },
      "temporal_metadata": {
        "type": "object",
        "required": ["effective_year", "expiration_year", "status"],
        "properties": {
          "effective_year": { "type": ["integer", "null"] },
          "expiration_year": { "type": ["integer", "null"] },
          "status": { "type": "string", "enum": ["active", "superseded", "partially_active", "pending"] },
          "amendment_history": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "amended_by": { "type": "string" },
                "amendment_year": { "type": "integer" },
                "description": { "type": "string" }
              }
            }
          }
        }
      },
      "relations": {
        "type": "object",
        "properties": {
          "cross_references": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "target_id": { "type": "string" },
                "relation_type": { "type": "string", "enum": ["modifies", "supersedes", "clarifies", "refers_to"] }
              }
            }
          }
        }
      },
      "hierarchical_metadata": {
        "type": "object",
        "properties": {
          "chapter": { "type": "string" },
          "section": { "type": "string" },
          "part": { "type": "string" }
        }
      }
    }
  }
}
```

---

## 2. Target Files

* **New Schema Definition File**: `backend/data/corpus_schema_v3.json`
* **Modified Data Files**:
  * `backend/data/standardized_bo_luat_dan_su_2015.json`
  * `backend/data/standardized_bo_luat_to_tung_dan_su_2015.json`
  * `backend/data/standardized_luat_dat_dai_2013.json`
  * `backend/data/standardized_luat_dat_dai_2024.json`
  * `backend/data/standardized_luat_hon_nhan_gia_dinh_2014.json`

---

## 3. Migration Plan

### Backward Compatibility
* Legacy pipelines reading Schema v2 must run unimpeded. All fields in Schema v2 (`standard_law_id`, `title`, `law_name`, `content`, `summary`, `keywords`, `tags`) are preserved exactly.
* Default fallback values will be used for temporal and relation metadata if they are missing in incoming datasets during the transition.

### Migration Order
1. Deploy `corpus_schema_v3.json` schema validator.
2. Enrich standard datasets offline to add hierarchical and basic temporal properties.
3. Validate enriched standard datasets using the new schema.
4. Overwrite standard json datasets.
5. Rebuild FAISS indexes (`law_index.faiss`) using the enriched metadata.

### Rollback Strategy
* If validation or indexing fails:
  1. Restore legacy backups (automatically archived via git/backups prior to execution).
  2. Revert configuration pointers to target the backup files.
  3. Rebuild the FAISS index from legacy sources.

---

## 4. Evaluation Plan

### Expected Metric Impact
* **Recall_core**: +5% (due to cross-reference expansion in retrieval query expansion).
* **WeightedRecall**: +4% (owing to better retrieval of core laws).
* **nDCG**: +6% (since hierarchical metadata improves BM25 field boosting).
* **Semantic Drift Reduction**: Substantial reduction in semantic mismatch by filtering out inactive/superseded laws.

### Measurable Success Criteria
* 100% of standard corpus files validate successfully against `corpus_schema_v3.json`.
* Rebuilt FAISS index successfully loads all enriched items.

---

## 5. Acceptance Gate: P3-G1 Corpus Schema

* **PASS Criteria**:
  * Every article in the corpus successfully parses into the new schema structure.
  * Validation rate is exactly 100.0%.
* **FAIL Criteria**:
  * Any record is missing required `temporal_metadata` or `relations` keys.
  * Schema validation yields any exceptions.
* **Required Evidence**:
  * Output log of the corpus validation script displaying validation success.
