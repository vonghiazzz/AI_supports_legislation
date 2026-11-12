"""Pure R8 scaffold for a versioned source-review manifest migration.

The module deliberately cannot publish an acquisition manifest, perform a
source review, or promote eligibility.  It validates and transforms in-memory
copies, audits already-frozen local evidence, and writes only an isolated run
report when explicitly invoked.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import dedup_policy_v2 as policy_v2
from . import dedup_policy_v2_independent as r6
from . import dedup_policy_v2_safety as r7
from . import source_intake as v1


PROTOCOL_ID = "independent_batch2dc_versioned_manifest_migration_v1"
PARENT_GIT_SHA = "ef08da1ec82ef921a41fa9d9035ca0adea12ddc8"
SOURCE_MANIFEST_SHA256 = r6.MANIFEST_SHA256
POLICY_FINGERPRINT = (
    "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"
)
R7_RUN_FINGERPRINT = (
    "49487e1a02de084b2df57c37fddc2970c0e0713db7f46d7bfa26318573ac72a5"
)
R7_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_v2_targeted_adjudication_contract_safety_v1_"
    "20261111_155737_ef08da1e"
)
R7_TARGET_ARTIFACT = "targeted_dispute_adjudication.json"

V2_MANIFEST_VERSION = "independent_source_acquisition_v2_review_records"
REVIEW_COLLECTION_VERSION = "independent_source_review_collection_v2"
REVIEW_RECORD_VERSION = "independent_source_review_record_v2_evidence_only"
TRUSTED_ATTESTATION_VERSION = "independent_source_review_attestation_v1"
SCAFFOLD_STATUS = "SCAFFOLD_NOT_ACTIVATED"
POLICY_BINDING = {
    "policy_id": policy_v2.POLICY_ID,
    "policy_version": policy_v2.POLICY_VERSION,
    "policy_fingerprint": POLICY_FINGERPRINT,
}
V1_ROOT_FIELDS = {
    "manifest_version",
    "created_at",
    "sources",
    "immutable_provenance",
    "known_duplicate_links",
}
V2_ROOT_FIELDS = V1_ROOT_FIELDS | {"review_schema", "review_records"}
REVIEW_SCHEMA_FIELDS = {
    "schema_version",
    "source_manifest_version",
    "source_manifest_sha256",
    "review_record_version",
    "policy_binding_mode",
    "activation_status",
    "migration_protocol_id",
}
REVIEW_RECORD_FIELDS = {
    "review_record_version",
    "source_id",
    "acquisition_id",
    "policy_binding",
    "source_binding",
    "evidence_binding",
    "findings",
    "review_state",
    "authority_attestation",
    "eligibility_authorization",
    "record_fingerprint",
}
SOURCE_BINDING_FIELDS = {
    "source_manifest_sha256",
    "raw_sha256",
    "structured_sha256",
    "identity_fingerprint",
    "source_snapshot_fingerprint",
}
EVIDENCE_BINDING_FIELDS = {
    "complete_inventory_fingerprint",
    "candidate_set_fingerprint",
    "evidence_fingerprint",
    "pair_binding_fingerprints",
}
FINDINGS_FIELDS = {
    "canonical_disposition",
    "blocking_rejection_reasons",
    "nonblocking_findings",
}
DOCUMENT_TYPES = {"BAN_AN", "QUYET_DINH", "OTHER_SUPPORTED", "UNKNOWN"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SourceReviewMigrationError(ValueError):
    """A pure migration, evidence, or binding invariant failed."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise SourceReviewMigrationError(f"{label} must be lowercase SHA256")


def _identity_fingerprint(source: Mapping[str, Any]) -> str:
    return policy_v2.canonical_fingerprint(
        {
            field: source[field]
            for field in (
                "case_id",
                "judgment_number",
                "court",
                "decision_date",
                "procedure_stage",
                "underlying_dispute_id",
                "related_decision_ids",
            )
        }
    )


def _review_schema(source_manifest_sha256: str) -> dict:
    _require_sha256(source_manifest_sha256, "source_manifest_sha256")
    return {
        "schema_version": REVIEW_COLLECTION_VERSION,
        "source_manifest_version": v1.MANIFEST_VERSION,
        "source_manifest_sha256": source_manifest_sha256,
        "review_record_version": REVIEW_RECORD_VERSION,
        "policy_binding_mode": "EXPLICIT_PER_RECORD_NO_DEFAULT",
        "activation_status": SCAFFOLD_STATUS,
        "migration_protocol_id": PROTOCOL_ID,
    }


def project_v2_manifest_to_v1(manifest: Mapping[str, Any]) -> dict:
    """Project a V2 copy back to the exact historical V1 shape."""

    if not isinstance(manifest, Mapping) or set(manifest) != V2_ROOT_FIELDS:
        raise SourceReviewMigrationError("V2 manifest root fields are invalid")
    projected = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key not in {"review_schema", "review_records"}
    }
    projected["manifest_version"] = v1.MANIFEST_VERSION
    v1.validate_acquisition_manifest(projected)
    return projected


def _expected_source_binding(source: Mapping[str, Any], manifest_sha256: str) -> dict:
    return {
        "source_manifest_sha256": manifest_sha256,
        "raw_sha256": source["sha256"],
        "structured_sha256": source["structured_sha256"],
        "identity_fingerprint": _identity_fingerprint(source),
        "source_snapshot_fingerprint": policy_v2.canonical_fingerprint(source),
    }


def validate_review_record_v2(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    source_manifest_sha256: str,
) -> dict:
    """Validate an evidence-only V2 record; authority cannot be caller asserted."""

    if not isinstance(record, Mapping) or set(record) != REVIEW_RECORD_FIELDS:
        raise SourceReviewMigrationError("V2 review record fields are invalid")
    checked = copy.deepcopy(dict(record))
    if checked["review_record_version"] != REVIEW_RECORD_VERSION:
        raise SourceReviewMigrationError("Unsupported V2 review record version")
    if (
        checked["source_id"] != source["source_id"]
        or checked["acquisition_id"] != source["acquisition_id"]
    ):
        raise SourceReviewMigrationError("V2 review record source identity mismatch")
    if source["acquisition_status"] not in {
        "extracted",
        "metadata_verified",
        "dedup_checked",
        "eligible",
    } or not isinstance(source["structured_sha256"], str):
        raise SourceReviewMigrationError("V2 review record requires extracted source evidence")
    if checked["policy_binding"] != POLICY_BINDING:
        raise SourceReviewMigrationError("Unknown or mismatched V2 policy binding")
    binding = checked["source_binding"]
    if not isinstance(binding, dict) or set(binding) != SOURCE_BINDING_FIELDS:
        raise SourceReviewMigrationError("V2 source binding fields are invalid")
    if binding != _expected_source_binding(source, source_manifest_sha256):
        raise SourceReviewMigrationError("Stale or mismatched V2 source binding")
    evidence = checked["evidence_binding"]
    if not isinstance(evidence, dict) or set(evidence) != EVIDENCE_BINDING_FIELDS:
        raise SourceReviewMigrationError("V2 evidence binding fields are invalid")
    for field in (
        "complete_inventory_fingerprint",
        "candidate_set_fingerprint",
        "evidence_fingerprint",
    ):
        _require_sha256(evidence[field], f"evidence_binding.{field}")
    pair_fingerprints = evidence["pair_binding_fingerprints"]
    if (
        not isinstance(pair_fingerprints, list)
        or pair_fingerprints != sorted(set(pair_fingerprints))
    ):
        raise SourceReviewMigrationError("Pair bindings must be sorted and unique")
    for value in pair_fingerprints:
        _require_sha256(value, "pair_binding_fingerprint")
    findings = checked["findings"]
    if not isinstance(findings, dict) or set(findings) != FINDINGS_FIELDS:
        raise SourceReviewMigrationError("V2 findings fields are invalid")
    if findings["canonical_disposition"] not in {
        None,
        "CANONICAL_RETAINED",
        "NONCANONICAL_REJECTED",
    }:
        raise SourceReviewMigrationError("V2 canonical disposition is invalid")
    for field in ("blocking_rejection_reasons", "nonblocking_findings"):
        values = findings[field]
        if (
            not isinstance(values, list)
            or values != sorted(set(values))
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise SourceReviewMigrationError(f"V2 {field} must be sorted unique strings")
    if not set(source["rejection_reasons"]) <= set(
        findings["blocking_rejection_reasons"]
    ):
        raise SourceReviewMigrationError("Existing rejection reasons cannot be cleared")
    if checked["review_state"] != "EVIDENCE_ONLY_NOT_AUTHORITATIVE":
        raise SourceReviewMigrationError("R8 cannot issue authoritative review state")
    if checked["authority_attestation"] is not None:
        raise SourceReviewMigrationError(
            "Caller-controlled authority is forbidden; trusted attestation required later"
        )
    if checked["eligibility_authorization"] != {
        "authorized": False,
        "reason": "SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED",
    }:
        raise SourceReviewMigrationError("Review metadata cannot grant eligibility")
    expected_fingerprint = policy_v2.canonical_fingerprint(
        checked, excluded_key="record_fingerprint"
    )
    if checked["record_fingerprint"] != expected_fingerprint:
        raise SourceReviewMigrationError("V2 review record fingerprint mismatch")
    return checked


def build_evidence_only_review_record(
    source: Mapping[str, Any],
    source_manifest_sha256: str,
    *,
    complete_inventory_fingerprint: str,
    candidate_set_fingerprint: str,
    evidence_fingerprint: str,
    pair_binding_fingerprints: Sequence[str],
    canonical_disposition: str | None = None,
    blocking_rejection_reasons: Sequence[str] = (),
    nonblocking_findings: Sequence[str] = (),
) -> dict:
    """Build a non-authoritative record for isolated fixtures and dry runs."""

    record = {
        "review_record_version": REVIEW_RECORD_VERSION,
        "source_id": source["source_id"],
        "acquisition_id": source["acquisition_id"],
        "policy_binding": copy.deepcopy(POLICY_BINDING),
        "source_binding": _expected_source_binding(source, source_manifest_sha256),
        "evidence_binding": {
            "complete_inventory_fingerprint": complete_inventory_fingerprint,
            "candidate_set_fingerprint": candidate_set_fingerprint,
            "evidence_fingerprint": evidence_fingerprint,
            "pair_binding_fingerprints": sorted(set(pair_binding_fingerprints)),
        },
        "findings": {
            "canonical_disposition": canonical_disposition,
            "blocking_rejection_reasons": sorted(
                set(blocking_rejection_reasons) | set(source["rejection_reasons"])
            ),
            "nonblocking_findings": sorted(set(nonblocking_findings)),
        },
        "review_state": "EVIDENCE_ONLY_NOT_AUTHORITATIVE",
        "authority_attestation": None,
        "eligibility_authorization": {
            "authorized": False,
            "reason": "SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED",
        },
        "record_fingerprint": "PENDING",
    }
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    return validate_review_record_v2(record, source, source_manifest_sha256)


def validate_versioned_manifest(manifest: Mapping[str, Any]) -> dict:
    """Validate historical V1 or the inactive, exact-schema V2 scaffold."""

    if not isinstance(manifest, Mapping):
        raise SourceReviewMigrationError("Manifest must be an object")
    version = manifest.get("manifest_version")
    if version == v1.MANIFEST_VERSION:
        v1.validate_acquisition_manifest(dict(manifest))
        return copy.deepcopy(dict(manifest))
    if version != V2_MANIFEST_VERSION:
        raise SourceReviewMigrationError("Unsupported manifest schema version")
    if set(manifest) != V2_ROOT_FIELDS:
        raise SourceReviewMigrationError("V2 manifest root fields are invalid")
    schema = manifest["review_schema"]
    if not isinstance(schema, dict) or set(schema) != REVIEW_SCHEMA_FIELDS:
        raise SourceReviewMigrationError("V2 review schema fields are invalid")
    projected = project_v2_manifest_to_v1(manifest)
    projected_sha = sha256_bytes(canonical_json_bytes(projected))
    if schema != _review_schema(projected_sha):
        raise SourceReviewMigrationError("V2 review schema binding mismatch")
    records = manifest["review_records"]
    if not isinstance(records, list):
        raise SourceReviewMigrationError("V2 review_records must be an array")
    by_source = {source["source_id"]: source for source in projected["sources"]}
    seen = set()
    for record in records:
        source_id = record.get("source_id") if isinstance(record, Mapping) else None
        if source_id not in by_source or source_id in seen:
            raise SourceReviewMigrationError("V2 review record source is unknown or duplicated")
        validate_review_record_v2(record, by_source[source_id], projected_sha)
        seen.add(source_id)
    return copy.deepcopy(dict(manifest))


def migrate_manifest_v1_to_v2_copy(manifest: Mapping[str, Any]) -> dict:
    """Create and validate a deterministic V2 copy without writing any file."""

    if not isinstance(manifest, Mapping):
        raise SourceReviewMigrationError("Historical manifest must be an object")
    historical = copy.deepcopy(dict(manifest))
    v1.validate_acquisition_manifest(historical)
    source_sha = sha256_bytes(canonical_json_bytes(historical))
    migrated = copy.deepcopy(historical)
    migrated["manifest_version"] = V2_MANIFEST_VERSION
    migrated["review_schema"] = _review_schema(source_sha)
    migrated["review_records"] = []
    return validate_versioned_manifest(migrated)


def migration_plan(manifest: Mapping[str, Any]) -> dict:
    """Return a deterministic plan whose fingerprint excludes no live state."""

    v1.validate_acquisition_manifest(dict(manifest))
    plan = {
        "protocol_id": PROTOCOL_ID,
        "from_version": v1.MANIFEST_VERSION,
        "to_version": V2_MANIFEST_VERSION,
        "selected_design": "SEPARATE_VERSIONED_REVIEW_RECORD_COLLECTION",
        "source_manifest_sha256": sha256_bytes(canonical_json_bytes(manifest)),
        "operations": [
            "deep_copy_historical_manifest",
            "set_explicit_v2_manifest_version",
            "add_inactive_exact_review_schema",
            "add_empty_review_record_collection",
            "validate_v1_projection",
            "verify_source_provenance_identity_status_history_equality",
        ],
        "live_write_authorized": False,
        "v2_activation_authorized": False,
        "promotion_authorized": False,
    }
    plan["migration_plan_fingerprint"] = policy_v2.canonical_fingerprint(plan)
    return plan


def migration_dry_run(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    path = root / r6.MANIFEST_RELATIVE
    before_bytes = path.read_bytes()
    before_sha = sha256_bytes(before_bytes)
    if before_sha != SOURCE_MANIFEST_SHA256:
        raise SourceReviewMigrationError("Live V1 manifest binding changed")
    historical = json.loads(before_bytes.decode("utf-8"))
    v1.validate_acquisition_manifest(historical)
    migrated = migrate_manifest_v1_to_v2_copy(historical)
    projected = project_v2_manifest_to_v1(migrated)
    after_bytes = path.read_bytes()
    if after_bytes != before_bytes:
        raise SourceReviewMigrationError("Live manifest changed during dry run")
    before_by_id = {item["acquisition_id"]: item for item in historical["sources"]}
    after_by_id = {item["acquisition_id"]: item for item in projected["sources"]}
    identity_fields = {
        "acquisition_id",
        "source_id",
        "case_id",
        "judgment_number",
        "court",
        "decision_date",
        "procedure_stage",
        "underlying_dispute_id",
        "related_decision_ids",
    }
    result = {
        "protocol_id": PROTOCOL_ID,
        "input_mode": "READ_ONLY_LIVE_INPUT_DEEP_COPIED_BEFORE_TRANSFORMATION",
        "input_manifest_sha256": before_sha,
        "input_manifest_version": historical["manifest_version"],
        "migrated_manifest_version": migrated["manifest_version"],
        "source_count": len(historical["sources"]),
        "review_record_count": len(migrated["review_records"]),
        "v1_projection_byte_equal": canonical_json_bytes(projected) == before_bytes,
        "v1_projection_semantic_equal": projected == historical,
        "immutable_provenance_equal": (
            projected["immutable_provenance"] == historical["immutable_provenance"]
        ),
        "identity_change_count": sum(
            any(before_by_id[key][field] != after_by_id[key][field] for field in identity_fields)
            for key in before_by_id
        ),
        "status_change_count": sum(
            before_by_id[key]["acquisition_status"]
            != after_by_id[key]["acquisition_status"]
            for key in before_by_id
        ),
        "history_change_count": sum(
            before_by_id[key]["status_history"] != after_by_id[key]["status_history"]
            for key in before_by_id
        ),
        "v1_review_semantics_change_count": sum(
            (
                before_by_id[key]["metadata_verification"],
                before_by_id[key]["rejection_reasons"],
            )
            != (
                after_by_id[key]["metadata_verification"],
                after_by_id[key]["rejection_reasons"],
            )
            for key in before_by_id
        ),
        "eligibility_change_count": sum(
            (before_by_id[key]["acquisition_status"] == "eligible")
            != (after_by_id[key]["acquisition_status"] == "eligible")
            for key in before_by_id
        ),
        "acquisition_id_change_count": len(set(before_by_id) ^ set(after_by_id)),
        "live_manifest_sha256_after": sha256_bytes(after_bytes),
        "live_manifest_mutated": after_bytes != before_bytes,
        "migration_plan": migration_plan(historical),
    }
    invariant_fields = (
        "v1_projection_byte_equal",
        "v1_projection_semantic_equal",
        "immutable_provenance_equal",
    )
    zero_fields = (
        "identity_change_count",
        "status_change_count",
        "history_change_count",
        "v1_review_semantics_change_count",
        "eligibility_change_count",
        "acquisition_id_change_count",
    )
    result["result"] = (
        "PASS"
        if all(result[field] is True for field in invariant_fields)
        and all(result[field] == 0 for field in zero_fields)
        and result["live_manifest_mutated"] is False
        else "FAIL"
    )
    if result["result"] != "PASS":
        raise SourceReviewMigrationError("Migration rehearsal did not preserve V1")
    result["artifact_fingerprint"] = policy_v2.canonical_fingerprint(result)
    return result


def _ascii_with_offsets(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(text):
        replacement = "D" if char in "Đđ" else char
        normalized = "".join(
            item
            for item in unicodedata.normalize("NFD", replacement)
            if unicodedata.category(item) != "Mn"
        ).upper()
        for item in normalized:
            chars.append(item)
            offsets.append(index)
    return "".join(chars), offsets


def classify_document_type_from_header(page_text: str) -> dict:
    """Classify only explicit first-page header markers with mapped offsets."""

    if not isinstance(page_text, str):
        raise SourceReviewMigrationError("Header page text must be a string")
    header = page_text[:1500]
    ascii_text, offsets = _ascii_with_offsets(header)
    patterns = {
        "BAN_AN": re.compile(r"\bBAN\s+AN\s+SO\s*:?"),
        # A bare occurrence of "quyết định" is not enough: a judgment header
        # may describe a request to cancel an administrative decision.  These
        # explicit title continuations are present in the frozen evidence and
        # keep the classifier fail-closed for other document types.
        "QUYET_DINH": re.compile(
            r"\bQUYET\s+DINH[\W_O]{0,20}(?:"
            r"GIAM\s+DOC\s+THAM|"
            r"DINH\s+CHI|"
            r"CONG\s+NHAN"
            r")\b"
        ),
    }
    matches = []
    for document_type, pattern in patterns.items():
        match = pattern.search(ascii_text)
        if match is None:
            continue
        start = offsets[match.start()]
        end = offsets[match.end() - 1] + 1
        matches.append((start, end, document_type))
    matches.sort(key=lambda item: (item[0], item[2]))
    if not matches:
        return {
            "document_type": "UNKNOWN",
            "evidence_supported": False,
            "page": 1,
            "character_start": None,
            "character_end": None,
            "exact_marker": None,
            "evidence_excerpt": None,
            "conflicting_evidence": [],
        }
    start, end, selected = matches[0]
    conflicting = [
        {
            "document_type": document_type,
            "character_start": other_start,
            "character_end": other_end,
            "exact_marker": header[other_start:other_end],
        }
        for other_start, other_end, document_type in matches[1:]
        if document_type != selected and other_start < 500
    ]
    if conflicting:
        selected = "UNKNOWN"
    excerpt_start = max(0, start - 80)
    excerpt_end = min(len(header), end + 140)
    return {
        "document_type": selected,
        "evidence_supported": selected != "UNKNOWN",
        "page": 1,
        "character_start": start,
        "character_end": end,
        "exact_marker": header[start:end],
        "evidence_excerpt": " ".join(header[excerpt_start:excerpt_end].split()),
        "conflicting_evidence": conflicting,
    }


def _current_r6_context(project_root: Path) -> dict:
    records, context = r6._load_inventory(project_root)
    targets = {
        item["source_id"]
        for item in records
        if isinstance(item["acquisition_id"], str)
        and item["acquisition_id"].startswith("batch2dc-20261105-")
    }
    if len(targets) != 46:
        raise SourceReviewMigrationError("Batch2DC target set changed")
    generated = r6.generate_candidates(records, targets)
    evaluated = r6.evaluate_candidates(context["policy"], generated)
    counterparts = {
        source_id
        for item in generated["pairs"]
        for source_id in (
            item["evidence"]["left_source_id"],
            item["evidence"]["right_source_id"],
        )
        if source_id not in targets
    }
    return {
        "records": records,
        "context": context,
        "targets": targets,
        "generated": generated,
        "evaluated": evaluated,
        "counterparts": counterparts,
    }


def document_type_evidence_audit(project_root: Path, current: Mapping[str, Any]) -> dict:
    root = Path(project_root).resolve()
    selected_ids = sorted(current["targets"] | current["counterparts"])
    by_id = {item["source_id"]: item for item in current["records"]}
    rows = []
    for source_id in selected_ids:
        record = by_id[source_id]
        raw_path = root / record["source_path"]
        structured_path = root / record["structured_path"]
        if raw_path.is_symlink() or structured_path.is_symlink():
            raise SourceReviewMigrationError("Symlink document-type evidence is forbidden")
        actual_raw_sha = policy_v2.sha256_file(raw_path)
        if actual_raw_sha != record["raw_sha256"]:
            raise SourceReviewMigrationError("Raw document-type evidence changed")
        actual_sha = policy_v2.sha256_file(structured_path)
        if actual_sha != record["structured_sha256"]:
            raise SourceReviewMigrationError("Structured document-type evidence changed")
        payload = json.loads(structured_path.read_text(encoding="utf-8"))
        v1.validate_structured_judgment(payload, record["source_name"])
        evidence = classify_document_type_from_header(payload["pages"][0]["text"])
        rows.append(
            {
                "source_id": source_id,
                "acquisition_id": record["acquisition_id"],
                "population": record["population"],
                "document_type": evidence["document_type"],
                "evidence_supported": evidence["evidence_supported"],
                "evidence_authority": "VERIFIED_STRUCTURED_EXTRACTION_OF_OFFICIAL_PDF",
                "structured_path": record["structured_path"],
                "structured_sha256": actual_sha,
                "raw_path": record["source_path"],
                "raw_sha256": actual_raw_sha,
                "evidence_location": {
                    key: evidence[key]
                    for key in (
                        "page",
                        "character_start",
                        "character_end",
                        "exact_marker",
                        "evidence_excerpt",
                    )
                },
                "conflicting_evidence": evidence["conflicting_evidence"],
                "procedure_stage_separate_verified_field": record["procedure_stage"],
                "case_domain": None,
                "case_domain_status": "NOT_AUDITED_AND_NOT_INFERRED_FROM_DOCUMENT_TYPE",
                "filename_or_number_used_as_authority": False,
            }
        )
    doc_types = {item["source_id"]: item["document_type"] for item in rows}
    relation_coverage = []
    for item in current["evaluated"]:
        left = item["evidence"]["left_source_id"]
        right = item["evidence"]["right_source_id"]
        relation_coverage.append(
            {
                "pair_id": item["pair_id"],
                "left_source_id": left,
                "right_source_id": right,
                "left_document_type": doc_types[left],
                "right_document_type": doc_types[right],
                "document_type_equal": (
                    None
                    if "UNKNOWN" in {doc_types[left], doc_types[right]}
                    else doc_types[left] == doc_types[right]
                ),
            }
        )
    target_rows = [item for item in rows if item["source_id"] in current["targets"]]
    counterpart_rows = [
        item for item in rows if item["source_id"] in current["counterparts"]
    ]
    output = {
        "protocol_id": PROTOCOL_ID,
        "scope": "46_BATCH2DC_PLUS_CURRENT_V2_CANDIDATE_COUNTERPARTS",
        "batch2dc_count": len(target_rows),
        "comparison_counterpart_count": len(counterpart_rows),
        "candidate_pair_count": len(relation_coverage),
        "authoritatively_supported_count": sum(
            item["evidence_supported"] for item in rows
        ),
        "unknown_count": sum(not item["evidence_supported"] for item in rows),
        "conflict_count": sum(bool(item["conflicting_evidence"]) for item in rows),
        "normalized_value_counts": {
            value: sum(item["document_type"] == value for item in rows)
            for value in sorted(DOCUMENT_TYPES)
        },
        "sources": rows,
        "candidate_relations": relation_coverage,
        "frozen_policy_semantics": {
            "document_type_is_contextual_field": True,
            "minimum_verified_contextual_conflicts": 1,
            "every_contextual_field_required_for_every_relation": False,
            "r7_universal_document_type_completion_requirement_matches_policy": False,
            "future_versioned_contract_amendment_required": True,
            "frozen_policy_modified": False,
            "amendment": (
                "Evidence completeness must be relation-specific: require enough "
                "verified evidence for the invoked rule, not every optional contextual field."
            ),
        },
        "live_source_metadata_mutated": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def targeted_adjudication_bindings(project_root: Path, current: Mapping[str, Any]) -> dict:
    root = Path(project_root).resolve()
    run_dir = root / R7_RUN_RELATIVE
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("run_fingerprint") != R7_RUN_FINGERPRINT:
        raise SourceReviewMigrationError("R7 run fingerprint changed")
    if policy_v2.canonical_fingerprint(
        run_manifest, excluded_key="run_fingerprint"
    ) != R7_RUN_FINGERPRINT:
        raise SourceReviewMigrationError("R7 run manifest is internally inconsistent")
    targeted_path = run_dir / R7_TARGET_ARTIFACT
    targeted_sha = policy_v2.sha256_file(targeted_path)
    if run_manifest["artifact_sha256"].get(R7_TARGET_ARTIFACT) != targeted_sha:
        raise SourceReviewMigrationError("R7 targeted artifact SHA changed")
    targeted = json.loads(targeted_path.read_text(encoding="utf-8"))
    if targeted.get("artifact_fingerprint") != policy_v2.canonical_fingerprint(
        targeted, excluded_key="artifact_fingerprint"
    ):
        raise SourceReviewMigrationError("R7 targeted artifact fingerprint mismatch")
    current_pairs = {
        tuple(
            sorted(
                (
                    item["evidence"]["left_source_id"],
                    item["evidence"]["right_source_id"],
                )
            )
        ): item
        for item in current["evaluated"]
    }
    bindings = []
    for adjudication in targeted["adjudications"]:
        left = adjudication["left"]
        right = adjudication["right"]
        pair = tuple(sorted((left["source_id"], right["source_id"])))
        if pair not in current_pairs:
            raise SourceReviewMigrationError("R7 adjudication pair is no longer a candidate")
        for summary in (left, right):
            if policy_v2.sha256_file(root / summary["raw_pdf"]["path"]) != summary[
                "raw_pdf"
            ]["sha256"]:
                raise SourceReviewMigrationError("R7 raw adjudication evidence changed")
            if policy_v2.sha256_file(root / summary["structured"]["path"]) != summary[
                "structured"
            ]["sha256"]:
                raise SourceReviewMigrationError("R7 structured adjudication evidence changed")
        binding = {
            "left_source_id": pair[0],
            "right_source_id": pair[1],
            "left_raw_sha256": next(
                item["raw_pdf"]["sha256"]
                for item in (left, right)
                if item["source_id"] == pair[0]
            ),
            "right_raw_sha256": next(
                item["raw_pdf"]["sha256"]
                for item in (left, right)
                if item["source_id"] == pair[1]
            ),
            "left_structured_sha256": next(
                item["structured"]["sha256"]
                for item in (left, right)
                if item["source_id"] == pair[0]
            ),
            "right_structured_sha256": next(
                item["structured"]["sha256"]
                for item in (left, right)
                if item["source_id"] == pair[1]
            ),
            "r7_adjudication_fingerprint": policy_v2.canonical_fingerprint(adjudication),
            "current_candidate_evidence_fingerprint": policy_v2.canonical_fingerprint(
                current_pairs[pair]["evidence"]
            ),
            "classification": adjudication["classification"],
            "same_legal_document": adjudication["same_legal_document"],
            "same_underlying_dispute": adjudication["same_underlying_dispute"],
            "reuse_guard": (
                "EXACT_PAIR_AND_RAW_STRUCTURED_HASHES_AND_CURRENT_COMPLETE_"
                "CANDIDATE_RECOMPUTATION_REQUIRED"
            ),
        }
        binding["pair_binding_fingerprint"] = policy_v2.canonical_fingerprint(binding)
        bindings.append(binding)
    expected_pairs = {
        tuple(sorted((left, right))) for left, right, _ in r7.TARGET_PAIRS
    }
    if {tuple((item["left_source_id"], item["right_source_id"])) for item in bindings} != expected_pairs:
        raise SourceReviewMigrationError("R7 targeted pair set changed")
    output = {
        "protocol_id": PROTOCOL_ID,
        "r7_run_path": R7_RUN_RELATIVE.as_posix(),
        "r7_run_fingerprint": R7_RUN_FINGERPRINT,
        "r7_target_artifact": R7_TARGET_ARTIFACT,
        "r7_target_artifact_sha256": targeted_sha,
        "r7_target_artifact_fingerprint": targeted["artifact_fingerprint"],
        "bindings": sorted(
            bindings, key=lambda item: (item["left_source_id"], item["right_source_id"])
        ),
        "complete_candidate_set_fingerprint": policy_v2.canonical_fingerprint(
            current["evaluated"]
        ),
        "unresolved_disputes_retained": sorted(r7.ORIGINAL_UNRESOLVED),
        "analytical_adjudication_became_authoritative_review": False,
        "live_review_applied": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def schema_compatibility_matrix() -> dict:
    rows = [
        ("registration", "writes exact V1 root/source records and provenance anchors", "V1 unchanged; future writer must explicitly select schema"),
        ("extraction", "updates structured path/hash/status/history under V1 validator and journal", "V1 unchanged; future V2 writer must preserve review collection atomically"),
        ("identity", "updates only verified legal identity fields under mutation lock", "review records become stale when source binding changes"),
        ("review", "record_source_review writes V1 booleans/rejections only", "must remain V1-only until separately authorized V2 producer exists"),
        ("promotion", "promote_source_to_eligible invokes V1 gates and atomic replacement", "future version selector must reject unreviewed/stale/unknown policy"),
        ("benchmark_inventory", "authoritative load validates V1, anchors, artifacts and status", "must use V1 projection or version-aware reader"),
        ("deduplication", "V1 eligible inventory; frozen V2 is pure shadow evaluation", "cross-version complete inventory recomputation remains mandatory"),
        ("historical_tests", "assert exact V1 root/source fields and transitions", "must remain unchanged and passing"),
        ("provenance_anchors", "one immutable external anchor per source", "unchanged because review records do not alter acquisition provenance"),
        ("transaction_recovery", "journals/locks protect V1 registration/extraction/review", "future V2 publication requires a new authorized atomic/recovery design"),
    ]
    output = {
        "protocol_id": PROTOCOL_ID,
        "v1_exact_root_fields": sorted(V1_ROOT_FIELDS),
        "v1_exact_source_fields": sorted(v1.SOURCE_REQUIRED_FIELDS),
        "selected_design": "B_SEPARATE_VERSIONED_REVIEW_RECORD_COLLECTION",
        "options": {
            "A_INLINE_SOURCE_FIELDS": {
                "selected": False,
                "risk": "breaks exact source schema and touches every source reader/writer",
            },
            "B_ROOT_REVIEW_COLLECTION": {
                "selected": True,
                "reason": "smallest one-manifest extension with exact V1 projection and no source/provenance mutation",
            },
            "C_SIDECAR_LEDGER": {
                "selected": False,
                "risk": "adds cross-file transaction, stale-reference and half-publication states",
            },
        },
        "dependencies": [
            {"dependency": name, "v1_contract": current, "future_v2_impact": future}
            for name, current, future in rows
        ],
        "production_v1_functions_modified": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def schema_spec() -> dict:
    output = {
        "protocol_id": PROTOCOL_ID,
        "manifest_versions": {
            v1.MANIFEST_VERSION: "HISTORICAL_EXACT_SCHEMA_ACTIVE",
            V2_MANIFEST_VERSION: SCAFFOLD_STATUS,
        },
        "v2_root_fields": sorted(V2_ROOT_FIELDS),
        "review_schema_fields": sorted(REVIEW_SCHEMA_FIELDS),
        "review_record_fields": sorted(REVIEW_RECORD_FIELDS),
        "review_record_version": REVIEW_RECORD_VERSION,
        "policy_binding": POLICY_BINDING,
        "implicit_v2_default": False,
        "unknown_versions_fail_closed": True,
        "identity_and_acquisition_provenance_immutable": True,
        "historical_status_transitions_preserved": True,
        "review_metadata_can_grant_eligibility": False,
        "cross_version_checks_mandatory": True,
        "actual_disk_migration_authorized": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def review_authority_trust_boundary() -> dict:
    output = {
        "protocol_id": PROTOCOL_ID,
        "current_r7_string_is_authority": False,
        "caller_role": "SUPPLY_EVIDENCE_AND_REQUEST_ONLY",
        "trusted_producer_requirements": [
            "repository-controlled committed producer code",
            "clean validated HEAD and exact producer code hash",
            "frozen policy fingerprint validation",
            "authoritative manifest, source, raw and structured hash validation",
            "complete current cross-version candidate recomputation",
            "relation-specific evidence sufficiency and explicit adjudication",
            "preservation of existing rejection reasons",
            "immutable signed-or-create-once attestation publication",
        ],
        "attestation_required_fields": [
            "attestation_version",
            "producer_id",
            "producer_version",
            "execution_git_sha",
            "producer_code_sha256",
            "policy_fingerprint",
            "source_binding_fingerprint",
            "complete_inventory_fingerprint",
            "candidate_set_fingerprint",
            "evidence_fingerprint",
            "decision_fingerprint",
            "issued_at",
            "attestation_sha256",
        ],
        "attestation_version": TRUSTED_ATTESTATION_VERSION,
        "caller_controlled_authority_field_accepted": False,
        "stale_source_evidence_policy_or_inventory_invalidates_readiness": True,
        "rejection_reasons_may_be_silently_cleared": False,
        "trusted_context_activated": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def eligibility_gate_design() -> dict:
    output = {
        "protocol_id": PROTOCOL_ID,
        "selection": [
            {
                "record": "existing V1-reviewed source with no V2 record",
                "gate": "V1",
            },
            {
                "record": "valid trusted V2 review record with exact bindings",
                "gate": "V2_FUTURE",
            },
            {
                "record": "unknown/missing/mismatched policy or unreviewed source",
                "gate": "REJECT",
            },
        ],
        "future_promotion_order": [
            "load authoritative manifest under mutation synchronization",
            "validate schema, policy and trusted review attestation",
            "verify source/raw/structured/evidence integrity",
            "recompute complete current candidate set and V2 findings",
            "check development overlap and all V1/V2 independent sources",
            "check underlying-dispute leakage and canonical disposition",
            "preserve every unresolved existing rejection reason",
            "reject stale or incomplete authority",
            "require separate promotion authorization",
        ],
        "canonical_duplicate_controls": {
            "retained": "independent:batch2dc-20261105-025",
            "noncanonical_rejected": "independent:batch2dc-20261105-026",
        },
        "unresolved_disputes": sorted(r7.ORIGINAL_UNRESOLVED),
        "promotion_executed": False,
        "v2_gate_activated": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def _protected_state(project_root: Path, manifest: Mapping[str, Any]) -> dict:
    root = Path(project_root).resolve()
    benchmark = json.loads(
        (root / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    sampling = json.loads(
        (root / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    counts = {
        status: sum(item["acquisition_status"] == status for item in manifest["sources"])
        for status in v1.ACQUISITION_STATUSES
    }
    batch = [
        item
        for item in manifest["sources"]
        if item["acquisition_id"].startswith("batch2dc-20261105-")
    ]
    identity_ready = sum(not v1._source_identity_failures(item) for item in batch)
    state = {
        "global": {
            "total": len(manifest["sources"]),
            "eligible": counts["eligible"],
            "extracted": counts["extracted"],
            "hash_verified": counts["hash_verified"],
        },
        "batch2dc": {
            "total": len(batch),
            "extracted": sum(item["acquisition_status"] == "extracted" for item in batch),
            "identity_ready": identity_ready,
            "eligible": sum(item["acquisition_status"] == "eligible" for item in batch),
        },
        "benchmark_state": benchmark["state"],
        "sampling_state": sampling["state"],
        "selection_status": sampling["selection_status"],
        "sampling_seed": sampling["sampling_seed"],
        "selected_cases": benchmark["current_counts"]["selected_cases"],
        "query_records": benchmark["current_counts"]["query_records"],
        "ground_truth_records": benchmark["current_counts"]["ground_truth_records"],
    }
    expected = {
        "global": {"total": 363, "eligible": 257, "extracted": 105, "hash_verified": 1},
        "batch2dc": {"total": 46, "extracted": 46, "identity_ready": 46, "eligible": 0},
        "benchmark_state": "DRAFT",
        "sampling_state": "DRAFT",
        "selection_status": "NOT_STARTED",
        "sampling_seed": None,
        "selected_cases": 0,
        "query_records": 0,
        "ground_truth_records": 0,
    }
    if state != expected:
        raise SourceReviewMigrationError("Protected source/benchmark state changed")
    return state


def build_outputs(project_root: Path, execution_git_sha: str, run_id: str) -> dict[str, dict]:
    root = Path(project_root).resolve()
    if len(execution_git_sha) != 40 or not run_id.startswith(PROTOCOL_ID + "_"):
        raise SourceReviewMigrationError("Execution identity is invalid")
    manifest_path = root / r6.MANIFEST_RELATIVE
    if policy_v2.sha256_file(manifest_path) != SOURCE_MANIFEST_SHA256:
        raise SourceReviewMigrationError("Live manifest binding changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v1.validate_acquisition_manifest(manifest)
    if policy_v2.load_and_validate_policy(root)["policy_fingerprint"] != POLICY_FINGERPRINT:
        raise SourceReviewMigrationError("Frozen V2 policy binding changed")
    current = _current_r6_context(root)
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r7_run_fingerprint": R7_RUN_FINGERPRINT,
    }
    outputs = {
        "schema_compatibility_matrix.json": schema_compatibility_matrix(),
        "versioned_manifest_schema_spec.json": schema_spec(),
        "migration_dry_run.json": migration_dry_run(root),
        "document_type_evidence_audit.json": document_type_evidence_audit(root, current),
        "review_authority_trust_boundary.json": review_authority_trust_boundary(),
        "targeted_adjudication_bindings.json": targeted_adjudication_bindings(root, current),
        "eligibility_gate_migration_design.json": eligibility_gate_design(),
    }
    protected = _protected_state(root, manifest)
    validation = {
        **common,
        "result": "PASS_VERSIONED_MIGRATION_SCAFFOLD_NOT_ACTIVATED",
        "selected_design": "SEPARATE_VERSIONED_REVIEW_RECORD_COLLECTION",
        "historical_v1_valid": True,
        "migration_dry_run_passed": outputs["migration_dry_run.json"]["result"] == "PASS",
        "document_type_audit_complete": (
            outputs["document_type_evidence_audit.json"]["batch2dc_count"] == 46
        ),
        "trusted_review_authority_activated": False,
        "v2_review_applied": False,
        "eligibility_promotion_performed": False,
        "live_manifest_mutated": False,
        "frozen_policy_modified": False,
        "benchmark_or_sampling_mutated": False,
        "network_used": False,
        "retrieval_or_model_used": False,
        "protected_state": protected,
    }
    validation["artifact_fingerprint"] = policy_v2.canonical_fingerprint(validation)
    outputs["validation_summary.json"] = validation
    for payload in outputs.values():
        for key, value in common.items():
            payload.setdefault(key, value)
        payload["artifact_fingerprint"] = policy_v2.canonical_fingerprint(
            payload, excluded_key="artifact_fingerprint"
        )
    return outputs


def write_run(
    project_root: Path,
    output_dir: Path,
    execution_git_sha: str,
    run_id: str,
) -> dict:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    expected_parent = (root / "backend/data/runs").resolve()
    if destination.parent != expected_parent or destination.name != run_id:
        raise SourceReviewMigrationError("Run directory must be a direct runs child")
    if destination.exists() or destination.is_symlink():
        raise SourceReviewMigrationError("Run directory already exists")
    manifest_path = root / r6.MANIFEST_RELATIVE
    before = manifest_path.read_bytes()
    outputs = build_outputs(root, execution_git_sha, run_id)
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            final = destination / name
            partial.write_bytes(canonical_json_bytes(payload))
            os.replace(partial, final)
            hashes[name] = policy_v2.sha256_file(final)
        if manifest_path.read_bytes() != before:
            raise SourceReviewMigrationError("Live manifest changed during R8")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "r7_run_fingerprint": R7_RUN_FINGERPRINT,
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "VERSIONED_MIGRATION_SCAFFOLD_VERIFIED_NOT_ACTIVATED",
        }
        run_manifest["run_fingerprint"] = policy_v2.canonical_fingerprint(run_manifest)
        (destination / "run_manifest.json").write_bytes(canonical_json_bytes(run_manifest))
        return run_manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-git-sha", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            write_run(
                args.project_root,
                args.output_dir,
                args.execution_git_sha,
                args.run_id,
            ),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
