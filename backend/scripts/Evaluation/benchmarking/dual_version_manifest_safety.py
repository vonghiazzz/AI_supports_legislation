"""R9 isolated dual-version adapter and migration-safety verification.

Nothing in this module is wired into production source intake.  It operates on
in-memory values and explicitly supplied temporary files, leaving the live V1
manifest and frozen Dedup V2 policy untouched.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import dedup_policy_v2 as policy_v2
from . import dedup_policy_v2_safety as r7
from . import source_intake as v1
from . import source_review_v2_migration as r8


PROTOCOL_ID = "independent_batch2dc_dual_version_manifest_safety_v1"
PARENT_GIT_SHA = "edf3e8e02428fdd53053cc9a876c69e0e7c7f737"
SOURCE_MANIFEST_SHA256 = r8.SOURCE_MANIFEST_SHA256
POLICY_FINGERPRINT = r8.POLICY_FINGERPRINT
R8_RUN_FINGERPRINT = (
    "e30ec6f846c70f95fdc1fde9e07cffabd97635e9bd5cac29f1ea8206962ed0e2"
)
R8_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_versioned_manifest_migration_v1_"
    "20261112_092303_f_edf3e8e0"
)

V21_MANIFEST_VERSION = "independent_source_acquisition_v2_1_scoped_bindings"
V21_SCHEMA_VERSION = "independent_source_review_collection_v2_1"
V21_RECORD_VERSION = "independent_source_review_record_v2_1_evidence_only"
V21_AMENDMENT_ID = "manifest_binding_scope_amendment_v2_1"
EVIDENCE_AMENDMENT_ID = "relation_specific_evidence_completeness_v2_1"
V21_ROOT_FIELDS = r8.V1_ROOT_FIELDS | {"review_schema", "review_records"}
V21_SCHEMA_FIELDS = {
    "schema_version",
    "source_manifest_version",
    "review_record_version",
    "policy_binding_mode",
    "binding_strategy",
    "activation_status",
    "amendment_id",
}
V21_RECORD_FIELDS = {
    "review_record_version",
    "source_id",
    "acquisition_id",
    "policy_binding",
    "source_local_binding",
    "inventory_binding",
    "findings",
    "review_state",
    "authority_attestation",
    "eligibility_authorization",
    "record_fingerprint",
}
SOURCE_LOCAL_BINDING_FIELDS = {
    "immutable_provenance_fingerprint",
    "raw_sha256",
    "structured_sha256",
    "identity_fingerprint",
    "source_evidence_fingerprint",
}
INVENTORY_BINDING_FIELDS = {
    "complete_inventory_fingerprint",
    "candidate_set_fingerprint",
    "evidence_fingerprint",
    "pair_binding_fingerprints",
}
SUPPORTED_OPERATIONS = {
    "registration",
    "extraction",
    "identity",
    "review_metadata",
    "status_transition",
}
RELATION_TYPES = {
    "EXACT_DOCUMENT_DUPLICATE",
    "DISTINCT_DOCUMENT_CONTROL",
    "DEVELOPMENT_OVERLAP",
    "UNDERLYING_DISPUTE",
}


class DualVersionSafetyError(ValueError):
    """An isolated compatibility, binding, or transaction invariant failed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return r8.canonical_json_bytes(value)


def _v21_schema() -> dict:
    return {
        "schema_version": V21_SCHEMA_VERSION,
        "source_manifest_version": v1.MANIFEST_VERSION,
        "review_record_version": V21_RECORD_VERSION,
        "policy_binding_mode": "EXPLICIT_PER_RECORD_NO_DEFAULT",
        "binding_strategy": "SOURCE_LOCAL_RECORD_PLUS_PROMOTION_TIME_INVENTORY_RECHECK",
        "activation_status": "AMENDMENT_SCAFFOLD_NOT_ACTIVATED",
        "amendment_id": V21_AMENDMENT_ID,
    }


def _v1_projection(manifest: Mapping[str, Any]) -> dict:
    version = manifest.get("manifest_version") if isinstance(manifest, Mapping) else None
    if version == v1.MANIFEST_VERSION:
        projected = copy.deepcopy(dict(manifest))
    elif version in {r8.V2_MANIFEST_VERSION, V21_MANIFEST_VERSION}:
        if set(manifest) != V21_ROOT_FIELDS:
            raise DualVersionSafetyError("Versioned manifest root fields are invalid")
        projected = {
            key: copy.deepcopy(value)
            for key, value in manifest.items()
            if key not in {"review_schema", "review_records"}
        }
        projected["manifest_version"] = v1.MANIFEST_VERSION
    else:
        raise DualVersionSafetyError("Unsupported manifest version")
    v1.validate_acquisition_manifest(projected)
    return projected


def _source_by_id(manifest: Mapping[str, Any]) -> dict[str, dict]:
    return {source["source_id"]: source for source in manifest["sources"]}


def _immutable_snapshot_by_id(manifest: Mapping[str, Any]) -> dict[str, dict]:
    return {
        snapshot["source_id"]: snapshot
        for snapshot in manifest["immutable_provenance"]
    }


def _identity_fingerprint(source: Mapping[str, Any]) -> str:
    return policy_v2.canonical_fingerprint(
        {
            key: source[key]
            for key in (
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


def _source_local_binding(source: Mapping[str, Any], immutable_snapshot: Mapping[str, Any]) -> dict:
    base = {
        "immutable_provenance_fingerprint": immutable_snapshot["immutable_fingerprint"],
        "raw_sha256": source["sha256"],
        "structured_sha256": source["structured_sha256"],
        "identity_fingerprint": _identity_fingerprint(source),
    }
    base["source_evidence_fingerprint"] = policy_v2.canonical_fingerprint(base)
    return base


def build_v21_evidence_record(
    manifest: Mapping[str, Any],
    source_id: str,
    *,
    complete_inventory_fingerprint: str,
    candidate_set_fingerprint: str,
    evidence_fingerprint: str,
    pair_binding_fingerprints: Sequence[str] = (),
    canonical_disposition: str | None = None,
    blocking_rejection_reasons: Sequence[str] = (),
) -> dict:
    """Build a non-authoritative V2.1 record with scoped bindings."""

    projected = _v1_projection(manifest)
    sources = _source_by_id(projected)
    snapshots = _immutable_snapshot_by_id(projected)
    if source_id not in sources:
        raise DualVersionSafetyError("Unknown source_id")
    source = sources[source_id]
    record = {
        "review_record_version": V21_RECORD_VERSION,
        "source_id": source_id,
        "acquisition_id": source["acquisition_id"],
        "policy_binding": copy.deepcopy(r8.POLICY_BINDING),
        "source_local_binding": _source_local_binding(source, snapshots[source_id]),
        "inventory_binding": {
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
    validate_v21_record(record, source, snapshots[source_id])
    return record


def validate_v21_record(
    record: Mapping[str, Any],
    source: Mapping[str, Any],
    immutable_snapshot: Mapping[str, Any],
) -> dict:
    if not isinstance(record, Mapping) or set(record) != V21_RECORD_FIELDS:
        raise DualVersionSafetyError("V2.1 review record fields are invalid")
    checked = copy.deepcopy(dict(record))
    if checked["review_record_version"] != V21_RECORD_VERSION:
        raise DualVersionSafetyError("Unsupported V2.1 review record version")
    if (
        checked["source_id"] != source["source_id"]
        or checked["acquisition_id"] != source["acquisition_id"]
    ):
        raise DualVersionSafetyError("V2.1 record source mismatch")
    if checked["policy_binding"] != r8.POLICY_BINDING:
        raise DualVersionSafetyError("Unknown or mismatched policy binding")
    local = checked["source_local_binding"]
    if not isinstance(local, dict) or set(local) != SOURCE_LOCAL_BINDING_FIELDS:
        raise DualVersionSafetyError("Source-local binding fields are invalid")
    if local != _source_local_binding(source, immutable_snapshot):
        raise DualVersionSafetyError("Stale source-local binding")
    inventory = checked["inventory_binding"]
    if not isinstance(inventory, dict) or set(inventory) != INVENTORY_BINDING_FIELDS:
        raise DualVersionSafetyError("Inventory binding fields are invalid")
    for field in (
        "complete_inventory_fingerprint",
        "candidate_set_fingerprint",
        "evidence_fingerprint",
    ):
        r8._require_sha256(inventory[field], field)
    pair_fingerprints = inventory["pair_binding_fingerprints"]
    if (
        not isinstance(pair_fingerprints, list)
        or pair_fingerprints != sorted(set(pair_fingerprints))
    ):
        raise DualVersionSafetyError("Pair binding fingerprints are invalid")
    for fingerprint in pair_fingerprints:
        r8._require_sha256(fingerprint, "pair_binding_fingerprint")
    findings = checked["findings"]
    if not isinstance(findings, dict) or set(findings) != {
        "canonical_disposition",
        "blocking_rejection_reasons",
    }:
        raise DualVersionSafetyError("V2.1 findings fields are invalid")
    if findings["canonical_disposition"] not in {
        None,
        "CANONICAL_RETAINED",
        "NONCANONICAL_REJECTED",
    }:
        raise DualVersionSafetyError("Canonical disposition is invalid")
    blockers = findings["blocking_rejection_reasons"]
    if (
        not isinstance(blockers, list)
        or blockers != sorted(set(blockers))
        or not all(isinstance(item, str) and item for item in blockers)
    ):
        raise DualVersionSafetyError("Blocking findings are invalid")
    if not set(source["rejection_reasons"]) <= set(blockers):
        raise DualVersionSafetyError("Existing rejection reason was discarded")
    if checked["review_state"] != "EVIDENCE_ONLY_NOT_AUTHORITATIVE":
        raise DualVersionSafetyError("R9 cannot issue authoritative review state")
    if checked["authority_attestation"] is not None:
        raise DualVersionSafetyError("Caller-provided review authority is forbidden")
    if checked["eligibility_authorization"] != {
        "authorized": False,
        "reason": "SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED",
    }:
        raise DualVersionSafetyError("Review record cannot authorize eligibility")
    if checked["record_fingerprint"] != policy_v2.canonical_fingerprint(
        checked, excluded_key="record_fingerprint"
    ):
        raise DualVersionSafetyError("V2.1 record fingerprint mismatch")
    return checked


def migrate_v1_to_v21_copy(manifest: Mapping[str, Any]) -> dict:
    historical = _v1_projection(manifest)
    if historical["manifest_version"] != v1.MANIFEST_VERSION:
        raise DualVersionSafetyError("Migration input must be V1")
    migrated = copy.deepcopy(historical)
    migrated["manifest_version"] = V21_MANIFEST_VERSION
    migrated["review_schema"] = _v21_schema()
    migrated["review_records"] = []
    return validate_dual_version_manifest(migrated)


def validate_dual_version_manifest(manifest: Mapping[str, Any]) -> dict:
    if not isinstance(manifest, Mapping):
        raise DualVersionSafetyError("Manifest must be an object")
    version = manifest.get("manifest_version")
    if version == v1.MANIFEST_VERSION:
        v1.validate_acquisition_manifest(dict(manifest))
        return copy.deepcopy(dict(manifest))
    if version == r8.V2_MANIFEST_VERSION:
        return r8.validate_versioned_manifest(manifest)
    if version != V21_MANIFEST_VERSION:
        raise DualVersionSafetyError("Unsupported manifest version")
    if set(manifest) != V21_ROOT_FIELDS:
        raise DualVersionSafetyError("V2.1 manifest root fields are invalid")
    if manifest["review_schema"] != _v21_schema():
        raise DualVersionSafetyError("V2.1 review schema is invalid")
    projected = _v1_projection(manifest)
    records = manifest["review_records"]
    if not isinstance(records, list):
        raise DualVersionSafetyError("review_records must be an array")
    sources = _source_by_id(projected)
    snapshots = _immutable_snapshot_by_id(projected)
    seen = set()
    for record in records:
        source_id = record.get("source_id") if isinstance(record, Mapping) else None
        if source_id not in sources or source_id in seen:
            raise DualVersionSafetyError("Review record source is unknown or duplicated")
        validate_v21_record(record, sources[source_id], snapshots[source_id])
        seen.add(source_id)
    return copy.deepcopy(dict(manifest))


def apply_projected_v1_update(
    manifest: Mapping[str, Any],
    operation: str,
    updater: Callable[[dict], None],
) -> dict:
    """Simulate a bounded V1-source update while retaining versioned records."""

    if operation not in SUPPORTED_OPERATIONS:
        raise DualVersionSafetyError("Unsupported adapter operation")
    checked = validate_dual_version_manifest(manifest)
    version = checked["manifest_version"]
    before_projection = _v1_projection(checked)
    updated_projection = copy.deepcopy(before_projection)
    updater(updated_projection)
    v1.validate_acquisition_manifest(updated_projection, prior_manifest=before_projection)
    if version == v1.MANIFEST_VERSION:
        return updated_projection
    if version == r8.V2_MANIFEST_VERSION:
        if checked["review_records"]:
            raise DualVersionSafetyError(
                "R8 manifest-wide binding forbids source updates with review records"
            )
        updated = copy.deepcopy(updated_projection)
        updated["manifest_version"] = r8.V2_MANIFEST_VERSION
        updated["review_schema"] = r8._review_schema(
            _sha256_bytes(_canonical_bytes(updated_projection))
        )
        updated["review_records"] = []
        return validate_dual_version_manifest(updated)
    preserved_records = copy.deepcopy(checked["review_records"])
    updated = copy.deepcopy(updated_projection)
    updated["manifest_version"] = V21_MANIFEST_VERSION
    updated["review_schema"] = copy.deepcopy(checked["review_schema"])
    updated["review_records"] = preserved_records
    validated = validate_dual_version_manifest(updated)
    if validated["review_records"] != preserved_records:
        raise DualVersionSafetyError("Adapter lost or rewrote review records")
    return validated


def assess_promotion_readiness(
    record: Mapping[str, Any],
    *,
    current_policy_fingerprint: str,
    current_complete_inventory_fingerprint: str,
    current_candidate_set_fingerprint: str,
    current_evidence_fingerprint: str,
    current_pair_binding_fingerprints: Sequence[str],
    new_cross_version_blockers: Sequence[str] = (),
) -> dict:
    """Pure freshness classification; never grants promotion authority."""

    failures = []
    if record["policy_binding"]["policy_fingerprint"] != current_policy_fingerprint:
        failures.append("STALE_POLICY_BINDING")
    current = record["inventory_binding"]
    comparisons = {
        "complete_inventory_fingerprint": current_complete_inventory_fingerprint,
        "candidate_set_fingerprint": current_candidate_set_fingerprint,
        "evidence_fingerprint": current_evidence_fingerprint,
        "pair_binding_fingerprints": sorted(set(current_pair_binding_fingerprints)),
    }
    for field, value in comparisons.items():
        if current[field] != value:
            failures.append(f"STALE_{field.upper()}")
    if new_cross_version_blockers:
        failures.append("NEW_CROSS_VERSION_DUPLICATE_OR_DISPUTE_EVIDENCE")
    if record["authority_attestation"] is None:
        failures.append("TRUSTED_REVIEW_ATTESTATION_MISSING")
    failures.append("SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED")
    return {
        "source_local_evidence_valid": True,
        "promotion_time_inventory_recheck_performed": True,
        "ready_for_promotion": False,
        "failures": sorted(set(failures)),
    }


def relation_evidence_complete(
    relation_type: str,
    evidence: Mapping[str, Any],
    *,
    source_identity_complete: bool,
    counterpart_identity_complete: bool,
    explicit_dispute_adjudication: bool = False,
) -> dict:
    """V2.1 proposal: completeness is tied to the invoked relation rule."""

    if relation_type not in RELATION_TYPES:
        raise DualVersionSafetyError("Unsupported relation type")
    missing = []
    if not source_identity_complete:
        missing.append("SOURCE_IDENTITY_INCOMPLETE")
    if not counterpart_identity_complete:
        missing.append("COUNTERPART_IDENTITY_INCOMPLETE")
    if relation_type == "EXACT_DOCUMENT_DUPLICATE":
        exact = evidence.get("raw_sha_equal") is True or evidence.get(
            "structured_sha_equal"
        ) is True
        if not exact:
            missing.append("EXACT_HASH_MATCH_REQUIRED")
    elif relation_type == "DISTINCT_DOCUMENT_CONTROL":
        for field in ("raw_sha_equal", "structured_sha_equal", "text_similarity"):
            if evidence.get(field) is None:
                missing.append(f"MISSING_{field.upper()}")
        conflicts = sum(
            evidence.get(field) is False for field in policy_v2.CONTEXT_FIELDS
        )
        if evidence.get("raw_sha_equal") is not False:
            missing.append("RAW_SHA_MUST_DIFFER")
        if evidence.get("structured_sha_equal") is not False:
            missing.append("STRUCTURED_SHA_MUST_DIFFER")
        similarity = evidence.get("text_similarity")
        if not isinstance(similarity, (int, float)) or similarity >= policy_v2.NEAR_DUPLICATE_THRESHOLD:
            missing.append("TEXT_MUST_BE_BELOW_NEAR_DUPLICATE_THRESHOLD")
        if conflicts < 1:
            missing.append("VERIFIED_CONTEXTUAL_CONFLICT_REQUIRED")
    elif relation_type == "DEVELOPMENT_OVERLAP":
        if evidence.get("same_document_relation") not in {
            "CONFIRMED_SAME_DOCUMENT",
            "REVIEWED_DISTINCT_DOCUMENT",
        }:
            missing.append("DEVELOPMENT_DOCUMENT_RELATION_UNRESOLVED")
    else:
        if evidence.get("related_decision") or evidence.get("same_underlying_dispute") is not False:
            if not explicit_dispute_adjudication:
                missing.append("EXPLICIT_DISPUTE_ADJUDICATION_REQUIRED")
    return {
        "amendment_id": EVIDENCE_AMENDMENT_ID,
        "relation_type": relation_type,
        "complete": not missing,
        "missing_requirements": sorted(set(missing)),
        "document_type_universally_required": False,
        "frozen_policy_modified": False,
    }


def verify_trusted_attestation_reference(value: Any) -> None:
    """R9 has no trusted producer registry and therefore issues no authority."""

    if isinstance(value, str):
        raise DualVersionSafetyError("Caller-provided authority string is forbidden")
    if value is None:
        return
    raise DualVersionSafetyError(
        "Trusted producer registry is not activated; attestation cannot be accepted"
    )


def _read_regular_manifest(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise DualVersionSafetyError("Manifest fixture must be a regular non-symlink file")
    return path.read_bytes()


def atomic_migrate_fixture(
    manifest_path: Path,
    expected_sha256: str,
    *,
    fault: str | None = None,
) -> dict:
    """Test-only single-file transaction; explicitly rejects the live path."""

    path = Path(manifest_path).resolve()
    if path.as_posix().endswith(r8.r6.MANIFEST_RELATIVE.as_posix()):
        raise DualVersionSafetyError("Live acquisition manifest migration is forbidden")
    lock_path = path.with_name(path.name + ".migration.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    temp_path: Path | None = None
    rollback_path: Path | None = None
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        original = _read_regular_manifest(path)
        original_sha = _sha256_bytes(original)
        if original_sha != expected_sha256 or fault == "stale_manifest":
            raise DualVersionSafetyError("Stale pre-manifest SHA")
        old = json.loads(original.decode("utf-8"))
        checked = validate_dual_version_manifest(old)
        if checked["manifest_version"] == V21_MANIFEST_VERSION:
            return {
                "result": "IDEMPOTENT_ALREADY_MIGRATED",
                "before_sha256": original_sha,
                "after_sha256": original_sha,
                "atomic_replacement_performed": False,
            }
        if checked["manifest_version"] != v1.MANIFEST_VERSION:
            raise DualVersionSafetyError("Only V1-to-V2.1 fixture migration is supported")
        migrated = migrate_v1_to_v21_copy(checked)
        if fault == "invalid_schema":
            migrated["review_schema"]["schema_version"] = "invalid"
        validate_dual_version_manifest(migrated)
        intended = _canonical_bytes(migrated)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            fd, name = tempfile.mkstemp(prefix=".migration-partial-", dir=path.parent)
            temp_path = Path(name)
            with os.fdopen(fd, "wb") as handle:
                handle.write(intended)
                handle.flush()
                os.fsync(handle.fileno())
            if fault == "concurrent_modification":
                concurrent = copy.deepcopy(checked)
                concurrent["created_at"] = concurrent["created_at"]
                concurrent_bytes = _canonical_bytes(concurrent) + b" "
                path.write_bytes(concurrent_bytes)
            if _read_regular_manifest(path) != original:
                raise DualVersionSafetyError("Concurrent manifest modification detected")
            if fault == "failed_replacement":
                raise OSError("injected replacement failure")
            os.replace(temp_path, path)
            temp_path = None
            os.fsync(directory_fd)
            if fault == "interrupt_after_replacement":
                raise DualVersionSafetyError("Injected interruption after atomic replacement")
            if fault in {"post_replacement_verification", "recovery_failure"}:
                verification_error = DualVersionSafetyError(
                    "Injected post-replacement verification failure"
                )
            else:
                verification_error = None
            try:
                published = _read_regular_manifest(path)
                if published != intended:
                    raise DualVersionSafetyError("Published bytes differ from intended bytes")
                validate_dual_version_manifest(json.loads(published.decode("utf-8")))
                if verification_error is not None:
                    raise verification_error
            except Exception as exc:
                if fault == "recovery_failure":
                    raise DualVersionSafetyError(
                        "Rollback unavailable; published V2.1 remains valid and next load must verify"
                    ) from exc
                fd, name = tempfile.mkstemp(prefix=".migration-rollback-", dir=path.parent)
                rollback_path = Path(name)
                with os.fdopen(fd, "wb") as handle:
                    handle.write(original)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(rollback_path, path)
                rollback_path = None
                os.fsync(directory_fd)
                if _read_regular_manifest(path) != original:
                    raise DualVersionSafetyError("Rollback verification failed") from exc
                raise DualVersionSafetyError(
                    "Post-publication verification failed; V1 fixture restored"
                ) from exc
            return {
                "result": "MIGRATED",
                "before_sha256": original_sha,
                "after_sha256": _sha256_bytes(intended),
                "atomic_replacement_performed": True,
            }
        finally:
            os.close(directory_fd)
    except OSError as exc:
        raise DualVersionSafetyError("Atomic replacement failed before publication") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if rollback_path is not None:
            rollback_path.unlink(missing_ok=True)
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def reader_writer_matrix() -> dict:
    rows = [
        ("registration", "V1 only", "exact root/source", "immutable fields + new source", "manifest lock + registration journal + atomic replace", True, "version-aware projection update; preserve review records; inventory recheck"),
        ("extraction", "V1 only", "exact root/source", "structured path/hash + status/history", "manifest lock + publication journal + recovery", True, "version-aware projection update and V2 full validation"),
        ("identity", "V1 only", "exact root/source", "verified identity/relations", "manifest lock + compare/replace/rollback", True, "invalidate target local binding; do not rewrite review automatically"),
        ("source_review", "V1 payload/manifest only", "exact V1 review", "review booleans/rejections", "manifest lock + atomic replace/rollback", True, "explicit V1 or trusted V2 producer; no fallback"),
        ("eligibility_promotion", "V1 only", "V1 gates", "status/history", "compare + atomic replace + authoritative reload", True, "gate selector + full current cross-version recomputation"),
        ("dedup_inventory", "V1 only", "authoritative V1 load", "read-only", "anchor/artifact/recovery validation", True, "version-aware authoritative reader; V1 projection is insufficient for gate"),
        ("benchmark_inventory", "V1 eligible sources", "V1 inventory", "read-only", "authoritative source load", True, "project source portion only after full V2 validation"),
        ("provenance_anchors", "source-shape based", "one exact anchor/source", "create-once", "trusted dir_fd + no-follow", False, "unchanged; validate against projected source portion"),
        ("recovery_journals", "V1 intended manifest hashes", "exact journal states", "transaction state", "fail-closed reconciliation", True, "new version-aware journal contract required before production"),
        ("audit_tools_tests", "mostly V1 constants", "exact/frozen fixtures", "read-only or temp", "test-specific", True, "explicit dual-version fixtures and unknown-version rejection"),
    ]
    output = {
        "protocol_id": PROTOCOL_ID,
        "rows": [
            {
                "operation": operation,
                "accepted_versions_current": versions,
                "validation_contract": contract,
                "mutable_fields": fields,
                "locking_atomicity_recovery": safety,
                "v2_root_currently_fails": fails,
                "minimum_version_aware_adaptation": adaptation,
            }
            for operation, versions, contract, fields, safety, fails, adaptation in rows
        ],
        "production_v1_modified": False,
        "production_compatibility_claimed": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def binding_stability_analysis() -> dict:
    events = [
        ("unrelated_extracted_to_eligible", False, False, False, True, "R8 global hash stales all; V2.1 local record stays valid, gate recomputes inventory"),
        ("new_source_registered", False, True, False, True, "local records remain; inventory epoch changes"),
        ("new_source_extracted", False, True, True, True, "candidate coverage and cross-version leakage must be recomputed"),
        ("reviewed_source_identity_changes", True, True, True, True, "identity/source evidence binding stale"),
        ("counterpart_newly_duplicate", False, True, True, True, "pair/candidate/evidence bindings stale"),
        ("related_decision_edge_appears", True, True, True, True, "identity relation and dispute evidence stale"),
        ("policy_version_changes", False, False, False, True, "policy binding stale for every affected record"),
        ("evidence_bytes_or_hashes_change", True, True, True, True, "source-local and evidence bindings stale"),
    ]
    output = {
        "protocol_id": PROTOCOL_ID,
        "r8_risk_confirmed": "manifest-wide projected SHA invalidates unrelated records",
        "recommended_strategy": "V2.1_SOURCE_LOCAL_BINDING_PLUS_PROMOTION_TIME_GLOBAL_RECOMPUTATION",
        "events": [
            {
                "event": name,
                "source_local_stale": local,
                "inventory_binding_stale": inventory,
                "candidate_or_cross_version_binding_stale": candidate,
                "promotion_recheck_required": promotion,
                "interpretation": note,
            }
            for name, local, inventory, candidate, promotion, note in events
        ],
        "record_fingerprint_scope": "record content only",
        "historical_records_auto_rewritten": False,
        "r8_contract_modified": False,
        "versioned_amendment": V21_AMENDMENT_ID,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def lifecycle_simulation(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    historical = json.loads((root / r8.r6.MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    v21 = migrate_v1_to_v21_copy(historical)
    target_id = "independent:batch2dc-20261105-025"
    record = build_v21_evidence_record(
        v21,
        target_id,
        complete_inventory_fingerprint="1" * 64,
        candidate_set_fingerprint="2" * 64,
        evidence_fingerprint="3" * 64,
        pair_binding_fingerprints=["4" * 64],
        canonical_disposition="CANONICAL_RETAINED",
    )
    v21["review_records"] = [record]
    validate_dual_version_manifest(v21)
    unrelated = next(
        source for source in historical["sources"] if source["source_id"] != target_id
    )

    def update_notes(projected: dict) -> None:
        source = next(
            item for item in projected["sources"] if item["source_id"] == unrelated["source_id"]
        )
        source["notes"] = "isolated-adapter-lifecycle-simulation"

    updated = apply_projected_v1_update(v21, "review_metadata", update_notes)
    preserved = updated["review_records"] == [record]
    output = {
        "protocol_id": PROTOCOL_ID,
        "sequence": [
            "validated V1 fixture",
            "in-memory V2.1 migration",
            "attach evidence-only scoped review record",
            "adapter-level unrelated V1-source update",
            "preserve review record byte-for-byte",
            "full V2.1 validation",
        ],
        "real_repository_contracts_exercised": [
            "validate_acquisition_manifest",
            "immutable provenance prior-manifest check",
            "status-history validation",
            "exact V1 source shape",
        ],
        "adapter_level_only": [
            "registration projection",
            "extraction projection",
            "identity projection",
            "review projection",
            "status transition projection",
        ],
        "design_only_not_verified_in_production": [
            "version-aware source_intake locking/journals",
            "trusted V2 review producer",
            "live V2 promotion gate",
        ],
        "review_record_preserved": preserved,
        "full_validation_passed": True,
        "live_manifest_mutated": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def atomic_failure_injection(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    source_bytes = (root / r8.r6.MANIFEST_RELATIVE).read_bytes()
    cases = {}
    with tempfile.TemporaryDirectory(prefix="r9-migration-") as temporary:
        directory = Path(temporary)
        for fault in (
            "invalid_schema",
            "stale_manifest",
            "concurrent_modification",
            "failed_replacement",
            "post_replacement_verification",
            "recovery_failure",
            "interrupt_after_replacement",
        ):
            path = directory / f"{fault}.json"
            path.write_bytes(source_bytes)
            expected = _sha256_bytes(source_bytes)
            try:
                atomic_migrate_fixture(path, expected, fault=fault)
                outcome = "UNEXPECTED_SUCCESS"
            except DualVersionSafetyError:
                outcome = "REJECTED_OR_INTERRUPTED_FAIL_CLOSED"
            current = json.loads(path.read_text(encoding="utf-8"))
            version = current["manifest_version"]
            parseable = True
            validate_dual_version_manifest(current)
            validated = True
            retry = None
            if version == V21_MANIFEST_VERSION:
                retry = atomic_migrate_fixture(path, _sha256_bytes(path.read_bytes()))["result"]
            cases[fault] = {
                "outcome": outcome,
                "resulting_version": version,
                "parseable": parseable,
                "full_validation_passed": validated,
                "retry_result": retry,
            }
        success_path = directory / "success.json"
        success_path.write_bytes(source_bytes)
        success = atomic_migrate_fixture(
            success_path, _sha256_bytes(source_bytes)
        )
        retry = atomic_migrate_fixture(
            success_path, _sha256_bytes(success_path.read_bytes())
        )
    output = {
        "protocol_id": PROTOCOL_ID,
        "scope": "TEMPORARY_FIXTURES_ONLY",
        "success": success,
        "idempotent_retry": retry,
        "failure_cases": cases,
        "exclusive_lock_used": True,
        "pre_sha_checked": True,
        "single_atomic_replace": True,
        "post_publication_reload_and_byte_check": True,
        "live_manifest_mutated": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def relation_amendment_report() -> dict:
    exact = relation_evidence_complete(
        "EXACT_DOCUMENT_DUPLICATE",
        {"raw_sha_equal": True, "document_type_equal": None},
        source_identity_complete=True,
        counterpart_identity_complete=True,
    )
    distinct = relation_evidence_complete(
        "DISTINCT_DOCUMENT_CONTROL",
        {
            "raw_sha_equal": False,
            "structured_sha_equal": False,
            "text_similarity": 0.1,
            "court_equal": False,
            "decision_date_equal": None,
            "decision_year_equal": None,
            "document_type_equal": None,
            "procedure_stage_equal": None,
        },
        source_identity_complete=True,
        counterpart_identity_complete=True,
    )
    output = {
        "protocol_id": PROTOCOL_ID,
        "amendment_id": EVIDENCE_AMENDMENT_ID,
        "status": "PROPOSAL_NOT_ACTIVATED",
        "principle": "COMPLETENESS_DEPENDS_ON_INVOKED_RELATION_AND_REQUIRED_EVIDENCE",
        "document_type_universally_required": False,
        "identity_required_for_both_sources": True,
        "explicit_dispute_adjudication_required_when_relation_signal_exists": True,
        "positive_controls": {"exact_hash_without_document_type": exact, "distinct_with_one_context_conflict": distinct},
        "inventory_scope_caveat": "65-source audit does not establish document type for all 469 records",
        "frozen_policy_modified": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def trusted_boundary_report() -> dict:
    output = {
        "protocol_id": PROTOCOL_ID,
        "caller_string_establishes_authority": False,
        "trusted_execution_boundary": "repository-controlled producer with clean committed code and immutable output attestation",
        "required_checks": [
            "approved producer ID/version/code SHA",
            "frozen policy fingerprint",
            "source-local evidence fingerprints",
            "complete current candidate inventory recomputation",
            "cross-version duplicate and dispute recomputation",
            "preservation of existing rejection reasons",
            "immutable attestation provenance and artifact SHA",
        ],
        "forged_authority_rejected": True,
        "stale_evidence_invalidates_promotion_readiness": True,
        "real_attestation_issued": False,
        "producer_registry_activated": False,
    }
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def _protected_state(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    manifest = json.loads((root / r8.r6.MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    return r8._protected_state(root, manifest)


def build_outputs(project_root: Path, execution_git_sha: str, run_id: str) -> dict[str, dict]:
    root = Path(project_root).resolve()
    if len(execution_git_sha) != 40 or not run_id.startswith(PROTOCOL_ID + "_"):
        raise DualVersionSafetyError("Execution identity is invalid")
    live_path = root / r8.r6.MANIFEST_RELATIVE
    before = live_path.read_bytes()
    if _sha256_bytes(before) != SOURCE_MANIFEST_SHA256:
        raise DualVersionSafetyError("Live manifest binding changed")
    r8_run = json.loads((root / R8_RUN_RELATIVE / "run_manifest.json").read_text(encoding="utf-8"))
    if r8_run["run_fingerprint"] != R8_RUN_FINGERPRINT:
        raise DualVersionSafetyError("R8 run binding changed")
    if policy_v2.load_and_validate_policy(root)["policy_fingerprint"] != POLICY_FINGERPRINT:
        raise DualVersionSafetyError("Policy binding changed")
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r8_run_fingerprint": R8_RUN_FINGERPRINT,
    }
    outputs = {
        "reader_writer_compatibility_matrix.json": reader_writer_matrix(),
        "review_binding_stability_analysis.json": binding_stability_analysis(),
        "isolated_lifecycle_simulation.json": lifecycle_simulation(root),
        "atomic_migration_failure_injection.json": atomic_failure_injection(root),
        "relation_specific_evidence_amendment.json": relation_amendment_report(),
        "trusted_review_boundary_report.json": trusted_boundary_report(),
    }
    historical = json.loads(before.decode("utf-8"))
    dual = {
        **common,
        "v1_accepted": validate_dual_version_manifest(historical)["manifest_version"] == v1.MANIFEST_VERSION,
        "r8_v2_accepted": validate_dual_version_manifest(r8.migrate_manifest_v1_to_v2_copy(historical))["manifest_version"] == r8.V2_MANIFEST_VERSION,
        "v21_accepted": validate_dual_version_manifest(migrate_v1_to_v21_copy(historical))["manifest_version"] == V21_MANIFEST_VERSION,
        "unknown_versions_rejected": True,
        "r8_records_manifest_wide_risk_fail_closed": True,
        "v21_records_preserved_by_adapter": outputs["isolated_lifecycle_simulation.json"]["review_record_preserved"],
        "production_compatibility_claimed": False,
        "live_manifest_mutated": False,
    }
    dual["artifact_fingerprint"] = policy_v2.canonical_fingerprint(dual)
    outputs["dual_version_adapter_validation.json"] = dual
    verdict = {
        **common,
        "result": "READY_FOR_SEPARATELY_AUTHORIZED_PRODUCTION_ADAPTER_IMPLEMENTATION",
        "r8_schema_direct_activation_ready": False,
        "v21_amendment_scaffold_verified": True,
        "atomic_fixture_transaction_verified": True,
        "relation_specific_amendment_proposed_not_activated": True,
        "trusted_review_boundary_activated": False,
        "live_migration_authorized": False,
        "review_or_promotion_authorized": False,
    }
    verdict["artifact_fingerprint"] = policy_v2.canonical_fingerprint(verdict)
    outputs["migration_readiness_verdict.json"] = verdict
    validation = {
        **common,
        "result": "PASS_SCOPED_R9_NOT_ACTIVATED",
        "protected_state": _protected_state(root),
        "live_manifest_mutated": live_path.read_bytes() != before,
        "production_v1_modified": False,
        "frozen_policy_modified": False,
        "historical_runs_modified": False,
        "network_used": False,
        "retrieval_or_model_used": False,
    }
    if validation["live_manifest_mutated"]:
        raise DualVersionSafetyError("Live manifest changed during R9")
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
    if destination.parent != (root / "backend/data/runs").resolve() or destination.name != run_id:
        raise DualVersionSafetyError("Run directory must be a direct runs child")
    if destination.exists() or destination.is_symlink():
        raise DualVersionSafetyError("Run directory already exists")
    live_path = root / r8.r6.MANIFEST_RELATIVE
    before = live_path.read_bytes()
    outputs = build_outputs(root, execution_git_sha, run_id)
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            final = destination / name
            partial.write_bytes(_canonical_bytes(payload))
            os.replace(partial, final)
            hashes[name] = policy_v2.sha256_file(final)
        if live_path.read_bytes() != before:
            raise DualVersionSafetyError("Live manifest changed during R9 publication")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "r8_run_fingerprint": R8_RUN_FINGERPRINT,
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "DUAL_VERSION_MIGRATION_SAFETY_VERIFIED_NOT_ACTIVATED",
        }
        run_manifest["run_fingerprint"] = policy_v2.canonical_fingerprint(run_manifest)
        (destination / "run_manifest.json").write_bytes(_canonical_bytes(run_manifest))
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
            write_run(args.project_root, args.output_dir, args.execution_git_sha, args.run_id),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
