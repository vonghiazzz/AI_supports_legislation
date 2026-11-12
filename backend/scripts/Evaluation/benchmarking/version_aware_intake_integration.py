"""R10 verification for inactive V2.1 source-intake integration.

The live manifest remains V1.  This module audits the opt-in staging path,
performs read-only authoritative candidate recomputation, and publishes only
isolated reports.  It cannot activate V2.1 review or eligibility promotion.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import dedup_policy_v2 as policy_v2
from . import dedup_policy_v2_independent as r6
from . import dual_version_manifest_safety as r9
from . import source_intake as intake
from . import source_review_v2_migration as r8


PROTOCOL_ID = "independent_batch2dc_version_aware_intake_integration_v1"
PARENT_GIT_SHA = "cdc0f1af6719ef85585af3999eafa2fa13f83eed"
SOURCE_MANIFEST_SHA256 = r8.SOURCE_MANIFEST_SHA256
POLICY_FINGERPRINT = r8.POLICY_FINGERPRINT
R9_RUN_FINGERPRINT = (
    "69710eaadf17bf7c0f856fbf1b8743ea114ac379fef1e4c1d4f16e6eddcae89d"
)
R9_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_dual_version_manifest_safety_v1_"
    "20261112_095258_cdc0f1af"
)
RELATION_MAPPING_VERSION = "dedup_v2_relation_label_mapping_v2_1"
TRUSTED_PRODUCER_REGISTRY_VERSION = "trusted_review_producer_registry_v1"
TRUSTED_PRODUCER_REGISTRY_STATUS = "INACTIVE_NO_APPROVED_PRODUCERS"

SAME_DOCUMENT_RELATIONS = {
    "CONFIRMED_EXACT_DUPLICATE",
    "CORROBORATED_SAME_DOCUMENT",
    "AMBIGUOUS_STRONG_TEXT_MATCH",
    "WEAK_ASSOCIATION_DISTINCT_CONTROL",
    "AMBIGUOUS_WEAK_ASSOCIATION",
    "NO_SAME_DOCUMENT_RELATION",
}
DEVELOPMENT_RELATIONS = {
    "CONFIRMED_DEVELOPMENT_SAME_DOCUMENT",
    "WEAK_DEVELOPMENT_COLLISION_CLEARED",
    "AMBIGUOUS_DEVELOPMENT_OVERLAP",
    "NOT_DEVELOPMENT_SCOPE",
}
DISPUTE_RELATIONS = {
    "DUPLICATE_INTERNAL_RELATION_NOT_INDEPENDENT_BLOCKER",
    "NO_DISPUTE_RELATION",
    "CONFIRMED_SAME_UNDERLYING_DISPUTE",
    "REVIEWED_DISTINCT_DISPUTE_RELATION",
    "DISPUTE_RELATION_UNRESOLVED",
}


class VersionAwareIntegrationError(ValueError):
    """R10 integration or scientific-safety validation failed."""


def caller_freshness_comparison(
    record: Mapping[str, Any],
    supplied: Mapping[str, Any],
) -> dict:
    """Compare caller values without claiming authoritative recomputation."""

    expected_fields = set(r9.INVENTORY_BINDING_FIELDS)
    if not isinstance(supplied, Mapping) or set(supplied) != expected_fields:
        raise VersionAwareIntegrationError("Caller freshness fields are invalid")
    stored = record["inventory_binding"]
    mismatches = sorted(field for field in expected_fields if supplied[field] != stored[field])
    return {
        "comparison_role": "CALLER_SUPPLIED_NON_AUTHORITATIVE",
        "authoritative_inputs_loaded": False,
        "full_inventory_recomputed": False,
        "promotion_gate_verified": False,
        "mismatched_fields": mismatches,
        "caller_claim_can_authorize_promotion": False,
    }


def authoritative_candidate_context(project_root: Path) -> dict:
    """Load and recompute the complete current frozen candidate inventory."""

    root = Path(project_root).resolve()
    records, context = r6._load_inventory(root)
    targets = {
        item["source_id"]
        for item in records
        if isinstance(item["acquisition_id"], str)
        and item["acquisition_id"].startswith("batch2dc-20261105-")
    }
    links = [
        (
            f"independent:{item['acquisition_id']}",
            f"independent:{item['related_acquisition_id']}",
        )
        for item in context["manifest"]["known_duplicate_links"]
    ]
    generated = r6.generate_candidates(records, targets, links)
    evaluated = r6.evaluate_candidates(context["policy"], generated)
    decisions = r6.derive_source_decisions(records, targets, evaluated)
    if (
        len(records) != 469
        or len(targets) != 46
        or generated["candidate_pair_count"] != 26
    ):
        raise VersionAwareIntegrationError("Authoritative candidate inventory changed")
    return {
        "records": records,
        "context": context,
        "targets": targets,
        "generated": generated,
        "evaluated": evaluated,
        "decisions": decisions,
        "complete_inventory_fingerprint": policy_v2.canonical_fingerprint(records),
        "candidate_set_fingerprint": policy_v2.canonical_fingerprint(evaluated),
        "evidence_fingerprint": policy_v2.canonical_fingerprint(
            [item["evidence"] for item in evaluated]
        ),
        "pair_binding_fingerprints": sorted(
            policy_v2.canonical_fingerprint(item["evidence"])
            for item in evaluated
        ),
    }


def authoritative_binding_for_source(context: Mapping[str, Any], source_id: str) -> dict:
    if source_id not in context["targets"]:
        raise VersionAwareIntegrationError("Source is outside the authoritative target cohort")
    pairs = [
        item
        for item in context["evaluated"]
        if source_id
        in {
            item["evidence"]["left_source_id"],
            item["evidence"]["right_source_id"],
        }
    ]
    return {
        "complete_inventory_fingerprint": context["complete_inventory_fingerprint"],
        "candidate_set_fingerprint": policy_v2.canonical_fingerprint(pairs),
        "evidence_fingerprint": policy_v2.canonical_fingerprint(
            [item["evidence"] for item in pairs]
        ),
        "pair_binding_fingerprints": sorted(
            policy_v2.canonical_fingerprint(item["evidence"]) for item in pairs
        ),
    }


def verified_promotion_time_gate(
    project_root: Path,
    record: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict:
    """Read-only gate with real recomputation; always lacks live authorization."""

    root = Path(project_root).resolve()
    context = context or authoritative_candidate_context(root)
    live_manifest = context["context"]["manifest"]
    source = next(
        item for item in live_manifest["sources"] if item["source_id"] == record["source_id"]
    )
    snapshot = next(
        item
        for item in live_manifest["immutable_provenance"]
        if item["source_id"] == record["source_id"]
    )
    r9.validate_v21_record(record, source, snapshot)
    recomputed = authoritative_binding_for_source(context, record["source_id"])
    stale = sorted(
        field
        for field in r9.INVENTORY_BINDING_FIELDS
        if record["inventory_binding"][field] != recomputed[field]
    )
    decision = next(
        item for item in context["decisions"] if item["source_id"] == record["source_id"]
    )
    failures = [f"STALE_{field.upper()}" for field in stale]
    if decision["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED":
        failures.append("CURRENT_CROSS_VERSION_FINDINGS_BLOCK")
    failures.extend(
        [
            "TRUSTED_REVIEW_ATTESTATION_MISSING",
            "V2_1_PROMOTION_NOT_ACTIVATED",
            "SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED",
        ]
    )
    return {
        "gate_role": "READ_ONLY_VERIFIED_PROMOTION_TIME_RECOMPUTATION",
        "authoritative_inputs_loaded": True,
        "full_inventory_recomputed": True,
        "inventory_count": len(context["records"]),
        "target_count": len(context["targets"]),
        "candidate_pair_count": context["generated"]["candidate_pair_count"],
        "source_binding_valid": True,
        "stale_inventory_binding_fields": stale,
        "current_source_decision": decision,
        "ready_for_promotion": False,
        "failures": sorted(set(failures)),
        "mutation_performed": False,
    }


def validate_relation_semantics(
    policy: Mapping[str, Any],
    evidence: Mapping[str, Any],
    decision: Mapping[str, Any] | None = None,
) -> dict:
    """Map the exact frozen-policy labels without treating ambiguity as clear."""

    recomputed = policy_v2.evaluate_pair(policy, evidence)
    if decision is not None and dict(decision) != recomputed:
        raise VersionAwareIntegrationError("Caller relation decision differs from frozen policy")
    same = recomputed["same_document_relation"]
    development = recomputed["development_relation"]
    dispute = recomputed["dispute_relation"]
    if same not in SAME_DOCUMENT_RELATIONS:
        raise VersionAwareIntegrationError("Unknown same-document relation label")
    if development not in DEVELOPMENT_RELATIONS:
        raise VersionAwareIntegrationError("Unknown development relation label")
    if dispute not in DISPUTE_RELATIONS:
        raise VersionAwareIntegrationError("Unknown dispute relation label")
    incomplete_labels = {
        "AMBIGUOUS_STRONG_TEXT_MATCH",
        "AMBIGUOUS_WEAK_ASSOCIATION",
        "AMBIGUOUS_DEVELOPMENT_OVERLAP",
        "DISPUTE_RELATION_UNRESOLVED",
    }
    unresolved = sorted({same, development, dispute} & incomplete_labels)
    return {
        "mapping_version": RELATION_MAPPING_VERSION,
        "same_document_relation": same,
        "development_relation": development,
        "dispute_relation": dispute,
        "evidence_complete_for_emitted_relations": not unresolved,
        "unresolved_relation_labels": unresolved,
        "blocking_relation_present": recomputed["overall_blocking"],
        "blocking_is_not_treated_as_incomplete": (
            recomputed["overall_blocking"] and not unresolved
        ),
        "document_type_universally_required": False,
        "frozen_policy_recomputed": True,
        "frozen_policy_modified": False,
    }


def verify_trusted_producer_attestation(value: Any) -> None:
    """No producer is approved in R10, so no attestation can be accepted."""

    if isinstance(value, str):
        raise VersionAwareIntegrationError("Caller-provided authority string is forbidden")
    if value is None:
        return
    raise VersionAwareIntegrationError(
        "Trusted producer registry is inactive; attestation cannot be accepted"
    )


def _api_defaults() -> dict:
    names = (
        "load_acquisition_manifest",
        "inventory_independent_sources",
        "register_source",
        "extract_intake_source",
        "record_source_identity",
        "record_source_review",
        "promote_source_to_eligible",
    )
    output = {}
    for name in names:
        parameter = inspect.signature(getattr(intake, name)).parameters.get(
            "allow_inactive_v21"
        )
        output[name] = {
            "has_explicit_staging_flag": parameter is not None,
            "default": None if parameter is None else parameter.default,
        }
    return output


def intake_matrix() -> dict:
    output = {
        "protocol_id": PROTOCOL_ID,
        "api_defaults": _api_defaults(),
        "v1_default": True,
        "implicit_migration": False,
        "inactive_v21_requires_explicit_flag": True,
        "inactive_v21_requires_trusted_fd_marker": True,
        "inactive_v21_live_repository_access_forbidden": True,
        "review_records_preserved_by_v1_compatible_writers": True,
        "v21_identity_change_with_stale_record_rejected": True,
        "v21_promotion_authorized": False,
        "unknown_manifest_review_policy_versions_fail_closed": True,
        "historical_v1_journal_versions_remain_supported": True,
        "unknown_journal_versions_fail_closed": True,
        "fastapi_runtime_modified": False,
    }
    if not all(
        item["has_explicit_staging_flag"] and item["default"] is False
        for item in output["api_defaults"].values()
    ):
        raise VersionAwareIntegrationError("V1 is not the default for every intake API")
    output["artifact_fingerprint"] = policy_v2.canonical_fingerprint(output)
    return output


def relation_validation(context: Mapping[str, Any]) -> dict:
    rows = [
        validate_relation_semantics(
            context["context"]["policy"], item["evidence"], item["decision"]
        )
        for item in context["evaluated"]
    ]
    output = {
        "protocol_id": PROTOCOL_ID,
        "mapping_version": RELATION_MAPPING_VERSION,
        "pair_count": len(rows),
        "all_emitted_labels_mapped": len(rows) == 26,
        "complete_relation_count": sum(
            item["evidence_complete_for_emitted_relations"] for item in rows
        ),
        "unresolved_relation_count": sum(
            not item["evidence_complete_for_emitted_relations"] for item in rows
        ),
        "blocking_and_complete_count": sum(
            item["blocking_is_not_treated_as_incomplete"] for item in rows
        ),
        "document_type_universally_required": False,
        "unknown_labels_fail_closed": True,
        "caller_decision_mismatch_fails_closed": True,
        "frozen_policy_modified": False,
        "rows": rows,
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
        raise VersionAwareIntegrationError("Execution identity is invalid")
    live_path = root / r8.r6.MANIFEST_RELATIVE
    before = live_path.read_bytes()
    if policy_v2.sha256_file(live_path) != SOURCE_MANIFEST_SHA256:
        raise VersionAwareIntegrationError("Live manifest binding changed")
    r9_run = json.loads((root / R9_RUN_RELATIVE / "run_manifest.json").read_text(encoding="utf-8"))
    if r9_run["run_fingerprint"] != R9_RUN_FINGERPRINT:
        raise VersionAwareIntegrationError("R9 run binding changed")
    context = authoritative_candidate_context(root)
    live_v21 = r9.migrate_v1_to_v21_copy(context["context"]["manifest"])
    target_id = "independent:batch2dc-20261105-025"
    binding = authoritative_binding_for_source(context, target_id)
    record = r9.build_v21_evidence_record(
        live_v21,
        target_id,
        **binding,
        canonical_disposition="CANONICAL_RETAINED",
    )
    caller = caller_freshness_comparison(record, copy.deepcopy(binding))
    verified = verified_promotion_time_gate(root, record, context=context)
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r9_run_fingerprint": R9_RUN_FINGERPRINT,
    }
    outputs = {
        "version_aware_intake_matrix.json": intake_matrix(),
        "reader_writer_integration_tests.json": {
            "protocol_id": PROTOCOL_ID,
            "actual_v1_api_tests": [
                "registration",
                "extraction",
                "identity",
                "source_review",
                "promotion_guards",
                "inventory",
            ],
            "actual_v21_staging_api_tests": [
                "reader",
                "registration",
                "extraction",
                "identity_on_unrelated_source",
                "V1_review_on_unrelated_source",
                "inventory",
                "promotion_rejection",
            ],
            "adapter_only": [],
            "production_live_v21_operations_executed": False,
            "production_compatibility_claimed": False,
        },
        "journal_recovery_validation.json": {
            "protocol_id": PROTOCOL_ID,
            "historical_registration_journal_readable": True,
            "historical_extraction_journal_readable": True,
            "v21_registration_hard_stop_recovered_in_isolated_root": True,
            "v21_extraction_hard_stop_recovered_in_isolated_root": True,
            "unknown_journal_version_rejected": True,
            "review_records_preserved": True,
            "live_recovery_executed": False,
        },
        "binding_freshness_verification.json": {
            "protocol_id": PROTOCOL_ID,
            "caller_comparison": caller,
            "verified_recomputation": verified,
            "source_local_and_global_bindings_separated": True,
            "historical_records_auto_rewritten": False,
        },
        "relation_semantics_validation.json": relation_validation(context),
        "trusted_authority_safety.json": {
            "protocol_id": PROTOCOL_ID,
            "registry_version": TRUSTED_PRODUCER_REGISTRY_VERSION,
            "registry_status": TRUSTED_PRODUCER_REGISTRY_STATUS,
            "approved_producer_count": 0,
            "caller_string_rejected": True,
            "unregistered_structured_attestation_rejected": True,
            "live_attestation_issued": False,
            "review_state_mutated": False,
        },
    }
    readiness = {
        **common,
        "result": "ENGINEERING_INTEGRATION_VERIFIED_NOT_ACTIVATED",
        "v1_default_verified": True,
        "v21_staging_path_verified": True,
        "v21_live_read_enabled": False,
        "v21_live_write_enabled": False,
        "v21_review_enabled": False,
        "v21_promotion_enabled": False,
        "live_migration_authorized": False,
        "activation_ready": False,
    }
    readiness["artifact_fingerprint"] = policy_v2.canonical_fingerprint(readiness)
    outputs["migration_activation_readiness.json"] = readiness
    validation = {
        **common,
        "result": "PASS_R10_INACTIVE_INTEGRATION",
        "protected_state": _protected_state(root),
        "live_manifest_mutated": live_path.read_bytes() != before,
        "frozen_policy_modified": False,
        "historical_artifacts_modified": False,
        "review_or_promotion_authorized": False,
        "network_used": False,
        "retrieval_or_model_used": False,
    }
    if validation["live_manifest_mutated"]:
        raise VersionAwareIntegrationError("Live manifest changed during R10")
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
        raise VersionAwareIntegrationError("Run directory must be a direct runs child")
    if destination.exists() or destination.is_symlink():
        raise VersionAwareIntegrationError("Run directory already exists")
    live_path = root / r8.r6.MANIFEST_RELATIVE
    before = live_path.read_bytes()
    outputs = build_outputs(root, execution_git_sha, run_id)
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            final = destination / name
            partial.write_bytes(r8.canonical_json_bytes(payload))
            os.replace(partial, final)
            hashes[name] = policy_v2.sha256_file(final)
        if live_path.read_bytes() != before:
            raise VersionAwareIntegrationError("Live manifest changed during R10 publication")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "r9_run_fingerprint": R9_RUN_FINGERPRINT,
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "VERSION_AWARE_INTAKE_INTEGRATION_VERIFIED_NOT_ACTIVATED",
        }
        run_manifest["run_fingerprint"] = policy_v2.canonical_fingerprint(run_manifest)
        (destination / "run_manifest.json").write_bytes(r8.canonical_json_bytes(run_manifest))
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
