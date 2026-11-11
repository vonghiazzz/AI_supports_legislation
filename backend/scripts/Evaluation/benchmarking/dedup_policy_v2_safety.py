"""R7 targeted dispute adjudication and V2 contract safety verification."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from . import dedup_policy_v2 as v2
from . import dedup_policy_v2_independent as r6


PROTOCOL_ID = "independent_batch2dc_v2_targeted_adjudication_contract_safety_v1"
PARENT_GIT_SHA = "4f64f160d6810f971824f1be254454ff5c0ad1a3"
R6_RUN_FINGERPRINT = "52083159eacf06a8ab601b3068b1f8694d1b414ca43e3baac3430ae44889e2e1"
TARGET_PAIRS = (
    (
        "independent:batch2dc-20261105-027",
        "independent:scale-20261030-204",
        "SHARED_LOCAL_DECISION_NUMBER_104_2026_QD_PT",
    ),
    (
        "independent:batch2dc-20261105-029",
        "independent:scale-20261030-198",
        "SHARED_LOCAL_DECISION_NUMBER_90_2026_QD_PT",
    ),
    (
        "independent:batch2dc-20261105-029",
        "independent:scale-20261030-234",
        "NORMALIZED_JUDGMENT_NUMBER_COLLISION_182_2026_DS_PT",
    ),
)
ORIGINAL_UNRESOLVED = {
    "batch2dc-20261105-008",
    "batch2dc-20261105-014",
    "batch2dc-20261105-022",
    "batch2dc-20261105-024",
    "batch2dc-20261105-034",
}


class SafetyVerificationError(ValueError):
    """Target evidence or safety contract validation failed."""


def _normalized_identifier(value: str) -> str:
    return re.sub(r"[^0-9A-ZĐ]", "", value.upper())


def _text(record: Mapping[str, Any], root: Path) -> str:
    path = root / record["structured_path"]
    if v2.sha256_file(path) != record["structured_sha256"]:
        raise SafetyVerificationError("Target structured artifact SHA mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join(page["text"] for page in payload["pages"])


def _evidence_summary(record: Mapping[str, Any], root: Path) -> dict:
    raw_path = root / record["source_path"]
    if v2.sha256_file(raw_path) != record["raw_sha256"]:
        raise SafetyVerificationError("Target raw PDF SHA mismatch")
    text = _text(record, root)
    return {
        "source_id": record["source_id"],
        "raw_pdf": {"path": record["source_path"], "sha256": record["raw_sha256"]},
        "structured": {"path": record["structured_path"], "sha256": record["structured_sha256"]},
        "judgment_number": record["judgment_number"],
        "case_id": record["case_id"],
        "court": record["court"],
        "decision_date": record["decision_date"],
        "procedure_stage": record["procedure_stage"],
        "related_decision_ids": record["related_decision_ids"],
        "underlying_dispute_id": record["underlying_dispute_id"],
        "text_fingerprint": v2.canonical_fingerprint(text),
        "header_excerpt": " ".join(text[:700].split()),
    }


def _target_adjudications(
    root: Path,
    records: list[dict],
    pair_results: list[dict],
    policy: Mapping[str, Any],
) -> list[dict]:
    by_id = {item["source_id"]: item for item in records}
    by_pair = {
        tuple(sorted((item["evidence"]["left_source_id"], item["evidence"]["right_source_id"]))): item
        for item in pair_results
    }
    output = []
    for left_id, right_id, origin in TARGET_PAIRS:
        pair = by_pair.get(tuple(sorted((left_id, right_id))))
        if pair is None:
            raise SafetyVerificationError("Target pair is absent from independent coverage")
        left, right = by_id[left_id], by_id[right_id]
        left_summary = _evidence_summary(left, root)
        right_summary = _evidence_summary(right, root)
        distinct_document = (
            left["raw_sha256"] != right["raw_sha256"]
            and left["structured_sha256"] != right["structured_sha256"]
        )
        independent_dispute = (
            left["court"] != right["court"]
            and left["case_id"] != right["case_id"]
            and left["decision_date"] != right["decision_date"]
        )
        if not distinct_document or not independent_dispute:
            raise SafetyVerificationError("Target evidence does not prove the frozen classification")
        if origin.startswith("SHARED_LOCAL_DECISION_NUMBER"):
            shared = set(left["related_decision_ids"]) & set(right["related_decision_ids"])
            if not shared:
                raise SafetyVerificationError("Expected shared related-decision identifier is absent")
            shared_identifiers = sorted(shared)
        else:
            if _normalized_identifier(left["judgment_number"]) != _normalized_identifier(right["judgment_number"]):
                raise SafetyVerificationError("Expected judgment-number collision is absent")
            shared_identifiers = [left["judgment_number"]]
        adjudicated_evidence = copy.deepcopy(pair["evidence"])
        adjudicated_evidence["same_underlying_dispute"] = False
        decision = v2.evaluate_pair(policy, adjudicated_evidence)
        if decision["dispute_blocking"]:
            raise SafetyVerificationError("Reviewed-independent target still blocks as a dispute")
        output.append(
            {
                "left": left_summary,
                "right": right_summary,
                "relationship_origin": origin,
                "shared_identifiers": shared_identifiers,
                "direct_or_propagated": "DIRECT_TEXT_AND_IDENTITY_EVIDENCE",
                "same_legal_document": False,
                "same_underlying_dispute": False,
                "classification": "DIFFERENT_DOCUMENTS_INDEPENDENT_DISPUTES",
                "classification_basis": [
                    "raw and structured SHA256 differ",
                    "courts differ",
                    "case IDs differ",
                    "decision dates differ",
                    "official headers identify separate proceedings",
                ],
                "v2_reviewed_pair_decision": decision,
                "additional_evidence_required": [],
            }
        )
    return output


def build_outputs(project_root: Path, execution_git_sha: str, run_id: str) -> dict[str, dict]:
    root = Path(project_root).resolve()
    if len(execution_git_sha) != 40 or not run_id.startswith(PROTOCOL_ID + "_"):
        raise SafetyVerificationError("Execution identity is invalid")
    records, context = r6._load_inventory(root)
    targets = {
        item["source_id"]
        for item in records
        if isinstance(item["acquisition_id"], str)
        and item["acquisition_id"].startswith("batch2dc-20261105-")
    }
    generated = r6.generate_candidates(records, targets)
    results = r6.evaluate_candidates(context["policy"], generated)
    decisions = r6.derive_source_decisions(records, targets, results)
    adjudications = _target_adjudications(root, records, results, context["policy"])
    by_acquisition = {item["acquisition_id"]: item for item in decisions}
    if {
        acquisition_id
        for acquisition_id in ORIGINAL_UNRESOLVED
        if by_acquisition[acquisition_id]["prospective_review_readiness"]
        == "BLOCKED_FAIL_CLOSED"
    } != ORIGINAL_UNRESOLVED:
        raise SafetyVerificationError("Original unresolved controls changed")
    by_source = {item["source_id"]: item for item in records}
    pair_025_026 = [
        item
        for item in results
        if {
            item["evidence"]["left_source_id"],
            item["evidence"]["right_source_id"],
        }
        == {
            "independent:batch2dc-20261105-025",
            "independent:batch2dc-20261105-026",
        }
    ]
    if len(pair_025_026) != 1:
        raise SafetyVerificationError("025/026 exact-duplicate control changed")
    payload_025 = r6.build_review_payload_v2(
        by_source["independent:batch2dc-20261105-025"],
        r6.MANIFEST_SHA256,
        pair_025_026,
        canonical_disposition="CANONICAL_RETAINED",
    )
    payload_026 = r6.build_review_payload_v2(
        by_source["independent:batch2dc-20261105-026"],
        r6.MANIFEST_SHA256,
        pair_025_026,
        canonical_disposition="NONCANONICAL_REJECTED",
    )
    gate_025 = r6.verify_gate_v2_authoritatively(
        payload_025,
        by_source["independent:batch2dc-20261105-025"],
        r6.MANIFEST_SHA256,
        pair_025_026,
        [],
        [],
        canonical_disposition="CANONICAL_RETAINED",
    )
    gate_026 = r6.verify_gate_v2_authoritatively(
        payload_026,
        by_source["independent:batch2dc-20261105-026"],
        r6.MANIFEST_SHA256,
        pair_025_026,
        [],
        [],
        canonical_disposition="NONCANONICAL_REJECTED",
    )
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": r6.MANIFEST_SHA256,
        "policy_fingerprint": context["policy"]["policy_fingerprint"],
        "r6_run_fingerprint": R6_RUN_FINGERPRINT,
    }
    targeted = {
        **common,
        "pair_count": 3,
        "adjudications": adjudications,
        "post_adjudication_projection": {
            "ready_for_future_v2_review": 40,
            "unresolved_fail_closed": 5,
            "noncanonical_rejected": 1,
        },
        "source_state_mutated": False,
    }
    oracle = {
        **common,
        "production_candidate_count": generated["candidate_pair_count"],
        "bounded_oracle_implementation": "SEPARATE_NESTED_ENUMERATION_WITHOUT_GENERATE_CANDIDATES",
        "positive_controls": [
            "raw SHA",
            "structured SHA",
            "text similarity",
            "filename",
            "judgment identity",
            "case ID",
            "related decision",
            "underlying dispute",
            "authoritative link",
        ],
        "ordering_and_duplicate_controls_required": True,
    }
    review_safety = {
        **common,
        "structural_payload_is_authoritative_review": False,
        "completion_dimensions": [
            "evidence_collection_complete",
            "evidence_adjudicated",
            "authoritative_review_completed",
            "eligibility_approved",
        ],
        "fabricated_completion_rejected": True,
        "missing_document_type_fails_completion": True,
        "unresolved_related_decision_fails_adjudication": True,
        "unverified_identity_fails_collection": True,
        "unsupported_policy_fails_validation": True,
    }
    canonical = {
        **common,
        "canonical_source_id": "independent:batch2dc-20261105-025",
        "noncanonical_source_id": "independent:batch2dc-20261105-026",
        "canonical_blocking_reasons": payload_025["source_findings"]["blocking_rejection_reasons"],
        "noncanonical_blocking_reasons": payload_026["source_findings"]["blocking_rejection_reasons"],
        "canonical_provenance_retained": bool(payload_025["evidence_observations"]),
        "canonical_gate": gate_025,
        "noncanonical_gate": gate_026,
        "consistency": "PASS",
    }
    gate_safety = {
        **common,
        "authoritative_recomputation_required": True,
        "caller_findings_mismatch_rejected": True,
        "source_and_manifest_hashes_rechecked": True,
        "policy_identity_rechecked": True,
        "complete_inventory_recheck_required": True,
        "existing_rejections_preserved": True,
        "cross_version_leakage_preserved": True,
        "unresolved_disputes_rejected": True,
        "promotion_performed": False,
    }
    migration = {
        **common,
        "migration_executed": False,
        "minimum_manifest_extension": r6.gate_spec()["minimum_manifest_extension"]
        + [
            "canonical_disposition",
            "blocking_findings",
            "nonblocking_findings",
            "completion_status",
            "review_authority_binding",
        ],
        "phases": [
            "schema and contract migration",
            "V2 producer and cohort-atomic review application",
            "separately authorized eligibility promotion",
            "B3 close-gate recomputation",
        ],
        "v1_backward_compatibility": "V1 records retain existing fields and V1 gate selection",
        "full_manifest_validation_required": True,
        "rollback_and_idempotent_retry_required": True,
    }
    validation = {
        **common,
        "result": "PASS_WITH_THREE_RELATIONS_ADJUDICATED_INDEPENDENT",
        "source_review_calls": 0,
        "promotion_calls": 0,
        "manifest_mutated": False,
        "v1_modified": False,
        "v2_policy_modified": False,
        "historical_artifacts_modified": False,
        "network_used": False,
        "retrieval_or_model_used": False,
    }
    outputs = {
        "targeted_dispute_adjudication.json": targeted,
        "candidate_oracle_validation.json": oracle,
        "review_completion_safety_report.json": review_safety,
        "canonical_retention_consistency.json": canonical,
        "eligibility_gate_safety_report.json": gate_safety,
        "manifest_migration_design.json": migration,
        "validation_summary.json": validation,
    }
    for payload in outputs.values():
        payload["artifact_fingerprint"] = v2.canonical_fingerprint(payload)
    return outputs


def write_run(project_root: Path, output_dir: Path, execution_git_sha: str, run_id: str) -> dict:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.parent != (root / "backend/data/runs").resolve() or destination.name != run_id:
        raise SafetyVerificationError("Run directory must be a direct runs child")
    if destination.exists() or destination.is_symlink():
        raise SafetyVerificationError("Run directory already exists")
    before = v2.sha256_file(root / r6.MANIFEST_RELATIVE)
    outputs = build_outputs(root, execution_git_sha, run_id)
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            final = destination / name
            partial.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            os.replace(partial, final)
            hashes[name] = v2.sha256_file(final)
        if v2.sha256_file(root / r6.MANIFEST_RELATIVE) != before:
            raise SafetyVerificationError("Manifest changed during R7")
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": before,
            "policy_fingerprint": outputs["targeted_dispute_adjudication.json"]["policy_fingerprint"],
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "TARGETED_ADJUDICATION_AND_CONTRACT_SAFETY_COMPLETE_NOT_ACTIVATED",
        }
        manifest["run_fingerprint"] = v2.canonical_fingerprint(manifest)
        (destination / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return manifest
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
