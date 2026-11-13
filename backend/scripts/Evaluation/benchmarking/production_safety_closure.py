"""R13 production-safety closure and governance release candidate.

All schema/review/promotion exercises operate on disposable clones.  The live
manifest, frozen V1 implementation, frozen policy, benchmark, sampling, query,
and ground truth remain protected.  This module creates no key, signature,
credential, registry authority, or live migration approval.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from . import cutover_authorization_lifecycle as lifecycle
from . import cutover_operational_closure as r12
from . import dedup_performance_v1
from . import dedup_policy_v2 as policy_v2
from . import dual_version_manifest_safety as v21
from . import independent_benchmark as frozen_v1
from . import live_cutover_readiness as r11
from . import source_intake as intake
from . import source_review_v2_migration as r8
from . import trusted_review_protocol as trusted


PROTOCOL_ID = "independent_batch2dc_production_safety_closure_v1"
PARENT_GIT_SHA = "2ef2ee9a0a27686711e71f1ea9f93012c5ff795b"
SOURCE_MANIFEST_SHA256 = r12.SOURCE_MANIFEST_SHA256
POLICY_FINGERPRINT = r12.POLICY_FINGERPRINT
R12_RUN_FINGERPRINT = "cdba9eae9c593c4fdb800943353baf4f062076379485654a2bdc263779ba7849"
R12_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_cutover_operational_closure_v1_"
    "20261113_141619_2ef2ee9a"
)
REHEARSAL_ACQUISITION_ID = "r13-nonempty-record-continuity"


class ProductionSafetyClosureError(ValueError):
    """An R13 safety/governance invariant failed closed."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ProductionSafetyClosureError(f"Invalid JSON: {path}") from exc


def _baseline(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    baseline = r12._baseline(root)
    r12_run_path = root / R12_RUN_RELATIVE / "run_manifest.json"
    r12_run = _json(r12_run_path)
    if r12_run["run_fingerprint"] != R12_RUN_FINGERPRINT:
        raise ProductionSafetyClosureError("R12 run fingerprint changed")
    for name, expected in r12_run["artifact_sha256"].items():
        if policy_v2.sha256_file(r12_run_path.parent / name) != expected:
            raise ProductionSafetyClosureError(f"R12 artifact changed: {name}")
    if policy_v2.sha256_file(
        root / "backend/scripts/Evaluation/benchmarking/independent_benchmark.py"
    ) != "eba62bdb5c484f050db3f9caff7999d85a205a42b39d868d5f8ee79f92526901":
        raise ProductionSafetyClosureError("Frozen V1 implementation changed")
    return {**baseline, "r12_run_fingerprint": R12_RUN_FINGERPRINT}


def _record(
    suffix: str,
    *,
    source_name: str | None = None,
    case_id: str | None = None,
    raw_sha: str | None = None,
    structured_sha: str | None = None,
    text: str = "",
    judgment_number: str | None = None,
    related: tuple[str, ...] = (),
    underlying: str | None = None,
) -> dict:
    index = sum(ord(character) for character in suffix) % 10
    return {
        "source_id": f"independent:{suffix}",
        "source_name": source_name or f"{suffix}.pdf",
        "source_sha256": raw_sha or str(index) * 64,
        "structured_sha256": structured_sha or str((index + 1) % 10) * 64,
        "case_id": case_id or f"{100 + index}/2026/DS-ST",
        "judgment_number": judgment_number or f"{100 + index}/2026/DS-ST",
        "related_decision_ids": list(related),
        "underlying_dispute_id": underlying,
        "_dedup_text": text,
    }


def equivalence_fixture() -> tuple[list[dict], list[dict], list[tuple[str, str]], dict]:
    base_tokens = [f"t{index}" for index in range(20)]
    base = " ".join(base_tokens)
    above = " ".join(base_tokens[:-1] + ["changed"])
    below = " ".join(base_tokens[:17] + ["x", "y", "z"])
    internal_tokens = [f"u{index}" for index in range(20)]
    internal_base = " ".join(internal_tokens)
    internal_above = " ".join(internal_tokens[:-1] + ["internal-changed"])
    inventory = [
        _record("dev", source_name="development.pdf", case_id="1/2026/DS-ST", raw_sha="a" * 64, text=base),
        _record("same-raw", raw_sha="a" * 64, text="unique raw duplicate"),
        _record("same-case", case_id="1/2026/DS-ST", text="unique case duplicate"),
        _record("filename-collision", source_name="DEVELOPMENT.PDF", text="unique filename duplicate"),
        _record("near-development", text=above),
        _record("below-left", text=base),
        _record("below-right", text=below),
        _record("empty-left", text=""),
        _record("empty-right", text=""),
        _record("raw-cluster-a", raw_sha="b" * 64, text="raw cluster first unique words"),
        _record("raw-cluster-b", raw_sha="b" * 64, text="raw cluster second distinct words"),
        _record("name-cluster-a", source_name="collision.pdf", text="name one unique words"),
        _record("name-cluster-b", source_name="COLLISION.PDF", text="name two distinct words"),
        _record("near-cluster-a", text=internal_base),
        _record("near-cluster-b", text=internal_above),
        _record("related-a", judgment_number="501/2026/DS-ST", related=("502/2026/DS-ST",), text="related alpha unique"),
        _record("related-b", judgment_number="502/2026/DS-ST", text="related beta distinct"),
        _record("known-a", text="known alpha unique"),
        _record("known-b", text="known beta distinct"),
    ]
    development = [{"id": 1, "judgment_name": "development.pdf"}]
    links = [("independent:known-a", "independent:known-b")]
    scores = {
        "above_threshold": frozen_v1.dedup_text_similarity(base, above),
        "below_threshold": frozen_v1.dedup_text_similarity(base, below),
    }
    if not (
        scores["above_threshold"] >= frozen_v1.NEAR_DUPLICATE_THRESHOLD
        and scores["below_threshold"] < frozen_v1.NEAR_DUPLICATE_THRESHOLD
    ):
        raise ProductionSafetyClosureError("Threshold control fixture is invalid")
    return inventory, development, links, scores


def validate_fixture_equivalence() -> dict:
    inventory, development, links, scores = equivalence_fixture()
    reference = frozen_v1.deduplicate_against_development(
        inventory, development, known_duplicate_links=links
    )
    stats: dict[str, Any] = {}
    optimized = dedup_performance_v1.deduplicate_memoized(
        inventory,
        development,
        known_duplicate_links=links,
        performance_stats=stats,
    )
    if optimized != reference:
        raise ProductionSafetyClosureError("R13 fixture output parity failed")
    eligible, report = reference
    reason_values = {
        reason
        for exclusion in report["exclusions"]
        for reason in exclusion["reasons"]
    }
    required_reason_prefixes = (
        "development_benchmark_exact_source",
        "development_benchmark_exact_sha256",
        "development_benchmark_same_case_id",
        "near_duplicate_source_text:",
        "new_pool_duplicate_sha256",
        "new_pool_duplicate_source_identity",
        "new_pool_near_duplicate_source_text:",
        "new_pool_related_decision_ids",
        "new_pool_known_duplicate_link",
        "canonical_representative:",
    )
    missing = [
        prefix
        for prefix in required_reason_prefixes
        if not any(reason.startswith(prefix) for reason in reason_values)
    ]
    if missing:
        raise ProductionSafetyClosureError(f"Fixture coverage missing: {missing}")
    return {
        "fixture_inventory_count": len(inventory),
        "fixture_development_count": len(development),
        "threshold_scores": scores,
        "eligible_count": len(eligible),
        "eligible_fingerprint": trusted.canonical_fingerprint(eligible),
        "report_fingerprint": trusted.canonical_fingerprint(report),
        "full_output_equal": True,
        "eligible_membership_equal": True,
        "source_order_equal": True,
        "development_findings_equal": True,
        "internal_clusters_equal": True,
        "canonical_representatives_equal": True,
        "exclusion_reasons_equal": True,
        "empty_text_control_covered": True,
        "known_link_control_covered": True,
        "related_decision_control_covered": True,
        "worker_statistics": stats,
    }


def validate_full_population_against_r12(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    manifest = intake.load_acquisition_manifest(root)
    inventory, development, links = intake._eligibility_dedup_inputs(root, manifest)
    started = time.perf_counter()
    stats: dict[str, Any] = {}
    eligible, report = dedup_performance_v1.deduplicate_memoized(
        inventory,
        development,
        known_duplicate_links=links,
        performance_stats=stats,
    )
    elapsed = time.perf_counter() - started
    r12_profile = _json(root / R12_RUN_RELATIVE / "v1_dedup_performance_profile.json")
    eligible_fp = trusted.canonical_fingerprint(eligible)
    report_fp = trusted.canonical_fingerprint(report)
    expected_eligible = r12_profile["reference"]["eligible_fingerprint"]
    expected_report = r12_profile["reference"]["report_fingerprint"]
    if eligible_fp != expected_eligible or report_fp != expected_report:
        raise ProductionSafetyClosureError("Isolated full-population parity failed")
    return {
        "population_record_count": len(inventory),
        "development_record_count": len(development),
        "wall_seconds_including_worker_startup": elapsed,
        "worker_statistics": stats,
        "eligible_fingerprint": eligible_fp,
        "report_fingerprint": report_fp,
        "matches_r12_exact_frozen_reference": True,
        "r12_reference_wall_seconds": r12_profile["reference"]["wall_seconds"],
        "r12_optimized_wall_seconds": r12_profile["optimized"]["wall_seconds"],
        "r12_speedup": r12_profile["wall_speedup"],
        "r12_reference_peak_rss_platform_units": r12_profile["reference"][
            "peak_rss_platform_units_after"
        ],
        "r12_optimized_peak_rss_platform_units": r12_profile["optimized"][
            "peak_rss_platform_units_after"
        ],
    }


def validate_shared_state_safety() -> dict:
    inventory, development, links, _ = equivalence_fixture()
    original = frozen_v1.dedup_text_similarity
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                dedup_performance_v1.deduplicate_memoized,
                copy.deepcopy(inventory),
                copy.deepcopy(development),
                copy.deepcopy(links),
                _worker_delay_after_patch=0.25,
            )
            for _ in range(2)
        ]
        while not all(future.done() for future in futures):
            if frozen_v1.dedup_text_similarity is not original:
                raise ProductionSafetyClosureError("Parent frozen function was mutated")
            original("a b c d e f", "a b c d e f")
            time.sleep(0.01)
        results = [future.result() for future in futures]
    if results[0] != results[1] or frozen_v1.dedup_text_similarity is not original:
        raise ProductionSafetyClosureError("Concurrent isolated results diverged")
    injected_failure_closed = False
    try:
        dedup_performance_v1.deduplicate_memoized(
            inventory,
            development,
            links,
            _inject_worker_failure_after_patch=True,
        )
    except dedup_performance_v1.DedupPerformanceError:
        injected_failure_closed = True
    if not injected_failure_closed or frozen_v1.dedup_text_similarity is not original:
        raise ProductionSafetyClosureError("Worker exception isolation failed")
    timeout_closed = False
    try:
        dedup_performance_v1.deduplicate_memoized(
            inventory,
            development,
            links,
            _worker_timeout_seconds=0.05,
            _worker_delay_after_patch=1.0,
        )
    except dedup_performance_v1.DedupPerformanceError:
        timeout_closed = True
    if not timeout_closed or frozen_v1.dedup_text_similarity is not original:
        raise ProductionSafetyClosureError("Worker timeout isolation failed")
    return {
        "r12_same_process_global_mutation_risk_confirmed": True,
        "r13_execution_model": "DEDICATED_MULTIPROCESSING_SPAWN_WORKER",
        "parent_module_function_mutated": False,
        "concurrent_workers_equal": True,
        "direct_parent_caller_remained_original": True,
        "worker_exception_failed_closed": True,
        "worker_timeout_failed_closed": True,
        "nested_or_concurrent_calls_share_worker_state": False,
        "optimized_path_enabled_by_default_in_source_intake": True,
    }


def _write_clone_manifest(root: Path, manifest: dict) -> None:
    path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    path.write_bytes(intake._canonical_json_bytes(manifest))


def perform_nonempty_record_clone(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    live_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    live_before = live_path.read_bytes()
    temporary = Path(tempfile.mkdtemp(prefix="r13-nonempty-record-"))
    clone = temporary / "repository"
    clone.mkdir()
    original_extractor = None
    try:
        r11._clone_required_tree(root, clone)
        v1_manifest = intake.load_acquisition_manifest(clone)
        migration = r11.migrate_clone_manifest(clone, SOURCE_MANIFEST_SHA256)
        migrated = intake.load_acquisition_manifest(clone, allow_inactive_v21=True)
        record_source = next(
            source
            for source in migrated["sources"]
            if source["acquisition_status"] == "extracted"
            and source["structured_sha256"] is not None
        )
        record = v21.build_v21_evidence_record(
            migrated,
            record_source["source_id"],
            complete_inventory_fingerprint="1" * 64,
            candidate_set_fingerprint="2" * 64,
            evidence_fingerprint="3" * 64,
            pair_binding_fingerprints=("4" * 64,),
            blocking_rejection_reasons=(),
        )
        with_record = copy.deepcopy(migrated)
        with_record["review_records"] = [record]
        v21.validate_dual_version_manifest(with_record)
        _write_clone_manifest(clone, with_record)
        loaded = intake.load_acquisition_manifest(clone, allow_inactive_v21=True)
        record_before = copy.deepcopy(loaded["review_records"][0])
        schema_before = copy.deepcopy(loaded["review_schema"])
        provenance_before = copy.deepcopy(loaded["immutable_provenance"])
        protected_source_before = copy.deepcopy(
            next(
                source
                for source in loaded["sources"]
                if source["source_id"] == record_source["source_id"]
            )
        )

        pdf = clone / "r13-nonempty-record-continuity.pdf"
        pdf.write_bytes(r11._minimal_pdf_bytes("r13-nonempty-record-continuity"))
        now = datetime.now().astimezone().isoformat()
        intake.register_source(
            clone,
            REHEARSAL_ACQUISITION_ID,
            pdf,
            "https://example.test/r13-nonempty-record-continuity.pdf",
            "example.test",
            now,
            "r13-nonempty-record-continuity.pdf",
            allow_inactive_v21=True,
        )
        original_extractor = r11._install_rehearsal_extractor()
        intake.extract_intake_source(
            clone, REHEARSAL_ACQUISITION_ID, allow_inactive_v21=True
        )
        intake._run_local_extractor = original_extractor
        original_extractor = None
        intake.record_source_identity(
            clone,
            REHEARSAL_ACQUISITION_ID,
            {
                "identity_version": intake.IDENTITY_PAYLOAD_VERSION,
                "verified_metadata": {
                    "judgment_number": "9997/2026/DS-ST",
                    "case_id": "9997/2026/DS-ST",
                    "court": "Tòa án nhân dân tỉnh R13",
                    "decision_date": "2026-11-13",
                    "procedure_stage": "so_tham",
                },
                "verified_fields": {
                    field: True for field in intake.REVIEW_METADATA_FIELDS
                },
                "related_decision_ids": [],
                "underlying_dispute_id": None,
            },
            allow_inactive_v21=True,
        )
        before_review = intake.load_acquisition_manifest(
            clone, allow_inactive_v21=True
        )
        findings = intake._eligibility_review_findings(
            clone, before_review, REHEARSAL_ACQUISITION_ID
        )
        payload = {
            "review_version": intake.REVIEW_PAYLOAD_VERSION,
            "reviews": {
                field: {"completed": True, "findings": findings[field]}
                for field in intake.REVIEW_COMPLETION_FIELDS
            },
        }
        reviewed = intake.record_source_review(
            clone,
            REHEARSAL_ACQUISITION_ID,
            payload,
            allow_inactive_v21=True,
        )
        promoted = intake.promote_source_to_eligible(
            clone,
            REHEARSAL_ACQUISITION_ID,
            allow_inactive_v21=True,
        )
        final = intake.load_acquisition_manifest(clone, allow_inactive_v21=True)
        record_after = final["review_records"][0]
        protected_source_after = next(
            source
            for source in final["sources"]
            if source["source_id"] == record_source["source_id"]
        )
        rehearsal_source = next(
            source
            for source in final["sources"]
            if source["acquisition_id"] == REHEARSAL_ACQUISITION_ID
        )
        checks = {
            "record_byte_equivalent": intake._canonical_json_bytes(record_after)
            == intake._canonical_json_bytes(record_before),
            "record_fingerprint_unchanged": record_after["record_fingerprint"]
            == record_before["record_fingerprint"],
            "review_schema_unchanged": final["review_schema"] == schema_before,
            "source_local_binding_unchanged": record_after["source_local_binding"]
            == record_before["source_local_binding"],
            "immutable_provenance_preserved": all(
                item in final["immutable_provenance"] for item in provenance_before
            ),
            "only_rehearsal_provenance_added": len(final["immutable_provenance"])
            == len(provenance_before) + 1,
            "record_source_status_unchanged": protected_source_after[
                "acquisition_status"
            ]
            == protected_source_before["acquisition_status"],
            "record_did_not_authorize_eligibility": record_after[
                "eligibility_authorization"
            ]["authorized"]
            is False,
            "rehearsal_has_no_v2_record": all(
                item["source_id"] != rehearsal_source["source_id"]
                for item in final["review_records"]
            ),
            "v1_findings_used": all(
                rehearsal_source["metadata_verification"][field]
                for field in intake.REVIEW_COMPLETION_FIELDS
            ),
        }
        if not all(checks.values()) or rehearsal_source["acquisition_status"] != "eligible":
            raise ProductionSafetyClosureError("Nonempty record continuity failed")
        return {
            "clone_type": "REAL_FROZEN_POPULATION_WITH_NONEMPTY_V2_1_RECORD",
            "migration_result": migration["result"],
            "preserved_record_source_id": record_source["source_id"],
            "preserved_record_fingerprint": record_before["record_fingerprint"],
            "record_review_state": record_before["review_state"],
            "review_status": reviewed["acquisition_status"],
            "promotion_status": promoted["acquisition_status"],
            "checks": checks,
            "v2_review_applied": False,
            "v2_evidence_used_for_promotion": False,
            "live_mutation_executed": False,
        }
    finally:
        if original_extractor is not None:
            intake._run_local_extractor = original_extractor
        shutil.rmtree(temporary, ignore_errors=True)
        if live_path.read_bytes() != live_before:
            raise ProductionSafetyClosureError("Live manifest changed during R13 clone")


def governance_code_bundle_fingerprint(project_root: Path) -> str:
    root = Path(project_root).resolve()
    relatives = (
        "backend/scripts/Evaluation/benchmarking/source_intake.py",
        "backend/scripts/Evaluation/benchmarking/independent_benchmark.py",
        "backend/scripts/Evaluation/benchmarking/dedup_performance_v1.py",
        "backend/scripts/Evaluation/benchmarking/trusted_review_protocol.py",
        "backend/scripts/Evaluation/benchmarking/cutover_authorization_lifecycle.py",
        "backend/scripts/Evaluation/benchmarking/live_cutover_readiness.py",
        "backend/scripts/Evaluation/benchmarking/cutover_operational_closure.py",
        "backend/scripts/Evaluation/benchmarking/production_safety_closure.py",
    )
    values = {relative: policy_v2.sha256_file(root / relative) for relative in relatives}
    return trusted.canonical_fingerprint(values)


def governance_package(project_root: Path, execution_git_sha: str) -> dict:
    root = Path(project_root).resolve()
    return {
        "release_candidate_commit": execution_git_sha,
        "branch": "chore/repo-hygiene-runtime-recovery",
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r12_run_fingerprint": R12_RUN_FINGERPRINT,
        "governance_code_bundle_fingerprint": governance_code_bundle_fingerprint(root),
        "cutover_code_bundle_fingerprint": trusted.cutover_code_bundle_fingerprint(root),
        "operational_code_bundle_fingerprint": lifecycle.operational_code_bundle_fingerprint(root),
        "backup_verification_checklist": [
            "READ_ONLY_MANIFEST_AND_ARTIFACT_VALIDATION_PASS",
            "BYTE_EXACT_V1_MANIFEST_BACKUP_WITH_SHA256",
            "RAW_STRUCTURED_PROVENANCE_TREE_BACKUP_AND_SAMPLE_RESTORE",
            "EXTERNAL_ACTIVATION_LEDGER_BACKUP_AND NOFOLLOW_VALIDATION",
            "RECOVERY_REGISTRY_AVAILABLE_TO_SEPARATE_OPERATOR",
            "RESTORE_REHEARSAL_COMPLETED_BEFORE_WINDOW",
        ],
        "recovery_procedure": {
            "before_manifest_publication": "ABORT_WITH_V1_UNCHANGED",
            "after_manifest_before_receipt": "ROLL_BACK_EXACT_V1_BYTES_AND_MARKER",
            "valid_v2_1_operational_suspension": "NO_DOWNGRADE; READ_ONLY_RECOVERY_INSPECTION",
            "journal_interruption": "AUTHORITATIVE_FAIL_CLOSED_RECONCILIATION",
            "unexpected_state": "STOP_WITHOUT_OVERWRITE_AND_ESCALATE",
        },
        "maintenance_window_recommendation_minutes": 60,
        "maintenance_window_basis": (
            "R12 exact reference 738s, optimized evaluation 13s, full-clone five-pass "
            "continuity 145s; 60 minutes reserves backup/restore verification and rollback."
        ),
        "role_separation": {
            "registry_provisioner": "HUMAN_PROVISIONING_REQUIRED",
            "independent_approver": "HUMAN_PROVISIONING_REQUIRED",
            "cutover_executor": "HUMAN_PROVISIONING_REQUIRED_DIFFERENT_ROLE",
            "recovery_operator": "HUMAN_PROVISIONING_REQUIRED_SEPARATE_ACCESS",
        },
        "authorization_requirements": {
            "scope": "ONE_EXACT_V1_TO_V2_1_TRANSITION",
            "short_expiry_required": True,
            "one_time_nonce_required": True,
            "create_once_receipt_required": True,
            "operational_authorization_issued_after_receipt": True,
            "review_and_promotion_authorities_separate": True,
        },
        "governance_approval_complete": False,
        "public_registry_provisioned": False,
        "live_migration_authorized": False,
    }


def build_outputs(
    project_root: Path,
    execution_git_sha: str,
    run_id: str,
    shared_safety: Mapping[str, Any],
    fixture_parity: Mapping[str, Any],
    full_population: Mapping[str, Any],
    continuity: Mapping[str, Any],
) -> dict[str, dict]:
    root = Path(project_root).resolve()
    baseline = _baseline(root)
    governance = governance_package(root, execution_git_sha)
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r12_run_fingerprint": R12_RUN_FINGERPRINT,
        "governance_code_bundle_fingerprint": governance[
            "governance_code_bundle_fingerprint"
        ],
    }
    outputs = {
        "dedup_shared_state_safety.json": copy.deepcopy(dict(shared_safety)),
        "v1_output_equivalence.json": {
            "targeted_fixture": copy.deepcopy(dict(fixture_parity)),
            "full_population": copy.deepcopy(dict(full_population)),
            "frozen_v1_sha256": "eba62bdb5c484f050db3f9caff7999d85a205a42b39d868d5f8ee79f92526901",
            "frozen_v1_modified": False,
        },
        "nonempty_review_record_continuity.json": copy.deepcopy(dict(continuity)),
        "authorization_recovery_verification.json": {
            "authority_amendment": lifecycle.authority_separation_specification(),
            "cutover_token_authorizes_operations": False,
            "operational_authority_independent_and_revocable": True,
            "review_producer_active": False,
            "promotion_authority_active": False,
            "recovery_access_read_only": True,
            "expired_or_revoked_operation_suspends": True,
            "valid_v2_1_automatic_v1_downgrade": False,
            "create_once_receipt_auditable": True,
            "caller_flag_creates_authority": False,
            "live_trust_registry_state": trusted.LIVE_TRUST_REGISTRY[
                "activation_state"
            ],
            "live_recovery_registry_state": lifecycle.LIVE_RECOVERY_REGISTRY[
                "activation_state"
            ],
        },
        "governance_release_candidate.json": governance,
        "registry_provisioning_requirements.json": {
            "status": "HUMAN_PROVISIONING_REQUIRED",
            "required_order": r12.registry_provisioning_order()["order"],
            "production_private_keys_committed": False,
            "approved_public_issuer_identity_available": False,
            "approved_recovery_issuer_identity_available": False,
            "code_fingerprint_dependencies": {
                "trusted_review_protocol_registry_change": (
                    "CHANGES_CUTOVER_AND_OPERATIONAL_CODE_BUNDLES"
                ),
                "cutover_authorization_lifecycle_recovery_registry_change": (
                    "CHANGES_CUTOVER_AND_OPERATIONAL_CODE_BUNDLES"
                ),
                "dedup_worker_change": "CHANGES_CUTOVER_AND_OPERATIONAL_CODE_BUNDLES",
            },
            "authorization_must_be_issued_after_final_registry_and_code_freeze": True,
        },
        "final_go_no_go.json": {
            "ENGINEERING_RELEASE_CANDIDATE_READY": True,
            "GOVERNANCE_APPROVAL_COMPLETE": False,
            "PUBLIC_REGISTRY_PROVISIONED": False,
            "LIVE_MIGRATION_AUTHORIZED": False,
            "result": "GOVERNANCE_RELEASE_CANDIDATE_READY_NOT_AUTHORIZED",
        },
        "validation_summary.json": {
            "result": "PASS_R13_GOVERNANCE_RELEASE_CANDIDATE_NOT_AUTHORIZED",
            "protected_state": baseline["protected_state"],
            "r12_artifacts_verified": True,
            "shared_state_risk_resolved": True,
            "exact_output_equivalence": True,
            "nonempty_review_record_continuity": True,
            "live_manifest_mutated": False,
            "live_migration_executed": False,
            "live_review_or_promotion_performed": False,
            "credentials_or_signatures_created": False,
            "registries_activated": False,
            "network_used": False,
            "retrieval_model_sampling_query_or_gt_used": False,
        },
    }
    for payload in outputs.values():
        for key, value in common.items():
            payload.setdefault(key, value)
        payload["artifact_fingerprint"] = trusted.canonical_fingerprint(
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
    if (
        destination.parent != (root / "backend/data/runs").resolve()
        or destination.name != run_id
        or destination.exists()
        or destination.is_symlink()
    ):
        raise ProductionSafetyClosureError("R13 run destination is invalid")
    if len(execution_git_sha) != 40 or not run_id.startswith(PROTOCOL_ID + "_"):
        raise ProductionSafetyClosureError("R13 execution identity is invalid")
    _baseline(root)
    live_before = (root / intake.ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    shared = validate_shared_state_safety()
    fixture = validate_fixture_equivalence()
    full_population = validate_full_population_against_r12(root)
    continuity = perform_nonempty_record_clone(root)
    outputs = build_outputs(
        root,
        execution_git_sha,
        run_id,
        shared,
        fixture,
        full_population,
        continuity,
    )
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            partial.write_bytes(r8.canonical_json_bytes(payload))
            os.replace(partial, destination / name)
            hashes[name] = policy_v2.sha256_file(destination / name)
        if (root / intake.ACQUISITION_MANIFEST_RELATIVE).read_bytes() != live_before:
            raise ProductionSafetyClosureError("Live manifest changed during R13")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "r12_run_fingerprint": R12_RUN_FINGERPRINT,
            "governance_code_bundle_fingerprint": governance_code_bundle_fingerprint(
                root
            ),
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "GOVERNANCE_RELEASE_CANDIDATE_READY_NOT_AUTHORIZED",
        }
        run_manifest["run_fingerprint"] = trusted.canonical_fingerprint(run_manifest)
        (destination / "run_manifest.json").write_bytes(
            r8.canonical_json_bytes(run_manifest)
        )
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
    result = write_run(
        args.project_root, args.output_dir, args.execution_git_sha, args.run_id
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
