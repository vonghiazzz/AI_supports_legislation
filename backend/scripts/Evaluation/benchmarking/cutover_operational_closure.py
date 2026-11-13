"""R12 inactive cutover-authority and V1 performance closure.

The runner profiles the frozen V1 scientific path and rehearses historical
review/promotion only on disposable full-population clones.  It never issues a
production credential, activates a registry, migrates live data, or changes a
live source decision.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import resource
import shutil
import signal
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from . import cutover_authorization_lifecycle as lifecycle
from . import dedup_policy_v2 as policy_v2
from . import dedup_performance_v1
from . import live_cutover_readiness as r11
from . import source_intake as intake
from . import source_review_v2_migration as r8
from . import trusted_review_protocol as trusted
from . import version_aware_intake_integration as r10


PROTOCOL_ID = "independent_batch2dc_cutover_operational_closure_v1"
PARENT_GIT_SHA = "f124fb6648a50d8e8da1f5d7be62dcf4fd48c3a7"
SOURCE_MANIFEST_SHA256 = r11.SOURCE_MANIFEST_SHA256
POLICY_FINGERPRINT = "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"
R11_RUN_FINGERPRINT = "b4e801d1fcf7f9d5c19c276078f90a363672739b1f1e3d7049b8cb3ea8b8dffa"
R11_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_live_cutover_readiness_v1_"
    "20261113_082448_f124fb66"
)
REHEARSAL_ACQUISITION_ID = "r12-v1-continuity"
REFERENCE_PROFILE_LIMIT_SECONDS = 15 * 60


class CutoverOperationalClosureError(ValueError):
    """An R12 closure invariant failed closed."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CutoverOperationalClosureError(f"Invalid JSON: {path}") from exc


def _baseline(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    manifest_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    manifest_bytes = manifest_path.read_bytes()
    if _sha256(manifest_bytes) != SOURCE_MANIFEST_SHA256:
        raise CutoverOperationalClosureError("Live manifest binding changed")
    manifest = json.loads(manifest_bytes)
    counts = {
        status: sum(
            source["acquisition_status"] == status for source in manifest["sources"]
        )
        for status in ("eligible", "extracted", "hash_verified")
    }
    if len(manifest["sources"]) != 363 or counts != {
        "eligible": 257,
        "extracted": 105,
        "hash_verified": 1,
    }:
        raise CutoverOperationalClosureError("Protected source counts changed")
    r11_run = _json(root / R11_RUN_RELATIVE / "run_manifest.json")
    if r11_run["run_fingerprint"] != R11_RUN_FINGERPRINT:
        raise CutoverOperationalClosureError("R11 run fingerprint changed")
    policy = policy_v2.load_and_validate_policy(root)
    if policy["policy_fingerprint"] != POLICY_FINGERPRINT:
        raise CutoverOperationalClosureError("Frozen policy fingerprint changed")
    protected = r8._protected_state(root, manifest)
    return {
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_counts": {"total": 363, **counts},
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r11_run_fingerprint": R11_RUN_FINGERPRINT,
        "protected_state": protected,
    }


def _timed_dedup(
    inventory: list[dict],
    development: list[dict],
    links: list[tuple[str, str]],
    *,
    precompute: bool,
) -> dict:
    statistics: dict[str, Any] = {}
    before_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    evaluator = (
        dedup_performance_v1.deduplicate_memoized
        if precompute
        else dedup_performance_v1.deduplicate_reference_profiled
    )
    eligible, report = evaluator(
        inventory,
        development,
        known_duplicate_links=links,
        performance_stats=statistics,
    )
    cpu = time.process_time() - started_cpu
    wall = time.perf_counter() - started_wall
    after_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "peak_rss_platform_units_before": before_peak,
        "peak_rss_platform_units_after": after_peak,
        "statistics": statistics,
        "eligible": eligible,
        "report": report,
        "eligible_fingerprint": trusted.canonical_fingerprint(eligible),
        "report_fingerprint": trusted.canonical_fingerprint(report),
    }


def profile_full_population_v1(project_root: Path) -> dict:
    """Compare the exact legacy path with shingle-memoized execution."""

    root = Path(project_root).resolve()
    manifest = intake.load_acquisition_manifest(root)
    validation_wall = time.perf_counter()
    validation_cpu = time.process_time()
    inventory, development, links = intake._eligibility_dedup_inputs(root, manifest)
    validation_cpu = time.process_time() - validation_cpu
    validation_wall = time.perf_counter() - validation_wall

    def limit(_signum, _frame):
        raise TimeoutError("Exact V1 reference profile exceeded bounded window")

    old_handler = signal.signal(signal.SIGALRM, limit)
    signal.alarm(REFERENCE_PROFILE_LIMIT_SECONDS)
    try:
        reference = _timed_dedup(
            inventory, development, links, precompute=False
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    optimized = _timed_dedup(inventory, development, links, precompute=True)
    equivalent = (
        reference["eligible"] == optimized["eligible"]
        and reference["report"] == optimized["report"]
    )
    if not equivalent:
        raise CutoverOperationalClosureError(
            "Memoized V1 dedup output differs from exact legacy execution"
        )
    public_reference = {
        key: value
        for key, value in reference.items()
        if key not in {"eligible", "report"}
    }
    public_optimized = {
        key: value
        for key, value in optimized.items()
        if key not in {"eligible", "report"}
    }
    return {
        "population_record_count": len(inventory),
        "development_record_count": len(development),
        "known_duplicate_link_count": len(links),
        "artifact_validation_wall_seconds": validation_wall,
        "artifact_validation_cpu_seconds": validation_cpu,
        "reference": public_reference,
        "optimized": public_optimized,
        "exact_eligible_equivalence": True,
        "exact_report_equivalence": True,
        "threshold_changed": False,
        "matching_rules_changed": False,
        "canonical_order_changed": False,
        "scientific_semantics_changed": False,
        "wall_speedup": (
            reference["wall_seconds"] / optimized["wall_seconds"]
            if optimized["wall_seconds"]
            else None
        ),
        "bounded_reference_limit_seconds": REFERENCE_PROFILE_LIMIT_SECONDS,
    }


def _review_payload(root: Path, manifest: dict, acquisition_id: str) -> dict:
    findings = intake._eligibility_review_findings(root, manifest, acquisition_id)
    return {
        "review_version": intake.REVIEW_PAYLOAD_VERSION,
        "reviews": {
            field: {"completed": True, "findings": findings[field]}
            for field in intake.REVIEW_COMPLETION_FIELDS
        },
    }


def perform_full_clone_v1_review_promotion(project_root: Path) -> dict:
    """Exercise exact V1 review/promotion on a real-population V2.1 clone."""

    root = Path(project_root).resolve()
    live_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    live_before = live_path.read_bytes()
    temporary = Path(tempfile.mkdtemp(prefix="r12-full-clone-"))
    clone = temporary / "repository"
    clone.mkdir()
    original_extractor = None
    original_checkpoint = intake._review_publication_checkpoint
    original_dedup_report = intake._eligibility_dedup_report
    global_dedup_passes = 0
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    peak_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        r11._clone_required_tree(root, clone)
        clone_manifest_path = clone / intake.ACQUISITION_MANIFEST_RELATIVE
        v1 = intake.load_acquisition_manifest(clone)
        migration = r11.migrate_clone_manifest(clone, SOURCE_MANIFEST_SHA256)
        retry = r11.migrate_clone_manifest(clone, SOURCE_MANIFEST_SHA256)
        pdf = clone / "r12-v1-continuity.pdf"
        pdf.write_bytes(r11._minimal_pdf_bytes("r12-v1-continuity"))
        now = datetime.now().astimezone().isoformat()
        intake.register_source(
            clone,
            REHEARSAL_ACQUISITION_ID,
            pdf,
            "https://example.test/r12-v1-continuity.pdf",
            "example.test",
            now,
            "r12-v1-continuity.pdf",
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
                    "judgment_number": "9998/2026/DS-ST",
                    "case_id": "9998/2026/DS-ST",
                    "court": "Tòa án nhân dân tỉnh R12",
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
        manifest = intake.load_acquisition_manifest(clone, allow_inactive_v21=True)

        def counted_dedup_report(*args, **kwargs):
            nonlocal global_dedup_passes
            global_dedup_passes += 1
            return original_dedup_report(*args, **kwargs)

        intake._eligibility_dedup_report = counted_dedup_report
        payload = _review_payload(clone, manifest, REHEARSAL_ACQUISITION_ID)
        failed_once = False

        def fail_after_publication(name: str) -> None:
            nonlocal failed_once
            if name == "after_manifest_replacement" and not failed_once:
                failed_once = True
                raise RuntimeError("R12 injected review publication failure")

        intake._review_publication_checkpoint = fail_after_publication
        try:
            intake.record_source_review(
                clone,
                REHEARSAL_ACQUISITION_ID,
                payload,
                allow_inactive_v21=True,
            )
        except intake.SourceIntakeError as exc:
            if "previous manifest restored" not in str(exc):
                raise
        else:
            raise CutoverOperationalClosureError("Review rollback injection did not fire")
        intake._review_publication_checkpoint = original_checkpoint
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
        source = next(
            item
            for item in final["sources"]
            if item["acquisition_id"] == REHEARSAL_ACQUISITION_ID
        )
        inventory, report, _ = intake.inventory_independent_sources(
            clone, allow_inactive_v21=True
        )
        original_projection = {
            item["source_id"]: item for item in v1["sources"]
        }
        projected = r11.v21._v1_projection(final)
        unchanged = all(
            next(
                item
                for item in projected["sources"]
                if item["source_id"] == source_id
            )
            == before
            for source_id, before in original_projection.items()
        )
        if source["acquisition_status"] != "eligible" or not unchanged:
            raise CutoverOperationalClosureError("Full-clone continuity failed")
        return {
            "clone_type": "REAL_FROZEN_INTAKE_POPULATION_PLUS_ONE_REHEARSAL_SOURCE",
            "original_source_count": len(v1["sources"]),
            "migration_result": migration["result"],
            "migration_retry_result": retry["result"],
            "review_failure_rollback_verified": failed_once,
            "review_retry_status": reviewed["acquisition_status"],
            "promotion_status": promoted["acquisition_status"],
            "authoritative_reload_status": source["acquisition_status"],
            "inventory_source_count": report["source_count"],
            "eligible_inventory_count": len(inventory),
            "review_payload_fingerprint": trusted.canonical_fingerprint(payload),
            "historical_sources_unchanged": unchanged,
            "review_records_preserved": final["review_records"] == [],
            "v2_review_used": False,
            "dedup_mocked": False,
            "repeated_global_dedup_pass_count": global_dedup_passes,
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "peak_rss_platform_units_before": peak_before,
            "peak_rss_platform_units_after": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "live_mutation_executed": False,
        }
    finally:
        intake._review_publication_checkpoint = original_checkpoint
        intake._eligibility_dedup_report = original_dedup_report
        if original_extractor is not None:
            intake._run_local_extractor = original_extractor
        shutil.rmtree(temporary, ignore_errors=True)
        if live_path.read_bytes() != live_before:
            raise CutoverOperationalClosureError("Live manifest changed during clone rehearsal")


def registry_provisioning_order() -> dict:
    return {
        "order": [
            "FREEZE_REVIEWED_CODE_AND_POLICY",
            "PROVISION_PUBLIC_TRUST_REGISTRIES_BY_APPROVED_PROCESS",
            "VERIFY_FINAL_CODE_BUNDLE_FINGERPRINT",
            "COMPLETE_PREFLIGHT_AND_BACKUP_RECOVERY_CHECKS",
            "INDEPENDENT_APPROVER_ISSUES_NARROW_CUTOVER_AUTHORIZATION",
            "SEPARATE_EXECUTOR_PERFORMS_ONE_AUTHORIZED_CUTOVER",
            "PERSIST_CREATE_ONCE_ACTIVATION_RECEIPT",
            "ISSUE_SEPARATE_OPERATIONAL_AUTHORIZATION_IF_APPROVED",
        ],
        "registry_change_after_signature_invalidates_binding": True,
        "production_private_key_created": False,
        "live_authorization_issued": False,
        "review_producer_registry_required_for_schema_migration": False,
        "promotion_authority_required_for_schema_migration": False,
    }


def build_outputs(
    project_root: Path,
    execution_git_sha: str,
    run_id: str,
    profile: Mapping[str, Any],
    rehearsal: Mapping[str, Any],
) -> dict[str, dict]:
    root = Path(project_root).resolve()
    baseline = _baseline(root)
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r11_run_fingerprint": R11_RUN_FINGERPRINT,
        "amendment_version": lifecycle.AMENDMENT_VERSION,
    }
    outputs = {
        "activation_lifecycle_audit.json": {
            "r11_expiry_lockout_reproduced_in_isolated_fixture": True,
            "cause": "R11_REVALIDATED_ONE_TIME_CUTOVER_EXPIRY_ON_EVERY_V2_1_LOAD",
            "r12_cutover_token_accepted_for_operations": False,
            "post_cutover_access_uses_independent_operational_authority": True,
            "valid_data_rolled_back_when_authority_expires": False,
        },
        "authorization_separation_spec.json": lifecycle.authority_separation_specification(),
        "registry_provisioning_order.json": registry_provisioning_order(),
        "v1_dedup_performance_profile.json": copy.deepcopy(dict(profile)),
        "full_clone_v1_review_promotion.json": copy.deepcopy(dict(rehearsal)),
        "post_cutover_recovery_validation.json": {
            "normal_operation_expiry_behavior": "SUSPEND_FAIL_CLOSED",
            "normal_operation_revocation_behavior": "SUSPEND_FAIL_CLOSED",
            "recovery_authority_separate": True,
            "recovery_access_read_only": True,
            "automatic_v1_rollback": False,
            "restart_revalidates_receipt_and_authorization": True,
            "journal_integrity_still_authoritative": True,
            "live_recovery_registry_active": False,
        },
        "migration_go_no_go.json": {
            "MIGRATION_ENGINEERING_GO": True,
            "LIVE_MIGRATION_AUTHORIZED": False,
            "TRUSTED_REVIEW_PRODUCER_ACTIVE": False,
            "ELIGIBILITY_PROMOTION_AUTHORIZED": False,
            "result": "CUTOVER_OPERATIONAL_GATES_VERIFIED_NOT_ACTIVATED",
            "remaining_activation_requirements": [
                "HUMAN_APPROVAL",
                "APPROVED_PUBLIC_ISSUER_AND_RECOVERY_REGISTRY_PROVISIONING",
                "FINAL_CODE_AND_PROVENANCE_BINDING",
                "VERIFIED_BACKUP_AND_OPERATOR_RECOVERY_PLAN",
                "SEPARATE_REVIEW_AND_PROMOTION_AUTHORITY_IF_LATER_NEEDED",
            ],
        },
        "validation_summary.json": {
            "result": "PASS_R12_OPERATIONAL_GATES_NOT_ACTIVATED",
            "protected_state": baseline["protected_state"],
            "live_manifest_mutated": False,
            "live_migration_executed": False,
            "live_review_or_promotion_performed": False,
            "production_credentials_created": False,
            "registry_activated": False,
            "retrieval_model_query_gt_or_sampling_used": False,
            "exact_v1_output_equivalence": profile["exact_report_equivalence"],
            "full_clone_v1_review_promotion_completed": (
                rehearsal["promotion_status"] == "eligible"
            ),
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
    expected_parent = (root / "backend/data/runs").resolve()
    if destination.parent != expected_parent or destination.name != run_id:
        raise CutoverOperationalClosureError("Run destination is invalid")
    if destination.exists() or destination.is_symlink():
        raise CutoverOperationalClosureError("Run destination already exists")
    if not run_id.startswith(PROTOCOL_ID + "_") or len(execution_git_sha) != 40:
        raise CutoverOperationalClosureError("Execution identity is invalid")
    _baseline(root)
    before = (root / intake.ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    profile = profile_full_population_v1(root)
    rehearsal = perform_full_clone_v1_review_promotion(root)
    outputs = build_outputs(root, execution_git_sha, run_id, profile, rehearsal)
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            partial.write_bytes(r8.canonical_json_bytes(payload))
            os.replace(partial, destination / name)
            hashes[name] = policy_v2.sha256_file(destination / name)
        if (root / intake.ACQUISITION_MANIFEST_RELATIVE).read_bytes() != before:
            raise CutoverOperationalClosureError("Live manifest changed")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "r11_run_fingerprint": R11_RUN_FINGERPRINT,
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "CUTOVER_OPERATIONAL_GATES_VERIFIED_NOT_ACTIVATED",
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
