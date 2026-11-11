"""Atomic Batch2DC identity application for independent-benchmark intake.

This module is deliberately an orchestration layer over ``source_intake``.
It reconstructs the frozen B3.4 evidence, validates all 46 identities before
mutation, and publishes one manifest replacement.  It never reviews, promotes,
downloads, samples, or generates benchmark records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import source_intake
from .identity_normalization_policy import (
    EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT,
    EXPECTED_BLOCKED_RUN_FINGERPRINT,
    EXPECTED_BLOCKED_RUN_ID,
    EXPECTED_CANONICAL_VALUE,
    EXPECTED_FIELD,
    EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT,
    EXPECTED_RAW_VALUE,
    EXPECTED_SOURCE_INTAKE_SHA256,
    load_and_validate_policy,
    normalize_allowed_identity_value,
    policy_fingerprint,
    sha256_file,
)


PROTOCOL_ID = "independent_batch2dc_identity_application_v1"
BASELINE_HEAD = "812428f61668979dd17bed47922e12d559391d8d"
BASELINE_PARENT = "93b5f260b34974da02ed291423ceef0a3cd3a330"
EXPECTED_PRE_MANIFEST_SHA256 = (
    "6933ef27c489ba51da1d3f2d157b60ca93a6ae57a51eb8e67570fda0d68add5d"
)
FROZEN_POLICY_FINGERPRINT = (
    "1033d13d11c0c7b3353a371ca236579ce821d310ec86a0873a47d6661bf9656c"
)
BLOCKED_RUN_RELATIVE = Path("backend/data/runs") / EXPECTED_BLOCKED_RUN_ID
LEDGER_RELATIVE = BLOCKED_RUN_RELATIVE / "source_evidence_ledger.json"
POLICY_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/"
    "independent_identity_normalization_policy_v1.yaml"
)
RUNS_RELATIVE = Path("backend/data/runs")
BENCHMARK_RELATIVE = Path(
    "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
)
SAMPLING_RELATIVE = Path(
    "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
)
SOURCE039_ID = "batch2dc-20261105-039"
EXCLUDED_IDS = frozenset(
    f"batch2dc-20261105-{index:03d}" for index in (23, 40, 47, 50)
)
EXPECTED_SOURCE_IDS = tuple(
    f"batch2dc-20261105-{index:03d}"
    for index in range(1, 51)
    if f"batch2dc-20261105-{index:03d}" not in EXCLUDED_IDS
)
IDENTITY_FIELDS = tuple(source_intake.REVIEW_METADATA_FIELDS)
ALLOWED_SOURCE_CHANGES = frozenset(IDENTITY_FIELDS) | {
    "underlying_dispute_id",
    "related_decision_ids",
    "metadata_verification",
}


class Batch2DCIdentityApplicationError(source_intake.SourceIntakeError):
    """Raised when the frozen Batch2DC identity transaction fails closed."""


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    if pretty:
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
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_regular(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Batch2DCIdentityApplicationError(
            f"Required JSON artifact is not a regular file: {path}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Batch2DCIdentityApplicationError(
            f"Required JSON artifact is invalid: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise Batch2DCIdentityApplicationError(f"JSON root must be an object: {path}")
    return value


def _payload_from_record(record: Mapping[str, Any]) -> dict:
    return {
        "identity_version": source_intake.IDENTITY_PAYLOAD_VERSION,
        "verified_metadata": {field: record[field] for field in IDENTITY_FIELDS},
        "verified_fields": {field: True for field in IDENTITY_FIELDS},
        "related_decision_ids": list(record["related_decision_ids"]),
        "underlying_dispute_id": record["underlying_dispute_id"],
    }


def _plan_fingerprint(plan: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(plan))
    unsigned.pop("plan_fingerprint", None)
    return canonical_fingerprint(unsigned)


def _validate_exact_source_ids(values: Sequence[str]) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise Batch2DCIdentityApplicationError("Source IDs must be a sequence")
    if not all(isinstance(value, str) for value in values):
        raise Batch2DCIdentityApplicationError("Every source ID must be a string")
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count != 1)
    if duplicates:
        raise Batch2DCIdentityApplicationError(
            f"Duplicate acquisition IDs: {duplicates}"
        )
    missing = sorted(set(EXPECTED_SOURCE_IDS) - set(values))
    extra = sorted(set(values) - set(EXPECTED_SOURCE_IDS))
    if missing or extra:
        raise Batch2DCIdentityApplicationError(
            f"Batch2DC source set mismatch; missing={missing}, extra={extra}"
        )


def _validate_plan_shape(plan: Mapping[str, Any]) -> dict:
    root_fields = {
        "protocol_id",
        "baseline_head",
        "baseline_parent",
        "blocked_run_id",
        "blocked_protocol_fingerprint",
        "blocked_run_fingerprint",
        "policy_fingerprint",
        "provisional_45_payload_fingerprint",
        "source039_normalization",
        "source_ids",
        "payloads",
        "evidence_bindings",
        "pre_manifest_sha256",
        "post_manifest_sha256",
        "plan_fingerprint",
    }
    if not isinstance(plan, Mapping) or set(plan) != root_fields:
        raise Batch2DCIdentityApplicationError("Application plan fields are invalid")
    exact = {
        "protocol_id": PROTOCOL_ID,
        "baseline_head": BASELINE_HEAD,
        "baseline_parent": BASELINE_PARENT,
        "blocked_run_id": EXPECTED_BLOCKED_RUN_ID,
        "blocked_protocol_fingerprint": EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT,
        "blocked_run_fingerprint": EXPECTED_BLOCKED_RUN_FINGERPRINT,
        "policy_fingerprint": FROZEN_POLICY_FINGERPRINT,
        "provisional_45_payload_fingerprint": (
            EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT
        ),
        "pre_manifest_sha256": EXPECTED_PRE_MANIFEST_SHA256,
    }
    for field, expected in exact.items():
        if plan[field] != expected:
            raise Batch2DCIdentityApplicationError(
                f"Application plan {field} differs from frozen value"
            )
    source_ids = plan["source_ids"]
    _validate_exact_source_ids(source_ids)
    if list(source_ids) != sorted(source_ids):
        raise Batch2DCIdentityApplicationError("Application source IDs are not sorted")
    payloads = plan["payloads"]
    evidence = plan["evidence_bindings"]
    if not isinstance(payloads, dict) or set(payloads) != set(source_ids):
        raise Batch2DCIdentityApplicationError("Application payload set is invalid")
    if not isinstance(evidence, dict) or set(evidence) != set(source_ids):
        raise Batch2DCIdentityApplicationError("Evidence binding set is invalid")
    checked_payloads = {
        acquisition_id: source_intake.validate_source_identity_payload(
            payloads[acquisition_id]
        )
        for acquisition_id in source_ids
    }
    for acquisition_id, binding in evidence.items():
        if not isinstance(binding, dict) or set(binding) != {
            "source_sha256",
            "structured_sha256",
        }:
            raise Batch2DCIdentityApplicationError(
                f"Evidence binding is invalid: {acquisition_id}"
            )
        if not all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in binding.values()
        ):
            raise Batch2DCIdentityApplicationError(
                f"Evidence hash is invalid: {acquisition_id}"
            )
    normalization = plan["source039_normalization"]
    if normalization != {
        "acquisition_id": SOURCE039_ID,
        "field": EXPECTED_FIELD,
        "raw_value": EXPECTED_RAW_VALUE,
        "canonical_value": EXPECTED_CANONICAL_VALUE,
        "transformation": "replace exactly one ASCII-space boundary with one slash",
    }:
        raise Batch2DCIdentityApplicationError("Source039 normalization is invalid")
    if checked_payloads[SOURCE039_ID]["verified_metadata"][EXPECTED_FIELD] != (
        EXPECTED_CANONICAL_VALUE
    ):
        raise Batch2DCIdentityApplicationError("Source039 canonical value is invalid")
    for field in ("post_manifest_sha256", "plan_fingerprint"):
        value = plan[field]
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise Batch2DCIdentityApplicationError(f"{field} is invalid")
    if plan["plan_fingerprint"] != _plan_fingerprint(plan):
        raise Batch2DCIdentityApplicationError("Application plan fingerprint mismatch")
    checked = deepcopy(dict(plan))
    checked["payloads"] = checked_payloads
    return checked


def _assert_source_change_contract(before: dict, after: dict) -> None:
    for field in source_intake.SOURCE_REQUIRED_FIELDS - ALLOWED_SOURCE_CHANGES:
        if after[field] != before[field]:
            raise Batch2DCIdentityApplicationError(
                f"Identity plan attempted forbidden mutation: {field}"
            )
    for field in source_intake.REVIEW_COMPLETION_FIELDS:
        if after["metadata_verification"][field] != before[
            "metadata_verification"
        ][field]:
            raise Batch2DCIdentityApplicationError(
                f"Identity plan attempted review mutation: {field}"
            )


def _apply_plan_in_memory(
    project_root: Path,
    manifest: dict,
    plan: Mapping[str, Any],
) -> dict:
    updated = deepcopy(manifest)
    original_by_id = {
        source["acquisition_id"]: source for source in manifest["sources"]
    }
    updated_by_id = {
        source["acquisition_id"]: source for source in updated["sources"]
    }
    for acquisition_id in plan["source_ids"]:
        if acquisition_id not in original_by_id:
            raise Batch2DCIdentityApplicationError(
                f"Unknown acquisition_id: {acquisition_id}"
            )
        source = original_by_id[acquisition_id]
        if source["acquisition_status"] != "extracted":
            raise Batch2DCIdentityApplicationError(
                f"Source identity requires extracted status: {acquisition_id}"
            )
        binding = plan["evidence_bindings"][acquisition_id]
        if source["sha256"] != binding["source_sha256"] or source[
            "structured_sha256"
        ] != binding["structured_sha256"]:
            raise Batch2DCIdentityApplicationError(
                f"Frozen evidence binding mismatch: {acquisition_id}"
            )
        integrity_failures, structured = source_intake._source_integrity_failures(
            source, project_root
        )
        if integrity_failures or structured is None:
            raise Batch2DCIdentityApplicationError(
                "Source identity integrity gates failed for "
                f"{acquisition_id}: {', '.join(integrity_failures)}"
            )
        target = updated_by_id[acquisition_id]
        source_intake._apply_verified_identity(
            target, plan["payloads"][acquisition_id]
        )
        _assert_source_change_contract(source, target)
        _atomic_application_checkpoint("after_source_application", acquisition_id)
    for acquisition_id, before in original_by_id.items():
        if acquisition_id not in plan["source_ids"] and updated_by_id[
            acquisition_id
        ] != before:
            raise Batch2DCIdentityApplicationError(
                f"Unrelated source changed: {acquisition_id}"
            )
    if updated.get("immutable_provenance") != manifest.get("immutable_provenance"):
        raise Batch2DCIdentityApplicationError("Immutable provenance changed")
    source_intake.validate_acquisition_manifest(updated, prior_manifest=manifest)
    source_intake._validate_manifest_anchors(project_root, updated)
    source_intake._validate_manifest_artifacts(project_root, updated)
    return updated


def reconstruct_frozen_payloads(
    ledger: Mapping[str, Any], policy: Mapping[str, Any]
) -> dict:
    """Purely reconstruct 45 frozen payloads plus the one allowed normalization."""

    if policy_fingerprint(policy) != FROZEN_POLICY_FINGERPRINT:
        raise Batch2DCIdentityApplicationError("Frozen policy fingerprint mismatch")
    if set(ledger) != {
        "record_count",
        "records",
        "resolved_count",
        "run_id",
        "unresolved_count",
    }:
        raise Batch2DCIdentityApplicationError("Blocked source ledger fields differ")
    if (
        ledger["run_id"] != EXPECTED_BLOCKED_RUN_ID
        or ledger["record_count"] != 46
        or ledger["resolved_count"] != 45
        or ledger["unresolved_count"] != 1
        or not isinstance(ledger["records"], list)
    ):
        raise Batch2DCIdentityApplicationError("Blocked source ledger summary differs")
    records = ledger["records"]
    ids = [record.get("acquisition_id") for record in records]
    _validate_exact_source_ids(ids)
    by_id = {record["acquisition_id"]: record for record in records}
    resolved = {
        acquisition_id: _payload_from_record(record)
        for acquisition_id, record in by_id.items()
        if record.get("status") == "RESOLVED_NOT_APPLIED"
    }
    if len(resolved) != 45 or SOURCE039_ID in resolved:
        raise Batch2DCIdentityApplicationError("Resolved 45-source ledger set differs")
    if canonical_fingerprint(resolved) != EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT:
        raise Batch2DCIdentityApplicationError(
            "Provisional 45-payload fingerprint mismatch"
        )
    source039 = by_id[SOURCE039_ID]
    if (
        source039.get("status") != "UNRESOLVED_CONTRACT_INCOMPATIBLE"
        or source039.get("judgment_number") is not None
        or source039.get("observed_judgment_header") != f"Số: {EXPECTED_RAW_VALUE}"
    ):
        raise Batch2DCIdentityApplicationError("Frozen source039 evidence differs")
    canonical = normalize_allowed_identity_value(
        policy, SOURCE039_ID, EXPECTED_FIELD, EXPECTED_RAW_VALUE
    )
    normalized_record = deepcopy(source039)
    normalized_record[EXPECTED_FIELD] = canonical
    resolved[SOURCE039_ID] = _payload_from_record(normalized_record)
    sorted_ids = sorted(resolved)
    _validate_exact_source_ids(sorted_ids)
    payloads = {
        acquisition_id: source_intake.validate_source_identity_payload(
            resolved[acquisition_id]
        )
        for acquisition_id in sorted_ids
    }
    evidence = {
        acquisition_id: {
            "source_sha256": by_id[acquisition_id]["source_sha256"],
            "structured_sha256": by_id[acquisition_id]["structured_sha256"],
        }
        for acquisition_id in sorted_ids
    }
    return {
        "source_ids": sorted_ids,
        "payloads": payloads,
        "evidence_bindings": evidence,
        "source039_normalization": {
            "acquisition_id": SOURCE039_ID,
            "field": EXPECTED_FIELD,
            "raw_value": EXPECTED_RAW_VALUE,
            "canonical_value": canonical,
            "transformation": (
                "replace exactly one ASCII-space boundary with one slash"
            ),
        },
    }


def build_batch2dc_application_plan(project_root: Path) -> dict:
    """Reconstruct, validate, and freeze the exact 46-source application plan."""

    project_root = Path(project_root).resolve()
    policy = load_and_validate_policy(project_root, project_root / POLICY_RELATIVE)
    ledger = _read_json_regular(project_root / LEDGER_RELATIVE)
    reconstructed = reconstruct_frozen_payloads(ledger, policy)
    sorted_ids = reconstructed["source_ids"]
    payloads = reconstructed["payloads"]
    evidence = reconstructed["evidence_bindings"]
    manifest_path = project_root / source_intake.ACQUISITION_MANIFEST_RELATIVE
    if sha256_file(manifest_path) != EXPECTED_PRE_MANIFEST_SHA256:
        raise Batch2DCIdentityApplicationError("Pre-R2 manifest SHA256 mismatch")
    manifest = source_intake.load_acquisition_manifest(project_root)
    batch_sources = {
        source["acquisition_id"]: source
        for source in manifest["sources"]
        if source["acquisition_id"].startswith("batch2dc-20261105-")
        and source["acquisition_id"] not in EXCLUDED_IDS
    }
    if set(batch_sources) != set(sorted_ids):
        raise Batch2DCIdentityApplicationError("Live Batch2DC source set differs")
    if any(
        source["acquisition_status"] != "extracted"
        or any(
            source["metadata_verification"][field] for field in IDENTITY_FIELDS
        )
        for source in batch_sources.values()
    ):
        raise Batch2DCIdentityApplicationError(
            "Pre-R2 Batch2DC identity state differs"
        )
    plan = {
        "protocol_id": PROTOCOL_ID,
        "baseline_head": BASELINE_HEAD,
        "baseline_parent": BASELINE_PARENT,
        "blocked_run_id": EXPECTED_BLOCKED_RUN_ID,
        "blocked_protocol_fingerprint": EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT,
        "blocked_run_fingerprint": EXPECTED_BLOCKED_RUN_FINGERPRINT,
        "policy_fingerprint": FROZEN_POLICY_FINGERPRINT,
        "provisional_45_payload_fingerprint": (
            EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT
        ),
        "source039_normalization": reconstructed["source039_normalization"],
        "source_ids": sorted_ids,
        "payloads": payloads,
        "evidence_bindings": evidence,
        "pre_manifest_sha256": EXPECTED_PRE_MANIFEST_SHA256,
        "post_manifest_sha256": "0" * 64,
        "plan_fingerprint": "0" * 64,
    }
    preview = _apply_plan_in_memory(project_root, manifest, plan)
    plan["post_manifest_sha256"] = _sha256_bytes(
        source_intake._canonical_json_bytes(preview)
    )
    plan["plan_fingerprint"] = _plan_fingerprint(plan)
    return _validate_plan_shape(plan)


def _atomic_application_checkpoint(
    name: str, acquisition_id: str | None = None
) -> None:
    """Test seam for application, stale-manifest, and publication failures."""


def _identity_ready_exact(source: dict, payload: Mapping[str, Any]) -> bool:
    return (
        all(
            source[field] == payload["verified_metadata"][field]
            and source["metadata_verification"][field] is True
            for field in IDENTITY_FIELDS
        )
        and source["related_decision_ids"] == payload["related_decision_ids"]
        and source["underlying_dispute_id"] == payload["underlying_dispute_id"]
    )


def apply_identity_plan_atomically(project_root: Path, plan: Mapping[str, Any]) -> dict:
    """Apply all 46 identities with one guarded manifest replacement.

    Exact replay of the already-published post-manifest is a byte-stable no-op.
    Any other manifest hash is rejected before mutation.
    """

    project_root = Path(project_root).resolve()
    checked_plan = _validate_plan_shape(plan)
    _, intake_chain = source_intake._open_trusted_directory(
        project_root, source_intake.INTAKE_ROOT_RELATIVE, create=False
    )
    source_intake._acquire_manifest_mutation_lock(intake_chain)
    try:
        manifest, original_bytes = source_intake._load_authoritative_manifest_locked(
            project_root, intake_chain
        )
        original_sha = _sha256_bytes(original_bytes)
        if original_sha == checked_plan["post_manifest_sha256"]:
            by_id = {
                source["acquisition_id"]: source for source in manifest["sources"]
            }
            if not all(
                acquisition_id in by_id
                and _identity_ready_exact(
                    by_id[acquisition_id], checked_plan["payloads"][acquisition_id]
                )
                for acquisition_id in checked_plan["source_ids"]
            ):
                raise Batch2DCIdentityApplicationError(
                    "Post-manifest hash matched but exact identity state did not"
                )
            return {
                "applied_count": 0,
                "identity_ready_count": len(checked_plan["source_ids"]),
                "idempotent_noop": True,
                "pre_manifest_sha256": original_sha,
                "post_manifest_sha256": original_sha,
                "source_ids": list(checked_plan["source_ids"]),
            }
        if original_sha != checked_plan["pre_manifest_sha256"]:
            raise Batch2DCIdentityApplicationError("Manifest SHA mismatch")
        updated = _apply_plan_in_memory(project_root, manifest, checked_plan)
        intended_bytes = source_intake._canonical_json_bytes(updated)
        if _sha256_bytes(intended_bytes) != checked_plan["post_manifest_sha256"]:
            raise Batch2DCIdentityApplicationError(
                "Prepared manifest differs from frozen post-manifest SHA"
            )
        try:
            _atomic_application_checkpoint("before_manifest_compare")
            current_bytes = source_intake._read_regular_bytes_at(
                intake_chain.fd,
                source_intake.ACQUISITION_MANIFEST_RELATIVE.name,
            )
            if current_bytes != original_bytes:
                raise Batch2DCIdentityApplicationError(
                    "Acquisition manifest changed during atomic identity application"
                )
            intake_chain.assert_identity()
            _atomic_application_checkpoint("before_manifest_replacement")
            source_intake._atomic_write_bytes_at(
                intake_chain.fd,
                source_intake.ACQUISITION_MANIFEST_RELATIVE.name,
                intended_bytes,
            )
            _atomic_application_checkpoint("after_manifest_replacement")
            if source_intake._read_regular_bytes_at(
                intake_chain.fd,
                source_intake.ACQUISITION_MANIFEST_RELATIVE.name,
            ) != intended_bytes:
                raise Batch2DCIdentityApplicationError(
                    "Published identity manifest verification failed"
                )
            checked_manifest, checked_bytes = (
                source_intake._load_authoritative_manifest_locked(
                    project_root, intake_chain
                )
            )
            _atomic_application_checkpoint("after_authoritative_reload")
            if checked_bytes != intended_bytes:
                raise Batch2DCIdentityApplicationError(
                    "Published identities failed authoritative reload"
                )
            checked_by_id = {
                source["acquisition_id"]: source
                for source in checked_manifest["sources"]
            }
            if not all(
                _identity_ready_exact(
                    checked_by_id[acquisition_id],
                    checked_plan["payloads"][acquisition_id],
                )
                for acquisition_id in checked_plan["source_ids"]
            ):
                raise Batch2DCIdentityApplicationError(
                    "Published identities are not exact"
                )
        except Exception as publication_error:
            current_bytes = source_intake._read_regular_bytes_at(
                intake_chain.fd,
                source_intake.ACQUISITION_MANIFEST_RELATIVE.name,
            )
            if current_bytes == intended_bytes:
                try:
                    source_intake._atomic_write_bytes_at(
                        intake_chain.fd,
                        source_intake.ACQUISITION_MANIFEST_RELATIVE.name,
                        original_bytes,
                    )
                    if source_intake._read_regular_bytes_at(
                        intake_chain.fd,
                        source_intake.ACQUISITION_MANIFEST_RELATIVE.name,
                    ) != original_bytes:
                        raise Batch2DCIdentityApplicationError(
                            "Atomic identity rollback verification failed"
                        )
                    source_intake._load_authoritative_manifest_locked(
                        project_root, intake_chain
                    )
                except Exception as rollback_error:
                    raise Batch2DCIdentityApplicationError(
                        "Atomic identity rollback failed"
                    ) from rollback_error
                raise Batch2DCIdentityApplicationError(
                    "Atomic identity publication failed; previous manifest restored"
                ) from publication_error
            if current_bytes != original_bytes:
                raise Batch2DCIdentityApplicationError(
                    "Atomic identity publication failed with unexpected manifest "
                    "state; refusing unsafe overwrite"
                ) from publication_error
            raise
        return {
            "applied_count": len(checked_plan["source_ids"]),
            "identity_ready_count": len(checked_plan["source_ids"]),
            "idempotent_noop": False,
            "pre_manifest_sha256": original_sha,
            "post_manifest_sha256": _sha256_bytes(intended_bytes),
            "source_ids": list(checked_plan["source_ids"]),
        }
    finally:
        source_intake._release_manifest_mutation_lock(intake_chain)
        intake_chain.close()


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_clean_committed_execution(root: Path) -> dict:
    branch = _run_git(root, "branch", "--show-current")
    if branch != "chore/repo-hygiene-runtime-recovery":
        raise Batch2DCIdentityApplicationError(f"Unexpected branch: {branch}")
    head = _run_git(root, "rev-parse", "HEAD")
    origin = _run_git(root, "rev-parse", f"origin/{branch}")
    if head != origin:
        raise Batch2DCIdentityApplicationError("Local HEAD and origin differ")
    if _run_git(root, "status", "--porcelain"):
        raise Batch2DCIdentityApplicationError("Execution worktree is not clean")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASELINE_HEAD, head],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise Batch2DCIdentityApplicationError(
            "Frozen R1 baseline is not an execution ancestor"
        )
    return {"branch": branch, "execution_commit": head, "origin_commit": origin}


def _protected_state(root: Path) -> dict:
    manifest = source_intake.load_acquisition_manifest(root)
    statuses = Counter(source["acquisition_status"] for source in manifest["sources"])
    batch = [
        source
        for source in manifest["sources"]
        if source["acquisition_id"] in EXPECTED_SOURCE_IDS
    ]
    benchmark_path = root / BENCHMARK_RELATIVE
    sampling_path = root / SAMPLING_RELATIVE
    benchmark = _read_json_regular(benchmark_path)
    sampling = _read_json_regular(sampling_path)
    counts = benchmark.get("current_counts", {})
    return {
        "global_total": len(manifest["sources"]),
        "global_eligible": statuses["eligible"],
        "global_extracted": statuses["extracted"],
        "global_hash_verified": statuses["hash_verified"],
        "batch2dc_count": len(batch),
        "batch2dc_extracted": sum(
            source["acquisition_status"] == "extracted" for source in batch
        ),
        "batch2dc_eligible": sum(
            source["acquisition_status"] == "eligible" for source in batch
        ),
        "batch2dc_identity_ready": sum(
            all(source["metadata_verification"][field] for field in IDENTITY_FIELDS)
            for source in batch
        ),
        "benchmark_state": benchmark.get("state"),
        "sampling_state": sampling.get("state"),
        "selection_status": sampling.get("selection_status"),
        "sampling_seed": sampling.get("sampling_seed"),
        "selected_count": counts.get("selected_cases"),
        "query_count": counts.get("query_records"),
        "ground_truth_count": counts.get("ground_truth_records"),
        "benchmark_sha256": sha256_file(benchmark_path),
        "sampling_sha256": sha256_file(sampling_path),
    }


def _assert_protected_transition(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    immutable = {
        "global_total": 363,
        "global_eligible": 257,
        "global_extracted": 105,
        "global_hash_verified": 1,
        "batch2dc_count": 46,
        "batch2dc_extracted": 46,
        "batch2dc_eligible": 0,
        "benchmark_state": "DRAFT",
        "sampling_state": "DRAFT",
        "selection_status": "NOT_STARTED",
        "sampling_seed": None,
        "selected_count": 0,
        "query_count": 0,
        "ground_truth_count": 0,
    }
    for field, expected in immutable.items():
        if before[field] != expected or after[field] != expected:
            raise Batch2DCIdentityApplicationError(
                f"Protected state differs for {field}"
            )
    if before["batch2dc_identity_ready"] != 0 or after[
        "batch2dc_identity_ready"
    ] != 46:
        raise Batch2DCIdentityApplicationError("Identity-ready transition differs")
    for field in ("benchmark_sha256", "sampling_sha256"):
        if before[field] != after[field]:
            raise Batch2DCIdentityApplicationError(f"Protected {field} changed")


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value, pretty=True))


def _load_test_evidence(path: Path) -> dict:
    evidence = _read_json_regular(path)
    if set(evidence) != {"schema_version", "results"} or evidence[
        "schema_version"
    ] != "batch2dc_r2_test_evidence_v1":
        raise Batch2DCIdentityApplicationError("Test evidence fields are invalid")
    results = evidence["results"]
    if not isinstance(results, list) or not results:
        raise Batch2DCIdentityApplicationError("Test evidence results are empty")
    required = {
        "name",
        "command",
        "outcome",
        "passed",
        "failed",
        "duration_seconds",
        "notes",
    }
    for result in results:
        if not isinstance(result, dict) or set(result) != required:
            raise Batch2DCIdentityApplicationError(
                "Test evidence result fields are invalid"
            )
        if result["outcome"] not in {
            "PASS",
            "HISTORICAL_INCOMPATIBILITY",
            "INTERRUPTED_COST_BOUND",
        }:
            raise Batch2DCIdentityApplicationError("Test evidence outcome is invalid")
        if not isinstance(result["passed"], int) or result["passed"] < 0:
            raise Batch2DCIdentityApplicationError("Test pass count is invalid")
        if not isinstance(result["failed"], int) or result["failed"] < 0:
            raise Batch2DCIdentityApplicationError("Test failure count is invalid")
        if not isinstance(result["duration_seconds"], (int, float)) or result[
            "duration_seconds"
        ] < 0:
            raise Batch2DCIdentityApplicationError("Test duration is invalid")
        for field in ("name", "command", "notes"):
            if not isinstance(result[field], str):
                raise Batch2DCIdentityApplicationError(
                    f"Test evidence {field} is invalid"
                )
    if not any(result["outcome"] == "PASS" for result in results):
        raise Batch2DCIdentityApplicationError("Test evidence has no passing suite")
    return evidence


def execute_r2(
    project_root: Path,
    test_evidence_path: Path,
    run_id: str | None = None,
) -> dict:
    """Execute the committed R2 runner and publish an ignored audit run."""

    root = Path(project_root).resolve()
    git_state = _require_clean_committed_execution(root)
    test_evidence = _load_test_evidence(test_evidence_path)
    plan = build_batch2dc_application_plan(root)
    pre_state = _protected_state(root)
    timestamp = datetime.now(timezone.utc)
    if run_id is None:
        run_id = (
            "independent_batch2dc_identity_application_v1_"
            f"{timestamp:%Y%m%d_%H%M%S_%f}_{git_state['execution_commit'][:8]}"
        )
    if not run_id.startswith("independent_batch2dc_identity_application_v1_"):
        raise Batch2DCIdentityApplicationError("Run ID is invalid")
    run_root = root / RUNS_RELATIVE
    final_dir = run_root / run_id
    partial_dir = run_root / f".{run_id}.{uuid.uuid4().hex}.partial"
    if final_dir.exists() or final_dir.is_symlink():
        raise Batch2DCIdentityApplicationError("Final R2 run directory already exists")
    partial_dir.mkdir(parents=False, exist_ok=False)
    try:
        protocol = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "created_at_utc": timestamp.isoformat(),
            "git": git_state,
            "baseline_head": BASELINE_HEAD,
            "baseline_parent": BASELINE_PARENT,
            "source_intake_sha256": sha256_file(
                root / "backend/scripts/Evaluation/benchmarking/source_intake.py"
            ),
            "frozen_source_intake_sha256": EXPECTED_SOURCE_INTAKE_SHA256,
            "policy_path": POLICY_RELATIVE.as_posix(),
            "policy_fingerprint": FROZEN_POLICY_FINGERPRINT,
            "blocked_run_id": EXPECTED_BLOCKED_RUN_ID,
            "network_forbidden": True,
            "source_review_forbidden": True,
            "promotion_forbidden": True,
            "query_gt_retrieval_model_forbidden": True,
        }
        _write_json(partial_dir / "identity_protocol.json", protocol)
        _write_json(partial_dir / "identity_application_plan.json", plan)
        _write_json(
            partial_dir / "regression_test_evidence.json", test_evidence
        )
        _write_json(
            partial_dir / "source_evidence_ledger.json",
            _read_json_regular(root / LEDGER_RELATIVE),
        )
        result = apply_identity_plan_atomically(root, plan)
        post_state = _protected_state(root)
        _assert_protected_transition(pre_state, post_state)
        changes = {
            acquisition_id: {
                "verified_metadata": plan["payloads"][acquisition_id][
                    "verified_metadata"
                ],
                "verified_fields": plan["payloads"][acquisition_id][
                    "verified_fields"
                ],
                "related_decision_ids": plan["payloads"][acquisition_id][
                    "related_decision_ids"
                ],
                "underlying_dispute_id": plan["payloads"][acquisition_id][
                    "underlying_dispute_id"
                ],
            }
            for acquisition_id in plan["source_ids"]
        }
        results = {
            "run_id": run_id,
            "status": "IDENTITY_APPLICATION_COMPLETE",
            "plan_fingerprint": plan["plan_fingerprint"],
            "resolved_count": 46,
            "unresolved_count": 0,
            "validation_failures": 0,
            "atomic_manifest_replacements": 1,
            "application": result,
            "field_level_changes": changes,
            "pre_state": pre_state,
            "post_state": post_state,
            "rollback_contract": (
                "tested fail-closed by tracked regression tests; no production "
                "rollback was required"
            ),
        }
        _write_json(partial_dir / "identity_application_results.json", results)
        artifact_sha256 = {
            path.name: sha256_file(path)
            for path in sorted(partial_dir.iterdir())
            if path.is_file()
        }
        run_manifest = {
            "run_id": run_id,
            "run_status": "COMPLETE",
            "execution_commit": git_state["execution_commit"],
            "plan_fingerprint": plan["plan_fingerprint"],
            "pre_manifest_sha256": result["pre_manifest_sha256"],
            "post_manifest_sha256": result["post_manifest_sha256"],
            "artifact_sha256": artifact_sha256,
        }
        run_manifest["canonical_run_fingerprint"] = canonical_fingerprint(
            run_manifest
        )
        _write_json(partial_dir / "run_manifest.json", run_manifest)
        os.replace(partial_dir, final_dir)
        return {"run_dir": final_dir.as_posix(), **run_manifest}
    except Exception:
        if partial_dir.exists() and partial_dir.is_dir() and not partial_dir.is_symlink():
            for path in partial_dir.iterdir():
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            partial_dir.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--run-id")
    arguments = parser.parse_args()
    result = execute_r2(
        arguments.project_root,
        arguments.test_evidence,
        arguments.run_id,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
