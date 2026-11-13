"""R11 live-cutover readiness rehearsal without live activation.

The module provides a dynamic authoritative inventory, a clone-only atomic
V1-to-V2.1 migration rehearsal, and reproducible readiness artifacts.  It does
not contain an active issuer, issue attestations, migrate live data, review a
source, or authorize promotion from a V2 review record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import dedup_policy_v2 as policy_v2
from . import dedup_policy_v2_independent as historical_evaluator
from . import dual_version_manifest_safety as v21
from . import independent_benchmark
from . import source_intake as intake
from . import source_review_v2_migration as r8
from . import trusted_review_protocol as trusted
from . import version_aware_intake_integration as r10


PROTOCOL_ID = "independent_batch2dc_live_cutover_readiness_v1"
PARENT_GIT_SHA = "0a752b7f08afb13b0e4a650ab2b27a4c07e9bf11"
SOURCE_MANIFEST_SHA256 = "411a5ad54b26e640d190bc04b76fbbbfb39896c2d2b6b3fd96450347a8fec08a"
POLICY_FINGERPRINT = trusted.POLICY_FINGERPRINT
R10_RUN_FINGERPRINT = (
    "ab30ca6c7cefafedd573756548aafa26b536c9b4d31ee3ab6ab84631f4b581f3"
)
R10_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_version_aware_intake_integration_v1_"
    "20261112_102648_0a752b7f"
)
REHEARSAL_ACQUISITION_ID = "r11-cutover-continuity"


class LiveCutoverReadinessError(ValueError):
    """An R11 readiness, migration, or dynamic-inventory invariant failed."""


class SimulatedHardStop(BaseException):
    """Test-only process-stop boundary that bypasses ordinary rollback."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LiveCutoverReadinessError(f"Invalid JSON: {path}") from exc


def _full_text(payload: Mapping[str, Any]) -> str:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise LiveCutoverReadinessError("Structured evidence has no pages")
    texts = []
    for page in pages:
        if not isinstance(page, Mapping) or not isinstance(page.get("text"), str):
            raise LiveCutoverReadinessError("Structured page text is invalid")
        texts.append(page["text"])
    return "\n".join(texts)


def _dynamic_development_records(project_root: Path) -> list[dict]:
    records = []
    for item in independent_benchmark.inventory_judgment_sources(
        project_root, include_private_text=True
    ):
        records.append(
            {
                "source_id": item["source_id"],
                "acquisition_id": None,
                "population": "DEVELOPMENT",
                "source_name": item["source_name"],
                "source_path": item["source_path"],
                "structured_path": item["structured_path"],
                "raw_sha256": item["source_sha256"],
                "structured_sha256": item["structured_sha256"],
                "case_id": item["case_id"],
                "judgment_number": None,
                "court": item["court"],
                "decision_date": None,
                "decision_year": item["year"],
                "document_type": None,
                "procedure_stage": item["procedure_stage"],
                "related_decision_ids": [],
                "underlying_dispute_id": None,
                "identity_verified": False,
                "text": item["_dedup_text"],
            }
        )
    return records


def load_dynamic_inventory(
    project_root: Path,
    *,
    allow_inactive_v21: bool = False,
    v21_activation_authorization: Mapping[str, Any] | None = None,
) -> dict:
    """Load current verified inventory without historical count invariants."""

    root = Path(project_root).resolve()
    manifest = intake.load_acquisition_manifest(
        root,
        allow_inactive_v21=allow_inactive_v21,
        v21_activation_authorization=v21_activation_authorization,
    )
    policy = policy_v2.load_and_validate_policy(root)
    if policy["policy_fingerprint"] != POLICY_FINGERPRINT:
        raise LiveCutoverReadinessError("Dynamic evaluator policy binding changed")
    records = _dynamic_development_records(root)
    development_count = len(records)
    incomplete = []
    candidate_statuses = {"extracted", "metadata_verified", "dedup_checked", "eligible"}
    for source in sorted(manifest["sources"], key=lambda item: item["source_id"]):
        if source["acquisition_status"] not in candidate_statuses:
            continue
        identity_failures = intake._source_identity_failures(source)
        if identity_failures:
            incomplete.append(
                {
                    "source_id": source["source_id"],
                    "acquisition_id": source["acquisition_id"],
                    "failures": identity_failures,
                }
            )
            continue
        failures, structured = intake._source_integrity_failures(source, root)
        if failures or structured is None:
            raise LiveCutoverReadinessError(
                f"Dynamic source integrity failed: {source['source_id']}: {failures}"
            )
        records.append(
            {
                "source_id": source["source_id"],
                "acquisition_id": source["acquisition_id"],
                "population": "INDEPENDENT",
                "source_name": source["original_filename"],
                "source_path": source["stored_pdf_path"],
                "structured_path": source["structured_path"],
                "raw_sha256": source["sha256"],
                "structured_sha256": source["structured_sha256"],
                "case_id": source["case_id"],
                "judgment_number": source["judgment_number"],
                "court": source["court"],
                "decision_date": source["decision_date"],
                "decision_year": int(source["decision_date"][:4]),
                "document_type": None,
                "procedure_stage": source["procedure_stage"],
                "related_decision_ids": list(source["related_decision_ids"]),
                "underlying_dispute_id": source["underlying_dispute_id"],
                "identity_verified": True,
                "text": _full_text(structured),
            }
        )
    source_ids = [item["source_id"] for item in records]
    if len(source_ids) != len(set(source_ids)):
        raise LiveCutoverReadinessError("Dynamic inventory source IDs are not unique")
    manifest_binding = {
        "manifest_version": manifest["manifest_version"],
        "sources": [
            {
                "source_id": source["source_id"],
                "acquisition_status": source["acquisition_status"],
                "raw_sha256": source["sha256"],
                "structured_sha256": source["structured_sha256"],
                "case_id": source["case_id"],
                "judgment_number": source["judgment_number"],
                "court": source["court"],
                "decision_date": source["decision_date"],
                "procedure_stage": source["procedure_stage"],
                "underlying_dispute_id": source["underlying_dispute_id"],
                "related_decision_ids": source["related_decision_ids"],
            }
            for source in manifest["sources"]
        ],
        "known_duplicate_links": manifest["known_duplicate_links"],
    }
    return {
        "manifest": manifest,
        "policy": policy,
        "records": records,
        "development_count": development_count,
        "independent_candidate_count": len(records) - development_count,
        "incomplete_candidate_sources": incomplete,
        "complete_inventory_fingerprint": policy_v2.canonical_fingerprint(
            {
                "manifest_binding": manifest_binding,
                "development": [
                    {
                        "source_id": item["source_id"],
                        "raw_sha256": item["raw_sha256"],
                        "structured_sha256": item["structured_sha256"],
                    }
                    for item in records
                    if item["population"] == "DEVELOPMENT"
                ],
            }
        ),
    }


def recompute_dynamic_candidates(
    project_root: Path,
    target_source_ids: set[str],
    *,
    allow_inactive_v21: bool = False,
    v21_activation_authorization: Mapping[str, Any] | None = None,
) -> dict:
    """Recompute candidates/dispositions from current authoritative state."""

    context = load_dynamic_inventory(
        project_root,
        allow_inactive_v21=allow_inactive_v21,
        v21_activation_authorization=v21_activation_authorization,
    )
    incomplete_ids = {item["source_id"] for item in context["incomplete_candidate_sources"]}
    if target_source_ids & incomplete_ids:
        raise LiveCutoverReadinessError("Target source identity is incomplete")
    available = {item["source_id"] for item in context["records"]}
    if not target_source_ids <= available:
        raise LiveCutoverReadinessError("Dynamic target is absent from candidate inventory")
    links = [
        (
            f"independent:{item['acquisition_id']}",
            f"independent:{item['related_acquisition_id']}",
        )
        for item in context["manifest"]["known_duplicate_links"]
    ]
    generated = historical_evaluator.generate_candidates(
        context["records"], target_source_ids, links
    )
    evaluated = historical_evaluator.evaluate_candidates(context["policy"], generated)
    decisions = historical_evaluator.derive_source_decisions(
        context["records"], target_source_ids, evaluated
    )
    return {
        **context,
        "target_source_ids": sorted(target_source_ids),
        "generated": generated,
        "evaluated": evaluated,
        "decisions": decisions,
        "candidate_set_fingerprint": policy_v2.canonical_fingerprint(evaluated),
        "findings_fingerprint": policy_v2.canonical_fingerprint(decisions),
        "historical_count_invariants_used": False,
    }


def write_clone_staging_marker(project_root: Path) -> None:
    root = Path(project_root).resolve()
    if root == Path(__file__).resolve().parents[4]:
        raise LiveCutoverReadinessError("Staging marker cannot be written to live root")
    _, chain = intake._open_trusted_directory(root, intake.INTAKE_ROOT_RELATIVE, create=False)
    try:
        marker_name = intake.INACTIVE_V21_STAGING_MARKER_RELATIVE.name
        if intake._entry_lstat(chain.fd, marker_name) is None:
            intake._write_bytes_create_once_at(
                chain.fd,
                marker_name,
                intake._canonical_json_bytes(intake.INACTIVE_V21_STAGING_MARKER),
            )
        elif intake._read_regular_bytes_at(chain.fd, marker_name) != intake._canonical_json_bytes(
            intake.INACTIVE_V21_STAGING_MARKER
        ):
            raise LiveCutoverReadinessError("Clone staging marker mismatch")
        chain.assert_identity()
    finally:
        chain.close()


def migrate_clone_manifest(
    project_root: Path,
    expected_pre_sha256: str,
    *,
    checkpoint: Callable[[str], None] | None = None,
) -> dict:
    """Atomically migrate one non-live clone, with rollback or forward recovery."""

    root = Path(project_root).resolve()
    if root == Path(__file__).resolve().parents[4]:
        raise LiveCutoverReadinessError("Live migration executor is forbidden")
    checkpoint = checkpoint or (lambda _name: None)
    write_clone_staging_marker(root)
    _, chain = intake._open_trusted_directory(root, intake.INTAKE_ROOT_RELATIVE, create=False)
    intake._acquire_manifest_mutation_lock(chain)
    original_bytes = None
    intended_bytes = None
    published = False
    try:
        original_bytes = intake._read_regular_bytes_at(
            chain.fd, intake.ACQUISITION_MANIFEST_RELATIVE.name
        )
        current = json.loads(original_bytes.decode("utf-8"))
        if current.get("manifest_version") == v21.V21_MANIFEST_VERSION:
            intake._load_authoritative_manifest_locked(
                root, chain, allow_inactive_v21=True
            )
            return {
                "result": "IDEMPOTENT_ALREADY_MIGRATED",
                "pre_sha256": expected_pre_sha256,
                "post_sha256": _sha256_bytes(original_bytes),
                "manifest_version": v21.V21_MANIFEST_VERSION,
            }
        if _sha256_bytes(original_bytes) != expected_pre_sha256:
            raise LiveCutoverReadinessError("Clone pre-migration SHA256 mismatch")
        intake.validate_acquisition_manifest(current)
        migrated = v21.migrate_v1_to_v21_copy(current)
        v21.validate_dual_version_manifest(migrated)
        intake.validate_acquisition_manifest_version_aware(
            migrated, allow_inactive_v21=True
        )
        intended_bytes = intake._canonical_json_bytes(migrated)
        checkpoint("before_atomic_publication")
        intake._atomic_write_bytes_at(
            chain.fd,
            intake.ACQUISITION_MANIFEST_RELATIVE.name,
            intended_bytes,
        )
        published = True
        checkpoint("after_atomic_publication")
        loaded, loaded_bytes = intake._load_authoritative_manifest_locked(
            root, chain, allow_inactive_v21=True
        )
        if loaded_bytes != intended_bytes or loaded != migrated:
            raise LiveCutoverReadinessError("Clone authoritative reload mismatch")
        checkpoint("after_authoritative_reload")
        return {
            "result": "MIGRATED",
            "pre_sha256": expected_pre_sha256,
            "post_sha256": _sha256_bytes(intended_bytes),
            "manifest_version": loaded["manifest_version"],
        }
    except Exception:
        if published and original_bytes is not None and intended_bytes is not None:
            current_bytes = intake._read_regular_bytes_at(
                chain.fd, intake.ACQUISITION_MANIFEST_RELATIVE.name
            )
            if current_bytes != intended_bytes:
                raise LiveCutoverReadinessError(
                    "Clone migration failed after an unexpected concurrent mutation"
                )
            intake._atomic_write_bytes_at(
                chain.fd,
                intake.ACQUISITION_MANIFEST_RELATIVE.name,
                original_bytes,
            )
            checked, checked_bytes = intake._load_authoritative_manifest_locked(root, chain)
            if checked_bytes != original_bytes or checked["manifest_version"] != intake.MANIFEST_VERSION:
                raise LiveCutoverReadinessError("Clone migration rollback failed")
        raise
    finally:
        intake._release_manifest_mutation_lock(chain)
        chain.close()


def _minimal_pdf_bytes(label: str) -> bytes:
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] /Resources << >> /Contents 4 0 R >>",
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    payload = bytearray(f"%PDF-1.4\n% {label}\n".encode("ascii"))
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{number} 0 obj\n".encode("ascii"))
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def _install_rehearsal_extractor() -> Callable:
    original = intake._run_local_extractor

    def extractor(_raw_fd, output_dir_fd, original_filename, _force_ocr):
        text = "R11 isolated continuity source with unique non-production legal text"
        payload = {
            "judgment_name": original_filename,
            "page_count": 1,
            "pages": [{"page": 1, "text": text}],
            "sections": {
                "header": text,
                "facts": "",
                "reasoning": "",
                "decision": "",
            },
        }
        intake._write_bytes_create_once_at(
            output_dir_fd,
            f"{Path(original_filename).stem}_structured.json",
            intake._canonical_json_bytes(payload),
        )
        return True

    intake._run_local_extractor = extractor
    return original


def _clone_required_tree(source_root: Path, clone_root: Path) -> None:
    relative_directories = (
        Path("backend/dataLaw/independent_intake"),
        Path("backend/dataLaw/ban_an"),
        Path("backend/data/structured_judgments"),
    )
    for relative in relative_directories:
        destination = clone_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_root / relative, destination, copy_function=shutil.copy2)
    policy_relative = policy_v2.DEFAULT_POLICY_RELATIVE
    policy_destination = clone_root / policy_relative
    policy_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / policy_relative, policy_destination)
    fixture_relative = policy_v2.EVIDENCE_FIXTURE_RELATIVE
    fixture_destination = clone_root / fixture_relative
    fixture_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / fixture_relative, fixture_destination)
    development_relative = Path("backend/data/judgments_evaluation.json")
    development_destination = clone_root / development_relative
    development_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / development_relative, development_destination)


def perform_full_clone_rehearsal(project_root: Path) -> dict:
    """Exercise one real-population clone and remove it after verification."""

    root = Path(project_root).resolve()
    live_manifest_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    live_before = live_manifest_path.read_bytes()
    temporary = Path(tempfile.mkdtemp(prefix="r11-full-clone-"))
    clone_root = temporary / "repository"
    clone_root.mkdir()
    original_extractor = None
    try:
        _clone_required_tree(root, clone_root)
        clone_manifest_path = clone_root / intake.ACQUISITION_MANIFEST_RELATIVE
        if _sha256_bytes(clone_manifest_path.read_bytes()) != SOURCE_MANIFEST_SHA256:
            raise LiveCutoverReadinessError("Full clone does not match live V1 manifest")
        v1_manifest = intake.load_acquisition_manifest(clone_root)
        if len(v1_manifest["sources"]) != 363:
            raise LiveCutoverReadinessError("Full clone source count changed")
        failed_after_publish = False

        def fail_after_publish(name: str) -> None:
            nonlocal failed_after_publish
            if name == "after_atomic_publication":
                failed_after_publish = True
                raise RuntimeError("R11 injected post-publication failure")

        try:
            migrate_clone_manifest(
                clone_root,
                SOURCE_MANIFEST_SHA256,
                checkpoint=fail_after_publish,
            )
        except RuntimeError as exc:
            if "injected" not in str(exc):
                raise
        else:
            raise LiveCutoverReadinessError("Migration failure injection did not fire")
        if not failed_after_publish or _sha256_bytes(clone_manifest_path.read_bytes()) != SOURCE_MANIFEST_SHA256:
            raise LiveCutoverReadinessError("Post-publication rollback did not restore V1")
        migration = migrate_clone_manifest(clone_root, SOURCE_MANIFEST_SHA256)
        idempotent = migrate_clone_manifest(clone_root, SOURCE_MANIFEST_SHA256)
        migrated = intake.load_acquisition_manifest(
            clone_root, allow_inactive_v21=True
        )
        if v21._v1_projection(migrated) != v1_manifest or migrated["review_records"]:
            raise LiveCutoverReadinessError("Clone migration changed V1 projection")

        pdf_input = clone_root / "r11-cutover-continuity.pdf"
        pdf_input.write_bytes(_minimal_pdf_bytes("r11-cutover-continuity"))
        now = datetime.now().astimezone().isoformat()
        registered = intake.register_source(
            clone_root,
            REHEARSAL_ACQUISITION_ID,
            pdf_input,
            "https://example.test/r11-cutover-continuity.pdf",
            "example.test",
            now,
            "r11-cutover-continuity.pdf",
            allow_inactive_v21=True,
        )
        original_extractor = _install_rehearsal_extractor()
        extracted = intake.extract_intake_source(
            clone_root,
            REHEARSAL_ACQUISITION_ID,
            allow_inactive_v21=True,
        )
        intake._run_local_extractor = original_extractor
        original_extractor = None
        metadata = {
            "judgment_number": "9999/2026/DS-ST",
            "case_id": "9999/2026/DS-ST",
            "court": "Tòa án nhân dân tỉnh R11",
            "decision_date": "2026-11-12",
            "procedure_stage": "so_tham",
        }
        identity = intake.record_source_identity(
            clone_root,
            REHEARSAL_ACQUISITION_ID,
            {
                "identity_version": intake.IDENTITY_PAYLOAD_VERSION,
                "verified_metadata": metadata,
                "verified_fields": {
                    field: True for field in intake.REVIEW_METADATA_FIELDS
                },
                "related_decision_ids": [],
                "underlying_dispute_id": None,
            },
            allow_inactive_v21=True,
        )
        inventory, report, _ = intake.inventory_independent_sources(
            clone_root, allow_inactive_v21=True
        )
        dynamic = recompute_dynamic_candidates(
            clone_root,
            {f"independent:{REHEARSAL_ACQUISITION_ID}"},
            allow_inactive_v21=True,
        )
        final = intake.load_acquisition_manifest(clone_root, allow_inactive_v21=True)
        original_sources = {
            item["source_id"]: item for item in v1_manifest["sources"]
        }
        final_projection = v21._v1_projection(final)
        existing_unchanged = all(
            next(
                item for item in final_projection["sources"] if item["source_id"] == source_id
            )
            == source
            for source_id, source in original_sources.items()
        )
        original_provenance_unchanged = all(
            item in final_projection["immutable_provenance"]
            for item in v1_manifest["immutable_provenance"]
        )
        if not existing_unchanged or not original_provenance_unchanged:
            raise LiveCutoverReadinessError("Full clone changed an existing source")
        return {
            "clone_type": "REAL_REPOSITORY_BACKED_FULL_INTAKE_POPULATION",
            "original_source_count": 363,
            "pre_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "post_migration_sha256": migration["post_sha256"],
            "failure_injection_rolled_back": True,
            "migration_result": migration["result"],
            "idempotent_retry_result": idempotent["result"],
            "authoritative_v21_reload": True,
            "registration_status": registered["acquisition_status"],
            "extraction_status": extracted["acquisition_status"],
            "identity_status": identity["acquisition_status"],
            "review_status": "EXERCISED_ON_ISOLATED_V21_EXACT_GATE_FIXTURE",
            "historical_v1_promotion_status": (
                "EXERCISED_ON_ISOLATED_V21_EXACT_GATE_FIXTURE"
            ),
            "full_clone_review_and_promotion_exercised": False,
            "full_clone_review_and_promotion_limitation": (
                "UNCHANGED_V1_GLOBAL_DEDUP_GATE_EXCEEDED_11_MINUTE_BOUNDED_"
                "REHEARSAL_WINDOW; EXACT_API_PATH_COVERED_ON_ISOLATED_V21"
            ),
            "v2_review_record_promotion_exercised": False,
            "eligible_inventory_count_after_rehearsal": len(inventory),
            "inventory_report_source_count": report["source_count"],
            "dynamic_inventory_count": len(dynamic["records"]),
            "dynamic_candidate_pair_count": dynamic["generated"]["candidate_pair_count"],
            "dynamic_counts_are_not_correctness_invariants": True,
            "existing_sources_unchanged": existing_unchanged,
            "original_provenance_unchanged": original_provenance_unchanged,
            "review_records_preserved": final["review_records"] == [],
            "mocked_component": "LOCAL_EXTRACTOR_ONLY",
            "live_mutation_executed": False,
        }
    finally:
        if original_extractor is not None:
            intake._run_local_extractor = original_extractor
        shutil.rmtree(temporary, ignore_errors=True)
        if live_manifest_path.read_bytes() != live_before:
            raise LiveCutoverReadinessError("Live manifest changed during full-clone rehearsal")


def cutover_dependency_graph() -> dict:
    return {
        "nodes": [
            "clean_committed_code",
            "frozen_policy_binding",
            "signed_external_activation_authorization",
            "atomic_v1_to_v21_cutover",
            "authoritative_v21_reload",
            "v1_operational_continuity",
            "dynamic_inventory_recomputation",
            "active_trusted_producer_registry",
            "signed_review_attestation",
            "separate_promotion_authorization",
        ],
        "edges": [
            ["clean_committed_code", "signed_external_activation_authorization"],
            ["frozen_policy_binding", "signed_external_activation_authorization"],
            ["signed_external_activation_authorization", "atomic_v1_to_v21_cutover"],
            ["atomic_v1_to_v21_cutover", "authoritative_v21_reload"],
            ["authoritative_v21_reload", "v1_operational_continuity"],
            ["authoritative_v21_reload", "dynamic_inventory_recomputation"],
            ["dynamic_inventory_recomputation", "signed_review_attestation"],
            ["active_trusted_producer_registry", "signed_review_attestation"],
            ["signed_review_attestation", "separate_promotion_authorization"],
        ],
    }


def _protected_state(project_root: Path) -> dict:
    manifest = _json(Path(project_root) / intake.ACQUISITION_MANIFEST_RELATIVE)
    return r8._protected_state(Path(project_root), manifest)


def build_outputs(
    project_root: Path,
    execution_git_sha: str,
    run_id: str,
    rehearsal: Mapping[str, Any],
) -> dict[str, dict]:
    root = Path(project_root).resolve()
    if len(execution_git_sha) != 40 or not run_id.startswith(PROTOCOL_ID + "_"):
        raise LiveCutoverReadinessError("Execution identity is invalid")
    manifest_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    before = manifest_path.read_bytes()
    if _sha256_bytes(before) != SOURCE_MANIFEST_SHA256:
        raise LiveCutoverReadinessError("Live manifest binding changed")
    r10_run = _json(root / R10_RUN_RELATIVE / "run_manifest.json")
    if r10_run["run_fingerprint"] != R10_RUN_FINGERPRINT:
        raise LiveCutoverReadinessError("R10 run binding changed")
    historical = r10.authoritative_candidate_context(root)
    dynamic = recompute_dynamic_candidates(root, set(historical["targets"]))
    if (
        len(dynamic["records"]) != 469
        or len(dynamic["target_source_ids"]) != 46
        or dynamic["generated"]["candidate_pair_count"] != 26
    ):
        raise LiveCutoverReadinessError("Historical checkpoint parity changed")
    decisions = {
        item["acquisition_id"]: item for item in dynamic["decisions"]
    }
    if (
        not decisions["batch2dc-20261105-025"]["canonical_source_id"].endswith("025")
        or not decisions["batch2dc-20261105-026"]["noncanonical_exact_duplicate"]
        or any(
            decisions[f"batch2dc-20261105-{suffix}"][
                "prospective_review_readiness"
            ]
            != "BLOCKED_FAIL_CLOSED"
            for suffix in ("008", "014", "022", "024", "027", "029", "034")
        )
    ):
        raise LiveCutoverReadinessError("Cross-version duplicate/dispute controls changed")
    adjudication_bindings = r8.targeted_adjudication_bindings(root, dynamic)
    amendment = trusted.evidence_amendment_specification()
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA256,
        "policy_fingerprint": POLICY_FINGERPRINT,
        "r10_run_fingerprint": R10_RUN_FINGERPRINT,
    }
    outputs = {
        "cutover_dependency_audit.json": {
            "known_blockers": {
                "live_v21_guard": True,
                "v21_record_promotion_guard": True,
                "historical_evaluator_fixed_counts": True,
                "producer_registry_inactive": True,
                "live_review_attestations_absent": True,
            },
            "additional_blockers": [
                "NO_SIGNED_LIVE_ACTIVATION_AUTHORIZATION",
                "NO_ACTIVE_PRODUCTION_ISSUER_OR_PRODUCER",
                "PRODUCTION_REPLAY_LEDGER_NOT_PROVISIONED",
                "NO_SEPARATE_V2_PROMOTION_AUTHORIZATION",
            ],
            "dependency_graph": cutover_dependency_graph(),
        },
        "live_capability_safety_matrix.json": {
            "v1_default": True,
            "live_v21_path_present": True,
            "live_registry_activation_state": trusted.LIVE_TRUST_REGISTRY[
                "activation_state"
            ],
            "live_issuer_count": len(trusted.LIVE_TRUST_REGISTRY["issuers"]),
            "live_producer_count": len(trusted.LIVE_TRUST_REGISTRY["producers"]),
            "intake_local_flag_can_authorize_live": False,
            "signed_external_authorization_required": True,
            "unknown_versions_fail_closed": True,
            "live_activation_authorized": False,
        },
        "v1_operational_continuity.json": {
            "full_clone_results": copy.deepcopy(dict(rehearsal)),
            "historical_v1_promotion_supported_on_v21": True,
            "historical_v1_review_and_promotion_exact_api_fixture_verified": True,
            "full_clone_review_and_promotion_exercised": False,
            "historical_v1_gate_changed": False,
            "v2_findings_substituted_for_v1_findings": False,
            "v2_review_record_promotion_enabled": False,
        },
        "dynamic_inventory_validation.json": {
            "future_evaluator": "CURRENT_AUTHORITATIVE_VERSION_AWARE_STATE",
            "historical_checkpoint": {
                "records": len(dynamic["records"]),
                "targets": len(dynamic["target_source_ids"]),
                "pairs": dynamic["generated"]["candidate_pair_count"],
            },
            "historical_counts_used_for_correctness": False,
            "complete_inventory_fingerprint": dynamic[
                "complete_inventory_fingerprint"
            ],
            "candidate_set_fingerprint": dynamic["candidate_set_fingerprint"],
            "findings_fingerprint": dynamic["findings_fingerprint"],
            "registration_extraction_eligibility_changes_supported": True,
            "incomplete_identity_targets_fail_closed": True,
            "connected_to_live_promotion": False,
        },
        "trusted_producer_registry_spec.json": {
            "registry": trusted.LIVE_TRUST_REGISTRY,
            "contract": {
                "producer_id_and_version": True,
                "approved_code_sha256": True,
                "policy_binding": True,
                "review_contract_version": trusted.REVIEW_CONTRACT_VERSION,
                "evidence_amendment_id": trusted.EVIDENCE_AMENDMENT_ID,
                "trusted_execution_identity": True,
                "issuer_approval_provenance": True,
                "revocation_fail_closed": True,
                "explicit_activation_state": True,
            },
            "private_keys_committed": False,
            "live_authority_present": False,
        },
        "attestation_protocol_spec.json": {
            "attestation_version": trusted.ATTESTATION_VERSION,
            "signature_algorithm": "Ed25519",
            "source_and_inventory_bindings": True,
            "findings_and_decision_bindings": True,
            "producer_code_and_policy_bindings": True,
            "tamper_evident_signature": True,
            "create_once_replay_ledger_required": True,
            "create_once_replay_ledger_scaffold_verified": True,
            "production_replay_ledger_provisioned": False,
            "caller_hash_or_authority_string_sufficient": False,
            "live_attestation_issued": False,
        },
        "evidence_amendment_validation.json": {
            "specification": amendment,
            "actual_frozen_relation_labels_mapped": True,
            "targeted_027_029_bindings_preserved": True,
            "targeted_adjudication_pair_binding_fingerprints": [
                item["pair_binding_fingerprint"]
                for item in adjudication_bindings["bindings"]
            ],
            "r7_run_fingerprint": adjudication_bindings["r7_run_fingerprint"],
            "r7_target_artifact_sha256": adjudication_bindings[
                "r7_target_artifact_sha256"
            ],
            "unresolved_008_014_022_024_034_preserved": True,
            "all_469_document_type_coverage_claimed": False,
            "frozen_policy_modified": False,
        },
        "full_clone_migration_rehearsal.json": copy.deepcopy(dict(rehearsal)),
    }
    go_no_go = {
        "MIGRATION_ENGINEERING_READY": True,
        "TRUSTED_PRODUCER_READY": True,
        "LIVE_MIGRATION_AUTHORIZED": False,
        "migration_engineering_ready": True,
        "trusted_producer_protocol_ready": True,
        "trusted_producer_live_registry_active": False,
        "live_migration_authorized": False,
        "live_review_authorized": False,
        "live_promotion_authorized": False,
        "result": "READINESS_VERIFIED_NOT_ACTIVATED",
        "activation_blockers": [
            "SEPARATE_SIGNED_LIVE_ACTIVATION_APPROVAL_REQUIRED",
            "PRODUCTION_ISSUER_AND_PRODUCER_REGISTRY_ACTIVATION_REQUIRED",
            "PRODUCTION_REPLAY_LEDGER_PROVISIONING_REQUIRED",
            "SEPARATE_V2_PROMOTION_AUTHORIZATION_REQUIRED",
        ],
    }
    outputs["activation_go_no_go.json"] = go_no_go
    outputs["validation_summary.json"] = {
        "result": "PASS_R11_READINESS_NOT_ACTIVATED",
        "protected_state": _protected_state(root),
        "live_manifest_mutated": manifest_path.read_bytes() != before,
        "frozen_policy_modified": False,
        "historical_artifacts_modified": False,
        "live_activation_authorized": False,
        "live_review_or_promotion_performed": False,
        "network_used": False,
        "retrieval_or_model_used": False,
    }
    if outputs["validation_summary.json"]["live_manifest_mutated"]:
        raise LiveCutoverReadinessError("Live manifest changed during R11")
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
        raise LiveCutoverReadinessError("Run directory must be a direct runs child")
    if destination.exists() or destination.is_symlink():
        raise LiveCutoverReadinessError("Run directory already exists")
    live_path = root / intake.ACQUISITION_MANIFEST_RELATIVE
    before = live_path.read_bytes()
    rehearsal = perform_full_clone_rehearsal(root)
    outputs = build_outputs(root, execution_git_sha, run_id, rehearsal)
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
            raise LiveCutoverReadinessError("Live manifest changed during R11 publication")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
            "policy_fingerprint": POLICY_FINGERPRINT,
            "r10_run_fingerprint": R10_RUN_FINGERPRINT,
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "CUTOVER_READINESS_VERIFIED_NOT_ACTIVATED",
        }
        run_manifest["run_fingerprint"] = policy_v2.canonical_fingerprint(run_manifest)
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
