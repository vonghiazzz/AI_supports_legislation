import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as policy_v2
from backend.scripts.Evaluation.benchmarking import dual_version_manifest_safety as r9
from backend.scripts.Evaluation.benchmarking import source_intake
from backend.scripts.Evaluation.benchmarking import version_aware_intake_integration as r10
from backend.scripts.Evaluation.benchmarking.source_intake import SourceIntakeError
from backend.scripts.Tests.test_independent_source_intake import (
    FIXED_TIME,
    _extracted_fixture,
    _minimal_pdf_bytes,
    _write_valid_extractor_output_fd,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / source_intake.ACQUISITION_MANIFEST_RELATIVE
RUN_ID = r10.PROTOCOL_ID + "_test"


class SimulatedCrash(BaseException):
    pass


def _write_marker(root):
    marker = root / source_intake.INACTIVE_V21_STAGING_MARKER_RELATIVE
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_bytes(r10.r8.canonical_json_bytes(source_intake.INACTIVE_V21_STAGING_MARKER))
    return marker


def _install_v21_fixture(root, *, with_record=True):
    source, manifest = _extracted_fixture(
        root,
        acquisition_id="recorded-source",
        filename="recorded.pdf",
        text="Nội dung nguồn được review độc lập",
    )
    migrated = r9.migrate_v1_to_v21_copy(manifest)
    if with_record:
        migrated["review_records"] = [
            r9.build_v21_evidence_record(
                migrated,
                source["source_id"],
                complete_inventory_fingerprint="1" * 64,
                candidate_set_fingerprint="2" * 64,
                evidence_fingerprint="3" * 64,
                pair_binding_fingerprints=["4" * 64],
            )
        ]
    r9.validate_dual_version_manifest(migrated)
    (root / source_intake.ACQUISITION_MANIFEST_RELATIVE).write_bytes(
        r10.r8.canonical_json_bytes(migrated)
    )
    _write_marker(root)
    return source, migrated


def _register_new_source(root, *, crash_checkpoint=None, monkeypatch=None):
    input_path = root / "new-source-input.pdf"
    input_path.write_bytes(_minimal_pdf_bytes("new-source"))
    if crash_checkpoint is not None:
        def checkpoint(name):
            if name == crash_checkpoint:
                raise SimulatedCrash(name)

        monkeypatch.setattr(
            source_intake, "_registration_publication_checkpoint", checkpoint
        )
    return source_intake.register_source(
        root,
        "new-source",
        input_path,
        "https://example.test/new-source.pdf",
        "example.test",
        FIXED_TIME,
        "new-source.pdf",
        allow_inactive_v21=True,
    )


def _install_extractor(monkeypatch):
    def succeeds(raw_fd, output_dir_fd, original_filename, force_ocr):
        _write_valid_extractor_output_fd(output_dir_fd, original_filename)
        return True

    monkeypatch.setattr(source_intake, "_run_local_extractor", succeeds)


def _identity_payload():
    metadata = {
        "judgment_number": "10/2026/DS-ST",
        "case_id": "10/2026/DS-ST",
        "court": "Tòa án nhân dân tỉnh Test",
        "decision_date": "2026-01-10",
        "procedure_stage": "so_tham",
    }
    return {
        "identity_version": source_intake.IDENTITY_PAYLOAD_VERSION,
        "verified_metadata": metadata,
        "verified_fields": {
            field: True for field in source_intake.REVIEW_METADATA_FIELDS
        },
        "related_decision_ids": [],
        "underlying_dispute_id": None,
    }


def _review_payload(root, manifest, acquisition_id):
    findings = source_intake._eligibility_review_findings(
        root, manifest, acquisition_id
    )
    return {
        "review_version": source_intake.REVIEW_PAYLOAD_VERSION,
        "reviews": {
            field: {"completed": True, "findings": findings[field]}
            for field in source_intake.REVIEW_COMPLETION_FIELDS
        },
    }


@pytest.fixture(scope="module")
def authoritative_context():
    return r10.authoritative_candidate_context(PROJECT_ROOT)


@pytest.fixture(scope="module")
def outputs():
    return r10.build_outputs(PROJECT_ROOT, "a" * 40, RUN_ID)


def test_baseline_and_r9_provenance_are_exact():
    assert policy_v2.sha256_file(MANIFEST_PATH) == r10.SOURCE_MANIFEST_SHA256
    run = json.loads(
        (PROJECT_ROOT / r10.R9_RUN_RELATIVE / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run["run_fingerprint"] == r10.R9_RUN_FINGERPRINT


def test_v1_remains_default_for_every_integrated_api(outputs):
    matrix = outputs["version_aware_intake_matrix.json"]
    assert matrix["v1_default"] is True
    assert all(item["default"] is False for item in matrix["api_defaults"].values())
    assert matrix["implicit_migration"] is False


def test_default_loader_rejects_v21_but_staging_loader_accepts(tmp_path):
    _, migrated = _install_v21_fixture(tmp_path)
    with pytest.raises(SourceIntakeError, match="Unsupported"):
        source_intake.load_acquisition_manifest(tmp_path)
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded == migrated


def test_staging_flag_without_marker_fails_closed(tmp_path):
    _install_v21_fixture(tmp_path)
    (tmp_path / source_intake.INACTIVE_V21_STAGING_MARKER_RELATIVE).unlink()
    with pytest.raises(SourceIntakeError, match="staging marker"):
        source_intake.load_acquisition_manifest(
            tmp_path, allow_inactive_v21=True
        )


def test_symlinked_staging_marker_fails_closed(tmp_path):
    _install_v21_fixture(tmp_path)
    marker = tmp_path / source_intake.INACTIVE_V21_STAGING_MARKER_RELATIVE
    target = tmp_path / "marker-target.json"
    target.write_bytes(marker.read_bytes())
    marker.unlink()
    marker.symlink_to(target)
    with pytest.raises(SourceIntakeError, match="staging marker"):
        source_intake.load_acquisition_manifest(
            tmp_path, allow_inactive_v21=True
        )


def test_live_repository_cannot_enable_inactive_v21_path():
    with pytest.raises(SourceIntakeError, match="live repository"):
        source_intake.load_acquisition_manifest(
            PROJECT_ROOT, allow_inactive_v21=True
        )


def test_unknown_manifest_version_rejected_even_with_staging_marker(tmp_path):
    _install_v21_fixture(tmp_path)
    path = tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE
    value = json.loads(path.read_text())
    value["manifest_version"] = "future-unknown"
    path.write_bytes(r10.r8.canonical_json_bytes(value))
    with pytest.raises(SourceIntakeError, match="Unsupported"):
        source_intake.load_acquisition_manifest(
            tmp_path, allow_inactive_v21=True
        )


def test_actual_v21_registration_preserves_review_records(tmp_path):
    _, before = _install_v21_fixture(tmp_path)
    result = _register_new_source(tmp_path)
    assert result["acquisition_status"] == "hash_verified"
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded["review_records"] == before["review_records"]
    assert any(item["acquisition_id"] == "new-source" for item in loaded["sources"])


def test_actual_v21_extraction_preserves_review_records(tmp_path, monkeypatch):
    _, before = _install_v21_fixture(tmp_path)
    _register_new_source(tmp_path)
    _install_extractor(monkeypatch)
    result = source_intake.extract_intake_source(
        tmp_path, "new-source", allow_inactive_v21=True
    )
    assert result["acquisition_status"] == "extracted"
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded["review_records"] == before["review_records"]


def test_actual_v21_identity_on_unrelated_source_preserves_record(
    tmp_path, monkeypatch
):
    _, before = _install_v21_fixture(tmp_path)
    _register_new_source(tmp_path)
    _install_extractor(monkeypatch)
    source_intake.extract_intake_source(
        tmp_path, "new-source", allow_inactive_v21=True
    )
    result = source_intake.record_source_identity(
        tmp_path,
        "new-source",
        _identity_payload(),
        allow_inactive_v21=True,
    )
    assert result["acquisition_status"] == "extracted"
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded["review_records"] == before["review_records"]


def test_target_identity_change_with_record_is_rejected_and_rolled_back(tmp_path):
    source, before = _install_v21_fixture(tmp_path)
    path = tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE
    original = path.read_bytes()
    payload = _identity_payload()
    payload["verified_metadata"]["court"] = "Tòa án khác"
    with pytest.raises(SourceIntakeError):
        source_intake.record_source_identity(
            tmp_path,
            source["acquisition_id"],
            payload,
            allow_inactive_v21=True,
        )
    assert path.read_bytes() == original
    assert source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )["review_records"] == before["review_records"]


def test_actual_v1_review_on_unrelated_v21_source_preserves_record(
    tmp_path, monkeypatch
):
    _, before = _install_v21_fixture(tmp_path)
    _register_new_source(tmp_path)
    _install_extractor(monkeypatch)
    source_intake.extract_intake_source(
        tmp_path, "new-source", allow_inactive_v21=True
    )
    source_intake.record_source_identity(
        tmp_path,
        "new-source",
        _identity_payload(),
        allow_inactive_v21=True,
    )
    manifest = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    payload = _review_payload(tmp_path, manifest, "new-source")
    result = source_intake.record_source_review(
        tmp_path,
        "new-source",
        payload,
        allow_inactive_v21=True,
    )
    assert result["acquisition_status"] == "extracted"
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded["review_records"] == before["review_records"]


def test_actual_v21_inventory_read_preserves_manifest(tmp_path):
    _install_v21_fixture(tmp_path)
    path = tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE
    before = path.read_bytes()
    inventory, report, links = source_intake.inventory_independent_sources(
        tmp_path, allow_inactive_v21=True
    )
    assert inventory == []
    assert report["source_count"] == 1
    assert links == []
    assert path.read_bytes() == before


def test_v21_promotion_is_explicitly_not_authorized(tmp_path):
    source, _ = _install_v21_fixture(tmp_path)
    with pytest.raises(SourceIntakeError, match="not authorized"):
        source_intake.promote_source_to_eligible(
            tmp_path,
            source["acquisition_id"],
            allow_inactive_v21=True,
        )


def test_registration_prepublication_failure_is_clean_and_preserves_record(
    tmp_path, monkeypatch
):
    _, before = _install_v21_fixture(tmp_path)

    def fail(name):
        if name == "before_manifest_publication":
            raise RuntimeError("injected")

    monkeypatch.setattr(source_intake, "_registration_publication_checkpoint", fail)
    with pytest.raises(RuntimeError, match="injected"):
        _register_new_source(tmp_path)
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded["review_records"] == before["review_records"]
    assert not any(item["acquisition_id"] == "new-source" for item in loaded["sources"])


def test_registration_manifest_published_crash_recovers_v21(
    tmp_path, monkeypatch
):
    _, before = _install_v21_fixture(tmp_path)
    with pytest.raises(SimulatedCrash):
        _register_new_source(
            tmp_path,
            crash_checkpoint="after_manifest_published",
            monkeypatch=monkeypatch,
        )
    journals = list((tmp_path / source_intake.TRANSACTION_ROOT_RELATIVE).glob("*.register.json"))
    assert len(journals) == 1
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    assert loaded["review_records"] == before["review_records"]
    assert any(item["acquisition_id"] == "new-source" for item in loaded["sources"])
    assert not journals[0].exists()
    assert source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    ) == loaded


def test_extraction_artifact_published_crash_rolls_back_v21(
    tmp_path, monkeypatch
):
    _, before = _install_v21_fixture(tmp_path)
    _register_new_source(tmp_path)
    _install_extractor(monkeypatch)

    def crash(name):
        if name == "after_artifact_published":
            raise SimulatedCrash(name)

    monkeypatch.setattr(source_intake, "_publication_checkpoint", crash)
    with pytest.raises(SimulatedCrash):
        source_intake.extract_intake_source(
            tmp_path, "new-source", allow_inactive_v21=True
        )
    loaded = source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )
    source = next(item for item in loaded["sources"] if item["acquisition_id"] == "new-source")
    assert source["acquisition_status"] == "hash_verified"
    assert loaded["review_records"] == before["review_records"]
    assert not list((tmp_path / source_intake.TRANSACTION_ROOT_RELATIVE).glob("*.json"))
    monkeypatch.setattr(source_intake, "_publication_checkpoint", lambda name: None)
    retried = source_intake.extract_intake_source(
        tmp_path, "new-source", allow_inactive_v21=True
    )
    assert retried["acquisition_status"] == "extracted"
    assert source_intake.load_acquisition_manifest(
        tmp_path, allow_inactive_v21=True
    )["review_records"] == before["review_records"]


def test_unknown_journal_version_fails_closed(tmp_path):
    source, _ = _install_v21_fixture(tmp_path)
    manifest_bytes = (tmp_path / source_intake.ACQUISITION_MANIFEST_RELATIVE).read_bytes()
    journal = source_intake._journal_payload(
        {
            "journal_version": "future_unknown",
            "transaction_id": "0" * 32,
            "acquisition_id": source["acquisition_id"],
            "state": "PREPARED",
            "previous_manifest_sha256": r10.r8.sha256_bytes(manifest_bytes),
            "intended_manifest_sha256": "0" * 64,
            "previous_status": "hash_verified",
            "intended_status": "extracted",
            "temp_directory_name": ".extract-partial-" + "0" * 32,
            "final_artifact_name": "recorded_structured.json",
            "final_relative_path": source["structured_path"],
            "structured_sha256": source["structured_sha256"],
            "created_at": FIXED_TIME,
            "updated_at": FIXED_TIME,
        }
    )
    path = tmp_path / source_intake.TRANSACTION_ROOT_RELATIVE / f"{source['acquisition_id']}.json"
    path.write_bytes(r10.r8.canonical_json_bytes(journal))
    with pytest.raises(SourceIntakeError, match="Unsupported publication journal"):
        source_intake.load_acquisition_manifest(
            tmp_path, allow_inactive_v21=True
        )


def test_caller_freshness_comparison_never_claims_authoritative(authoritative_context):
    binding = r10.authoritative_binding_for_source(
        authoritative_context, "independent:batch2dc-20261105-025"
    )
    manifest = r9.migrate_v1_to_v21_copy(authoritative_context["context"]["manifest"])
    record = r9.build_v21_evidence_record(
        manifest,
        "independent:batch2dc-20261105-025",
        **binding,
    )
    result = r10.caller_freshness_comparison(record, binding)
    assert result["authoritative_inputs_loaded"] is False
    assert result["full_inventory_recomputed"] is False
    assert result["promotion_gate_verified"] is False
    assert result["caller_claim_can_authorize_promotion"] is False


def test_verified_gate_recomputes_full_inventory_but_never_promotes(authoritative_context):
    source_id = "independent:batch2dc-20261105-025"
    binding = r10.authoritative_binding_for_source(authoritative_context, source_id)
    manifest = r9.migrate_v1_to_v21_copy(authoritative_context["context"]["manifest"])
    record = r9.build_v21_evidence_record(
        manifest,
        source_id,
        **binding,
        canonical_disposition="CANONICAL_RETAINED",
    )
    result = r10.verified_promotion_time_gate(
        PROJECT_ROOT, record, context=authoritative_context
    )
    assert result["authoritative_inputs_loaded"] is True
    assert result["full_inventory_recomputed"] is True
    assert result["inventory_count"] == 469
    assert result["candidate_pair_count"] == 26
    assert result["stale_inventory_binding_fields"] == []
    assert result["ready_for_promotion"] is False
    assert "TRUSTED_REVIEW_ATTESTATION_MISSING" in result["failures"]
    assert result["mutation_performed"] is False


def test_stale_inventory_binding_is_detected_by_verified_gate(authoritative_context):
    source_id = "independent:batch2dc-20261105-025"
    binding = r10.authoritative_binding_for_source(authoritative_context, source_id)
    manifest = r9.migrate_v1_to_v21_copy(authoritative_context["context"]["manifest"])
    record = r9.build_v21_evidence_record(manifest, source_id, **binding)
    record["inventory_binding"]["candidate_set_fingerprint"] = "0" * 64
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    result = r10.verified_promotion_time_gate(
        PROJECT_ROOT, record, context=authoritative_context
    )
    assert result["stale_inventory_binding_fields"] == ["candidate_set_fingerprint"]


def test_all_actual_frozen_relation_labels_are_mapped(authoritative_context):
    results = [
        r10.validate_relation_semantics(
            authoritative_context["context"]["policy"],
            item["evidence"],
            item["decision"],
        )
        for item in authoritative_context["evaluated"]
    ]
    assert len(results) == 26
    assert all(item["mapping_version"] == r10.RELATION_MAPPING_VERSION for item in results)


def test_relation_decision_mismatch_fails_closed(authoritative_context):
    item = authoritative_context["evaluated"][0]
    forged = copy.deepcopy(item["decision"])
    forged["same_document_relation"] = "UNKNOWN_OR_CLEARED"
    with pytest.raises(r10.VersionAwareIntegrationError, match="differs"):
        r10.validate_relation_semantics(
            authoritative_context["context"]["policy"], item["evidence"], forged
        )


def test_ambiguous_relations_are_never_treated_as_complete(authoritative_context):
    rows = [
        r10.validate_relation_semantics(
            authoritative_context["context"]["policy"], item["evidence"]
        )
        for item in authoritative_context["evaluated"]
    ]
    ambiguous = [item for item in rows if item["unresolved_relation_labels"]]
    assert ambiguous
    assert all(item["evidence_complete_for_emitted_relations"] is False for item in ambiguous)


def test_canonical_and_dispute_controls_remain_fail_closed(authoritative_context):
    decisions = {item["acquisition_id"]: item for item in authoritative_context["decisions"]}
    assert decisions["batch2dc-20261105-025"]["canonical_source_id"].endswith("025")
    assert decisions["batch2dc-20261105-026"]["noncanonical_exact_duplicate"] is True
    assert decisions["batch2dc-20261105-027"]["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED"
    assert decisions["batch2dc-20261105-029"]["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED"
    for acquisition_id in ("008", "014", "022", "024", "034"):
        assert decisions[f"batch2dc-20261105-{acquisition_id}"]["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED"


def test_trusted_registry_rejects_caller_authority():
    with pytest.raises(r10.VersionAwareIntegrationError, match="authority string"):
        r10.verify_trusted_producer_attestation("AUTHORIZED")
    with pytest.raises(r10.VersionAwareIntegrationError, match="registry is inactive"):
        r10.verify_trusted_producer_attestation({"producer": "caller"})


def test_outputs_are_inactive_and_self_fingerprinted(outputs):
    assert set(outputs) == {
        "version_aware_intake_matrix.json",
        "reader_writer_integration_tests.json",
        "journal_recovery_validation.json",
        "binding_freshness_verification.json",
        "relation_semantics_validation.json",
        "trusted_authority_safety.json",
        "migration_activation_readiness.json",
        "validation_summary.json",
    }
    readiness = outputs["migration_activation_readiness.json"]
    assert readiness["result"] == "ENGINEERING_INTEGRATION_VERIFIED_NOT_ACTIVATED"
    assert readiness["v21_live_read_enabled"] is False
    assert readiness["v21_live_write_enabled"] is False
    assert readiness["activation_ready"] is False
    for payload in outputs.values():
        assert payload["artifact_fingerprint"] == policy_v2.canonical_fingerprint(
            payload, excluded_key="artifact_fingerprint"
        )


def test_protected_state_and_live_manifest_remain_unchanged(outputs):
    summary = outputs["validation_summary.json"]
    assert summary["protected_state"]["global"] == {
        "total": 363,
        "eligible": 257,
        "extracted": 105,
        "hash_verified": 1,
    }
    assert summary["protected_state"]["batch2dc"] == {
        "total": 46,
        "extracted": 46,
        "identity_ready": 46,
        "eligible": 0,
    }
    assert summary["protected_state"]["benchmark_state"] == "DRAFT"
    assert summary["protected_state"]["sampling_state"] == "DRAFT"
    assert summary["protected_state"]["selection_status"] == "NOT_STARTED"
    assert summary["protected_state"]["sampling_seed"] is None
    assert summary["live_manifest_mutated"] is False
    assert summary["review_or_promotion_authorized"] is False
    assert policy_v2.sha256_file(MANIFEST_PATH) == r10.SOURCE_MANIFEST_SHA256
