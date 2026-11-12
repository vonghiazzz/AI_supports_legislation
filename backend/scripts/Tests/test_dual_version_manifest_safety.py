import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as policy_v2
from backend.scripts.Evaluation.benchmarking import dual_version_manifest_safety as r9
from backend.scripts.Evaluation.benchmarking import source_intake as v1
from backend.scripts.Evaluation.benchmarking import source_review_v2_migration as r8


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / r8.r6.MANIFEST_RELATIVE
RUN_ID = r9.PROTOCOL_ID + "_test"


@pytest.fixture(scope="module")
def historical():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v21(historical):
    return r9.migrate_v1_to_v21_copy(historical)


@pytest.fixture(scope="module")
def v21_with_record(v21):
    value = copy.deepcopy(v21)
    value["review_records"] = [
        r9.build_v21_evidence_record(
            value,
            "independent:batch2dc-20261105-025",
            complete_inventory_fingerprint="1" * 64,
            candidate_set_fingerprint="2" * 64,
            evidence_fingerprint="3" * 64,
            pair_binding_fingerprints=["4" * 64],
            canonical_disposition="CANONICAL_RETAINED",
        )
    ]
    return r9.validate_dual_version_manifest(value)


@pytest.fixture(scope="module")
def outputs():
    return r9.build_outputs(PROJECT_ROOT, "a" * 40, RUN_ID)


def _new_source_from(source):
    value = copy.deepcopy(source)
    timestamp = source["download_timestamp"]
    value.update(
        {
            "acquisition_id": "fixture-new-source",
            "source_id": v1.stable_source_id("fixture-new-source"),
            "original_filename": "fixture-new-source.pdf",
            "stored_pdf_path": "backend/dataLaw/independent_intake/raw/fixture-new-source/fixture-new-source.pdf",
            "source_url": None,
            "offline_provenance": {"description": "isolated R9 fixture"},
            "source_domain": "offline.local",
            "download_timestamp": timestamp,
            "sha256": "a" * 64,
            "case_id": None,
            "judgment_number": None,
            "court": None,
            "decision_date": None,
            "procedure_stage": None,
            "underlying_dispute_id": None,
            "related_decision_ids": [],
            "structured_path": "backend/dataLaw/independent_intake/structured/fixture-new-source/fixture-new-source_structured.json",
            "structured_sha256": None,
            "metadata_verification": {field: False for field in v1.VERIFICATION_FIELDS},
            "acquisition_status": "downloaded",
            "status_history": [
                {"status": "downloaded", "timestamp": timestamp, "reasons": []}
            ],
            "rejection_reasons": [],
            "notes": "isolated fixture",
        }
    )
    return value


def test_baseline_and_r8_provenance_are_frozen():
    assert policy_v2.sha256_file(MANIFEST_PATH) == r9.SOURCE_MANIFEST_SHA256
    assert policy_v2.load_and_validate_policy(PROJECT_ROOT)["policy_fingerprint"] == (
        r9.POLICY_FINGERPRINT
    )
    run = json.loads(
        (PROJECT_ROOT / r9.R8_RUN_RELATIVE / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run["run_fingerprint"] == r9.R8_RUN_FINGERPRINT


def test_dual_reader_accepts_exact_v1_r8_and_v21(historical, v21):
    assert r9.validate_dual_version_manifest(historical)["manifest_version"] == (
        v1.MANIFEST_VERSION
    )
    r8_manifest = r8.migrate_manifest_v1_to_v2_copy(historical)
    assert r9.validate_dual_version_manifest(r8_manifest)["manifest_version"] == (
        r8.V2_MANIFEST_VERSION
    )
    assert r9.validate_dual_version_manifest(v21)["manifest_version"] == (
        r9.V21_MANIFEST_VERSION
    )


@pytest.mark.parametrize("version", ["unknown", "", None, 3])
def test_unknown_manifest_versions_fail_closed(historical, version):
    value = copy.deepcopy(historical)
    value["manifest_version"] = version
    with pytest.raises(r9.DualVersionSafetyError, match="Unsupported"):
        r9.validate_dual_version_manifest(value)


def test_v21_migration_preserves_exact_v1_projection(historical, v21):
    assert r9._v1_projection(v21) == historical
    assert v21["review_records"] == []
    assert v21["review_schema"]["binding_strategy"].startswith("SOURCE_LOCAL")


def test_v1_projected_update_remains_v1(historical):
    def update(manifest):
        manifest["sources"][0]["notes"] = "isolated update"

    result = r9.apply_projected_v1_update(historical, "review_metadata", update)
    assert result["manifest_version"] == v1.MANIFEST_VERSION
    v1.validate_acquisition_manifest(result, prior_manifest=historical)


def test_r8_empty_review_collection_can_be_updated_without_record_loss(historical):
    r8_manifest = r8.migrate_manifest_v1_to_v2_copy(historical)

    def update(manifest):
        manifest["sources"][0]["notes"] = "isolated update"

    result = r9.apply_projected_v1_update(r8_manifest, "review_metadata", update)
    assert result["manifest_version"] == r8.V2_MANIFEST_VERSION
    assert result["review_records"] == []


def test_r8_manifest_wide_binding_fails_closed_when_records_exist(historical):
    r8_manifest = r8.migrate_manifest_v1_to_v2_copy(historical)
    source = next(
        item
        for item in historical["sources"]
        if item["acquisition_id"] == "batch2dc-20261105-025"
    )
    r8_manifest["review_records"] = [
        r8.build_evidence_only_review_record(
            source,
            r9.SOURCE_MANIFEST_SHA256,
            complete_inventory_fingerprint="1" * 64,
            candidate_set_fingerprint="2" * 64,
            evidence_fingerprint="3" * 64,
            pair_binding_fingerprints=["4" * 64],
        )
    ]
    r8.validate_versioned_manifest(r8_manifest)
    with pytest.raises(r9.DualVersionSafetyError, match="manifest-wide"):
        r9.apply_projected_v1_update(
            r8_manifest,
            "review_metadata",
            lambda manifest: manifest["sources"][0].update(notes="changed"),
        )


def test_v21_unrelated_update_preserves_review_record(v21_with_record):
    original_record = copy.deepcopy(v21_with_record["review_records"][0])

    def update(manifest):
        source = next(
            item
            for item in manifest["sources"]
            if item["source_id"] != original_record["source_id"]
        )
        source["notes"] = "unrelated isolated update"

    result = r9.apply_projected_v1_update(v21_with_record, "review_metadata", update)
    assert result["review_records"] == [original_record]
    r9.validate_dual_version_manifest(result)


def test_unrelated_extracted_to_eligible_preserves_local_record(v21_with_record):
    record = copy.deepcopy(v21_with_record["review_records"][0])

    def promote_unrelated(manifest):
        source = next(
            item
            for item in manifest["sources"]
            if item["source_id"] != record["source_id"]
            and item["acquisition_status"] == "extracted"
        )
        timestamp = source["status_history"][-1]["timestamp"]
        source["acquisition_status"] = "eligible"
        source["status_history"].extend(
            [
                {"status": "metadata_verified", "timestamp": timestamp, "reasons": []},
                {"status": "dedup_checked", "timestamp": timestamp, "reasons": []},
                {"status": "eligible", "timestamp": timestamp, "reasons": []},
            ]
        )

    result = r9.apply_projected_v1_update(
        v21_with_record, "status_transition", promote_unrelated
    )
    assert result["review_records"] == [record]


def test_v21_target_identity_change_stales_local_record(v21_with_record):
    target = v21_with_record["review_records"][0]["source_id"]

    def update(manifest):
        source = next(item for item in manifest["sources"] if item["source_id"] == target)
        source["court"] = source["court"] + " changed"

    with pytest.raises(r9.DualVersionSafetyError, match="source-local"):
        r9.apply_projected_v1_update(v21_with_record, "identity", update)


def test_v21_target_artifact_hash_change_stales_local_record(v21_with_record):
    target = v21_with_record["review_records"][0]["source_id"]

    def update(manifest):
        source = next(item for item in manifest["sources"] if item["source_id"] == target)
        source["structured_sha256"] = "0" * 64

    with pytest.raises(r9.DualVersionSafetyError, match="source-local"):
        r9.apply_projected_v1_update(v21_with_record, "extraction", update)


def test_v21_registration_projection_preserves_existing_record(v21_with_record):
    source_template = r9._v1_projection(v21_with_record)["sources"][0]

    def register(manifest):
        source = _new_source_from(source_template)
        manifest["sources"].append(source)
        manifest["immutable_provenance"].append(v1.immutable_provenance_snapshot(source))

    result = r9.apply_projected_v1_update(v21_with_record, "registration", register)
    assert len(result["sources"]) == len(v21_with_record["sources"]) + 1
    assert result["review_records"] == v21_with_record["review_records"]


def test_v21_extraction_projection_preserves_existing_record(v21_with_record):
    source_template = r9._v1_projection(v21_with_record)["sources"][0]

    def register_and_extract(manifest):
        source = _new_source_from(source_template)
        timestamp = source["download_timestamp"]
        source["structured_sha256"] = "b" * 64
        source["acquisition_status"] = "extracted"
        source["status_history"].extend(
            [
                {"status": "hash_verified", "timestamp": timestamp, "reasons": []},
                {"status": "extracted", "timestamp": timestamp, "reasons": []},
            ]
        )
        manifest["sources"].append(source)
        manifest["immutable_provenance"].append(v1.immutable_provenance_snapshot(source))

    result = r9.apply_projected_v1_update(
        v21_with_record, "extraction", register_and_extract
    )
    new_source = next(
        item for item in result["sources"] if item["acquisition_id"] == "fixture-new-source"
    )
    assert new_source["acquisition_status"] == "extracted"
    assert result["review_records"] == v21_with_record["review_records"]


def test_unsupported_adapter_operation_does_not_fallback(v21):
    with pytest.raises(r9.DualVersionSafetyError, match="Unsupported adapter"):
        r9.apply_projected_v1_update(v21, "promotion", lambda manifest: None)


def test_unknown_policy_binding_is_rejected(v21_with_record):
    value = copy.deepcopy(v21_with_record)
    record = value["review_records"][0]
    record["policy_binding"]["policy_fingerprint"] = "0" * 64
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r9.DualVersionSafetyError, match="policy binding"):
        r9.validate_dual_version_manifest(value)


def test_forged_authority_string_is_rejected(v21_with_record):
    value = copy.deepcopy(v21_with_record)
    record = value["review_records"][0]
    record["authority_attestation"] = "AUTHORIZED_VERSIONED_REVIEW_PRODUCER"
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r9.DualVersionSafetyError, match="Caller-provided"):
        r9.validate_dual_version_manifest(value)
    with pytest.raises(r9.DualVersionSafetyError, match="authority string"):
        r9.verify_trusted_attestation_reference("FORGED")


def test_structural_attestation_is_not_accepted_without_registry():
    with pytest.raises(r9.DualVersionSafetyError, match="registry is not activated"):
        r9.verify_trusted_attestation_reference({"producer": "caller"})


def test_inventory_staleness_is_separate_from_source_local_validity(v21_with_record):
    record = v21_with_record["review_records"][0]
    result = r9.assess_promotion_readiness(
        record,
        current_policy_fingerprint=r9.POLICY_FINGERPRINT,
        current_complete_inventory_fingerprint="9" * 64,
        current_candidate_set_fingerprint="8" * 64,
        current_evidence_fingerprint="7" * 64,
        current_pair_binding_fingerprints=["6" * 64],
        new_cross_version_blockers=["new duplicate"],
    )
    assert result["source_local_evidence_valid"] is True
    assert "STALE_COMPLETE_INVENTORY_FINGERPRINT" in result["failures"]
    assert "NEW_CROSS_VERSION_DUPLICATE_OR_DISPUTE_EVIDENCE" in result["failures"]
    assert result["ready_for_promotion"] is False


def test_policy_change_stales_promotion_readiness(v21_with_record):
    record = v21_with_record["review_records"][0]
    inventory = record["inventory_binding"]
    result = r9.assess_promotion_readiness(
        record,
        current_policy_fingerprint="0" * 64,
        current_complete_inventory_fingerprint=inventory[
            "complete_inventory_fingerprint"
        ],
        current_candidate_set_fingerprint=inventory["candidate_set_fingerprint"],
        current_evidence_fingerprint=inventory["evidence_fingerprint"],
        current_pair_binding_fingerprints=inventory["pair_binding_fingerprints"],
    )
    assert "STALE_POLICY_BINDING" in result["failures"]


def test_exact_duplicate_does_not_require_document_type():
    result = r9.relation_evidence_complete(
        "EXACT_DOCUMENT_DUPLICATE",
        {"raw_sha_equal": True, "document_type_equal": None},
        source_identity_complete=True,
        counterpart_identity_complete=True,
    )
    assert result["complete"] is True
    assert result["document_type_universally_required"] is False


def test_distinct_control_needs_only_one_verified_contextual_conflict():
    result = r9.relation_evidence_complete(
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
    assert result["complete"] is True


def test_distinct_control_without_context_conflict_fails_closed():
    result = r9.relation_evidence_complete(
        "DISTINCT_DOCUMENT_CONTROL",
        {
            "raw_sha_equal": False,
            "structured_sha_equal": False,
            "text_similarity": 0.1,
            "court_equal": None,
            "decision_date_equal": None,
            "decision_year_equal": None,
            "document_type_equal": None,
            "procedure_stage_equal": None,
        },
        source_identity_complete=True,
        counterpart_identity_complete=True,
    )
    assert result["complete"] is False
    assert "VERIFIED_CONTEXTUAL_CONFLICT_REQUIRED" in result["missing_requirements"]


def test_dispute_signal_requires_explicit_adjudication_and_identity():
    incomplete = r9.relation_evidence_complete(
        "UNDERLYING_DISPUTE",
        {"related_decision": True, "same_underlying_dispute": None},
        source_identity_complete=True,
        counterpart_identity_complete=False,
        explicit_dispute_adjudication=False,
    )
    assert incomplete["complete"] is False
    assert "COUNTERPART_IDENTITY_INCOMPLETE" in incomplete["missing_requirements"]
    assert "EXPLICIT_DISPUTE_ADJUDICATION_REQUIRED" in incomplete[
        "missing_requirements"
    ]


def _write_fixture(tmp_path, historical):
    path = tmp_path / "acquisition_manifest.fixture.json"
    path.write_bytes(r9._canonical_bytes(historical))
    return path


def test_atomic_fixture_migration_and_idempotent_retry(tmp_path, historical):
    path = _write_fixture(tmp_path, historical)
    first = r9.atomic_migrate_fixture(path, r9._sha256_bytes(path.read_bytes()))
    second = r9.atomic_migrate_fixture(path, r9._sha256_bytes(path.read_bytes()))
    assert first["result"] == "MIGRATED"
    assert second["result"] == "IDEMPOTENT_ALREADY_MIGRATED"
    assert json.loads(path.read_text())["manifest_version"] == r9.V21_MANIFEST_VERSION


@pytest.mark.parametrize(
    "fault",
    [
        "invalid_schema",
        "stale_manifest",
        "failed_replacement",
        "post_replacement_verification",
    ],
)
def test_atomic_failures_leave_original_v1_fixture(tmp_path, historical, fault):
    path = _write_fixture(tmp_path, historical)
    original = path.read_bytes()
    with pytest.raises(r9.DualVersionSafetyError):
        r9.atomic_migrate_fixture(path, r9._sha256_bytes(original), fault=fault)
    assert path.read_bytes() == original


def test_concurrent_modification_is_not_overwritten(tmp_path, historical):
    path = _write_fixture(tmp_path, historical)
    original = path.read_bytes()
    with pytest.raises(r9.DualVersionSafetyError, match="Concurrent"):
        r9.atomic_migrate_fixture(
            path, r9._sha256_bytes(original), fault="concurrent_modification"
        )
    assert path.read_bytes() != original
    assert json.loads(path.read_text())["manifest_version"] == v1.MANIFEST_VERSION


@pytest.mark.parametrize("fault", ["interrupt_after_replacement", "recovery_failure"])
def test_post_replace_interruption_leaves_valid_v21_and_retryable(
    tmp_path, historical, fault
):
    path = _write_fixture(tmp_path, historical)
    with pytest.raises(r9.DualVersionSafetyError):
        r9.atomic_migrate_fixture(
            path, r9._sha256_bytes(path.read_bytes()), fault=fault
        )
    current = json.loads(path.read_text())
    assert r9.validate_dual_version_manifest(current)["manifest_version"] == (
        r9.V21_MANIFEST_VERSION
    )
    retry = r9.atomic_migrate_fixture(path, r9._sha256_bytes(path.read_bytes()))
    assert retry["result"] == "IDEMPOTENT_ALREADY_MIGRATED"


def test_live_manifest_path_is_explicitly_forbidden():
    with pytest.raises(r9.DualVersionSafetyError, match="Live acquisition"):
        r9.atomic_migrate_fixture(MANIFEST_PATH, r9.SOURCE_MANIFEST_SHA256)


def test_binding_stability_covers_all_required_events(outputs):
    report = outputs["review_binding_stability_analysis.json"]
    assert {item["event"] for item in report["events"]} == {
        "unrelated_extracted_to_eligible",
        "new_source_registered",
        "new_source_extracted",
        "reviewed_source_identity_changes",
        "counterpart_newly_duplicate",
        "related_decision_edge_appears",
        "policy_version_changes",
        "evidence_bytes_or_hashes_change",
    }
    assert report["historical_records_auto_rewritten"] is False
    assert report["r8_contract_modified"] is False


def test_exact_duplicate_and_dispute_controls_remain_bound():
    gate = json.loads(
        (
            PROJECT_ROOT
            / r8.R7_RUN_RELATIVE
            / "eligibility_gate_safety_report.json"
        ).read_text(encoding="utf-8")
    )
    targeted = json.loads(
        (
            PROJECT_ROOT / r8.R7_RUN_RELATIVE / "targeted_dispute_adjudication.json"
        ).read_text(encoding="utf-8")
    )
    canonical = json.loads(
        (
            PROJECT_ROOT
            / r8.R7_RUN_RELATIVE
            / "canonical_retention_consistency.json"
        ).read_text(encoding="utf-8")
    )
    assert canonical["canonical_source_id"].endswith("025")
    assert canonical["noncanonical_source_id"].endswith("026")
    assert len(targeted["adjudications"]) == 3
    assert {item[0] for item in r9.r7.TARGET_PAIRS} == {
        "independent:batch2dc-20261105-027",
        "independent:batch2dc-20261105-029",
    }
    assert gate["unresolved_disputes_rejected"] is True
    assert sorted(r9.r7.ORIGINAL_UNRESOLVED) == [
        "batch2dc-20261105-008",
        "batch2dc-20261105-014",
        "batch2dc-20261105-022",
        "batch2dc-20261105-024",
        "batch2dc-20261105-034",
    ]


def test_outputs_distinguish_simulation_from_production(outputs):
    lifecycle = outputs["isolated_lifecycle_simulation.json"]
    matrix = outputs["reader_writer_compatibility_matrix.json"]
    verdict = outputs["migration_readiness_verdict.json"]
    assert lifecycle["review_record_preserved"] is True
    assert lifecycle["design_only_not_verified_in_production"]
    assert matrix["production_compatibility_claimed"] is False
    assert verdict["r8_schema_direct_activation_ready"] is False
    assert verdict["v21_amendment_scaffold_verified"] is True
    assert verdict["live_migration_authorized"] is False


def test_atomic_failure_injection_artifact_is_complete(outputs):
    report = outputs["atomic_migration_failure_injection.json"]
    assert report["success"]["result"] == "MIGRATED"
    assert report["idempotent_retry"]["result"] == "IDEMPOTENT_ALREADY_MIGRATED"
    assert set(report["failure_cases"]) == {
        "invalid_schema",
        "stale_manifest",
        "concurrent_modification",
        "failed_replacement",
        "post_replacement_verification",
        "recovery_failure",
        "interrupt_after_replacement",
    }
    assert report["live_manifest_mutated"] is False


def test_relation_amendment_is_proposal_only(outputs):
    report = outputs["relation_specific_evidence_amendment.json"]
    assert report["status"] == "PROPOSAL_NOT_ACTIVATED"
    assert report["document_type_universally_required"] is False
    assert report["frozen_policy_modified"] is False
    assert "65-source audit does not establish" in report["inventory_scope_caveat"]


def test_protected_state_and_live_manifest_are_unchanged(outputs):
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
    assert policy_v2.sha256_file(MANIFEST_PATH) == r9.SOURCE_MANIFEST_SHA256


def test_all_nine_outputs_have_deterministic_fingerprints(outputs):
    assert set(outputs) == {
        "reader_writer_compatibility_matrix.json",
        "dual_version_adapter_validation.json",
        "review_binding_stability_analysis.json",
        "isolated_lifecycle_simulation.json",
        "atomic_migration_failure_injection.json",
        "relation_specific_evidence_amendment.json",
        "trusted_review_boundary_report.json",
        "migration_readiness_verdict.json",
        "validation_summary.json",
    }
    for payload in outputs.values():
        assert payload["artifact_fingerprint"] == policy_v2.canonical_fingerprint(
            payload, excluded_key="artifact_fingerprint"
        )
