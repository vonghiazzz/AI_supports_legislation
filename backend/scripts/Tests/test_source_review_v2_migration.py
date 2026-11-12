import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as policy_v2
from backend.scripts.Evaluation.benchmarking import source_intake as v1
from backend.scripts.Evaluation.benchmarking import source_review_v2_migration as r8


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = PROJECT_ROOT / r8.r6.MANIFEST_RELATIVE
RUN_ID = r8.PROTOCOL_ID + "_test"


@pytest.fixture(scope="module")
def historical_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def migrated_manifest(historical_manifest):
    return r8.migrate_manifest_v1_to_v2_copy(historical_manifest)


@pytest.fixture(scope="module")
def outputs():
    return r8.build_outputs(PROJECT_ROOT, "a" * 40, RUN_ID)


def _reviewable_source(manifest):
    return next(
        source
        for source in manifest["sources"]
        if source["acquisition_id"] == "batch2dc-20261105-025"
    )


def _record(source):
    return r8.build_evidence_only_review_record(
        source,
        r8.SOURCE_MANIFEST_SHA256,
        complete_inventory_fingerprint="1" * 64,
        candidate_set_fingerprint="2" * 64,
        evidence_fingerprint="3" * 64,
        pair_binding_fingerprints=["4" * 64],
        canonical_disposition="CANONICAL_RETAINED",
        nonblocking_findings=["RAW_SHA_CLUSTER_CANONICAL_RETAINED"],
    )


def test_frozen_baseline_bindings_are_exact():
    assert policy_v2.sha256_file(MANIFEST_PATH) == r8.SOURCE_MANIFEST_SHA256
    assert policy_v2.load_and_validate_policy(PROJECT_ROOT)["policy_fingerprint"] == (
        r8.POLICY_FINGERPRINT
    )
    run_manifest = json.loads(
        (PROJECT_ROOT / r8.R7_RUN_RELATIVE / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert run_manifest["run_fingerprint"] == r8.R7_RUN_FINGERPRINT


def test_historical_v1_remains_valid_and_exact_schema(historical_manifest):
    assert set(historical_manifest) == r8.V1_ROOT_FIELDS
    assert historical_manifest["manifest_version"] == v1.MANIFEST_VERSION
    v1.validate_acquisition_manifest(historical_manifest)
    assert r8.validate_versioned_manifest(historical_manifest) == historical_manifest


def test_migration_is_in_memory_and_does_not_mutate_input(historical_manifest):
    before = copy.deepcopy(historical_manifest)
    migrated = r8.migrate_manifest_v1_to_v2_copy(historical_manifest)
    assert historical_manifest == before
    assert migrated is not historical_manifest
    assert migrated["manifest_version"] == r8.V2_MANIFEST_VERSION
    assert migrated["review_records"] == []


def test_v1_projection_is_byte_and_semantic_equivalent(
    historical_manifest, migrated_manifest
):
    projected = r8.project_v2_manifest_to_v1(migrated_manifest)
    assert projected == historical_manifest
    assert r8.canonical_json_bytes(projected) == MANIFEST_PATH.read_bytes()


def test_source_identity_provenance_status_and_history_are_preserved(
    historical_manifest, migrated_manifest
):
    projected = r8.project_v2_manifest_to_v1(migrated_manifest)
    assert projected["sources"] == historical_manifest["sources"]
    assert projected["immutable_provenance"] == historical_manifest["immutable_provenance"]
    assert [item["acquisition_status"] for item in projected["sources"]] == [
        item["acquisition_status"] for item in historical_manifest["sources"]
    ]
    assert [item["status_history"] for item in projected["sources"]] == [
        item["status_history"] for item in historical_manifest["sources"]
    ]


def test_migration_and_plan_fingerprints_are_deterministic(historical_manifest):
    first = r8.migrate_manifest_v1_to_v2_copy(historical_manifest)
    second = r8.migrate_manifest_v1_to_v2_copy(historical_manifest)
    assert r8.canonical_json_bytes(first) == r8.canonical_json_bytes(second)
    assert r8.migration_plan(historical_manifest) == r8.migration_plan(
        historical_manifest
    )


@pytest.mark.parametrize("version", ["unknown", "", None, 2])
def test_unknown_manifest_versions_fail_closed(historical_manifest, version):
    invalid = copy.deepcopy(historical_manifest)
    invalid["manifest_version"] = version
    with pytest.raises(r8.SourceReviewMigrationError, match="Unsupported"):
        r8.validate_versioned_manifest(invalid)


def test_v2_root_exact_schema_rejects_unknown_field(migrated_manifest):
    invalid = copy.deepcopy(migrated_manifest)
    invalid["unexpected"] = True
    with pytest.raises(r8.SourceReviewMigrationError, match="root fields"):
        r8.validate_versioned_manifest(invalid)


def test_v2_review_schema_binding_is_exact(migrated_manifest):
    invalid = copy.deepcopy(migrated_manifest)
    invalid["review_schema"]["source_manifest_sha256"] = "0" * 64
    with pytest.raises(r8.SourceReviewMigrationError, match="schema binding"):
        r8.validate_versioned_manifest(invalid)


def test_valid_evidence_only_record_has_explicit_policy_and_no_authority(
    historical_manifest,
):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    assert record["policy_binding"] == r8.POLICY_BINDING
    assert record["review_state"] == "EVIDENCE_ONLY_NOT_AUTHORITATIVE"
    assert record["authority_attestation"] is None
    assert record["eligibility_authorization"]["authorized"] is False


def test_review_record_can_be_added_only_to_the_in_memory_v2_copy(
    historical_manifest, migrated_manifest
):
    source = _reviewable_source(historical_manifest)
    candidate = copy.deepcopy(migrated_manifest)
    candidate["review_records"] = [_record(source)]
    r8.validate_versioned_manifest(candidate)
    assert migrated_manifest["review_records"] == []
    assert json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) == historical_manifest


def test_unknown_policy_binding_is_rejected(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    record["policy_binding"]["policy_fingerprint"] = "0" * 64
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="policy binding"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


def test_missing_evidence_binding_is_rejected(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    del record["evidence_binding"]["candidate_set_fingerprint"]
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="evidence binding"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


def test_malformed_pair_bindings_are_rejected(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    record["evidence_binding"]["pair_binding_fingerprints"] = ["4" * 64, "4" * 64]
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="sorted and unique"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


def test_forged_review_authority_string_is_rejected(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    record["authority_attestation"] = "AUTHORIZED_VERSIONED_REVIEW_PRODUCER"
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="Caller-controlled"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


def test_authoritative_state_cannot_be_self_asserted(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    record["review_state"] = "AUTHORITATIVE_REVIEW_COMPLETE"
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="cannot issue"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


def test_stale_source_evidence_is_rejected(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    stale = copy.deepcopy(source)
    stale["structured_sha256"] = "0" * 64
    with pytest.raises(r8.SourceReviewMigrationError, match="Stale or mismatched"):
        r8.validate_review_record_v2(record, stale, r8.SOURCE_MANIFEST_SHA256)


def test_existing_rejection_reasons_cannot_be_silently_cleared(historical_manifest):
    source = copy.deepcopy(_reviewable_source(historical_manifest))
    source["rejection_reasons"] = ["EXISTING_BLOCKER"]
    record = _record(source)
    record["findings"]["blocking_rejection_reasons"] = []
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="cannot be cleared"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


def test_review_record_cannot_authorize_promotion(historical_manifest):
    source = _reviewable_source(historical_manifest)
    record = _record(source)
    record["eligibility_authorization"]["authorized"] = True
    record["record_fingerprint"] = policy_v2.canonical_fingerprint(
        record, excluded_key="record_fingerprint"
    )
    with pytest.raises(r8.SourceReviewMigrationError, match="cannot grant"):
        r8.validate_review_record_v2(record, source, r8.SOURCE_MANIFEST_SHA256)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("TÒA ÁN NHÂN DÂN\nBản án số: 01/2026/DS-ST", "BAN_AN"),
        ("TÒA ÁN NHÂN DÂN\nQUYÉT ĐỊNH ĐÌNH CHỈ GIẢI QUYẾT", "QUYET_DINH"),
        ("TÒA ÁN NHÂN DÂN\nQuyết định giám đốc thẩm Số: 1", "QUYET_DINH"),
        ("TÒA ÁN NHÂN DÂN\nkhông có tiêu đề loại văn bản", "UNKNOWN"),
    ],
)
def test_document_type_classifier_uses_explicit_header_evidence(header, expected):
    result = r8.classify_document_type_from_header(header)
    assert result["document_type"] == expected
    assert result["evidence_supported"] is (expected != "UNKNOWN")
    if expected != "UNKNOWN":
        assert result["exact_marker"]
        assert result["page"] == 1


def test_conflicting_header_markers_remain_unknown():
    result = r8.classify_document_type_from_header(
        "Bản án số: 01/2026/DS-ST QUYẾT ĐỊNH ĐÌNH CHỈ giả định"
    )
    assert result["document_type"] == "UNKNOWN"
    assert result["conflicting_evidence"]


def test_real_document_type_audit_covers_batch_and_counterparts(outputs):
    audit = outputs["document_type_evidence_audit.json"]
    assert audit["batch2dc_count"] == 46
    assert audit["comparison_counterpart_count"] == 19
    assert audit["candidate_pair_count"] == 26
    assert len(audit["sources"]) == 65
    assert all(item["filename_or_number_used_as_authority"] is False for item in audit["sources"])
    assert all(item["structured_sha256"] for item in audit["sources"])


def test_document_type_is_not_confused_with_stage_or_domain(outputs):
    audit = outputs["document_type_evidence_audit.json"]
    assert all(item["procedure_stage_separate_verified_field"] for item in audit["sources"])
    assert all(item["case_domain"] is None for item in audit["sources"])
    assert all(item["case_domain_status"].startswith("NOT_AUDITED") for item in audit["sources"])


def test_frozen_policy_does_not_require_document_type_universally(outputs):
    semantics = outputs["document_type_evidence_audit.json"]["frozen_policy_semantics"]
    assert semantics["document_type_is_contextual_field"] is True
    assert semantics["minimum_verified_contextual_conflicts"] == 1
    assert semantics["every_contextual_field_required_for_every_relation"] is False
    assert semantics["future_versioned_contract_amendment_required"] is True
    assert semantics["frozen_policy_modified"] is False


def test_targeted_027_029_adjudications_are_bound_to_exact_evidence(outputs):
    report = outputs["targeted_adjudication_bindings.json"]
    assert len(report["bindings"]) == 3
    assert all(item["pair_binding_fingerprint"] for item in report["bindings"])
    assert len({item["pair_binding_fingerprint"] for item in report["bindings"]}) == 3
    assert all(
        item["reuse_guard"].startswith("EXACT_PAIR_AND_RAW_STRUCTURED_HASHES")
        for item in report["bindings"]
    )
    assert report["analytical_adjudication_became_authoritative_review"] is False


def test_five_unresolved_disputes_remain_blocked(outputs):
    report = outputs["targeted_adjudication_bindings.json"]
    assert report["unresolved_disputes_retained"] == [
        "batch2dc-20261105-008",
        "batch2dc-20261105-014",
        "batch2dc-20261105-022",
        "batch2dc-20261105-024",
        "batch2dc-20261105-034",
    ]


def test_canonical_025_026_and_cross_version_checks_remain_mandatory(outputs):
    design = outputs["eligibility_gate_migration_design.json"]
    assert design["canonical_duplicate_controls"] == {
        "retained": "independent:batch2dc-20261105-025",
        "noncanonical_rejected": "independent:batch2dc-20261105-026",
    }
    assert "check development overlap and all V1/V2 independent sources" in design[
        "future_promotion_order"
    ]
    assert design["promotion_executed"] is False


def test_caller_string_is_explicitly_not_a_trust_boundary(outputs):
    trust = outputs["review_authority_trust_boundary.json"]
    assert trust["current_r7_string_is_authority"] is False
    assert trust["caller_controlled_authority_field_accepted"] is False
    assert trust["trusted_context_activated"] is False
    assert "complete current cross-version candidate recomputation" in trust[
        "trusted_producer_requirements"
    ]


def test_schema_design_selects_root_collection_not_inline_or_sidecar(outputs):
    matrix = outputs["schema_compatibility_matrix.json"]
    assert matrix["selected_design"] == "B_SEPARATE_VERSIONED_REVIEW_RECORD_COLLECTION"
    assert matrix["options"]["B_ROOT_REVIEW_COLLECTION"]["selected"] is True
    assert matrix["options"]["A_INLINE_SOURCE_FIELDS"]["selected"] is False
    assert matrix["options"]["C_SIDECAR_LEDGER"]["selected"] is False
    assert matrix["production_v1_functions_modified"] is False


def test_migration_dry_run_has_zero_live_or_semantic_changes(outputs):
    dry_run = outputs["migration_dry_run.json"]
    assert dry_run["result"] == "PASS"
    assert dry_run["v1_projection_byte_equal"] is True
    assert dry_run["v1_projection_semantic_equal"] is True
    assert dry_run["identity_change_count"] == 0
    assert dry_run["status_change_count"] == 0
    assert dry_run["history_change_count"] == 0
    assert dry_run["v1_review_semantics_change_count"] == 0
    assert dry_run["eligibility_change_count"] == 0
    assert dry_run["live_manifest_mutated"] is False


def test_protected_state_and_benchmark_remain_unchanged(outputs):
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
    assert summary["benchmark_or_sampling_mutated"] is False


def test_all_r8_outputs_have_deterministic_fingerprints(outputs):
    assert set(outputs) == {
        "schema_compatibility_matrix.json",
        "versioned_manifest_schema_spec.json",
        "migration_dry_run.json",
        "document_type_evidence_audit.json",
        "review_authority_trust_boundary.json",
        "targeted_adjudication_bindings.json",
        "eligibility_gate_migration_design.json",
        "validation_summary.json",
    }
    for payload in outputs.values():
        assert payload["artifact_fingerprint"] == policy_v2.canonical_fingerprint(
            payload, excluded_key="artifact_fingerprint"
        )


def test_no_live_manifest_or_benchmark_mutation_after_all_checks():
    assert policy_v2.sha256_file(MANIFEST_PATH) == r8.SOURCE_MANIFEST_SHA256
    benchmark = json.loads(
        (
            PROJECT_ROOT
            / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
        ).read_text(encoding="utf-8")
    )
    sampling = json.loads(
        (
            PROJECT_ROOT
            / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert benchmark["state"] == "DRAFT"
    assert benchmark["current_counts"]["selected_cases"] == 0
    assert benchmark["current_counts"]["query_records"] == 0
    assert benchmark["current_counts"]["ground_truth_records"] == 0
    assert sampling["state"] == "DRAFT"
    assert sampling["selection_status"] == "NOT_STARTED"
    assert sampling["sampling_seed"] is None
