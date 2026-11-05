import ast
import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import topup_acquisition_cohort as cohort_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COHORT_PATH = PROJECT_ROOT / cohort_module.DEFAULT_COHORT_RELATIVE
INTAKE_PATH = PROJECT_ROOT / cohort_module.DEFAULT_INTAKE_RELATIVE
DEVELOPMENT_PATH = PROJECT_ROOT / cohort_module.DEFAULT_DEVELOPMENT_DEDUP_RELATIVE
PLAN_PATH = PROJECT_ROOT / cohort_module.DEFAULT_PLAN_RELATIVE


@pytest.fixture(scope="module")
def cohort():
    return cohort_module.strict_json_load(COHORT_PATH)


@pytest.fixture(scope="module")
def intake():
    return cohort_module.strict_json_load(INTAKE_PATH)


@pytest.fixture(scope="module")
def development():
    return cohort_module.strict_json_load(DEVELOPMENT_PATH)


@pytest.fixture(scope="module")
def validated(cohort, intake, development):
    return cohort_module.validate_cohort(cohort, intake, development)


def test_cohort_identity_status_and_size_are_frozen(cohort):
    assert cohort["cohort_version"] == "independent_batch2dc_topup_cohort_v1"
    assert cohort["cohort_id"] == "batch2dc-20261105"
    assert cohort["status"] == "COHORT_DECLARED_NOT_ACQUIRED"
    assert cohort["target_raw_count"] == 50
    assert cohort["outcome_adaptive_expansion_forbidden"] is True


def test_parent_topup_plan_binding_is_exact(cohort):
    assert cohort["parent_topup_plan_commit"] == cohort_module.PARENT_PLAN_COMMIT
    assert cohort["parent_topup_plan_sha256"] == cohort_module.PARENT_PLAN_SHA256
    assert cohort["parent_topup_plan_fingerprint"] == cohort_module.PARENT_PLAN_FINGERPRINT
    assert cohort_module.sha256_file(PLAN_PATH) == cohort_module.PARENT_PLAN_SHA256
    assert cohort_module.canonical_json_fingerprint(
        cohort_module.strict_yaml_load(PLAN_PATH)
    ) == cohort_module.PARENT_PLAN_FINGERPRINT


def test_r12_listing_pool_binding_is_exact(cohort):
    parent = cohort["parent_listing_pool"]
    assert parent["run_id"] == cohort_module.PARENT_POOL_RUN_ID
    assert parent["listing_pool_fingerprint"] == cohort_module.PARENT_POOL_FINGERPRINT
    assert parent["unique_pool_fingerprint"] == cohort_module.PARENT_UNIQUE_POOL_FINGERPRINT
    assert parent["position_count"] == 160
    assert parent["unique_canonical_listing_url_count"] == 160
    assert parent["duplicate_url_group_count"] == 0


def test_r2_detail_validation_binding_and_pdf_firewall_are_exact(cohort):
    detail = cohort["detail_validation"]
    assert detail == {
        "run_id": cohort_module.DETAIL_RUN_ID,
        "protocol_fingerprint": cohort_module.DETAIL_PROTOCOL_FINGERPRINT,
        "detail_fetch_ledger_fingerprint": cohort_module.DETAIL_LEDGER_FINGERPRINT,
        "all_160_outcome_fingerprint": cohort_module.OUTCOMES_FINGERPRINT,
        "accepted_pool_fingerprint": cohort_module.ACCEPTED_POOL_FINGERPRINT,
        "selection_proof_fingerprint": cohort_module.SELECTION_PROOF_FINGERPRINT,
        "detail_requests_attempted": 160,
        "detail_pages_completed": 160,
        "pdf_requests": 0,
    }


def test_exact_50_batch2dc_ids_and_stable_source_ids(cohort):
    expected = [f"batch2dc-20261105-{index:03d}" for index in range(1, 51)]
    assert [entry["acquisition_id"] for entry in cohort["entries"]] == expected
    assert all(
        cohort_module.stable_source_id(identifier) == f"independent:{identifier}"
        for identifier in expected
    )


def test_official_domain_and_url_shapes_only(cohort, validated):
    assert len(validated["source_urls"]) == 50
    assert all(url.startswith("https://congbobanan.toaan.gov.vn/5ta") for url in validated["source_urls"])
    assert all(url.lower().endswith(".pdf") for url in validated["source_urls"])
    assert all(url.startswith("https://congbobanan.toaan.gov.vn/2ta") for url in validated["listing_urls"])
    assert all(" " not in url for url in validated["source_urls"] + validated["listing_urls"])


def test_source_and_listing_urls_are_unique(validated):
    assert len(set(validated["source_urls"])) == 50
    assert len(set(validated["listing_urls"])) == 50


def test_all_selected_listing_urls_belong_to_frozen_160(cohort):
    proof = cohort["full_pool_selection_proof"]
    frozen = proof["frozen_listing_urls"]
    assert len(frozen) == len(set(frozen)) == 160
    assert frozen == sorted(frozen)
    assert cohort_module.canonical_json_fingerprint(frozen) == cohort_module.PARENT_UNIQUE_POOL_FINGERPRINT
    assert set(proof["selected_listing_urls"]).issubset(frozen)


def test_deterministic_first_50_selection_is_self_contained(cohort):
    proof = cohort["full_pool_selection_proof"]
    accepted = proof["accepted_sorted_listing_urls"]
    selected = proof["selected_listing_urls"]
    assert len(accepted) == 140
    assert accepted == sorted(accepted)
    assert selected == accepted[:50]
    assert proof["candidate_51_listing_url"] == accepted[50]
    assert proof["selected_is_exact_accepted_prefix"] is True
    assert proof["earlier_accepted_url_skipped"] is False
    assert [entry["official_listing_url"] for entry in cohort["entries"]] == selected


def test_full_pool_completion_and_rejection_summary_are_frozen(cohort):
    proof = cohort["full_pool_selection_proof"]
    assert (proof["candidate_validation_complete"], proof["accepted_count"], proof["rejected_count"]) == (
        "160/160", 140, 20
    )
    assert cohort["discovery_summary"] == {
        "frozen_listing_candidates": 160,
        "detail_pages_inspected": 160,
        "detail_fetch_failures": 0,
        "metadata_qualified_candidates": 160,
        "missing_pdf_link_count": 0,
        "non_land_intent_metadata_rejections": 0,
        "existing_intake_url_duplicates": 19,
        "exact_intake_identity_duplicates": 0,
        "exact_development_duplicates": 0,
        "within_pool_pdf_duplicates": 0,
        "within_pool_exact_identity_duplicates": 1,
        "accepted_total": 140,
        "rejected_total": 20,
        "selected_total": 50,
        "rejection_reason_counts": {
            "DUPLICATE_EXACT_IDENTITY_WITHIN_FROZEN_POOL": 1,
            "EXISTING_INTAKE_SOURCE_URL": 19,
        },
    }


def test_only_land_targeted_discovery_priority_is_present(cohort, validated):
    assert cohort["discovery_priority_order"] == ["LAND_2024_TARGETED_DISCOVERY"]
    assert validated["priority_counts"] == {"LAND_2024_TARGETED_DISCOVERY": 50}
    assert all("land-related legal relation" in entry["discovery_basis"] for entry in cohort["entries"])
    serialized = json.dumps(cohort, ensure_ascii=False)
    assert "confirmed_legal_family" not in serialized
    assert "raw_family_quota" not in serialized


def test_pre_download_duplicate_checks_preserve_metadata_limitations(cohort):
    for entry in cohort["entries"]:
        precheck = entry["pre_download_duplicate_check"]
        assert precheck["existing_intake_url_overlap"] is False
        assert precheck["new_cohort_url_overlap"] is False
        assert precheck["result"] == "NO_OBVIOUS_DUPLICATE_IN_AVAILABLE_METADATA"
        expected = None if precheck["case_identity_comparison_limited_by_metadata"] else False
        assert precheck["exact_known_intake_case_overlap"] is expected
        assert precheck["exact_development_case_overlap"] is expected


def test_existing_intake_url_overlap_is_rejected(cohort, intake, development):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["source_url"] = cohort_module.normalize_existing_url(intake["sources"][0]["source_url"])
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="Existing intake"):
        cohort_module.validate_cohort(changed, intake, development)


def test_within_cohort_source_url_duplicate_is_rejected(cohort, intake, development):
    changed = copy.deepcopy(cohort)
    changed["entries"][1]["source_url"] = changed["entries"][0]["source_url"]
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="Duplicate source_url"):
        cohort_module.validate_cohort(changed, intake, development)


def test_within_cohort_listing_duplicate_is_rejected(cohort, intake, development):
    changed = copy.deepcopy(cohort)
    changed["entries"][1]["official_listing_url"] = changed["entries"][0]["official_listing_url"]
    changed["entries"][1]["discovery_metadata_source"] = changed["entries"][0]["official_listing_url"]
    with pytest.raises(cohort_module.TopupAcquisitionCohortError):
        cohort_module.validate_cohort(changed, intake, development)


def test_exact_known_intake_identity_overlap_is_rejected(cohort, intake, development):
    changed = copy.deepcopy(cohort)
    entry = changed["entries"][0]
    entry["official_case_or_judgment_id"] = intake["sources"][0]["judgment_number"]
    precheck = entry["pre_download_duplicate_check"]
    precheck["case_identity_comparison_limited_by_metadata"] = False
    precheck["exact_known_intake_case_overlap"] = False
    precheck["exact_development_case_overlap"] = False
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="Exact known intake"):
        cohort_module.validate_cohort(changed, intake, development)


def test_exact_development_identity_overlap_is_rejected(cohort, intake, development):
    existing = {
        cohort_module._identity(value)
        for source in intake["sources"]
        for value in (source.get("case_id"), source.get("judgment_number"))
        if value
    }
    development_id = next(
        item["case_id"] for item in development["development_source_mapping"]
        if cohort_module._identity(item["case_id"]) not in existing
    )
    changed = copy.deepcopy(cohort)
    entry = changed["entries"][0]
    entry["official_case_or_judgment_id"] = development_id
    precheck = entry["pre_download_duplicate_check"]
    precheck["case_identity_comparison_limited_by_metadata"] = False
    precheck["exact_known_intake_case_overlap"] = False
    precheck["exact_development_case_overlap"] = False
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="Exact development"):
        cohort_module.validate_cohort(changed, intake, development)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confirmed_legal_family", "LAND_2024"),
        ("family_membership", ["LAND_2024"]),
        ("ground_truth", ["law"]),
        ("query", "hidden query"),
        ("retrieval_result", []),
        ("retrieval_score", 1.0),
        ("retrieval_rank", 1),
        ("model_output", "land"),
        ("annotation", "confirmed"),
        ("sampling_seed", 42),
        ("selected_benchmark_ids", [1]),
    ],
)
def test_forbidden_scientific_fields_are_rejected(cohort, intake, development, field, value):
    changed = copy.deepcopy(cohort)
    changed["entries"][0][field] = value
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="Forbidden"):
        cohort_module.validate_cohort(changed, intake, development)


def test_unknown_priority_is_rejected(cohort, intake, development):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["discovery_priority"] = "CONFIRMED_LAND_2024"
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="priority"):
        cohort_module.validate_cohort(changed, intake, development)


def test_fingerprint_is_deterministic_and_excludes_declaration_timestamp(cohort, intake, development):
    assert cohort_module.cohort_fingerprint(cohort) == cohort["cohort_fingerprint"]
    changed = copy.deepcopy(cohort)
    changed["declared_at"] = "2026-11-05T23:59:59Z"
    assert cohort_module.cohort_fingerprint(changed) == cohort["cohort_fingerprint"]
    cohort_module.validate_cohort(changed, intake, development)


def test_bound_metadata_tampering_fails_fingerprint(cohort, intake, development):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["court"] = "Tòa án nhân dân tampered"
    with pytest.raises(cohort_module.TopupAcquisitionCohortError, match="fingerprint"):
        cohort_module.validate_cohort(changed, intake, development)


def test_protected_scientific_state_and_no_artifacts_are_unchanged(cohort):
    _, result = cohort_module.load_and_validate_cohort(PROJECT_ROOT, COHORT_PATH)
    state = result["protected_state"]
    assert state == {
        "benchmark": "DRAFT",
        "sampling": "DRAFT",
        "selection": "NOT_STARTED",
        "sampling_seed": None,
        "selected_cases": 0,
        "query_records": 0,
        "ground_truth_records": 0,
        "batch2dc_acquisition_artifacts": 0,
    }


def test_manifest_is_byte_identical_and_intake_counts_are_frozen(validated):
    assert cohort_module.sha256_file(INTAKE_PATH) == cohort_module.ACQUISITION_MANIFEST_SHA256
    assert validated["intake_counts"] == {
        "total": 317, "eligible": 257, "extracted": 59, "hash_verified": 1
    }


def test_validator_has_no_network_pdf_or_model_dependencies():
    source = Path(cohort_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({
        "requests", "httpx", "urllib3", "selenium", "playwright", "pdfplumber",
        "pypdf", "fitz", "torch", "transformers", "sentence_transformers",
    })


def test_canonical_audit_proves_zero_pdf_and_zero_registration(cohort):
    audit = cohort_module.build_audit(PROJECT_ROOT, COHORT_PATH)
    assert audit["pdf_requests"] == 0
    assert audit["pdf_body_fetched"] is False
    assert audit["sources_registered"] == 0
    assert audit["selection_benchmark_performed"] is False
    assert len(audit["selected_50_ids"]) == 50
    assert len(audit["selected_50_listing_urls"]) == 50
    assert len(audit["selected_50_pdf_urls"]) == 50
    assert audit["cohort_fingerprint"] == cohort["cohort_fingerprint"]
