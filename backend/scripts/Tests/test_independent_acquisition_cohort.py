import ast
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import acquisition_cohort as cohort_module


PROJECT_ROOT = Path(__file__).resolve().parents[3]
COHORT_PATH = PROJECT_ROOT / cohort_module.DEFAULT_COHORT_RELATIVE
INTAKE_PATH = PROJECT_ROOT / cohort_module.DEFAULT_INTAKE_RELATIVE
PLAN_PATH = PROJECT_ROOT / cohort_module.PARENT_PLAN_RELATIVE


@pytest.fixture(scope="module")
def cohort():
    return cohort_module.strict_json_load(COHORT_PATH)


@pytest.fixture(scope="module")
def intake():
    return cohort_module.strict_json_load(INTAKE_PATH)


@pytest.fixture(scope="module")
def validated(cohort, intake):
    return cohort_module.validate_cohort(cohort, intake)


def test_cohort_identity_and_status_are_frozen(cohort):
    assert cohort["cohort_version"] == "independent_batch2db_cohort_v1"
    assert cohort["cohort_id"] == "batch2db-20261103"
    assert cohort["status"] == "COHORT_DECLARED_NOT_ACQUIRED"
    assert cohort["target_raw_count"] == 50
    assert cohort["outcome_adaptive_expansion_forbidden"] is True


def test_parent_plan_identity_is_exact(cohort):
    assert cohort["parent_acquisition_plan_commit"] == (
        "a00a08b727db2ee55f347f9e09b8893ccb972e00"
    )
    assert cohort["parent_acquisition_plan_sha256"] == (
        "9379d96f99e3291e38d2bcda0fbe1a9566e2b781d3093b14e5fa91ad9da0658e"
    )
    assert cohort["plan_authoring_base_commit"] == (
        "a5ebb4a03f35a4c33596faf93b8d5fa82d6d076f"
    )
    assert cohort["declaration_execution_commit"] == (
        "a00a08b727db2ee55f347f9e09b8893ccb972e00"
    )
    assert cohort_module.sha256_file(PLAN_PATH) == (
        cohort["parent_acquisition_plan_sha256"]
    )


def test_parent_plan_frozen_numerical_contract():
    from backend.scripts.Evaluation.benchmarking import acquisition_planning

    audit = acquisition_planning.build_acquisition_plan_audit(PROJECT_ROOT, PLAN_PATH)
    plan = audit["acquisition_plan.json"]
    assert (
        plan["raw_count_hard_floor"],
        plan["final_recommended_raw_batch_target"],
        plan["conservative_expected_eligible_additions"],
        plan["projected_final_eligible"],
        plan["projected_selection_reserve"],
    ) == (37, 50, 37, 260, 10)
    assert plan["targeting"]["family_specific_raw_yield_calibration_status"] == (
        "FAMILY_SPECIFIC_RAW_YIELD_UNCALIBRATED"
    )
    assert plan["targeting"]["raw_family_quota"] is None


def test_exact_cohort_size_and_ids(cohort):
    entries = cohort["entries"]
    assert len(entries) == 50
    assert [entry["acquisition_id"] for entry in entries] == [
        f"batch2db-20261103-{index:03d}" for index in range(1, 51)
    ]
    assert all(
        cohort_module.stable_source_id(entry["acquisition_id"])
        == f"independent:{entry['acquisition_id']}"
        for entry in entries
    )


def test_official_domain_only(cohort, validated):
    assert all(
        entry["source_domain"] == "congbobanan.toaan.gov.vn"
        for entry in cohort["entries"]
    )
    assert len(validated["source_urls"]) == 50
    assert all(url.startswith("https://congbobanan.toaan.gov.vn/5ta") for url in validated["source_urls"])
    assert all(" " not in url for url in validated["source_urls"])


def test_unique_source_and_listing_urls(validated):
    assert len(set(validated["source_urls"])) == 50
    assert len(set(validated["listing_urls"])) == 50


def test_duplicate_source_url_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][1]["source_url"] = changed["entries"][0]["source_url"]
    with pytest.raises(cohort_module.AcquisitionCohortError, match="Duplicate source_url"):
        cohort_module.validate_cohort(changed, intake)


def test_duplicate_listing_url_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][1]["official_listing_url"] = changed["entries"][0][
        "official_listing_url"
    ]
    changed["entries"][1]["discovery_metadata_source"] = changed["entries"][0][
        "official_listing_url"
    ]
    with pytest.raises(cohort_module.AcquisitionCohortError, match="listing_url"):
        cohort_module.validate_cohort(changed, intake)


def test_duplicate_acquisition_id_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][1]["acquisition_id"] = changed["entries"][0][
        "acquisition_id"
    ]
    with pytest.raises(cohort_module.AcquisitionCohortError, match="IDs/order"):
        cohort_module.validate_cohort(changed, intake)


def test_existing_intake_url_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    existing = intake["sources"][0]["source_url"]
    changed["entries"][0]["source_url"] = cohort_module.normalize_existing_url(existing)
    with pytest.raises(cohort_module.AcquisitionCohortError, match="already exists"):
        cohort_module.validate_cohort(changed, intake)


def test_non_official_source_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["source_url"] = "https://example.test/5ta1t1cvn/a.pdf"
    with pytest.raises(cohort_module.AcquisitionCohortError, match="official HTTPS"):
        cohort_module.validate_cohort(changed, intake)


def test_unescaped_or_noncanonical_url_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["source_url"] = changed["entries"][0][
        "source_url"
    ].replace("%20", " ", 1)
    with pytest.raises(cohort_module.AcquisitionCohortError, match="whitespace"):
        cohort_module.validate_cohort(changed, intake)


def test_forbidden_legal_confirmation_field_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["confirmed_legal_family"] = "LAND_2024"
    with pytest.raises(cohort_module.AcquisitionCohortError, match="Forbidden"):
        cohort_module.validate_cohort(changed, intake)


@pytest.mark.parametrize(
    "field,value",
    [
        ("retrieval_score", 1.0),
        ("retrieval_rank", 1),
        ("model_output", "land"),
        ("query", "hidden query"),
        ("ground_truth", ["LAND_2024"]),
        ("annotation", "confirmed"),
    ],
)
def test_retrieval_model_query_gt_annotation_fields_are_rejected(
    cohort, intake, field, value
):
    changed = copy.deepcopy(cohort)
    changed["entries"][0][field] = value
    with pytest.raises(cohort_module.AcquisitionCohortError, match="Forbidden"):
        cohort_module.validate_cohort(changed, intake)


def test_discovery_priorities_are_exact_and_descriptive(cohort, validated):
    assert cohort["discovery_priority_order"] == [
        "LAND_2024_TARGETED_DISCOVERY",
        "LAND_HALL_TARGETED_DISCOVERY",
    ]
    assert validated["priority_counts"] == {"LAND_2024_TARGETED_DISCOVERY": 50}
    assert cohort["discovery_summary"][
        "priority_counts_are_descriptive_not_quotas"
    ] is True
    serialized = json.dumps(cohort, ensure_ascii=False)
    assert "raw_family_quota" not in serialized
    assert "confirmed_legal_family" not in serialized


def test_unknown_discovery_priority_is_rejected(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["discovery_priority"] = "CONFIRMED_LAND_2024"
    with pytest.raises(cohort_module.AcquisitionCohortError, match="Unknown"):
        cohort_module.validate_cohort(changed, intake)


def test_ordering_is_deterministic(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0], changed["entries"][1] = (
        changed["entries"][1],
        changed["entries"][0],
    )
    with pytest.raises(cohort_module.AcquisitionCohortError, match="IDs/order"):
        cohort_module.validate_cohort(changed, intake)


def test_fingerprint_is_deterministic_and_excludes_timestamp(cohort, intake):
    assert cohort_module.cohort_fingerprint(cohort) == cohort["cohort_fingerprint"]
    changed = copy.deepcopy(cohort)
    changed["declared_at"] = "2026-11-03T23:59:59Z"
    assert cohort_module.cohort_fingerprint(changed) == cohort["cohort_fingerprint"]
    cohort_module.validate_cohort(changed, intake)


def test_tampered_bound_content_fails_fingerprint(cohort, intake):
    changed = copy.deepcopy(cohort)
    changed["entries"][0]["court"] = "TAND tampered"
    with pytest.raises(cohort_module.AcquisitionCohortError, match="fingerprint"):
        cohort_module.validate_cohort(changed, intake)


def test_discovery_prescreen_summary_is_frozen(cohort):
    summary = cohort["discovery_summary"]
    assert summary["candidate_links_inspected"] == 80
    assert summary["unique_candidate_listing_links"] == 79
    assert summary["candidate_detail_pages_inspected"] == 56
    assert summary["pre_download_duplicates_rejected"] == 6
    assert summary["development_exact_duplicates_rejected"] == 0
    assert summary["rejection_reason_counts"] == {
        "EXISTING_INTAKE_SOURCE_URL": 6
    }


def test_optional_metadata_is_complete(validated):
    assert validated["missing_optional_metadata"] == {}


def test_validator_has_no_network_or_acquisition_dependencies():
    source = Path(cohort_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert imported_modules.isdisjoint(
        {
            "http.client",
            "httpx",
            "requests",
            "socket",
            "urllib.request",
            "aiohttp",
            "pdfplumber",
            "pytesseract",
            "torch",
            "transformers",
        }
    )


def test_build_audit_does_not_mutate_intake_or_parent_plan():
    paths = [INTAKE_PATH, PLAN_PATH, COHORT_PATH]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    audit = cohort_module.build_cohort_audit(PROJECT_ROOT, COHORT_PATH)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before
    assert set(audit) == {
        "cohort_summary.json",
        "duplicate_prescreen.json",
        "discovery_priority_summary.json",
    }


def test_audit_summary_contains_no_acquisition_or_selection(cohort):
    audit = cohort_module.build_cohort_audit(PROJECT_ROOT, COHORT_PATH)
    summary = audit["cohort_summary.json"]
    assert summary["declared_cohort_size"] == 50
    assert summary["unique_source_urls"] == 50
    assert summary["official_domain_count"] == 50
    assert summary["pdf_downloaded_or_stored"] is False
    assert summary["source_registered"] is False
    assert summary["selection_performed"] is False
    assert summary["cohort_fingerprint"] == cohort["cohort_fingerprint"]


def test_atomic_audit_publication(cohort):
    audit = cohort_module.build_cohort_audit(PROJECT_ROOT, COHORT_PATH)
    runs = PROJECT_ROOT / "backend/data/runs"
    with tempfile.TemporaryDirectory(prefix="phase2c-test-", dir=runs) as temp:
        result = cohort_module.publish_cohort_audit(
            PROJECT_ROOT, COHORT_PATH, Path(temp), audit
        )
        assert result.is_dir()
        assert not result.with_name(result.name + ".partial").exists()
        assert {path.name for path in result.iterdir()} == {
            "cohort_summary.json",
            "duplicate_prescreen.json",
            "discovery_priority_summary.json",
            "run_manifest.json",
        }
        manifest = cohort_module.strict_json_load(result / "run_manifest.json")
        assert manifest["parent_acquisition_plan_commit"] == (
            "a00a08b727db2ee55f347f9e09b8893ccb972e00"
        )
        assert manifest["tracked_worktree_clean"] is True
        assert manifest["network_accessed_by_validator"] is False
        assert manifest["pdf_downloaded"] is False
        assert manifest["sources_registered"] == 0
        assert manifest["selection_performed"] is False
        assert manifest["sampling_seed_generated"] is False
        assert manifest["query_or_gt_generated"] is False


def test_existing_intake_has_no_batch2db_ids(intake):
    assert len(intake["sources"]) == 271
    assert not any(
        source["acquisition_id"].startswith("batch2db-20261103-")
        for source in intake["sources"]
    )


def test_no_batch2db_pdf_structured_provenance_or_journal_artifacts():
    intake_root = PROJECT_ROOT / "backend/dataLaw/independent_intake"
    forbidden = []
    for relative in ("raw", "structured", "provenance", "transactions"):
        root = intake_root / relative
        if root.exists():
            forbidden.extend(root.glob("batch2db-20261103-*"))
    assert forbidden == []


def test_protected_benchmark_state_remains_draft():
    benchmark = cohort_module.strict_json_load(
        PROJECT_ROOT
        / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
    )
    sampling = cohort_module.strict_json_load(
        PROJECT_ROOT
        / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
    )
    assert benchmark["state"] == "DRAFT"
    assert sampling["state"] == "DRAFT"
    assert sampling["selection_status"] == "NOT_STARTED"
    assert sampling["sampling_seed"] is None
    assert benchmark["current_counts"]["development_sources_consumed"] == 107
    assert [
        benchmark["current_counts"][key]
        for key in ("selected_cases", "query_records", "ground_truth_records")
    ] == [0, 0, 0]
