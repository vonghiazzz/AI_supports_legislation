import ast
import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking.independent_benchmark import (
    canonical_json_fingerprint,
)
from backend.scripts.Evaluation.benchmarking.preliminary_stratification import (
    DEFAULT_ARTIFACT_RELATIVE,
    PreliminaryStratificationError,
    build_preliminary_stratification,
    classify_preliminary_source,
    validate_preliminary_stratification,
)
from backend.scripts.Evaluation.benchmarking.source_intake import (
    load_acquisition_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/benchmarking/preliminary_stratification.py"
)


def _source(judgment_number="10/2026/DS-ST", procedure_stage="so_tham"):
    return {
        "acquisition_id": "test-001",
        "source_id": "independent:test-001",
        "case_id": judgment_number,
        "judgment_number": judgment_number,
        "decision_date": "2026-01-02",
        "procedure_stage": procedure_stage,
    }


def _structured(*texts):
    return {
        "judgment_name": "test.pdf",
        "page_count": len(texts),
        "pages": [
            {
                "page": index,
                "text": text,
                "extraction_status": "native_text",
            }
            for index, text in enumerate(texts, start=1)
        ],
        "sections": {"header": "", "facts": "", "reasoning": "", "decision": ""},
    }


@pytest.fixture(scope="module")
def generated_artifacts():
    first = build_preliminary_stratification(PROJECT_ROOT)
    second = build_preliminary_stratification(PROJECT_ROOT)
    return first, second


def test_unknown_fails_closed_without_documentary_law_evidence():
    result = classify_preliminary_source(
        _source(),
        _structured("Bản án dân sự nhưng không có tên luật minh thị."),
    )
    assert result["classification_status"] == "UNKNOWN"
    assert result["primary_family_preliminary"] == "UNKNOWN"
    assert result["rule_ids"] == ["PF-UNKNOWN-01"]


def test_blds_and_incidental_blttds_are_not_conflated():
    result = classify_preliminary_source(
        _source("10/2026/DS-ST"),
        _structured(
            "Căn cứ Điều 463 Bộ luật Dân sự năm 2015 và "
            "Điều 147 Bộ luật Tố tụng dân sự."
        ),
    )
    assert result["primary_family_preliminary"] == "BLDS_CORE"
    assert result["classification_status"] == "CLASSIFIED"
    assert "PF-BLTTDS-01" not in result["rule_ids"]


def test_blttds_primary_requires_explicit_law_and_documentary_decision_code():
    result = classify_preliminary_source(
        _source("10/2026/QĐST-DS"),
        _structured("Căn cứ Điều 41 Bộ luật Tố tụng dân sự năm 2015."),
    )
    assert result["primary_family_preliminary"] == "BLTTDS_PRIMARY"
    assert result["rule_ids"] == ["PF-BLTTDS-01"]


def test_land_version_uses_explicit_reference_not_decision_year():
    result = classify_preliminary_source(
        _source("10/2026/DS-ST"),
        _structured("Áp dụng Điều 167 Luật Đất đai năm 2013."),
    )
    assert result["decision_year"] == 2026
    assert result["primary_family_preliminary"] == "LAND_2013"
    assert "LAND_2024" not in result["secondary_families_preliminary"]


def test_land_version_without_documentary_year_fails_closed():
    result = classify_preliminary_source(
        _source("10/2026/DS-ST"),
        _structured("Căn cứ Luật Đất đai và các văn bản hướng dẫn."),
    )
    assert result["primary_family_preliminary"] == "UNKNOWN"
    assert result["classification_status"] == "UNKNOWN"
    assert result["ambiguity_flags"] == ["LAND_VERSION_UNRESOLVED"]


def test_multiple_explicit_family_rules_preserve_multi_family_evidence():
    result = classify_preliminary_source(
        _source("10/2026/HNGĐ-ST"),
        _structured("Áp dụng Điều 167 Luật Đất đai năm 2013."),
    )
    assert result["primary_family_preliminary"] == "MULTI_FAMILY"
    assert result["classification_status"] == "MULTI_FAMILY"
    assert result["secondary_families_preliminary"] == [
        "HNGD_2014",
        "LAND_2013",
    ]


def test_module_has_no_network_model_or_retriever_dependency():
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {
        "requests",
        "httpx",
        "torch",
        "transformers",
        "sentence_transformers",
        "FlagEmbedding",
        "google",
    }


def test_production_backfill_is_exactly_live_eligible_and_reproducible(
    generated_artifacts,
):
    first, second = generated_artifacts
    assert first == second
    assert first["source_count"] == 186
    assert first["records_fingerprint"] == canonical_json_fingerprint(
        first["records"]
    )
    assert first["records"] == sorted(
        first["records"], key=lambda item: (item["case_id"], item["source_id"])
    )
    assert all(
        record["source_id"].startswith("independent:")
        for record in first["records"]
    )


def test_production_backfill_excludes_development_blocked_and_hash_verified(
    generated_artifacts,
):
    artifact, _ = generated_artifacts
    output_ids = {record["source_id"] for record in artifact["records"]}
    manifest = load_acquisition_manifest(PROJECT_ROOT)
    excluded = {
        source["source_id"]
        for source in manifest["sources"]
        if source["acquisition_status"] != "eligible"
    }
    assert not output_ids & excluded
    assert "independent:scale-20261029-178" not in output_ids
    assert not any(source_id.startswith("judgment:") for source_id in output_ids)


def test_checked_in_artifact_matches_deterministic_generation(generated_artifacts):
    generated, _ = generated_artifacts
    checked_in = json.loads(
        (PROJECT_ROOT / DEFAULT_ARTIFACT_RELATIVE).read_text(encoding="utf-8")
    )
    assert checked_in == generated


def test_artifact_fingerprint_tampering_fails_closed(generated_artifacts):
    generated, _ = generated_artifacts
    tampered = copy.deepcopy(generated)
    tampered["records"][0]["decision_year"] += 1
    with pytest.raises(PreliminaryStratificationError, match="fingerprint mismatch"):
        validate_preliminary_stratification(tampered)


def test_multi_family_is_not_counted_as_single_family_quota(generated_artifacts):
    generated, _ = generated_artifacts
    for family in generated["quota_observability"]["families"].values():
        assert family["multi_family_counted_toward_minimum"] is False
