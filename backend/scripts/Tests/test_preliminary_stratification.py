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
    assemble_preliminary_stratification,
    build_preliminary_stratification,
    classify_preliminary_records,
    classify_preliminary_source,
    validate_checkpoint_frame_against_live,
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


def _eligible_source(index, text="Căn cứ Điều 463 Bộ luật Dân sự năm 2015."):
    case_id = f"{index:02d}/2026/DS-ST"
    source_id = f"independent:test-{index:03d}"
    return (
        {
            "acquisition_id": f"test-{index:03d}",
            "source_id": source_id,
            "case_id": case_id,
            "judgment_number": case_id,
            "decision_date": "2026-01-02",
            "procedure_stage": "so_tham",
            "acquisition_status": "eligible",
        },
        _structured(text),
    )


def _frame_record(source, index):
    return {
        "source_id": source["source_id"],
        "acquisition_id": source["acquisition_id"],
        "case_id": source["case_id"],
        "judgment_number": source["judgment_number"],
        "underlying_dispute_id": None,
        "related_decision_ids": [],
        "source_name": f"test-{index:03d}.pdf",
        "source_path": f"raw/test-{index:03d}.pdf",
        "structured_path": f"structured/test-{index:03d}.json",
        "court": "TAND test",
        "procedure_stage": source["procedure_stage"],
        "year": int(source["decision_date"][:4]),
        "case_type": None,
        "preliminary_law_families": [],
        "preliminary_legal_functions": ["unknown"],
        "source_sha256": f"{index:064x}",
        "structured_sha256": f"{index + 100:064x}",
        "blank_page_numbers": [],
        "metadata_source": "verified_independent_acquisition_manifest_v1",
    }


def _explicit_fixture(count):
    sources = []
    structured = {}
    frame = []
    for index in range(1, count + 1):
        source, payload = _eligible_source(index)
        sources.append(source)
        structured[source["source_id"]] = payload
        frame.append(_frame_record(source, index))
    frame.sort(key=lambda item: (item["case_id"], item["source_id"]))
    return sources, structured, frame


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
    sampling = json.loads(
        (
            PROJECT_ROOT
            / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert first == second
    assert first["source_count"] == len(sampling["eligible_sources"])
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


def test_active_growth_and_explicit_reconciliation_are_both_valid():
    sources, structured, live_frame = _explicit_fixture(3)
    checkpoint_frame = live_frame[:2]

    validate_checkpoint_frame_against_live(checkpoint_frame, live_frame)
    checkpoint_records = classify_preliminary_records(sources[:2], {
        source["source_id"]: structured[source["source_id"]]
        for source in sources[:2]
    })
    checkpoint_artifact = assemble_preliminary_stratification(
        checkpoint_frame,
        checkpoint_records,
    )
    validate_preliminary_stratification(checkpoint_artifact, checkpoint_frame)

    reconciled_records = classify_preliminary_records(sources, structured)
    reconciled_artifact = assemble_preliminary_stratification(
        live_frame,
        reconciled_records,
    )
    validate_checkpoint_frame_against_live(live_frame, live_frame)
    validate_preliminary_stratification(reconciled_artifact, live_frame)
    assert checkpoint_artifact["source_count"] == 2
    assert reconciled_artifact["source_count"] == 3


def test_new_live_sources_can_be_classified_without_mutating_checkpoint():
    sources, structured, live_frame = _explicit_fixture(3)
    checkpoint_frame = copy.deepcopy(live_frame[:2])
    checkpoint_records = classify_preliminary_records(sources[:2], {
        source["source_id"]: structured[source["source_id"]]
        for source in sources[:2]
    })
    checkpoint_artifact = assemble_preliminary_stratification(
        checkpoint_frame,
        checkpoint_records,
    )
    frozen_frame = copy.deepcopy(checkpoint_frame)
    frozen_artifact = copy.deepcopy(checkpoint_artifact)

    new_records = classify_preliminary_records(
        sources[2:],
        {sources[2]["source_id"]: structured[sources[2]["source_id"]]},
    )

    assert len(new_records) == 1
    assert new_records[0]["source_id"] not in {
        record["source_id"] for record in checkpoint_frame
    }
    assert checkpoint_frame == frozen_frame
    assert checkpoint_artifact == frozen_artifact


@pytest.mark.parametrize("relationship", ["artifact_extra", "artifact_missing"])
def test_stratification_and_checkpoint_source_ids_must_match(relationship):
    sources, structured, live_frame = _explicit_fixture(3)
    records = classify_preliminary_records(sources, structured)
    artifact = assemble_preliminary_stratification(live_frame, records)
    wrong_frame = live_frame[:2] if relationship == "artifact_extra" else live_frame + [
        _frame_record(_eligible_source(4)[0], 4)
    ]
    with pytest.raises(
        PreliminaryStratificationError,
        match="inventory fingerprint|source IDs",
    ):
        validate_preliminary_stratification(artifact, wrong_frame)


def test_duplicate_stratification_source_id_is_rejected():
    sources, structured, frame = _explicit_fixture(2)
    artifact = assemble_preliminary_stratification(
        frame,
        classify_preliminary_records(sources, structured),
    )
    artifact["records"].append(copy.deepcopy(artifact["records"][0]))
    artifact["records"].sort(key=lambda item: (item["case_id"], item["source_id"]))
    artifact["source_count"] += 1
    with pytest.raises(PreliminaryStratificationError, match="Duplicate"):
        validate_preliminary_stratification(artifact)


def test_checkpoint_source_missing_or_no_longer_eligible_is_rejected():
    _, _, frame = _explicit_fixture(2)
    with pytest.raises(PreliminaryStratificationError, match="no longer live eligible"):
        validate_checkpoint_frame_against_live(frame, frame[1:])


def test_noneligible_explicit_source_is_rejected():
    sources, structured, _ = _explicit_fixture(1)
    sources[0]["acquisition_status"] = "extracted"
    with pytest.raises(PreliminaryStratificationError, match="eligible sources"):
        classify_preliminary_records(sources, structured)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_count", 999, "source count"),
        ("source_inventory_fingerprint", "0" * 64, "inventory fingerprint"),
        ("records_fingerprint", "0" * 64, "fingerprint mismatch"),
        (
            "classification_status_distribution",
            {"CLASSIFIED": 0, "MULTI_FAMILY": 0, "UNKNOWN": 0},
            "classification summary",
        ),
        ("classification_coverage", 0.0, "coverage mismatch"),
        (
            "family_distribution",
            {family: 0 for family in (
                "BLDS_CORE",
                "BLTTDS_PRIMARY",
                "LAND_2013",
                "LAND_2024",
                "HNGD_2014",
                "MULTI_FAMILY",
                "OTHER",
                "UNKNOWN",
            )},
            "family summary",
        ),
        ("quota_observability", {}, "quota summary"),
    ],
)
def test_artifact_self_consistency_tampering_is_rejected(field, value, message):
    sources, structured, frame = _explicit_fixture(2)
    artifact = assemble_preliminary_stratification(
        frame,
        classify_preliminary_records(sources, structured),
    )
    artifact[field] = value
    with pytest.raises(PreliminaryStratificationError, match=message):
        validate_preliminary_stratification(artifact, frame)


def test_checkpoint_source_mutation_is_rejected():
    _, _, frame = _explicit_fixture(2)
    live = copy.deepcopy(frame)
    live[0]["structured_sha256"] = "f" * 64
    with pytest.raises(PreliminaryStratificationError, match="changed"):
        validate_checkpoint_frame_against_live(frame, live)


def test_development_source_contamination_is_rejected():
    _, _, frame = _explicit_fixture(1)
    frame[0]["source_id"] = "judgment:development.pdf"
    with pytest.raises(PreliminaryStratificationError, match="Development source"):
        validate_checkpoint_frame_against_live(frame, frame)
