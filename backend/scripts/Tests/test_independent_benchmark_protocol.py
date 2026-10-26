import copy
import hashlib
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking.independent_benchmark import (
    BENCHMARK_ID,
    CORPUS_FINGERPRINT,
    BenchmarkProtocolError,
    canonical_json_fingerprint,
    draft_benchmark_manifest,
    dedup_text_similarity,
    deduplicate_against_development,
    inventory_judgment_sources,
    phase1_payloads,
    selected_records_fingerprint,
    sha256_file,
    transition_manifest,
    validate_benchmark_records,
    validate_manifest_state,
    validate_sampling_records,
    validate_schema_v2,
    validate_selection_guardrails,
    validate_unique_selected_cases,
    verify_sealed_artifacts,
)
from backend.scripts.Evaluation.core.normalization import canonical_retrieval_id


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"


def _sampling_record(source_id="judgment:new.pdf", case_id="NEW/2025/DS-GDT"):
    return {
        "source_id": source_id,
        "case_id": case_id,
        "source_name": source_id.split(":", 1)[-1],
        "source_path": f"sources/{source_id.split(':', 1)[-1]}",
        "structured_path": None,
        "court": "toa_an_cap_cao",
        "procedure_stage": "giam_doc_tham",
        "year": 2025,
        "case_type": "tranh chap",
        "preliminary_law_families": ["BLDS"],
        "preliminary_legal_functions": ["substantive"],
        "source_sha256": "a" * 64,
        "structured_sha256": None,
        "metadata_source": "test",
    }


def _draft_manifest():
    return {
        "benchmark_id": BENCHMARK_ID,
        "schema_version": "gt_schema_v2",
        "state": "DRAFT",
        "git_commit": "a" * 40,
        "generation_timestamp": "2026-10-24T00:00:00+00:00",
        "corpus_fingerprint": CORPUS_FINGERPRINT,
        "deterministic_ordering": "case_id ASC, then source_id ASC",
        "targets": {"total": 250, "main": 200, "stress": 50},
    }


def _write_artifact(root, name, content):
    path = root / name
    path.write_text(content, encoding="utf-8")
    return {"path": name, "sha256": sha256_file(path)}


def _selection_records(count=250):
    records = []
    for index in range(count):
        record = _sampling_record(
            source_id=f"judgment:{index:03d}.pdf",
            case_id=f"{index:03d}/2025/DS-GDT",
        )
        record["source_path"] = f"sources/{index:03d}.pdf"
        record["source_sha256"] = hashlib.sha256(str(index).encode()).hexdigest()
        record["slice"] = "main" if index < min(200, count) else "stress"
        record["blttds_only_procedural"] = index < 40
        record["anticipated_substantive_core"] = index < 130
        records.append(record)
    return records


def _freeze_inputs(tmp_path, records=None):
    records = records or _selection_records()
    sampling = _write_artifact(tmp_path, "sampling_manifest.json", "{}")
    dedup = _write_artifact(tmp_path, "source_dedup_report.json", "{}")
    manifest = {
        **_draft_manifest(),
        "blocked_transitions": [],
        "blockers": [],
        "current_counts": {"eligible_sources": len(records)},
        "phase1_artifacts": [sampling, dedup],
    }
    updates = {
        "sampling_seed": 42,
        "selection_mode": "target",
        "selected_case_count": len(records),
        "selected_case_fingerprint": selected_records_fingerprint(records),
        "sampling_manifest_sha256": sampling["sha256"],
        "source_dedup_report_sha256": dedup["sha256"],
    }
    return manifest, updates, records, copy.deepcopy(records)


def _complete_sealed_manifest(tmp_path, state="SEALED"):
    entries = {
        "sampling": _write_artifact(tmp_path, "sampling_manifest.json", "sampling"),
        "dedup": _write_artifact(tmp_path, "source_dedup_report.json", "dedup"),
        "schema": _write_artifact(tmp_path, "schema.json", "{}"),
        "guidelines": _write_artifact(tmp_path, "guidelines.md", "guidelines"),
        "benchmark": _write_artifact(tmp_path, "benchmark.json", "[]"),
        "agreement": _write_artifact(tmp_path, "agreement.json", "{}"),
    }
    manifest = {
        **_draft_manifest(),
        "state": state,
        "sampling_seed": 42,
        "selection_mode": "target",
        "selected_case_count": 250,
        "selected_case_fingerprint": "b" * 64,
        "sampling_manifest_sha256": entries["sampling"]["sha256"],
        "source_dedup_report_sha256": entries["dedup"]["sha256"],
        "phase1_artifacts": [
            entries["sampling"],
            entries["dedup"],
            entries["schema"],
            entries["guidelines"],
        ],
        "annotation_started_at": "2026-10-24T01:00:00+00:00",
        "annotation_guidelines_path": entries["guidelines"]["path"],
        "annotation_guidelines_sha256": entries["guidelines"]["sha256"],
        "annotator_role_assignments_fingerprint": "e" * 64,
        "annotation_frozen_at": "2026-10-24T02:00:00+00:00",
        "benchmark_path": entries["benchmark"]["path"],
        "benchmark_sha256": entries["benchmark"]["sha256"],
        "gt_schema_path": entries["schema"]["path"],
        "gt_schema_sha256": entries["schema"]["sha256"],
        "annotation_agreement_path": entries["agreement"]["path"],
        "annotation_agreement_sha256": entries["agreement"]["sha256"],
        "sealed_at": "2026-10-24T03:00:00+00:00",
    }
    if state == "OPENED_FOR_CONFIRMATORY_EVALUATION":
        entries["preregistration"] = _write_artifact(
            tmp_path, "preregistration.json", "{}"
        )
        manifest.update({
            "opened_at": "2026-10-24T04:00:00+00:00",
            "preregistration_path": entries["preregistration"]["path"],
            "preregistration_sha256": entries["preregistration"]["sha256"],
        })
    return manifest, entries


def test_inventory_and_dedup_are_deterministic():
    first_inventory = inventory_judgment_sources(PROJECT_ROOT)
    second_inventory = inventory_judgment_sources(PROJECT_ROOT)
    assert canonical_json_fingerprint(first_inventory) == canonical_json_fingerprint(second_inventory)

    development = json.loads(
        (PROJECT_ROOT / "backend/data/judgments_evaluation.json").read_text(encoding="utf-8")
    )
    first_eligible, first_report = deduplicate_against_development(first_inventory, development)
    second_eligible, second_report = deduplicate_against_development(second_inventory, development)
    assert first_eligible == second_eligible
    assert first_report == second_report


def test_phase1_inventory_combines_development_and_eligible_independent_sources():
    sampling, report = phase1_payloads(
        PROJECT_ROOT, generation_timestamp="2026-10-24T00:00:00+00:00"
    )
    assert len(sampling["available_source_inventory"]) == 132
    assert report["development_sources_matched"] == 107
    assert report["development_sources_unmatched"] == 0
    eligible_ids = [record["source_id"] for record in sampling["eligible_sources"]]
    assert len(eligible_ids) == 25
    assert set(eligible_ids) == {
        f"independent:pilot-20261025-{index:03d}" for index in range(1, 25)
    } | {"independent:scale-20261026-001"}
    assert report["eligible_source_count"] == 25
    assert not [
        exclusion
        for exclusion in report["exclusions"]
        if exclusion["source_id"].startswith("independent:")
    ]


def test_known_duplicate_and_paraphrase_detection_is_deterministic():
    source_text = "Tranh chấp hợp đồng vay tài sản giữa nguyên đơn và bị đơn " * 20
    assert dedup_text_similarity(source_text, "  " + source_text.upper()) == 1.0

    consumed = _sampling_record("judgment:old.pdf", "OLD/2024/DS-GDT")
    consumed["source_name"] = "old.pdf"
    consumed["_dedup_text"] = source_text
    consumed["source_sha256"] = "b" * 64
    candidate = _sampling_record("judgment:new.pdf", "NEW/2025/DS-GDT")
    candidate["_dedup_text"] = source_text.upper()
    development = [{"id": 1, "judgment_name": "old.pdf"}]
    eligible, report = deduplicate_against_development([consumed, candidate], development)
    assert eligible == []
    assert any(
        reason.startswith("near_duplicate_source_text:")
        for item in report["exclusions"]
        if item["source_id"] == "judgment:new.pdf"
        for reason in item["reasons"]
    )


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("source_name", "Same.PDF", "new_pool_duplicate_source_identity"),
        ("source_sha256", "c" * 64, "new_pool_duplicate_sha256"),
        ("case_id", "SAME/2025/DS-GDT", "new_pool_duplicate_case_id"),
    ],
)
def test_new_pool_exact_duplicates_have_one_deterministic_representative(
    field, value, expected_reason
):
    left = _sampling_record("judgment:a.pdf", "1/2025/DS-GDT")
    right = _sampling_record("judgment:b.pdf", "2/2025/DS-GDT")
    left["source_sha256"] = hashlib.sha256(b"left").hexdigest()
    right["source_sha256"] = hashlib.sha256(b"right").hexdigest()
    left[field] = value
    right[field] = value

    eligible, report = deduplicate_against_development([right, left], [])

    assert [record["source_id"] for record in eligible] == ["judgment:a.pdf"]
    assert report["internal_duplicate_clusters"] == [{
        "canonical_source_id": "judgment:a.pdf",
        "member_source_ids": ["judgment:a.pdf", "judgment:b.pdf"],
    }]
    assert any(
        expected_reason in item["reasons"] for item in report["exclusions"]
    )


def test_near_duplicate_new_sources_are_deduplicated_deterministically():
    shared_text = "Tranh chấp hợp đồng vay tài sản giữa nguyên đơn và bị đơn " * 20
    left = _sampling_record("judgment:a.pdf", "1/2025/DS-GDT")
    right = _sampling_record("judgment:b.pdf", "2/2025/DS-GDT")
    left["source_sha256"] = hashlib.sha256(b"left").hexdigest()
    right["source_sha256"] = hashlib.sha256(b"right").hexdigest()
    left["_dedup_text"] = shared_text
    right["_dedup_text"] = shared_text.upper()

    forward = deduplicate_against_development([left, right], [])
    reverse = deduplicate_against_development([right, left], [])

    assert forward == reverse
    assert [record["source_id"] for record in forward[0]] == ["judgment:a.pdf"]
    assert any(
        reason.startswith("new_pool_near_duplicate_source_text:")
        for item in forward[1]["exclusions"]
        for reason in item["reasons"]
    )


def test_duplicate_selected_case_ids_and_development_overlap_are_rejected():
    duplicate = [_sampling_record(), _sampling_record("judgment:second.pdf")]
    with pytest.raises(BenchmarkProtocolError, match="Duplicate case_id"):
        validate_sampling_records(duplicate)
    with pytest.raises(BenchmarkProtocolError, match="overlap development"):
        validate_unique_selected_cases([_sampling_record()], {"NEW/2025/DS-GDT"})


def test_schema_snapshot_preserves_v2_semantics():
    schema = json.loads((BENCHMARK_DIR / "gt_schema_v2.snapshot.json").read_text(encoding="utf-8"))
    validate_schema_v2(schema)
    gt = schema["items"]["properties"]["ground_truth"]["properties"]
    assert gt["core"]["items"]["properties"]["importance_score"]["const"] == 3
    assert gt["supporting"]["items"]["properties"]["importance_score"]["const"] == 1
    for partition in ("core", "supporting"):
        assert gt[partition]["items"]["properties"]["domain"]["enum"] == [
            "substantive",
            "procedural",
        ]


def test_future_benchmark_records_are_validated_against_schema_snapshot():
    schema = json.loads((BENCHMARK_DIR / "gt_schema_v2.snapshot.json").read_text(encoding="utf-8"))
    valid = [{
        "id": 1,
        "judgment_name": "new.pdf",
        "query": "A human-written legal query",
        "ground_truth": {
            "core": [{
                "law_code": "bo_luat_dan_su_2015",
                "article": "Khoản 1 Điều 1",
                "reasoning": "Human annotation rationale",
                "importance": "decisive",
                "importance_score": 3,
                "source_type": "explicit",
                "domain": "substantive",
            }],
            "supporting": [],
        },
    }]
    validate_benchmark_records(valid, schema)
    invalid = copy.deepcopy(valid)
    invalid[0]["ground_truth"]["core"][0]["importance_score"] = 1
    with pytest.raises(BenchmarkProtocolError, match="schema validation failed"):
        validate_benchmark_records(invalid, schema)


def test_core_support_and_legal_function_dimensions_cannot_be_conflated():
    schema = json.loads((BENCHMARK_DIR / "gt_schema_v2.snapshot.json").read_text(encoding="utf-8"))
    core = schema["items"]["properties"]["ground_truth"]["properties"]["core"]["items"]
    assert core["properties"]["importance"]["const"] == "decisive"
    assert core["properties"]["domain"]["enum"] == ["substantive", "procedural"]


def test_canonical_mapping_preserves_raw_subarticle_annotation_at_article_level():
    assert canonical_retrieval_id(
        "Điểm a khoản 1 Điều 337 bo_luat_to_tung_dan_su_2015"
    ) == "điều 337 bo_luat_to_tung_dan_su_2015"


def test_quota_guardrails_fail_closed_for_insufficient_eligible_frame():
    sampling = json.loads((BENCHMARK_DIR / "sampling_manifest.json").read_text(encoding="utf-8"))
    feasibility = sampling["quota_feasibility"]
    assert feasibility["status"] == "INFEASIBLE"
    assert feasibility["eligible_count"] == 25
    assert feasibility["minimum_total"] == 180
    assert "eligible source count 25 is below minimum total 180" in feasibility["blockers"]
    assert feasibility["main_guardrails"]["BLTTDS_only_procedural_max_ratio"] == 0.20
    assert feasibility["main_guardrails"]["substantive_core_min_ratio"] == 0.65


def test_selection_guardrail_validator_enforces_target_and_minimum_contracts():
    records = []
    for index in range(250):
        record = _sampling_record(
            source_id=f"judgment:{index}.pdf",
            case_id=f"{index}/2025/DS-GDT",
        )
        record["slice"] = "main" if index < 200 else "stress"
        record["blttds_only_procedural"] = index < 40
        record["anticipated_substantive_core"] = index < 130
        records.append(record)
    result = validate_selection_guardrails(records)
    assert result["BLTTDS_only_procedural_ratio"] == 0.20
    assert result["substantive_core_ratio"] == 0.65

    with pytest.raises(BenchmarkProtocolError, match="limitation reason"):
        validate_selection_guardrails(records[:180], mode="minimum")
    with pytest.raises(BenchmarkProtocolError, match="below 65%"):
        bad = copy.deepcopy(records)
        bad[129]["anticipated_substantive_core"] = False
        validate_selection_guardrails(bad)


def test_manifest_state_transitions_are_strict_and_require_fields(tmp_path):
    draft = _draft_manifest()
    validate_manifest_state(draft)
    with pytest.raises(BenchmarkProtocolError, match="Invalid state transition"):
        transition_manifest(draft, "ANNOTATION_IN_PROGRESS", {})
    with pytest.raises(BenchmarkProtocolError, match="missing fields"):
        transition_manifest(draft, "SAMPLING_FROZEN", {})
    manifest, updates, selected, eligible = _freeze_inputs(tmp_path)
    frozen = transition_manifest(
        manifest,
        "SAMPLING_FROZEN",
        updates,
        selected_records=selected,
        eligible_records=eligible,
        project_root=tmp_path,
    )
    assert frozen["state"] == "SAMPLING_FROZEN"


def test_live_draft_manifest_blocked_transition_is_rejected():
    manifest = json.loads(
        (BENCHMARK_DIR / "benchmark_manifest.json").read_text(encoding="utf-8")
    )
    with pytest.raises(BenchmarkProtocolError, match="transition is blocked"):
        transition_manifest(manifest, "SAMPLING_FROZEN", {})


def test_single_selected_case_cannot_bypass_target_quota(tmp_path):
    records = _selection_records(1)
    manifest, updates, selected, eligible = _freeze_inputs(tmp_path, records)
    with pytest.raises(BenchmarkProtocolError, match="exactly 250"):
        transition_manifest(
            manifest,
            "SAMPLING_FROZEN",
            updates,
            selected_records=selected,
            eligible_records=eligible,
            project_root=tmp_path,
        )


def test_selected_case_count_must_equal_selected_records(tmp_path):
    manifest, updates, selected, eligible = _freeze_inputs(tmp_path)
    updates["selected_case_count"] = len(selected) - 1
    with pytest.raises(BenchmarkProtocolError, match="must equal"):
        transition_manifest(
            manifest,
            "SAMPLING_FROZEN",
            updates,
            selected_records=selected,
            eligible_records=eligible,
            project_root=tmp_path,
        )


@pytest.mark.parametrize(
    "fingerprint",
    ["abc", "a" * 63, "a" * 65, "z" * 64],
)
def test_sampling_freeze_rejects_malformed_fingerprints(tmp_path, fingerprint):
    manifest, updates, selected, eligible = _freeze_inputs(tmp_path)
    updates["selected_case_fingerprint"] = fingerprint
    with pytest.raises(BenchmarkProtocolError, match="64 lowercase hexadecimal"):
        transition_manifest(
            manifest,
            "SAMPLING_FROZEN",
            updates,
            selected_records=selected,
            eligible_records=eligible,
            project_root=tmp_path,
        )


def test_sampling_freeze_rejects_recomputed_fingerprint_mismatch(tmp_path):
    manifest, updates, selected, eligible = _freeze_inputs(tmp_path)
    updates["selected_case_fingerprint"] = "f" * 64
    with pytest.raises(BenchmarkProtocolError, match="fingerprint mismatch"):
        transition_manifest(
            manifest,
            "SAMPLING_FROZEN",
            updates,
            selected_records=selected,
            eligible_records=eligible,
            project_root=tmp_path,
        )


def test_minimum_quota_cannot_be_bypassed_without_limitation_reason(tmp_path):
    records = _selection_records(180)
    for index, record in enumerate(records):
        record["slice"] = "main" if index < 150 else "stress"
    manifest, updates, selected, eligible = _freeze_inputs(tmp_path, records)
    updates["selection_mode"] = "minimum"
    with pytest.raises(BenchmarkProtocolError, match="limitation reason"):
        transition_manifest(
            manifest,
            "SAMPLING_FROZEN",
            updates,
            selected_records=selected,
            eligible_records=eligible,
            project_root=tmp_path,
        )


def test_fingerprint_determinism():
    left = {"b": [2, 1], "a": "x"}
    right = {"a": "x", "b": [2, 1]}
    assert canonical_json_fingerprint(left) == canonical_json_fingerprint(right)


@pytest.mark.parametrize(
    ("state", "artifact_name"),
    [
        ("SEALED", "sampling"),
        ("SEALED", "dedup"),
        ("SEALED", "schema"),
        ("SEALED", "guidelines"),
        ("SEALED", "agreement"),
        ("OPENED_FOR_CONFIRMATORY_EVALUATION", "preregistration"),
    ],
)
def test_each_sealed_artifact_is_tamper_detected(tmp_path, state, artifact_name):
    manifest, entries = _complete_sealed_manifest(tmp_path, state)
    verify_sealed_artifacts(manifest, tmp_path)
    (tmp_path / entries[artifact_name]["path"]).write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(BenchmarkProtocolError, match="changed"):
        verify_sealed_artifacts(manifest, tmp_path)


def test_missing_required_sealed_artifact_fails(tmp_path):
    manifest, entries = _complete_sealed_manifest(tmp_path)
    (tmp_path / entries["agreement"]["path"]).unlink()
    with pytest.raises(BenchmarkProtocolError, match="missing"):
        verify_sealed_artifacts(manifest, tmp_path)


@pytest.mark.parametrize("state", ["SEALED", "OPENED_FOR_CONFIRMATORY_EVALUATION"])
def test_valid_complete_sealing_contract_succeeds(tmp_path, state):
    manifest, _ = _complete_sealed_manifest(tmp_path, state)
    verify_sealed_artifacts(manifest, tmp_path)


@pytest.mark.parametrize(
    "forbidden_key",
    ["bm25_rank", "dense_score", "hybrid_candidates", "retrieval_metrics"],
)
def test_sampling_records_reject_retrieval_output_fields(forbidden_key):
    record = _sampling_record()
    record[forbidden_key] = 1
    with pytest.raises(BenchmarkProtocolError, match="forbidden"):
        validate_sampling_records([record])


def test_checked_in_phase1_artifacts_match_live_inventory():
    generated_sampling, generated_dedup = phase1_payloads(
        PROJECT_ROOT, generation_timestamp="2026-10-24T00:00:00+00:00"
    )
    checked_sampling = json.loads((BENCHMARK_DIR / "sampling_manifest.json").read_text(encoding="utf-8"))
    checked_dedup = json.loads((BENCHMARK_DIR / "source_dedup_report.json").read_text(encoding="utf-8"))
    for key in (
        "inventory_fingerprint",
        "eligible_fingerprint",
        "available_source_inventory",
        "eligible_sources",
        "available_breakdown",
        "eligible_breakdown",
        "quota_feasibility",
    ):
        assert checked_sampling[key] == generated_sampling[key]
    for key in (
        "algorithm",
        "inventory_fingerprint",
        "eligible_fingerprint",
        "development_source_mapping",
        "exclusions",
        "internal_duplicate_clusters",
        "ordering_rule",
        "reason_counts",
    ):
        assert checked_dedup[key] == generated_dedup[key]


def test_draft_manifest_hashes_all_phase1_artifacts():
    manifest = draft_benchmark_manifest(
        PROJECT_ROOT, generation_timestamp="2026-10-24T00:00:00+07:00"
    )
    checked = json.loads((BENCHMARK_DIR / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert checked["state"] == "DRAFT"
    assert checked["corpus_fingerprint"] == CORPUS_FINGERPRINT
    assert checked["phase1_artifacts"] == manifest["phase1_artifacts"]
    assert checked["source_files"] == manifest["source_files"]
    assert checked["current_counts"]["available_sources"] == 132
    assert checked["current_counts"]["development_sources_consumed"] == 107
    assert checked["current_counts"]["eligible_sources"] == 25
    assert checked["current_counts"]["selected_cases"] == 0
    assert checked["current_counts"]["query_records"] == 0
    assert checked["current_counts"]["ground_truth_records"] == 0


def test_no_final_benchmark_or_gt_records_exist():
    assert not (BENCHMARK_DIR / "judgments_evaluation_independent_v1.json").exists()
    sampling = json.loads((BENCHMARK_DIR / "sampling_manifest.json").read_text(encoding="utf-8"))
    assert sampling["selection_status"] == "NOT_STARTED"
    assert len(sampling["eligible_sources"]) == 25
    assert "selected_cases" not in sampling

