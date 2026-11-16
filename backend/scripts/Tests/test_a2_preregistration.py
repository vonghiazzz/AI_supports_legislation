import ast
import copy
import json
from pathlib import Path

import pytest

from scripts.Evaluation.experiments import a2_dense_encoder_swap_preregistration as a2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_PATH = PROJECT_ROOT / "backend/scripts/Evaluation/configs/a2_preregistered_protocol.yaml"
BENCHMARK_MANIFEST_PATH = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
SAMPLING_MANIFEST_PATH = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
ARTIFACT_ROOT = PROJECT_ROOT / "backend/data/preregistrations" / a2.EXPERIMENT_ID
VALIDATION_PATH = ARTIFACT_ROOT / "a2_protocol_validation.json"
READINESS_PATH = ARTIFACT_ROOT / "a2_experiment_readiness.json"


@pytest.fixture
def protocol():
    return a2.load_protocol(PROTOCOL_PATH)


def test_frozen_protocol_is_valid_and_self_fingerprinting(protocol):
    result = a2.validate_protocol(protocol)
    assert result == {
        "experiment_id": a2.EXPERIMENT_ID,
        "protocol_fingerprint": protocol["integrity"]["protocol_fingerprint"],
        "valid": True,
        "treatment_difference": "dense_encoder_only",
    }


def test_yaml_duplicate_keys_fail_closed(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("protocol: 1\nprotocol: 2\n", encoding="utf-8")
    with pytest.raises(a2.A2ProtocolError, match="duplicate YAML key"):
        a2.load_protocol(path)


def test_protocol_fingerprint_ignores_only_its_own_value(protocol):
    changed = copy.deepcopy(protocol)
    changed["integrity"]["protocol_fingerprint"] = "0" * 64
    assert a2.protocol_fingerprint(changed) == a2.protocol_fingerprint(protocol)
    changed["candidate_generation"]["fusion"]["w_dense"] = 0.6
    assert a2.protocol_fingerprint(changed) != a2.protocol_fingerprint(protocol)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("scope", "reranking"), True, "reranking"),
        (("candidate_generation", "fusion", "output_depth"), 30, "candidate-generation"),
        (("statistics", "bootstrap_resamples"), 1000, "statistics"),
        (("evaluation", "primary_metric"), "weighted_recall_at_10", "primary metric"),
        (("integrity", "source_base_commit"), "0" * 40, "base commit"),
    ],
)
def test_protocol_drift_fails_closed(protocol, path, value, message):
    changed = copy.deepcopy(protocol)
    cursor = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    changed["integrity"]["protocol_fingerprint"] = a2.protocol_fingerprint(changed)
    with pytest.raises(a2.A2ProtocolError, match=message):
        a2.validate_protocol(changed)


def test_control_and_treatment_differ_only_by_dense_encoder(protocol):
    control = copy.deepcopy(protocol["systems"]["control"])
    treatment = copy.deepcopy(protocol["systems"]["treatment"])
    treatment["dense_encoder"] = copy.deepcopy(control["dense_encoder"])
    assert treatment == control


def test_non_dense_treatment_change_is_rejected(protocol):
    changed = copy.deepcopy(protocol)
    changed["systems"]["treatment"]["dense_depth"] = 30
    changed["integrity"]["protocol_fingerprint"] = a2.protocol_fingerprint(changed)
    with pytest.raises(a2.A2ProtocolError, match="differ only"):
        a2.validate_protocol(changed)


def test_current_independent_benchmark_is_blocked_without_content_access(protocol):
    benchmark = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))
    sampling = json.loads(SAMPLING_MANIFEST_PATH.read_text(encoding="utf-8"))
    result = a2.assess_independent_benchmark_readiness(protocol, benchmark, sampling)
    assert result["verdict"] == a2.BLOCKED_VERDICT
    assert result["ready"] is False
    assert result["metadata_only"] is True
    assert result["query_or_gt_content_read"] is False
    assert benchmark["state"] == "DRAFT"
    assert sampling["selection_status"] == "NOT_STARTED"
    assert sampling["sampling_seed"] is None
    assert benchmark["current_counts"]["selected_cases"] == 0
    assert benchmark["current_counts"]["query_records"] == 0
    assert benchmark["current_counts"]["ground_truth_records"] == 0


@pytest.mark.parametrize("path", [VALIDATION_PATH, READINESS_PATH])
def test_tracked_artifact_fingerprint_is_canonical(path):
    artifact = json.loads(path.read_text(encoding="utf-8"))
    recorded = artifact.pop("artifact_fingerprint")
    assert recorded == a2.canonical_json_fingerprint(artifact)


def test_tracked_readiness_artifact_matches_live_metadata_assessment(protocol):
    benchmark = json.loads(BENCHMARK_MANIFEST_PATH.read_text(encoding="utf-8"))
    sampling = json.loads(SAMPLING_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = a2.assess_independent_benchmark_readiness(protocol, benchmark, sampling)
    tracked = json.loads(READINESS_PATH.read_text(encoding="utf-8"))
    for key in (
        "experiment_id",
        "protocol_fingerprint",
        "metadata_only",
        "query_or_gt_content_read",
        "checks",
        "missing_requirements",
        "required_future_metadata_interface",
        "ready",
        "verdict",
    ):
        assert tracked[key] == expected[key]


def _selection_records(total=250, main=200):
    records = []
    for index in range(total):
        records.append({
            "source_id": f"source-{index:03d}",
            "case_id": f"case-{index:03d}",
            "source_name": f"case-{index:03d}.pdf",
            "source_path": f"sources/case-{index:03d}.pdf",
            "source_sha256": f"{index + 1:064x}",
            "slice": "main" if index < main else "stress",
            "blttds_only_procedural": index < min(40, main),
            "anticipated_substantive_core": index < min(130, main),
        })
    return records


def _seal_evidence(evidence):
    evidence["evidence_fingerprint"] = a2._evidence_fingerprint(evidence)


def _ready_metadata(protocol, total=250, main=200):
    fingerprint = protocol["integrity"]["protocol_fingerprint"]
    selected = _selection_records(total, main)
    final_frame = copy.deepcopy(selected)
    selected_fingerprint = a2.selected_records_fingerprint(selected)
    frame_fingerprint = a2.canonical_json_fingerprint(final_frame)
    gt_count = max(total, 1)
    selection_evidence = {
        "schema_version": "independent_selection_evidence_v1",
        "validation_status": "PASSED",
        "validator": "independent_benchmark.validate_selection_guardrails:target",
        "selection_mode": "TARGET_ONLY",
        "target_counts": {"total": 250, "main": 200, "stress": 50},
        "selected_case_count": total,
        "slice_counts": {"main": main, "stress": total - main},
        "slices_disjoint": True,
        "selected_case_fingerprint": selected_fingerprint,
        "eligible_frame_fingerprint": frame_fingerprint,
        "fallback_controls": {
            "fallback_authorized": False,
            "legacy_minimum_mode_authorized": False,
            "silent_quota_reduction_authorized": False,
            "proportional_quota_scaling_authorized": False,
        },
        "selected_records": selected,
    }
    final_frame_evidence = {
        "schema_version": "final_eligible_frame_evidence_v1",
        "state": "FINALIZED",
        "validation_status": "VERIFIED",
        "source_provenance_status": "VERIFIED",
        "eligible_fingerprint": frame_fingerprint,
        "eligible_records": final_frame,
    }
    annotation_evidence = {
        "schema_version": "independent_annotation_validation_evidence_v1",
        "validation_status": "PASSED",
        "validator": "independent_benchmark.validate_one_case_one_query_contract",
        "one_query_per_selected_case": True,
        "selected_case_fingerprint": selected_fingerprint,
        "query_count": total,
        "ground_truth_records": gt_count,
        "gt_schema_version": "gt_schema_v2",
    }
    for evidence in (selection_evidence, final_frame_evidence, annotation_evidence):
        _seal_evidence(evidence)
    benchmark = {
        "benchmark_id": "retrieval_independent_v1",
        "state": "OPENED_FOR_CONFIRMATORY_EVALUATION",
        "selection_mode": "TARGET_ONLY",
        "targets": {"total": 250, "main": 200, "stress": 50, "minimum_total": 180},
        "selected_case_count": total,
        "selected_case_fingerprint": selected_fingerprint,
        "final_eligible_frame_fingerprint": frame_fingerprint,
        "selection_evidence_fingerprint": selection_evidence["evidence_fingerprint"],
        "final_frame_evidence_fingerprint": final_frame_evidence["evidence_fingerprint"],
        "annotation_validation_evidence_fingerprint": annotation_evidence["evidence_fingerprint"],
        "current_counts": {
            "selected_cases": total,
            "query_records": total,
            "ground_truth_records": gt_count,
        },
        "a2_protocol_fingerprint": fingerprint,
        "close_gate_status": "PASSED",
        "source_provenance_status": "VERIFIED",
        "leakage_audit_status": "PASSED",
        "a2_control_model_status": "VERIFIED",
        "a2_treatment_model_status": "VERIFIED",
        "a2_parameter_change_status": "UNCHANGED",
    }
    sampling = {
        "state": "SEALED",
        "selection_status": "COMPLETED",
        "sampling_seed": 20261115,
        "selection_mode": "TARGET_ONLY",
        "targets": {"total": 250, "main": 200, "stress": 50},
        "selected_case_count": total,
        "selected_case_fingerprint": selected_fingerprint,
        "eligible_fingerprint": frame_fingerprint,
        "selection_evidence_fingerprint": selection_evidence["evidence_fingerprint"],
        "final_frame_evidence_fingerprint": final_frame_evidence["evidence_fingerprint"],
    }
    return benchmark, sampling, selection_evidence, final_frame_evidence, annotation_evidence


def _assess(protocol, bundle, *, authorized=True):
    benchmark, sampling, selection, frame, annotation = bundle
    return a2.assess_independent_benchmark_readiness(
        protocol,
        benchmark,
        sampling,
        selection_evidence=selection,
        final_frame_evidence=frame,
        annotation_evidence=annotation,
        execution_authorized=authorized,
    )


def test_all_frozen_metadata_and_separate_authorization_are_required(protocol):
    bundle = _ready_metadata(protocol)
    blocked = _assess(protocol, bundle, authorized=False)
    assert blocked["verdict"] == a2.BLOCKED_VERDICT
    assert blocked["missing_requirements"] == ["execution_authorized"]
    ready = _assess(protocol, bundle)
    assert ready["verdict"] == a2.READY_VERDICT
    assert ready["missing_requirements"] == []


@pytest.mark.parametrize(
    "total",
    [180, 200, 249, 251],
)
def test_non_target_selected_totals_are_blocked(protocol, total):
    result = _assess(protocol, _ready_metadata(protocol, total=total, main=min(200, total)))
    assert result["verdict"] == a2.BLOCKED_VERDICT
    assert result["checks"]["selected_total_exact"] is False


@pytest.mark.parametrize(("main", "stress"), [(199, 51), (201, 49)])
def test_wrong_main_stress_split_is_blocked(protocol, main, stress):
    result = _assess(protocol, _ready_metadata(protocol, total=main + stress, main=main))
    assert result["verdict"] == a2.BLOCKED_VERDICT
    assert result["checks"]["selected_slice_counts_exact"] is False


@pytest.mark.parametrize("failure", ["missing_counts", "unverified"])
def test_missing_or_unverified_slice_counts_are_blocked(protocol, failure):
    bundle = _ready_metadata(protocol)
    benchmark, sampling, selection, _, _ = bundle
    if failure == "missing_counts":
        selection["slice_counts"] = None
    else:
        selection["validation_status"] = "PENDING"
    _seal_evidence(selection)
    benchmark["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    sampling["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    if failure == "missing_counts":
        assert result["checks"]["selected_slice_counts_exact"] is False
    else:
        assert result["checks"]["selection_evidence_bound"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_caller_count_claims_cannot_override_249_actual_records(protocol):
    bundle = _ready_metadata(protocol, total=249, main=200)
    benchmark, sampling, selection, _, annotation = bundle
    benchmark["selected_case_count"] = 250
    benchmark["current_counts"]["selected_cases"] = 250
    benchmark["current_counts"]["query_records"] = 250
    sampling["selected_case_count"] = 250
    selection["selected_case_count"] = 250
    annotation["query_count"] = 250
    _seal_evidence(selection)
    _seal_evidence(annotation)
    benchmark["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    sampling["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    benchmark["annotation_validation_evidence_fingerprint"] = annotation["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["selected_total_exact"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_minimum_mode_fallback_is_blocked(protocol):
    bundle = _ready_metadata(protocol)
    benchmark, sampling, selection, _, _ = bundle
    benchmark["selection_mode"] = "MINIMUM"
    sampling["selection_mode"] = "MINIMUM"
    selection["selection_mode"] = "MINIMUM"
    selection["fallback_controls"]["legacy_minimum_mode_authorized"] = True
    _seal_evidence(selection)
    benchmark["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    sampling["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["target_mode_target_only"] is False
    assert result["checks"]["fallbacks_disabled"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_duplicate_or_overlapping_selected_cases_are_blocked(protocol):
    bundle = _ready_metadata(protocol)
    benchmark, sampling, selection, _, annotation = bundle
    selection["selected_records"][-1]["case_id"] = selection["selected_records"][0]["case_id"]
    selection["selected_case_fingerprint"] = a2.selected_records_fingerprint(selection["selected_records"])
    benchmark["selected_case_fingerprint"] = selection["selected_case_fingerprint"]
    sampling["selected_case_fingerprint"] = selection["selected_case_fingerprint"]
    annotation["selected_case_fingerprint"] = selection["selected_case_fingerprint"]
    _seal_evidence(selection)
    _seal_evidence(annotation)
    benchmark["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    sampling["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    benchmark["annotation_validation_evidence_fingerprint"] = annotation["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["selected_slices_disjoint"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_stale_eligible_frame_fingerprint_is_blocked(protocol):
    bundle = _ready_metadata(protocol)
    benchmark, sampling, selection, frame, _ = bundle
    stale = "9" * 64
    selection["eligible_frame_fingerprint"] = stale
    frame["eligible_fingerprint"] = stale
    sampling["eligible_fingerprint"] = stale
    benchmark["final_eligible_frame_fingerprint"] = stale
    _seal_evidence(selection)
    _seal_evidence(frame)
    benchmark["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    sampling["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    benchmark["final_frame_evidence_fingerprint"] = frame["evidence_fingerprint"]
    sampling["final_frame_evidence_fingerprint"] = frame["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["eligible_frame_fingerprint_matches_final"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_invalid_selected_case_fingerprint_is_blocked(protocol):
    bundle = _ready_metadata(protocol)
    benchmark, sampling, selection, _, annotation = bundle
    invalid = "f" * 64
    selection["selected_case_fingerprint"] = invalid
    benchmark["selected_case_fingerprint"] = invalid
    sampling["selected_case_fingerprint"] = invalid
    annotation["selected_case_fingerprint"] = invalid
    _seal_evidence(selection)
    _seal_evidence(annotation)
    benchmark["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    sampling["selection_evidence_fingerprint"] = selection["evidence_fingerprint"]
    benchmark["annotation_validation_evidence_fingerprint"] = annotation["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["selected_case_fingerprint_valid"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_incorrect_query_count_is_blocked(protocol):
    bundle = _ready_metadata(protocol)
    benchmark, _, _, _, annotation = bundle
    benchmark["current_counts"]["query_records"] = 249
    annotation["query_count"] = 249
    _seal_evidence(annotation)
    benchmark["annotation_validation_evidence_fingerprint"] = annotation["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["query_count_matches_selection"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_missing_gt_is_blocked(protocol):
    bundle = _ready_metadata(protocol)
    benchmark, _, _, _, annotation = bundle
    benchmark["current_counts"]["ground_truth_records"] = 0
    annotation["ground_truth_records"] = 0
    _seal_evidence(annotation)
    benchmark["annotation_validation_evidence_fingerprint"] = annotation["evidence_fingerprint"]
    result = _assess(protocol, bundle)
    assert result["checks"]["gt_count_positive"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_draft_benchmark_remains_blocked_even_with_synthetic_evidence(protocol):
    bundle = _ready_metadata(protocol)
    bundle[0]["state"] = "DRAFT"
    result = _assess(protocol, bundle)
    assert result["checks"]["benchmark_state"] is False
    assert result["verdict"] == a2.BLOCKED_VERDICT


def test_frozen_protocol_fingerprint_is_unchanged(protocol):
    assert protocol["integrity"]["protocol_fingerprint"] == (
        "b30f2ea89e545bda004dd6ea05e1c52ce7f958e664f823ed9a347febf670246f"
    )


def test_module_has_no_model_or_retrieval_imports():
    source = Path(a2.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = {
        "sentence_transformers",
        "transformers",
        "faiss",
        "scripts.Evaluation.retrieval",
        "scripts.Evaluation.core.corpus",
    }
    assert not any(
        module == name or module.startswith(name + ".")
        for module in imported
        for name in forbidden
    )
