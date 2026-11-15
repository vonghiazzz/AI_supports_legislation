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
        "ready",
        "verdict",
    ):
        assert tracked[key] == expected[key]


def _ready_metadata(protocol):
    fingerprint = protocol["integrity"]["protocol_fingerprint"]
    benchmark = {
        "benchmark_id": "retrieval_independent_v1",
        "state": "OPENED_FOR_CONFIRMATORY_EVALUATION",
        "targets": {"minimum_total": 180},
        "current_counts": {"selected_cases": 250, "query_records": 250, "ground_truth_records": 1500},
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
        "selected_case_fingerprint": "1" * 64,
        "eligible_fingerprint": "2" * 64,
    }
    return benchmark, sampling


def test_all_frozen_metadata_and_separate_authorization_are_required(protocol):
    benchmark, sampling = _ready_metadata(protocol)
    blocked = a2.assess_independent_benchmark_readiness(protocol, benchmark, sampling)
    assert blocked["verdict"] == a2.BLOCKED_VERDICT
    assert blocked["missing_requirements"] == ["execution_authorized"]
    ready = a2.assess_independent_benchmark_readiness(
        protocol, benchmark, sampling, execution_authorized=True
    )
    assert ready["verdict"] == a2.READY_VERDICT
    assert ready["missing_requirements"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "draft_state",
        "selection_not_started",
        "seed_missing",
        "count_below_minimum",
        "query_count_mismatch",
        "gt_missing",
        "fingerprint_missing",
        "leakage_audit_missing",
        "model_unverified",
        "parameters_changed",
    ],
)
def test_each_future_readiness_failure_blocks(protocol, mutation):
    benchmark, sampling = _ready_metadata(protocol)
    if mutation == "draft_state":
        benchmark["state"] = "DRAFT"
    elif mutation == "selection_not_started":
        sampling["selection_status"] = "NOT_STARTED"
    elif mutation == "seed_missing":
        sampling["sampling_seed"] = None
    elif mutation == "count_below_minimum":
        benchmark["current_counts"]["selected_cases"] = 179
        benchmark["current_counts"]["query_records"] = 179
    elif mutation == "query_count_mismatch":
        benchmark["current_counts"]["query_records"] = 249
    elif mutation == "gt_missing":
        benchmark["current_counts"]["ground_truth_records"] = 0
    elif mutation == "fingerprint_missing":
        benchmark.pop("a2_protocol_fingerprint")
    elif mutation == "leakage_audit_missing":
        benchmark["leakage_audit_status"] = "PENDING"
    elif mutation == "model_unverified":
        benchmark["a2_treatment_model_status"] = "PENDING"
    elif mutation == "parameters_changed":
        benchmark["a2_parameter_change_status"] = "CHANGED"
    result = a2.assess_independent_benchmark_readiness(
        protocol, benchmark, sampling, execution_authorized=True
    )
    assert result["verdict"] == a2.BLOCKED_VERDICT


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
