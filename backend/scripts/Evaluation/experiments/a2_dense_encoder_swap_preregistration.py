"""Fail-closed preregistration validator for RESEARCH-A2.

This module freezes metadata only.  It deliberately contains no retrieval,
model-loading, query-generation, ground-truth loading, or evaluation code.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import yaml


EXPERIMENT_ID = "candidate_generation_dense_encoder_swap_minmax50_v1"
BLOCKED_VERDICT = "A2_CONFIRMATORY_EVALUATION_BLOCKED"
READY_VERDICT = "A2_CONFIRMATORY_EVALUATION_READY"
EXPECTED_BASE_COMMIT = "2cbdddf0647bd8c9d3ff30c6973100905f7371c7"
EXPECTED_BRANCH = "research/a2-dense-encoder-swap-prereg"
EXPECTED_CORPUS_FINGERPRINT = (
    "66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58"
)
EXPECTED_CONTROL_MODEL = {
    "id": "keepitreal/vietnamese-sbert",
    "revision": "a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23",
    "architecture": "SentenceTransformer",
    "dimension": 768,
    "pooling": "model_native",
    "query_instruction": None,
    "normalize_embeddings": True,
    "max_seq_length": "model_native",
    "batch_size": 1,
    "device": "cpu",
    "dtype": "float32",
    "local_files_only": True,
}
EXPECTED_TREATMENT_MODEL = {
    "id": "BAAI/bge-m3",
    "revision": "5617a9f61b028005a4858fdac845db406aefb181",
    "architecture": "XLMRobertaModel",
    "dimension": 1024,
    "pooling": "cls",
    "query_instruction": None,
    "normalize_embeddings": True,
    "max_seq_length": 8192,
    "batch_size": 1,
    "device": "cpu",
    "dtype": "float32",
    "local_files_only": True,
}


class A2ProtocolError(ValueError):
    """Raised when the preregistered contract is incomplete or changed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise A2ProtocolError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_protocol(path: Path) -> dict:
    payload = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise A2ProtocolError("protocol must be a YAML mapping")
    return payload


def canonical_json_fingerprint(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_fingerprint(protocol: dict) -> str:
    portable = copy.deepcopy(protocol)
    portable.setdefault("integrity", {}).pop("protocol_fingerprint", None)
    return canonical_json_fingerprint(portable)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise A2ProtocolError(message)


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_only_dense_encoder_diff(protocol: dict) -> None:
    control = copy.deepcopy(protocol["systems"]["control"])
    treatment = copy.deepcopy(protocol["systems"]["treatment"])
    _require(control["dense_encoder"] == EXPECTED_CONTROL_MODEL, "control dense encoder drift")
    _require(
        treatment["dense_encoder"] == EXPECTED_TREATMENT_MODEL,
        "treatment dense encoder drift",
    )
    treatment["dense_encoder"] = copy.deepcopy(control["dense_encoder"])
    _require(
        treatment == control,
        "control and treatment may differ only in dense_encoder",
    )


def validate_protocol(protocol: dict) -> dict:
    """Validate the complete frozen contract and return stable validation data."""
    required_sections = {
        "protocol",
        "scope",
        "representations",
        "candidate_generation",
        "systems",
        "evaluation",
        "statistics",
        "success_contract",
        "independent_benchmark_gate",
        "prohibited_changes",
        "implementation_readiness",
        "integrity",
    }
    _require(set(protocol) == required_sections, "top-level protocol sections changed")
    meta = protocol["protocol"]
    _require(meta["id"] == EXPERIMENT_ID, "experiment id mismatch")
    _require(meta["version"] == 1, "protocol version must be 1")
    _require(meta["status"] == "PREREGISTERED_FROZEN", "protocol is not frozen")
    _require(meta["research_phase"] == "confirmatory_candidate_generation", "scope drift")
    _require(meta["automatic_runtime_promotion"] is False, "runtime promotion forbidden")
    _require(meta["development_benchmark_use"] == "prohibited", "DEV107 fallback forbidden")

    scope = protocol["scope"]
    _require(scope["treatment_variable"] == "dense_encoder_only", "treatment drift")
    _require(scope["reranking"] is False, "reranking is outside A2")
    _require(scope["candidate_generation_experiment"] is True, "scope must be retrieval")
    _require(scope["confirmatory_dataset"] == "retrieval_independent_v1", "dataset drift")

    _require(protocol["representations"] == {
        "query": "Q0",
        "document": "D0",
        "document_template": "Tiêu đề: {title}. Nội dung: {content}",
    }, "Q0/D0 representation drift")

    candidate = protocol["candidate_generation"]
    _require(candidate == {
        "bm25": {"tokenizer": "whitespace", "depth": 50},
        "dense": {"similarity": "inner_product", "faiss_index": "IndexFlatIP", "depth": 50},
        "fusion": {
            "method": "minmax",
            "normalization_pool": "full_source_top50_per_retriever_per_query",
            "w_bm25": 0.5,
            "w_dense": 0.5,
            "output_depth": 50,
        },
        "source_tie_break": "score_desc_then_standard_law_id_asc",
        "fusion_tie_break": "fused_score_desc_then_standard_law_id_asc",
        "score_rounding_before_sort": False,
    }, "candidate-generation contract drift")
    _validate_only_dense_encoder_diff(protocol)

    evaluation = protocol["evaluation"]
    _require(evaluation["world"] == "closed_world", "evaluation must be closed-world")
    _require(evaluation["primary_metric"] == "macro_candidate_recall_at_50", "primary metric drift")
    _require(evaluation["primary_denominator"] == "all_evaluable_queries", "query denominator drift")
    _require(evaluation["per_query_denominator"] == "in_corpus_ground_truth_instances", "GT denominator drift")
    _require(evaluation["relevance_weights"] == {"core": 3, "support": 1}, "weights drift")
    _require(evaluation["secondary_metrics"] == [
        "core_recall_at_50",
        "support_recall_at_50",
        "candidate_recall_at_10",
        "candidate_recall_at_20",
        "candidate_recall_at_30",
        "zero_hit_query_count_at_50",
        "raw_fused_weighted_recall_at_10",
        "micro_gt_coverage_at_50",
        "corpus_gt_coverage_at_50",
        "query_latency_p50_ms",
        "query_latency_p95_ms",
        "peak_memory_bytes",
    ], "secondary metrics drift")

    stats = protocol["statistics"]
    _require(stats == {
        "paired_unit": "query",
        "effect": "treatment_minus_control",
        "bootstrap_resamples": 20000,
        "random_seed": 42,
        "confidence_interval": "two_sided_95_percentile",
        "report_wins_ties_losses": True,
        "slice_reporting": "MAIN_and_STRESS_separate_only_when_frozen",
    }, "statistics contract drift")
    _require(protocol["success_contract"] == {
        "primary_point_estimate_above_zero": True,
        "primary_ci_lower_above_zero": True,
        "all_provenance_and_contract_gates_pass": True,
        "all_conditions_required": True,
    }, "success contract drift")

    gate = protocol["independent_benchmark_gate"]
    _require(gate["failure_verdict"] == BLOCKED_VERDICT, "failure verdict drift")
    _require(gate["no_dev107_fallback"] is True, "DEV107 fallback must fail closed")
    _require(gate["query_or_gt_content_read_during_readiness"] is False, "readiness leakage")
    _require(gate["required_state"] == "OPENED_FOR_CONFIRMATORY_EVALUATION", "state gate drift")
    _require(gate["required_selection_status"] == "COMPLETED", "selection gate drift")
    _require(gate["required_protocol_fingerprint_binding"] is True, "binding required")
    _require(gate["required_separate_execution_authorization"] is True, "authorization required")

    forbidden = set(protocol["prohibited_changes"])
    _require(forbidden == {
        "reranking", "rrf", "chunking", "temporal_filtering", "dynamic_fusion",
        "query_rewriting", "hyperparameter_tuning", "candidate_depth_tuning",
        "representation_tuning", "independent_query_or_gt_inspection",
    }, "prohibited-change list drift")

    fingerprint = protocol_fingerprint(protocol)
    _require(_is_sha256(protocol["integrity"]["protocol_fingerprint"]), "invalid fingerprint")
    _require(protocol["integrity"]["protocol_fingerprint"] == fingerprint, "fingerprint mismatch")
    _require(protocol["integrity"]["source_base_commit"] == EXPECTED_BASE_COMMIT, "base commit drift")
    _require(protocol["integrity"]["preregistration_branch"] == EXPECTED_BRANCH, "branch drift")
    _require(protocol["integrity"]["corpus_fingerprint"] == EXPECTED_CORPUS_FINGERPRINT, "corpus drift")
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_fingerprint": fingerprint,
        "valid": True,
        "treatment_difference": "dense_encoder_only",
    }

def assess_independent_benchmark_readiness(
    protocol: dict,
    benchmark_manifest: dict,
    sampling_manifest: dict,
    *,
    execution_authorized: bool = False,
) -> dict:
    """Inspect metadata only; never accept or load independent query/GT records."""
    validation = validate_protocol(protocol)
    missing = []
    expected_fp = validation["protocol_fingerprint"]
    counts = benchmark_manifest.get("current_counts", {})

    checks = {
        "benchmark_id": benchmark_manifest.get("benchmark_id") == "retrieval_independent_v1",
        "benchmark_state": benchmark_manifest.get("state") == "OPENED_FOR_CONFIRMATORY_EVALUATION",
        "sampling_state": sampling_manifest.get("state") in {"SEALED", "OPENED_FOR_CONFIRMATORY_EVALUATION"},
        "selection_completed": sampling_manifest.get("selection_status") == "COMPLETED",
        "sampling_seed_frozen": sampling_manifest.get("sampling_seed") is not None,
        "minimum_selected_count": counts.get("selected_cases", 0) >= benchmark_manifest.get("targets", {}).get("minimum_total", 180),
        "query_count_matches_selection": counts.get("query_records", 0) == counts.get("selected_cases", -1) and counts.get("query_records", 0) > 0,
        "gt_count_positive": counts.get("ground_truth_records", 0) > 0,
        "protocol_fingerprint_bound": benchmark_manifest.get("a2_protocol_fingerprint") == expected_fp,
        "selection_policy_frozen": bool(sampling_manifest.get("selected_case_fingerprint")),
        "eligible_frame_frozen": bool(sampling_manifest.get("eligible_fingerprint")),
        "close_gate_passed": benchmark_manifest.get("close_gate_status") == "PASSED",
        "source_provenance_attested": benchmark_manifest.get("source_provenance_status") == "VERIFIED",
        "leakage_audit_passed": benchmark_manifest.get("leakage_audit_status") == "PASSED",
        "control_model_verified": benchmark_manifest.get("a2_control_model_status") == "VERIFIED",
        "treatment_model_verified": benchmark_manifest.get("a2_treatment_model_status") == "VERIFIED",
        "parameters_unchanged": benchmark_manifest.get("a2_parameter_change_status") == "UNCHANGED",
        "execution_authorized": execution_authorized is True,
    }
    for name, passed in checks.items():
        if not passed:
            missing.append(name)
    ready = not missing
    return {
        "experiment_id": EXPERIMENT_ID,
        "protocol_fingerprint": expected_fp,
        "metadata_only": True,
        "query_or_gt_content_read": False,
        "checks": checks,
        "missing_requirements": missing,
        "ready": ready,
        "verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
    }
