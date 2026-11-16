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

from ..benchmarking.independent_benchmark import (
    BenchmarkProtocolError,
    selected_records_fingerprint,
    validate_sampling_records,
    validate_selection_guardrails,
)


EXPERIMENT_ID = "candidate_generation_dense_encoder_swap_minmax50_v1"
BLOCKED_VERDICT = "A2_CONFIRMATORY_EVALUATION_BLOCKED"
READY_VERDICT = "A2_CONFIRMATORY_EVALUATION_READY"
EXPECTED_BASE_COMMIT = "2cbdddf0647bd8c9d3ff30c6973100905f7371c7"
EXPECTED_BRANCH = "research/a2-dense-encoder-swap-prereg"
EXPECTED_SELECTION_MODE = "TARGET_ONLY"
EXPECTED_TARGET_COUNTS = {"total": 250, "main": 200, "stress": 50}
FUTURE_METADATA_INTERFACE = {
    "selection_evidence": {
        "schema_version": "independent_selection_evidence_v1",
        "authoritative_validator": "independent_benchmark.validate_selection_guardrails:target",
        "required_payload": [
            "selected_records",
            "selected_case_count",
            "slice_counts",
            "slices_disjoint",
            "selected_case_fingerprint",
            "eligible_frame_fingerprint",
            "selection_mode",
            "target_counts",
            "fallback_controls",
            "validation_status",
            "evidence_fingerprint",
        ],
        "required_manifest_bindings": [
            "benchmark.selection_evidence_fingerprint",
            "sampling.selection_evidence_fingerprint",
        ],
    },
    "final_frame_evidence": {
        "schema_version": "final_eligible_frame_evidence_v1",
        "required_payload": [
            "eligible_records",
            "eligible_fingerprint",
            "state=FINALIZED",
            "validation_status=VERIFIED",
            "source_provenance_status=VERIFIED",
            "evidence_fingerprint",
        ],
        "required_manifest_bindings": [
            "benchmark.final_frame_evidence_fingerprint",
            "benchmark.final_eligible_frame_fingerprint",
            "sampling.final_frame_evidence_fingerprint",
            "sampling.eligible_fingerprint",
        ],
    },
    "annotation_evidence": {
        "schema_version": "independent_annotation_validation_evidence_v1",
        "authoritative_validator": "independent_benchmark.validate_one_case_one_query_contract",
        "required_payload": [
            "one_query_per_selected_case=true",
            "selected_case_fingerprint",
            "query_count",
            "ground_truth_records",
            "gt_schema_version=gt_schema_v2",
            "validation_status=PASSED",
            "evidence_fingerprint",
        ],
        "required_manifest_bindings": [
            "benchmark.annotation_validation_evidence_fingerprint",
        ],
    },
}
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


def _evidence_fingerprint(evidence: dict | None) -> str | None:
    if not isinstance(evidence, dict):
        return None
    portable = copy.deepcopy(evidence)
    portable.pop("evidence_fingerprint", None)
    try:
        return canonical_json_fingerprint(portable)
    except (TypeError, ValueError):
        return None


def _valid_evidence_fingerprint(evidence: dict | None) -> bool:
    if not isinstance(evidence, dict):
        return False
    recorded = evidence.get("evidence_fingerprint")
    return _is_sha256(recorded) and recorded == _evidence_fingerprint(evidence)


def _selection_and_frame_checks(
    benchmark_manifest: dict,
    sampling_manifest: dict,
    selection_evidence: dict | None,
    final_frame_evidence: dict | None,
) -> dict[str, bool]:
    """Recompute selection invariants from records, not caller count claims."""
    checks = {
        "target_mode_target_only": False,
        "target_counts_exact": False,
        "fallbacks_disabled": False,
        "selection_evidence_bound": False,
        "selected_total_exact": False,
        "selected_slice_counts_exact": False,
        "selected_slices_disjoint": False,
        "selected_case_fingerprint_valid": False,
        "selected_cases_in_final_frame": False,
        "final_frame_evidence_bound": False,
        "final_frame_finalized": False,
        "eligible_frame_fingerprint_matches_final": False,
    }
    if not isinstance(selection_evidence, dict) or not isinstance(final_frame_evidence, dict):
        return checks

    selected = selection_evidence.get("selected_records")
    final_frame = final_frame_evidence.get("eligible_records")
    if not isinstance(selected, list) or not isinstance(final_frame, list):
        return checks
    if not all(isinstance(record, dict) for record in selected + final_frame):
        return checks

    selection_fp = selection_evidence.get("evidence_fingerprint")
    frame_evidence_fp = final_frame_evidence.get("evidence_fingerprint")
    selected_fp = None
    final_frame_fp = None
    try:
        validate_sampling_records(selected)
        guardrails = validate_selection_guardrails(selected, mode="target")
        validate_sampling_records(final_frame)
        selected_fp = selected_records_fingerprint(selected)
        final_frame_fp = canonical_json_fingerprint(final_frame)
        authoritative_selection_valid = True
    except (BenchmarkProtocolError, KeyError, TypeError, ValueError):
        guardrails = {}
        authoritative_selection_valid = False

    slice_counts = {
        "main": sum(record.get("slice") == "main" for record in selected),
        "stress": sum(record.get("slice") == "stress" for record in selected),
    }
    selected_ids = [(record.get("source_id"), record.get("case_id")) for record in selected]
    selected_disjoint = (
        authoritative_selection_valid
        and len(selected_ids) == len(set(selected_ids))
        and len({record.get("source_id") for record in selected}) == len(selected)
        and len({record.get("case_id") for record in selected}) == len(selected)
    )
    frame_by_source = {
        record.get("source_id"): record
        for record in final_frame
        if isinstance(record, dict)
    }
    selected_in_frame = authoritative_selection_valid and all(
        selected_record.get("source_id") in frame_by_source
        and all(
            selected_record.get(field) == frame_by_source[selected_record["source_id"]].get(field)
            for field in ("case_id", "source_path", "source_sha256")
        )
        for selected_record in selected
    )

    benchmark_targets = benchmark_manifest.get("targets", {})
    sampling_targets = sampling_manifest.get("targets", {})
    evidence_targets = selection_evidence.get("target_counts")
    checks.update({
        "target_mode_target_only": (
            benchmark_manifest.get("selection_mode") == EXPECTED_SELECTION_MODE
            and sampling_manifest.get("selection_mode") == EXPECTED_SELECTION_MODE
            and selection_evidence.get("selection_mode") == EXPECTED_SELECTION_MODE
        ),
        "target_counts_exact": (
            {key: benchmark_targets.get(key) for key in EXPECTED_TARGET_COUNTS}
            == EXPECTED_TARGET_COUNTS
            and {key: sampling_targets.get(key) for key in EXPECTED_TARGET_COUNTS}
            == EXPECTED_TARGET_COUNTS
            and evidence_targets == EXPECTED_TARGET_COUNTS
        ),
        "fallbacks_disabled": selection_evidence.get("fallback_controls") == {
            "fallback_authorized": False,
            "legacy_minimum_mode_authorized": False,
            "silent_quota_reduction_authorized": False,
            "proportional_quota_scaling_authorized": False,
        },
        "selection_evidence_bound": (
            _valid_evidence_fingerprint(selection_evidence)
            and benchmark_manifest.get("selection_evidence_fingerprint") == selection_fp
            and sampling_manifest.get("selection_evidence_fingerprint") == selection_fp
            and selection_evidence.get("schema_version") == "independent_selection_evidence_v1"
            and selection_evidence.get("validation_status") == "PASSED"
            and selection_evidence.get("validator")
            == "independent_benchmark.validate_selection_guardrails:target"
        ),
        "selected_total_exact": (
            authoritative_selection_valid
            and len(selected) == 250
            and guardrails.get("total") == 250
            and selection_evidence.get("selected_case_count") == 250
            and sampling_manifest.get("selected_case_count") == 250
            and benchmark_manifest.get("selected_case_count") == 250
            and benchmark_manifest.get("current_counts", {}).get("selected_cases") == 250
        ),
        "selected_slice_counts_exact": (
            authoritative_selection_valid
            and slice_counts == {"main": 200, "stress": 50}
            and guardrails.get("main") == 200
            and guardrails.get("stress") == 50
            and selection_evidence.get("slice_counts") == slice_counts
        ),
        "selected_slices_disjoint": (
            selected_disjoint
            and selection_evidence.get("slices_disjoint") is True
        ),
        "selected_case_fingerprint_valid": (
            authoritative_selection_valid
            and _is_sha256(selected_fp)
            and selection_evidence.get("selected_case_fingerprint") == selected_fp
            and sampling_manifest.get("selected_case_fingerprint") == selected_fp
            and benchmark_manifest.get("selected_case_fingerprint") == selected_fp
        ),
        "selected_cases_in_final_frame": selected_in_frame,
        "final_frame_evidence_bound": (
            _valid_evidence_fingerprint(final_frame_evidence)
            and benchmark_manifest.get("final_frame_evidence_fingerprint") == frame_evidence_fp
            and sampling_manifest.get("final_frame_evidence_fingerprint") == frame_evidence_fp
            and final_frame_evidence.get("schema_version") == "final_eligible_frame_evidence_v1"
        ),
        "final_frame_finalized": (
            final_frame_evidence.get("state") == "FINALIZED"
            and final_frame_evidence.get("validation_status") == "VERIFIED"
            and final_frame_evidence.get("source_provenance_status") == "VERIFIED"
        ),
        "eligible_frame_fingerprint_matches_final": (
            _is_sha256(final_frame_fp)
            and final_frame_evidence.get("eligible_fingerprint") == final_frame_fp
            and selection_evidence.get("eligible_frame_fingerprint") == final_frame_fp
            and sampling_manifest.get("eligible_fingerprint") == final_frame_fp
            and benchmark_manifest.get("final_eligible_frame_fingerprint") == final_frame_fp
        ),
    })
    return checks


def _annotation_checks(
    benchmark_manifest: dict,
    selected_case_fingerprint: str | None,
    annotation_evidence: dict | None,
) -> dict[str, bool]:
    checks = {
        "query_count_matches_selection": False,
        "annotation_contract_verified": False,
        "gt_count_positive": False,
    }
    if not isinstance(annotation_evidence, dict):
        return checks
    counts = benchmark_manifest.get("current_counts", {})
    evidence_fp = annotation_evidence.get("evidence_fingerprint")
    query_count = counts.get("query_records")
    selected_count = counts.get("selected_cases")
    gt_count = counts.get("ground_truth_records")
    checks.update({
        "query_count_matches_selection": (
            query_count == selected_count == 250
            and annotation_evidence.get("query_count") == 250
        ),
        "annotation_contract_verified": (
            _valid_evidence_fingerprint(annotation_evidence)
            and benchmark_manifest.get("annotation_validation_evidence_fingerprint") == evidence_fp
            and annotation_evidence.get("schema_version") == "independent_annotation_validation_evidence_v1"
            and annotation_evidence.get("validation_status") == "PASSED"
            and annotation_evidence.get("validator")
            == "independent_benchmark.validate_one_case_one_query_contract"
            and annotation_evidence.get("one_query_per_selected_case") is True
            and annotation_evidence.get("selected_case_fingerprint") == selected_case_fingerprint
            and annotation_evidence.get("gt_schema_version") == "gt_schema_v2"
        ),
        "gt_count_positive": (
            isinstance(gt_count, int)
            and not isinstance(gt_count, bool)
            and gt_count > 0
            and annotation_evidence.get("ground_truth_records") == gt_count
        ),
    })
    return checks


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
        "hypothesis",
        "scope",
        "representations",
        "candidate_generation",
        "systems",
        "evaluation",
        "coverage_policy",
        "statistics",
        "success_contract",
        "stop_conditions",
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

    _require(protocol["hypothesis"] == {
        "research_question": "Does BGE-M3 improve frozen Hybrid 0.5 candidate generation relative to Vietnamese-SBERT?",
        "null": "treatment_macro_candidate_recall_at_50_delta_is_less_than_or_equal_to_zero",
        "alternative": "treatment_macro_candidate_recall_at_50_delta_is_greater_than_zero",
        "development_selection_basis": "A1_DEV107_exploratory_missing_candidate_analysis",
        "known_selection_bias": "BGE-M3_and_depth_50_were_selected_using_repeated_DEV107_exploration",
        "independent_result_tuning_permitted": False,
    }, "hypothesis or development-bias disclosure drift")

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
    _require(evaluation["report_evaluable_query_count"] is True, "evaluable query count must be reported")
    _require(evaluation["report_in_corpus_gt_instance_count"] is True, "GT instance count must be reported")
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
    _require(protocol["coverage_policy"] == {
        "closed_world_matching": "frozen_benchmark_canonical_retrieval_id",
        "corpus_missing_gt": "report_separately_and_exclude_from_retrieval_miss_count",
        "retrieval_miss": "in_corpus_gt_absent_from_candidate_set",
        "candidate_recall_not_final_answer_accuracy": True,
        "core_support_labels": "preserve_frozen_benchmark_labels",
    }, "corpus/GT coverage policy drift")

    stats = protocol["statistics"]
    _require(stats == {
        "paired_unit": "query",
        "effect": "treatment_minus_control",
        "bootstrap_resamples": 20000,
        "random_seed": 42,
        "confidence_interval": "two_sided_95_percentile",
        "report_mean_paired_delta": True,
        "report_wins_ties_losses": True,
        "slice_reporting": "MAIN_and_STRESS_separate_only_when_frozen",
    }, "statistics contract drift")
    _require(protocol["success_contract"] == {
        "primary_point_estimate_above_zero": True,
        "primary_ci_lower_above_zero": True,
        "all_provenance_and_contract_gates_pass": True,
        "all_conditions_required": True,
    }, "success contract drift")
    _require(protocol["stop_conditions"] == {
        "readiness_gate_failure": BLOCKED_VERDICT,
        "provenance_or_contract_failure": "invalidate_run_and_publish_no_confirmatory_claim",
        "primary_success_condition_failure": "report_failure_and_stop_without_retuning",
        "secondary_metrics_role": "diagnostic_only",
        "independent_result_parameter_tuning": "prohibited",
    }, "stop conditions drift")

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
    selection_evidence: dict | None = None,
    final_frame_evidence: dict | None = None,
    annotation_evidence: dict | None = None,
    execution_authorized: bool = False,
) -> dict:
    """Inspect metadata only; never accept or load independent query/GT records."""
    validation = validate_protocol(protocol)
    missing = []
    expected_fp = validation["protocol_fingerprint"]
    checks = {
        "benchmark_id": benchmark_manifest.get("benchmark_id") == "retrieval_independent_v1",
        "benchmark_state": benchmark_manifest.get("state") == "OPENED_FOR_CONFIRMATORY_EVALUATION",
        "sampling_state": sampling_manifest.get("state") in {"SEALED", "OPENED_FOR_CONFIRMATORY_EVALUATION"},
        "selection_completed": sampling_manifest.get("selection_status") == "COMPLETED",
        "sampling_seed_frozen": (
            isinstance(sampling_manifest.get("sampling_seed"), int)
            and not isinstance(sampling_manifest.get("sampling_seed"), bool)
        ),
        "protocol_fingerprint_bound": benchmark_manifest.get("a2_protocol_fingerprint") == expected_fp,
        "close_gate_passed": benchmark_manifest.get("close_gate_status") == "PASSED",
        "source_provenance_attested": benchmark_manifest.get("source_provenance_status") == "VERIFIED",
        "leakage_audit_passed": benchmark_manifest.get("leakage_audit_status") == "PASSED",
        "control_model_verified": benchmark_manifest.get("a2_control_model_status") == "VERIFIED",
        "treatment_model_verified": benchmark_manifest.get("a2_treatment_model_status") == "VERIFIED",
        "parameters_unchanged": benchmark_manifest.get("a2_parameter_change_status") == "UNCHANGED",
        "execution_authorized": execution_authorized is True,
    }
    checks.update(_selection_and_frame_checks(
        benchmark_manifest,
        sampling_manifest,
        selection_evidence,
        final_frame_evidence,
    ))
    selected_fp = (
        selection_evidence.get("selected_case_fingerprint")
        if isinstance(selection_evidence, dict)
        else None
    )
    checks.update(_annotation_checks(
        benchmark_manifest,
        selected_fp,
        annotation_evidence,
    ))
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
        "required_future_metadata_interface": copy.deepcopy(FUTURE_METADATA_INTERFACE),
        "ready": ready,
        "verdict": READY_VERDICT if ready else BLOCKED_VERDICT,
    }
