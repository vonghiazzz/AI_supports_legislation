"""EXP-04 selective Dynamic Hybrid protocol.

This module only reuses frozen EXP-03 candidate pools.  It never builds a
retriever, loads an embedding model, or changes runtime retrieval behavior.
"""

import datetime
import hashlib
import json
import math
import platform
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..core.corpus import sha256_file
from ..core.query import normalize_query_q0
from ..evaluation.metrics import build_metric_world_report, get_task_metrics, get_weighted_recall
from ..reporting.manifest import git_provenance, safe_package_version
from ..reporting.serialization import write_exp04_artifacts
from .baseline import BASELINE_PROTOCOL_ID
from .exp03_static_fusion_cv import (
    EXP03_CANDIDATE_FINGERPRINT_ALGORITHM,
    EXP03_FOLD_ALGORITHM,
    EXP03_RANKING_TIE_BREAK,
    GT_RESULT_KEYS,
    _record_query_id,
    _relative_path,
    _source_file_provenance,
    fingerprint_candidate_pools,
    fingerprint_fold_mapping,
    fuse_cached_query,
    validate_candidate_pools,
    validate_fold_mapping,
)


EXP04_MODE = "exp04_selective_dynamic_hybrid_nested_cv"
EXP04_EXPERIMENT_ID = "exp04_selective_dynamic_hybrid_v1"
EXP04_PROTOCOL_ID = "exp04_selective_dynamic_hybrid_nested_5fold_cv_v1"
EXP04_FEATURE_VERSION = "exp04_inference_features_v1"

EXP04_SOURCE_RUN_ID = "exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23"
EXP04_CANDIDATE_PATH = (
    "backend/data/runs/"
    f"{EXP04_SOURCE_RUN_ID}/candidate_pools.json"
)
EXP04_FOLD_PATH = f"backend/data/runs/{EXP04_SOURCE_RUN_ID}/fold_manifest.json"
EXP04_SOURCE_SUMMARY_PATH = f"backend/data/runs/{EXP04_SOURCE_RUN_ID}/summary_metrics.json"
EXP04_CANDIDATE_FINGERPRINT = (
    "4e94fc54bf2dc1ec09c4fbb605daadb1ba0dfa8c7c0c10a2d40ea34e90c2c805"
)
EXP04_FOLD_FINGERPRINT = (
    "94e9f68e7113f21b24e4cfbbccb2faacac1bfb106ac35e5c2167e7f51eee7310"
)

EXP04_QUERY_COUNT = 107
EXP04_POOL_K = 30
EXP04_STORED_K = 20
EXP04_REPORTED_K = (5, 10, 20)
EXP04_PRIMARY_K = 10
EXP04_FOLD_COUNT = 5
EXP04_FOLD_SIZES = (22, 22, 21, 21, 21)
EXP04_CONTROL_WEIGHT = 0.5
EXP04_WEIGHT_GRID = tuple(value / 10.0 for value in range(11))
EXP04_LABEL_ABS_TOL = 1.0e-12
EXP04_LABEL_REL_TOL = 0.0
EXP04_C_GRID = (0.01, 0.1, 1.0)
EXP04_THRESHOLD_GRID = (0.0, 0.5, 1.0)
EXP04_BOOTSTRAP_SAMPLES = 20_000
EXP04_SIGN_FLIP_SAMPLES = 100_000
EXP04_RANDOM_SEED = 42

EXP04_ACTION_PAIRS = (
    {"id": "A", "dense_action": 0.4, "bm25_action": 0.6},
    {"id": "B", "dense_action": 0.3, "bm25_action": 0.7},
    {"id": "C", "dense_action": 0.2, "bm25_action": 0.8},
)
EXP04_SELECTION_TIE_BREAK = (
    "highest_unrounded_pooled_inner_oof_wr10",
    "fewer_total_overrides",
    "smaller_action_closeness",
    "smaller_c",
    "larger_threshold",
    "lexical_ascending_config_id",
)

FEATURE_NAMES = (
    "query_log_char_count",
    "query_log_token_count",
    "query_type_token_ratio",
    "query_digit_token_fraction",
    "query_log_legal_ref_count",
    "query_log_legal_doc_count",
    "bm25_top1_score",
    "bm25_gap_1_2_norm",
    "bm25_gap_1_5_norm",
    "bm25_score_sd_norm",
    "dense_top1_score",
    "dense_gap_1_2_norm",
    "dense_gap_1_5_norm",
    "dense_score_sd_norm",
    "top1_id_agreement",
    "top10_id_jaccard",
    "top10_reciprocal_rank_overlap",
    "top30_union_fraction",
)

LEGAL_REF_PATTERN = r"(?<!\w)(?:điều|khoản|điểm)\s+\d+[a-z]?(?!\w)"
LEGAL_DOC_PATTERN = (
    r"(?<!\w)(?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|"
    r"nghị\s+quyết|pháp\s+lệnh)(?!\w)"
)
LEGAL_REF_RE = re.compile(LEGAL_REF_PATTERN, re.IGNORECASE | re.UNICODE)
LEGAL_DOC_RE = re.compile(LEGAL_DOC_PATTERN, re.IGNORECASE | re.UNICODE)
INFERENCE_VIEW_KEYS = frozenset(("query", "bm25_pool", "dense_pool"))


def _require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"EXP-04 requires {label}={expected!r}; got {actual!r}")


def _validate_exp04_protocol(config):
    protocol = config.get("protocol", {})
    evaluation = config.get("evaluation", {})
    candidate = config.get("candidate_source", {})
    folds = config.get("fold_source", {})
    control = config.get("control", {})
    labeling = config.get("labeling", {})
    policy = config.get("policy", {})
    feature = config.get("feature_contract", {})
    preprocessing = config.get("preprocessing", {})
    model = config.get("model", {})
    tuning = config.get("tuning", {})
    metric = config.get("metric_protocol", {})
    statistics = config.get("statistics", {})

    required = (
        (protocol.get("id"), EXP04_PROTOCOL_ID, "protocol.id"),
        (protocol.get("experiment_id"), EXP04_EXPERIMENT_ID, "protocol.experiment_id"),
        (protocol.get("control_protocol"), BASELINE_PROTOCOL_ID, "protocol.control_protocol"),
        (evaluation.get("mode"), EXP04_MODE, "evaluation.mode"),
        (evaluation.get("experiment_id"), EXP04_EXPERIMENT_ID, "evaluation.experiment_id"),
        (evaluation.get("query_representation"), "q0", "query_representation"),
        (evaluation.get("document_representation"), "d0", "document_representation"),
        (evaluation.get("temporal_filter"), "none", "temporal_filter"),
        (int(evaluation.get("expected_query_count", 0)), EXP04_QUERY_COUNT, "query_count"),
        (int(evaluation.get("pool_k", 0)), EXP04_POOL_K, "pool_k"),
        (int(evaluation.get("stored_k", 0)), EXP04_STORED_K, "stored_k"),
        (tuple(evaluation.get("reported_k", [])), EXP04_REPORTED_K, "reported_k"),
        (int(evaluation.get("alpha_core", 0)), 3, "alpha_core"),
        (int(evaluation.get("alpha_support", 0)), 1, "alpha_support"),
        (int(evaluation.get("random_seed", -1)), EXP04_RANDOM_SEED, "random_seed"),
        (candidate.get("run_id"), EXP04_SOURCE_RUN_ID, "candidate run"),
        (candidate.get("path"), EXP04_CANDIDATE_PATH, "candidate path"),
        (candidate.get("fingerprint"), EXP04_CANDIDATE_FINGERPRINT, "candidate fingerprint"),
        (folds.get("path"), EXP04_FOLD_PATH, "fold path"),
        (folds.get("mapping_fingerprint"), EXP04_FOLD_FINGERPRINT, "fold fingerprint"),
        (int(folds.get("fold_count", 0)), EXP04_FOLD_COUNT, "fold_count"),
        (tuple(folds.get("fold_sizes", [])), EXP04_FOLD_SIZES, "fold_sizes"),
        (control.get("name"), "Fixed_0_5", "control name"),
        (float(control.get("w_bm25", -1)), EXP04_CONTROL_WEIGHT, "control weight"),
        (control.get("source_summary"), EXP04_SOURCE_SUMMARY_PATH, "control source summary"),
        (float(control.get("parity_tolerance", -1)), 0.0, "parity tolerance"),
        (tuple(float(x) for x in labeling.get("grid", [])), EXP04_WEIGHT_GRID, "label grid"),
        (labeling.get("objective"), {"world": "closed_world", "metric": "weighted_recall", "k": 10}, "label objective"),
        (float(labeling.get("rel_tol", -1)), EXP04_LABEL_REL_TOL, "label rel_tol"),
        (float(labeling.get("abs_tol", -1)), EXP04_LABEL_ABS_TOL, "label abs_tol"),
        (labeling.get("ambiguous_treatment"), "STAY", "ambiguous treatment"),
        (policy.get("family"), "selective_directional_override", "policy family"),
        (float(policy.get("default_w_bm25", -1)), EXP04_CONTROL_WEIGHT, "default weight"),
        (policy.get("both_pass_action"), "STAY", "both-pass action"),
        (policy.get("neither_pass_action"), "STAY", "neither-pass action"),
        (tuple(policy.get("action_pairs", [])), EXP04_ACTION_PAIRS, "action pairs"),
        (feature.get("version"), EXP04_FEATURE_VERSION, "feature version"),
        (feature.get("dtype"), "float64", "feature dtype"),
        (feature.get("legal_ref_regex"), LEGAL_REF_PATTERN, "legal reference regex"),
        (feature.get("legal_doc_regex"), LEGAL_DOC_PATTERN, "legal document regex"),
        (tuple(feature.get("regex_flags", [])), ("IGNORECASE", "UNICODE"), "regex flags"),
        (tuple(feature.get("ordered_features", [])), FEATURE_NAMES, "ordered feature list"),
        (preprocessing, {"scaler": "StandardScaler", "with_mean": True, "with_std": True, "missing_values": "error"}, "preprocessing"),
        (model.get("class"), "LogisticRegression", "model class"),
        (model.get("penalty"), "l2", "model penalty"),
        (model.get("solver"), "liblinear", "model solver"),
        (model.get("dual"), False, "model dual"),
        (float(model.get("tol", -1)), 1.0e-4, "model tol"),
        (model.get("fit_intercept"), True, "fit_intercept"),
        (float(model.get("intercept_scaling", -1)), 1.0, "intercept_scaling"),
        (model.get("class_weight"), "balanced", "class_weight"),
        (int(model.get("random_state", -1)), 42, "model random_state"),
        (int(model.get("max_iter", 0)), 1000, "max_iter"),
        (int(model.get("verbose", -1)), 0, "verbose"),
        (model.get("warm_start"), False, "warm_start"),
        (int(model.get("n_jobs", 0)), 1, "n_jobs"),
        (model.get("score_method"), "decision_function", "score method"),
        (model.get("probability_calibration"), False, "probability calibration"),
        (tuple(float(x) for x in tuning.get("c_grid", [])), EXP04_C_GRID, "C grid"),
        (tuple(float(x) for x in tuning.get("threshold_grid", [])), EXP04_THRESHOLD_GRID, "threshold grid"),
        (int(tuning.get("config_count", 0)), 27, "config count"),
        (tuple(tuning.get("tie_break", [])), EXP04_SELECTION_TIE_BREAK, "selection tie-break"),
        (metric.get("primary_world"), "closed_world", "primary world"),
        (metric.get("primary_metric"), "weighted_recall", "primary metric"),
        (int(metric.get("primary_k", 0)), 10, "primary k"),
        (tuple(metric.get("secondary_metrics", [])), ("recall", "ndcg", "mrr"), "secondary metrics"),
        (tuple(metric.get("reported_k", [])), EXP04_REPORTED_K, "metric reported_k"),
        (int(statistics.get("bootstrap_samples", 0)), EXP04_BOOTSTRAP_SAMPLES, "bootstrap samples"),
        (int(statistics.get("bootstrap_seed", -1)), 42, "bootstrap seed"),
        (tuple(float(x) for x in statistics.get("bootstrap_ci_percentiles", [])), (2.5, 97.5), "bootstrap percentiles"),
        (int(statistics.get("sign_flip_samples", 0)), EXP04_SIGN_FLIP_SAMPLES, "sign-flip samples"),
        (int(statistics.get("sign_flip_seed", -1)), 42, "sign-flip seed"),
        (statistics.get("sign_flip_two_sided"), True, "sign-flip sidedness"),
    )
    for actual, expected, label in required:
        _require_equal(actual, expected, label)
    return {
        "protocol_id": EXP04_PROTOCOL_ID,
        "experiment_id": EXP04_EXPERIMENT_ID,
        "mode": EXP04_MODE,
        "query_count": EXP04_QUERY_COUNT,
        "pool_k": EXP04_POOL_K,
        "stored_k": EXP04_STORED_K,
        "reported_k": EXP04_REPORTED_K,
        "fold_count": EXP04_FOLD_COUNT,
        "fold_sizes": EXP04_FOLD_SIZES,
        "c_grid": EXP04_C_GRID,
        "threshold_grid": EXP04_THRESHOLD_GRID,
        "action_pairs": EXP04_ACTION_PAIRS,
    }


def make_inference_view(record):
    missing = INFERENCE_VIEW_KEYS - set(record)
    if missing:
        raise ValueError(f"Inference record is missing required fields: {sorted(missing)}")
    return {key: record[key] for key in ("query", "bm25_pool", "dense_pool")}


def _validate_pool(pool, name):
    if not isinstance(pool, list) or len(pool) != EXP04_POOL_K:
        raise ValueError(f"{name} must contain exactly {EXP04_POOL_K} records")
    ranks = [item.get("rank") for item in pool]
    if ranks != list(range(1, EXP04_POOL_K + 1)):
        raise ValueError(f"{name} ranks must be exactly 1..{EXP04_POOL_K}")
    identifiers = [item.get("standard_law_id") for item in pool]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError(f"{name} candidate IDs must be non-empty strings")
    if len(set(identifiers)) != EXP04_POOL_K:
        raise ValueError(f"{name} candidate IDs must be unique")
    try:
        scores = [float(item["score"]) for item in pool]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} scores must be numeric") from exc
    if not all(math.isfinite(score) for score in scores):
        raise ValueError(f"{name} scores must be finite")
    return scores


def validate_inference_view(view):
    if set(view) != INFERENCE_VIEW_KEYS:
        extras = sorted(set(view) - INFERENCE_VIEW_KEYS)
        missing = sorted(INFERENCE_VIEW_KEYS - set(view))
        raise ValueError(
            "Feature extraction requires the exact inference-only view; "
            f"missing={missing}, unexpected={extras}"
        )
    if not isinstance(view["query"], str):
        raise ValueError("Inference query must be present as a string")
    bm25_scores = _validate_pool(view["bm25_pool"], "bm25_pool")
    dense_scores = _validate_pool(view["dense_pool"], "dense_pool")
    return bm25_scores, dense_scores


def _normalized_score_features(scores):
    score_range = max(scores) - min(scores)
    if score_range == 0.0:
        return float(scores[0]), 0.0, 0.0, 0.0
    return (
        float(scores[0]),
        float((scores[0] - scores[1]) / score_range),
        float((scores[0] - scores[4]) / score_range),
        float(np.std(np.asarray(scores, dtype=np.float64), ddof=0) / score_range),
    )


def extract_feature_vector(view):
    bm25_scores, dense_scores = validate_inference_view(view)
    query = normalize_query_q0(view["query"])
    tokens = query.split()
    token_count = len(tokens)
    type_token_ratio = (
        len({token.casefold() for token in tokens}) / token_count if token_count else 0.0
    )
    digit_fraction = (
        sum(any(char.isdecimal() for char in token) for token in tokens) / token_count
        if token_count
        else 0.0
    )
    bm25_features = _normalized_score_features(bm25_scores)
    dense_features = _normalized_score_features(dense_scores)
    bm25_ids = [item["standard_law_id"] for item in view["bm25_pool"]]
    dense_ids = [item["standard_law_id"] for item in view["dense_pool"]]
    bm25_top10 = set(bm25_ids[:10])
    dense_top10 = set(dense_ids[:10])
    intersection = bm25_top10 & dense_top10
    union = bm25_top10 | dense_top10
    bm25_ranks = {identifier: rank for rank, identifier in enumerate(bm25_ids[:10], 1)}
    dense_ranks = {identifier: rank for rank, identifier in enumerate(dense_ids[:10], 1)}
    harmonic_10 = sum(1.0 / rank for rank in range(1, 11))
    reciprocal_overlap = (
        (2.0 / harmonic_10)
        * sum(
            1.0 / (bm25_ranks[identifier] + dense_ranks[identifier])
            for identifier in intersection
        )
        if intersection
        else 0.0
    )
    values = (
        math.log1p(len(query)),
        math.log1p(token_count),
        type_token_ratio,
        digit_fraction,
        math.log1p(sum(1 for _ in LEGAL_REF_RE.finditer(query))),
        math.log1p(sum(1 for _ in LEGAL_DOC_RE.finditer(query))),
        *bm25_features,
        *dense_features,
        1.0 if bm25_ids[0] == dense_ids[0] else 0.0,
        len(intersection) / len(union) if union else 0.0,
        reciprocal_overlap,
        len(set(bm25_ids) | set(dense_ids)) / 60.0,
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("EXP-04 feature vector has unexpected dimensionality")
    if not np.isfinite(vector).all():
        raise ValueError("EXP-04 derived features must all be finite")
    return vector


def extract_feature_matrix(inference_views):
    matrix = np.vstack([extract_feature_vector(view) for view in inference_views]).astype(
        np.float64, copy=False
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("EXP-04 feature matrix violates the frozen feature contract")
    return matrix


def _metric_result(record, ranking, configuration="Candidate"):
    result = {"query_id": _record_query_id(record), "metrics": {configuration: {"top_k": ranking}}}
    for key in GT_RESULT_KEYS:
        if key in record:
            result[key] = record[key]
    return result


def query_weighted_recall(record, ranking, k=EXP04_PRIMARY_K):
    result = _metric_result(record, ranking)
    return get_weighted_recall(
        [result],
        "Candidate",
        k,
        is_filtered=True,
        alpha_core=3,
        alpha_support=1,
    )


def construct_training_label(record):
    values = []
    for weight in EXP04_WEIGHT_GRID:
        ranking = fuse_cached_query(record, weight, stored_k=EXP04_PRIMARY_K)
        values.append((weight, query_weighted_recall(record, ranking)))
    maximum = max(value for _, value in values)
    best_set = tuple(
        weight
        for weight, value in values
        if math.isclose(
            value,
            maximum,
            rel_tol=EXP04_LABEL_REL_TOL,
            abs_tol=EXP04_LABEL_ABS_TOL,
        )
    )
    if EXP04_CONTROL_WEIGHT in best_set:
        oracle_label = "STAY"
        ambiguous = False
    elif max(best_set) < EXP04_CONTROL_WEIGHT:
        oracle_label = "DENSE_HEAVY"
        ambiguous = False
    elif min(best_set) > EXP04_CONTROL_WEIGHT:
        oracle_label = "BM25_HEAVY"
        ambiguous = False
    elif min(best_set) < EXP04_CONTROL_WEIGHT < max(best_set):
        oracle_label = "AMBIGUOUS"
        ambiguous = True
    else:
        raise RuntimeError("Oracle best set does not match a frozen EXP-04 label")
    training_label = "STAY" if ambiguous else oracle_label
    return {
        "query_id": _record_query_id(record),
        "training_label": training_label,
        "oracle_label": oracle_label,
        "ambiguous": ambiguous,
        "best_set": list(best_set),
        "maximum_wr10": maximum,
    }


def construct_training_labels(records):
    return [construct_training_label(record) for record in records]


@dataclass
class DirectionalModel:
    name: str
    model: object
    disabled: bool
    positive_count: int
    negative_count: int
    disable_reason: str | None = None


@dataclass
class RouterFit:
    scaler: StandardScaler
    dense: DirectionalModel
    bm25: DirectionalModel
    c_value: float
    label_records: list


def _new_logistic_regression(c_value):
    return LogisticRegression(
        penalty="l2",
        solver="liblinear",
        dual=False,
        tol=1.0e-4,
        C=float(c_value),
        fit_intercept=True,
        intercept_scaling=1.0,
        class_weight="balanced",
        random_state=EXP04_RANDOM_SEED,
        max_iter=1000,
        verbose=0,
        warm_start=False,
        n_jobs=1,
    )


def fit_directional_model(name, matrix, target, c_value):
    target = np.asarray(target, dtype=np.int64)
    positive_count = int(target.sum())
    negative_count = int(target.size - positive_count)
    if negative_count == 0:
        raise ValueError(f"{name} directional target has zero negatives")
    if positive_count == 0:
        return DirectionalModel(
            name=name,
            model=None,
            disabled=True,
            positive_count=0,
            negative_count=negative_count,
            disable_reason="zero_positive_training_examples",
        )
    model = _new_logistic_regression(c_value)
    model.fit(matrix, target)
    return DirectionalModel(
        name=name,
        model=model,
        disabled=False,
        positive_count=positive_count,
        negative_count=negative_count,
    )


def fit_router(training_records, c_value, label_builder=construct_training_labels):
    if not training_records:
        raise ValueError("Router fitting requires non-empty training records")
    views = [make_inference_view(record) for record in training_records]
    matrix = extract_feature_matrix(views)
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(matrix)
    labels = label_builder(training_records)
    if len(labels) != len(training_records):
        raise ValueError("Training label count must match the training feature count")
    dense_target = [item["training_label"] == "DENSE_HEAVY" for item in labels]
    bm25_target = [item["training_label"] == "BM25_HEAVY" for item in labels]
    return RouterFit(
        scaler=scaler,
        dense=fit_directional_model("DENSE_DIRECTION", scaled, dense_target, c_value),
        bm25=fit_directional_model("BM25_DIRECTION", scaled, bm25_target, c_value),
        c_value=float(c_value),
        label_records=labels,
    )


def _direction_score(direction, scaled_vector):
    if direction.disabled:
        return float("-inf")
    return float(direction.model.decision_function(scaled_vector.reshape(1, -1))[0])


def score_inference_view(router, view):
    raw_vector = extract_feature_vector(view)
    scaled_vector = router.scaler.transform(raw_vector.reshape(1, -1))[0]
    if not np.isfinite(scaled_vector).all():
        raise ValueError("Scaled inference features must be finite")
    return {
        "feature_vector": raw_vector.tolist(),
        "dense_score": _direction_score(router.dense, scaled_vector),
        "bm25_score": _direction_score(router.bm25, scaled_vector),
        "dense_disabled": router.dense.disabled,
        "bm25_disabled": router.bm25.disabled,
    }


def route_from_scores(dense_score, bm25_score, threshold, action_pair):
    dense_pass = float(dense_score) >= float(threshold)
    bm25_pass = float(bm25_score) >= float(threshold)
    if dense_pass and not bm25_pass:
        route = "DENSE_DIRECTION"
        weight = float(action_pair["dense_action"])
    elif bm25_pass and not dense_pass:
        route = "BM25_DIRECTION"
        weight = float(action_pair["bm25_action"])
    else:
        route = "STAY"
        weight = EXP04_CONTROL_WEIGHT
    return {
        "route": route,
        "w_bm25": weight,
        "dense_pass": dense_pass,
        "bm25_pass": bm25_pass,
    }


def deterministic_rule_weight(view):
    features = extract_feature_vector(view)
    bm25_gap = float(features[FEATURE_NAMES.index("bm25_gap_1_2_norm")])
    dense_gap = float(features[FEATURE_NAMES.index("dense_gap_1_2_norm")])
    top1_agreement = bool(features[FEATURE_NAMES.index("top1_id_agreement")])
    if top1_agreement:
        return EXP04_CONTROL_WEIGHT
    if dense_gap > bm25_gap:
        return 0.4
    if bm25_gap > dense_gap:
        return 0.6
    return EXP04_CONTROL_WEIGHT


def action_closeness(action_pair):
    return abs(float(action_pair["dense_action"]) - 0.5) + abs(
        float(action_pair["bm25_action"]) - 0.5
    )


def _c_code(c_value):
    return {0.01: "C001", 0.1: "C010", 1.0: "C100"}[float(c_value)]


def _threshold_code(threshold):
    return {0.0: "T000", 0.5: "T050", 1.0: "T100"}[float(threshold)]


def canonical_config_id(action_pair, c_value, threshold):
    return f"{action_pair['id']}__{_c_code(c_value)}__{_threshold_code(threshold)}"


def enumerate_inner_configs():
    configs = []
    for c_value in EXP04_C_GRID:
        for threshold in EXP04_THRESHOLD_GRID:
            for action_pair in EXP04_ACTION_PAIRS:
                configs.append(
                    {
                        "config_id": canonical_config_id(action_pair, c_value, threshold),
                        "C": c_value,
                        "threshold": threshold,
                        "action_pair": dict(action_pair),
                        "action_closeness": action_closeness(action_pair),
                    }
                )
    if len(configs) != 27 or len({item["config_id"] for item in configs}) != 27:
        raise RuntimeError("EXP-04 requires exactly 27 unique inner configurations")
    return configs


def validate_train_validation_split(training_ids, validation_ids, expected_ids):
    training = set(training_ids)
    validation = set(validation_ids)
    expected = set(expected_ids)
    if training & validation:
        raise ValueError("Training and validation query IDs overlap")
    if training | validation != expected:
        raise ValueError("Training and validation IDs do not cover the expected universe")


def build_inner_oof_scores(outer_training_records, inner_fold_mapping, c_value):
    record_by_id = {_record_query_id(record): record for record in outer_training_records}
    if len(record_by_id) != len(outer_training_records):
        raise ValueError("Outer-training records must have unique query IDs")
    if set(inner_fold_mapping) != set(record_by_id):
        raise ValueError("Inner fold mapping must cover exactly the outer-training records")
    inner_folds = sorted(set(inner_fold_mapping.values()))
    if len(inner_folds) != EXP04_FOLD_COUNT - 1:
        raise ValueError("Each outer fold requires exactly four inner validation blocks")

    scores = []
    split_diagnostics = []
    expected_ids = set(record_by_id)
    for validation_fold in inner_folds:
        validation_ids = sorted(
            query_id
            for query_id, fold in inner_fold_mapping.items()
            if fold == validation_fold
        )
        training_ids = sorted(expected_ids - set(validation_ids))
        validate_train_validation_split(training_ids, validation_ids, expected_ids)
        router = fit_router([record_by_id[query_id] for query_id in training_ids], c_value)
        for query_id in validation_ids:
            score = score_inference_view(router, make_inference_view(record_by_id[query_id]))
            scores.append(
                {
                    "query_id": query_id,
                    "inner_validation_fold": validation_fold,
                    **score,
                }
            )
        split_diagnostics.append(
            {
                "inner_validation_fold": validation_fold,
                "training_query_ids": training_ids,
                "validation_query_ids": validation_ids,
                "dense_positive_count": router.dense.positive_count,
                "dense_negative_count": router.dense.negative_count,
                "dense_disabled": router.dense.disabled,
                "dense_disable_reason": router.dense.disable_reason,
                "bm25_positive_count": router.bm25.positive_count,
                "bm25_negative_count": router.bm25.negative_count,
                "bm25_disabled": router.bm25.disabled,
                "bm25_disable_reason": router.bm25.disable_reason,
                "ambiguous_count": sum(item["ambiguous"] for item in router.label_records),
            }
        )
    scores.sort(key=lambda item: item["query_id"])
    actual_ids = [item["query_id"] for item in scores]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != expected_ids:
        raise RuntimeError("Inner OOF scoring requires exactly one score pair per query")
    return scores, split_diagnostics


def _results_for_weighted_recall(records, rankings, configuration="Policy"):
    return [
        _metric_result(record, rankings[_record_query_id(record)], configuration)
        for record in records
    ]


def evaluate_inner_configuration(outer_training_records, inner_scores, config):
    record_by_id = {_record_query_id(record): record for record in outer_training_records}
    inference_by_id = {
        query_id: make_inference_view(record) for query_id, record in record_by_id.items()
    }
    rankings = {}
    override_count = 0
    routes = []
    for score in inner_scores:
        query_id = score["query_id"]
        decision = route_from_scores(
            score["dense_score"],
            score["bm25_score"],
            config["threshold"],
            config["action_pair"],
        )
        if decision["route"] != "STAY":
            override_count += 1
        rankings[query_id] = fuse_cached_query(
            inference_by_id[query_id], decision["w_bm25"], stored_k=EXP04_STORED_K
        )
        routes.append({"query_id": query_id, **decision})
    results = _results_for_weighted_recall(outer_training_records, rankings)
    pooled_wr10 = get_weighted_recall(
        results,
        "Policy",
        EXP04_PRIMARY_K,
        is_filtered=True,
        alpha_core=3,
        alpha_support=1,
    )
    return {
        **config,
        "pooled_inner_oof_wr10": pooled_wr10,
        "override_count": override_count,
        "routes": routes,
    }


def choose_inner_configuration(surface):
    if not surface:
        raise ValueError("Inner selection requires a non-empty configuration surface")
    return min(
        surface,
        key=lambda item: (
            -item["pooled_inner_oof_wr10"],
            item["override_count"],
            item["action_closeness"],
            item["C"],
            -item["threshold"],
            item["config_id"],
        ),
    )


def select_from_frozen_inner_scores(outer_training_records, scores_by_c):
    surface = []
    for config in enumerate_inner_configs():
        evaluated = evaluate_inner_configuration(
            outer_training_records, scores_by_c[config["C"]], config
        )
        surface.append(evaluated)
    return {"selected": choose_inner_configuration(surface), "surface": surface}


def select_inner_configuration(outer_training_records, inner_fold_mapping):
    scores_by_c = {}
    diagnostics_by_c = {}
    for c_value in EXP04_C_GRID:
        scores, diagnostics = build_inner_oof_scores(
            outer_training_records, inner_fold_mapping, c_value
        )
        scores_by_c[c_value] = scores
        diagnostics_by_c[c_value] = diagnostics

    selection = select_from_frozen_inner_scores(outer_training_records, scores_by_c)
    return {
        **selection,
        "scores_by_c": scores_by_c,
        "split_diagnostics_by_c": diagnostics_by_c,
    }


def freeze_outer_predictions(router, outer_test_inference, selected_config, outer_fold):
    predictions = []
    for query_id, view in outer_test_inference:
        score = score_inference_view(router, view)
        decision = route_from_scores(
            score["dense_score"],
            score["bm25_score"],
            selected_config["threshold"],
            selected_config["action_pair"],
        )
        dynamic_ranking = fuse_cached_query(view, decision["w_bm25"], EXP04_STORED_K)
        fixed_ranking = fuse_cached_query(view, EXP04_CONTROL_WEIGHT, EXP04_STORED_K)
        rule_weight = deterministic_rule_weight(view)
        rule_ranking = fuse_cached_query(view, rule_weight, EXP04_STORED_K)
        serialized_score = {
            **score,
            "dense_score": None if score["dense_disabled"] else score["dense_score"],
            "bm25_score": None if score["bm25_disabled"] else score["bm25_score"],
        }
        predictions.append(
            {
                "query_id": query_id,
                "outer_fold": outer_fold,
                "selected_config_id": selected_config["config_id"],
                "selected_C": selected_config["C"],
                "selected_threshold": selected_config["threshold"],
                "selected_action_pair": selected_config["action_pair"],
                **serialized_score,
                **decision,
                "dynamic_top_k": dynamic_ranking,
                "fixed_0_5_top_k": fixed_ranking,
                "rule_w_bm25": rule_weight,
                "rule_top_k": rule_ranking,
            }
        )
    predictions.sort(key=lambda item: item["query_id"])
    fingerprint_payload = json.dumps(
        predictions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return predictions, hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()


def _evaluation_result(record, prediction):
    result = {
        "query_id": _record_query_id(record),
        "query": record.get("query", ""),
        "outer_fold": prediction["outer_fold"],
        "selected_config_id": prediction["selected_config_id"],
        "route": prediction["route"],
        "w_bm25": prediction["w_bm25"],
        "dense_score": prediction["dense_score"],
        "bm25_score": prediction["bm25_score"],
        "dense_pass": prediction["dense_pass"],
        "bm25_pass": prediction["bm25_pass"],
        "feature_vector": prediction["feature_vector"],
    }
    for key in GT_RESULT_KEYS:
        if key in record:
            result[key] = record[key]
    result["metrics"] = {
        "Dynamic_OOF": {"top_k": prediction["dynamic_top_k"]},
        "Fixed_0_5": {"top_k": prediction["fixed_0_5_top_k"]},
        "Rule_Baseline": {"top_k": prediction["rule_top_k"]},
    }
    return result


def _serialize_direction(direction):
    payload = {
        "disabled": direction.disabled,
        "positive_count": direction.positive_count,
        "negative_count": direction.negative_count,
        "disable_reason": direction.disable_reason,
    }
    if not direction.disabled:
        payload.update(
            {
                "classes": direction.model.classes_.tolist(),
                "coefficients": direction.model.coef_.tolist(),
                "intercept": direction.model.intercept_.tolist(),
                "n_iter": direction.model.n_iter_.tolist(),
                "converged": bool(np.all(direction.model.n_iter_ < direction.model.max_iter)),
            }
        )
    return payload


def build_outer_fold_result(record_by_id, fold_mapping, outer_fold):
    test_ids = sorted(query_id for query_id, fold in fold_mapping.items() if fold == outer_fold)
    training_ids = sorted(set(record_by_id) - set(test_ids))
    validate_train_validation_split(training_ids, test_ids, record_by_id)
    outer_training = [record_by_id[query_id] for query_id in training_ids]
    inner_mapping = {query_id: fold_mapping[query_id] for query_id in training_ids}
    selection = select_inner_configuration(outer_training, inner_mapping)
    selected = selection["selected"]
    router = fit_router(outer_training, selected["C"])

    inference_only_test = [
        (query_id, make_inference_view(record_by_id[query_id])) for query_id in test_ids
    ]
    frozen, prediction_fingerprint = freeze_outer_predictions(
        router, inference_only_test, selected, outer_fold
    )

    # Ground truth is attached only after the outer prediction payload is frozen.
    evaluation_results = [
        _evaluation_result(record_by_id[prediction["query_id"]], prediction)
        for prediction in frozen
    ]
    fold_result = {
        "outer_fold": outer_fold,
        "training_query_ids": training_ids,
        "heldout_query_ids": test_ids,
        "selected_config": {
            key: selected[key]
            for key in (
                "config_id",
                "C",
                "threshold",
                "action_pair",
                "action_closeness",
                "pooled_inner_oof_wr10",
                "override_count",
            )
        },
        "inner_selection_surface": [
            {key: item[key] for key in item if key != "routes"}
            for item in selection["surface"]
        ],
        "inner_split_diagnostics_by_c": selection["split_diagnostics_by_c"],
        "outer_training_labels": router.label_records,
        "scaler": {
            "mean": router.scaler.mean_.tolist(),
            "scale": router.scaler.scale_.tolist(),
        },
        "dense_model": _serialize_direction(router.dense),
        "bm25_model": _serialize_direction(router.bm25),
        "prediction_fingerprint": prediction_fingerprint,
    }
    return frozen, evaluation_results, fold_result


def build_nested_oof_results(records, fold_mapping):
    record_by_id = {_record_query_id(record): record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("EXP-04 candidate records must have unique query IDs")
    validate_fold_mapping(fold_mapping, record_by_id, EXP04_FOLD_COUNT)
    predictions = []
    evaluation_results = []
    fold_results = []
    for outer_fold in range(EXP04_FOLD_COUNT):
        frozen, evaluated, fold_result = build_outer_fold_result(
            record_by_id, fold_mapping, outer_fold
        )
        predictions.extend(frozen)
        evaluation_results.extend(evaluated)
        fold_results.append(fold_result)
    predictions.sort(key=lambda item: item["query_id"])
    evaluation_results.sort(key=lambda item: item["query_id"])
    actual_ids = [item["query_id"] for item in predictions]
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(record_by_id):
        raise RuntimeError("EXP-04 requires exactly one outer OOF prediction per query")
    return predictions, evaluation_results, fold_results


def query_metric_values(result, configuration, k=EXP04_PRIMARY_K):
    recall, ndcg, mrr, valid = get_task_metrics(
        [result], configuration, k, is_filtered=True, task_type="overall"
    )
    weighted_recall = get_weighted_recall(
        [result], configuration, k, is_filtered=True, alpha_core=3, alpha_support=1
    )
    return {
        "recall": recall,
        "ndcg": ndcg,
        "mrr": mrr,
        "weighted_recall": weighted_recall,
        "valid_queries": valid,
    }


def paired_bootstrap(deltas, samples=EXP04_BOOTSTRAP_SAMPLES, seed=42):
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Paired bootstrap requires a finite non-empty delta vector")
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    chunk_size = 2_000
    for start in range(0, int(samples), chunk_size):
        size = min(chunk_size, int(samples) - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        means[start : start + size] = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5))
    return {
        "samples": int(samples),
        "seed": int(seed),
        "mean_delta": float(values.mean()),
        "ci_95_percentile": [float(lower), float(upper)],
    }


def paired_sign_flip(deltas, samples=EXP04_SIGN_FLIP_SAMPLES, seed=42):
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Paired sign-flip requires a finite non-empty delta vector")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme_count = 0
    chunk_size = 5_000
    for start in range(0, int(samples), chunk_size):
        size = min(chunk_size, int(samples) - start)
        signs = rng.integers(0, 2, size=(size, values.size), dtype=np.int8)
        signs = signs * 2 - 1
        flipped = np.abs((signs * values).mean(axis=1))
        extreme_count += int(np.count_nonzero(flipped >= observed))
    return {
        "samples": int(samples),
        "seed": int(seed),
        "observed_absolute_mean": observed,
        "extreme_count": extreme_count,
        "p_value": (1 + extreme_count) / (int(samples) + 1),
        "two_sided": True,
    }


def false_override_report(evaluation_results):
    report = {
        "route_count": 0,
        "stay_count": 0,
        "dense_override_count": 0,
        "bm25_override_count": 0,
        "beneficial_overrides": 0,
        "tied_overrides": 0,
        "harmful_overrides": 0,
        "total_positive_gain": 0.0,
        "total_negative_harm": 0.0,
    }
    query_deltas = []
    for result in evaluation_results:
        dynamic = query_metric_values(result, "Dynamic_OOF")["weighted_recall"]
        fixed = query_metric_values(result, "Fixed_0_5")["weighted_recall"]
        delta = dynamic - fixed
        query_deltas.append({"query_id": result["query_id"], "delta": delta})
        if result["route"] == "STAY":
            report["stay_count"] += 1
            continue
        report["route_count"] += 1
        if result["route"] == "DENSE_DIRECTION":
            report["dense_override_count"] += 1
        else:
            report["bm25_override_count"] += 1
        if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=EXP04_LABEL_ABS_TOL):
            report["tied_overrides"] += 1
        elif delta > 0.0:
            report["beneficial_overrides"] += 1
            report["total_positive_gain"] += delta
        else:
            report["harmful_overrides"] += 1
            report["total_negative_harm"] += -delta
    benefit = report["total_positive_gain"]
    harm = report["total_negative_harm"]
    if benefit == 0.0 and harm == 0.0:
        ratio = 0.0
    elif benefit == 0.0:
        ratio = math.inf
    else:
        ratio = harm / benefit
    report["harm_benefit_ratio"] = ratio
    report["query_wr10_deltas"] = query_deltas
    return report


def build_statistical_analysis(evaluation_results):
    metric_deltas = {name: [] for name in ("weighted_recall", "recall", "ndcg", "mrr")}
    per_query = []
    for result in evaluation_results:
        dynamic = query_metric_values(result, "Dynamic_OOF")
        fixed = query_metric_values(result, "Fixed_0_5")
        deltas = {name: dynamic[name] - fixed[name] for name in metric_deltas}
        for name, value in deltas.items():
            metric_deltas[name].append(value)
        per_query.append(
            {
                "query_id": result["query_id"],
                "outer_fold": result["outer_fold"],
                "deltas": deltas,
            }
        )
    metrics = {}
    for name, deltas in metric_deltas.items():
        metrics[name] = {
            "paired_bootstrap": paired_bootstrap(deltas),
            "paired_sign_flip": paired_sign_flip(deltas),
        }
    return {"metrics": metrics, "per_query": per_query}


def evaluate_success_contract(statistical_analysis, override_report):
    primary_rows = statistical_analysis["per_query"]
    primary_deltas = [row["deltas"]["weighted_recall"] for row in primary_rows]
    primary = statistical_analysis["metrics"]["weighted_recall"]
    positive = [delta for delta in primary_deltas if delta > 0.0]
    total_after_largest = sum(primary_deltas) - (max(positive) if positive else 0.0)
    fold_means = {}
    for fold in range(EXP04_FOLD_COUNT):
        values = [
            row["deltas"]["weighted_recall"]
            for row in primary_rows
            if row["outer_fold"] == fold
        ]
        fold_means[str(fold)] = float(np.mean(values)) if values else 0.0
    secondary_no_collapse = all(
        statistical_analysis["metrics"][name]["paired_bootstrap"][
            "ci_95_percentile"
        ][1]
        >= 0.0
        for name in ("recall", "ndcg", "mrr")
    )
    conditions = {
        "dynamic_wr10_above_fixed": float(np.mean(primary_deltas)) > 0.0,
        "bootstrap_lower_above_zero": primary["paired_bootstrap"][
            "ci_95_percentile"
        ][0]
        > 0.0,
        "sign_flip_p_below_0_05": primary["paired_sign_flip"]["p_value"] < 0.05,
        "nonzero_routing": override_report["route_count"] > 0,
        "benefit_exceeds_harm": override_report["total_positive_gain"]
        > override_report["total_negative_harm"],
        "positive_after_removing_largest_gain": total_after_largest > 0.0,
        "positive_in_at_least_three_folds": sum(value > 0.0 for value in fold_means.values())
        >= 3,
        "no_serious_secondary_collapse": secondary_no_collapse,
    }
    return {
        "conditions": conditions,
        "all_conditions_hold": all(conditions.values()),
        "decision": "EXPLORATORY_SUCCESS" if all(conditions.values()) else "DO_NOT_PROMOTE",
        "result_scope": "exploratory",
        "total_delta_after_removing_largest_positive_query": total_after_largest,
        "fold_mean_wr10_deltas": fold_means,
    }


def validate_fixed_control_parity(metric_worlds, source_summary):
    maximum_error = 0.0
    for world in ("closed_world", "open_world"):
        for k in EXP04_REPORTED_K:
            actual = metric_worlds[world]["Fixed_0_5"][str(k)]
            expected = source_summary[world]["Control_Fixed_0_5"][str(k)]
            for metric in ("recall", "ndcg", "mrr", "weighted_recall"):
                error = abs(float(actual[metric]) - float(expected[metric]))
                maximum_error = max(maximum_error, error)
                if error != 0.0:
                    raise RuntimeError(
                        f"EXP-04 fixed-0.5 parity failed for {world} {metric}@{k}: {error}"
                    )
            if actual["valid_queries"] != expected["valid_queries"]:
                raise RuntimeError(
                    f"EXP-04 fixed-0.5 parity failed for {world} valid_queries@{k}"
                )
    return {"passed": True, "maximum_metric_error": maximum_error}


def feature_contract_payload():
    return {
        "version": EXP04_FEATURE_VERSION,
        "dtype": "float64",
        "ordered_features": list(FEATURE_NAMES),
        "legal_ref_regex": LEGAL_REF_PATTERN,
        "legal_doc_regex": LEGAL_DOC_PATTERN,
        "regex_flags": ["IGNORECASE", "UNICODE"],
        "input_fields": ["query", "bm25_pool", "dense_pool"],
        "pool_k": EXP04_POOL_K,
        "formulas": {
            "query_log_char_count": "log1p(len(normalize_query_q0(query)))",
            "query_log_token_count": "log1p(len(q.split()))",
            "query_type_token_ratio": "unique_casefold_tokens/token_count; zero_if_empty",
            "query_digit_token_fraction": "tokens_with_any_char_isdecimal/token_count; zero_if_empty",
            "query_log_legal_ref_count": "log1p(non_overlapping_LEGAL_REF_RE_matches)",
            "query_log_legal_doc_count": "log1p(non_overlapping_LEGAL_DOC_RE_matches)",
            "bm25_top1_score": "raw_rank_1_score",
            "bm25_gap_1_2_norm": "(s1-s2)/(max30-min30); zero_if_range_zero",
            "bm25_gap_1_5_norm": "(s1-s5)/(max30-min30); zero_if_range_zero",
            "bm25_score_sd_norm": "population_sd30/(max30-min30); zero_if_range_zero",
            "dense_top1_score": "raw_rank_1_score",
            "dense_gap_1_2_norm": "(s1-s2)/(max30-min30); zero_if_range_zero",
            "dense_gap_1_5_norm": "(s1-s5)/(max30-min30); zero_if_range_zero",
            "dense_score_sd_norm": "population_sd30/(max30-min30); zero_if_range_zero",
            "top1_id_agreement": "float(bm25_rank1_id == dense_rank1_id)",
            "top10_id_jaccard": "len(B10_intersection_D10)/len(B10_union_D10)",
            "top10_reciprocal_rank_overlap": "(2/H10)*sum(1/(bm25_rank+dense_rank) for intersection)",
            "top30_union_fraction": "len(B30_union_D30)/60.0",
        },
        "scaler": {"class": "StandardScaler", "with_mean": True, "with_std": True},
    }


def _source_paths(config_path, project_root):
    base = "backend/scripts/Evaluation"
    return [
        f"{base}/run_experiment.py",
        _relative_path(config_path, project_root),
        f"{base}/core/query.py",
        f"{base}/retrieval/fusion.py",
        f"{base}/evaluation/metrics.py",
        f"{base}/reporting/manifest.py",
        f"{base}/reporting/serialization.py",
        f"{base}/experiments/exp03_static_fusion_cv.py",
        f"{base}/experiments/exp04_selective_dynamic_hybrid_nested_cv.py",
        "backend/scripts/Tests/test_exp03_protocol.py",
        "backend/scripts/Tests/test_exp04_protocol.py",
    ]


def validate_exp04_artifacts(artifacts):
    expected = {
        "feature_contract.json",
        "outer_fold_manifest.json",
        "inner_selection_results.json",
        "oof_predictions.json",
        "pipeline_results.json",
        "summary_metrics.json",
        "statistical_analysis.json",
        "run_manifest.json",
    }
    if set(artifacts) != expected:
        raise ValueError("EXP-04 artifact payload must contain the exact frozen artifact set")
    predictions = artifacts["oof_predictions.json"].get("predictions", [])
    query_ids = [item.get("query_id") for item in predictions]
    if len(query_ids) != EXP04_QUERY_COUNT or len(set(query_ids)) != EXP04_QUERY_COUNT:
        raise ValueError("EXP-04 artifacts require exactly 107 unique OOF predictions")
    if any(any(key in item for key in GT_RESULT_KEYS) for item in predictions):
        raise ValueError("EXP-04 frozen prediction artifact must not contain ground truth")
    fold_results = artifacts["inner_selection_results.json"].get("outer_folds", [])
    if len(fold_results) != EXP04_FOLD_COUNT:
        raise ValueError("EXP-04 artifacts require five outer-fold selection records")
    if artifacts["feature_contract.json"] != feature_contract_payload():
        raise ValueError("EXP-04 feature-contract artifact drift")
    manifest = artifacts["run_manifest.json"]
    required_manifest = {
        "run_id",
        "timestamp",
        "experiment_id",
        "protocol_id",
        "evaluation_mode",
        "control_protocol",
        "candidate_source_run",
        "candidate_path",
        "candidate_fingerprint",
        "fold_path",
        "fold_mapping_fingerprint",
        "feature_contract_version",
        "source_files",
        "git_commit",
        "git_dirty",
        "tracked_code_dirty",
        "exp04_source_dirty",
        "package_versions",
    }
    missing = sorted(required_manifest - set(manifest))
    if missing:
        raise ValueError(f"EXP-04 run manifest is missing required fields: {missing}")


def run_exp04_selective_dynamic_hybrid_nested_cv(config_path, max_queries=None):
    if max_queries is not None:
        raise ValueError("EXP-04 requires the complete frozen 107-query candidate cache")
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    _validate_exp04_protocol(config)

    project_root = Path(__file__).resolve().parents[4]
    candidate_path = project_root / EXP04_CANDIDATE_PATH
    fold_path = project_root / EXP04_FOLD_PATH
    source_summary_path = project_root / EXP04_SOURCE_SUMMARY_PATH
    with open(candidate_path, "r", encoding="utf-8") as handle:
        candidate_cache = json.load(handle)
    with open(fold_path, "r", encoding="utf-8") as handle:
        fold_manifest = json.load(handle)
    with open(source_summary_path, "r", encoding="utf-8") as handle:
        source_summary = json.load(handle)

    _require_equal(candidate_cache.get("query_count"), EXP04_QUERY_COUNT, "cache query count")
    _require_equal(candidate_cache.get("pool_k"), EXP04_POOL_K, "cache pool_k")
    _require_equal(
        candidate_cache.get("fingerprint_algorithm"),
        EXP03_CANDIDATE_FINGERPRINT_ALGORITHM,
        "candidate fingerprint algorithm",
    )
    candidate_records = candidate_cache.get("records", [])
    validate_candidate_pools(candidate_records, EXP04_QUERY_COUNT, EXP04_POOL_K)
    actual_candidate_fingerprint = fingerprint_candidate_pools(candidate_records)
    _require_equal(
        actual_candidate_fingerprint,
        EXP04_CANDIDATE_FINGERPRINT,
        "computed candidate fingerprint",
    )
    _require_equal(
        candidate_cache.get("candidate_cache_fingerprint"),
        EXP04_CANDIDATE_FINGERPRINT,
        "persisted candidate fingerprint",
    )

    _require_equal(fold_manifest.get("assignment_algorithm"), EXP03_FOLD_ALGORITHM, "fold algorithm")
    _require_equal(fold_manifest.get("fold_count"), EXP04_FOLD_COUNT, "persisted fold count")
    _require_equal(tuple(fold_manifest.get("fold_sizes", [])), EXP04_FOLD_SIZES, "persisted fold sizes")
    mapping_items = fold_manifest.get("mapping", [])
    fold_mapping = {item["query_id"]: int(item["fold"]) for item in mapping_items}
    if len(fold_mapping) != len(mapping_items):
        raise ValueError("EXP-04 fold source contains duplicate query IDs")
    _require_equal(
        fingerprint_fold_mapping(fold_mapping),
        EXP04_FOLD_FINGERPRINT,
        "computed fold fingerprint",
    )
    _require_equal(
        fold_manifest.get("mapping_fingerprint"),
        EXP04_FOLD_FINGERPRINT,
        "persisted fold fingerprint",
    )
    validate_fold_mapping(
        fold_mapping,
        [_record_query_id(record) for record in candidate_records],
        EXP04_FOLD_COUNT,
    )

    source_paths = _source_paths(config_path, project_root)
    provenance = git_provenance(project_root, source_paths)
    if provenance.get("baseline_source_dirty") is not False:
        raise RuntimeError("A real EXP-04 run requires committed, clean EXP-04 source files")

    predictions, evaluation_results, fold_results = build_nested_oof_results(
        candidate_records, fold_mapping
    )
    configurations = ["Dynamic_OOF", "Fixed_0_5", "Rule_Baseline"]
    metric_worlds = build_metric_world_report(
        evaluation_results,
        configurations,
        EXP04_REPORTED_K,
        alpha_core=3,
        alpha_support=1,
    )
    parity = validate_fixed_control_parity(metric_worlds, source_summary)
    override_report = false_override_report(evaluation_results)
    statistical_analysis = build_statistical_analysis(evaluation_results)
    success = evaluate_success_contract(statistical_analysis, override_report)

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"{EXP04_PROTOCOL_ID}_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    run_dir = project_root / config["paths"]["output_dir"] / "runs" / run_id
    exp04_source_dirty = provenance.pop("baseline_source_dirty")
    exp04_source_status = provenance.pop("baseline_source_status")
    manifest = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "experiment_id": EXP04_EXPERIMENT_ID,
        "protocol_id": EXP04_PROTOCOL_ID,
        "evaluation_mode": EXP04_MODE,
        "control_protocol": BASELINE_PROTOCOL_ID,
        "candidate_source_run": EXP04_SOURCE_RUN_ID,
        "candidate_path": EXP04_CANDIDATE_PATH,
        "candidate_file_sha256": sha256_file(candidate_path),
        "candidate_fingerprint": EXP04_CANDIDATE_FINGERPRINT,
        "fold_path": EXP04_FOLD_PATH,
        "fold_file_sha256": sha256_file(fold_path),
        "fold_mapping_fingerprint": EXP04_FOLD_FINGERPRINT,
        "query_count": EXP04_QUERY_COUNT,
        "pool_k": EXP04_POOL_K,
        "stored_k": EXP04_STORED_K,
        "query_representation": "q0",
        "document_representation": "d0",
        "fusion_contract": "minmax_full_top30_missing_zero_score_desc_id_asc",
        "ranking_tie_break": EXP03_RANKING_TIE_BREAK,
        "metric_contract": {
            "primary_world": "closed_world",
            "primary_metric": "weighted_recall",
            "primary_k": 10,
            "reported_k": list(EXP04_REPORTED_K),
            "alpha_core": 3,
            "alpha_support": 1,
        },
        "feature_contract_version": EXP04_FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "model_parameters": config["model"],
        "c_grid": list(EXP04_C_GRID),
        "threshold_grid": list(EXP04_THRESHOLD_GRID),
        "action_pairs": list(EXP04_ACTION_PAIRS),
        "selection_tie_break": list(EXP04_SELECTION_TIE_BREAK),
        "bootstrap": {"samples": EXP04_BOOTSTRAP_SAMPLES, "seed": 42},
        "sign_flip": {"samples": EXP04_SIGN_FLIP_SAMPLES, "seed": 42},
        "fixed_control_parity": parity,
        "config": {
            "path": _relative_path(config_path, project_root),
            "sha256": sha256_file(config_path),
        },
        "source_files": _source_file_provenance(project_root, source_paths),
        **provenance,
        "exp04_source_dirty": exp04_source_dirty,
        "exp04_source_status": exp04_source_status,
        "python_version": platform.python_version(),
        "package_versions": {
            name: safe_package_version(name)
            for name in ("numpy", "scikit-learn", "PyYAML")
        },
    }
    artifacts = {
        "feature_contract.json": feature_contract_payload(),
        "outer_fold_manifest.json": fold_manifest,
        "inner_selection_results.json": {
            "protocol_id": EXP04_PROTOCOL_ID,
            "outer_folds": fold_results,
        },
        "oof_predictions.json": {
            "run_id": run_id,
            "protocol_id": EXP04_PROTOCOL_ID,
            "predictions": predictions,
        },
        "pipeline_results.json": {
            "run_id": run_id,
            "protocol_id": EXP04_PROTOCOL_ID,
            "experiment_id": EXP04_EXPERIMENT_ID,
            "configurations": configurations,
            "results": evaluation_results,
        },
        "summary_metrics.json": {
            "run_id": run_id,
            "protocol_id": EXP04_PROTOCOL_ID,
            "experiment_id": EXP04_EXPERIMENT_ID,
            "primary_world": "closed_world",
            "primary_metric": "weighted_recall",
            "primary_k": 10,
            "reported_k": list(EXP04_REPORTED_K),
            "closed_world": metric_worlds["closed_world"],
            "open_world": metric_worlds["open_world"],
            "fixed_control_parity": parity,
            "false_override_report": override_report,
            "success_contract": success,
        },
        "statistical_analysis.json": {
            "run_id": run_id,
            "protocol_id": EXP04_PROTOCOL_ID,
            **statistical_analysis,
        },
        "run_manifest.json": manifest,
    }
    validate_exp04_artifacts(artifacts)
    return write_exp04_artifacts(run_dir, config_path, artifacts)
