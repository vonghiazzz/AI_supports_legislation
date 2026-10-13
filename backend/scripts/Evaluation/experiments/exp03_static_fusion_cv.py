import datetime
import hashlib
import json
import math
import platform
import random
import uuid
from pathlib import Path

import yaml

from ..core.corpus import load_documents, load_frozen_baseline_corpus, sha256_file
from ..core.ground_truth import build_canonical_ground_truth
from ..core.query import build_query_text
from ..evaluation.coverage import compute_ground_truth_coverage
from ..evaluation.metrics import build_metric_world_report, get_weighted_recall
from ..reporting.manifest import (
    enforce_full_run_provenance,
    fingerprint_records,
    git_provenance,
    resolved_model_revision,
    safe_package_version,
    verify_model_revision,
)
from ..reporting.serialization import write_exp03_artifacts
from ..retrieval.bm25 import build_bm25_index, search_bm25
from ..retrieval.dense import build_dense_retriever, encode_dense_query, search_faiss
from ..retrieval.fusion import fuse_pools
from .baseline import (
    BASELINE_PROTOCOL_ID,
    FROZEN_CORPUS_COUNTS,
    FROZEN_CORPUS_FINGERPRINT,
    FROZEN_CORPUS_TOTAL,
    FROZEN_EMBEDDING_MODEL,
    FROZEN_EMBEDDING_REVISION,
    _combined_corpus_fingerprint,
)


EXP03_MODE = "exp03_static_fusion_cv"
EXP03_EXPERIMENT_ID = "exp03_static_hybrid_weight_v1"
EXP03_PROTOCOL_ID = "exp03_static_hybrid_5fold_cv_v1"
EXP03_FOLD_ALGORITHM = "sha256_protocol_seed_query_id_contiguous_v1"
EXP03_FOLD_COUNT = 5
EXP03_FOLD_SEED = 42
EXP03_FOLD_SIZES = (22, 22, 21, 21, 21)
EXP03_QUERY_COUNT = 107
EXP03_POOL_K = 30
EXP03_STORED_K = 20
EXP03_REPORTED_K = (5, 10, 20)
EXP03_PRIMARY_K = 10
EXP03_WEIGHT_GRID = tuple(value / 10.0 for value in range(11))
EXP03_CONTROL_WEIGHT = 0.5
EXP03_WEIGHT_TIE_BREAK = (
    "highest_unrounded_objective",
    "smallest_distance_to_0.5",
    "smaller_w_bm25",
)
EXP03_RANKING_TIE_BREAK = "score_desc_then_standard_law_id_asc"
EXP03_EMBEDDING_DIMENSION = 768
EXP03_FAISS_INDEX_TYPE = "IndexFlatIP"
EXP03_CANDIDATE_FINGERPRINT_ALGORITHM = "sha256_canonical_json_v1"
EXP03_GOLDEN_RUN_ID = "exp02_d0_vs_d1_v1_d0_20261013_085034_383461_4828f45f"
EXP03_GOLDEN_METRIC_TOLERANCE = 1.0e-12
EXP03_GOLDEN_CONFIGURATIONS = ("BM25", "FAISS", "Hybrid_Static_Fixed")
EXP03_Q0_FINGERPRINT = "adf712e5d2636a960bbf454e1d32e6e8d665f3189a1c9776cf3cf84558ceff08"

GT_RESULT_KEYS = (
    "core_gt_raw",
    "supporting_gt_raw",
    "core_gt_filtered",
    "supporting_gt_filtered",
    "substance_gt_raw",
    "procedural_gt_raw",
    "substance_gt_filtered",
    "procedural_gt_filtered",
    "explicit_gts",
    "inferred_gts",
    "gt_weights",
)


def _require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"EXP-03 requires {label}={expected!r}; got {actual!r}")


def _validate_exp03_protocol(config):
    protocol = config.get("protocol", {})
    evaluation = config.get("evaluation", {})
    metrics = config.get("metric_protocol", {})
    cv = config.get("cross_validation", {})
    tuning = config.get("weight_tuning", {})
    objective = tuning.get("objective", {})
    models = config.get("models", {})
    corpus = config.get("corpus", {})

    _require_equal(protocol.get("id"), EXP03_PROTOCOL_ID, "protocol.id")
    _require_equal(
        protocol.get("experiment_id"), EXP03_EXPERIMENT_ID, "protocol.experiment_id"
    )
    _require_equal(
        protocol.get("control_protocol"), BASELINE_PROTOCOL_ID, "protocol.control_protocol"
    )
    _require_equal(evaluation.get("mode"), EXP03_MODE, "evaluation.mode")
    _require_equal(
        evaluation.get("experiment_id"), EXP03_EXPERIMENT_ID, "evaluation.experiment_id"
    )
    _require_equal(evaluation.get("query_representation"), "q0", "query_representation")
    _require_equal(evaluation.get("document_representation"), "d0", "document_representation")
    _require_equal(evaluation.get("temporal_filter"), "none", "temporal_filter")
    _require_equal(int(evaluation.get("pool_k", 0)), EXP03_POOL_K, "pool_k")
    _require_equal(int(evaluation.get("k", 0)), EXP03_STORED_K, "stored_k")
    _require_equal(int(evaluation.get("alpha_core", 0)), 3, "alpha_core")
    _require_equal(int(evaluation.get("alpha_support", 0)), 1, "alpha_support")
    _require_equal(int(evaluation.get("random_seed", -1)), 42, "random_seed")
    _require_equal(
        int(evaluation.get("expected_query_count", 0)), EXP03_QUERY_COUNT, "expected_query_count"
    )
    _require_equal(
        evaluation.get("query_fingerprint"), EXP03_Q0_FINGERPRINT, "Q0 fingerprint"
    )

    _require_equal(metrics.get("primary_world"), "closed_world", "primary_world")
    _require_equal(metrics.get("primary_metric"), "weighted_recall", "primary_metric")
    _require_equal(int(metrics.get("primary_k", 0)), EXP03_PRIMARY_K, "primary_k")
    _require_equal(
        tuple(int(value) for value in metrics.get("reported_k", [])),
        EXP03_REPORTED_K,
        "reported_k",
    )

    _require_equal(int(cv.get("fold_count", 0)), EXP03_FOLD_COUNT, "fold_count")
    _require_equal(int(cv.get("seed", -1)), EXP03_FOLD_SEED, "fold_seed")
    _require_equal(cv.get("algorithm"), EXP03_FOLD_ALGORITHM, "fold_algorithm")
    _require_equal(
        tuple(int(value) for value in cv.get("fold_sizes", [])),
        EXP03_FOLD_SIZES,
        "fold_sizes",
    )

    _require_equal(
        tuple(float(value) for value in tuning.get("grid", [])),
        EXP03_WEIGHT_GRID,
        "weight_grid",
    )
    _require_equal(objective.get("world"), "closed_world", "objective.world")
    _require_equal(objective.get("metric"), "weighted_recall", "objective.metric")
    _require_equal(int(objective.get("k", 0)), EXP03_PRIMARY_K, "objective.k")
    _require_equal(tuple(tuning.get("tie_break", [])), EXP03_WEIGHT_TIE_BREAK, "weight_tie_break")
    _require_equal(
        float(config.get("control", {}).get("static_w_bm25", -1)),
        EXP03_CONTROL_WEIGHT,
        "control.static_w_bm25",
    )

    _require_equal(models.get("embedding_model"), FROZEN_EMBEDDING_MODEL, "embedding_model")
    _require_equal(
        models.get("embedding_revision"), FROZEN_EMBEDDING_REVISION, "embedding_revision"
    )
    _require_equal(models.get("local_files_only"), True, "local_files_only")
    _require_equal(
        int(models.get("embedding_dimension", 0)),
        EXP03_EMBEDDING_DIMENSION,
        "embedding_dimension",
    )
    _require_equal(models.get("embedding_normalized"), True, "embedding_normalized")

    _require_equal(list(corpus.get("required_keys", [])), list(FROZEN_CORPUS_COUNTS), "corpus order")
    _require_equal(corpus.get("expected_counts"), FROZEN_CORPUS_COUNTS, "corpus counts")
    _require_equal(int(corpus.get("expected_total", 0)), FROZEN_CORPUS_TOTAL, "corpus total")
    _require_equal(corpus.get("fingerprint"), FROZEN_CORPUS_FINGERPRINT, "corpus fingerprint")
    _require_equal(config.get("bm25", {}).get("tokenizer"), "whitespace", "BM25 tokenizer")
    _require_equal(config.get("faiss", {}).get("index_type"), EXP03_FAISS_INDEX_TYPE, "FAISS type")
    _require_equal(config.get("fusion", {}).get("method"), "minmax", "fusion method")
    _require_equal(
        config.get("fusion", {}).get("ranking_tie_break"),
        EXP03_RANKING_TIE_BREAK,
        "ranking tie-break",
    )

    golden = config.get("golden_parity", {})
    _require_equal(golden.get("run_id"), EXP03_GOLDEN_RUN_ID, "golden run")
    _require_equal(
        tuple(golden.get("configurations", [])),
        EXP03_GOLDEN_CONFIGURATIONS,
        "golden configurations",
    )
    _require_equal(
        float(golden.get("metric_tolerance", 0.0)),
        EXP03_GOLDEN_METRIC_TOLERANCE,
        "golden metric tolerance",
    )
    _require_equal(
        config.get("paths", {}).get("golden_run_dir"),
        f"backend/data/runs/{EXP03_GOLDEN_RUN_ID}",
        "golden run path",
    )

    return {
        "protocol_id": EXP03_PROTOCOL_ID,
        "experiment_id": EXP03_EXPERIMENT_ID,
        "control_protocol": BASELINE_PROTOCOL_ID,
        "query_representation": "q0",
        "document_representation": "d0",
        "temporal_policy": "none",
        "pool_k": EXP03_POOL_K,
        "stored_k": EXP03_STORED_K,
        "reported_k": EXP03_REPORTED_K,
        "primary_k": EXP03_PRIMARY_K,
        "alpha_core": 3,
        "alpha_support": 1,
        "fold_count": EXP03_FOLD_COUNT,
        "fold_seed": EXP03_FOLD_SEED,
        "fold_algorithm": EXP03_FOLD_ALGORITHM,
        "fold_sizes": EXP03_FOLD_SIZES,
        "weight_grid": EXP03_WEIGHT_GRID,
        "weight_tie_break": EXP03_WEIGHT_TIE_BREAK,
        "ranking_tie_break": EXP03_RANKING_TIE_BREAK,
        "control_weight": EXP03_CONTROL_WEIGHT,
    }


def _record_query_id(record):
    if isinstance(record, dict):
        if "query_id" in record:
            return record["query_id"]
        if "id" in record:
            return record["id"]
        raise ValueError("Query record is missing query_id/id")
    return record


def _query_id_sort_key(query_id):
    if isinstance(query_id, bool) or not isinstance(query_id, (int, str)):
        raise ValueError("query_id must be an integer or string")
    return (0, query_id) if isinstance(query_id, int) else (1, query_id)


def _fold_hash(query_id, protocol_id, seed):
    payload = f"{protocol_id}|{seed}|{query_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_folds(
    records,
    protocol_id=EXP03_PROTOCOL_ID,
    seed=EXP03_FOLD_SEED,
    fold_sizes=EXP03_FOLD_SIZES,
):
    query_ids = [_record_query_id(record) for record in records]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("Fold assignment requires unique query IDs")
    if sum(fold_sizes) != len(query_ids):
        raise ValueError("Fold sizes must sum to the query count")
    ordered = sorted(
        query_ids,
        key=lambda query_id: (
            _fold_hash(query_id, protocol_id, seed),
            _query_id_sort_key(query_id),
        ),
    )
    mapping = {}
    start = 0
    for fold, size in enumerate(fold_sizes):
        for query_id in ordered[start : start + size]:
            mapping[query_id] = fold
        start += size
    return mapping


def _canonical_fold_mapping(mapping):
    return [
        {"query_id": query_id, "fold": int(mapping[query_id])}
        for query_id in sorted(mapping, key=_query_id_sort_key)
    ]


def fingerprint_fold_mapping(mapping):
    payload = json.dumps(
        _canonical_fold_mapping(mapping),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_fold_manifest(mapping, protocol):
    counts = [
        sum(1 for fold in mapping.values() if fold == index)
        for index in range(protocol["fold_count"])
    ]
    if tuple(counts) != tuple(protocol["fold_sizes"]):
        raise ValueError("Fold mapping does not match the predeclared fold sizes")
    return {
        "assignment_algorithm": protocol["fold_algorithm"],
        "seed": protocol["fold_seed"],
        "fold_count": protocol["fold_count"],
        "fold_sizes": counts,
        "mapping": _canonical_fold_mapping(mapping),
        "mapping_fingerprint": fingerprint_fold_mapping(mapping),
    }


def _canonical_pool(pool):
    return [
        {
            "rank": int(item["rank"]),
            "standard_law_id": item["standard_law_id"],
            "score": float(item["score"]),
        }
        for item in sorted(pool, key=lambda item: int(item["rank"]))
    ]


def _canonical_candidate_pools(records):
    canonical = []
    for record in sorted(records, key=lambda item: _query_id_sort_key(_record_query_id(item))):
        canonical.append(
            {
                "query_id": _record_query_id(record),
                "bm25_pool": _canonical_pool(record["bm25_pool"]),
                "dense_pool": _canonical_pool(record["dense_pool"]),
            }
        )
    return canonical


def fingerprint_candidate_pools(records):
    payload = json.dumps(
        _canonical_candidate_pools(records),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_candidate_pools(records, expected_query_count, pool_k):
    query_ids = [_record_query_id(record) for record in records]
    if len(records) != expected_query_count or len(set(query_ids)) != expected_query_count:
        raise ValueError("Candidate cache must contain the expected unique query IDs")
    for record in records:
        for name in ("bm25_pool", "dense_pool"):
            pool = record.get(name, [])
            if len(pool) != pool_k:
                raise ValueError(f"{name} must preserve exactly pool_k={pool_k} candidates")
            if [int(item["rank"]) for item in pool] != list(range(1, pool_k + 1)):
                raise ValueError(f"{name} ranks must be contiguous from 1 to pool_k")
            candidate_ids = [item["standard_law_id"] for item in pool]
            if len(set(candidate_ids)) != pool_k:
                raise ValueError(f"{name} candidate IDs must be unique")
            if not all(math.isfinite(float(item["score"])) for item in pool):
                raise ValueError(f"{name} scores must be finite")


def collect_candidate_pools(
    queries,
    documents,
    bm25_index,
    embedder,
    faiss_index,
    pool_k,
    query_representation,
    corpus_document_ids,
    alpha_core=3,
    alpha_support=1,
):
    records = []
    for index, query_record in enumerate(queries):
        query_text = build_query_text(query_record, query_representation)
        bm25_pool = search_bm25(bm25_index, documents, query_text, pool_k)
        query_embedding = encode_dense_query(embedder, query_text)
        dense_pool = search_faiss(faiss_index, query_embedding, documents, pool_k)
        records.append(
            {
                "query_id": query_record.get("id", index),
                "query": query_text,
                **build_canonical_ground_truth(
                    query_record,
                    corpus_document_ids,
                    alpha_core=alpha_core,
                    alpha_support=alpha_support,
                ),
                "bm25_pool": bm25_pool,
                "dense_pool": dense_pool,
            }
        )
    return records


def fuse_cached_query(record, w_bm25, stored_k=EXP03_STORED_K):
    fused = fuse_pools(
        record["bm25_pool"],
        record["dense_pool"],
        float(w_bm25),
        "minmax",
    )
    return [
        {"rank": rank, "standard_law_id": standard_id, "score": float(score)}
        for rank, (standard_id, score) in enumerate(fused[:stored_k], 1)
    ]


def _result_base(record):
    result = {
        "query_id": _record_query_id(record),
        "query": record.get("query", ""),
    }
    for key in GT_RESULT_KEYS:
        if key in record:
            result[key] = record[key]
    return result


def _single_weight_results(records, w_bm25, configuration, stored_k):
    return [
        {
            **_result_base(record),
            "metrics": {
                configuration: {"top_k": fuse_cached_query(record, w_bm25, stored_k)}
            },
        }
        for record in records
    ]


def _golden_control_results(records, stored_k):
    return [
        {
            **_result_base(record),
            "metrics": {
                "BM25": {"top_k": record["bm25_pool"][:stored_k]},
                "FAISS": {"top_k": record["dense_pool"][:stored_k]},
                "Hybrid_Static_Fixed": {
                    "top_k": fuse_cached_query(record, EXP03_CONTROL_WEIGHT, stored_k)
                },
            },
        }
        for record in records
    ]


def evaluate_weight_on_query_subset(
    records,
    w_bm25,
    k=EXP03_PRIMARY_K,
    alpha_core=3,
    alpha_support=1,
):
    configuration = "Candidate_Weight"
    results = _single_weight_results(records, w_bm25, configuration, stored_k=k)
    return get_weighted_recall(
        results,
        configuration,
        k,
        is_filtered=True,
        alpha_core=alpha_core,
        alpha_support=alpha_support,
    )


def select_fold_weight(
    training_records,
    weight_grid=EXP03_WEIGHT_GRID,
    k=EXP03_PRIMARY_K,
    alpha_core=3,
    alpha_support=1,
):
    if not training_records:
        raise ValueError("Weight selection requires non-empty training records")
    grid_scores = [
        {
            "w_bm25": float(weight),
            "weighted_recall_at_10": evaluate_weight_on_query_subset(
                training_records,
                weight,
                k=k,
                alpha_core=alpha_core,
                alpha_support=alpha_support,
            ),
        }
        for weight in weight_grid
    ]
    selected = max(
        grid_scores,
        key=lambda item: (
            item["weighted_recall_at_10"],
            -abs(item["w_bm25"] - EXP03_CONTROL_WEIGHT),
            -item["w_bm25"],
        ),
    )
    return {
        "selected_w_bm25": selected["w_bm25"],
        "objective_value": selected["weighted_recall_at_10"],
        "training_grid_scores": grid_scores,
    }


def validate_train_heldout_split(training_query_ids, heldout_query_ids, expected_query_ids):
    training = set(training_query_ids)
    heldout = set(heldout_query_ids)
    expected = set(expected_query_ids)
    if training & heldout:
        raise ValueError("Training and held-out query IDs overlap")
    if training | heldout != expected:
        raise ValueError("Training and held-out query IDs do not cover the expected fold universe")


def validate_fold_mapping(mapping, expected_query_ids, fold_count=EXP03_FOLD_COUNT):
    expected = set(expected_query_ids)
    if set(mapping) != expected:
        raise ValueError("Fold mapping does not cover exactly the expected query IDs")
    invalid = [fold for fold in mapping.values() if fold not in range(fold_count)]
    if invalid:
        raise ValueError("Fold mapping contains an invalid fold index")


def validate_oof_coverage(oof_results, expected_query_ids):
    actual = [_record_query_id(record) for record in oof_results]
    if len(actual) != len(set(actual)):
        raise ValueError("OOF results contain duplicate held-out query IDs")
    if set(actual) != set(expected_query_ids):
        raise ValueError("OOF results do not cover exactly the expected query IDs")


def build_oof_results(
    candidate_records,
    fold_mapping,
    weight_grid=EXP03_WEIGHT_GRID,
    fold_count=EXP03_FOLD_COUNT,
    stored_k=EXP03_STORED_K,
    objective_k=EXP03_PRIMARY_K,
    alpha_core=3,
    alpha_support=1,
):
    record_by_id = {_record_query_id(record): record for record in candidate_records}
    expected_query_ids = sorted(record_by_id, key=_query_id_sort_key)
    if len(record_by_id) != len(candidate_records):
        raise ValueError("Candidate records must have unique query IDs")
    validate_fold_mapping(fold_mapping, expected_query_ids, fold_count)

    oof_results = []
    fold_results = []
    for fold in range(fold_count):
        heldout_ids = [query_id for query_id in expected_query_ids if fold_mapping[query_id] == fold]
        training_ids = [query_id for query_id in expected_query_ids if fold_mapping[query_id] != fold]
        validate_train_heldout_split(training_ids, heldout_ids, expected_query_ids)
        selection = select_fold_weight(
            [record_by_id[query_id] for query_id in training_ids],
            weight_grid,
            k=objective_k,
            alpha_core=alpha_core,
            alpha_support=alpha_support,
        )
        selected_weight = selection["selected_w_bm25"]
        for query_id in heldout_ids:
            record = record_by_id[query_id]
            oof_results.append(
                {
                    **_result_base(record),
                    "cv_fold": fold,
                    "selected_w_bm25": selected_weight,
                    "metrics": {
                        "OOF_Selected_Policy": {
                            "top_k": fuse_cached_query(record, selected_weight, stored_k)
                        },
                        "Control_Fixed_0_5": {
                            "top_k": fuse_cached_query(record, EXP03_CONTROL_WEIGHT, stored_k)
                        },
                    },
                }
            )
        fold_results.append(
            {
                "fold": fold,
                "training_query_ids": sorted(training_ids, key=_query_id_sort_key),
                "heldout_query_ids": sorted(heldout_ids, key=_query_id_sort_key),
                **selection,
            }
        )

    oof_results.sort(key=lambda record: _query_id_sort_key(_record_query_id(record)))
    validate_oof_coverage(oof_results, expected_query_ids)
    return oof_results, fold_results


def validate_golden_parity(
    candidate_records,
    golden_pipeline,
    golden_summary=None,
    reported_k=EXP03_REPORTED_K,
    stored_k=EXP03_STORED_K,
    alpha_core=3,
    alpha_support=1,
    metric_tolerance=1.0e-12,
):
    golden_records = golden_pipeline.get("results", golden_pipeline)
    candidate_by_id = {_record_query_id(record): record for record in candidate_records}
    golden_by_id = {_record_query_id(record): record for record in golden_records}
    if (
        set(candidate_by_id) != set(golden_by_id)
        or len(candidate_by_id) != len(candidate_records)
        or len(golden_by_id) != len(golden_records)
    ):
        raise RuntimeError("Golden parity failed: query IDs do not align exactly")

    control_results = _golden_control_results(candidate_records, stored_k)
    control_by_id = {_record_query_id(record): record for record in control_results}
    for query_id in sorted(candidate_by_id, key=_query_id_sort_key):
        for configuration in EXP03_GOLDEN_CONFIGURATIONS:
            actual_ids = [
                item["standard_law_id"]
                for item in control_by_id[query_id]["metrics"][configuration]["top_k"]
            ]
            expected_ids = [
                item["standard_law_id"]
                for item in golden_by_id[query_id]["metrics"][configuration]["top_k"][
                    :stored_k
                ]
            ]
            if len(expected_ids) != stored_k:
                raise RuntimeError(
                    "Golden parity failed: "
                    f"query {query_id} lacks {configuration} top-{stored_k} results"
                )
            if actual_ids != expected_ids:
                raise RuntimeError(
                    "Golden parity failed: "
                    f"{configuration} top-{stored_k} ranking differs for query {query_id}"
                )

    metric_worlds = build_metric_world_report(
        control_results,
        EXP03_GOLDEN_CONFIGURATIONS,
        reported_k,
        alpha_core=alpha_core,
        alpha_support=alpha_support,
    )
    if golden_summary is not None:
        for world in ("closed_world", "open_world"):
            for configuration in EXP03_GOLDEN_CONFIGURATIONS:
                for k in reported_k:
                    actual = metric_worlds[world][configuration][str(k)]
                    expected = golden_summary[world][configuration][str(k)]
                    for metric in (
                        "recall",
                        "ndcg",
                        "mrr",
                        "weighted_recall",
                        "valid_queries",
                    ):
                        if metric not in actual or metric not in expected:
                            raise RuntimeError(
                                "Golden parity failed: "
                                f"{world} {configuration} {metric}@{k} is missing"
                            )
                        matches = (
                            actual[metric] == expected[metric]
                            if metric == "valid_queries"
                            else math.isclose(
                                float(actual[metric]),
                                float(expected[metric]),
                                rel_tol=0.0,
                                abs_tol=metric_tolerance,
                            )
                        )
                        if not matches:
                            raise RuntimeError(
                                "Golden parity failed: "
                                f"{world} {configuration} {metric}@{k} differs "
                                f"beyond {metric_tolerance}"
                            )
    return {"passed": True, "metric_worlds": metric_worlds}


def validate_exp03_artifact_schema(
    candidate_cache,
    fold_manifest,
    pipeline_results,
    summary,
    run_manifest,
):
    required_candidate = {
        "schema_version",
        "protocol_id",
        "query_count",
        "pool_k",
        "fingerprint_algorithm",
        "candidate_cache_fingerprint",
        "records",
    }
    required_fold = {
        "assignment_algorithm",
        "seed",
        "fold_count",
        "fold_sizes",
        "mapping",
        "mapping_fingerprint",
    }
    required_pipeline = {
        "run_id",
        "protocol_id",
        "experiment_id",
        "control_protocol",
        "configurations",
        "fold_results",
        "results",
    }
    required_summary = {
        "run_id",
        "protocol_id",
        "experiment_id",
        "primary_world",
        "primary_metric",
        "primary_k",
        "reported_k",
        "selected_weights_by_fold",
        "closed_world",
        "open_world",
    }
    required_manifest = {
        "run_id",
        "timestamp",
        "protocol_id",
        "experiment_id",
        "control_protocol",
        "query_count",
        "corpus_count",
        "query_representation",
        "document_representation",
        "temporal_filter",
        "embedding_model",
        "embedding_requested_revision",
        "embedding_resolved_revision",
        "embedding_revision_verified",
        "embedding_local_files_only",
        "embedding_normalized",
        "embedding_dimension",
        "faiss_index_type",
        "bm25_tokenizer",
        "pool_k",
        "stored_k",
        "candidate_cache_fingerprint",
        "fold_algorithm",
        "fold_seed",
        "fold_count",
        "fold_sizes",
        "fold_mapping_fingerprint",
        "weight_grid",
        "objective",
        "weight_tie_break",
        "selected_weights_by_fold",
        "ranking_tie_break",
        "primary_k",
        "reported_k",
        "alpha_core",
        "alpha_support",
        "combined_corpus_fingerprint",
        "query_representation_fingerprint",
        "benchmark",
        "gt_schema",
        "config",
        "source_files",
        "corpora",
        "ground_truth_coverage",
        "golden_parity",
        "git_commit",
        "git_dirty",
        "tracked_code_dirty",
        "random_seed",
        "package_versions",
    }
    payloads = (
        ("candidate cache", candidate_cache, required_candidate),
        ("fold manifest", fold_manifest, required_fold),
        ("pipeline results", pipeline_results, required_pipeline),
        ("summary", summary, required_summary),
        ("run manifest", run_manifest, required_manifest),
    )
    for label, payload, required in payloads:
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"EXP-03 {label} is missing required fields: {missing}")
    validate_candidate_pools(
        candidate_cache["records"],
        int(candidate_cache["query_count"]),
        int(candidate_cache["pool_k"]),
    )
    candidate_fingerprint = fingerprint_candidate_pools(candidate_cache["records"])
    if candidate_cache["candidate_cache_fingerprint"] != candidate_fingerprint:
        raise ValueError("EXP-03 candidate-cache fingerprint does not match its records")
    mapping = {
        item["query_id"]: int(item["fold"])
        for item in fold_manifest["mapping"]
    }
    if len(mapping) != len(fold_manifest["mapping"]):
        raise ValueError("EXP-03 fold manifest contains duplicate query IDs")
    if fingerprint_fold_mapping(mapping) != fold_manifest["mapping_fingerprint"]:
        raise ValueError("EXP-03 fold-mapping fingerprint does not match its mapping")
    actual_fold_sizes = [
        sum(1 for value in mapping.values() if value == fold)
        for fold in range(int(fold_manifest["fold_count"]))
    ]
    if actual_fold_sizes != list(fold_manifest["fold_sizes"]):
        raise ValueError("EXP-03 fold sizes do not match the persisted mapping")
    candidate_query_ids = [
        _record_query_id(record) for record in candidate_cache["records"]
    ]
    validate_fold_mapping(mapping, candidate_query_ids, int(fold_manifest["fold_count"]))
    validate_oof_coverage(pipeline_results["results"], candidate_query_ids)
    if run_manifest["candidate_cache_fingerprint"] != candidate_fingerprint:
        raise ValueError("EXP-03 run manifest candidate fingerprint is inconsistent")
    if run_manifest["fold_mapping_fingerprint"] != fold_manifest["mapping_fingerprint"]:
        raise ValueError("EXP-03 run manifest fold fingerprint is inconsistent")
    if pipeline_results["configurations"] != [
        "OOF_Selected_Policy",
        "Control_Fixed_0_5",
    ]:
        raise ValueError("EXP-03 pipeline results must separate OOF policy and fixed control")
    for record in pipeline_results["results"]:
        if not {"cv_fold", "selected_w_bm25", "metrics"} <= set(record):
            raise ValueError("Every EXP-03 OOF result must trace its fold and selected weight")
        if set(record["metrics"]) != {
            "OOF_Selected_Policy",
            "Control_Fixed_0_5",
        }:
            raise ValueError("Every EXP-03 OOF result must contain policy and control rankings")


def _relative_path(path, project_root):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(resolved)


def _source_paths(config_path, project_root):
    base = "backend/scripts/Evaluation"
    return [
        f"{base}/run_experiment.py",
        _relative_path(config_path, project_root),
        f"{base}/core/corpus.py",
        f"{base}/core/ground_truth.py",
        f"{base}/core/query.py",
        f"{base}/retrieval/bm25.py",
        f"{base}/retrieval/dense.py",
        f"{base}/retrieval/fusion.py",
        f"{base}/evaluation/metrics.py",
        f"{base}/reporting/manifest.py",
        f"{base}/reporting/serialization.py",
        f"{base}/experiments/baseline.py",
        f"{base}/experiments/exp03_static_fusion_cv.py",
        "backend/scripts/Tests/test_baseline_protocol.py",
        "backend/scripts/Tests/test_exp03_protocol.py",
    ]


def _source_file_provenance(project_root, source_paths):
    provenance = []
    for source_path in source_paths:
        resolved = Path(source_path)
        if not resolved.is_absolute():
            resolved = Path(project_root) / resolved
        provenance.append(
            {
                "path": _relative_path(resolved, project_root),
                "sha256": sha256_file(resolved),
            }
        )
    return provenance


def _portable_corpus_inventory(corpus_inventory, project_root):
    return [
        {
            **item,
            "path": _relative_path(item["path"], project_root),
        }
        for item in corpus_inventory
    ]


def run_exp03_static_fusion_cv(config_path, max_queries=None):
    if max_queries is not None:
        raise ValueError("EXP-03 requires the complete frozen 107-query benchmark")
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol = _validate_exp03_protocol(config)

    random.seed(int(config["evaluation"]["random_seed"]))
    import numpy as np
    import torch

    np.random.seed(int(config["evaluation"]["random_seed"]))
    torch.manual_seed(int(config["evaluation"]["random_seed"]))

    project_root = Path(__file__).resolve().parents[4]
    output_dir = project_root / config["paths"]["output_dir"]
    laws_config_path = project_root / config["paths"]["laws_config"]
    benchmark_path = project_root / config["paths"]["eval_file"]
    gt_schema_path = project_root / config["paths"]["gt_schema"]
    golden_run_dir = project_root / config["paths"]["golden_run_dir"]

    from ..validate_benchmark import validate_benchmark_file

    validate_benchmark_file(benchmark_path)
    law_files, corpus_inventory, corpus_count = load_frozen_baseline_corpus(
        laws_config_path, output_dir, config["corpus"]
    )
    corpus_fingerprint = _combined_corpus_fingerprint(corpus_inventory)
    if corpus_fingerprint != FROZEN_CORPUS_FINGERPRINT:
        raise RuntimeError("EXP-03 frozen corpus fingerprint mismatch")
    documents = load_documents(
        law_files,
        document_representation="d0",
        bm25_tokenizer="whitespace",
        laws_config_path=laws_config_path,
    )
    with open(benchmark_path, "r", encoding="utf-8") as handle:
        benchmark = json.load(handle)
    if len(benchmark) != EXP03_QUERY_COUNT:
        raise RuntimeError(f"EXP-03 requires exactly {EXP03_QUERY_COUNT} benchmark queries")
    query_fingerprint = fingerprint_records(
        [
            {"id": record.get("id"), "text": build_query_text(record, "q0")}
            for record in benchmark
        ]
    )
    if query_fingerprint != EXP03_Q0_FINGERPRINT:
        raise RuntimeError("EXP-03 Q0 fingerprint mismatch")

    source_paths = _source_paths(config_path, project_root)
    provenance = git_provenance(project_root, source_paths)
    enforce_full_run_provenance(None, len(benchmark), provenance)
    bm25_index = build_bm25_index(documents)
    embedder, faiss_index, embedding_dimension = build_dense_retriever(
        documents,
        config["models"]["embedding_model"],
        config["models"]["embedding_revision"],
        local_files_only=config["models"]["local_files_only"],
        device="cpu",
    )
    resolved_revision = resolved_model_revision(embedder)
    verify_model_revision(config["models"]["embedding_revision"], resolved_revision)
    if embedding_dimension != EXP03_EMBEDDING_DIMENSION:
        raise RuntimeError("EXP-03 embedding dimension drift")
    if type(faiss_index).__name__ != EXP03_FAISS_INDEX_TYPE:
        raise RuntimeError("EXP-03 FAISS index type drift")

    corpus_document_ids = {document["standard_law_id"] for document in documents}
    candidate_records = collect_candidate_pools(
        benchmark,
        documents,
        bm25_index,
        embedder,
        faiss_index,
        protocol["pool_k"],
        protocol["query_representation"],
        corpus_document_ids,
        protocol["alpha_core"],
        protocol["alpha_support"],
    )
    validate_candidate_pools(candidate_records, EXP03_QUERY_COUNT, EXP03_POOL_K)
    candidate_fingerprint = fingerprint_candidate_pools(candidate_records)

    with open(golden_run_dir / "pipeline_results.json", "r", encoding="utf-8") as handle:
        golden_pipeline = json.load(handle)
    with open(golden_run_dir / "summary_metrics.json", "r", encoding="utf-8") as handle:
        golden_summary = json.load(handle)
    parity = validate_golden_parity(
        candidate_records,
        golden_pipeline,
        golden_summary,
        reported_k=protocol["reported_k"],
        stored_k=protocol["stored_k"],
        alpha_core=protocol["alpha_core"],
        alpha_support=protocol["alpha_support"],
        metric_tolerance=float(config["golden_parity"]["metric_tolerance"]),
    )
    if not parity["passed"]:
        raise RuntimeError("EXP-03 golden parity gate failed")

    fold_mapping = assign_folds(
        candidate_records,
        protocol["protocol_id"],
        protocol["fold_seed"],
        protocol["fold_sizes"],
    )
    fold_manifest = build_fold_manifest(fold_mapping, protocol)
    oof_results, fold_results = build_oof_results(
        candidate_records,
        fold_mapping,
        protocol["weight_grid"],
        protocol["fold_count"],
        protocol["stored_k"],
        protocol["primary_k"],
        protocol["alpha_core"],
        protocol["alpha_support"],
    )
    configurations = ["OOF_Selected_Policy", "Control_Fixed_0_5"]
    metric_worlds = build_metric_world_report(
        oof_results,
        configurations,
        protocol["reported_k"],
        alpha_core=protocol["alpha_core"],
        alpha_support=protocol["alpha_support"],
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"{EXP03_PROTOCOL_ID}_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / "runs" / run_id
    candidate_cache = {
        "schema_version": "exp03_candidate_pools_v1",
        "protocol_id": EXP03_PROTOCOL_ID,
        "query_count": len(candidate_records),
        "pool_k": EXP03_POOL_K,
        "fingerprint_algorithm": EXP03_CANDIDATE_FINGERPRINT_ALGORITHM,
        "candidate_cache_fingerprint": candidate_fingerprint,
        "records": candidate_records,
    }
    pipeline_results = {
        "run_id": run_id,
        "protocol_id": EXP03_PROTOCOL_ID,
        "experiment_id": EXP03_EXPERIMENT_ID,
        "control_protocol": BASELINE_PROTOCOL_ID,
        "configurations": configurations,
        "fold_results": fold_results,
        "results": oof_results,
    }
    summary = {
        "run_id": run_id,
        "protocol_id": EXP03_PROTOCOL_ID,
        "experiment_id": EXP03_EXPERIMENT_ID,
        "primary_world": "closed_world",
        "primary_metric": "weighted_recall",
        "primary_k": EXP03_PRIMARY_K,
        "reported_k": list(EXP03_REPORTED_K),
        "selected_weights_by_fold": [
            {"fold": item["fold"], "w_bm25": item["selected_w_bm25"]}
            for item in fold_results
        ],
        "closed_world": metric_worlds["closed_world"],
        "open_world": metric_worlds["open_world"],
    }
    manifest = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "protocol_id": EXP03_PROTOCOL_ID,
        "experiment_id": EXP03_EXPERIMENT_ID,
        "control_protocol": BASELINE_PROTOCOL_ID,
        "query_count": len(candidate_records),
        "corpus_count": corpus_count,
        "query_representation": "q0",
        "document_representation": "d0",
        "temporal_filter": "none",
        "embedding_model": FROZEN_EMBEDDING_MODEL,
        "embedding_requested_revision": FROZEN_EMBEDDING_REVISION,
        "embedding_resolved_revision": resolved_revision,
        "embedding_revision_verified": resolved_revision == FROZEN_EMBEDDING_REVISION,
        "embedding_local_files_only": config["models"]["local_files_only"],
        "embedding_normalized": True,
        "embedding_dimension": embedding_dimension,
        "faiss_index_type": type(faiss_index).__name__,
        "bm25_tokenizer": "whitespace",
        "pool_k": EXP03_POOL_K,
        "stored_k": EXP03_STORED_K,
        "candidate_cache_fingerprint": candidate_fingerprint,
        "fold_algorithm": EXP03_FOLD_ALGORITHM,
        "fold_seed": EXP03_FOLD_SEED,
        "fold_count": EXP03_FOLD_COUNT,
        "fold_sizes": list(EXP03_FOLD_SIZES),
        "fold_mapping_fingerprint": fold_manifest["mapping_fingerprint"],
        "weight_grid": list(EXP03_WEIGHT_GRID),
        "objective": config["weight_tuning"]["objective"],
        "weight_tie_break": list(EXP03_WEIGHT_TIE_BREAK),
        "selected_weights_by_fold": [
            {"fold": item["fold"], "w_bm25": item["selected_w_bm25"]}
            for item in fold_results
        ],
        "ranking_tie_break": EXP03_RANKING_TIE_BREAK,
        "primary_k": EXP03_PRIMARY_K,
        "reported_k": list(EXP03_REPORTED_K),
        "alpha_core": 3,
        "alpha_support": 1,
        "combined_corpus_fingerprint": corpus_fingerprint,
        "query_representation_fingerprint": query_fingerprint,
        "benchmark": {
            "path": _relative_path(benchmark_path, project_root),
            "sha256": sha256_file(benchmark_path),
        },
        "gt_schema": {
            "path": _relative_path(gt_schema_path, project_root),
            "sha256": sha256_file(gt_schema_path),
        },
        "config": {
            "path": _relative_path(config_path, project_root),
            "sha256": sha256_file(config_path),
        },
        "corpora": _portable_corpus_inventory(corpus_inventory, project_root),
        "source_files": _source_file_provenance(project_root, source_paths),
        "ground_truth_coverage": compute_ground_truth_coverage(benchmark, corpus_document_ids),
        "golden_parity": {
            "passed": True,
            "run_id": config["golden_parity"]["run_id"],
            "configurations": list(EXP03_GOLDEN_CONFIGURATIONS),
            "ranking_depth": EXP03_STORED_K,
            "metric_tolerance": float(config["golden_parity"]["metric_tolerance"]),
        },
        **provenance,
        "random_seed": int(config["evaluation"]["random_seed"]),
        "python_version": platform.python_version(),
        "package_versions": {
            name: safe_package_version(name)
            for name in (
                "torch",
                "numpy",
                "faiss-cpu",
                "rank-bm25",
                "sentence-transformers",
                "PyYAML",
            )
        },
    }
    validate_exp03_artifact_schema(
        candidate_cache,
        fold_manifest,
        pipeline_results,
        summary,
        manifest,
    )
    return write_exp03_artifacts(
        run_dir,
        config_path,
        candidate_cache,
        fold_manifest,
        pipeline_results,
        summary,
        manifest,
    )
