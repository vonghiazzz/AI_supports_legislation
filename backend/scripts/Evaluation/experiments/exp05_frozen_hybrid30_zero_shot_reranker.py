"""EXP-05 zero-shot reranking over frozen Hybrid 0.5 candidates.

Candidate generation is immutable: this module consumes the persisted EXP-03
BM25/Dense pools, reconstructs Hybrid 0.5 top-30, and only reorders those IDs.
Ground truth is deliberately excluded from every tokenizer/model input.
"""

import datetime
import hashlib
import json
import math
import platform
import resource
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..core.corpus import load_documents, load_frozen_baseline_corpus, sha256_file
from ..core.query import normalize_query_q0
from ..evaluation.metrics import build_metric_world_report, get_task_metrics, get_weighted_recall
from ..reporting.manifest import fingerprint_records, git_provenance, safe_package_version
from ..reporting.serialization import write_exp05_artifacts
from .baseline import BASELINE_PROTOCOL_ID
from .exp03_static_fusion_cv import (
    EXP03_CANDIDATE_FINGERPRINT_ALGORITHM,
    EXP03_FOLD_ALGORITHM,
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


EXP05_MODE = "exp05_frozen_candidate_rerank"
EXP05_EXPERIMENT_ID = "exp05_frozen_hybrid30_zero_shot_reranker_v1"
EXP05_PROTOCOL_ID = "exp05_frozen_hybrid30_zero_shot_cross_encoder_eval_v1"
EXP05_CONTROL = "Control_Hybrid_0_5"
EXP05_TREATMENT = "ZeroShot_BGE_Reranker_V2_M3"

EXP05_SOURCE_RUN_ID = "exp03_static_hybrid_5fold_cv_v1_20261013_123303_632045_cef5ac23"
EXP05_CANDIDATE_PATH = f"backend/data/runs/{EXP05_SOURCE_RUN_ID}/candidate_pools.json"
EXP05_FOLD_PATH = f"backend/data/runs/{EXP05_SOURCE_RUN_ID}/fold_manifest.json"
EXP05_SOURCE_MANIFEST_PATH = f"backend/data/runs/{EXP05_SOURCE_RUN_ID}/run_manifest.json"
EXP05_CANDIDATE_FINGERPRINT = (
    "4e94fc54bf2dc1ec09c4fbb605daadb1ba0dfa8c7c0c10a2d40ea34e90c2c805"
)
EXP05_CORPUS_FINGERPRINT = (
    "66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58"
)
EXP05_QUERY_FINGERPRINT = (
    "adf712e5d2636a960bbf454e1d32e6e8d665f3189a1c9776cf3cf84558ceff08"
)
EXP05_FOLD_FINGERPRINT = (
    "94e9f68e7113f21b24e4cfbbccb2faacac1bfb106ac35e5c2167e7f51eee7310"
)

EXP05_QUERY_COUNT = 107
EXP05_POOL_K = 30
EXP05_CANDIDATE_COUNT = 30
EXP05_INFERENCE_PAIR_COUNT = 3_210
EXP05_REPORTED_K = (5, 10, 20)
EXP05_PRIMARY_K = 10
EXP05_FOLD_COUNT = 5
EXP05_FOLD_SIZES = (22, 22, 21, 21, 21)
EXP05_BOOTSTRAP_SAMPLES = 20_000
EXP05_SIGN_FLIP_SAMPLES = 100_000
EXP05_RANDOM_SEED = 42

EXP05_MODEL_ID = "BAAI/bge-reranker-v2-m3"
EXP05_MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
EXP05_MODEL_SNAPSHOT = (
    "/Users/tannghiavo/.cache/huggingface/hub/"
    "models--BAAI--bge-reranker-v2-m3/snapshots/"
    f"{EXP05_MODEL_REVISION}"
)
EXP05_MODEL_HASHES = {
    "config.json": "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",
    "model.safetensors": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    "tokenizer.json": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
}
EXP05_MAX_LENGTH = 1024
EXP05_BATCH_SIZE = 1
EXP05_DEVICE = "cpu"
EXP05_DTYPE = "float32"
EXP05_TRUNCATION = "only_second"
EXP05_RERANK_TIE_BREAK = (
    "reranker_score_desc",
    "original_hybrid_rank_asc",
    "standard_law_id_asc",
)

EXP05_CONTROL_AT_10 = {
    "recall": 0.08825383049544679,
    "ndcg": 0.05286971847087022,
    "mrr": 0.1900904910250704,
    "weighted_recall": 0.07248115500412532,
}
EXP05_PARITY_TOLERANCE = 1.0e-12
EXP05_INFERENCE_KEYS = frozenset(
    {
        "query_id",
        "query",
        "standard_law_id",
        "document",
        "original_hybrid_rank",
    }
)
EXP05_SCORED_KEYS = EXP05_INFERENCE_KEYS | {"reranker_score"}
EXP05_ARTIFACT_FILES = frozenset(
    {
        "run_manifest.json",
        "prediction_artifact.json",
        "summary_metrics.json",
        "statistical_analysis.json",
    }
)
EXP05_SUCCESS_CONDITIONS = (
    "treatment_closed_wr10_above_control",
    "bootstrap_lower_above_zero",
    "sign_flip_p_below_0_05",
    "wr10_wins_exceed_losses",
    "positive_after_removing_largest_gain",
    "positive_in_at_least_three_folds",
    "no_secondary_ci_entirely_below_zero",
)


def _require_equal(actual, expected, label):
    if actual != expected:
        raise ValueError(f"EXP-05 requires {label}={expected!r}; got {actual!r}")


def _validate_exp05_protocol(config):
    protocol = config.get("protocol", {})
    evaluation = config.get("evaluation", {})
    candidate = config.get("candidate_source", {})
    folds = config.get("fold_source", {})
    frozen = config.get("frozen_inputs", {})
    model = config.get("model", {})
    inference = config.get("inference", {})
    control = config.get("control", {})
    metric = config.get("metric_protocol", {})
    statistics = config.get("statistics", {})
    success = config.get("success_contract", {})
    artifacts = config.get("artifacts", {})
    required = (
        (protocol.get("id"), EXP05_PROTOCOL_ID, "protocol.id"),
        (protocol.get("experiment_id"), EXP05_EXPERIMENT_ID, "protocol.experiment_id"),
        (protocol.get("control_protocol"), BASELINE_PROTOCOL_ID, "control protocol"),
        (protocol.get("result_scope"), "exploratory", "result scope"),
        (protocol.get("automatic_runtime_promotion"), False, "automatic promotion"),
        (evaluation.get("mode"), EXP05_MODE, "evaluation.mode"),
        (evaluation.get("experiment_id"), EXP05_EXPERIMENT_ID, "evaluation experiment"),
        (evaluation.get("query_representation"), "q0", "query representation"),
        (evaluation.get("document_representation"), "d0", "document representation"),
        (evaluation.get("temporal_filter"), "none", "temporal filter"),
        (int(evaluation.get("expected_query_count", 0)), EXP05_QUERY_COUNT, "query count"),
        (int(evaluation.get("pool_k", 0)), EXP05_POOL_K, "pool_k"),
        (int(evaluation.get("candidate_count", 0)), EXP05_CANDIDATE_COUNT, "candidate count"),
        (int(evaluation.get("expected_inference_pairs", 0)), EXP05_INFERENCE_PAIR_COUNT, "pair count"),
        (tuple(evaluation.get("reported_k", [])), EXP05_REPORTED_K, "reported_k"),
        (int(evaluation.get("alpha_core", 0)), 3, "alpha_core"),
        (int(evaluation.get("alpha_support", 0)), 1, "alpha_support"),
        (candidate.get("run_id"), EXP05_SOURCE_RUN_ID, "candidate run"),
        (candidate.get("path"), EXP05_CANDIDATE_PATH, "candidate path"),
        (candidate.get("fingerprint"), EXP05_CANDIDATE_FINGERPRINT, "candidate fingerprint"),
        (candidate.get("fusion_method"), "minmax", "fusion method"),
        (float(candidate.get("w_bm25", -1)), 0.5, "BM25 weight"),
        (float(candidate.get("w_dense", -1)), 0.5, "Dense weight"),
        (folds.get("path"), EXP05_FOLD_PATH, "fold path"),
        (folds.get("mapping_fingerprint"), EXP05_FOLD_FINGERPRINT, "fold fingerprint"),
        (int(folds.get("fold_count", 0)), EXP05_FOLD_COUNT, "fold count"),
        (tuple(folds.get("fold_sizes", [])), EXP05_FOLD_SIZES, "fold sizes"),
        (folds.get("purpose"), "stability_reporting_only", "fold purpose"),
        (frozen.get("query_fingerprint"), EXP05_QUERY_FINGERPRINT, "query fingerprint"),
        (frozen.get("corpus_fingerprint"), EXP05_CORPUS_FINGERPRINT, "corpus fingerprint"),
        (frozen.get("d0_template"), "Tiêu đề: {title}. Nội dung: {content}", "D0 template"),
        (model.get("id"), EXP05_MODEL_ID, "model ID"),
        (model.get("revision"), EXP05_MODEL_REVISION, "model revision"),
        (model.get("snapshot_path"), EXP05_MODEL_SNAPSHOT, "model snapshot"),
        (model.get("tokenizer_class"), "transformers.AutoTokenizer", "tokenizer class"),
        (model.get("model_class"), "transformers.AutoModelForSequenceClassification", "model class"),
        (model.get("local_files_only"), True, "local-only loading"),
        (model.get("critical_file_sha256"), EXP05_MODEL_HASHES, "model hashes"),
        (inference.get("device"), EXP05_DEVICE, "device"),
        (inference.get("dtype"), EXP05_DTYPE, "dtype"),
        (int(inference.get("batch_size", 0)), EXP05_BATCH_SIZE, "batch size"),
        (inference.get("model_eval"), True, "model eval"),
        (inference.get("inference_mode"), True, "inference mode"),
        (int(inference.get("max_length", 0)), EXP05_MAX_LENGTH, "max_length"),
        (inference.get("truncation"), EXP05_TRUNCATION, "truncation"),
        (inference.get("padding"), True, "padding"),
        (inference.get("score"), "raw_scalar_classification_logit", "score"),
        (inference.get("sigmoid"), False, "sigmoid"),
        (inference.get("softmax"), False, "softmax"),
        (inference.get("normalize_scores"), False, "score normalization"),
        (inference.get("blend_with_hybrid"), False, "score blending"),
        (tuple(inference.get("rerank_tie_break", [])), EXP05_RERANK_TIE_BREAK, "rerank tie-break"),
        (control.get("name"), EXP05_CONTROL, "control name"),
        (control.get("treatment_name"), EXP05_TREATMENT, "treatment name"),
        (float(control.get("parity_tolerance", -1)), EXP05_PARITY_TOLERANCE, "parity tolerance"),
        (control.get("closed_world_at_10"), EXP05_CONTROL_AT_10, "control metrics"),
        (metric.get("primary_world"), "closed_world", "primary world"),
        (metric.get("primary_metric"), "weighted_recall", "primary metric"),
        (int(metric.get("primary_k", 0)), EXP05_PRIMARY_K, "primary k"),
        (tuple(metric.get("secondary_metrics", [])), ("recall", "ndcg", "mrr"), "secondary metrics"),
        (tuple(metric.get("reported_k", [])), EXP05_REPORTED_K, "metric reported_k"),
        (int(statistics.get("bootstrap_samples", 0)), EXP05_BOOTSTRAP_SAMPLES, "bootstrap samples"),
        (int(statistics.get("bootstrap_seed", -1)), EXP05_RANDOM_SEED, "bootstrap seed"),
        (tuple(float(x) for x in statistics.get("bootstrap_ci_percentiles", [])), (2.5, 97.5), "bootstrap percentiles"),
        (int(statistics.get("sign_flip_samples", 0)), EXP05_SIGN_FLIP_SAMPLES, "sign-flip samples"),
        (int(statistics.get("sign_flip_seed", -1)), EXP05_RANDOM_SEED, "sign-flip seed"),
        (statistics.get("sign_flip_two_sided"), True, "sign-flip sidedness"),
        (statistics.get("sign_flip_statistic"), "absolute_mean", "sign-flip statistic"),
        (success.get("require_all"), True, "success aggregation"),
        (tuple(success.get("conditions", [])), EXP05_SUCCESS_CONDITIONS, "success conditions"),
        (artifacts.get("strict_json"), True, "strict JSON"),
        (artifacts.get("atomic_partial_directory"), True, "atomic publication"),
        (set(artifacts.get("files", [])), set(EXP05_ARTIFACT_FILES), "artifact files"),
    )
    for actual, expected, label in required:
        _require_equal(actual, expected, label)
    return {
        "protocol_id": EXP05_PROTOCOL_ID,
        "experiment_id": EXP05_EXPERIMENT_ID,
        "mode": EXP05_MODE,
        "query_count": EXP05_QUERY_COUNT,
        "candidate_count": EXP05_CANDIDATE_COUNT,
        "inference_pair_count": EXP05_INFERENCE_PAIR_COUNT,
        "reported_k": EXP05_REPORTED_K,
    }


def validate_model_artifacts(snapshot_path=EXP05_MODEL_SNAPSHOT, expected_hashes=None):
    snapshot = Path(snapshot_path)
    if snapshot.name != EXP05_MODEL_REVISION:
        raise ValueError("EXP-05 model snapshot path must end in the frozen revision")
    hashes = expected_hashes or EXP05_MODEL_HASHES
    actual = {}
    for filename, expected in hashes.items():
        path = snapshot / filename
        if not path.is_file():
            raise FileNotFoundError(f"EXP-05 local model artifact is missing: {path}")
        actual[filename] = sha256_file(path)
        if actual[filename] != expected:
            raise RuntimeError(f"EXP-05 model artifact hash mismatch: {filename}")
    return actual


def validate_candidate_source(candidate_cache):
    _require_equal(candidate_cache.get("query_count"), EXP05_QUERY_COUNT, "cache query count")
    _require_equal(candidate_cache.get("pool_k"), EXP05_POOL_K, "cache pool_k")
    _require_equal(
        candidate_cache.get("fingerprint_algorithm"),
        EXP03_CANDIDATE_FINGERPRINT_ALGORITHM,
        "candidate fingerprint algorithm",
    )
    records = candidate_cache.get("records", [])
    validate_candidate_pools(records, EXP05_QUERY_COUNT, EXP05_POOL_K)
    actual = fingerprint_candidate_pools(records)
    _require_equal(actual, EXP05_CANDIDATE_FINGERPRINT, "computed candidate fingerprint")
    _require_equal(
        candidate_cache.get("candidate_cache_fingerprint"),
        EXP05_CANDIDATE_FINGERPRINT,
        "persisted candidate fingerprint",
    )
    query_fingerprint = fingerprint_records(
        [
            {"id": _record_query_id(record), "text": normalize_query_q0(record.get("query", ""))}
            for record in records
        ]
    )
    _require_equal(query_fingerprint, EXP05_QUERY_FINGERPRINT, "computed query fingerprint")
    return records


def validate_fold_source(fold_manifest, query_ids):
    _require_equal(fold_manifest.get("assignment_algorithm"), EXP03_FOLD_ALGORITHM, "fold algorithm")
    _require_equal(fold_manifest.get("fold_count"), EXP05_FOLD_COUNT, "persisted fold count")
    _require_equal(tuple(fold_manifest.get("fold_sizes", [])), EXP05_FOLD_SIZES, "persisted fold sizes")
    items = fold_manifest.get("mapping", [])
    mapping = {item["query_id"]: int(item["fold"]) for item in items}
    if len(mapping) != len(items):
        raise ValueError("EXP-05 fold source contains duplicate query IDs")
    _require_equal(fingerprint_fold_mapping(mapping), EXP05_FOLD_FINGERPRINT, "computed fold fingerprint")
    _require_equal(fold_manifest.get("mapping_fingerprint"), EXP05_FOLD_FINGERPRINT, "persisted fold fingerprint")
    validate_fold_mapping(mapping, query_ids, EXP05_FOLD_COUNT)
    return mapping


def validate_frozen_corpus(config, project_root):
    project_root = Path(project_root)
    laws_config = project_root / config["paths"]["laws_config"]
    law_files, inventory, total = load_frozen_baseline_corpus(
        laws_config,
        project_root / "backend/data",
        config["corpus"],
    )
    actual_fingerprint = fingerprint_records(
        [
            {"key": item["key"], "count": item["count"], "sha256": item["sha256"]}
            for item in inventory
        ]
    )
    _require_equal(total, 1811, "corpus document count")
    _require_equal(actual_fingerprint, EXP05_CORPUS_FINGERPRINT, "computed corpus fingerprint")
    documents = load_documents(
        law_files,
        document_representation="d0",
        bm25_tokenizer="whitespace",
        laws_config_path=laws_config,
    )
    counts = Counter(document["standard_law_id"] for document in documents)
    if len(documents) != 1811 or len(counts) != 1811 or any(count != 1 for count in counts.values()):
        raise RuntimeError("EXP-05 frozen D0 corpus must contain 1,811 unique document IDs")
    if any(document["bm25_text"] != document["vector_text"] for document in documents):
        raise RuntimeError("EXP-05 D0 BM25/vector text parity failed")
    return {document["standard_law_id"]: document["vector_text"] for document in documents}, inventory


def build_frozen_hybrid_top30(record):
    ranking = fuse_cached_query(record, 0.5, stored_k=EXP05_CANDIDATE_COUNT)
    if len(ranking) != EXP05_CANDIDATE_COUNT:
        raise RuntimeError("EXP-05 requires exactly 30 fused candidates per query")
    identifiers = [item["standard_law_id"] for item in ranking]
    if len(set(identifiers)) != EXP05_CANDIDATE_COUNT:
        raise RuntimeError("EXP-05 fused top-30 candidate IDs must be unique")
    return ranking


def build_inference_records(candidate_records, document_by_id, require_full=False):
    inference_records = []
    controls = []
    ordered = sorted(candidate_records, key=lambda record: (str(type(_record_query_id(record))), _record_query_id(record)))
    for record in ordered:
        query_id = _record_query_id(record)
        query = normalize_query_q0(record.get("query", ""))
        control = build_frozen_hybrid_top30(record)
        controls.append({"query_id": query_id, "top_30": control})
        for item in control:
            identifier = item["standard_law_id"]
            if identifier not in document_by_id:
                raise RuntimeError(f"EXP-05 candidate is absent from frozen D0 corpus: {identifier}")
            inference_records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "standard_law_id": identifier,
                    "document": document_by_id[identifier],
                    "original_hybrid_rank": int(item["rank"]),
                }
            )
    validate_inference_records(inference_records)
    if require_full:
        _require_equal(len(candidate_records), EXP05_QUERY_COUNT, "inference query count")
        _require_equal(len(inference_records), EXP05_INFERENCE_PAIR_COUNT, "inference pair count")
    return inference_records, controls


def validate_inference_records(records):
    for record in records:
        if set(record) != EXP05_INFERENCE_KEYS:
            raise ValueError("EXP-05 scorer requires the exact GT-free inference record schema")
        if not isinstance(record["query"], str) or not isinstance(record["document"], str):
            raise ValueError("EXP-05 query and D0 document inputs must be strings")
        if not isinstance(record["standard_law_id"], str) or not record["standard_law_id"]:
            raise ValueError("EXP-05 candidate ID must be a non-empty string")
        rank = record["original_hybrid_rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("EXP-05 original Hybrid rank must be a positive integer")
    return True


def _peak_rss_mb():
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def load_local_reranker(
    snapshot_path=EXP05_MODEL_SNAPSHOT,
    tokenizer_class=None,
    model_class=None,
    torch_module=None,
):
    if torch_module is None:
        import torch as torch_module
    if tokenizer_class is None or model_class is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer_class = tokenizer_class or AutoTokenizer
        model_class = model_class or AutoModelForSequenceClassification
    started = time.perf_counter()
    peak_before = _peak_rss_mb()
    tokenizer = tokenizer_class.from_pretrained(str(snapshot_path), local_files_only=True)
    model = model_class.from_pretrained(
        str(snapshot_path),
        local_files_only=True,
        dtype=torch_module.float32,
        use_safetensors=True,
    )
    model.to(device=EXP05_DEVICE, dtype=torch_module.float32)
    model.eval()
    elapsed = time.perf_counter() - started
    metadata = {
        "load_seconds": float(elapsed),
        "peak_rss_before_mb": float(peak_before),
        "peak_rss_after_mb": float(_peak_rss_mb()),
        "architecture": model.__class__.__name__,
        "device": EXP05_DEVICE,
        "dtype": EXP05_DTYPE,
    }
    return tokenizer, model, torch_module, metadata


def audit_token_lengths(inference_records, tokenizer):
    validate_inference_records(inference_records)
    special = int(tokenizer.num_special_tokens_to_add(pair=True))
    query_cache = {}
    document_cache = {}
    lengths = []
    query_lengths = []
    truncations = 0
    query_threats = 0
    query_preservation_failures = 0
    for record in inference_records:
        query = record["query"]
        document = record["document"]
        if query not in query_cache:
            query_cache[query] = tokenizer(
                query, add_special_tokens=False, truncation=False
            )["input_ids"]
        if document not in document_cache:
            document_cache[document] = tokenizer(
                document, add_special_tokens=False, truncation=False
            )["input_ids"]
        query_ids = query_cache[query]
        document_ids = document_cache[document]
        length = len(query_ids) + len(document_ids) + special
        lengths.append(length)
        query_lengths.append(len(query_ids))
        query_threats += int(len(query_ids) + special >= EXP05_MAX_LENGTH)
        encoded = tokenizer(
            query,
            document,
            add_special_tokens=True,
            max_length=EXP05_MAX_LENGTH,
            truncation=EXP05_TRUNCATION,
            padding=False,
        )
        sequence_ids = encoded.sequence_ids()
        query_preservation_failures += int(
            sum(value == 0 for value in sequence_ids) != len(query_ids)
        )
        truncations += int(length > EXP05_MAX_LENGTH)
    values = np.asarray(lengths, dtype=np.int64)
    percentiles = np.percentile(values, (50, 75, 90, 95, 99), method="linear")
    return {
        "total_pairs": int(values.size),
        "min": int(values.min()),
        "median": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "p99": float(percentiles[4]),
        "max": int(values.max()),
        "pairs_over_512": int(np.count_nonzero(values > 512)),
        "pairs_over_1024": int(np.count_nonzero(values > 1024)),
        "truncation_count": int(truncations),
        "truncation_rate": float(truncations / values.size),
        "max_query_tokens_without_specials": int(max(query_lengths)),
        "query_budget_threats": int(query_threats),
        "query_preservation_failures": int(query_preservation_failures),
        "truncation_side": "document_only",
    }


def enforce_query_preservation(token_audit):
    for field in ("query_budget_threats", "query_preservation_failures"):
        if token_audit.get(field) != 0:
            raise RuntimeError(f"EXP-05 query-preservation protocol failed: {field}")
    return True


def score_inference_records(inference_records, tokenizer, model, torch_module):
    validate_inference_records(inference_records)
    scored = []
    started = time.perf_counter()
    peak_before = _peak_rss_mb()
    with torch_module.inference_mode():
        for start in range(0, len(inference_records), EXP05_BATCH_SIZE):
            batch = inference_records[start : start + EXP05_BATCH_SIZE]
            encoded = tokenizer(
                [item["query"] for item in batch],
                [item["document"] for item in batch],
                max_length=EXP05_MAX_LENGTH,
                truncation=EXP05_TRUNCATION,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(EXP05_DEVICE) for key, value in encoded.items()}
            logits = model(**encoded).logits
            expected_shape = (len(batch), 1)
            if logits.ndim != 2 or tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    "EXP-05 model logits must have exact shape "
                    f"{expected_shape}; got {tuple(logits.shape)}"
                )
            scores = logits[:, 0].detach().cpu().tolist()
            for record, score in zip(batch, scores):
                score = float(score)
                if not math.isfinite(score):
                    raise ValueError("EXP-05 reranker produced NaN or Infinity")
                scored.append({**record, "reranker_score": score})
    elapsed = time.perf_counter() - started
    return scored, {
        "inference_seconds": float(elapsed),
        "pairs_per_second": float(len(scored) / elapsed) if elapsed else None,
        "pair_count": len(scored),
        "peak_rss_before_mb": float(peak_before),
        "peak_rss_after_mb": float(_peak_rss_mb()),
    }


def rerank_scored_candidates(scored_records):
    for record in scored_records:
        if set(record) != EXP05_SCORED_KEYS:
            raise ValueError("EXP-05 reranking requires the exact scored inference schema")
        if not math.isfinite(float(record["reranker_score"])):
            raise ValueError("EXP-05 reranker scores must be finite")
    return sorted(
        scored_records,
        key=lambda item: (
            -float(item["reranker_score"]),
            int(item["original_hybrid_rank"]),
            item["standard_law_id"],
        ),
    )


def fingerprint_predictions(predictions):
    payload = json.dumps(
        predictions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_prediction_artifact(scored_records, controls, candidate_count=EXP05_CANDIDATE_COUNT):
    by_query = defaultdict(list)
    for record in scored_records:
        by_query[record["query_id"]].append(record)
    control_by_query = {item["query_id"]: item["top_30"] for item in controls}
    if set(by_query) != set(control_by_query):
        raise ValueError("EXP-05 treatment/control query sets must be identical")
    predictions = []
    for query_id in sorted(by_query, key=lambda value: (str(type(value)), value)):
        treatment_rows = rerank_scored_candidates(by_query[query_id])
        control_rows = control_by_query[query_id]
        if len(treatment_rows) != candidate_count or len(control_rows) != candidate_count:
            raise ValueError("EXP-05 treatment/control must each contain exactly 30 candidates")
        treatment_ids = {row["standard_law_id"] for row in treatment_rows}
        control_ids = {row["standard_law_id"] for row in control_rows}
        if treatment_ids != control_ids or len(treatment_ids) != candidate_count:
            raise ValueError("EXP-05 treatment and control candidate sets must be identical")
        predictions.append(
            {
                "query_id": query_id,
                "control_top_30": [
                    {
                        "rank": int(row["rank"]),
                        "standard_law_id": row["standard_law_id"],
                        "score": float(row["score"]),
                    }
                    for row in control_rows
                ],
                "treatment_top_30": [
                    {
                        "rank": rank,
                        "standard_law_id": row["standard_law_id"],
                        "reranker_score": float(row["reranker_score"]),
                        "original_hybrid_rank": int(row["original_hybrid_rank"]),
                    }
                    for rank, row in enumerate(treatment_rows, 1)
                ],
            }
        )
    artifact = {
        "schema_version": "exp05_prediction_artifact_v1",
        "protocol_id": EXP05_PROTOCOL_ID,
        "experiment_id": EXP05_EXPERIMENT_ID,
        "control": EXP05_CONTROL,
        "treatment": EXP05_TREATMENT,
        "query_count": len(predictions),
        "candidate_count_per_query": candidate_count,
        "frozen": True,
        "predictions": predictions,
    }
    artifact["prediction_fingerprint"] = fingerprint_predictions(predictions)
    validate_prediction_artifact(artifact, expected_query_count=len(predictions))
    return json.loads(json.dumps(artifact, ensure_ascii=False, allow_nan=False))


def validate_prediction_artifact(artifact, expected_query_count=EXP05_QUERY_COUNT):
    if artifact.get("frozen") is not True:
        raise ValueError("EXP-05 evaluation requires a frozen prediction artifact")
    _require_equal(artifact.get("protocol_id"), EXP05_PROTOCOL_ID, "prediction protocol")
    predictions = artifact.get("predictions", [])
    if len(predictions) != expected_query_count:
        raise ValueError("EXP-05 prediction artifact has an unexpected query count")
    query_ids = [item.get("query_id") for item in predictions]
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("EXP-05 prediction artifact contains duplicate query IDs")
    forbidden = set(GT_RESULT_KEYS) | {"ground_truth", "oracle", "metric_outcome"}

    def contains_forbidden(value):
        if isinstance(value, dict):
            return bool(forbidden & set(value)) or any(
                contains_forbidden(item) for item in value.values()
            )
        if isinstance(value, list):
            return any(contains_forbidden(item) for item in value)
        return False

    for prediction in predictions:
        if contains_forbidden(prediction):
            raise ValueError("EXP-05 prediction artifact must be GT-free")
        if set(prediction) != {"query_id", "control_top_30", "treatment_top_30"}:
            raise ValueError("EXP-05 prediction rows must use the exact GT-free schema")
        expected_candidates = int(artifact.get("candidate_count_per_query", 0))
        if (
            len(prediction["control_top_30"]) != expected_candidates
            or len(prediction["treatment_top_30"]) != expected_candidates
        ):
            raise ValueError("EXP-05 prediction rows have an invalid candidate count")
    expected = fingerprint_predictions(predictions)
    if artifact.get("prediction_fingerprint") != expected:
        raise ValueError("EXP-05 prediction artifact fingerprint mismatch")
    json.dumps(artifact, ensure_ascii=False, allow_nan=False)
    return True


def attach_ground_truth_after_prediction_freeze(
    candidate_records,
    prediction_artifact,
    fold_mapping,
):
    validate_prediction_artifact(prediction_artifact, expected_query_count=len(candidate_records))
    record_by_id = {_record_query_id(record): record for record in candidate_records}
    if len(record_by_id) != len(candidate_records):
        raise ValueError("EXP-05 candidate records must have unique query IDs")
    results = []
    for prediction in prediction_artifact["predictions"]:
        query_id = prediction["query_id"]
        record = record_by_id[query_id]
        result = {
            "query_id": query_id,
            "query": record.get("query", ""),
            "outer_fold": int(fold_mapping[query_id]),
        }
        for key in GT_RESULT_KEYS:
            if key in record:
                result[key] = record[key]
        result["metrics"] = {
            EXP05_CONTROL: {
                "top_k": [dict(item) for item in prediction["control_top_30"]]
            },
            EXP05_TREATMENT: {
                "top_k": [
                    {
                        "rank": item["rank"],
                        "standard_law_id": item["standard_law_id"],
                        "score": item["reranker_score"],
                        "original_hybrid_rank": item["original_hybrid_rank"],
                    }
                    for item in prediction["treatment_top_30"]
                ]
            },
        }
        results.append(result)
    return results


def validate_control_parity(metric_worlds, tolerance=EXP05_PARITY_TOLERANCE):
    actual = metric_worlds["closed_world"][EXP05_CONTROL][str(EXP05_PRIMARY_K)]
    errors = {
        name: abs(float(actual[name]) - float(expected))
        for name, expected in EXP05_CONTROL_AT_10.items()
    }
    maximum = max(errors.values())
    if maximum > tolerance:
        raise RuntimeError(f"EXP-05 control parity failed; maximum_metric_error={maximum}")
    return {"passed": True, "maximum_metric_error": maximum, "metric_errors": errors}


def query_metric_values(result, configuration, k=EXP05_PRIMARY_K):
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


def paired_bootstrap(deltas, samples=EXP05_BOOTSTRAP_SAMPLES, seed=EXP05_RANDOM_SEED):
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("EXP-05 paired bootstrap requires finite non-empty deltas")
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    chunk = 2_000
    for start in range(0, int(samples), chunk):
        size = min(chunk, int(samples) - start)
        indices = rng.integers(0, values.size, size=(size, values.size))
        means[start : start + size] = values[indices].mean(axis=1)
    lower, upper = np.percentile(means, (2.5, 97.5))
    return {
        "samples": int(samples),
        "seed": int(seed),
        "mean_delta": float(values.mean()),
        "ci_95_percentile": [float(lower), float(upper)],
    }


def paired_sign_flip(deltas, samples=EXP05_SIGN_FLIP_SAMPLES, seed=EXP05_RANDOM_SEED):
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("EXP-05 paired sign-flip requires finite non-empty deltas")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 5_000
    for start in range(0, int(samples), chunk):
        size = min(chunk, int(samples) - start)
        signs = rng.integers(0, 2, size=(size, values.size), dtype=np.int8) * 2 - 1
        flipped = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(flipped >= observed))
    return {
        "samples": int(samples),
        "seed": int(seed),
        "observed_absolute_mean": observed,
        "extreme_count": extreme,
        "p_value": (1 + extreme) / (int(samples) + 1),
        "two_sided": True,
    }


def build_statistical_analysis(evaluation_results):
    metric_names = ("weighted_recall", "recall", "ndcg", "mrr")
    deltas_by_metric = {name: [] for name in metric_names}
    per_query = []
    for result in evaluation_results:
        treatment = query_metric_values(result, EXP05_TREATMENT)
        control = query_metric_values(result, EXP05_CONTROL)
        deltas = {name: treatment[name] - control[name] for name in metric_names}
        for name in metric_names:
            deltas_by_metric[name].append(deltas[name])
        per_query.append(
            {"query_id": result["query_id"], "outer_fold": result["outer_fold"], "deltas": deltas}
        )
    metrics = {
        name: {
            "paired_bootstrap": paired_bootstrap(values),
            "paired_sign_flip": paired_sign_flip(values),
        }
        for name, values in deltas_by_metric.items()
    }
    primary = deltas_by_metric["weighted_recall"]
    wins = sum(value > 0.0 for value in primary)
    losses = sum(value < 0.0 for value in primary)
    ties = len(primary) - wins - losses
    harmful = [
        {"query_id": row["query_id"], "delta": row["deltas"]["weighted_recall"]}
        for row in per_query
        if row["deltas"]["weighted_recall"] < 0.0
    ]
    fold_means = {}
    for fold in range(EXP05_FOLD_COUNT):
        fold_values = [
            row["deltas"]["weighted_recall"]
            for row in per_query
            if row["outer_fold"] == fold
        ]
        if not fold_values:
            raise ValueError("EXP-05 stability reporting requires every frozen outer fold")
        fold_means[str(fold)] = float(np.mean(fold_values))
    return {
        "metrics": metrics,
        "per_query": per_query,
        "wr10_win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
        "fold_mean_wr10_deltas": fold_means,
        "harmful_queries": harmful,
    }


def evaluate_success_contract(metric_worlds, statistical_analysis):
    closed = metric_worlds["closed_world"]
    treatment_wr10 = closed[EXP05_TREATMENT]["10"]["weighted_recall"]
    control_wr10 = closed[EXP05_CONTROL]["10"]["weighted_recall"]
    primary = statistical_analysis["metrics"]["weighted_recall"]
    primary_deltas = [row["deltas"]["weighted_recall"] for row in statistical_analysis["per_query"]]
    positives = [value for value in primary_deltas if value > 0.0]
    after_largest = sum(primary_deltas) - (max(positives) if positives else 0.0)
    counts = statistical_analysis["wr10_win_tie_loss"]
    fold_means = statistical_analysis["fold_mean_wr10_deltas"]
    no_secondary_collapse = all(
        statistical_analysis["metrics"][name]["paired_bootstrap"]["ci_95_percentile"][1]
        >= 0.0
        for name in ("recall", "ndcg", "mrr")
    )
    conditions = {
        "treatment_closed_wr10_above_control": treatment_wr10 > control_wr10,
        "bootstrap_lower_above_zero": primary["paired_bootstrap"]["ci_95_percentile"][0] > 0.0,
        "sign_flip_p_below_0_05": primary["paired_sign_flip"]["p_value"] < 0.05,
        "wr10_wins_exceed_losses": counts["wins"] > counts["losses"],
        "positive_after_removing_largest_gain": after_largest > 0.0,
        "positive_in_at_least_three_folds": sum(value > 0.0 for value in fold_means.values()) >= 3,
        "no_secondary_ci_entirely_below_zero": no_secondary_collapse,
    }
    success = all(conditions.values())
    return {
        "conditions": conditions,
        "all_conditions_hold": success,
        "decision": "EXPLORATORY_SUCCESS" if success else "DO_NOT_PROMOTE",
        "result_scope": "exploratory",
        "automatic_runtime_promotion": False,
        "total_delta_after_removing_largest_positive_query": float(after_largest),
    }


def _source_paths(config_path, project_root):
    base = "backend/scripts/Evaluation"
    return [
        f"{base}/run_experiment.py",
        _relative_path(config_path, project_root),
        f"{base}/core/corpus.py",
        f"{base}/core/query.py",
        f"{base}/core/representations.py",
        f"{base}/retrieval/fusion.py",
        f"{base}/evaluation/metrics.py",
        f"{base}/reporting/manifest.py",
        f"{base}/reporting/serialization.py",
        f"{base}/experiments/exp03_static_fusion_cv.py",
        f"{base}/experiments/exp05_frozen_hybrid30_zero_shot_reranker.py",
        "backend/scripts/Tests/test_exp03_protocol.py",
        "backend/scripts/Tests/test_exp05_protocol.py",
    ]


def _validate_real_run_provenance(provenance):
    commit = provenance.get("git_commit")
    if not (
        isinstance(commit, str)
        and len(commit) in (40, 64)
        and all(character in "0123456789abcdefABCDEF" for character in commit)
    ):
        raise RuntimeError("A real EXP-05 run requires a valid committed git HEAD")
    dirty_fields = ("git_dirty", "tracked_code_dirty", "baseline_source_dirty")
    if any(provenance.get(field) is not False for field in dirty_fields):
        raise RuntimeError("A real EXP-05 run requires a completely clean worktree")
    return commit


def _verify_provenance_unchanged(
    project_root,
    source_paths,
    expected_commit,
    expected_source_files,
):
    current = git_provenance(project_root, source_paths)
    current_commit = _validate_real_run_provenance(current)
    if current_commit != expected_commit:
        raise RuntimeError("EXP-05 git HEAD changed during inference")
    current_source_files = _source_file_provenance(project_root, source_paths)
    if current_source_files != expected_source_files:
        raise RuntimeError("EXP-05 source/config files changed during inference")
    return True


def validate_exp05_artifacts(artifacts):
    if set(artifacts) != EXP05_ARTIFACT_FILES:
        raise ValueError("EXP-05 artifact payload must contain the exact frozen artifact set")
    prediction_artifact = artifacts["prediction_artifact.json"]
    validate_prediction_artifact(
        prediction_artifact,
        expected_query_count=EXP05_QUERY_COUNT,
    )
    _require_equal(
        prediction_artifact.get("candidate_count_per_query"),
        EXP05_CANDIDATE_COUNT,
        "artifact candidate count",
    )
    manifest = artifacts["run_manifest.json"]
    required_manifest = {
        "run_id",
        "timestamp",
        "protocol_id",
        "experiment_id",
        "evaluation_mode",
        "candidate_fingerprint",
        "corpus_fingerprint",
        "query_fingerprint",
        "fold_mapping_fingerprint",
        "model_id",
        "model_revision",
        "model_file_sha256",
        "python_version",
        "package_versions",
        "device",
        "dtype",
        "batch_size",
        "max_length",
        "truncation",
        "candidate_count",
        "prediction_fingerprint",
        "git_commit",
        "git_dirty",
        "source_files",
    }
    missing = sorted(required_manifest - set(manifest))
    if missing:
        raise ValueError(f"EXP-05 run manifest is missing required fields: {missing}")
    _require_equal(
        manifest.get("prediction_fingerprint"),
        prediction_artifact.get("prediction_fingerprint"),
        "manifest prediction fingerprint",
    )
    json.dumps(artifacts, ensure_ascii=False, allow_nan=False)
    return True


def run_exp05_frozen_hybrid30_zero_shot_reranker(config_path, max_queries=None):
    if max_queries is not None:
        raise ValueError("EXP-05 requires the complete frozen 107-query candidate cache")
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    _validate_exp05_protocol(config)

    project_root = Path(__file__).resolve().parents[4]
    candidate_path = project_root / EXP05_CANDIDATE_PATH
    fold_path = project_root / EXP05_FOLD_PATH
    source_manifest_path = project_root / EXP05_SOURCE_MANIFEST_PATH
    with open(candidate_path, "r", encoding="utf-8") as handle:
        candidate_cache = json.load(handle)
    with open(fold_path, "r", encoding="utf-8") as handle:
        fold_manifest = json.load(handle)
    with open(source_manifest_path, "r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)

    candidate_records = validate_candidate_source(candidate_cache)
    fold_mapping = validate_fold_source(
        fold_manifest, [_record_query_id(record) for record in candidate_records]
    )
    _require_equal(source_manifest.get("combined_corpus_fingerprint"), EXP05_CORPUS_FINGERPRINT, "source corpus fingerprint")
    _require_equal(source_manifest.get("query_representation_fingerprint"), EXP05_QUERY_FINGERPRINT, "source query fingerprint")
    _require_equal(source_manifest.get("fold_mapping_fingerprint"), EXP05_FOLD_FINGERPRINT, "source fold fingerprint")
    document_by_id, corpus_inventory = validate_frozen_corpus(config, project_root)
    inference_records, controls = build_inference_records(
        candidate_records, document_by_id, require_full=True
    )

    source_paths = _source_paths(config_path, project_root)
    pre_run_provenance = git_provenance(project_root, source_paths)
    pre_run_commit = _validate_real_run_provenance(pre_run_provenance)
    pre_run_source_files = _source_file_provenance(project_root, source_paths)

    model_hashes = validate_model_artifacts(config["model"]["snapshot_path"])
    tokenizer, model, torch_module, load_timing = load_local_reranker(
        config["model"]["snapshot_path"]
    )
    token_audit = audit_token_lengths(inference_records, tokenizer)
    enforce_query_preservation(token_audit)
    scored_records, inference_timing = score_inference_records(
        inference_records, tokenizer, model, torch_module
    )
    _verify_provenance_unchanged(
        project_root,
        source_paths,
        pre_run_commit,
        pre_run_source_files,
    )

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"{EXP05_PROTOCOL_ID}_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    prediction_artifact = freeze_prediction_artifact(scored_records, controls)

    # The fingerprinted GT-free artifact is validated before any GT-bearing
    # candidate record is joined for evaluation.
    evaluation_results = attach_ground_truth_after_prediction_freeze(
        candidate_records, prediction_artifact, fold_mapping
    )
    control_worlds = build_metric_world_report(
        evaluation_results,
        [EXP05_CONTROL],
        EXP05_REPORTED_K,
        alpha_core=3,
        alpha_support=1,
    )
    parity = validate_control_parity(control_worlds)
    metric_worlds = build_metric_world_report(
        evaluation_results,
        [EXP05_CONTROL, EXP05_TREATMENT],
        EXP05_REPORTED_K,
        alpha_core=3,
        alpha_support=1,
    )
    statistical_analysis = build_statistical_analysis(evaluation_results)
    success = evaluate_success_contract(metric_worlds, statistical_analysis)

    manifest_provenance = {
        key: value
        for key, value in pre_run_provenance.items()
        if key not in {"baseline_source_dirty", "baseline_source_status"}
    }
    manifest = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "protocol_id": EXP05_PROTOCOL_ID,
        "experiment_id": EXP05_EXPERIMENT_ID,
        "evaluation_mode": EXP05_MODE,
        "control_protocol": BASELINE_PROTOCOL_ID,
        "candidate_source_run": EXP05_SOURCE_RUN_ID,
        "candidate_path": EXP05_CANDIDATE_PATH,
        "candidate_file_sha256": sha256_file(candidate_path),
        "candidate_fingerprint": EXP05_CANDIDATE_FINGERPRINT,
        "corpus_fingerprint": EXP05_CORPUS_FINGERPRINT,
        "query_fingerprint": EXP05_QUERY_FINGERPRINT,
        "fold_mapping_fingerprint": EXP05_FOLD_FINGERPRINT,
        "model_id": EXP05_MODEL_ID,
        "model_revision": EXP05_MODEL_REVISION,
        "model_snapshot_path": EXP05_MODEL_SNAPSHOT,
        "model_file_sha256": model_hashes,
        "python_version": platform.python_version(),
        "package_versions": {
            name: safe_package_version(name)
            for name in ("torch", "transformers", "tokenizers")
        },
        "device": EXP05_DEVICE,
        "dtype": EXP05_DTYPE,
        "batch_size": EXP05_BATCH_SIZE,
        "max_length": EXP05_MAX_LENGTH,
        "truncation": EXP05_TRUNCATION,
        "padding": True,
        "score": "raw_scalar_classification_logit",
        "candidate_count": EXP05_CANDIDATE_COUNT,
        "query_count": EXP05_QUERY_COUNT,
        "inference_pair_count": EXP05_INFERENCE_PAIR_COUNT,
        "prediction_fingerprint": prediction_artifact["prediction_fingerprint"],
        "rerank_tie_break": list(EXP05_RERANK_TIE_BREAK),
        "model_load_timing": load_timing,
        "inference_timing": inference_timing,
        "truncation_audit": token_audit,
        "control_parity": parity,
        "corpus_inventory": corpus_inventory,
        "config": {
            "path": _relative_path(config_path, project_root),
            "sha256": sha256_file(config_path),
        },
        "source_files": pre_run_source_files,
        **manifest_provenance,
        "exp05_source_dirty": pre_run_provenance["baseline_source_dirty"],
        "exp05_source_status": pre_run_provenance["baseline_source_status"],
    }
    summary = {
        "run_id": run_id,
        "protocol_id": EXP05_PROTOCOL_ID,
        "experiment_id": EXP05_EXPERIMENT_ID,
        "primary_world": "closed_world",
        "primary_metric": "weighted_recall",
        "primary_k": EXP05_PRIMARY_K,
        "reported_k": list(EXP05_REPORTED_K),
        "closed_world": metric_worlds["closed_world"],
        "open_world": metric_worlds["open_world"],
        "control_parity": parity,
        "wr10_win_tie_loss": statistical_analysis["wr10_win_tie_loss"],
        "fold_mean_wr10_deltas": statistical_analysis["fold_mean_wr10_deltas"],
        "harmful_queries": statistical_analysis["harmful_queries"],
        "truncation_audit": token_audit,
        "model_load_timing": load_timing,
        "inference_timing": inference_timing,
        "success_contract": success,
    }
    statistical_artifact = {
        "run_id": run_id,
        "protocol_id": EXP05_PROTOCOL_ID,
        **statistical_analysis,
        "success_contract": success,
    }
    artifacts = {
        "run_manifest.json": manifest,
        "prediction_artifact.json": prediction_artifact,
        "summary_metrics.json": summary,
        "statistical_analysis.json": statistical_artifact,
    }
    validate_exp05_artifacts(artifacts)
    run_dir = project_root / config["paths"]["output_dir"] / "runs" / run_id
    return write_exp05_artifacts(run_dir, config_path, artifacts)
