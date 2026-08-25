import datetime
import json
import platform
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import yaml

from ..core.corpus import load_documents, load_frozen_baseline_corpus, sha256_file
from ..core.ground_truth import build_canonical_ground_truth
from ..core.query import build_query_text
from ..evaluation.coverage import compute_ground_truth_coverage
from ..evaluation.latency import build_latency_aggregates
from ..evaluation.metrics import build_metric_world_report
from ..reporting.manifest import (
    enforce_full_run_provenance,
    fingerprint_records,
    git_provenance,
    resolved_model_revision,
    safe_package_version,
    verify_model_revision,
)
from ..reporting.serialization import assemble_baseline_summary, write_baseline_artifacts
from ..retrieval.bm25 import build_bm25_index, search_bm25
from ..retrieval.dense import build_dense_retriever, encode_dense_query, search_faiss
from ..retrieval.fusion import fuse_pools


BASELINE_MODE = "baseline"
BASELINE_CONFIGURATIONS = ["BM25", "FAISS", "Hybrid_Static_Fixed"]
BASELINE_REPORTED_K = (5, 10, 20)
BASELINE_PRIMARY_K = 10
BASELINE_PRIMARY_WORLD = "closed_world"


def get_enabled_configurations(mode):
    if mode == BASELINE_MODE:
        return list(BASELINE_CONFIGURATIONS)
    return [
        "BM25",
        "FAISS",
        "Hybrid_Static_Fixed",
        "Hybrid_Static_Opt",
        "Hybrid_Dynamic",
        "Hybrid_Dynamic_Rerank",
    ]


def should_initialize_reranker(mode):
    return mode != BASELINE_MODE


def should_run_tuning(mode):
    return mode != BASELINE_MODE


def _validate_baseline_protocol(config):
    evaluation = config["evaluation"]
    if evaluation.get("mode") != BASELINE_MODE:
        raise ValueError("run_baseline_evaluation_pipeline requires evaluation.mode=baseline")
    query_representation = evaluation.get("query_representation")
    document_representation = evaluation.get("document_representation")
    temporal_policy = evaluation.get("temporal_filter")
    if query_representation != "q0":
        raise ValueError("Baseline mode requires query_representation=q0")
    if document_representation not in {"d0", "d1"}:
        raise ValueError("Baseline mode requires document_representation=d0 or d1")
    if temporal_policy != "none":
        raise ValueError("Baseline B0/B1/E1 requires temporal_filter=none")

    bm25_tokenizer = config.get("bm25", {}).get("tokenizer")
    if bm25_tokenizer != "whitespace":
        raise ValueError("Baseline mode requires deterministic bm25.tokenizer=whitespace")
    fusion_method = config["fusion"].get("method")
    static_w_bm25 = float(config["fusion"].get("static_w_bm25"))
    if fusion_method != "minmax" or static_w_bm25 != 0.5:
        raise ValueError("Baseline E1 requires minmax fusion with static_w_bm25=0.5")

    metric_protocol = config.get("metric_protocol", {})
    primary_world = metric_protocol.get("primary_world")
    primary_k = int(metric_protocol.get("primary_k", 0))
    reported_k = tuple(int(value) for value in metric_protocol.get("reported_k", []))
    if (
        primary_world != BASELINE_PRIMARY_WORLD
        or primary_k != BASELINE_PRIMARY_K
        or reported_k != BASELINE_REPORTED_K
    ):
        raise ValueError(
            "Baseline metric protocol requires primary_world=closed_world, "
            "primary_k=10, reported_k=[5, 10, 20]"
        )
    if int(evaluation["k"]) < max(reported_k):
        raise ValueError("Stored retrieval depth must cover every reported metric K")
    return {
        "query_representation": query_representation,
        "document_representation": document_representation,
        "temporal_policy": temporal_policy,
        "bm25_tokenizer": bm25_tokenizer,
        "fusion_method": fusion_method,
        "static_w_bm25": static_w_bm25,
        "primary_world": primary_world,
        "primary_k": primary_k,
        "reported_k": reported_k,
    }


def _source_paths():
    base = "backend/scripts/Evaluation"
    return [
        f"{base}/run_rag_evaluation.py",
        f"{base}/run_experiment.py",
        f"{base}/eval_config.yaml",
        f"{base}/configs/baseline_d0.yaml",
        f"{base}/requirements-baseline.txt",
        f"{base}/core/normalization.py",
        f"{base}/core/query.py",
        f"{base}/core/representations.py",
        f"{base}/core/corpus.py",
        f"{base}/core/ground_truth.py",
        f"{base}/retrieval/bm25.py",
        f"{base}/retrieval/dense.py",
        f"{base}/retrieval/fusion.py",
        f"{base}/retrieval/temporal.py",
        f"{base}/evaluation/metrics.py",
        f"{base}/evaluation/coverage.py",
        f"{base}/evaluation/latency.py",
        f"{base}/reporting/manifest.py",
        f"{base}/reporting/serialization.py",
        f"{base}/experiments/baseline.py",
        "backend/scripts/Tests/test_baseline_protocol.py",
    ]


def _warm_up(
    selected_queries,
    warmup_queries,
    query_representation,
    bm25_index,
    embedder,
    faiss_index,
    documents,
    pool_k,
    static_w_bm25,
    fusion_method,
):
    if warmup_queries < 1:
        raise ValueError("Baseline latency protocol requires at least one warm-up query")
    for warmup_index in range(warmup_queries):
        record = selected_queries[warmup_index % len(selected_queries)]
        query_text = build_query_text(record, query_representation)
        bm25_pool = search_bm25(bm25_index, documents, query_text, pool_k)
        query_embedding = encode_dense_query(embedder, query_text)
        faiss_pool = search_faiss(faiss_index, query_embedding, documents, pool_k)
        fuse_pools(bm25_pool, faiss_pool, static_w_bm25, fusion_method)


def _retrieve_query(
    record,
    index,
    query_representation,
    documents,
    bm25_index,
    embedder,
    faiss_index,
    pool_k,
    stored_k,
    static_w_bm25,
    fusion_method,
    corpus_document_ids,
    alpha_core,
    alpha_support,
):
    query_text = build_query_text(record, query_representation)

    started = time.perf_counter_ns()
    bm25_pool = search_bm25(bm25_index, documents, query_text, pool_k)
    bm25_retrieval_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    started = time.perf_counter_ns()
    query_embedding = encode_dense_query(embedder, query_text)
    dense_encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0

    started = time.perf_counter_ns()
    faiss_pool = search_faiss(faiss_index, query_embedding, documents, pool_k)
    faiss_search_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    dense_total_ms = dense_encode_ms + faiss_search_ms

    started = time.perf_counter_ns()
    hybrid = fuse_pools(bm25_pool, faiss_pool, static_w_bm25, fusion_method)
    hybrid_pool = [
        {"rank": rank, "standard_law_id": standard_id, "score": float(score)}
        for rank, (standard_id, score) in enumerate(hybrid[:stored_k], 1)
    ]
    fusion_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    hybrid_serial_ms = bm25_retrieval_ms + dense_total_ms + fusion_ms
    hybrid_parallel_estimate_ms = max(bm25_retrieval_ms, dense_total_ms) + fusion_ms
    gt_contract = build_canonical_ground_truth(
        record,
        corpus_document_ids,
        alpha_core=alpha_core,
        alpha_support=alpha_support,
    )
    return {
        "query_id": record.get("id", index),
        "query": query_text,
        **gt_contract,
        "latency_ms": {
            "bm25_retrieval_ms": bm25_retrieval_ms,
            "dense_encode_ms": dense_encode_ms,
            "faiss_search_ms": faiss_search_ms,
            "dense_total_ms": dense_total_ms,
            "fusion_ms": fusion_ms,
            "hybrid_serial_ms": hybrid_serial_ms,
            "hybrid_parallel_estimate_ms": hybrid_parallel_estimate_ms,
        },
        "metrics": {
            "BM25": {"top_k": bm25_pool[:stored_k]},
            "FAISS": {"top_k": faiss_pool[:stored_k]},
            "Hybrid_Static_Fixed": {"top_k": hybrid_pool},
        },
    }


def run_baseline_evaluation_pipeline(config_path, max_queries=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    protocol = _validate_baseline_protocol(config)
    evaluation = config["evaluation"]
    random_seed = int(evaluation.get("random_seed", 42))
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    project_root = Path(__file__).resolve().parents[4]
    output_dir = project_root / config["paths"]["output_dir"]
    laws_config_path = project_root / config["paths"]["laws_config"]
    benchmark_path = project_root / config["paths"]["eval_file"]
    schema_path = output_dir / "gt_schema_v2.json"

    from ..validate_benchmark import validate_benchmark_file

    validate_benchmark_file(benchmark_path)
    law_files, corpus_inventory, corpus_count = load_frozen_baseline_corpus(
        laws_config_path, output_dir, config["corpus"]
    )
    documents = load_documents(
        law_files,
        document_representation=protocol["document_representation"],
        bm25_tokenizer=protocol["bm25_tokenizer"],
        laws_config_path=laws_config_path,
    )
    if len(documents) != corpus_count:
        raise ValueError(
            f"Document count changed during representation build: {len(documents)} != {corpus_count}"
        )
    if any(document["bm25_text"] != document["vector_text"] for document in documents):
        raise AssertionError("Baseline BM25 and dense must use the same document representation")

    with open(benchmark_path, "r", encoding="utf-8") as handle:
        benchmark = json.load(handle)
    selected_queries = benchmark[:max_queries] if max_queries is not None else benchmark
    if not selected_queries:
        raise ValueError("No benchmark queries selected")
    provenance = git_provenance(project_root, _source_paths())
    enforce_full_run_provenance(max_queries, len(benchmark), provenance)

    bm25_index = build_bm25_index(documents)
    embedding_model = config["models"]["embedding_model"]
    requested_revision = config["models"].get("embedding_revision")
    local_files_only = bool(config["models"].get("local_files_only", True))
    embedder, faiss_index, embedding_dimension = build_dense_retriever(
        documents,
        embedding_model,
        requested_revision,
        local_files_only=local_files_only,
        device="cpu",
    )
    resolved_revision = resolved_model_revision(embedder)
    verify_model_revision(requested_revision, resolved_revision)

    alpha_core = int(evaluation.get("alpha_core", 3))
    alpha_support = int(evaluation.get("alpha_support", 1))
    stored_k = int(evaluation["k"])
    pool_k = int(evaluation["pool_k"])
    corpus_document_ids = {document["standard_law_id"] for document in documents}
    gt_coverage = compute_ground_truth_coverage(benchmark, corpus_document_ids)
    warmup_queries = int(config.get("latency", {}).get("warmup_queries", 1))
    _warm_up(
        selected_queries,
        warmup_queries,
        protocol["query_representation"],
        bm25_index,
        embedder,
        faiss_index,
        documents,
        pool_k,
        protocol["static_w_bm25"],
        protocol["fusion_method"],
    )

    final_results = [
        _retrieve_query(
            record,
            index,
            protocol["query_representation"],
            documents,
            bm25_index,
            embedder,
            faiss_index,
            pool_k,
            stored_k,
            protocol["static_w_bm25"],
            protocol["fusion_method"],
            corpus_document_ids,
            alpha_core,
            alpha_support,
        )
        for index, record in enumerate(selected_queries)
    ]
    configurations = get_enabled_configurations(BASELINE_MODE)
    metric_worlds = build_metric_world_report(
        final_results,
        configurations,
        protocol["reported_k"],
        alpha_core=alpha_core,
        alpha_support=alpha_support,
    )
    latency_aggregates = build_latency_aggregates(final_results)

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"baseline_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / "runs" / run_id
    summary_metadata = {
        "run_id": run_id,
        "evaluation_mode": BASELINE_MODE,
        "query_representation": protocol["query_representation"],
        "document_representation": protocol["document_representation"],
        "temporal_filter": protocol["temporal_policy"],
        "corpus_count": corpus_count,
        "benchmark_query_count": len(benchmark),
        "evaluated_query_count": len(selected_queries),
        "primary_world": protocol["primary_world"],
        "primary_k": protocol["primary_k"],
        "metric_protocol": {
            "primary_world": protocol["primary_world"],
            "primary_k": protocol["primary_k"],
            "reported_k": list(protocol["reported_k"]),
            "stored_ranking_k": stored_k,
        },
        "embedding_model": embedding_model,
        "embedding_revision": resolved_revision,
        "embedding_requested_revision": requested_revision,
        "fusion_method": protocol["fusion_method"],
        "static_w_bm25": protocol["static_w_bm25"],
    }
    summary = assemble_baseline_summary(
        summary_metadata, metric_worlds, gt_coverage, latency_aggregates
    )
    results_output = {
        "run_id": run_id,
        "experiment_id": evaluation["experiment_id"],
        "evaluation_mode": BASELINE_MODE,
        "configurations": configurations,
        "results": final_results,
    }
    query_fingerprint = fingerprint_records([
        {
            "id": record.get("id"),
            "text": build_query_text(record, protocol["query_representation"]),
        }
        for record in selected_queries
    ])
    combined_corpus_fingerprint = fingerprint_records([
        {"key": item["key"], "count": item["count"], "sha256": item["sha256"]}
        for item in corpus_inventory
    ])
    manifest = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "evaluation_mode": BASELINE_MODE,
        "experiment_id": evaluation["experiment_id"],
        "query_count": len(selected_queries),
        "benchmark_query_count": len(benchmark),
        "corpus_count": corpus_count,
        "query_representation": protocol["query_representation"],
        "document_representation": protocol["document_representation"],
        "temporal_filter": protocol["temporal_policy"],
        "embedding_model": embedding_model,
        "embedding_requested_revision": requested_revision,
        "embedding_resolved_revision": resolved_revision,
        "embedding_revision_verified": requested_revision == resolved_revision,
        "embedding_local_files_only": local_files_only,
        "embedding_dimension": embedding_dimension,
        "embedding_normalized": True,
        "faiss_index_type": type(faiss_index).__name__,
        "bm25_tokenizer": protocol["bm25_tokenizer"],
        "bm25_tokenizer_version": None,
        "fusion_method": protocol["fusion_method"],
        "static_w_bm25": protocol["static_w_bm25"],
        "alpha_core": alpha_core,
        "alpha_support": alpha_support,
        "metric_protocol": summary["metric_protocol"],
        "ground_truth_coverage": gt_coverage,
        "latency_ms": latency_aggregates,
        "warmup_policy": {
            "enabled": True,
            "query_count": warmup_queries,
            "query_selection": "first_selected_queries_cycled_if_needed",
            "components": ["bm25_retrieval", "dense_encode", "faiss_search", "fusion"],
            "excluded_from_per_query_latency": True,
            "model_initialization_excluded": True,
            "corpus_embedding_build_excluded": True,
        },
        "benchmark": {"path": str(benchmark_path), "sha256": sha256_file(benchmark_path)},
        "gt_schema": {"path": str(schema_path), "sha256": sha256_file(schema_path)},
        "corpora": corpus_inventory,
        "combined_corpus_fingerprint": combined_corpus_fingerprint,
        "query_representation_fingerprint": query_fingerprint,
        **provenance,
        "python_version": platform.python_version(),
        "package_versions": {
            name: safe_package_version(name)
            for name in [
                "torch",
                "numpy",
                "faiss-cpu",
                "rank-bm25",
                "sentence-transformers",
                "PyYAML",
            ]
        },
        "random_seed": random_seed,
        "retrieval_configurations": configurations,
        "graded_relevance_preserved": all("gt_weights" in item for item in final_results),
    }
    results_path = write_baseline_artifacts(
        run_dir,
        config_path,
        results_output,
        summary,
        manifest,
    )
    print(f"BASELINE RUN: {run_id}")
    print(f"query_count={len(selected_queries)} corpus_count={corpus_count}")
    print(
        f"query_representation={protocol['query_representation']} "
        f"document_representation={protocol['document_representation']}"
    )
    print(
        f"temporal_filter={protocol['temporal_policy']} "
        f"embedding_dimension={embedding_dimension}"
    )
    print(f"retrieval_configurations={configurations}")
    print(f"manifest={run_dir / 'run_manifest.json'}")
    return results_path
