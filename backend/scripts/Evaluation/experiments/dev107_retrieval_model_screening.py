"""Development-only zero-shot retrieval/reranking model screening.

This runner intentionally evaluates only the frozen 107-query historical
development benchmark.  It never reads the independent-benchmark intake or
sampling artifacts.  Results are exploratory and cannot promote runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import time
import uuid
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from ..core.corpus import load_documents, load_frozen_baseline_corpus, sha256_file
from ..core.query import build_query_text
from ..evaluation.metrics import calculate_ndcg, is_match
from ..reporting.manifest import fingerprint_records, git_provenance, safe_package_version
from ..retrieval.bm25 import build_bm25_index, search_bm25
from ..retrieval.fusion import fuse_pools
from .exp05_frozen_hybrid30_zero_shot_reranker import (
    EXP05_CANDIDATE_FINGERPRINT,
    EXP05_CORPUS_FINGERPRINT,
    EXP05_MODEL_ID,
    EXP05_MODEL_HASHES,
    EXP05_MODEL_REVISION,
    EXP05_PARITY_TOLERANCE,
    EXP05_QUERY_FINGERPRINT,
    load_local_reranker,
    validate_candidate_source,
    validate_exp05_artifacts,
    validate_model_artifacts,
    validate_prediction_artifact,
)
from .exp03_static_fusion_cv import _source_file_provenance


PROTOCOL_ID = "dev107_retrieval_model_screening_v1"
SYSTEMS = (
    "BM25",
    "Vietnamese_SBERT",
    "Current_Hybrid",
    "BGE_M3",
    "BGE_M3_Hybrid_RRF",
)
RERANK_GENERATORS = ("Current_Hybrid", "BGE_M3_Hybrid_RRF")
STAT_METRICS = ("recall", "weighted_recall", "mrr", "ndcg")
EXPECTED_EXP05_CONTROL = {
    "recall": 0.08825383049544679,
    "weighted_recall": 0.07248115500412532,
    "mrr": 0.1900904910250704,
    "ndcg": 0.05286971847087022,
}
EXP05_OFFICIAL_INFERENCE_SECONDS = 4572.643784703003
EXP05_OFFICIAL_PAIR_COUNT = 3210
EMBEDDING_CACHE_SCHEMA = "dev107_embedding_cache_v2"
RETRIEVAL_CACHE_SCHEMA = "dev107_retrieval_candidates_v2"


class Dev107IntegrityError(RuntimeError):
    """Fail-closed integrity error with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str):
    raise Dev107IntegrityError(code, message)


def _strict_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_json(value) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentiles(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "document_count": len(documents),
        "query_count": len(records),
        "mean_ms": float(array.mean() * 1000.0),
        "p50_ms": float(np.percentile(array, 50) * 1000.0),
        "p95_ms": float(np.percentile(array, 95) * 1000.0),
    }


def _require_exact_keys(value: dict, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        missing = sorted(expected - set(value or {}))
        unknown = sorted(set(value or {}) - expected)
        raise ValueError(f"{label} keys mismatch; missing={missing}, unknown={unknown}")


def _config_fingerprint(config: dict) -> str:
    portable = {key: value for key, value in config.items() if not key.startswith("_")}
    return _sha256_json(portable)


def _validate_config(config: dict) -> None:
    _require_exact_keys(
        config,
        {
            "protocol", "evaluation", "frozen_inputs", "corpus", "bm25",
            "vietnamese_sbert", "bge_m3", "fusion", "reranker", "metrics",
            "statistics", "embedding_cache", "paths",
        },
        "screening config",
    )
    _require_exact_keys(
        config["protocol"],
        {"id", "result_scope", "confirmatory", "automatic_runtime_promotion"},
        "protocol",
    )
    _require_exact_keys(
        config["evaluation"],
        {
            "query_count", "query_representation", "document_representation",
            "corpus_count", "candidate_depths", "reported_k", "alpha_core",
            "alpha_support", "random_seed",
        },
        "evaluation",
    )
    _require_exact_keys(
        config["frozen_inputs"],
        {
            "development_path", "exp03_candidate_path", "exp03_candidate_fingerprint",
            "candidate_records_fingerprint", "exp05_prediction_path", "query_fingerprint",
            "corpus_fingerprint", "d0_template",
        },
        "frozen_inputs",
    )
    if config.get("protocol", {}).get("id") != PROTOCOL_ID:
        raise ValueError(f"protocol.id must be {PROTOCOL_ID}")
    if config["protocol"].get("result_scope") != "development_exploratory":
        raise ValueError("Screening must remain DEVELOPMENT / EXPLORATORY")
    if config["protocol"].get("confirmatory") is not False:
        raise ValueError("Screening cannot be confirmatory")
    if config["protocol"].get("automatic_runtime_promotion") is not False:
        raise ValueError("Screening cannot promote runtime")
    evaluation = config["evaluation"]
    expected = {
        "query_count": 107,
        "query_representation": "q0",
        "document_representation": "d0",
        "corpus_count": 1811,
        "candidate_depths": [20, 30, 50, 100],
        "reported_k": [5, 10, 20],
        "alpha_core": 3,
        "alpha_support": 1,
        "random_seed": 42,
    }
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise ValueError(f"evaluation.{key} must be {value!r}")
    if config["fusion"]["bge_hybrid"] != {"method": "rrf", "rrf_k": 60}:
        raise ValueError("BGE hybrid must use fixed RRF k=60")
    if config["fusion"]["current_hybrid"] != {
        "method": "minmax", "w_bm25": 0.5, "w_dense": 0.5
    }:
        raise ValueError("Current Hybrid must remain min-max 0.5/0.5")
    if config["fusion"].get("tie_break") != "score_desc_then_standard_law_id_asc":
        raise ValueError("Fusion tie-break mismatch")
    if config["reranker"]["max_length"] != 1024:
        raise ValueError("Reranker must preserve EXP-05 max_length=1024")
    if config["reranker"]["truncation"] != "only_second":
        raise ValueError("Reranker must preserve the query and truncate documents only")
    if config["embedding_cache"] != {
        "schema_version": EMBEDDING_CACHE_SCHEMA,
        "allow_legacy": False,
    }:
        raise ValueError("Unsupported embedding-cache configuration")
    if config["paths"]["output_dir"] != "backend/data/runs":
        raise ValueError("paths.output_dir must remain the canonical run root")
    if config["bm25"] != {"tokenizer": "whitespace"}:
        raise ValueError("BM25 tokenizer must remain whitespace")
    for name in ("vietnamese_sbert", "bge_m3", "reranker"):
        model = config[name]
        if model.get("local_files_only") is not True or model.get("device") != "cpu":
            raise ValueError(f"{name} must remain local-only on CPU")
        if model.get("dtype") != "float32":
            raise ValueError(f"{name} must remain float32")
    model_ids = [
        config["vietnamese_sbert"].get("id"),
        config["bge_m3"].get("id"),
        config["reranker"].get("id"),
    ]
    if None in model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("Screening model IDs must be present and unique")
    candidate_records_fingerprint = config["frozen_inputs"].get(
        "candidate_records_fingerprint"
    )
    if not (
        isinstance(candidate_records_fingerprint, str)
        and len(candidate_records_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in candidate_records_fingerprint)
    ):
        raise ValueError("Frozen candidate-record fingerprint must be canonical SHA256")


def _validate_candidate_and_source(
    cache: dict,
    candidate_path: Path,
    frozen: dict,
) -> tuple[list[dict], Path, dict]:
    try:
        records = validate_candidate_source(cache)
    except Exception as exc:
        _fail("CANDIDATE_INTEGRITY_FAILURE", str(exc))
    if _sha256_json(records) != frozen["candidate_records_fingerprint"]:
        _fail("CANDIDATE_INTEGRITY_FAILURE", "candidate record order/content changed")
    source_manifest_path = candidate_path.with_name("run_manifest.json")
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail("CANDIDATE_INTEGRITY_FAILURE", f"source manifest unavailable: {exc}")
    required = {
        "candidate_cache_fingerprint": frozen["exp03_candidate_fingerprint"],
        "combined_corpus_fingerprint": frozen["corpus_fingerprint"],
        "query_representation_fingerprint": frozen["query_fingerprint"],
    }
    for key, expected in required.items():
        if source_manifest.get(key) != expected:
            _fail("CANDIDATE_INTEGRITY_FAILURE", f"candidate source {key} mismatch")
    return records, source_manifest_path, source_manifest


def _load_inputs(config: dict, project_root: Path):
    frozen = config["frozen_inputs"]
    candidate_path = project_root / frozen["exp03_candidate_path"]
    cache = json.loads(candidate_path.read_text(encoding="utf-8"))
    records, source_manifest_path, source_manifest = _validate_candidate_and_source(
        cache, candidate_path, frozen
    )
    development_path = project_root / frozen["development_path"]
    benchmark = json.loads(development_path.read_text(encoding="utf-8"))
    query_view = [
        {"id": record.get("id"), "text": build_query_text(record, "q0")}
        for record in benchmark
    ]
    if fingerprint_records(query_view) != frozen["query_fingerprint"]:
        raise ValueError("Historical Q0 query fingerprint mismatch")
    record_query_view = [
        {"id": record.get("query_id"), "text": record.get("query", "")}
        for record in records
    ]
    if record_query_view != query_view:
        _fail(
            "CANDIDATE_INTEGRITY_FAILURE",
            "candidate query identity/order differs from the frozen benchmark",
        )

    laws, corpus_inventory, total = load_frozen_baseline_corpus(
        project_root / config["paths"]["laws_config"],
        project_root / config["paths"]["data_dir"],
        config["corpus"],
    )
    documents = load_documents(
        laws,
        document_representation="d0",
        bm25_tokenizer="whitespace",
        laws_config_path=project_root / config["paths"]["laws_config"],
    )
    if total != 1811 or len(documents) != 1811:
        raise ValueError("Frozen corpus must contain exactly 1,811 D0 documents")
    corpus_fingerprint = fingerprint_records(
        [
            {"key": item["key"], "count": item["count"], "sha256": item["sha256"]}
            for item in corpus_inventory
        ]
    )
    if corpus_fingerprint != frozen["corpus_fingerprint"]:
        raise ValueError("Frozen corpus fingerprint mismatch")
    benchmark_sha = sha256_file(development_path)
    return (
        records,
        documents,
        corpus_inventory,
        candidate_path,
        source_manifest_path,
        development_path,
        benchmark_sha,
    )


def _search_faiss(index, query_vector, documents, depth: int):
    scores, indices = index.search(query_vector.astype("float32"), depth)
    return [
        {
            "rank": rank,
            "standard_law_id": documents[int(index_value)]["standard_law_id"],
            "score": float(score),
        }
        for rank, (score, index_value) in enumerate(zip(scores[0], indices[0]), 1)
        if index_value >= 0
    ]


def _array_fingerprint(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _embedding_cache_context(
    config: dict,
    documents: list[dict],
    records: list[dict],
) -> dict:
    document_ids = [item["standard_law_id"] for item in documents]
    query_ids = [item["query_id"] for item in records]
    return {
        "document_count": len(document_ids),
        "query_count": len(query_ids),
        "corpus_fingerprint": config["frozen_inputs"]["corpus_fingerprint"],
        "document_representation_id": config["evaluation"]["document_representation"],
        "document_representation_fingerprint": fingerprint_records(
            [
                {
                    "standard_law_id": item["standard_law_id"],
                    "text": item["vector_text"],
                }
                for item in documents
            ]
        ),
        "query_fingerprint": config["frozen_inputs"]["query_fingerprint"],
        "config_fingerprint": _config_fingerprint(config),
        "document_ids_fingerprint": _sha256_json(document_ids),
        "query_ids_fingerprint": _sha256_json(query_ids),
    }


def _expected_embedding_metadata(
    model_name: str,
    model_config: dict,
    context: dict,
) -> dict:
    return {
        "cache_schema_version": EMBEDDING_CACHE_SCHEMA,
        "model_name": model_name,
        "model_id": model_config["id"],
        "model_revision": model_config["revision"],
        "embedding_dimension": int(model_config["dimension"]),
        "normalization": bool(model_config["normalize_embeddings"]),
        "pooling": model_config.get("pooling", "mean"),
        "query_instruction": model_config.get("query_instruction"),
        **context,
    }


def _resolve_local_snapshot(model_config: dict) -> Path:
    declared = model_config.get("snapshot_path")
    revision = model_config.get("revision")
    model_id = model_config.get("id")
    if not revision:
        _fail(
            "PRE_RUN_PROVENANCE_FAILURE",
            f"model revision is unknown for {model_id!r}",
        )
    if declared:
        snapshot = Path(declared).expanduser()
    else:
        if os.environ.get("HF_HUB_CACHE"):
            hub_root = Path(os.environ["HF_HUB_CACHE"]).expanduser()
        elif os.environ.get("HF_HOME"):
            hub_root = Path(os.environ["HF_HOME"]).expanduser() / "hub"
        elif os.environ.get("XDG_CACHE_HOME"):
            hub_root = Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
        else:
            hub_root = Path.home() / ".cache" / "huggingface" / "hub"
        repository = "models--" + str(model_id).replace("/", "--")
        snapshot = hub_root / repository / "snapshots" / str(revision)
    if snapshot.name != revision or not snapshot.is_dir():
        _fail(
            "PRE_RUN_PROVENANCE_FAILURE",
            f"local snapshot unavailable for {model_id}@{revision}",
        )
    return snapshot


def _write_embedding_cache(
    path: Path,
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    query_seconds: list[float],
    index_seconds: float,
    expected_metadata: dict,
    creation_source: dict,
) -> dict:
    metadata = {
        **expected_metadata,
        "document_shape": list(document_vectors.shape),
        "query_shape": list(query_vectors.shape),
        "dtype": str(document_vectors.dtype),
        "document_matrix_fingerprint": _array_fingerprint(document_vectors),
        "query_matrix_fingerprint": _array_fingerprint(query_vectors),
        "creation_source": creation_source,
    }
    metadata["metadata_fingerprint"] = _sha256_json(metadata)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez(
            handle,
            documents=document_vectors,
            queries=query_vectors,
            query_seconds=np.asarray(query_seconds, dtype=np.float64),
            index_seconds=np.asarray(float(index_seconds), dtype=np.float64),
            metadata_json=np.asarray(
                json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
            ),
        )
    os.replace(temporary, path)
    return metadata


def _load_embedding_cache(
    path: Path,
    expected_metadata: dict,
) -> tuple[np.ndarray, np.ndarray, list[float], float, dict]:
    for count_key in ("document_count", "query_count"):
        count = expected_metadata.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            _fail(
                "EMBEDDING_PROVENANCE_FAILURE",
                f"authoritative {count_key} is missing or invalid",
            )
    try:
        with np.load(path, allow_pickle=False) as cached:
            required = {
                "documents", "queries", "query_seconds", "index_seconds", "metadata_json"
            }
            if set(cached.files) != required:
                _fail(
                    "CACHE_PROVENANCE_INCOMPLETE",
                    f"{path.name} lacks the authoritative embedding-cache envelope",
                )
            documents = np.asarray(cached["documents"])
            queries = np.asarray(cached["queries"])
            query_seconds = np.asarray(cached["query_seconds"], dtype=np.float64)
            index_seconds = float(cached["index_seconds"])
            metadata = json.loads(str(cached["metadata_json"].item()))
    except Dev107IntegrityError:
        raise
    except Exception as exc:
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"cannot read {path.name}: {exc}")
    stored_fingerprint = metadata.pop("metadata_fingerprint", None)
    if stored_fingerprint != _sha256_json(metadata):
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} metadata integrity mismatch")
    metadata["metadata_fingerprint"] = stored_fingerprint
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            _fail(
                "EMBEDDING_PROVENANCE_FAILURE",
                f"{path.name} {key} mismatch",
            )
    expected_document_shape = (
        int(expected_metadata["document_count"]),
        int(expected_metadata["embedding_dimension"]),
    )
    expected_query_shape = (
        int(expected_metadata["query_count"]),
        int(expected_metadata["embedding_dimension"]),
    )
    if documents.shape != expected_document_shape or queries.shape != expected_query_shape:
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} matrix shape mismatch")
    if documents.dtype != np.float32 or queries.dtype != np.float32:
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} matrix dtype must be float32")
    if not np.isfinite(documents).all() or not np.isfinite(queries).all():
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} contains NaN/Infinity")
    if query_seconds.shape != (int(expected_metadata["query_count"]),) or not np.isfinite(query_seconds).all():
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} timing vector is invalid")
    if not math.isfinite(index_seconds):
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} index timing is invalid")
    if metadata.get("document_shape") != list(documents.shape):
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} document shape metadata mismatch")
    if metadata.get("query_shape") != list(queries.shape):
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} query shape metadata mismatch")
    if metadata.get("dtype") != "float32":
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} dtype metadata mismatch")
    if metadata.get("document_matrix_fingerprint") != _array_fingerprint(documents):
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} document matrix fingerprint mismatch")
    if metadata.get("query_matrix_fingerprint") != _array_fingerprint(queries):
        _fail("EMBEDDING_PROVENANCE_FAILURE", f"{path.name} query matrix fingerprint mismatch")
    creation_source = metadata.get("creation_source")
    if not isinstance(creation_source, dict) or not creation_source.get("runner_sha256"):
        _fail("CACHE_PROVENANCE_INCOMPLETE", f"{path.name} creation provenance missing")
    return documents, queries, query_seconds.tolist(), index_seconds, metadata


def _build_dense(model_path: str, documents: list[dict], queries: list[str], config: dict):
    import faiss
    from sentence_transformers import SentenceTransformer

    started = time.perf_counter()
    model = SentenceTransformer(model_path, device="cpu", local_files_only=True)
    model.max_seq_length = int(config["max_seq_length"])
    vectors = model.encode(
        [record["vector_text"] for record in documents],
        batch_size=int(config["batch_size"]),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    if vectors.shape != (1811, int(config["dimension"])) or not np.isfinite(vectors).all():
        raise RuntimeError(f"Unexpected dense corpus matrix: {vectors.shape}")
    query_vectors = []
    query_seconds = []
    for number, query in enumerate(queries, 1):
        query_started = time.perf_counter()
        vector = model.encode(
            [query], batch_size=1, show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        query_seconds.append(time.perf_counter() - query_started)
        query_vectors.append(vector[0])
        if number % 20 == 0 or number == len(queries):
            print(f"BGE-M3 query encoding {number}/{len(queries)}", flush=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return (
        vectors,
        np.asarray(query_vectors, dtype="float32"),
        index,
        float(time.perf_counter() - started),
        query_seconds,
    )


def _rrf(left: list[dict], right: list[dict], rrf_k: int = 60):
    left_rank = {item["standard_law_id"]: rank for rank, item in enumerate(left, 1)}
    right_rank = {item["standard_law_id"]: rank for rank, item in enumerate(right, 1)}
    ids = sorted(set(left_rank) | set(right_rank))
    scored = []
    for document_id in ids:
        score = 0.0
        if document_id in left_rank:
            score += 1.0 / (rrf_k + left_rank[document_id])
        if document_id in right_rank:
            score += 1.0 / (rrf_k + right_rank[document_id])
        scored.append((document_id, score))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def _as_ranking(scored, score_key="score"):
    return [
        {"rank": rank, "standard_law_id": document_id, score_key: float(score)}
        for rank, (document_id, score) in enumerate(scored, 1)
    ]


def _retrieve_all(
    config,
    records,
    documents,
    work_dir: Path,
    cache_context: dict,
    creation_source: dict,
):
    import faiss
    from sentence_transformers import SentenceTransformer

    depth = max(config["evaluation"]["candidate_depths"])
    timings = defaultdict(list)
    index_times = {}
    started = time.perf_counter()
    bm25_index = build_bm25_index(documents)
    index_times["BM25"] = time.perf_counter() - started

    queries = [record["query"] for record in records]
    vs = config["vietnamese_sbert"]
    vs_cache = work_dir / "vietnamese_sbert_embeddings.npz"
    vs_expected = _expected_embedding_metadata("Vietnamese_SBERT", vs, cache_context)
    if vs_cache.exists():
        (
            vs_vectors,
            vs_queries,
            vs_query_seconds,
            index_times["Vietnamese_SBERT"],
            vs_metadata,
        ) = _load_embedding_cache(vs_cache, vs_expected)
        print("loaded Vietnamese-SBERT embedding checkpoint", flush=True)
    else:
        started = time.perf_counter()
        vs_model = SentenceTransformer(
            config["_resolved_snapshot_paths"]["vietnamese_sbert"],
            device="cpu",
            local_files_only=True,
        )
        vs_vectors = vs_model.encode(
            [record["vector_text"] for record in documents],
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        vs_queries = []
        vs_query_seconds = []
        for query in queries:
            query_started = time.perf_counter()
            vs_queries.append(
                vs_model.encode(
                    [query], show_progress_bar=False, convert_to_numpy=True,
                    normalize_embeddings=True,
                )[0]
            )
            vs_query_seconds.append(time.perf_counter() - query_started)
        vs_queries = np.asarray(vs_queries, dtype="float32")
        index_times["Vietnamese_SBERT"] = time.perf_counter() - started
        vs_metadata = _write_embedding_cache(
            vs_cache,
            vs_vectors,
            vs_queries,
            vs_query_seconds,
            index_times["Vietnamese_SBERT"],
            vs_expected,
            creation_source,
        )
        (
            vs_vectors,
            vs_queries,
            vs_query_seconds,
            _,
            vs_metadata,
        ) = _load_embedding_cache(vs_cache, vs_expected)
        del vs_model
        gc.collect()
    if vs_vectors.shape != (1811, 768) or vs_queries.shape != (107, 768):
        raise RuntimeError("Unexpected Vietnamese-SBERT embedding checkpoint shape")
    vs_index = faiss.IndexFlatIP(768)
    vs_index.add(vs_vectors)

    bge = config["bge_m3"]
    bge_cache = work_dir / "bge_m3_embeddings.npz"
    bge_expected = _expected_embedding_metadata("BGE_M3", bge, cache_context)
    if bge_cache.exists():
        (
            bge_vectors,
            bge_queries,
            bge_query_seconds,
            index_times["BGE_M3"],
            bge_metadata,
        ) = _load_embedding_cache(bge_cache, bge_expected)
        print("loaded BGE-M3 embedding checkpoint", flush=True)
    else:
        print("encoding frozen D0 corpus with BGE-M3", flush=True)
        bge_vectors, bge_queries, _, bge_build_seconds, bge_query_seconds = _build_dense(
            config["_resolved_snapshot_paths"]["bge_m3"], documents, queries, bge
        )
        index_times["BGE_M3"] = bge_build_seconds
        bge_metadata = _write_embedding_cache(
            bge_cache,
            bge_vectors,
            bge_queries,
            bge_query_seconds,
            bge_build_seconds,
            bge_expected,
            creation_source,
        )
        (
            bge_vectors,
            bge_queries,
            bge_query_seconds,
            _,
            bge_metadata,
        ) = _load_embedding_cache(bge_cache, bge_expected)
        gc.collect()
    if bge_vectors.shape != (1811, 1024) or bge_queries.shape != (107, 1024):
        raise RuntimeError("Unexpected BGE-M3 embedding checkpoint shape")
    bge_index = faiss.IndexFlatIP(1024)
    bge_index.add(bge_vectors)
    timings["Vietnamese_SBERT"].extend(vs_query_seconds)
    timings["BGE_M3"].extend(bge_query_seconds)

    output = []
    for number, record in enumerate(records, 1):
        query = record["query"]
        started = time.perf_counter()
        bm25 = search_bm25(bm25_index, documents, query, depth)
        timings["BM25"].append(time.perf_counter() - started)

        vs_pool = _search_faiss(vs_index, vs_queries[number - 1:number], documents, depth)
        bge_pool = _search_faiss(bge_index, bge_queries[number - 1:number], documents, depth)

        depth_pools = {}
        for candidate_depth in config["evaluation"]["candidate_depths"]:
            current = fuse_pools(
                bm25[:candidate_depth], vs_pool[:candidate_depth], 0.5, "minmax"
            )[:candidate_depth]
            modern = _rrf(
                bm25[:candidate_depth], bge_pool[:candidate_depth], rrf_k=60
            )[:candidate_depth]
            depth_pools[str(candidate_depth)] = {
                "Current_Hybrid": _as_ranking(current),
                "BGE_M3_Hybrid_RRF": _as_ranking(modern),
            }
        output.append(
            {
                "query_id": record["query_id"],
                "query": query,
                "bm25": bm25,
                "vietnamese_sbert": vs_pool,
                "bge_m3": bge_pool,
                "candidate_pools": depth_pools,
            }
        )
        print(f"retrieval {number}/107", flush=True)
    return (
        output,
        {
            "index_build_seconds": index_times,
            "query_latency": {key: _percentiles(value) for key, value in timings.items()},
        },
        {
            "Vietnamese_SBERT": {
                "path": vs_cache.name,
                "sha256": sha256_file(vs_cache),
                "metadata": vs_metadata,
            },
            "BGE_M3": {
                "path": bge_cache.name,
                "sha256": sha256_file(bge_cache),
                "metadata": bge_metadata,
            },
        },
    )


def _retrieval_provenance(config: dict, cache_context: dict, embedding_caches: dict) -> dict:
    return {
        "config_fingerprint": _config_fingerprint(config),
        "candidate_source_fingerprint": config["frozen_inputs"]["exp03_candidate_fingerprint"],
        "candidate_records_fingerprint": config["frozen_inputs"]["candidate_records_fingerprint"],
        "corpus_fingerprint": config["frozen_inputs"]["corpus_fingerprint"],
        "query_fingerprint": config["frozen_inputs"]["query_fingerprint"],
        "document_ids_fingerprint": cache_context["document_ids_fingerprint"],
        "query_ids_fingerprint": cache_context["query_ids_fingerprint"],
        "candidate_depths": list(config["evaluation"]["candidate_depths"]),
        "fusion": config["fusion"],
        "embedding_cache_sha256": {
            name: value["sha256"] for name, value in sorted(embedding_caches.items())
        },
    }


def _validate_ranked_pool(pool: list[dict], expected_length: int, label: str, valid_ids: set[str]) -> None:
    if len(pool) != expected_length:
        _fail("RESUME_INTEGRITY_FAILURE", f"{label} length mismatch")
    if [int(item.get("rank", 0)) for item in pool] != list(range(1, expected_length + 1)):
        _fail("RESUME_INTEGRITY_FAILURE", f"{label} ranks are not contiguous")
    ids = [item.get("standard_law_id") for item in pool]
    if len(set(ids)) != expected_length or not set(ids) <= valid_ids:
        _fail("RESUME_INTEGRITY_FAILURE", f"{label} candidate identities are invalid")
    score_key = "score" if all("score" in item for item in pool) else "reranker_score"
    if not all(math.isfinite(float(item.get(score_key, math.nan))) for item in pool):
        _fail("RESUME_INTEGRITY_FAILURE", f"{label} contains non-finite scores")


def _load_resume_embedding_caches(
    run_dir: Path,
    config: dict,
    cache_context: dict,
) -> tuple[dict, dict]:
    arrays = {}
    provenance = {}
    for model_name, config_key, filename in (
        ("Vietnamese_SBERT", "vietnamese_sbert", "vietnamese_sbert_embeddings.npz"),
        ("BGE_M3", "bge_m3", "bge_m3_embeddings.npz"),
    ):
        path = run_dir / filename
        if not path.is_file():
            _fail("RESUME_INTEGRITY_FAILURE", f"resume cache missing: {filename}")
        expected = _expected_embedding_metadata(model_name, config[config_key], cache_context)
        documents, queries, query_seconds, index_seconds, metadata = _load_embedding_cache(
            path, expected
        )
        arrays[model_name] = (documents, queries)
        provenance[model_name] = {
            "path": filename,
            "sha256": sha256_file(path),
            "metadata": metadata,
        }
    return arrays, provenance


def _validate_retrieval_artifact(
    artifact: dict,
    config: dict,
    source_records: list[dict],
    documents: list[dict],
    expected_provenance: dict,
    embedding_arrays: dict | None = None,
) -> list[dict]:
    if artifact.get("schema_version") != RETRIEVAL_CACHE_SCHEMA:
        _fail("RESUME_INTEGRITY_FAILURE", "retrieval artifact schema mismatch")
    if artifact.get("protocol_id") != PROTOCOL_ID:
        _fail("RESUME_INTEGRITY_FAILURE", "retrieval artifact protocol mismatch")
    records = artifact.get("records", [])
    if artifact.get("fingerprint") != _sha256_json(records):
        _fail("RESUME_INTEGRITY_FAILURE", "retrieval record fingerprint mismatch")
    if artifact.get("provenance") != expected_provenance:
        _fail("RESUME_INTEGRITY_FAILURE", "retrieval provenance mismatch")
    if len(records) != len(source_records):
        _fail("RESUME_INTEGRITY_FAILURE", "retrieval query count mismatch")
    valid_ids = {item["standard_law_id"] for item in documents}
    max_depth = max(config["evaluation"]["candidate_depths"])
    for index, (record, source) in enumerate(zip(records, source_records)):
        if record.get("query_id") != source.get("query_id") or record.get("query") != source.get("query"):
            _fail("RESUME_INTEGRITY_FAILURE", f"query identity/order mismatch at row {index}")
        for key in ("bm25", "vietnamese_sbert", "bge_m3"):
            _validate_ranked_pool(record.get(key, []), max_depth, f"{key}[{index}]", valid_ids)
        pools = record.get("candidate_pools", {})
        if set(pools) != {str(value) for value in config["evaluation"]["candidate_depths"]}:
            _fail("RESUME_INTEGRITY_FAILURE", "candidate-depth keys mismatch")
        for depth in config["evaluation"]["candidate_depths"]:
            depth_pools = pools[str(depth)]
            if set(depth_pools) != set(RERANK_GENERATORS):
                _fail("RESUME_INTEGRITY_FAILURE", "fusion generator keys mismatch")
            current = _as_ranking(
                fuse_pools(
                    record["bm25"][:depth],
                    record["vietnamese_sbert"][:depth],
                    0.5,
                    "minmax",
                )[:depth]
            )
            modern = _as_ranking(
                _rrf(record["bm25"][:depth], record["bge_m3"][:depth], rrf_k=60)[:depth]
            )
            if depth_pools["Current_Hybrid"] != current:
                _fail("RESUME_INTEGRITY_FAILURE", "Current Hybrid candidates fail recomputation")
            if depth_pools["BGE_M3_Hybrid_RRF"] != modern:
                _fail("RESUME_INTEGRITY_FAILURE", "BGE RRF candidates fail recomputation")

    # Coordinated metadata + record rewrites are detected by regenerating raw
    # retrieval from the frozen corpus and validated embedding matrices.
    if embedding_arrays is not None:
        import faiss

        bm25_index = build_bm25_index(documents)
        dense_indices = {}
        for name in ("Vietnamese_SBERT", "BGE_M3"):
            matrix = embedding_arrays[name][0]
            index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            dense_indices[name] = index
        for row, source in zip(records, source_records):
            expected_bm25 = search_bm25(bm25_index, documents, source["query"], max_depth)
            if row["bm25"] != expected_bm25:
                _fail("RESUME_INTEGRITY_FAILURE", "BM25 resume pool differs from recomputation")
            row_number = source_records.index(source)
            expected_vs = _search_faiss(
                dense_indices["Vietnamese_SBERT"],
                embedding_arrays["Vietnamese_SBERT"][1][row_number:row_number + 1],
                documents,
                max_depth,
            )
            expected_bge = _search_faiss(
                dense_indices["BGE_M3"],
                embedding_arrays["BGE_M3"][1][row_number:row_number + 1],
                documents,
                max_depth,
            )
            if row["vietnamese_sbert"] != expected_vs or row["bge_m3"] != expected_bge:
                _fail("RESUME_INTEGRITY_FAILURE", "dense resume pool differs from recomputation")
    return records


def _preseed_exp05(path: Path, candidate_records: list[dict], config: dict):
    run_dir = path.parent
    project_root = Path(config.get("_project_root", Path(__file__).resolve().parents[4]))
    required_files = {
        "run_manifest.json",
        "prediction_artifact.json",
        "summary_metrics.json",
        "statistical_analysis.json",
    }
    try:
        artifacts = {
            name: json.loads((run_dir / name).read_text(encoding="utf-8"))
            for name in required_files
        }
        artifact = artifacts["prediction_artifact.json"]
        validate_prediction_artifact(artifact, expected_query_count=107)
        validate_exp05_artifacts(artifacts)
    except Exception as exc:
        _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", str(exc))
    manifest = artifacts["run_manifest.json"]
    expected_manifest = {
        "candidate_fingerprint": config["frozen_inputs"]["exp03_candidate_fingerprint"],
        "corpus_fingerprint": config["frozen_inputs"]["corpus_fingerprint"],
        "query_fingerprint": config["frozen_inputs"]["query_fingerprint"],
        "model_id": EXP05_MODEL_ID,
        "model_revision": EXP05_MODEL_REVISION,
        "prediction_fingerprint": artifact["prediction_fingerprint"],
        "inference_pair_count": 3210,
        "model_file_sha256": EXP05_MODEL_HASHES,
        "candidate_file_sha256": sha256_file(
            project_root / config["frozen_inputs"]["exp03_candidate_path"]
        ),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", f"EXP-05 {key} mismatch")
    source_by_id = {record["query_id"]: record for record in candidate_records}
    if len(source_by_id) != 107:
        _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", "candidate source query IDs are invalid")
    scores = {}
    for prediction in artifact["predictions"]:
        query_id = str(prediction["query_id"])
        source = source_by_id.get(prediction["query_id"])
        if source is None:
            _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", "prediction query is absent from candidate source")
        expected_control = _as_ranking(
            fuse_pools(source["bm25_pool"], source["dense_pool"], 0.5, "minmax")[:30]
        )
        if prediction["control_top_30"] != expected_control:
            _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", "EXP-05 control candidates mismatch")
        expected_ids = {item["standard_law_id"] for item in expected_control}
        treatment_ids = [item["standard_law_id"] for item in prediction["treatment_top_30"]]
        if len(treatment_ids) != 30 or set(treatment_ids) != expected_ids:
            _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", "EXP-05 treatment candidate identity mismatch")
        for item in prediction["treatment_top_30"]:
            if not math.isfinite(float(item["reranker_score"])):
                _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", "EXP-05 score is not finite")
            scores[f"{query_id}\t{item['standard_law_id']}"] = float(item["reranker_score"])
    if len(scores) != 3210:
        _fail("EXP05_ARTIFACT_INTEGRITY_FAILURE", "EXP-05 must contain 3,210 unique pairs")
    return scores, {
        "prediction_fingerprint": artifact["prediction_fingerprint"],
        "manifest_sha256": sha256_file(run_dir / "run_manifest.json"),
        "prediction_sha256": sha256_file(path),
        "candidate_fingerprint": manifest["candidate_fingerprint"],
        "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
    }


def _required_rerank_pairs(retrieval_records):
    pairs = {}
    origins = defaultdict(list)
    for record in retrieval_records:
        query_id = str(record["query_id"])
        for depth, generators in record["candidate_pools"].items():
            for generator, candidates in generators.items():
                for candidate in candidates:
                    key = f"{query_id}\t{candidate['standard_law_id']}"
                    pairs[key] = {
                        "query_id": record["query_id"],
                        "query": record["query"],
                        "standard_law_id": candidate["standard_law_id"],
                    }
                    origins[key].append(f"{generator}@{depth}")
    return pairs, origins


def _score_missing(config, retrieval_records, documents, candidate_records, checkpoint_path: Path):
    document_by_id = {record["standard_law_id"]: record for record in documents}
    required, origins = _required_rerank_pairs(retrieval_records)
    exp05_path = Path(config["_project_root"]) / config["frozen_inputs"]["exp05_prediction_path"]
    scores, exp05_provenance = _preseed_exp05(exp05_path, candidate_records, config)
    reused = len(set(required) & set(scores))
    scores = {key: value for key, value in scores.items() if key in required}
    # EXP-05 persisted aggregate, rather than per-pair, timing.  Allocate its
    # official mean seconds/pair to reused logits so downstream latency is not
    # silently reported as zero.  This is explicitly an estimate.
    exp05_seconds_per_pair = (
        EXP05_OFFICIAL_INFERENCE_SECONDS / EXP05_OFFICIAL_PAIR_COUNT
    )
    elapsed_by_key = {key: exp05_seconds_per_pair for key in scores}
    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if saved.get("required_pair_fingerprint") != _sha256_json(sorted(required)):
            raise RuntimeError("Reranker checkpoint belongs to a different candidate set")
        saved_scores = {key: float(value) for key, value in saved["scores"].items()}
        if not set(saved_scores) <= set(required) or not all(
            math.isfinite(value) for value in saved_scores.values()
        ):
            _fail("RESUME_INTEGRITY_FAILURE", "reranker checkpoint scores are invalid")
        scores.update(saved_scores)
        elapsed_by_key.update(
            {key: float(value) for key, value in saved.get("elapsed_seconds", {}).items()}
        )
    missing = [key for key in sorted(required) if key not in scores]
    print(
        f"reranker required={len(required)} reused_exp05={reused} "
        f"already_checkpointed={len(scores)-reused} missing={len(missing)}",
        flush=True,
    )
    if not missing:
        return scores, elapsed_by_key, {
            "required": len(required),
            "reused_exp05": reused,
            "exp05_provenance": exp05_provenance,
            "reused_latency_method": "official_EXP05_mean_seconds_per_pair",
            "reused_seconds_per_pair": exp05_seconds_per_pair,
        }

    reranker_snapshot = config["_resolved_snapshot_paths"]["reranker"]
    validate_model_artifacts(reranker_snapshot)
    tokenizer, model, torch, load_metadata = load_local_reranker(
        reranker_snapshot
    )
    checkpoint_every = int(config["reranker"]["checkpoint_every_pairs"])
    with torch.inference_mode():
        for number, key in enumerate(missing, 1):
            record = required[key]
            document = document_by_id[record["standard_law_id"]]["vector_text"]
            started = time.perf_counter()
            encoded = tokenizer(
                [record["query"]], [document], max_length=1024,
                truncation="only_second", padding=True, return_tensors="pt",
            )
            logits = model(**{name: value.to("cpu") for name, value in encoded.items()}).logits
            if logits.ndim != 2 or tuple(logits.shape) != (1, 1):
                raise RuntimeError(f"Unexpected reranker logits shape: {tuple(logits.shape)}")
            score = float(logits[0, 0].detach().cpu())
            if not math.isfinite(score):
                raise RuntimeError("Reranker returned NaN/Infinity")
            scores[key] = score
            elapsed_by_key[key] = time.perf_counter() - started
            if number % checkpoint_every == 0 or number == len(missing):
                _strict_write(
                    checkpoint_path,
                    {
                        "schema_version": "dev107_reranker_checkpoint_v1",
                        "required_pair_fingerprint": _sha256_json(sorted(required)),
                        "scores": scores,
                        "elapsed_seconds": elapsed_by_key,
                    },
                )
                print(f"reranker {number}/{len(missing)}", flush=True)
    return scores, elapsed_by_key, {
        "required": len(required),
        "reused_exp05": reused,
        "model_load": load_metadata,
        "newly_scored": len(missing),
        "exp05_provenance": exp05_provenance,
        "reused_latency_method": "official_EXP05_mean_seconds_per_pair",
        "reused_seconds_per_pair": exp05_seconds_per_pair,
    }


def _rerank(candidates, query_id, scores):
    ranked = []
    for candidate in candidates:
        key = f"{query_id}\t{candidate['standard_law_id']}"
        ranked.append(
            (
                candidate["standard_law_id"],
                scores[key],
                int(candidate["rank"]),
            )
        )
    ranked.sort(key=lambda item: (-item[1], item[2], item[0]))
    return [
        {
            "rank": rank,
            "standard_law_id": document_id,
            "reranker_score": float(score),
            "original_rank": original_rank,
        }
        for rank, (document_id, score, original_rank) in enumerate(ranked, 1)
    ]


def _query_metrics(ranking, record, k: int):
    core = record["core_gt_filtered"]
    support = record["supporting_gt_filtered"]
    ground_truth = core + support
    if not ground_truth:
        return None
    ids = [item["standard_law_id"] for item in ranking[:k]]
    relevant = [any(is_match(document_id, gt) for gt in ground_truth) for document_id in ids]
    hits = sum(
        any(is_match(document_id, gt) for document_id in ids) for gt in ground_truth
    )
    core_hits = sum(any(is_match(document_id, gt) for document_id in ids) for gt in core)
    support_hits = sum(
        any(is_match(document_id, gt) for document_id in ids) for gt in support
    )
    denominator = 3 * len(core) + len(support)
    reciprocal_rank = next((1.0 / rank for rank, hit in enumerate(relevant, 1) if hit), 0.0)
    precision = sum(relevant) / k
    recall = hits / len(ground_truth)
    beta2 = 4.0
    f2 = (
        (1 + beta2) * precision * recall / (beta2 * precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "recall": recall,
        "precision": precision,
        "weighted_recall": (3 * core_hits + support_hits) / denominator,
        "mrr": reciprocal_rank,
        "ndcg": calculate_ndcg(ids, ground_truth, k, gt_weights=record["gt_weights"]),
        "f2": f2,
    }


def _average_precision(ranking, record, k=20):
    gt = record["core_gt_filtered"] + record["supporting_gt_filtered"]
    if not gt:
        return None
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranking[:k], 1):
        if any(is_match(item["standard_law_id"], value) for value in gt):
            hits += 1
            total += hits / rank
    return total / len(gt)


def _aggregate(rankings, gt_by_query, reported_k=(5, 10, 20)):
    by_k = {}
    per_query = {}
    for k in reported_k:
        values = defaultdict(list)
        for query_id, ranking in rankings.items():
            metrics = _query_metrics(ranking, gt_by_query[query_id], k)
            if metrics is None:
                continue
            per_query.setdefault(str(query_id), {})[str(k)] = metrics
            for name, value in metrics.items():
                values[name].append(value)
        by_k[str(k)] = {name: float(np.mean(items)) for name, items in values.items()}
        by_k[str(k)]["valid_queries"] = len(values["recall"])
    aps = [
        value for query_id, ranking in rankings.items()
        if (value := _average_precision(ranking, gt_by_query[query_id], 20)) is not None
    ]
    return {"at_k": by_k, "map_at_20": float(np.mean(aps)), "per_query": per_query}


def _candidate_analysis(candidates, record):
    gt = record["core_gt_filtered"] + record["supporting_gt_filtered"]
    ids = [item["standard_law_id"] for item in candidates]
    present = sum(any(is_match(document_id, value) for document_id in ids) for value in gt)
    return {
        "gt_total": len(gt),
        "gt_present": present,
        "gt_absent": len(gt) - present,
        "candidate_recall": present / len(gt) if gt else None,
        "zero_relevant": bool(gt) and present == 0,
    }


def _bootstrap(deltas, samples=20000, seed=42):
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2000):
        size = min(2000, samples - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    relation = "entirely_above_zero" if low > 0 else "entirely_below_zero" if high < 0 else "crosses_zero"
    return {
        "mean_delta": float(values.mean()),
        "ci_95": [float(low), float(high)],
        "ci_relation": relation,
        "samples": samples,
        "seed": seed,
    }


def _assemble_results(config, records, retrieval_records, scores, elapsed_by_key, timing):
    gt = {record["query_id"]: record for record in records}
    retrieval_by_id = {record["query_id"]: record for record in retrieval_records}
    rankings = {name: {} for name in SYSTEMS}
    depth_rankings = {}
    candidate_analysis = {}
    for query_id, record in retrieval_by_id.items():
        rankings["BM25"][query_id] = record["bm25"]
        rankings["Vietnamese_SBERT"][query_id] = record["vietnamese_sbert"]
        rankings["BGE_M3"][query_id] = record["bge_m3"]
        rankings["Current_Hybrid"][query_id] = record["candidate_pools"]["30"]["Current_Hybrid"]
        rankings["BGE_M3_Hybrid_RRF"][query_id] = record["candidate_pools"]["100"]["BGE_M3_Hybrid_RRF"]
    aggregate = {name: _aggregate(value, gt) for name, value in rankings.items()}

    for generator in RERANK_GENERATORS:
        for depth in config["evaluation"]["candidate_depths"]:
            name = f"{generator}_Reranked_N{depth}"
            control = {}
            treatment = {}
            analyses = []
            latency = []
            control_hits = 0
            reranked_hits = 0
            ranking_failure_queries = 0
            for query_id, record in retrieval_by_id.items():
                candidates = record["candidate_pools"][str(depth)][generator]
                control[query_id] = candidates
                treatment[query_id] = _rerank(candidates, query_id, scores)
                analysis = _candidate_analysis(candidates, gt[query_id])
                analyses.append(analysis)
                ground_truth = (
                    gt[query_id]["core_gt_filtered"]
                    + gt[query_id]["supporting_gt_filtered"]
                )
                control_ids = [item["standard_law_id"] for item in candidates[:10]]
                reranked_ids = [
                    item["standard_law_id"] for item in treatment[query_id][:10]
                ]
                query_control_hits = sum(
                    any(is_match(document_id, value) for document_id in control_ids)
                    for value in ground_truth
                )
                query_reranked_hits = sum(
                    any(is_match(document_id, value) for document_id in reranked_ids)
                    for value in ground_truth
                )
                control_hits += query_control_hits
                reranked_hits += query_reranked_hits
                ranking_failure_queries += int(
                    analysis["gt_present"] > 0 and query_reranked_hits == 0
                )
                latency.append(
                    sum(elapsed_by_key.get(f"{query_id}\t{x['standard_law_id']}", 0.0) for x in candidates)
                )
            control_metrics = _aggregate(control, gt)
            treatment_metrics = _aggregate(treatment, gt)
            depth_rankings[name] = treatment_metrics
            recalls = [item["candidate_recall"] for item in analyses if item["candidate_recall"] is not None]
            candidate_analysis[f"{generator}@{depth}"] = {
                "candidate_recall": float(np.mean(recalls)),
                "gt_present": sum(item["gt_present"] for item in analyses),
                "gt_absent": sum(item["gt_absent"] for item in analyses),
                "queries_zero_relevant": sum(item["zero_relevant"] for item in analyses),
                "control_gt_hits_at_10": control_hits,
                "reranked_gt_hits_at_10": reranked_hits,
                "ranking_miss_instances_after_rerank": (
                    sum(item["gt_present"] for item in analyses) - reranked_hits
                ),
                "reranker_hit_rate_conditional_on_candidate_presence": (
                    reranked_hits / sum(item["gt_present"] for item in analyses)
                    if sum(item["gt_present"] for item in analyses) else None
                ),
                "queries_with_candidates_but_no_reranked_top10_hit": ranking_failure_queries,
                "control": control_metrics["at_k"],
                "reranked": treatment_metrics["at_k"],
                "reranker_latency": _percentiles(latency),
            }

    # EXP-05 reproduction gate for exact pool-30 current Hybrid.
    exp05 = depth_rankings["Current_Hybrid_Reranked_N30"]["at_k"]["10"]
    exp05_reproduction = {
        name: {"actual": exp05[name], "expected": expected, "absolute_error": abs(exp05[name] - expected)}
        for name, expected in {
            "recall": 0.1352983893080289,
            "weighted_recall": 0.09941759776486084,
            "mrr": 0.37295653463877754,
            "ndcg": 0.08628390747714257,
        }.items()
    }

    comparison_systems = dict(aggregate)
    comparison_systems.update(depth_rankings)
    statistics_output = {}
    for challenger, result in comparison_systems.items():
        if challenger in {"BM25", "Current_Hybrid"}:
            continue
        statistics_output[challenger] = {}
        for reference in ("BM25", "Current_Hybrid"):
            statistics_output[challenger][reference] = {}
            for metric in STAT_METRICS:
                deltas = []
                for query_id in sorted(gt):
                    left = result["per_query"][str(query_id)]["10"][metric]
                    right = aggregate[reference]["per_query"][str(query_id)]["10"][metric]
                    deltas.append(left - right)
                statistics_output[challenger][reference][metric] = _bootstrap(deltas)
    return {
        "retrieval_systems": aggregate,
        "reranked_systems": depth_rankings,
        "candidate_analysis": candidate_analysis,
        "statistics": statistics_output,
        "exp05_reproduction": exp05_reproduction,
        "timing": timing,
    }


def _enforce_exp05_parity(exp05_reproduction: dict) -> float:
    errors = [float(item["absolute_error"]) for item in exp05_reproduction.values()]
    maximum = max(errors, default=math.inf)
    if not math.isfinite(maximum) or maximum > EXP05_PARITY_TOLERANCE:
        _fail(
            "EXP05_PARITY_FAILURE",
            f"maximum_metric_error={maximum}; tolerance={EXP05_PARITY_TOLERANCE}",
        )
    return maximum


def _source_paths(config_path: Path, project_root: Path) -> list[str]:
    paths = [
        config_path,
        Path(__file__),
        project_root / "backend/scripts/Evaluation/core/corpus.py",
        project_root / "backend/scripts/Evaluation/core/query.py",
        project_root / "backend/scripts/Evaluation/retrieval/bm25.py",
        project_root / "backend/scripts/Evaluation/retrieval/fusion.py",
        project_root / "backend/scripts/Evaluation/evaluation/metrics.py",
        project_root / "backend/scripts/Evaluation/reporting/manifest.py",
        project_root / "backend/scripts/Evaluation/experiments/exp03_static_fusion_cv.py",
        project_root / "backend/scripts/Evaluation/experiments/exp05_frozen_hybrid30_zero_shot_reranker.py",
        project_root / "backend/scripts/Tests/test_dev107_retrieval_model_screening.py",
    ]
    return [str(path.resolve().relative_to(project_root.resolve())) for path in paths]


def _validate_pre_run_provenance(provenance: dict) -> str:
    commit = provenance.get("git_commit")
    if not (
        isinstance(commit, str)
        and len(commit) in (40, 64)
        and all(character in "0123456789abcdefABCDEF" for character in commit)
    ):
        _fail("PRE_RUN_PROVENANCE_FAILURE", "valid committed git HEAD required")
    if any(
        provenance.get(key) is not False
        for key in ("git_dirty", "tracked_code_dirty", "baseline_source_dirty")
    ):
        _fail("PRE_RUN_PROVENANCE_FAILURE", "new runs require a completely clean worktree")
    return commit


def _capture_pre_run_provenance(
    project_root: Path,
    config_path: Path,
    config_fingerprint: str,
) -> dict:
    source_paths = _source_paths(config_path, project_root)
    git_state = git_provenance(project_root, source_paths)
    commit = _validate_pre_run_provenance(git_state)
    return {
        "git_commit": commit,
        "git_state": git_state,
        "source_paths": source_paths,
        "source_files": _source_file_provenance(project_root, source_paths),
        "config_fingerprint": config_fingerprint,
    }


def _capture_input_provenance(
    candidate_path: Path,
    candidate_manifest_path: Path,
    development_path: Path,
    exp05_prediction_path: Path,
    corpus_inventory: list[dict],
) -> dict:
    exp05_manifest_path = exp05_prediction_path.with_name("run_manifest.json")
    return {
        "candidate_cache_sha256": sha256_file(candidate_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "benchmark_sha256": sha256_file(development_path),
        "exp05_prediction_sha256": sha256_file(exp05_prediction_path),
        "exp05_manifest_sha256": sha256_file(exp05_manifest_path),
        "corpus_inventory": [
            {"key": item["key"], "count": item["count"], "sha256": item["sha256"]}
            for item in corpus_inventory
        ],
    }


def _capture_embedding_model_provenance(config: dict) -> dict:
    output = {}
    for name in ("vietnamese_sbert", "bge_m3"):
        model = config[name]
        snapshot = Path(
            config.get("_resolved_snapshot_paths", {}).get(name)
            or _resolve_local_snapshot(model)
        )
        files = {}
        for relative in ("config.json", "modules.json", "1_Pooling/config.json", "pytorch_model.bin"):
            path = snapshot / relative
            if not path.exists() or not path.is_file():
                _fail("PRE_RUN_PROVENANCE_FAILURE", f"{name} artifact missing: {relative}")
            resolved = path.resolve(strict=True)
            entry = {
                "relative_path": relative,
                "resolved_blob": resolved.name,
                "size": resolved.stat().st_size,
            }
            if relative != "pytorch_model.bin":
                entry["sha256"] = sha256_file(path)
            files[relative] = entry
        output[name] = {
            "model_id": model["id"],
            "model_revision": model["revision"],
            "snapshot_path": str(snapshot),
            "files": files,
        }
    return output


def _verify_prepublication_provenance(
    project_root: Path,
    expected: dict,
    current_input_provenance: dict,
) -> None:
    current_git = git_provenance(project_root, expected["source_paths"])
    current_commit = _validate_pre_run_provenance(current_git)
    if current_commit != expected["git_commit"]:
        _fail("TOCTOU_PUBLICATION_FAILURE", "git HEAD changed during execution")
    current_sources = _source_file_provenance(project_root, expected["source_paths"])
    if current_sources != expected["source_files"]:
        _fail("TOCTOU_PUBLICATION_FAILURE", "source/config changed during execution")
    if current_input_provenance != expected["input_provenance"]:
        _fail("TOCTOU_PUBLICATION_FAILURE", "critical input changed during execution")
    current_models = _capture_embedding_model_provenance(expected["config"])
    if current_models != expected["model_provenance"]:
        _fail("TOCTOU_PUBLICATION_FAILURE", "embedding model provenance changed")


def _assert_no_symlink_chain(path: Path, root: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        _fail("UNSAFE_RESUME_PATH", "resume directory is outside the canonical run root")
    current = root
    if current.is_symlink():
        _fail("UNSAFE_RESUME_PATH", "canonical run root may not be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            _fail("UNSAFE_RESUME_PATH", f"symlink component rejected: {current}")


def _resolve_resume_dir(resume_dir: Path, run_root: Path) -> Path:
    raw = Path(resume_dir)
    if ".." in raw.parts:
        _fail("UNSAFE_RESUME_PATH", "parent traversal is forbidden")
    root = run_root.resolve(strict=True)
    try:
        lexical = raw if raw.is_absolute() else run_root / raw
        _assert_no_symlink_chain(lexical, run_root)
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(root)
    except Dev107IntegrityError:
        raise
    except Exception as exc:
        _fail("UNSAFE_RESUME_PATH", str(exc))
    if not resolved.is_dir() or not resolved.name.endswith(".partial"):
        _fail("UNSAFE_RESUME_PATH", "resume directory must be an existing .partial directory")
    return resolved


def _completed_result_fingerprint(
    retrieval_fingerprint: str,
    score_fingerprint: str,
    summary: dict,
    statistics_output: dict,
) -> str:
    return _sha256_json(
        {
            "retrieval_fingerprint": retrieval_fingerprint,
            "score_fingerprint": score_fingerprint,
            "summary_fingerprint": _sha256_json(summary),
            "statistics_fingerprint": _sha256_json(statistics_output),
        }
    )


def _verify_embedding_assets(run_dir: Path, provenance: dict) -> None:
    for value in provenance.values():
        path = run_dir / value["path"]
        if not path.is_file() or sha256_file(path) != value["sha256"]:
            _fail("TOCTOU_PUBLICATION_FAILURE", f"embedding cache changed: {path.name}")


def _validate_partial_publication(run_dir: Path, config_path: Path) -> None:
    required = {
        "retrieval_candidates.json",
        "reranker_scores.json",
        "summary_metrics.json",
        "statistical_analysis.json",
        "run_manifest.json",
        "eval_config.snapshot.yaml",
    }
    actual = {path.name for path in run_dir.iterdir()}
    if not required <= actual:
        _fail("TOCTOU_PUBLICATION_FAILURE", f"partial artifact set incomplete: {sorted(required - actual)}")
    retrieval = json.loads((run_dir / "retrieval_candidates.json").read_text(encoding="utf-8"))
    scores = json.loads((run_dir / "reranker_scores.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8"))
    statistics_output = json.loads(
        (run_dir / "statistical_analysis.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if retrieval.get("fingerprint") != _sha256_json(retrieval.get("records", [])):
        _fail("TOCTOU_PUBLICATION_FAILURE", "retrieval fingerprint changed before publication")
    if scores.get("pair_count") != len(scores.get("scores", {})):
        _fail("TOCTOU_PUBLICATION_FAILURE", "reranker pair count mismatch")
    if scores.get("fingerprint") != _sha256_json(scores.get("scores", {})):
        _fail("TOCTOU_PUBLICATION_FAILURE", "reranker fingerprint changed before publication")
    expected_result = _completed_result_fingerprint(
        retrieval["fingerprint"],
        scores["fingerprint"],
        summary,
        statistics_output,
    )
    if manifest.get("final_result_fingerprint") != expected_result:
        _fail("TOCTOU_PUBLICATION_FAILURE", "final result fingerprint mismatch")
    if sha256_file(run_dir / "eval_config.snapshot.yaml") != sha256_file(config_path):
        _fail("TOCTOU_PUBLICATION_FAILURE", "config snapshot mismatch")


def run(
    config_path: Path,
    resume_dir: Path | None = None,
    *,
    _project_root: Path | None = None,
):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    project_root = (
        Path(_project_root).resolve()
        if _project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    config_fingerprint = _config_fingerprint(config)
    pre_run = _capture_pre_run_provenance(project_root, config_path, config_fingerprint)
    config["_project_root"] = str(project_root)
    config["_resolved_snapshot_paths"] = {
        name: str(_resolve_local_snapshot(config[name]))
        for name in ("vietnamese_sbert", "bge_m3", "reranker")
    }
    (
        records,
        documents,
        corpus_inventory,
        candidate_path,
        candidate_manifest_path,
        development_path,
        benchmark_sha,
    ) = _load_inputs(config, project_root)
    exp05_prediction_path = project_root / config["frozen_inputs"]["exp05_prediction_path"]
    input_provenance = _capture_input_provenance(
        candidate_path,
        candidate_manifest_path,
        development_path,
        exp05_prediction_path,
        corpus_inventory,
    )
    pre_run["input_provenance"] = input_provenance
    pre_run["config"] = config
    pre_run["model_provenance"] = _capture_embedding_model_provenance(config)
    cache_context = _embedding_cache_context(config, documents, records)
    runner_source = next(
        item for item in pre_run["source_files"]
        if item["path"].endswith("dev107_retrieval_model_screening.py")
    )
    creation_source = {
        "git_commit": pre_run["git_commit"],
        "runner_sha256": runner_source["sha256"],
        "config_fingerprint": config_fingerprint,
    }
    configured_run_root = project_root / config["paths"]["output_dir"]
    _assert_no_symlink_chain(configured_run_root, project_root)
    run_root = configured_run_root.resolve(strict=True)

    if resume_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"
        run_dir = run_root / (run_id + ".partial")
        run_dir.mkdir(parents=False, exist_ok=False)
    else:
        run_dir = _resolve_resume_dir(resume_dir, run_root)
        run_id = run_dir.name[:-8]
    retrieval_path = run_dir / "retrieval_candidates.json"
    if retrieval_path.exists():
        embedding_arrays, embedding_provenance = _load_resume_embedding_caches(
            run_dir, config, cache_context
        )
        expected_retrieval_provenance = _retrieval_provenance(
            config, cache_context, embedding_provenance
        )
        retrieval_artifact = json.loads(retrieval_path.read_text(encoding="utf-8"))
        retrieval_records = _validate_retrieval_artifact(
            retrieval_artifact,
            config,
            records,
            documents,
            expected_retrieval_provenance,
            embedding_arrays=embedding_arrays,
        )
        retrieval_timing = retrieval_artifact["timing"]
    else:
        retrieval_records, retrieval_timing, embedding_provenance = _retrieve_all(
            config,
            records,
            documents,
            run_dir,
            cache_context,
            creation_source,
        )
        expected_retrieval_provenance = _retrieval_provenance(
            config, cache_context, embedding_provenance
        )
        retrieval_artifact = {
            "schema_version": RETRIEVAL_CACHE_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "records": retrieval_records,
            "timing": retrieval_timing,
            "provenance": expected_retrieval_provenance,
            "fingerprint": _sha256_json(retrieval_records),
        }
        _validate_retrieval_artifact(
            retrieval_artifact,
            config,
            records,
            documents,
            expected_retrieval_provenance,
        )
        _strict_write(
            retrieval_path,
            retrieval_artifact,
        )
    scores, elapsed, scorer_metadata = _score_missing(
        config,
        retrieval_records,
        documents,
        records,
        run_dir / "reranker_scores.checkpoint.json",
    )
    results = _assemble_results(
        config, records, retrieval_records, scores, elapsed,
        {**retrieval_timing, "reranker": scorer_metadata},
    )
    parity_maximum_error = _enforce_exp05_parity(results["exp05_reproduction"])
    _strict_write(run_dir / "summary_metrics.json", results)
    _strict_write(run_dir / "statistical_analysis.json", results["statistics"])
    score_fingerprint = _sha256_json(scores)
    _strict_write(
        run_dir / "reranker_scores.json",
        {
            "schema_version": "dev107_reranker_scores_v1",
            "pair_count": len(scores),
            "scores": scores,
            "fingerprint": score_fingerprint,
        },
    )
    retrieval_fingerprint = retrieval_artifact["fingerprint"]
    result_fingerprint = _completed_result_fingerprint(
        retrieval_fingerprint,
        score_fingerprint,
        results,
        results["statistics"],
    )
    manifest = {
        "run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "scope": "DEVELOPMENT / EXPLORATORY",
        "confirmatory": False,
        "git_commit": pre_run["git_commit"],
        "git_dirty": pre_run["git_state"]["git_dirty"],
        "tracked_code_dirty": pre_run["git_state"]["tracked_code_dirty"],
        "config": {
            "path": str(config_path.relative_to(project_root)),
            "sha256": sha256_file(config_path),
            "fingerprint": config_fingerprint,
        },
        "source_files": pre_run["source_files"],
        "benchmark": {"path": str(development_path.relative_to(project_root)), "sha256": benchmark_sha},
        "candidate_cache": {"path": str(candidate_path.relative_to(project_root)), "sha256": sha256_file(candidate_path)},
        "candidate_source_fingerprint": config["frozen_inputs"]["exp03_candidate_fingerprint"],
        "candidate_records_fingerprint": config["frozen_inputs"]["candidate_records_fingerprint"],
        "query_fingerprint": config["frozen_inputs"]["query_fingerprint"],
        "corpus_fingerprint": config["frozen_inputs"]["corpus_fingerprint"],
        "corpus_inventory": corpus_inventory,
        "input_provenance": input_provenance,
        "embedding_caches": embedding_provenance,
        "embedding_models": pre_run["model_provenance"],
        "retrieval_candidate_fingerprint": retrieval_fingerprint,
        "reranker_prediction_fingerprint": scorer_metadata["exp05_provenance"]["prediction_fingerprint"],
        "reranker_score_fingerprint": score_fingerprint,
        "final_result_fingerprint": result_fingerprint,
        "exp05_parity_tolerance": EXP05_PARITY_TOLERANCE,
        "exp05_parity_maximum_error": parity_maximum_error,
        "query_count": 107,
        "corpus_count": 1811,
        "seed": 42,
        "python": platform.python_version(),
        "packages": {name: safe_package_version(name) for name in ("torch", "transformers", "sentence-transformers", "faiss-cpu", "numpy", "scipy")},
        "models": {name: config[name] for name in ("vietnamese_sbert", "bge_m3", "reranker")},
        "resolved_model_snapshots": config["_resolved_snapshot_paths"],
        "independent_benchmark_used": False,
        "gemini_used": False,
        "training_on_development_queries": False,
    }
    _strict_write(run_dir / "run_manifest.json", manifest)
    shutil.copy2(config_path, run_dir / "eval_config.snapshot.yaml")
    _validate_partial_publication(run_dir, config_path)
    current_input_provenance = _capture_input_provenance(
        candidate_path,
        candidate_manifest_path,
        development_path,
        exp05_prediction_path,
        corpus_inventory,
    )
    _verify_embedding_assets(run_dir, embedding_provenance)
    _verify_prepublication_provenance(
        project_root,
        pre_run,
        current_input_provenance,
    )
    final_dir = run_dir.with_name(run_id)
    if final_dir.exists():
        _fail("TOCTOU_PUBLICATION_FAILURE", "final run directory already exists")
    os.replace(run_dir, final_dir)
    print(final_dir)
    return final_dir


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).parents[1] / "configs/dev107_retrieval_model_screening.yaml",
    )
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args(argv)
    return run(args.config.resolve(), args.resume_dir)


if __name__ == "__main__":
    main()
