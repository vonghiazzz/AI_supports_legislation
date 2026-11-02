"""BSARD Phase 1F: frozen BGE reranking over BGE/RRF Top-20 and Top-50.

The module deliberately separates candidate generation from reranking.  It
never invokes a retriever: two committed, fingerprinted rankings are the only
candidate sources, and one authoritative union score cache feeds all four
derived systems.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..benchmarking.bsard_adapter import load_bsard
from ..evaluation.bsard_metrics import (
    evaluate_official_compat,
    evaluate_safe_canonical,
    paired_bootstrap_mean_delta,
)
from .bsard_bm25_smoke import ranking_fingerprint, sha256_file, sha256_json, validate_rankings


PROTOCOL_ID = "bsard_bge_reranker_v1"
CACHE_SCHEMA = "bsard_bge_reranker_score_cache_v1"
MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODEL_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots" / MODEL_REVISION
MODEL_HASHES = {
    "config.json": "13dcd6c31d9fec9d1d8e158702072f62d7fa7d312a64b9fe057bec9a08cfe41a",
    "model.safetensors": "d9e3e081faff1eefb84019509b2f5558fd74c1a05a2c7db22f74174fcedb5286",
    "tokenizer.json": "69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15",
    "tokenizer_config.json": "7e4c1cc848840aeccdd763458c18dd525eb0f795c992e00ebe9c28554e7db2d4",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "8c785abebea9ae3257b61681b4e6fd8365ceafde980c21970d001e834cf10835",
}
BGE_RUN_ID = "bsard_bge_m3_dense_v1_20261102_064758_329094_fccb32cc"
BGE_RETRIEVAL_FP = "092721506b540caa02a0d20117de8e9a2dac824e6624414837388da99fe29664"
BGE_RESULT_FP = "a0d71b9080f86048f6d1e5e4cfbaf19da70507a57638c19a8393cbf0bbbbdc36"
RRF_RUN_ID = "bsard_rrf_hybrid_v1_20261102_082257_638404_1dddd7c5"
RRF_RETRIEVAL_FP = "771a64288847ef5148f4a1fe37fced9c2d7d98410a710e04b8bb1d8eaeac9500"
RRF_RESULT_FP = "928d1892b3f0912a3718f8646b709da3057a6734ec536ee486151f52cceb82ad"
CORPUS_FP = "13192650b6a488221b5d98654f6274354910f509ac83b49b9e9bbaff8cccaaf3"
TEST_FP = "38865923e39d24b23f140be7a3bd90618aed483c3da7def248186abec7c97c21"
QREL_FP = "c1fbd3251a5c1a806bb85d66c8e8a73fd84b864ca6ee5ee5bc072fc3dc2cab0a"
EXPECTED_CEILINGS = {
    "bge_n20": (272, 79), "bge_n50": (357, 55),
    "rrf_n20": (255, 84), "rrf_n50": (372, 60),
}
SOURCE_DEPTH = 500
CANDIDATE_DEPTHS = (20, 50)
MAX_LENGTH = 1024
TRUNCATION = "only_second"
BATCH_SIZE = 1
DEVICE = "cpu"
DTYPE = "float32"
SHARD_SIZE = 100
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 42
SYSTEMS = ("bge_rerank_n20", "bge_rerank_n50", "rrf_rerank_n20", "rrf_rerank_n50")
SENSITIVE_METRICS = ("recall@10", "map@100", "mrr@100", "ndcg@10")


class BSARDRerankerError(RuntimeError):
    pass


def strict_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(BSARDRerankerError(f"non-finite JSON constant: {value}")))


def strict_write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def git_head(root: Path) -> str:
    value = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise BSARDRerankerError("valid git commit required")
    return value


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if platform.system() == "Darwin" else value / 1024.0


def validate_model_files(snapshot: Path = MODEL_SNAPSHOT, expected: Mapping[str, str] = MODEL_HASHES) -> dict[str, str]:
    try:
        resolved = snapshot.resolve(strict=True)
    except FileNotFoundError as exc:
        raise BSARDRerankerError("BGE_RERANKER_EXACT_REVISION_CACHE_MISSING") from exc
    if resolved.name != MODEL_REVISION or not resolved.is_dir():
        raise BSARDRerankerError("BGE_RERANKER_EXACT_REVISION_CACHE_MISSING")
    actual = {}
    for name, expected_hash in expected.items():
        path = resolved / name
        if not path.is_file():
            raise BSARDRerankerError(f"missing model artifact: {name}")
        actual[name] = sha256_file(path)
        if actual[name] != expected_hash:
            raise BSARDRerankerError(f"model artifact hash mismatch: {name}")
    return actual


def _rrf_result_fingerprint(summary: Mapping[str, Any], retrieval_fp: str) -> str:
    payload = {
        "retrieval_fingerprint": retrieval_fp,
        "official_compat": summary["official_compat"],
        "safe_canonical": summary["safe_canonical"],
        "coverage": summary["coverage"],
        "source_membership": {
            "source_top500_qrel_partition": summary["fusion_analysis"]["source_complementarity"],
            "source_union_relevant_qrels": summary["fusion_analysis"]["source_union"]["source_union_relevant_qrels"],
            "source_union_micro_coverage": summary["fusion_analysis"]["source_union"]["source_union_micro_coverage"],
            "source_union_zero_relevant_queries": summary["fusion_analysis"]["source_union"]["source_union_zero_relevant_queries"],
            "top500_retention": summary["fusion_analysis"]["complement_retention"],
            "rrf_top500_relevant_qrels": summary["fusion_analysis"]["source_union_loss"]["rrf_top500_relevant_qrels"],
            "relevant_qrels_discarded_by_rrf_top500": summary["fusion_analysis"]["source_union_loss"]["relevant_qrels_discarded_by_rrf_top500"],
            "rrf_top500_zero_relevant_queries": summary["fusion_analysis"]["source_union_loss"]["rrf_top500_zero_relevant_queries"],
            "relevant_source_category_contribution": summary["fusion_analysis"]["source_category_contribution"],
            "bge_top500_qrel_rank_movement": summary["fusion_analysis"]["rank_movement"],
            "bm25_only_qrels_entering_rrf": summary["fusion_analysis"]["bm25_only_entry_depths"],
        },
        "primary_paired": summary["rrf_vs_bge_m3"],
        "secondary_paired": summary["rrf_vs_bm25"],
        "three_way": summary["three_way_safe_canonical"],
        "depth_comparison": summary["depth_comparison"],
        "decision": summary["decision"],
    }
    return sha256_json(payload)


def load_source(path: Path) -> dict[str, Any]:
    if not path.is_dir() or path.name.endswith(".partial"):
        raise BSARDRerankerError(f"frozen source unavailable: {path}")
    return {name: strict_load(path / filename) for name, filename in (("manifest", "run_manifest.json"), ("summary", "summary_metrics.json"), ("rankings", "retrieval_rankings.json"))}


def validate_source(source: Mapping[str, Any], *, kind: str, dataset: Any) -> None:
    expected = {
        "bge": (BGE_RUN_ID, BGE_RETRIEVAL_FP, BGE_RESULT_FP),
        "rrf": (RRF_RUN_ID, RRF_RETRIEVAL_FP, RRF_RESULT_FP),
    }[kind]
    manifest, summary, rankings = source["manifest"], source["summary"], source["rankings"]
    if (manifest.get("run_id"), manifest.get("retrieval_fingerprint"), manifest.get("result_fingerprint")) != expected:
        raise BSARDRerankerError(f"{kind} source identity mismatch")
    records = rankings.get("records")
    if ranking_fingerprint(records) != expected[1] or rankings.get("retrieval_fingerprint") != expected[1]:
        raise BSARDRerankerError(f"{kind} source ranking fingerprint mismatch")
    if kind == "bge":
        result = sha256_json({"retrieval_fingerprint": expected[1], "official": summary["official_compat"], "safe": summary["safe_canonical"], "coverage": summary["coverage"], "complementarity": summary["complementarity"], "bootstrap": summary["bm25_control"]["paired_bootstrap"]})
    else:
        result = _rrf_result_fingerprint(summary, expected[1])
    if result != expected[2]:
        raise BSARDRerankerError(f"{kind} source result content mismatch")
    identity = (manifest.get("corpus_fingerprint"), manifest.get("test_split_fingerprint"), manifest.get("test_unique_qrel_fingerprint"))
    if identity != (CORPUS_FP, TEST_FP, QREL_FP) or manifest.get("raw_dataset_sha256") != dataset.provenance["raw_sha256"]:
        raise BSARDRerankerError(f"{kind} source benchmark identity mismatch")
    known = [document["document_id"] for document in dataset.documents]
    validate_rankings(records, dataset.human_test, known, maximum_depth=SOURCE_DEPTH)
    if len(records) != 222 or any(len(record["ranking"]) != SOURCE_DEPTH for record in records):
        raise BSARDRerankerError(f"{kind} source shape mismatch")
    if any(not math.isfinite(float(item["score"])) for record in records for item in record["ranking"]):
        raise BSARDRerankerError(f"{kind} source has non-finite score")


def ranking_map(records: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {str(record["query_id"]): list(record["ranking"]) for record in records}


def extract_candidate_pools(source_records: Sequence[Mapping[str, Any]], query_ids: Sequence[str]) -> dict[int, dict[str, list[str]]]:
    source = ranking_map(source_records)
    if set(source) != set(query_ids):
        raise BSARDRerankerError("source qid mismatch")
    output = {depth: {} for depth in CANDIDATE_DEPTHS}
    for qid in query_ids:
        ids = [str(item["standard_law_id"]) for item in source[qid]]
        if len(ids) != SOURCE_DEPTH or len(ids) != len(set(ids)):
            raise BSARDRerankerError("source ranking shape/uniqueness mismatch")
        for depth in CANDIDATE_DEPTHS:
            output[depth][qid] = ids[:depth]
        if output[20][qid] != output[50][qid][:20]:
            raise BSARDRerankerError("N20 is not exact N50 prefix")
    return output


def candidate_ceiling(pool: Mapping[str, Sequence[str]], queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relevant_pairs = 0
    zero = 0
    for query in queries:
        present = set(pool[str(query["query_id"])]) & set(query["relevant_document_ids"])
        relevant_pairs += len(present)
        zero += not present
    return {
        "candidate_pairs": sum(len(values) for values in pool.values()),
        "relevant_qrel_pairs_available": relevant_pairs,
        "missed_qrel_pairs": 1060 - relevant_pairs,
        "queries_with_relevant_candidate": len(queries) - zero,
        "zero_relevant_queries": zero,
    }


def build_unique_pairs(bge50: Mapping[str, Sequence[str]], rrf50: Mapping[str, Sequence[str]], query_ids: Sequence[str]) -> tuple[list[tuple[str, str]], dict[str, int]]:
    pairs = []
    overlap = 0
    for qid in query_ids:
        left, right = list(bge50[qid]), list(rrf50[qid])
        overlap += len(set(left) & set(right))
        pairs.extend((qid, document_id) for document_id in sorted(set(left) | set(right), key=int))
    if len(pairs) != len(set(pairs)):
        raise BSARDRerankerError("duplicate authoritative pair")
    return pairs, {
        "bge_n20": len(query_ids) * 20,
        "bge_n50": len(query_ids) * 50,
        "rrf_n20": len(query_ids) * 20,
        "rrf_n50": len(query_ids) * 50,
        "unique_union": len(pairs),
        "bge_rrf_n50_overlap": overlap,
        "duplicate_scoring_avoided_vs_n50_separate": len(query_ids) * 100 - len(pairs),
        "duplicate_scoring_avoided_vs_all_four_separate": len(query_ids) * 140 - len(pairs),
    }


def pair_content_records(pairs: Sequence[tuple[str, str]], queries: Mapping[str, str], documents: Mapping[str, str]) -> list[dict[str, str]]:
    records = []
    for qid, document_id in pairs:
        query, document = queries[qid], documents[document_id]
        qfp = sha256_json(query); dfp = sha256_json(document)
        records.append({"query_id": qid, "document_id": document_id, "query_text_fingerprint": qfp, "document_text_fingerprint": dfp, "pair_fingerprint": sha256_json([qid, document_id, qfp, dfp])})
    return records


def pair_set_fingerprint(pairs: Sequence[tuple[str, str]]) -> str:
    return sha256_json(sorted(([qid, document_id] for qid, document_id in pairs), key=lambda value: (int(value[0]), int(value[1]))))


def ordered_pair_fingerprint(pairs: Sequence[tuple[str, str]]) -> str:
    return sha256_json([[qid, document_id] for qid, document_id in pairs])


def audit_token_lengths(pairs: Sequence[tuple[str, str]], queries: Mapping[str, str], documents: Mapping[str, str], tokenizer: Any) -> dict[str, Any]:
    special = int(tokenizer.num_special_tokens_to_add(pair=True))
    query_lengths = {qid: len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]) for qid, text in queries.items()}
    needed_documents = {document_id for _, document_id in pairs}
    document_lengths = {document_id: len(tokenizer(documents[document_id], add_special_tokens=False, truncation=False)["input_ids"]) for document_id in needed_documents}
    lengths = np.asarray([query_lengths[qid] + document_lengths[document_id] + special for qid, document_id in pairs], dtype=np.int64)
    affected_pairs = [(qid, document_id) for qid, document_id in pairs if query_lengths[qid] + document_lengths[document_id] + special > MAX_LENGTH]
    query_threats = [qid for qid, length in query_lengths.items() if length + special >= MAX_LENGTH]
    values = np.percentile(lengths, (50, 90, 95, 99), method="linear")
    return {
        "pair_count": len(pairs), "min": int(lengths.min()), "median": float(values[0]),
        "p90": float(values[1]), "p95": float(values[2]), "p99": float(values[3]), "max": int(lengths.max()),
        "effective_max_length": MAX_LENGTH, "truncation": TRUNCATION, "truncation_side": "document_only",
        "pairs_exceeding_max": len(affected_pairs), "fraction_exceeding_max": len(affected_pairs) / len(pairs),
        "affected_query_count": len({qid for qid, _ in affected_pairs}),
        "affected_document_count": len({document_id for _, document_id in affected_pairs}),
        "affected_pair_count": len(affected_pairs), "max_query_tokens_without_specials": max(query_lengths.values()),
        "query_budget_threats": len(query_threats), "query_preservation_failures": 0,
        "tokenizer_class": tokenizer.__class__.__name__, "tokenizer_model_max_length": int(tokenizer.model_max_length),
    }


def enforce_query_preservation(audit: Mapping[str, Any]) -> None:
    if audit.get("query_budget_threats") != 0 or audit.get("query_preservation_failures") != 0:
        raise BSARDRerankerError("query preservation gate failed")


def score_vector_fingerprint(scores: np.ndarray) -> str:
    values = np.asarray(scores)
    return sha256_json({"dtype": str(values.dtype), "shape": list(values.shape), "bytes_sha256": __import__("hashlib").sha256(values.tobytes(order="C")).hexdigest()})


def cache_context(*, pairs: Sequence[tuple[str, str]], contents: Sequence[Mapping[str, str]], model_hashes: Mapping[str, str], runner: Path, config: Path, commit: str) -> dict[str, Any]:
    context = {
        "cache_schema_version": CACHE_SCHEMA, "status": "complete", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "model_artifact_fingerprints": dict(model_hashes),
        "tokenizer_fingerprint": sha256_json({key: model_hashes[key] for key in sorted(model_hashes) if "token" in key or "sentencepiece" in key or "special" in key}),
        "scoring_contract": {"input_order": ["official_question", "official_article"], "max_length": MAX_LENGTH, "truncation": TRUNCATION, "padding": True, "dtype": DTYPE, "device": DEVICE, "batch_size": BATCH_SIZE, "model_eval": True, "inference_mode": True, "output": "raw_scalar_logit", "normalization": None},
        "corpus_fingerprint": CORPUS_FP, "test_split_fingerprint": TEST_FP, "test_unique_qrel_fingerprint": QREL_FP,
        "bge_source_retrieval_fingerprint": BGE_RETRIEVAL_FP, "rrf_source_retrieval_fingerprint": RRF_RETRIEVAL_FP,
        "candidate_pair_set_fingerprint": pair_set_fingerprint(pairs), "ordered_pair_fingerprint": ordered_pair_fingerprint(pairs),
        "pair_content_fingerprint": sha256_json(list(contents)), "runner_source_fingerprint": sha256_file(runner),
        "config_fingerprint": sha256_file(config), "git_commit": commit, "pair_count": len(pairs), "score_dtype": DTYPE,
    }
    context["context_fingerprint"] = sha256_json(context)
    return context


def validate_context(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if actual != expected or actual.get("context_fingerprint") != sha256_json({key: value for key, value in actual.items() if key != "context_fingerprint"}):
        raise BSARDRerankerError("score cache provenance mismatch")


def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _shard_metadata(context: Mapping[str, Any], pairs: Sequence[tuple[str, str]], contents: Sequence[Mapping[str, str]], start: int, end: int, scores: np.ndarray | None = None) -> dict[str, Any]:
    metadata = {
        "context_fingerprint": context["context_fingerprint"], "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "start": start, "end": end, "ordered_pair_fingerprint": ordered_pair_fingerprint(pairs[start:end]),
        "pair_content_fingerprint": sha256_json(list(contents[start:end])), "score_dtype": DTYPE,
    }
    if scores is not None:
        metadata["score_vector_fingerprint"] = score_vector_fingerprint(scores)
    metadata["metadata_fingerprint"] = sha256_json(metadata)
    return metadata


def save_shard(path: Path, scores: np.ndarray, metadata: Mapping[str, Any]) -> None:
    values = np.asarray(scores)
    if values.dtype != np.float32 or values.ndim != 1 or not np.isfinite(values).all():
        raise BSARDRerankerError("invalid score shard")
    _atomic_npz(path, scores=values, metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True, allow_nan=False)))


def load_shard(path: Path, expected: Mapping[str, Any], count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"scores", "metadata_json"}:
            raise BSARDRerankerError("incomplete score shard")
        scores = np.asarray(archive["scores"]); metadata = json.loads(str(archive["metadata_json"].item()))
    if metadata != expected or metadata.get("metadata_fingerprint") != sha256_json({key: value for key, value in metadata.items() if key != "metadata_fingerprint"}):
        raise BSARDRerankerError("score shard provenance mismatch")
    if scores.shape != (count,) or scores.dtype != np.float32 or not np.isfinite(scores).all() or metadata.get("score_vector_fingerprint") != score_vector_fingerprint(scores):
        raise BSARDRerankerError("score shard validation failed")
    return scores


def save_final_cache(path: Path, scores: np.ndarray, context: Mapping[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
    values = np.asarray(scores)
    metadata = {**context, "score_vector_fingerprint": score_vector_fingerprint(values), "timing": dict(timing)}
    metadata["metadata_fingerprint"] = sha256_json(metadata)
    _atomic_npz(path, scores=values, metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))
    return metadata


def load_final_cache(cache_dir: Path, context: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if cache_dir.name.endswith(".partial") or not cache_dir.is_dir() or not (cache_dir / "scores.npz").is_file():
        raise BSARDRerankerError("partial cache cannot be interpreted as final")
    with np.load(cache_dir / "scores.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"scores", "metadata_json"}:
            raise BSARDRerankerError("incomplete final score cache")
        scores = np.asarray(archive["scores"]); metadata = json.loads(str(archive["metadata_json"].item()))
    stored = metadata.get("metadata_fingerprint")
    if stored != sha256_json({key: value for key, value in metadata.items() if key != "metadata_fingerprint"}):
        raise BSARDRerankerError("cache metadata fingerprint mismatch")
    for key, value in context.items():
        if metadata.get(key) != value:
            raise BSARDRerankerError(f"cache context mismatch: {key}")
    if scores.shape != (context["pair_count"],) or scores.dtype != np.float32 or not np.isfinite(scores).all() or metadata.get("score_vector_fingerprint") != score_vector_fingerprint(scores):
        raise BSARDRerankerError("final score cache vector validation failed")
    return scores, metadata


def load_model(snapshot: Path = MODEL_SNAPSHOT):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    started = time.perf_counter(); before = peak_rss_mb()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot.resolve()), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(snapshot.resolve()), local_files_only=True, dtype=torch.float32, use_safetensors=True)
    model.to(device=DEVICE, dtype=torch.float32); model.eval()
    return tokenizer, model, torch, {"load_seconds": time.perf_counter() - started, "peak_rss_before_mb": before, "peak_rss_after_mb": peak_rss_mb(), "architecture": model.__class__.__name__, "tokenizer": tokenizer.__class__.__name__, "device": DEVICE, "dtype": DTYPE}


def score_batch(batch: Sequence[tuple[str, str]], queries: Mapping[str, str], documents: Mapping[str, str], tokenizer: Any, model: Any, torch_module: Any) -> np.ndarray:
    encoded = tokenizer([queries[qid] for qid, _ in batch], [documents[document_id] for _, document_id in batch], max_length=MAX_LENGTH, truncation=TRUNCATION, padding=True, return_tensors="pt")
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch_module.inference_mode():
        logits = model(**encoded).logits
    expected = (len(batch), 1)
    if logits.ndim != 2 or tuple(logits.shape) != expected:
        raise BSARDRerankerError(f"model logits must have exact shape {expected}; got {tuple(logits.shape)}")
    values = np.asarray(logits[:, 0].detach().cpu().tolist(), dtype=np.float32)
    if not np.isfinite(values).all():
        raise BSARDRerankerError("reranker produced NaN or Infinity")
    return values


def build_score_cache(cache_root: Path, context: Mapping[str, Any], pairs: Sequence[tuple[str, str]], contents: Sequence[Mapping[str, str]], queries: Mapping[str, str], documents: Mapping[str, str], tokenizer: Any, model: Any, torch_module: Any) -> tuple[np.ndarray, dict[str, Any]]:
    final = cache_root / CACHE_SCHEMA; partial = cache_root / f"{CACHE_SCHEMA}.partial"
    if final.exists():
        return load_final_cache(final, context)
    partial.mkdir(parents=True, exist_ok=True)
    context_path = partial / "context.json"
    if context_path.exists(): validate_context(strict_load(context_path), context)
    else: strict_write(context_path, context)
    chunks = []; scoring_started = time.perf_counter(); scored_now = 0
    for start in range(0, len(pairs), SHARD_SIZE):
        end = min(start + SHARD_SIZE, len(pairs)); path = partial / f"scores_{start:06d}_{end:06d}.npz"
        empty_meta = _shard_metadata(context, pairs, contents, start, end)
        if path.exists():
            # The score fingerprint is recovered first, then the complete bound metadata is validated.
            with np.load(path, allow_pickle=False) as archive:
                probe = np.asarray(archive["scores"])
            expected = _shard_metadata(context, pairs, contents, start, end, probe)
            values = load_shard(path, expected, end - start)
        else:
            values_list = []
            for offset in range(start, end, BATCH_SIZE):
                values_list.append(score_batch(pairs[offset:min(offset + BATCH_SIZE, end)], queries, documents, tokenizer, model, torch_module))
            values = np.concatenate(values_list).astype(np.float32, copy=False)
            metadata = _shard_metadata(context, pairs, contents, start, end, values)
            save_shard(path, values, metadata); scored_now += len(values)
        chunks.append(values)
        if end % 500 == 0 or end == len(pairs): print(f"BSARD reranker pairs {end}/{len(pairs)}", flush=True)
    scores = np.concatenate(chunks).astype(np.float32, copy=False)
    timing = {"pair_scoring_seconds_this_invocation": time.perf_counter() - scoring_started, "pairs_scored_this_invocation": scored_now, "mean_seconds_per_new_pair": (time.perf_counter() - scoring_started) / scored_now if scored_now else 0.0, "peak_rss_mb": peak_rss_mb()}
    metadata = save_final_cache(partial / "scores.npz", scores, context, timing)
    load_final_cache(partial.with_name("__invalid_final_name__"), context) if False else None
    os.replace(partial, final)
    replay_scores, replay_metadata = load_final_cache(final, context)
    if not np.array_equal(scores, replay_scores) or replay_metadata != metadata:
        raise BSARDRerankerError("final cache publication replay mismatch")
    return replay_scores, replay_metadata


def score_map(pairs: Sequence[tuple[str, str]], scores: np.ndarray) -> dict[tuple[str, str], float]:
    if len(pairs) != len(scores) or len(pairs) != len(set(pairs)):
        raise BSARDRerankerError("pair/score cardinality mismatch")
    output = {(qid, document_id): float(score) for (qid, document_id), score in zip(pairs, scores)}
    if not all(math.isfinite(value) for value in output.values()):
        raise BSARDRerankerError("non-finite authoritative score")
    return output


def rerank_one(source: Sequence[Mapping[str, Any]], qid: str, depth: int, scores: Mapping[tuple[str, str], float]) -> list[dict[str, Any]]:
    if depth not in CANDIDATE_DEPTHS or len(source) != SOURCE_DEPTH:
        raise BSARDRerankerError("invalid rerank depth/source")
    head = [(str(item["standard_law_id"]), rank, scores[(qid, str(item["standard_law_id"]))]) for rank, item in enumerate(source[:depth], 1)]
    head.sort(key=lambda item: (-item[2], item[1], int(item[0])))
    output = [{"rank": rank, "standard_law_id": document_id, "score": float(score), "reranker_score": float(score), "source_rank": source_rank} for rank, (document_id, source_rank, score) in enumerate(head, 1)]
    for rank, item in enumerate(source[depth:], depth + 1):
        output.append({"rank": rank, "standard_law_id": str(item["standard_law_id"]), "score": float(item["score"]), "reranker_score": None, "source_rank": rank})
    ids = [item["standard_law_id"] for item in output]
    source_ids = [str(item["standard_law_id"]) for item in source]
    if len(ids) != SOURCE_DEPTH or len(ids) != len(set(ids)) or set(ids[:depth]) != set(source_ids[:depth]) or ids[depth:] != source_ids[depth:] or set(ids) != set(source_ids):
        raise BSARDRerankerError("reranked-head/tail invariant failed")
    return output


def derive_systems(bge_records: Sequence[Mapping[str, Any]], rrf_records: Sequence[Mapping[str, Any]], query_ids: Sequence[str], scores: Mapping[tuple[str, str], float]) -> dict[str, list[dict[str, Any]]]:
    sources = {"bge": ranking_map(bge_records), "rrf": ranking_map(rrf_records)}; systems = {}
    for generator in ("bge", "rrf"):
        for depth in CANDIDATE_DEPTHS:
            name = f"{generator}_rerank_n{depth}"
            systems[name] = [{"query_id": qid, "ranking": rerank_one(sources[generator][qid], qid, depth, scores)} for qid in query_ids]
    return systems


def evaluate(records: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]], known_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_qid = {str(record["query_id"]): [str(item["standard_law_id"]) for item in record["ranking"]] for record in records}
    qids = [str(query["query_id"]) for query in queries]
    official = evaluate_official_compat([query["raw_relevant_document_ids"] for query in queries], [by_qid[qid] for qid in qids], known_document_ids=known_ids, query_ids=qids)
    safe = evaluate_safe_canonical({str(query["query_id"]): query["relevant_document_ids"] for query in queries}, by_qid, known_document_ids=known_ids)
    return official, safe


def paired(control: Mapping[str, Any], treatment: Mapping[str, Any], metrics: Iterable[str]) -> dict[str, Any]:
    qids = sorted(control["per_query"], key=int)
    return {metric: paired_bootstrap_mean_delta([control["per_query"][qid]["metrics"][metric] for qid in qids], [treatment["per_query"][qid]["metrics"][metric] for qid in qids], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED) for metric in metrics}


def failure_and_promotion(source_records: Sequence[Mapping[str, Any]], reranked_records: Sequence[Mapping[str, Any]], depth: int, queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = {str(record["query_id"]): [str(item["standard_law_id"]) for item in record["ranking"]] for record in source_records}
    reranked = {str(record["query_id"]): [str(item["standard_law_id"]) for item in record["ranking"]] for record in reranked_records}
    query_counts = {"candidate_failure": 0, "ranking_failure": 0, "success": 0}; reference = {"absent_candidate_pool": 0, "present_below_top10": 0, "present_top10": 0}
    changes = []; improved = worsened = tied = promoted = demoted = 0
    for query in queries:
        qid = str(query["query_id"]); relevant = set(query["relevant_document_ids"]); candidates = set(source[qid][:depth]); top10 = set(reranked[qid][:10])
        if not relevant & candidates: query_counts["candidate_failure"] += 1
        elif not relevant & top10: query_counts["ranking_failure"] += 1
        else: query_counts["success"] += 1
        for document_id in relevant:
            if document_id not in candidates: reference["absent_candidate_pool"] += 1; continue
            old = source[qid].index(document_id) + 1; new = reranked[qid].index(document_id) + 1; change = old - new; changes.append(change)
            improved += change > 0; worsened += change < 0; tied += change == 0
            promoted += old > 10 and new <= 10; demoted += old <= 10 and new > 10
            if new <= 10: reference["present_top10"] += 1
            else: reference["present_below_top10"] += 1
    available = len(queries) - query_counts["candidate_failure"]
    return {
        "candidate_depth": depth,
        "query_level": {**query_counts, "total": len(queries), "conditional_top10_success_rate": query_counts["success"] / available if available else 0.0},
        "reference_level": {**reference, "total_unique_qrels": 1060},
        "promotion": {"available_relevant_pairs": len(changes), "improved": improved, "worsened": worsened, "tied": tied, "median_source_rank_minus_reranked_rank": float(median(changes)) if changes else 0.0, "promoted_into_top10": promoted, "demoted_out_of_top10": demoted},
    }


def score_distribution(pairs: Sequence[tuple[str, str]], scores: np.ndarray, queries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    relevant = {(str(query["query_id"]), document_id) for query in queries for document_id in query["relevant_document_ids"]}
    groups = {"relevant": [], "nonrelevant": []}
    for pair, score in zip(pairs, scores): groups["relevant" if pair in relevant else "nonrelevant"].append(float(score))
    output = {}
    for name, values in groups.items():
        array = np.asarray(values, dtype=float); output[name] = {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)), "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}
    return output


def classify_reranker(comparisons: Mapping[str, Any]) -> str:
    def reliable(name: str) -> int:
        rows = [comparisons[name][metric] for metric in SENSITIVE_METRICS]
        if any(row["ci_upper"] < 0 for row in rows): return -1
        if any(row["ci_lower"] > 0 for row in rows) and sum(row["mean_delta"] < 0 for row in rows) == 0: return 1
        return 0
    bge, rrf = reliable("bge_n50_vs_source"), reliable("rrf_n50_vs_source")
    if bge < 0 and rrf < 0: return "RERANKER_HURTS"
    if bge > 0 and rrf > 0: return "RERANKER_IMPROVES_BOTH_GENERATORS"
    if bge > 0 and rrf == 0: return "RERANKER_IMPROVES_BGE_ONLY"
    if rrf > 0 and bge == 0: return "RERANKER_IMPROVES_RRF_ONLY"
    if bge == 0 and rrf == 0: return "RERANKER_NO_RELIABLE_GAIN"
    return "RERANKER_MIXED"


def classify_generator(comparison: Mapping[str, Any]) -> str:
    rows = [comparison[metric] for metric in SENSITIVE_METRICS]
    if all(row["mean_delta"] >= 0 for row in rows) and any(row["ci_lower"] > 0 for row in rows): return "RRF_BETTER_RERANKER_INPUT"
    if all(row["mean_delta"] <= 0 for row in rows) and any(row["ci_upper"] < 0 for row in rows): return "BGE_BETTER_RERANKER_INPUT"
    if any(row["mean_delta"] > 0 for row in rows) and any(row["mean_delta"] < 0 for row in rows): return "GENERATOR_TRADEOFF"
    return "NO_CLEAR_GENERATOR_WINNER"


def classify_depth(comparisons: Mapping[str, Any], ceilings: Mapping[str, Any]) -> str:
    rows = [comparisons[name][metric] for name in ("bge_n50_vs_n20", "rrf_n50_vs_n20") for metric in SENSITIVE_METRICS]
    if all(row["mean_delta"] >= 0 for row in rows) and any(row["ci_lower"] > 0 for row in rows): return "N50_PREFERRED"
    if all(row["mean_delta"] <= 0 for row in rows) and all(ceilings[f"{generator}_n50"]["relevant_qrel_pairs_available"] == ceilings[f"{generator}_n20"]["relevant_qrel_pairs_available"] for generator in ("bge", "rrf")): return "N20_PREFERRED"
    if any(row["mean_delta"] > 0 for row in rows) and any(row["mean_delta"] < 0 for row in rows): return "DEPTH_TRADEOFF"
    return "NO_CLEAR_DEPTH_WINNER"


def publish(final: Path, config: Path, artifacts: Mapping[str, Any]) -> Path:
    expected = {"system_rankings.json", "summary_metrics.json", "failure_decomposition.json", "candidate_analysis.json", "resource_metrics.json", "run_manifest.json"}
    partial = final.with_name(final.name + ".partial")
    if set(artifacts) != expected or final.exists() or partial.exists(): raise BSARDRerankerError("unsafe artifact publication")
    partial.mkdir(parents=True)
    try:
        for name, payload in artifacts.items(): strict_write(partial / name, payload)
        shutil.copy2(config, partial / "eval_config.snapshot.yaml")
        if {path.name for path in partial.iterdir()} != expected | {"eval_config.snapshot.yaml"}: raise BSARDRerankerError("incomplete artifact publication")
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True); raise
    return final


def run(*, raw_root: Path, config_path: Path, output_root: Path) -> Path:
    total_started = time.perf_counter(); root = Path(__file__).resolve().parents[4]; runner = Path(__file__).resolve(); commit = git_head(root)
    model_hashes = validate_model_files(); dataset = load_bsard(raw_root, strict_identity=True)
    if (len(dataset.documents), len(dataset.human_test), sum(len(query["relevant_document_ids"]) for query in dataset.human_test)) != (22633, 222, 1060): raise BSARDRerankerError("canonical BSARD counts changed")
    if (dataset.fingerprints["corpus_ordered"], dataset.fingerprints["test_split"], dataset.fingerprints["test_unique_qrels"]) != (CORPUS_FP, TEST_FP, QREL_FP): raise BSARDRerankerError("canonical BSARD fingerprints changed")
    runs = root / "backend/data/runs"; bge = load_source(runs / BGE_RUN_ID); rrf = load_source(runs / RRF_RUN_ID)
    validate_source(bge, kind="bge", dataset=dataset); validate_source(rrf, kind="rrf", dataset=dataset)
    queries = {str(query["query_id"]): query["query_text"] for query in dataset.human_test}; documents = {str(document["document_id"]): document["document_text"] for document in dataset.documents}; query_ids = list(queries)
    bge_pools = extract_candidate_pools(bge["rankings"]["records"], query_ids); rrf_pools = extract_candidate_pools(rrf["rankings"]["records"], query_ids)
    ceilings = {f"{generator}_n{depth}": candidate_ceiling(pools[depth], dataset.human_test) for generator, pools in (("bge", bge_pools), ("rrf", rrf_pools)) for depth in CANDIDATE_DEPTHS}
    for name, expected in EXPECTED_CEILINGS.items():
        if (ceilings[name]["relevant_qrel_pairs_available"], ceilings[name]["zero_relevant_queries"]) != expected: raise BSARDRerankerError(f"candidate ceiling changed: {name}")
    pairs, pair_counts = build_unique_pairs(bge_pools[50], rrf_pools[50], query_ids); contents = pair_content_records(pairs, queries, documents)
    tokenizer, model, torch_module, model_load = load_model()
    token_started = time.perf_counter(); token_audit = audit_token_lengths(pairs, queries, documents, tokenizer); token_seconds = time.perf_counter() - token_started; enforce_query_preservation(token_audit)
    context = cache_context(pairs=pairs, contents=contents, model_hashes=model_hashes, runner=runner, config=config_path, commit=commit)
    scores, cache_meta = build_score_cache(output_root, context, pairs, contents, queries, documents, tokenizer, model, torch_module)
    del model; gc.collect()
    cache_validation_started = time.perf_counter(); replay_scores, replay_meta = load_final_cache(output_root / CACHE_SCHEMA, context); cache_validation_seconds = time.perf_counter() - cache_validation_started
    if not np.array_equal(scores, replay_scores) or replay_meta["score_vector_fingerprint"] != cache_meta["score_vector_fingerprint"]: raise BSARDRerankerError("score cache replay mismatch")
    authoritative = score_map(pairs, scores); derive_started = time.perf_counter(); systems = derive_systems(bge["rankings"]["records"], rrf["rankings"]["records"], query_ids, authoritative); derivation_seconds = time.perf_counter() - derive_started
    known = list(documents); source_records = {"bge_source": bge["rankings"]["records"], "rrf_source": rrf["rankings"]["records"]}
    metric_started = time.perf_counter(); evaluations = {}
    for name, records in {**source_records, **systems}.items():
        official, safe = evaluate(records, dataset.human_test, known); evaluations[name] = {"official_compat": official, "safe_canonical": safe}
    for generator in ("bge", "rrf"):
        for depth in CANDIDATE_DEPTHS:
            metrics = evaluations[f"{generator}_rerank_n{depth}"]["safe_canonical"]["metrics"]; source_metrics = evaluations[f"{generator}_source"]["safe_canonical"]["metrics"]
            if any(metrics[key] != source_metrics[key] for key in ("recall@100", "recall@200", "recall@500")): raise BSARDRerankerError("deep recall invariant failed")
    comparisons = {}
    for generator in ("bge", "rrf"):
        for depth in CANDIDATE_DEPTHS:
            metrics = SENSITIVE_METRICS + (("recall@20", "recall@30") if depth == 50 else ())
            comparisons[f"{generator}_n{depth}_vs_source"] = paired(evaluations[f"{generator}_source"]["safe_canonical"], evaluations[f"{generator}_rerank_n{depth}"]["safe_canonical"], metrics)
        comparisons[f"{generator}_n50_vs_n20"] = paired(evaluations[f"{generator}_rerank_n20"]["safe_canonical"], evaluations[f"{generator}_rerank_n50"]["safe_canonical"], ("recall@10", "recall@20", "recall@30", "recall@50", "map@100", "mrr@100", "ndcg@10"))
    comparisons["rrf_vs_bge_n50"] = paired(evaluations["bge_rerank_n50"]["safe_canonical"], evaluations["rrf_rerank_n50"]["safe_canonical"], SENSITIVE_METRICS)
    comparisons["rrf_vs_bge_n20"] = paired(evaluations["bge_rerank_n20"]["safe_canonical"], evaluations["rrf_rerank_n20"]["safe_canonical"], SENSITIVE_METRICS)
    failure = {name: failure_and_promotion(source_records[f"{name.split('_')[0]}_source"], records, int(name.rsplit("n", 1)[1]), dataset.human_test) for name, records in systems.items()}
    distribution = score_distribution(pairs, scores, dataset.human_test); metric_seconds = time.perf_counter() - metric_started
    ranking_fps = {name: ranking_fingerprint(records) for name, records in systems.items()}
    result_fps = {name: sha256_json({"ranking_fingerprint": ranking_fps[name], **evaluations[name]}) for name in systems}
    replay_systems = derive_systems(bge["rankings"]["records"], rrf["rankings"]["records"], query_ids, score_map(pairs, replay_scores)); replay_ranking_fps = {name: ranking_fingerprint(records) for name, records in replay_systems.items()}
    replay_result_fps = {}; replay_metrics = {}
    for name, records in replay_systems.items():
        official, safe = evaluate(records, dataset.human_test, known); replay_metrics[name] = {"official_compat": official, "safe_canonical": safe}; replay_result_fps[name] = sha256_json({"ranking_fingerprint": replay_ranking_fps[name], **replay_metrics[name]})
    if replay_ranking_fps != ranking_fps or replay_result_fps != result_fps or replay_metrics != {name: evaluations[name] for name in systems}: raise BSARDRerankerError("ranking/metric replay mismatch")
    decisions = {"reranker": classify_reranker(comparisons), "candidate_generator": classify_generator(comparisons["rrf_vs_bge_n50"]), "candidate_depth": classify_depth(comparisons, ceilings)}
    run_id = f"{PROTOCOL_ID}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    system_artifact = {"run_id": run_id, "query_count": 222, "ranking_depth": 500, "systems": systems, "ranking_fingerprints": ranking_fps}
    resource_metrics = {"model_load": model_load, "pair_token_audit_seconds": token_seconds, "pair_scoring": cache_meta["timing"], "cache_validation_seconds": cache_validation_seconds, "system_derivation_seconds": derivation_seconds, "metric_analysis_seconds": metric_seconds, "total_incremental_seconds": time.perf_counter() - total_started, "unique_pairs_scored": len(pairs), "score_cache_bytes": (output_root / CACHE_SCHEMA / "scores.npz").stat().st_size, "peak_rss_mb": peak_rss_mb(), "conceptual_pair_budget_per_query": {"n20": 20, "n50": 50}, "actual_unique_total_scoring_work": len(pairs)}
    summary = {"metric_semantics": ["bsard_official_compat", "bsard_safe_canonical"], "evaluations": evaluations, "six_system_table": {name: evaluations[name]["safe_canonical"]["metrics"] for name in evaluations}, "paired_bootstrap": comparisons, "decisions": decisions, "deep_recall_generator_invariant": True}
    candidate_analysis = {"candidate_counts": pair_counts, "ceilings": ceilings, "score_distribution": distribution, "promotion": {name: value["promotion"] for name, value in failure.items()}, "n20_vs_n50_incremental_pairs_per_generator": 222 * 30, "unique_pair_work_not_double_counted": True}
    manifest = {"run_id": run_id, "protocol_id": PROTOCOL_ID, "git_commit": commit, "runner_source_fingerprint": sha256_file(runner), "config_fingerprint": sha256_file(config_path), "raw_dataset_sha256": dataset.provenance["raw_sha256"], "corpus_fingerprint": CORPUS_FP, "test_split_fingerprint": TEST_FP, "test_unique_qrel_fingerprint": QREL_FP, "source_runs": {"bge": {"run_id": BGE_RUN_ID, "retrieval_fingerprint": BGE_RETRIEVAL_FP, "result_fingerprint": BGE_RESULT_FP}, "rrf": {"run_id": RRF_RUN_ID, "retrieval_fingerprint": RRF_RETRIEVAL_FP, "result_fingerprint": RRF_RESULT_FP}}, "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": str(MODEL_SNAPSHOT.resolve()), "files": model_hashes}, "scoring_contract": context["scoring_contract"], "candidate_pair_set_fingerprint": context["candidate_pair_set_fingerprint"], "ordered_pair_fingerprint": context["ordered_pair_fingerprint"], "pair_content_fingerprint": context["pair_content_fingerprint"], "score_vector_fingerprint": cache_meta["score_vector_fingerprint"], "score_cache_metadata_fingerprint": cache_meta["metadata_fingerprint"], "score_cache_path": str((output_root / CACHE_SCHEMA).resolve()), "ranking_fingerprints": ranking_fps, "result_fingerprints": result_fps, "replay_ranking_fingerprints": replay_ranking_fps, "replay_result_fingerprints": replay_result_fps, "deterministic_replay_passed": True, "token_audit": token_audit, "models_executed": [MODEL_ID], "retrievers_executed": [], "reranker_used": True}
    artifacts = {"system_rankings.json": system_artifact, "summary_metrics.json": summary, "failure_decomposition.json": failure, "candidate_analysis.json": candidate_analysis, "resource_metrics.json": resource_metrics, "run_manifest.json": manifest}
    return publish(output_root / run_id, config_path, artifacts)


def main(argv=None):
    root = Path(__file__).resolve().parents[4]; parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True); parser.add_argument("--config", type=Path, default=root / "backend/scripts/Evaluation/configs/bsard_external_zero_shot_v1.yaml"); parser.add_argument("--output-root", type=Path, default=root / "backend/data/runs")
    args = parser.parse_args(argv); result = run(raw_root=args.raw_root.resolve(), config_path=args.config.resolve(), output_root=args.output_root.resolve()); print(result); return result


if __name__ == "__main__":
    main()
