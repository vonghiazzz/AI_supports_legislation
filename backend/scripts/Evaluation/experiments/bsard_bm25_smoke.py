"""BSARD Phase 1C: deterministic BM25-only end-to-end smoke evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from ..benchmarking.bsard_adapter import load_bsard
from ..evaluation.bsard_metrics import (
    evaluate_official_compat,
    evaluate_safe_canonical,
)
from ..retrieval.bm25 import build_bm25_index, search_bm25


PROTOCOL_ID = "bsard_bm25_smoke_v1"
TOP_K = 500
DIAGNOSTIC_DEPTHS = (20, 30, 50, 100, 200, 500)
EXPECTED_DOCUMENT_COUNT = 22633
EXPECTED_TRAIN_QUERY_COUNT = 886
EXPECTED_TEST_QUERY_COUNT = 222
EXPECTED_TEST_RAW_QRELS = 1061
EXPECTED_TEST_UNIQUE_QRELS = 1060
BM25_PARAMETERS = {"implementation": "rank_bm25.BM25Okapi", "k1": 1.5, "b": 0.75, "epsilon": 0.25}
TOKENIZATION_POLICY = {
    "tokenizer": "python_str_split_whitespace",
    "normalization": "unicode_text_unchanged_except_lowercase",
    "case_handling": "str.lower",
    "punctuation_handling": "retained_and_attached_by_whitespace_tokenization",
    "stopword_handling": "retained",
    "stemming": False,
    "lemmatization": False,
    "document_representation": "official_zenodo_article_only",
    "query_representation": "official_french_question_only",
}
PAPER_BM25 = {
    "source": "Louis and Spanakis, ACL 2022, Table 2",
    "preprocessing": {
        "tool": "spaCy fr_core_news_md",
        "lowercase": True,
        "remove_punctuation": True,
        "remove_numbers": True,
        "remove_stopwords": True,
        "lemmatization": True,
    },
    "parameters": {"k1": 1.0, "b": 0.6, "tuned_on": "BSARD train"},
    "metrics": {
        "recall@100": 0.5133,
        "recall@200": 0.5678,
        "recall@500": 0.6471,
        "map@100": 0.1604,
        "mrr@100": 0.2459,
    },
}


class BSARDBM25SmokeError(RuntimeError):
    """Raised when the smoke-run protocol fails closed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json_write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _git_head(project_root: Path) -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise BSARDBM25SmokeError("valid git HEAD is required")
    return value


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024.0
    return value / (1024.0 * 1024.0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise BSARDBM25SmokeError("latency percentile requires non-empty values")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def map_documents(documents: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Thin compatibility mapping into the shared BM25 contract."""
    return [
        {
            "standard_law_id": str(document["document_id"]),
            "bm25_text": str(document["document_text"]),
        }
        for document in documents
    ]


def retrieve_rankings(
    index: Any,
    documents: Sequence[Mapping[str, str]],
    queries: Sequence[Mapping[str, Any]],
    *,
    top_k: int = TOP_K,
) -> tuple[list[dict[str, Any]], list[float]]:
    records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in queries:
        start = time.perf_counter()
        ranking = search_bm25(index, documents, query["query_text"], top_k)
        latencies.append(time.perf_counter() - start)
        records.append({"query_id": str(query["query_id"]), "ranking": ranking})
    return records, latencies


def validate_rankings(
    records: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    known_document_ids: Iterable[str],
    *,
    maximum_depth: int = TOP_K,
) -> None:
    expected_qids = [str(query["query_id"]) for query in queries]
    actual_qids = [str(record.get("query_id", "")) for record in records]
    if actual_qids != expected_qids or len(actual_qids) != len(set(actual_qids)):
        raise BSARDBM25SmokeError("ranking artifact requires exact ordered qid coverage")
    known = set(known_document_ids)
    for record in records:
        ranking = record.get("ranking")
        if not isinstance(ranking, list) or len(ranking) > maximum_depth:
            raise BSARDBM25SmokeError("invalid ranking length")
        ids = [str(item.get("standard_law_id", "")) for item in ranking]
        if len(ids) != len(set(ids)):
            raise BSARDBM25SmokeError("duplicate predicted article ID")
        if set(ids) - known:
            raise BSARDBM25SmokeError("unknown predicted article ID")
        for expected_rank, item in enumerate(ranking, 1):
            if item.get("rank") != expected_rank:
                raise BSARDBM25SmokeError("non-contiguous BM25 rank")
            score = item.get("score")
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise BSARDBM25SmokeError("BM25 score must be finite")


def ranking_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        {
            "query_id": str(record["query_id"]),
            "ordered_document_ids": [
                str(item["standard_law_id"]) for item in record["ranking"]
            ],
        }
        for record in records
    ]
    return sha256_json(identity)


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    known_document_ids: Iterable[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    predictions = {
        str(record["query_id"]): [
            str(item["standard_law_id"]) for item in record["ranking"]
        ]
        for record in records
    }
    official = evaluate_official_compat(
        [query["raw_relevant_document_ids"] for query in queries],
        [predictions[str(query["query_id"])] for query in queries],
        known_document_ids=known_document_ids,
        query_ids=[query["query_id"] for query in queries],
    )
    safe = evaluate_safe_canonical(
        {query["query_id"]: query["relevant_document_ids"] for query in queries},
        predictions,
        known_document_ids=known_document_ids,
        recall_cutoffs=DIAGNOSTIC_DEPTHS,
        map_cutoff=100,
        mrr_cutoff=100,
        ndcg_cutoffs=(10,),
    )
    return official, safe


def coverage_diagnostics(
    records: Sequence[Mapping[str, Any]],
    queries: Sequence[Mapping[str, Any]],
    *,
    depths: Sequence[int] = DIAGNOSTIC_DEPTHS,
) -> dict[str, Any]:
    predictions = {
        str(record["query_id"]): [
            str(item["standard_law_id"]) for item in record["ranking"]
        ]
        for record in records
    }
    total_references = sum(len(query["relevant_document_ids"]) for query in queries)
    if total_references == 0:
        raise BSARDBM25SmokeError("coverage requires non-empty unique qrels")
    output: dict[str, Any] = {}
    for depth in depths:
        missed_references = 0
        zero_queries = 0
        for query in queries:
            relevant = set(query["relevant_document_ids"])
            retrieved = set(predictions[str(query["query_id"])][:depth])
            hits = len(relevant & retrieved)
            missed_references += len(relevant) - hits
            zero_queries += int(hits == 0)
        output[str(depth)] = {
            "unique_relevant_reference_denominator": total_references,
            "retrieved_relevant_references": total_references - missed_references,
            "missed_relevant_references": missed_references,
            "reference_level_candidate_miss_rate": missed_references / total_references,
            "candidate_recall": (total_references - missed_references) / total_references,
            "query_denominator": len(queries),
            "zero_relevant_query_count": zero_queries,
            "zero_relevant_query_rate": zero_queries / len(queries),
        }
    return output


def latency_summary(
    *,
    index_build_seconds: float,
    query_latencies_seconds: Sequence[float],
    total_retrieval_seconds: float,
    replay_seconds: float,
    peak_rss_mb: float,
) -> dict[str, Any]:
    return {
        "index_build_seconds": index_build_seconds,
        "query_latency_ms": {
            "mean": statistics.fmean(query_latencies_seconds) * 1000.0,
            "p50": _percentile(query_latencies_seconds, 50.0) * 1000.0,
            "p95": _percentile(query_latencies_seconds, 95.0) * 1000.0,
        },
        "total_test_retrieval_seconds": total_retrieval_seconds,
        "deterministic_replay_seconds": replay_seconds,
        "peak_process_rss_mb": peak_rss_mb,
        "resource_note": "ru_maxrss process peak; includes adapter, index, retrieval, and evaluation",
    }


def compare_with_paper(official_metrics: Mapping[str, float]) -> dict[str, Any]:
    deltas = {
        name: float(official_metrics[name]) - paper_value
        for name, paper_value in PAPER_BM25["metrics"].items()
    }
    return {
        "paper": PAPER_BM25,
        "fresh_official_compat_metrics": dict(official_metrics),
        "absolute_delta_fresh_minus_paper": deltas,
        "classification": "NOT_DIRECTLY_COMPARABLE",
        "reason": (
            "The fresh smoke uses project-default lowercase whitespace tokens and "
            "rank-bm25 defaults k1=1.5/b=0.75; the paper uses spaCy filtering/lemmatization, "
            "a different BM25 implementation, and train-tuned k1=1.0/b=0.6."
        ),
    }


def atomic_publish(
    run_dir: Path,
    config_path: Path,
    artifacts: Mapping[str, Any],
) -> Path:
    expected = {
        "retrieval_rankings.json",
        "summary_metrics.json",
        "run_manifest.json",
    }
    if set(artifacts) != expected:
        raise BSARDBM25SmokeError("exact BSARD smoke artifact set is required")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"final run directory already exists: {run_dir}")
    partial = run_dir.with_name(run_dir.name + ".partial")
    if partial.exists():
        raise FileExistsError(f"partial run directory already exists: {partial}")
    partial.mkdir(parents=False, exist_ok=False)
    try:
        for filename, payload in artifacts.items():
            _strict_json_write(partial / filename, payload)
        shutil.copy2(config_path, partial / "eval_config.snapshot.yaml")
        actual = {path.name for path in partial.iterdir()}
        if actual != expected | {"eval_config.snapshot.yaml"}:
            raise BSARDBM25SmokeError("incomplete BSARD smoke artifact set")
        for filename in expected:
            json.loads(
                (partial / filename).read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    BSARDBM25SmokeError(f"non-finite JSON constant: {value}")
                ),
            )
        os.replace(partial, run_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return run_dir


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("protocol", {}).get("id") != "bsard_external_zero_shot_v1":
        raise BSARDBM25SmokeError("unexpected BSARD config protocol")
    if config["protocol"].get("execute_models") is not False:
        raise BSARDBM25SmokeError("BSARD smoke config must keep model execution disabled")
    if config["dataset"].get("include_synthetic_by_default") is not False:
        raise BSARDBM25SmokeError("synthetic questions must remain disabled")
    if tuple(config["evaluation"]["candidate_diagnostic_depths"]) != DIAGNOSTIC_DEPTHS:
        raise BSARDBM25SmokeError("candidate diagnostic depths changed")


def run(
    *,
    raw_root: Path,
    config_path: Path,
    output_root: Path,
) -> Path:
    project_root = Path(__file__).resolve().parents[4]
    git_commit = _git_head(project_root)
    config_path = config_path.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    benchmark_manifest_path = (
        project_root / "backend/data/benchmarks/bsard/benchmark_manifest.json"
    ).resolve(strict=True)

    dataset = load_bsard(raw_root, strict_identity=True)
    if len(dataset.documents) != EXPECTED_DOCUMENT_COUNT:
        raise BSARDBM25SmokeError("canonical document count changed")
    if (
        len(dataset.human_train) != EXPECTED_TRAIN_QUERY_COUNT
        or len(dataset.human_test) != EXPECTED_TEST_QUERY_COUNT
    ):
        raise BSARDBM25SmokeError("canonical human split count changed")
    test_stats = dataset.qrel_statistics["test"]
    if (
        test_stats["raw_qrel_occurrence_count"] != EXPECTED_TEST_RAW_QRELS
        or test_stats["unique_qrel_count"] != EXPECTED_TEST_UNIQUE_QRELS
    ):
        raise BSARDBM25SmokeError("canonical test qrel identity changed")

    documents = map_documents(dataset.documents)
    known_document_ids = [document["standard_law_id"] for document in documents]
    build_start = time.perf_counter()
    index = build_bm25_index(documents)
    index_build_seconds = time.perf_counter() - build_start

    retrieval_start = time.perf_counter()
    records, query_latencies = retrieve_rankings(
        index, documents, dataset.human_test, top_k=TOP_K
    )
    total_retrieval_seconds = time.perf_counter() - retrieval_start
    validate_rankings(records, dataset.human_test, known_document_ids)
    retrieval_fingerprint = ranking_fingerprint(records)

    replay_start = time.perf_counter()
    replay_records, _ = retrieve_rankings(
        index, documents, dataset.human_test, top_k=TOP_K
    )
    replay_seconds = time.perf_counter() - replay_start
    validate_rankings(replay_records, dataset.human_test, known_document_ids)
    replay_fingerprint = ranking_fingerprint(replay_records)
    if replay_fingerprint != retrieval_fingerprint:
        raise BSARDBM25SmokeError("BM25 deterministic replay fingerprint mismatch")

    official, safe = evaluate_records(
        records, dataset.human_test, known_document_ids
    )
    coverage = coverage_diagnostics(records, dataset.human_test)
    latency = latency_summary(
        index_build_seconds=index_build_seconds,
        query_latencies_seconds=query_latencies,
        total_retrieval_seconds=total_retrieval_seconds,
        replay_seconds=replay_seconds,
        peak_rss_mb=_peak_rss_mb(),
    )
    paper_comparison = compare_with_paper(official["metrics"])
    qid274_official = next(
        record for record in official["per_query"] if record["query_id"] == "274"
    )
    qid274_safe = safe["per_query"]["274"]

    summary = {
        "protocol_id": PROTOCOL_ID,
        "scope": "BSARD_TEST_BM25_SMOKE",
        "official_compat": official,
        "safe_canonical": safe,
        "coverage_diagnostics": coverage,
        "latency": latency,
        "determinism": {
            "replay_performed": True,
            "identical": True,
            "first_fingerprint": retrieval_fingerprint,
            "replay_fingerprint": replay_fingerprint,
        },
        "qid_274_semantics": {
            "raw_relevant_count": qid274_official["raw_relevant_count"],
            "unique_relevant_count": qid274_safe["unique_relevant_count"],
            "official_compat": qid274_official["metrics"],
            "safe_canonical": qid274_safe["metrics"],
        },
        "paper_comparison": paper_comparison,
    }
    result_fingerprint = sha256_json(
        {
            "retrieval_fingerprint": retrieval_fingerprint,
            "official_compat": official,
            "safe_canonical": safe,
            "coverage_diagnostics": coverage,
        }
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"
    rankings_artifact = {
        "schema_version": "bsard_bm25_rankings_v1",
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "query_count": len(records),
        "ranking_depth": TOP_K,
        "records": records,
        "retrieval_fingerprint": retrieval_fingerprint,
    }
    manifest = {
        "schema_version": "bsard_bm25_run_manifest_v1",
        "run_id": run_id,
        "protocol_id": PROTOCOL_ID,
        "git_commit": git_commit,
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(project_root)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "config": {
            "path": str(config_path.relative_to(project_root)),
            "sha256": sha256_file(config_path),
        },
        "benchmark_manifest": {
            "path": str(benchmark_manifest_path.relative_to(project_root)),
            "sha256": sha256_file(benchmark_manifest_path),
        },
        "raw_dataset_sha256": dict(dataset.provenance["raw_sha256"]),
        "corpus_fingerprint": dataset.fingerprints["corpus_ordered"],
        "test_split_fingerprint": dataset.fingerprints["test_split"],
        "test_unique_qrel_fingerprint": dataset.fingerprints["test_unique_qrels"],
        "bm25_parameters": BM25_PARAMETERS,
        "tokenization_policy": TOKENIZATION_POLICY,
        "document_count": len(documents),
        "query_count": len(dataset.human_test),
        "ranking_count": len(records),
        "ranking_depth": TOP_K,
        "metric_semantics": [
            official["metric_semantics"], safe["metric_semantics"]
        ],
        "retrieval_fingerprint": retrieval_fingerprint,
        "result_fingerprint": result_fingerprint,
        "deterministic_replay_passed": True,
        "synthetic_questions_used": False,
        "negative_pools_used": False,
        "train_queries_used_for_retrieval_or_tuning": False,
        "models_used": [],
        "python": platform.python_version(),
        "packages": {"rank-bm25": _package_version("rank-bm25")},
    }
    artifacts = {
        "retrieval_rankings.json": rankings_artifact,
        "summary_metrics.json": summary,
        "run_manifest.json": manifest,
    }
    run_dir = output_root.resolve() / run_id
    return atomic_publish(run_dir, config_path, artifacts)


def main(argv: Sequence[str] | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root
        / "backend/scripts/Evaluation/configs/bsard_external_zero_shot_v1.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path, default=project_root / "backend/data/runs"
    )
    args = parser.parse_args(argv)
    raw_root = args.raw_root or (
        Path(os.environ["BSARD_RAW_ROOT"]) if "BSARD_RAW_ROOT" in os.environ else None
    )
    if raw_root is None:
        raise BSARDBM25SmokeError("explicit --raw-root or BSARD_RAW_ROOT is required")
    final_dir = run(
        raw_root=raw_root,
        config_path=args.config,
        output_root=args.output_root,
    )
    print(final_dir)
    return final_dir


if __name__ == "__main__":
    main()
