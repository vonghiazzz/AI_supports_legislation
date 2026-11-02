"""BSARD Phase 1E: fixed equal-weight RRF over frozen BM25/BGE rankings."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from ..benchmarking.bsard_adapter import load_bsard
from ..evaluation.bsard_metrics import paired_bootstrap_mean_delta
from . import bsard_bm25_smoke as bm25_runner
from .bsard_bm25_smoke import (
    DIAGNOSTIC_DEPTHS,
    coverage_diagnostics,
    evaluate_records,
    ranking_fingerprint,
    sha256_file,
    sha256_json,
    validate_rankings,
)

PROTOCOL_ID = "bsard_rrf_hybrid_v1"
RRF_K = 60
SOURCE_DEPTH = 500
OUTPUT_DEPTH = 500
SOURCE_WEIGHTS = {"bm25": 1.0, "bge_m3": 1.0}
TIE_POLICY = "rrf_score_desc_then_numeric_article_id_asc"
BM25_RUN_ID = "bsard_bm25_smoke_v1_20261102_024146_040921_1e759195"
BGE_RUN_ID = "bsard_bge_m3_dense_v1_20261102_064758_329094_fccb32cc"
BM25_RETRIEVAL_FP = "306b06754cecf8f564352bb3b9a817e28a2884c9301ecf16b67dea74a820ea5e"
BM25_RESULT_FP = "0a7109f7e48587d64690721447de59a93c43970300a70b2a0d9433f46779b150"
BGE_RETRIEVAL_FP = "092721506b540caa02a0d20117de8e9a2dac824e6624414837388da99fe29664"
BGE_RESULT_FP = "a0d71b9080f86048f6d1e5e4cfbaf19da70507a57638c19a8393cbf0bbbbdc36"
CORPUS_FP = "13192650b6a488221b5d98654f6274354910f509ac83b49b9e9bbaff8cccaaf3"
TEST_FP = "38865923e39d24b23f140be7a3bd90618aed483c3da7def248186abec7c97c21"
QREL_FP = "c1fbd3251a5c1a806bb85d66c8e8a73fd84b864ca6ee5ee5bc072fc3dc2cab0a"
EXPECTED_SOURCE_UNION = {"both": 346, "bm25_only": 44, "bge_only": 324, "missed_both": 346}
EXPECTED_UNIQUE_QRELS = 1060
EXPECTED_SOURCE_UNION_ZERO = 10
PAIRED_METRICS = ("recall@100", "recall@200", "recall@500", "map@100", "mrr@100", "ndcg@10")


class BSARDRRFError(RuntimeError):
    pass


def strict_load(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            BSARDRRFError(f"non-finite JSON constant: {value}")
        ),
    )


def strict_write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def git_head(root: Path) -> str:
    value = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if len(value) != 40:
        raise BSARDRRFError("valid git commit required")
    return value


def rrf_fuse(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    known_ids: Iterable[str],
    *,
    expected_depth: int = SOURCE_DEPTH,
    output_depth: int = OUTPUT_DEPTH,
    rrf_k: int = RRF_K,
) -> list[dict[str, Any]]:
    if rrf_k != 60:
        raise BSARDRRFError("RRF k must remain 60")
    known = set(map(str, known_ids))
    rank_maps = []
    for source in (left, right):
        ids = [str(item.get("standard_law_id", "")) for item in source]
        if len(ids) != expected_depth:
            raise BSARDRRFError("source ranking depth mismatch")
        if len(ids) != len(set(ids)):
            raise BSARDRRFError("duplicate source prediction")
        if set(ids) - known:
            raise BSARDRRFError("unknown source prediction")
        rank_maps.append({document_id: rank for rank, document_id in enumerate(ids, 1)})
    scored = []
    for document_id in set(rank_maps[0]) | set(rank_maps[1]):
        score = sum(
            1.0 / (rrf_k + ranks[document_id])
            for ranks in rank_maps
            if document_id in ranks
        )
        if not math.isfinite(score):
            raise BSARDRRFError("non-finite RRF score")
        scored.append((document_id, score))
    scored.sort(key=lambda item: (-item[1], int(item[0])))
    return [
        {"rank": rank, "standard_law_id": document_id, "score": float(score)}
        for rank, (document_id, score) in enumerate(scored[:output_depth], 1)
    ]


def validate_fused(records, queries, known_ids) -> None:
    validate_rankings(records, queries, known_ids, maximum_depth=OUTPUT_DEPTH)
    for record in records:
        ranking = record["ranking"]
        if len(ranking) != OUTPUT_DEPTH:
            raise BSARDRRFError("fused ranking must contain exactly 500 documents")
        for left, right in zip(ranking, ranking[1:]):
            valid = left["score"] > right["score"] or (
                left["score"] == right["score"]
                and int(left["standard_law_id"]) < int(right["standard_law_id"])
            )
            if not valid:
                raise BSARDRRFError("RRF deterministic order violation")


def fuse_records(bm25_records, bge_records, queries, known_ids):
    bm25 = {str(record["query_id"]): record["ranking"] for record in bm25_records}
    bge = {str(record["query_id"]): record["ranking"] for record in bge_records}
    expected = [str(query["query_id"]) for query in queries]
    if set(bm25) != set(expected) or set(bge) != set(expected):
        raise BSARDRRFError("missing or extra source qid")
    records, latencies = [], []
    for query_id in expected:
        started = time.perf_counter()
        ranking = rrf_fuse(bm25[query_id], bge[query_id], known_ids)
        latencies.append(time.perf_counter() - started)
        records.append({"query_id": query_id, "ranking": ranking})
    validate_fused(records, queries, known_ids)
    ordered = sorted(latencies)
    percentile = lambda p: ordered[min(len(ordered) - 1, int(math.ceil(p / 100 * len(ordered))) - 1)]
    return records, {
        "total_seconds": sum(latencies),
        "mean_ms": sum(latencies) / len(latencies) * 1000,
        "p50_ms": percentile(50) * 1000,
        "p95_ms": percentile(95) * 1000,
    }


def _result_fingerprint(manifest, summary, *, kind: str) -> str:
    if kind == "bm25":
        return sha256_json(
            {
                "retrieval_fingerprint": manifest["retrieval_fingerprint"],
                "official_compat": summary["official_compat"],
                "safe_canonical": summary["safe_canonical"],
                "coverage_diagnostics": summary["coverage_diagnostics"],
            }
        )
    return sha256_json(
        {
            "retrieval_fingerprint": manifest["retrieval_fingerprint"],
            "official": summary["official_compat"],
            "safe": summary["safe_canonical"],
            "coverage": summary["coverage"],
            "complementarity": summary["complementarity"],
            "bootstrap": summary["bm25_control"]["paired_bootstrap"],
        }
    )


def load_source(path: Path) -> dict[str, Any]:
    return {
        "manifest": strict_load(path / "run_manifest.json"),
        "summary": strict_load(path / "summary_metrics.json"),
        "rankings": strict_load(path / "retrieval_rankings.json"),
    }


def validate_source(source, *, kind, expected_run, expected_retrieval, expected_result, dataset):
    manifest, summary, rankings = source["manifest"], source["summary"], source["rankings"]
    if manifest.get("run_id") != expected_run:
        raise BSARDRRFError(f"{kind} source run mismatch")
    if manifest.get("retrieval_fingerprint") != expected_retrieval:
        raise BSARDRRFError(f"{kind} source retrieval fingerprint mismatch")
    if manifest.get("result_fingerprint") != expected_result:
        raise BSARDRRFError(f"{kind} source result fingerprint mismatch")
    if ranking_fingerprint(rankings.get("records", [])) != expected_retrieval:
        raise BSARDRRFError(f"{kind} ranking content mismatch")
    if _result_fingerprint(manifest, summary, kind=kind) != expected_result:
        raise BSARDRRFError(f"{kind} result content mismatch")
    expected_identity = {
        "corpus_fingerprint": CORPUS_FP,
        "test_split_fingerprint": TEST_FP,
        "test_unique_qrel_fingerprint": QREL_FP,
    }
    if any(manifest.get(key) != value for key, value in expected_identity.items()):
        raise BSARDRRFError(f"{kind} benchmark identity mismatch")
    if manifest.get("raw_dataset_sha256") != dataset.provenance["raw_sha256"]:
        raise BSARDRRFError(f"{kind} raw provenance mismatch")
    validate_rankings(
        rankings["records"], dataset.human_test, [doc["document_id"] for doc in dataset.documents]
    )
    if len(rankings["records"]) != 222 or any(
        len(record["ranking"]) != SOURCE_DEPTH for record in rankings["records"]
    ):
        raise BSARDRRFError(f"{kind} frozen ranking shape mismatch")


def source_provenance(root: Path, bm25_source, bge_source) -> dict[str, Any]:
    bm25_path = root / "backend/scripts/Evaluation/experiments/bsard_bm25_smoke.py"
    bge_path = root / "backend/scripts/Evaluation/experiments/bsard_bge_m3_dense.py"
    bm_manifest, bge_manifest = bm25_source["manifest"], bge_source["manifest"]
    semantic_match = (
        bm_manifest["bm25_parameters"] == bm25_runner.BM25_PARAMETERS
        and bm_manifest["tokenization_policy"] == bm25_runner.TOKENIZATION_POLICY
        and bm25_runner.TOP_K == SOURCE_DEPTH
    )
    if not semantic_match:
        raise BSARDRRFError("committed BM25 semantic contract differs from source artifact")
    current_bge = sha256_file(bge_path)
    if current_bge != bge_manifest["runner_sha256"]:
        raise BSARDRRFError("committed BGE analysis runner fingerprint mismatch")
    return {
        "bm25": {
            "recorded_generation_runner_sha256": bm_manifest["runner"]["sha256"],
            "current_committed_runner_sha256": sha256_file(bm25_path),
            "byte_identical": bm_manifest["runner"]["sha256"] == sha256_file(bm25_path),
            "semantic_contract_validated": semantic_match,
        },
        "bge_m3": {
            "recorded_analysis_runner_sha256": bge_manifest["runner_sha256"],
            "current_committed_runner_sha256": current_bge,
            "byte_identical": True,
            "recorded_retrieval_runner_sha256": bge_manifest["source_retrieval_runner_sha256"],
            "source_retrieval_run_id": bge_manifest["source_retrieval_run_id"],
        },
    }


def source_membership_analysis(bm25_records, bge_records, fused_records, queries):
    bm = {r["query_id"]: [x["standard_law_id"] for x in r["ranking"]] for r in bm25_records}
    bg = {r["query_id"]: [x["standard_law_id"] for x in r["ranking"]] for r in bge_records}
    rr = {r["query_id"]: [x["standard_law_id"] for x in r["ranking"]] for r in fused_records}
    categories = {"shared": set(), "bm25_only": set(), "bge_only": set(), "missed_both": set()}
    union_zero = 0
    for query in queries:
        qid = str(query["query_id"]); relevant = set(query["relevant_document_ids"])
        bms, bgs = set(bm[qid]), set(bg[qid])
        if not relevant & (bms | bgs): union_zero += 1
        for document_id in relevant:
            pair = (qid, document_id)
            if document_id in bms and document_id in bgs: categories["shared"].add(pair)
            elif document_id in bms: categories["bm25_only"].add(pair)
            elif document_id in bgs: categories["bge_only"].add(pair)
            else: categories["missed_both"].add(pair)
    counts = {
        "both": len(categories["shared"]),
        "bm25_only": len(categories["bm25_only"]),
        "bge_only": len(categories["bge_only"]),
        "missed_both": len(categories["missed_both"]),
    }
    if counts != EXPECTED_SOURCE_UNION or sum(counts.values()) != EXPECTED_UNIQUE_QRELS or union_zero != EXPECTED_SOURCE_UNION_ZERO:
        raise BSARDRRFError("frozen source complementarity gate failed")
    retained = {}
    fused_500 = {(qid, document_id) for qid, ids in rr.items() for document_id in ids[:500]}
    for category in ("shared", "bm25_only", "bge_only"):
        count = len(categories[category] & fused_500); denominator = len(categories[category])
        retained[category] = {"count": count, "denominator": denominator, "rate": count / denominator if denominator else 0.0}
    contributions = {}
    for depth in (100, 500):
        present = {(qid, document_id) for qid, ids in rr.items() for document_id in ids[:depth]}
        contributions[str(depth)] = {
            category: len(categories[category] & present)
            for category in ("shared", "bm25_only", "bge_only")
        }
    bge_changes, improved, worsened, tied, dropped = [], 0, 0, 0, 0
    for qid, document_id in categories["shared"] | categories["bge_only"]:
        bge_rank = bg[qid].index(document_id) + 1
        if document_id not in rr[qid]: dropped += 1; continue
        rrf_rank = rr[qid].index(document_id) + 1
        change = bge_rank - rrf_rank; bge_changes.append(change)
        improved += change > 0; worsened += change < 0; tied += change == 0
    bm_entries = {
        str(depth): len(categories["bm25_only"] & {(qid, doc) for qid, ids in rr.items() for doc in ids[:depth]})
        for depth in (20, 50, 100, 200, 500)
    }
    fused_zero = sum(
        not (set(query["relevant_document_ids"]) & set(rr[str(query["query_id"])][:500]))
        for query in queries
    )
    fused_relevant = sum(retained[key]["count"] for key in retained)
    return {
        "source_top500_qrel_partition": counts,
        "source_union_relevant_qrels": 714,
        "source_union_micro_coverage": 714 / EXPECTED_UNIQUE_QRELS,
        "source_union_zero_relevant_queries": union_zero,
        "top500_retention": retained,
        "rrf_top500_relevant_qrels": fused_relevant,
        "relevant_qrels_discarded_by_rrf_top500": 714 - fused_relevant,
        "rrf_top500_zero_relevant_queries": fused_zero,
        "relevant_source_category_contribution": contributions,
        "bge_top500_qrel_rank_movement": {
            "median_bge_rank_minus_rrf_rank_retained": float(median(bge_changes)),
            "improved": improved, "worsened": worsened, "tied": tied,
            "dropped_from_rrf_top500": dropped,
        },
        "bm25_only_qrels_entering_rrf": bm_entries,
    }


def source_complementarity_control(bm25_records, bge_records, queries):
    bm = {r["query_id"]: {x["standard_law_id"] for x in r["ranking"]} for r in bm25_records}
    bg = {r["query_id"]: {x["standard_law_id"] for x in r["ranking"]} for r in bge_records}
    counts = {"both": 0, "bm25_only": 0, "bge_only": 0, "missed_both": 0}; zero = 0
    for query in queries:
        qid = str(query["query_id"]); relevant = set(query["relevant_document_ids"])
        if not relevant & (bm[qid] | bg[qid]): zero += 1
        for document_id in relevant:
            if document_id in bm[qid] and document_id in bg[qid]: counts["both"] += 1
            elif document_id in bm[qid]: counts["bm25_only"] += 1
            elif document_id in bg[qid]: counts["bge_only"] += 1
            else: counts["missed_both"] += 1
    if counts != EXPECTED_SOURCE_UNION or sum(counts.values()) != EXPECTED_UNIQUE_QRELS or zero != EXPECTED_SOURCE_UNION_ZERO:
        raise BSARDRRFError("frozen source complementarity gate failed")
    return {"qrel_partition": counts, "relevant_qrels": sum(counts[key] for key in ("both", "bm25_only", "bge_only")), "zero_relevant_queries": zero}


def paired(control_safe, treatment_safe):
    output = {}
    qids = sorted(treatment_safe["per_query"], key=int)
    for metric in PAIRED_METRICS:
        control = [control_safe["per_query"][qid]["metrics"][metric] for qid in qids]
        treatment = [treatment_safe["per_query"][qid]["metrics"][metric] for qid in qids]
        output[metric] = paired_bootstrap_mean_delta(control, treatment, samples=20000, seed=42)
    return output


def classify(primary):
    deltas = {key: value["mean_delta"] for key, value in primary.items()}
    quality = (deltas["map@100"], deltas["mrr@100"], deltas["ndcg@10"])
    if deltas["recall@500"] > 0 and any(value < 0 for value in quality):
        return "RRF_IMPROVES_DEEP_COVERAGE_BUT_NOT_RANKING"
    if all(value >= 0 for value in deltas.values()) and any(value["ci_lower"] > 0 for value in primary.values()):
        return "RRF_IMPROVES_BGE"
    if sum(value < 0 for value in deltas.values()) >= 4:
        return "RRF_HURTS_BGE"
    if max(map(abs, deltas.values())) < 0.01 and all(
        row["ci_lower"] <= 0 <= row["ci_upper"] for row in primary.values()
    ):
        return "RRF_LARGELY_REDUNDANT"
    return "RRF_MIXED_OR_INCONCLUSIVE"


def publish(final: Path, config: Path, artifacts: Mapping[str, Any]) -> Path:
    expected = {"retrieval_rankings.json", "summary_metrics.json", "fusion_analysis.json", "run_manifest.json"}
    partial = final.with_name(final.name + ".partial")
    if set(artifacts) != expected or final.exists() or partial.exists():
        raise BSARDRRFError("unsafe artifact publication")
    partial.mkdir(parents=True)
    try:
        for name, payload in artifacts.items(): strict_write(partial / name, payload)
        shutil.copy2(config, partial / "eval_config.snapshot.yaml")
        if {p.name for p in partial.iterdir()} != expected | {"eval_config.snapshot.yaml"}:
            raise BSARDRRFError("incomplete artifact publication")
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True); raise
    return final


def run(*, raw_root: Path, config_path: Path, output_root: Path) -> Path:
    started = time.perf_counter(); root = Path(__file__).resolve().parents[4]; commit = git_head(root)
    if len(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()) != 0:
        raise BSARDRRFError("tracked worktree must be clean")
    identity_started = time.perf_counter(); dataset = load_bsard(raw_root, strict_identity=True)
    if len(dataset.documents) != 22633 or len(dataset.human_test) != 222 or sum(len(q["relevant_document_ids"]) for q in dataset.human_test) != 1060:
        raise BSARDRRFError("canonical BSARD identity changed")
    if dataset.fingerprints != {**dataset.fingerprints, "corpus_ordered": CORPUS_FP, "test_split": TEST_FP, "test_unique_qrels": QREL_FP}:
        raise BSARDRRFError("canonical BSARD fingerprint changed")
    identity_seconds = time.perf_counter() - identity_started
    load_started = time.perf_counter(); runs = root / "backend/data/runs"
    bm25_source = load_source(runs / BM25_RUN_ID); bge_source = load_source(runs / BGE_RUN_ID)
    load_seconds = time.perf_counter() - load_started
    validation_started = time.perf_counter()
    validate_source(bm25_source, kind="bm25", expected_run=BM25_RUN_ID, expected_retrieval=BM25_RETRIEVAL_FP, expected_result=BM25_RESULT_FP, dataset=dataset)
    validate_source(bge_source, kind="bge", expected_run=BGE_RUN_ID, expected_retrieval=BGE_RETRIEVAL_FP, expected_result=BGE_RESULT_FP, dataset=dataset)
    provenance = source_provenance(root, bm25_source, bge_source)
    source_control = source_complementarity_control(bm25_source["rankings"]["records"], bge_source["rankings"]["records"], dataset.human_test)
    validation_seconds = time.perf_counter() - validation_started
    known = [doc["document_id"] for doc in dataset.documents]
    fusion_started = time.perf_counter(); records, fusion_latency = fuse_records(bm25_source["rankings"]["records"], bge_source["rankings"]["records"], dataset.human_test, known); fusion_seconds = time.perf_counter() - fusion_started
    replay, _ = fuse_records(bm25_source["rankings"]["records"], bge_source["rankings"]["records"], dataset.human_test, known)
    retrieval_fp = ranking_fingerprint(records)
    if ranking_fingerprint(replay) != retrieval_fp or replay != records:
        raise BSARDRRFError("RRF deterministic replay failed")
    analysis_started = time.perf_counter(); membership = source_membership_analysis(bm25_source["rankings"]["records"], bge_source["rankings"]["records"], records, dataset.human_test)
    if membership["source_top500_qrel_partition"] != source_control["qrel_partition"]: raise BSARDRRFError("source complementarity replay mismatch")
    official, safe = evaluate_records(records, dataset.human_test, known); coverage = coverage_diagnostics(records, dataset.human_test)
    primary = paired(bge_source["summary"]["safe_canonical"], safe); secondary = paired(bm25_source["summary"]["safe_canonical"], safe)
    three_way = {metric: {"bm25": bm25_source["summary"]["safe_canonical"]["metrics"][metric], "bge_m3": bge_source["summary"]["safe_canonical"]["metrics"][metric], "rrf": safe["metrics"][metric]} for metric in ("recall@20", "recall@30", "recall@50", "recall@100", "recall@200", "recall@500", "map@100", "mrr@100", "ndcg@10")}
    depth_comparison = {str(depth): {"qrel_coverage": {"bm25": bm25_source["summary"]["coverage_diagnostics"][str(depth)]["retrieved_relevant_references"], "bge_m3": bge_source["summary"]["coverage"][str(depth)]["retrieved_relevant_references"], "rrf": coverage[str(depth)]["retrieved_relevant_references"]}, "zero_relevant_queries": {"bm25": bm25_source["summary"]["coverage_diagnostics"][str(depth)]["zero_relevant_query_count"], "bge_m3": bge_source["summary"]["coverage"][str(depth)]["zero_relevant_query_count"], "rrf": coverage[str(depth)]["zero_relevant_query_count"]}} for depth in DIAGNOSTIC_DEPTHS}
    decision = classify(primary); analysis_seconds = time.perf_counter() - analysis_started
    result_payload = {"retrieval_fingerprint": retrieval_fp, "official_compat": official, "safe_canonical": safe, "coverage": coverage, "source_membership": membership, "primary_paired": primary, "secondary_paired": secondary, "three_way": three_way, "depth_comparison": depth_comparison, "decision": decision}; result_fp = sha256_json(result_payload)
    replay_started = time.perf_counter(); replay_membership = source_membership_analysis(bm25_source["rankings"]["records"], bge_source["rankings"]["records"], replay, dataset.human_test); replay_official, replay_safe = evaluate_records(replay, dataset.human_test, known); replay_coverage = coverage_diagnostics(replay, dataset.human_test); replay_primary = paired(bge_source["summary"]["safe_canonical"], replay_safe); replay_secondary = paired(bm25_source["summary"]["safe_canonical"], replay_safe)
    replay_three_way = {metric: {"bm25": bm25_source["summary"]["safe_canonical"]["metrics"][metric], "bge_m3": bge_source["summary"]["safe_canonical"]["metrics"][metric], "rrf": replay_safe["metrics"][metric]} for metric in three_way}
    replay_depth = {str(depth): {"qrel_coverage": {"bm25": bm25_source["summary"]["coverage_diagnostics"][str(depth)]["retrieved_relevant_references"], "bge_m3": bge_source["summary"]["coverage"][str(depth)]["retrieved_relevant_references"], "rrf": replay_coverage[str(depth)]["retrieved_relevant_references"]}, "zero_relevant_queries": {"bm25": bm25_source["summary"]["coverage_diagnostics"][str(depth)]["zero_relevant_query_count"], "bge_m3": bge_source["summary"]["coverage"][str(depth)]["zero_relevant_query_count"], "rrf": replay_coverage[str(depth)]["zero_relevant_query_count"]}} for depth in DIAGNOSTIC_DEPTHS}
    replay_payload = {"retrieval_fingerprint": ranking_fingerprint(replay), "official_compat": replay_official, "safe_canonical": replay_safe, "coverage": replay_coverage, "source_membership": replay_membership, "primary_paired": replay_primary, "secondary_paired": replay_secondary, "three_way": replay_three_way, "depth_comparison": replay_depth, "decision": classify(replay_primary)}
    replay_result_fp = sha256_json(replay_payload); replay_seconds = time.perf_counter() - replay_started
    if replay_result_fp != result_fp: raise BSARDRRFError("RRF result fingerprint replay failed")
    run_id = f"{PROTOCOL_ID}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    rankings = {"run_id": run_id, "query_count": 222, "ranking_depth": 500, "records": records, "retrieval_fingerprint": retrieval_fp}
    analysis = {"source_complementarity": membership["source_top500_qrel_partition"], "source_union": {key: membership[key] for key in ("source_union_relevant_qrels", "source_union_micro_coverage", "source_union_zero_relevant_queries")}, "complement_retention": membership["top500_retention"], "source_union_loss": {key: membership[key] for key in ("rrf_top500_relevant_qrels", "relevant_qrels_discarded_by_rrf_top500", "rrf_top500_zero_relevant_queries")}, "source_category_contribution": membership["relevant_source_category_contribution"], "rank_movement": membership["bge_top500_qrel_rank_movement"], "bm25_only_entry_depths": membership["bm25_only_qrels_entering_rrf"]}
    timing = {"canonical_identity_seconds": identity_seconds, "artifact_loading_seconds": load_seconds, "source_validation_seconds": validation_seconds, "fusion_computation_seconds": fusion_seconds, "metric_analysis_seconds": analysis_seconds, "determinism_result_replay_seconds": replay_seconds, "per_query_fusion": fusion_latency, "total_incremental_seconds": time.perf_counter() - started}
    summary = {"official_compat": official, "safe_canonical": safe, "coverage": coverage, "three_way_safe_canonical": three_way, "depth_comparison": depth_comparison, "rrf_vs_bge_m3": primary, "rrf_vs_bm25": secondary, "fusion_analysis": analysis, "decision": decision, "timing": timing}
    manifest = {"run_id": run_id, "protocol_id": PROTOCOL_ID, "git_commit": commit, "runner_source_fingerprint": sha256_file(Path(__file__)), "benchmark_manifest_fingerprint": sha256_file(root / "backend/data/benchmarks/bsard/benchmark_manifest.json"), "raw_dataset_sha256": dataset.provenance["raw_sha256"], "corpus_fingerprint": CORPUS_FP, "test_split_fingerprint": TEST_FP, "test_unique_qrel_fingerprint": QREL_FP, "source_runs": {"bm25": {"run_id": BM25_RUN_ID, "retrieval_fingerprint": BM25_RETRIEVAL_FP, "result_fingerprint": BM25_RESULT_FP}, "bge_m3": {"run_id": BGE_RUN_ID, "retrieval_fingerprint": BGE_RETRIEVAL_FP, "result_fingerprint": BGE_RESULT_FP}}, "source_runner_provenance": provenance, "source_complementarity_control": source_control, "rrf": {"k": RRF_K, "weights": SOURCE_WEIGHTS, "source_depth": SOURCE_DEPTH, "output_depth": OUTPUT_DEPTH, "tie_policy": TIE_POLICY, "score_normalization": None}, "metric_semantics": ["bsard_official_compat", "bsard_safe_canonical"], "retrieval_fingerprint": retrieval_fp, "result_fingerprint": result_fp, "replay_retrieval_fingerprint": ranking_fingerprint(replay), "replay_result_fingerprint": replay_result_fp, "deterministic_replay_passed": True, "models_executed": [], "reranker_used": False}
    artifacts = {"retrieval_rankings.json": rankings, "summary_metrics.json": summary, "fusion_analysis.json": analysis, "run_manifest.json": manifest}
    return publish(output_root / run_id, config_path, artifacts)


def main(argv=None):
    root = Path(__file__).resolve().parents[4]; parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "backend/scripts/Evaluation/configs/bsard_external_zero_shot_v1.yaml")
    parser.add_argument("--output-root", type=Path, default=root / "backend/data/runs")
    args = parser.parse_args(argv); result = run(raw_root=args.raw_root, config_path=args.config.resolve(), output_root=args.output_root.resolve()); print(result); return result


if __name__ == "__main__":
    main()
