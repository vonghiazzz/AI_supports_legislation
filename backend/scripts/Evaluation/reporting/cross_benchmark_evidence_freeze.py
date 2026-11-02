"""Freeze cross-benchmark research evidence without executing retrieval or models."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cross_benchmark_evidence_freeze_v1"
TERMINAL_COMMIT = "22ee5a048349c74ebe3eaa2c5025374870b61b4c"
CLAIM_STATUSES = {
    "REPLICATED_PATTERN",
    "BENCHMARK_SPECIFIC",
    "SUPPORTED_SINGLE_BENCHMARK",
    "HYPOTHESIS_FOR_INDEPENDENT_TEST",
    "NOT_SUPPORTED",
}
BENCHMARK_ROLES = {
    "DEV107": "DEVELOPMENT_EXPLORATORY",
    "BSARD": "EXTERNAL_EXPLORATORY_FROZEN",
    "INDEPENDENT_VIETNAMESE": "FUTURE_CONFIRMATORY_TARGET",
}
DEV_RUN_ID = "dev107_retrieval_model_screening_v1_20261031_180636_318893_96458f45"
BM25_RUN_ID = "bsard_bm25_smoke_v1_20261102_024146_040921_1e759195"
BGE_RUN_ID = "bsard_bge_m3_dense_v1_20261102_064758_329094_fccb32cc"
RRF_RUN_ID = "bsard_rrf_hybrid_v1_20261102_082257_638404_1dddd7c5"
RERANK_RUN_ID = "bsard_bge_reranker_v1_20261102_144746_756758_dc945187"
EXPECTED = {
    "DEV107": {
        "run_id": DEV_RUN_ID,
        "git_commit": "5ce2ca633bac9481a4d62c733e13b08ac84b1b56",
        "corpus_fingerprint": "66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58",
        "query_fingerprint": "adf712e5d2636a960bbf454e1d32e6e8d665f3189a1c9776cf3cf84558ceff08",
        "retrieval_candidate_fingerprint": "54dd2d1ec3d365d179e3be2dd841033f0bf83cdddc96062c6a2c79e62f62ac52",
        "reranker_score_fingerprint": "a30ea268cdf9d3c461248296c28021a2c8dcba7b1347e5024c5f8cb3dc1ae711",
        "final_result_fingerprint": "0a60e3b3a38a5ef9527e77d6954edb86ba9fdc45a0c4c228dda5fd51c8032f0e",
    },
    "BM25": {"run_id": BM25_RUN_ID, "retrieval_fingerprint": "306b06754cecf8f564352bb3b9a817e28a2884c9301ecf16b67dea74a820ea5e", "result_fingerprint": "0a7109f7e48587d64690721447de59a93c43970300a70b2a0d9433f46779b150"},
    "BGE": {"run_id": BGE_RUN_ID, "retrieval_fingerprint": "092721506b540caa02a0d20117de8e9a2dac824e6624414837388da99fe29664", "result_fingerprint": "a0d71b9080f86048f6d1e5e4cfbaf19da70507a57638c19a8393cbf0bbbbdc36"},
    "RRF": {"run_id": RRF_RUN_ID, "retrieval_fingerprint": "771a64288847ef5148f4a1fe37fced9c2d7d98410a710e04b8bb1d8eaeac9500", "result_fingerprint": "928d1892b3f0912a3718f8646b709da3057a6734ec536ee486151f52cceb82ad"},
    "RERANKER": {"run_id": RERANK_RUN_ID, "score_vector_fingerprint": "e17c1c0f499c0d4fd67ea9d756da88032126eff2265d28720e76112a2b31701a"},
}
BSARD_CORPUS_FP = "13192650b6a488221b5d98654f6274354910f509ac83b49b9e9bbaff8cccaaf3"
BSARD_TEST_FP = "38865923e39d24b23f140be7a3bd90618aed483c3da7def248186abec7c97c21"
BSARD_QREL_FP = "c1fbd3251a5c1a806bb85d66c8e8a73fd84b864ca6ee5ee5bc072fc3dc2cab0a"


class EvidenceFreezeError(RuntimeError):
    pass


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    if not path.is_file() or path.is_symlink():
        raise EvidenceFreezeError(f"missing regular source artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(EvidenceFreezeError(f"non-finite JSON constant: {value}")))


def strict_write(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def git_head(root: Path) -> str:
    value = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
    if value != TERMINAL_COMMIT:
        raise EvidenceFreezeError(f"freeze must run at terminal commit {TERMINAL_COMMIT}")
    return value


def validate_source_identity(manifest: Mapping[str, Any], expected: Mapping[str, str], label: str) -> None:
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise EvidenceFreezeError(f"{label} {key} mismatch")


def ranking_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json([{"query_id": str(record["query_id"]), "ordered_document_ids": [str(item["standard_law_id"]) for item in record["ranking"]]} for record in records])


def paired_bootstrap(control: Mapping[str, Any], treatment: Mapping[str, Any], metrics=("recall", "weighted_recall", "mrr", "ndcg")) -> dict[str, Any]:
    qids = sorted(control, key=int)
    if set(qids) != set(treatment) or not qids:
        raise EvidenceFreezeError("paired evidence requires identical non-empty qids")
    output = {}
    for metric in metrics:
        deltas = np.asarray([treatment[qid]["10"][metric] - control[qid]["10"][metric] for qid in qids], dtype=float)
        if not np.isfinite(deltas).all():
            raise EvidenceFreezeError("paired evidence contains non-finite values")
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(deltas), size=(20000, len(deltas)))
        means = deltas[indices].mean(axis=1)
        output[metric] = {"mean_delta": float(deltas.mean()), "ci_lower": float(np.percentile(means, 2.5)), "ci_upper": float(np.percentile(means, 97.5)), "samples": 20000, "seed": 42}
    return output


def failure_ratio(absent: int, ranking_failure: int, success: int, total: int) -> dict[str, Any]:
    values = (absent, ranking_failure, success, total)
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in values) or total <= 0:
        raise EvidenceFreezeError("failure components must be non-negative integers with positive total")
    if absent + ranking_failure + success != total:
        raise EvidenceFreezeError("failure components do not sum to total")
    fractions = {"candidate_absence": absent / total, "ranking_failure": ranking_failure / total, "top10_success": success / total}
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceFreezeError("failure fractions do not sum to one")
    return {"total_relevant_references": total, "candidate_absent": absent, "present_below_top10": ranking_failure, "top10_relevant": success, "fractions": fractions, "candidate_absence_dominates_ranking_failure": absent > ranking_failure}


def _load_run(runs: Path, run_id: str, names: Sequence[str]) -> dict[str, Any]:
    directory = runs / run_id
    if not directory.is_dir() or directory.name.endswith(".partial"):
        raise EvidenceFreezeError(f"missing final source run: {run_id}")
    return {name: strict_load(directory / f"{name}.json") for name in names} | {"path": str(directory.resolve())}


def validate_dev107(root: Path, runs: Path) -> dict[str, Any]:
    source = _load_run(runs, DEV_RUN_ID, ("run_manifest", "retrieval_candidates", "reranker_scores", "summary_metrics", "statistical_analysis"))
    manifest, retrieval, scores = source["run_manifest"], source["retrieval_candidates"], source["reranker_scores"]
    validate_source_identity(manifest, EXPECTED["DEV107"], "DEV107")
    if manifest.get("query_count") != 107 or manifest.get("corpus_count") != 1811 or manifest.get("confirmatory") is not False or manifest.get("independent_benchmark_used") is not False:
        raise EvidenceFreezeError("DEV107 benchmark role/count mismatch")
    if retrieval.get("fingerprint") != sha256_json(retrieval.get("records")) or retrieval["fingerprint"] != manifest["retrieval_candidate_fingerprint"]:
        raise EvidenceFreezeError("DEV107 retrieval content mismatch")
    if scores.get("fingerprint") != sha256_json(scores.get("scores")) or scores["fingerprint"] != manifest["reranker_score_fingerprint"] or scores.get("pair_count") != len(scores.get("scores", {})):
        raise EvidenceFreezeError("DEV107 reranker score content mismatch")
    final = sha256_json({"retrieval_fingerprint": retrieval["fingerprint"], "score_fingerprint": scores["fingerprint"], "summary_fingerprint": sha256_json(source["summary_metrics"]), "statistics_fingerprint": sha256_json(source["statistical_analysis"])})
    if final != manifest["final_result_fingerprint"]:
        raise EvidenceFreezeError("DEV107 final result fingerprint mismatch")
    run_dir = Path(source["path"])
    if sha256_file(run_dir / "eval_config.snapshot.yaml") != manifest["config"]["sha256"]:
        raise EvidenceFreezeError("DEV107 config snapshot mismatch")
    for item in manifest["source_files"]:
        path = root / item["path"]
        if path.is_file() and sha256_file(path) != item["sha256"]:
            raise EvidenceFreezeError(f"DEV107 committed source changed: {item['path']}")
    candidate_path = root / manifest["candidate_cache"]["path"]
    if sha256_file(candidate_path) != manifest["candidate_cache"]["sha256"]:
        raise EvidenceFreezeError("DEV107 candidate source bytes changed")
    benchmark_path = root / manifest["benchmark"]["path"]
    if sha256_file(benchmark_path) != manifest["benchmark"]["sha256"]:
        raise EvidenceFreezeError("DEV107 benchmark bytes changed")
    benchmark = strict_load(benchmark_path)
    if len(benchmark) != 107:
        raise EvidenceFreezeError("DEV107 benchmark query count changed")
    source["qrel_fingerprint"] = sha256_json([
        {"query_id": str(record["id"]), "ground_truth": record["ground_truth"]}
        for record in benchmark
    ])
    source["in_corpus_relevant_reference_count"] = 1018
    for row in source["summary_metrics"]["candidate_analysis"].values():
        if row["gt_absent"] + row["gt_present"] != 1018:
            raise EvidenceFreezeError("DEV107 in-corpus relevant-reference count changed")
    return source


def _bm25_result(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    return sha256_json({"retrieval_fingerprint": manifest["retrieval_fingerprint"], "official_compat": summary["official_compat"], "safe_canonical": summary["safe_canonical"], "coverage_diagnostics": summary["coverage_diagnostics"]})


def _bge_result(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    return sha256_json({"retrieval_fingerprint": manifest["retrieval_fingerprint"], "official": summary["official_compat"], "safe": summary["safe_canonical"], "coverage": summary["coverage"], "complementarity": summary["complementarity"], "bootstrap": summary["bm25_control"]["paired_bootstrap"]})


def _rrf_result(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    analysis = summary["fusion_analysis"]
    membership = {
        "source_top500_qrel_partition": analysis["source_complementarity"],
        "source_union_relevant_qrels": analysis["source_union"]["source_union_relevant_qrels"],
        "source_union_micro_coverage": analysis["source_union"]["source_union_micro_coverage"],
        "source_union_zero_relevant_queries": analysis["source_union"]["source_union_zero_relevant_queries"],
        "top500_retention": analysis["complement_retention"],
        "rrf_top500_relevant_qrels": analysis["source_union_loss"]["rrf_top500_relevant_qrels"],
        "relevant_qrels_discarded_by_rrf_top500": analysis["source_union_loss"]["relevant_qrels_discarded_by_rrf_top500"],
        "rrf_top500_zero_relevant_queries": analysis["source_union_loss"]["rrf_top500_zero_relevant_queries"],
        "relevant_source_category_contribution": analysis["source_category_contribution"],
        "bge_top500_qrel_rank_movement": analysis["rank_movement"],
        "bm25_only_qrels_entering_rrf": analysis["bm25_only_entry_depths"],
    }
    return sha256_json({"retrieval_fingerprint": manifest["retrieval_fingerprint"], "official_compat": summary["official_compat"], "safe_canonical": summary["safe_canonical"], "coverage": summary["coverage"], "source_membership": membership, "primary_paired": summary["rrf_vs_bge_m3"], "secondary_paired": summary["rrf_vs_bm25"], "three_way": summary["three_way_safe_canonical"], "depth_comparison": summary["depth_comparison"], "decision": summary["decision"]})


def _validate_bsard_identity(manifest: Mapping[str, Any], label: str) -> None:
    if (manifest.get("corpus_fingerprint"), manifest.get("test_split_fingerprint"), manifest.get("test_unique_qrel_fingerprint")) != (BSARD_CORPUS_FP, BSARD_TEST_FP, BSARD_QREL_FP):
        raise EvidenceFreezeError(f"{label} BSARD identity mismatch")


def validate_bsard(runs: Path) -> dict[str, Any]:
    output = {}
    for label, run_id, result_builder in (("BM25", BM25_RUN_ID, _bm25_result), ("BGE", BGE_RUN_ID, _bge_result), ("RRF", RRF_RUN_ID, _rrf_result)):
        source = _load_run(runs, run_id, ("run_manifest", "summary_metrics", "retrieval_rankings"))
        manifest, summary, rankings = source["run_manifest"], source["summary_metrics"], source["retrieval_rankings"]
        validate_source_identity(manifest, EXPECTED[label], label); _validate_bsard_identity(manifest, label)
        if ranking_fingerprint(rankings["records"]) != manifest["retrieval_fingerprint"] or result_builder(manifest, summary) != manifest["result_fingerprint"]:
            raise EvidenceFreezeError(f"{label} content fingerprint mismatch")
        output[label] = source
    reranker = _load_run(runs, RERANK_RUN_ID, ("run_manifest", "summary_metrics", "failure_decomposition", "candidate_analysis", "system_rankings"))
    manifest = reranker["run_manifest"]; validate_source_identity(manifest, EXPECTED["RERANKER"], "RERANKER"); _validate_bsard_identity(manifest, "RERANKER")
    if manifest.get("replay_ranking_fingerprints") != manifest.get("ranking_fingerprints") or manifest.get("replay_result_fingerprints") != manifest.get("result_fingerprints"):
        raise EvidenceFreezeError("RERANKER deterministic replay mismatch")
    for name, records in reranker["system_rankings"]["systems"].items():
        if ranking_fingerprint(records) != manifest["ranking_fingerprints"][name]:
            raise EvidenceFreezeError(f"RERANKER ranking content mismatch: {name}")
        expected_result = sha256_json({"ranking_fingerprint": manifest["ranking_fingerprints"][name], **reranker["summary_metrics"]["evaluations"][name]})
        if expected_result != manifest["result_fingerprints"][name]:
            raise EvidenceFreezeError(f"RERANKER result content mismatch: {name}")
    cache = Path(manifest["score_cache_path"])
    context = strict_load(cache / "context.json")
    with np.load(cache / "scores.npz", allow_pickle=False) as archive:
        values = np.asarray(archive["scores"]); metadata = json.loads(str(archive["metadata_json"].item()))
    stored = metadata.pop("metadata_fingerprint", None)
    if stored != sha256_json(metadata) or metadata.get("context_fingerprint") != context["context_fingerprint"] or values.shape != (16199,) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise EvidenceFreezeError("RERANKER score cache validation failed")
    vector_fp = sha256_json({"dtype": str(values.dtype), "shape": list(values.shape), "bytes_sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest()})
    if vector_fp != manifest["score_vector_fingerprint"]:
        raise EvidenceFreezeError("RERANKER score vector fingerprint mismatch")
    metadata["metadata_fingerprint"] = stored
    output["RERANKER"] = reranker
    return output


def artifact_inventory(dev: Mapping[str, Any], bsard: Mapping[str, Any]) -> dict[str, Any]:
    dm = dev["run_manifest"]
    rows = [{
        "benchmark": "DEV107", "run_id": dm["run_id"], "run_path": dev["path"], "git_commit": dm["git_commit"],
        "retrieval_fingerprint": dm["retrieval_candidate_fingerprint"], "result_fingerprint": dm["final_result_fingerprint"],
        "config_fingerprint": dm["config"]["fingerprint"], "config_sha256": dm["config"]["sha256"],
        "runner_source_fingerprint": next(item["sha256"] for item in dm["source_files"] if item["path"].endswith("dev107_retrieval_model_screening.py")),
        "corpus_fingerprint": dm["corpus_fingerprint"], "query_fingerprint": dm["query_fingerprint"], "qrel_fingerprint": dev["qrel_fingerprint"],
        "models": dm["models"], "metric_semantics": ["DEV107 closed-world @10/@20 binary and weighted relevance"],
    }]
    for label in ("BM25", "BGE", "RRF", "RERANKER"):
        source = bsard[label]; manifest = source["run_manifest"]
        rows.append({
            "benchmark": f"BSARD_{label}", "run_id": manifest["run_id"], "run_path": source["path"], "git_commit": manifest["git_commit"],
            "retrieval_fingerprint": manifest.get("retrieval_fingerprint"), "result_fingerprint": manifest.get("result_fingerprint") or manifest.get("result_fingerprints"),
            "score_vector_fingerprint": manifest.get("score_vector_fingerprint"),
            "config_fingerprint": manifest.get("config_fingerprint") or manifest.get("config_sha256") or manifest.get("config", {}).get("sha256"),
            "runner_source_fingerprint": manifest.get("runner_source_fingerprint") or manifest.get("runner_sha256") or manifest.get("runner", {}).get("sha256"),
            "corpus_fingerprint": manifest["corpus_fingerprint"], "query_fingerprint": manifest["test_split_fingerprint"], "qrel_fingerprint": manifest["test_unique_qrel_fingerprint"],
            "models": manifest.get("model") or manifest.get("models_used") or manifest.get("models_executed"),
            "metric_semantics": manifest.get("metric_semantics") or source.get("summary_metrics", {}).get("metric_semantics") or ["bsard_official_compat", "bsard_safe_canonical"],
        })
    return {"artifacts": rows, "validation": "ALL_AUTHORITATIVE_FINGERPRINTS_VALID"}


def build_failure_summary(dev: Mapping[str, Any], reranker: Mapping[str, Any]) -> dict[str, Any]:
    dev_n50 = dev["summary_metrics"]["candidate_analysis"]["BGE_M3_Hybrid_RRF@50"]
    dev_current = dev["summary_metrics"]["candidate_analysis"]["Current_Hybrid@30"]
    failures = reranker["failure_decomposition"]
    rows = {
        "DEV107_BGE_RRF_N50": failure_ratio(dev_n50["gt_absent"], dev_n50["ranking_miss_instances_after_rerank"], dev_n50["reranked_gt_hits_at_10"], 1018),
        "DEV107_CURRENT_HYBRID_N30_SECONDARY": failure_ratio(dev_current["gt_absent"], dev_current["ranking_miss_instances_after_rerank"], dev_current["reranked_gt_hits_at_10"], 1018),
    }
    for system, key in (("BSARD_BGE_N50", "bge_rerank_n50"), ("BSARD_RRF_N50", "rrf_rerank_n50")):
        reference = failures[key]["reference_level"]
        rows[system] = failure_ratio(reference["absent_candidate_pool"], reference["present_below_top10"], reference["present_top10"], 1060)
    if not all(row["candidate_absence_dominates_ranking_failure"] for row in rows.values()):
        raise EvidenceFreezeError("candidate absence replication claim is not supported")
    return {"estimand": "within-system unique relevant-reference disposition", "no_cross_benchmark_count_comparison_without_denominators": True, "systems": rows}


def _claim(claim_id: str, text: str, dev_evidence: Any, bsard_evidence: Any, dev_status: str, bsard_status: str, cross_status: str, limitations: Sequence[str], conclusion: str) -> dict[str, Any]:
    statuses = (dev_status, bsard_status, cross_status)
    if any(status not in CLAIM_STATUSES for status in statuses):
        raise EvidenceFreezeError(f"invalid claim status for {claim_id}")
    return {"claim_id": claim_id, "claim_text": text, "DEV107_evidence": dev_evidence, "BSARD_evidence": bsard_evidence, "DEV107_status": dev_status, "BSARD_status": bsard_status, "cross_benchmark_status": cross_status, "limitations": list(limitations), "allowed_conclusion": conclusion}


def build_evidence_matrix(dev: Mapping[str, Any], bsard: Mapping[str, Any], failure: Mapping[str, Any]) -> dict[str, Any]:
    summary = dev["summary_metrics"]; rerank_summary = bsard["RERANKER"]["summary_metrics"]
    current_n30 = paired_bootstrap(summary["retrieval_systems"]["Current_Hybrid"]["per_query"], summary["reranked_systems"]["Current_Hybrid_Reranked_N30"]["per_query"])
    rrf_n50 = paired_bootstrap(summary["retrieval_systems"]["BGE_M3_Hybrid_RRF"]["per_query"], summary["reranked_systems"]["BGE_M3_Hybrid_RRF_Reranked_N50"]["per_query"])
    dev_rrf = paired_bootstrap(summary["retrieval_systems"]["BGE_M3"]["per_query"], summary["retrieval_systems"]["BGE_M3_Hybrid_RRF"]["per_query"])
    dev_bge_bm25 = paired_bootstrap(summary["retrieval_systems"]["BM25"]["per_query"], summary["retrieval_systems"]["BGE_M3"]["per_query"])
    bsard_effects = {key: rerank_summary["paired_bootstrap"][key] for key in ("bge_n20_vs_source", "bge_n50_vs_source", "rrf_n20_vs_source", "rrf_n50_vs_source")}
    complement = bsard["RRF"]["run_manifest"]["source_complementarity_control"]
    rows = [
        _claim("E1", "Zero-shot BGE reranking improves ranking conditional on candidate availability.", {"Current_Hybrid_N30_vs_generator": current_n30, "BGE_RRF_N50_vs_generator": rrf_n50}, bsard_effects, "SUPPORTED_SINGLE_BENCHMARK", "SUPPORTED_SINGLE_BENCHMARK", "REPLICATED_PATTERN", ["Both benchmarks are exploratory.", "Reranking cannot recover absent candidates."], "Reranking improved rank-sensitive metrics in both explored benchmarks."),
        _claim("E2", "Residual relevant-reference loss is dominated by candidate absence.", failure["systems"]["DEV107_BGE_RRF_N50"], {key: failure["systems"][key] for key in ("BSARD_BGE_N50", "BSARD_RRF_N50")}, "SUPPORTED_SINGLE_BENCHMARK", "SUPPORTED_SINGLE_BENCHMARK", "REPLICATED_PATTERN", ["Not confirmed on the independent Vietnamese benchmark."], "In both explored benchmarks, absent candidates exceed ranking failures among available candidates."),
        _claim("E3", "Fixed equal-weight RRF does not reliably outperform the stronger dense retriever.", dev_rrf, bsard["RRF"]["summary_metrics"]["rrf_vs_bge_m3"], "SUPPORTED_SINGLE_BENCHMARK", "SUPPORTED_SINGLE_BENCHMARK", "REPLICATED_PATTERN", ["Does not generalize to learned/adaptive fusion or other k/weights."], "Fixed equal-weight RRF showed no reliable overall advantage over BGE-M3 in either explored setting."),
        _claim("E4", "Lexical and dense retrievers provide complementary candidates.", {"authoritative_overlap_analysis": False}, complement, "NOT_SUPPORTED", "SUPPORTED_SINGLE_BENCHMARK", "SUPPORTED_SINGLE_BENCHMARK", ["Explicit frozen overlap evidence exists only for BSARD."], "BSARD shows complementary BM25-only and BGE-only qrels; cross-benchmark replication is not established."),
        _claim("E5", "BGE-M3 is stronger than BM25.", dev_bge_bm25, bsard["BGE"]["summary_metrics"]["bm25_control"], "BENCHMARK_SPECIFIC", "SUPPORTED_SINGLE_BENCHMARK", "BENCHMARK_SPECIFIC", ["Different languages, corpora, metrics and query construction."], "BGE clearly exceeded BM25 on BSARD; DEV107 evidence was mixed by metric."),
        _claim("E6", "One candidate depth is universally preferred.", {"N50": "exploratory effectiveness leader", "N20": "resource-efficient challenger"}, {"classification": rerank_summary["decisions"]["candidate_depth"], "N50_vs_N20": {key: rerank_summary["paired_bootstrap"][key] for key in ("bge_n50_vs_n20", "rrf_n50_vs_n20")}}, "BENCHMARK_SPECIFIC", "BENCHMARK_SPECIFIC", "BENCHMARK_SPECIFIC", ["BSARD Top10/MAP/MRR/nDCG depth CIs crossed zero.", "N50 costs substantially more."], "No universal N20/N50 choice is supported."),
        _claim("E7", "Adaptive or coverage-aware candidate generation may reduce the replicated candidate-absence bottleneck.", {"motivation": "candidate absence dominates"}, {"motivation": "candidate absence dominates"}, "HYPOTHESIS_FOR_INDEPENDENT_TEST", "HYPOTHESIS_FOR_INDEPENDENT_TEST", "HYPOTHESIS_FOR_INDEPENDENT_TEST", ["No adaptive candidate-generation treatment has been implemented or evaluated."], "Candidate generation/allocation is justified as the next research target, not as an achieved improvement."),
    ]
    return {"schema_version": "cross_benchmark_evidence_matrix_v1", "claim_status_vocabulary": sorted(CLAIM_STATUSES), "raw_metric_pooling": False, "common_estimand_declared": False, "comparison_mode": ["within_benchmark_treatment_direction", "failure_mode_structure", "qualitative_structural_replication"], "claims": rows}


def benchmark_role_freeze() -> dict[str, Any]:
    return {
        "roles": BENCHMARK_ROLES,
        "DEV107": {"status": "DEVELOPMENT_EXPLORATORY", "allowed": ["development", "debugging", "synthetic_or_unit_analysis"], "confirmatory": False},
        "BSARD": {"status": "EXTERNAL_EXPLORATORY_FROZEN", "declaration": "BSARD_EXPLORATORY_TRACK_FROZEN_AFTER_PHASE_1F", "terminal_commit": TERMINAL_COMMIT, "freeze_context": "2026-11-02 after sequential BM25, BGE-M3, RRF, reranker and candidate-depth inspection", "reason": "The test benchmark is now part of the method-development evidence base.", "allowed": ["reproduction_of_frozen_results", "artifact_validation", "non-decision-changing_bug_or_regression_verification", "reporting"], "prohibited_without_new_exploratory_protocol": ["parameter_search", "candidate_depth_selection", "model_selection", "fusion_tuning", "new_method_tuning"]},
        "INDEPENDENT_VIETNAMESE": {"status": "FUTURE_CONFIRMATORY_TARGET", "current_state": "DRAFT", "labels_and_results_must_remain_unavailable_for_tuning": True, "open_only_after_freeze": ["sampling_protocol", "annotation_protocol", "new_method", "baseline_systems", "evaluation_protocol"]},
        "firewall": {"new_method_development_inputs": ["DEV107", "synthetic_or_unit_level_analysis"], "BSARD_may_motivate_but_not_optimize": True, "freeze_before_confirmatory_evaluation": ["implementation_commit", "hyperparameters", "model_revisions", "candidate_budget", "fusion_or_routing_rules", "fallback_behavior", "latency_budget", "random_seeds", "evaluation_metrics"]},
    }


def hypotheses() -> dict[str, Any]:
    return {
        "OBSERVED": ["Zero-shot BGE reranking improved ranking metrics in both DEV107 exploratory and BSARD exploratory settings.", "Candidate absence exceeded ranking failure for the selected representative N50 systems."],
        "REPLICATED_PATTERN": ["Reranking improves ranking conditional on candidate availability.", "Candidate absence dominates residual relevant-reference loss.", "Fixed equal-weight RRF did not reliably outperform the stronger dense retriever."],
        "BENCHMARK_SPECIFIC_RESULT": ["BGE-vs-BM25 behavior differed by benchmark.", "No universal N20/N50 candidate depth was supported."],
        "HYPOTHESIS": ["Query-dependent lexical/dense allocation may reduce candidate misses.", "Adaptive candidate depth may improve coverage per unit cost.", "Coverage-aware union selection, budget-aware generation, or learned routing/fusion may reduce candidate absence."],
        "NOT_YET_TESTED": ["Whether any adaptive/coverage-aware method improves the independent Vietnamese benchmark.", "Whether the proposed directions outperform frozen baselines under a preregistered latency budget."],
        "future_independent_baselines": {"status": "PROPOSED_PRE_REGISTRATION_SET", "systems": ["BM25 historical baseline", "BGE-M3 dense", "fixed RRF k=60", "BGE reranker pipeline", "new candidate-generation method if developed"], "final_set_requires_supervisor_review": True},
        "policy": {"do_not_design_algorithm_in_this_phase": True, "BSARD_test_driven_optimization_prohibited": True},
    }


def render_markdown(roles: Mapping[str, Any], matrix: Mapping[str, Any], failure: Mapping[str, Any], hypotheses_payload: Mapping[str, Any]) -> str:
    lines = ["# Cross-benchmark evidence freeze", "", f"Terminal commit: `{TERMINAL_COMMIT}`", "", "## Benchmark roles", ""]
    for name, role in roles["roles"].items(): lines.append(f"- **{name}**: `{role}`")
    lines += ["", "## Evidence matrix", "", "| ID | Claim | Status | Allowed conclusion |", "|---|---|---|---|"]
    for row in matrix["claims"]: lines.append(f"| {row['claim_id']} | {row['claim_text']} | `{row['cross_benchmark_status']}` | {row['allowed_conclusion']} |")
    lines += ["", "## Representative failure ratios", ""]
    for name, row in failure["systems"].items():
        values = row["fractions"]; lines.append(f"- **{name}**: absent {values['candidate_absence']:.6f}; ranking failure {values['ranking_failure']:.6f}; Top-10 success {values['top10_success']:.6f}.")
    for heading in ("OBSERVED", "REPLICATED_PATTERN", "BENCHMARK_SPECIFIC_RESULT", "HYPOTHESIS", "NOT_YET_TESTED"):
        lines += ["", f"## {heading.replace('_', ' ')}", ""] + [f"- {value}" for value in hypotheses_payload[heading]]
    lines += ["", "## BSARD freeze declaration", "", "`BSARD_EXPLORATORY_TRACK_FROZEN_AFTER_PHASE_1F`", ""]
    return "\n".join(lines)


def publish(output_root: Path, structured: Mapping[str, Any], markdown: str, manifest_base: Mapping[str, Any]) -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f"); run_id = f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"; final = output_root / run_id; partial = output_root / f"{run_id}.partial"
    if final.exists() or partial.exists(): raise EvidenceFreezeError("unsafe freeze publication path")
    partial.mkdir(parents=True)
    try:
        for filename, payload in structured.items(): strict_write(partial / filename, payload)
        (partial / "evidence_summary.md").write_text(markdown, encoding="utf-8")
        file_fingerprints = {name: sha256_file(partial / name) for name in sorted((*structured, "evidence_summary.md"))}
        manifest = {**manifest_base, "run_id": run_id, "protocol_id": PROTOCOL_ID, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "output_file_fingerprints": file_fingerprints, "freeze_artifact_fingerprint": sha256_json(file_fingerprints), "models_executed": [], "retrievers_executed": [], "raw_metric_pooling": False}
        strict_write(partial / "run_manifest.json", manifest)
        if {path.name for path in partial.iterdir()} != set(structured) | {"evidence_summary.md", "run_manifest.json"}: raise EvidenceFreezeError("incomplete freeze artifact")
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True); raise
    return final


def run(*, output_root: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[4]; commit = git_head(root); runs = root / "backend/data/runs"; output_root = output_root or runs
    dev = validate_dev107(root, runs); bsard = validate_bsard(runs); inventory = artifact_inventory(dev, bsard); failure = build_failure_summary(dev, bsard["RERANKER"]); matrix = build_evidence_matrix(dev, bsard, failure); roles = benchmark_role_freeze(); hypotheses_payload = hypotheses()
    independent = strict_load(root / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json")
    sampling = strict_load(root / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json")
    if independent.get("state") != "DRAFT" or sampling.get("state") != "DRAFT" or sampling.get("selection_status") != "NOT_STARTED" or independent["current_counts"] != {"available_sources": 330, "development_sources_consumed": 107, "eligible_sources": 223, "selected_cases": 0, "query_records": 0, "ground_truth_records": 0}:
        raise EvidenceFreezeError("independent confirmatory target state changed")
    structured = {"evidence_matrix.json": matrix, "failure_pattern_summary.json": failure, "benchmark_role_freeze.json": roles, "research_hypotheses.json": hypotheses_payload, "source_artifact_inventory.json": inventory}
    evidence_fp = sha256_json(matrix); report_fp = sha256_json(structured)
    manifest_base = {"git_commit": commit, "runner_source_fingerprint": sha256_file(Path(__file__)), "source_artifacts": {row["benchmark"]: {key: row.get(key) for key in ("run_id", "retrieval_fingerprint", "result_fingerprint", "score_vector_fingerprint", "corpus_fingerprint", "query_fingerprint", "qrel_fingerprint")} for row in inventory["artifacts"]}, "evidence_matrix_fingerprint": evidence_fp, "structured_report_fingerprint": report_fp, "benchmark_roles": BENCHMARK_ROLES, "metric_policy": "NO_CROSS_BENCHMARK_RAW_METRIC_POOLING", "terminal_bsard_commit": TERMINAL_COMMIT}
    return publish(output_root, structured, render_markdown(roles, matrix, failure, hypotheses_payload), manifest_base)


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv); result = run(output_root=args.output_root.resolve() if args.output_root else None); print(result); return result


if __name__ == "__main__":
    main()
