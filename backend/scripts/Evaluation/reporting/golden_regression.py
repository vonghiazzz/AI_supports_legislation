import json
import math
from pathlib import Path


METHODS = ("BM25", "FAISS", "Hybrid_Static_Fixed")
WORLDS = ("closed_world", "open_world")
REPORTED_K = ("5", "10", "20")
METRIC_NAMES = ("recall", "ndcg", "mrr", "weighted_recall")


def _load_run(run_dir):
    run_dir = Path(run_dir)
    return {
        "results": json.loads((run_dir / "pipeline_results.json").read_text(encoding="utf-8")),
        "summary": json.loads((run_dir / "summary_metrics.json").read_text(encoding="utf-8")),
        "manifest": json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8")),
    }


def _metrics_equal(golden_summary, candidate_summary, method, tolerance):
    differences = []
    for world in WORLDS:
        for k in REPORTED_K:
            golden_metrics = golden_summary[world][method][k]
            candidate_metrics = candidate_summary[world][method][k]
            for metric_name in METRIC_NAMES:
                if not math.isclose(
                    float(golden_metrics[metric_name]),
                    float(candidate_metrics[metric_name]),
                    rel_tol=tolerance,
                    abs_tol=tolerance,
                ):
                    differences.append({
                        "world": world,
                        "k": int(k),
                        "metric": metric_name,
                        "golden": golden_metrics[metric_name],
                        "candidate": candidate_metrics[metric_name],
                    })
            if golden_metrics["valid_queries"] != candidate_metrics["valid_queries"]:
                differences.append({
                    "world": world,
                    "k": int(k),
                    "metric": "valid_queries",
                    "golden": golden_metrics["valid_queries"],
                    "candidate": candidate_metrics["valid_queries"],
                })
    return not differences, differences


def compare_runs(golden_run_dir, candidate_run_dir, tolerance=1e-12):
    golden = _load_run(golden_run_dir)
    candidate = _load_run(candidate_run_dir)
    golden_results = golden["results"]
    candidate_results = candidate["results"]
    golden_items = golden_results["results"]
    candidate_items = candidate_results["results"]
    golden_ids = [item["query_id"] for item in golden_items]
    candidate_ids = [item["query_id"] for item in candidate_items]

    method_reports = {}
    for method in METHODS:
        ranking_mismatches = []
        for golden_item, candidate_item in zip(golden_items, candidate_items):
            golden_ranking = [
                document["standard_law_id"]
                for document in golden_item["metrics"][method]["top_k"][:20]
            ]
            candidate_ranking = [
                document["standard_law_id"]
                for document in candidate_item["metrics"][method]["top_k"][:20]
            ]
            if golden_ranking != candidate_ranking:
                ranking_mismatches.append(golden_item["query_id"])
        metrics_equal, metric_differences = _metrics_equal(
            golden["summary"], candidate["summary"], method, tolerance
        )
        method_reports[method] = {
            "top20_rankings_equal": not ranking_mismatches,
            "ranking_mismatch_query_ids": ranking_mismatches,
            "metrics_equal": metrics_equal,
            "metric_differences": metric_differences,
        }

    manifest_fields = (
        "query_representation",
        "document_representation",
        "corpus_count",
        "embedding_model",
        "embedding_requested_revision",
        "embedding_resolved_revision",
        "fusion_method",
        "static_w_bm25",
    )
    protocol_differences = {
        field: {
            "golden": golden["manifest"].get(field),
            "candidate": candidate["manifest"].get(field),
        }
        for field in manifest_fields
        if golden["manifest"].get(field) != candidate["manifest"].get(field)
    }
    report = {
        "golden_run_id": golden["manifest"]["run_id"],
        "candidate_run_id": candidate["manifest"]["run_id"],
        "query_count_equal": len(golden_items) == len(candidate_items),
        "query_ids_equal": golden_ids == candidate_ids,
        "configuration_names_equal": golden_results["configurations"]
        == candidate_results["configurations"],
        "methods": method_reports,
        "ground_truth_coverage_equal": golden["summary"]["ground_truth_coverage"]
        == candidate["summary"]["ground_truth_coverage"],
        "corpus_fingerprint_equal": golden["manifest"]["combined_corpus_fingerprint"]
        == candidate["manifest"]["combined_corpus_fingerprint"],
        "query_fingerprint_equal": golden["manifest"]["query_representation_fingerprint"]
        == candidate["manifest"]["query_representation_fingerprint"],
        "protocol_equal": not protocol_differences,
        "protocol_differences": protocol_differences,
    }
    report["passed"] = all([
        report["query_count_equal"],
        report["query_ids_equal"],
        report["configuration_names_equal"],
        report["ground_truth_coverage_equal"],
        report["corpus_fingerprint_equal"],
        report["query_fingerprint_equal"],
        report["protocol_equal"],
        all(
            method["top20_rankings_equal"] and method["metrics_equal"]
            for method in method_reports.values()
        ),
    ])
    return report


def assert_runs_equal(golden_run_dir, candidate_run_dir, tolerance=1e-12):
    report = compare_runs(golden_run_dir, candidate_run_dir, tolerance=tolerance)
    if not report["passed"]:
        raise AssertionError(json.dumps(report, ensure_ascii=False, indent=2))
    return report
