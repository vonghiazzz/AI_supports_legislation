import json
import shutil


BASELINE_PRIMARY_WORLD = "closed_world"
BASELINE_PRIMARY_K = 10


def assemble_baseline_summary(metadata, metric_worlds, gt_coverage, latency_aggregates):
    if metadata.get("primary_world") != BASELINE_PRIMARY_WORLD:
        raise ValueError("Baseline summary primary_world must be closed_world")
    if int(metadata.get("primary_k", 0)) != BASELINE_PRIMARY_K:
        raise ValueError("Baseline summary primary_k must be 10")
    return {
        **metadata,
        "ground_truth_coverage": gt_coverage,
        "closed_world": metric_worlds["closed_world"],
        "open_world": metric_worlds["open_world"],
        "latency_ms": latency_aggregates,
    }


def write_baseline_artifacts(
    run_dir,
    config_path,
    results_output,
    summary,
    manifest,
):
    run_dir.mkdir(parents=True, exist_ok=False)
    results_path = run_dir / "pipeline_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_output, handle, ensure_ascii=False, indent=2)
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    shutil.copy2(config_path, run_dir / "eval_config.snapshot.yaml")
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return results_path
