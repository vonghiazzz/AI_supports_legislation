import json
import os
import shutil
import tempfile
from pathlib import Path


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


def write_exp03_artifacts(
    run_dir,
    config_path,
    candidate_cache,
    fold_manifest,
    results_output,
    summary,
    manifest,
):
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"EXP-03 run directory already exists: {run_dir}")

    artifacts = {
        "candidate_pools.json": candidate_cache,
        "fold_manifest.json": fold_manifest,
        "pipeline_results.json": results_output,
        "summary_metrics.json": summary,
        "run_manifest.json": manifest,
    }
    expected_artifacts = set(artifacts) | {"eval_config.snapshot.yaml"}
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{run_dir.name}.partial-",
            dir=run_dir.parent,
        )
    )
    try:
        for filename, payload in artifacts.items():
            with open(temp_dir / filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        shutil.copy2(config_path, temp_dir / "eval_config.snapshot.yaml")
        actual_artifacts = {path.name for path in temp_dir.iterdir()}
        if actual_artifacts != expected_artifacts:
            raise RuntimeError(
                "EXP-03 artifact publication requires the exact complete artifact set"
            )
        os.replace(temp_dir, run_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return run_dir / "pipeline_results.json"


def write_exp04_artifacts(run_dir, config_path, artifacts):
    """Atomically publish the complete EXP-04 run-scoped artifact set."""
    run_dir = Path(run_dir)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(f"EXP-04 run directory already exists: {run_dir}")
    expected_artifacts = set(artifacts) | {"eval_config.snapshot.yaml"}
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{run_dir.name}.partial-",
            dir=run_dir.parent,
        )
    )
    try:
        for filename, payload in artifacts.items():
            with open(temp_dir / filename, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        shutil.copy2(config_path, temp_dir / "eval_config.snapshot.yaml")
        actual_artifacts = {path.name for path in temp_dir.iterdir()}
        if actual_artifacts != expected_artifacts:
            raise RuntimeError(
                "EXP-04 artifact publication requires the exact complete artifact set"
            )
        os.replace(temp_dir, run_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return run_dir / "pipeline_results.json"
