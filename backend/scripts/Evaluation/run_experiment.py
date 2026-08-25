import argparse
import sys
from pathlib import Path

import yaml


if __package__:
    from .experiments.baseline import BASELINE_MODE, run_baseline_evaluation_pipeline
    from .experiments.legacy_advanced import run_legacy_advanced
else:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from backend.scripts.Evaluation.experiments.baseline import (
        BASELINE_MODE,
        run_baseline_evaluation_pipeline,
    )
    from backend.scripts.Evaluation.experiments.legacy_advanced import run_legacy_advanced


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "baseline_d0.yaml"


def run_experiment(config_path=DEFAULT_CONFIG_PATH, max_queries=None, temporal_filter="on"):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        mode = yaml.safe_load(handle).get("evaluation", {}).get("mode", "legacy")
    if mode == BASELINE_MODE:
        return run_baseline_evaluation_pipeline(config_path, max_queries=max_queries)
    return run_legacy_advanced(
        temporal_filter=temporal_filter,
        config_path=config_path,
        max_queries=max_queries,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an experimental legal retrieval evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--temporal-filter", choices=["on", "off"], default="on")
    args = parser.parse_args(argv)
    output_path = run_experiment(
        config_path=args.config,
        max_queries=args.max_queries,
        temporal_filter=args.temporal_filter,
    )
    print(f"Evaluation complete: {output_path}")
    return output_path


if __name__ == "__main__":
    main()
