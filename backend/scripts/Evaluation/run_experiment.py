import argparse
import sys
from pathlib import Path

import yaml


if __package__:
    from .experiments.baseline import (
        BASELINE_MODE,
        BASELINE_PROTOCOL_ID,
        EXP02_EXPERIMENT_ID,
        run_baseline_evaluation_pipeline,
    )
    from .experiments.exp03_static_fusion_cv import (
        EXP03_MODE,
        EXP03_PROTOCOL_ID,
        run_exp03_static_fusion_cv,
    )
    from .experiments.exp04_selective_dynamic_hybrid_nested_cv import (
        EXP04_MODE,
        EXP04_PROTOCOL_ID,
        run_exp04_selective_dynamic_hybrid_nested_cv,
    )
    from .experiments.exp05_frozen_hybrid30_zero_shot_reranker import (
        EXP05_MODE,
        EXP05_PROTOCOL_ID,
        run_exp05_frozen_hybrid30_zero_shot_reranker,
    )
    from .experiments.legacy_advanced import run_legacy_advanced
else:
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from backend.scripts.Evaluation.experiments.baseline import (
        BASELINE_MODE,
        BASELINE_PROTOCOL_ID,
        EXP02_EXPERIMENT_ID,
        run_baseline_evaluation_pipeline,
    )
    from backend.scripts.Evaluation.experiments.exp03_static_fusion_cv import (
        EXP03_MODE,
        EXP03_PROTOCOL_ID,
        run_exp03_static_fusion_cv,
    )
    from backend.scripts.Evaluation.experiments.exp04_selective_dynamic_hybrid_nested_cv import (
        EXP04_MODE,
        EXP04_PROTOCOL_ID,
        run_exp04_selective_dynamic_hybrid_nested_cv,
    )
    from backend.scripts.Evaluation.experiments.exp05_frozen_hybrid30_zero_shot_reranker import (
        EXP05_MODE,
        EXP05_PROTOCOL_ID,
        run_exp05_frozen_hybrid30_zero_shot_reranker,
    )
    from backend.scripts.Evaluation.experiments.legacy_advanced import run_legacy_advanced


DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "baseline_d0.yaml"
LEGACY_MODE = "legacy"
BASELINE_PROTOCOL_IDS = {BASELINE_PROTOCOL_ID, EXP02_EXPERIMENT_ID}


def run_experiment(config_path=DEFAULT_CONFIG_PATH, max_queries=None, temporal_filter="on"):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    mode = config.get("evaluation", {}).get("mode", LEGACY_MODE)
    protocol_id = config.get("protocol", {}).get("id")
    if mode == BASELINE_MODE and protocol_id in BASELINE_PROTOCOL_IDS:
        return run_baseline_evaluation_pipeline(config_path, max_queries=max_queries)
    if mode == EXP03_MODE and protocol_id == EXP03_PROTOCOL_ID:
        return run_exp03_static_fusion_cv(config_path, max_queries=max_queries)
    if mode == EXP04_MODE and protocol_id == EXP04_PROTOCOL_ID:
        return run_exp04_selective_dynamic_hybrid_nested_cv(
            config_path, max_queries=max_queries
        )
    if mode == EXP05_MODE and protocol_id == EXP05_PROTOCOL_ID:
        return run_exp05_frozen_hybrid30_zero_shot_reranker(
            config_path, max_queries=max_queries
        )
    if mode == LEGACY_MODE and protocol_id is None:
        return run_legacy_advanced(
            temporal_filter=temporal_filter,
            config_path=config_path,
            max_queries=max_queries,
        )
    raise ValueError(
        "Unsupported evaluation dispatch combination: "
        f"evaluation.mode={mode!r}, protocol.id={protocol_id!r}"
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
