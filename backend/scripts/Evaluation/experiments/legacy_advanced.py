"""Lazy boundary for the historical advanced evaluator.

The legacy implementation remains in the compatibility entrypoint during this
refactor so its research behavior is not rewritten accidentally.
"""


def run_legacy_advanced(temporal_filter="on", config_path=None, max_queries=None):
    from .. import run_rag_evaluation as compatibility

    return compatibility.run_evaluation_pipeline(
        temporal_filter=temporal_filter,
        config_path=config_path,
        max_queries=max_queries,
    )
