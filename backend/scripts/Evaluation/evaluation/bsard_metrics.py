"""Separated metric contracts for the BSARD benchmark.

``evaluate_official_compat`` reproduces the published scorer formulas on
well-formed positional inputs, including raw ground-truth occurrence
denominators.  ``evaluate_safe_canonical`` uses unique qrels and strict qid
alignment.  Values from the two APIs must never be mixed without retaining
their ``metric_semantics`` metadata.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


OFFICIAL_CUTOFFS = (100, 200, 500)
OFFICIAL_MAP_CUTOFF = 100
OFFICIAL_MRR_CUTOFF = 100


class BSARDMetricError(ValueError):
    """Raised when benchmark or prediction identity is unsafe."""


def _as_id(value: Any, *, context: str) -> str:
    if isinstance(value, bool):
        raise BSARDMetricError(f"{context} contains an invalid boolean ID")
    normalized = str(value)
    if not normalized or not normalized.isdecimal():
        raise BSARDMetricError(f"{context} contains a malformed ID: {value!r}")
    return normalized


def _normalize_ids(values: Iterable[Any], *, context: str) -> list[str]:
    return [_as_id(value, context=context) for value in values]


def _validate_prediction(
    values: Iterable[Any],
    *,
    context: str,
    known_document_ids: set[str],
    allow_empty: bool,
) -> list[str]:
    prediction = _normalize_ids(values, context=context)
    if not prediction and not allow_empty:
        raise BSARDMetricError(f"{context} is empty")
    if len(prediction) != len(set(prediction)):
        raise BSARDMetricError(f"{context} contains duplicate document IDs")
    unknown = sorted(set(prediction) - known_document_ids, key=int)
    if unknown:
        raise BSARDMetricError(f"{context} contains unknown document IDs: {unknown}")
    if len(prediction) > len(known_document_ids):
        raise BSARDMetricError(f"{context} is longer than the corpus")
    return prediction


def _recall_at(prediction: Sequence[str], relevant: Sequence[str] | set[str], k: int) -> float:
    return sum(document_id in relevant for document_id in prediction[:k]) / len(relevant)


def _average_precision_at(
    prediction: Sequence[str], relevant: Sequence[str] | set[str], k: int
) -> float:
    hits = 0
    precision_sum = 0.0
    for rank, document_id in enumerate(prediction[:k], 1):
        if document_id in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def _reciprocal_rank_at(
    prediction: Sequence[str], relevant: Sequence[str] | set[str], k: int
) -> float:
    for rank, document_id in enumerate(prediction[:k], 1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def _ndcg_at(prediction: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(prediction[:k], 1)
        if document_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_official_compat(
    raw_ground_truths: Sequence[Sequence[Any]],
    ranked_predictions: Sequence[Sequence[Any]],
    *,
    known_document_ids: Iterable[Any],
    query_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Evaluate valid positional inputs with historical BSARD formulas.

    Raw duplicate ground-truth occurrences remain in the Recall/AP denominator.
    Unlike the legacy utility, this wrapper rejects malformed predictions and
    positional length mismatches instead of silently truncating them.
    """
    if len(raw_ground_truths) != len(ranked_predictions):
        raise BSARDMetricError("query/result length mismatch is not permitted")
    if query_ids is not None and len(query_ids) != len(raw_ground_truths):
        raise BSARDMetricError("query_ids length does not match benchmark input")
    known = set(_normalize_ids(known_document_ids, context="known document IDs"))
    if not known:
        raise BSARDMetricError("known document ID set is empty")

    per_query: list[dict[str, Any]] = []
    for index, (raw_gt, raw_prediction) in enumerate(
        zip(raw_ground_truths, ranked_predictions)
    ):
        relevant = _normalize_ids(raw_gt, context=f"ground truth at position {index}")
        if not relevant:
            raise BSARDMetricError(f"ground truth at position {index} is empty")
        unknown_gt = sorted(set(relevant) - known, key=int)
        if unknown_gt:
            raise BSARDMetricError(
                f"ground truth at position {index} contains unknown IDs: {unknown_gt}"
            )
        prediction = _validate_prediction(
            raw_prediction,
            context=f"prediction at position {index}",
            known_document_ids=known,
            allow_empty=False,
        )
        metrics = {
            **{
                f"recall@{cutoff}": _recall_at(prediction, relevant, cutoff)
                for cutoff in OFFICIAL_CUTOFFS
            },
            "map@100": _average_precision_at(
                prediction, relevant, OFFICIAL_MAP_CUTOFF
            ),
            "mrr@100": _reciprocal_rank_at(
                prediction, relevant, OFFICIAL_MRR_CUTOFF
            ),
        }
        per_query.append(
            {
                "query_id": str(query_ids[index]) if query_ids is not None else None,
                "raw_relevant_count": len(relevant),
                "unique_relevant_count": len(set(relevant)),
                "metrics": metrics,
            }
        )

    metric_names = (*[f"recall@{k}" for k in OFFICIAL_CUTOFFS], "map@100", "mrr@100")
    return {
        "metric_semantics": "bsard_official_compat",
        "alignment": "positional_validated",
        "binary_relevance": True,
        "query_count": len(per_query),
        "metrics": {
            name: sum(record["metrics"][name] for record in per_query) / len(per_query)
            if per_query
            else 0.0
            for name in metric_names
        },
        "per_query": per_query,
    }


def evaluate_safe_canonical(
    ground_truths_by_qid: Mapping[Any, Iterable[Any]],
    predictions_by_qid: Mapping[Any, Iterable[Any]],
    *,
    known_document_ids: Iterable[Any],
    recall_cutoffs: Sequence[int] = (10, 20, 30, 50, 100, 200, 500),
    map_cutoff: int = 100,
    mrr_cutoff: int = 100,
    ndcg_cutoffs: Sequence[int] = (10,),
) -> dict[str, Any]:
    """Evaluate qid-aligned predictions with unique canonical qrels."""
    gt_by_qid = {
        _as_id(qid, context="ground-truth qid"): _normalize_ids(
            values, context=f"ground truth for qid {qid}"
        )
        for qid, values in ground_truths_by_qid.items()
    }
    predictions = {
        _as_id(qid, context="prediction qid"): values
        for qid, values in predictions_by_qid.items()
    }
    if not gt_by_qid:
        raise BSARDMetricError("benchmark has no queries")
    missing = sorted(set(gt_by_qid) - set(predictions), key=int)
    extra = sorted(set(predictions) - set(gt_by_qid), key=int)
    if missing or extra:
        raise BSARDMetricError(f"qid mismatch: missing={missing}, extra={extra}")
    known = set(_normalize_ids(known_document_ids, context="known document IDs"))
    if not known:
        raise BSARDMetricError("known document ID set is empty")
    if any(not isinstance(k, int) or isinstance(k, bool) or k <= 0 for k in (*recall_cutoffs, *ndcg_cutoffs, map_cutoff, mrr_cutoff)):
        raise BSARDMetricError("metric cutoffs must be positive integers")

    per_query: dict[str, dict[str, Any]] = {}
    for qid in sorted(gt_by_qid, key=int):
        raw_relevant = gt_by_qid[qid]
        relevant = set(raw_relevant)
        if not relevant:
            raise BSARDMetricError(f"ground truth for qid {qid} is empty")
        unknown_gt = sorted(relevant - known, key=int)
        if unknown_gt:
            raise BSARDMetricError(
                f"ground truth for qid {qid} contains unknown IDs: {unknown_gt}"
            )
        prediction = _validate_prediction(
            predictions[qid],
            context=f"prediction for qid {qid}",
            known_document_ids=known,
            allow_empty=True,
        )
        metrics = {
            **{
                f"recall@{cutoff}": _recall_at(prediction, relevant, cutoff)
                for cutoff in recall_cutoffs
            },
            **{
                f"candidate_recall@{cutoff}": _recall_at(
                    prediction, relevant, cutoff
                )
                for cutoff in recall_cutoffs
            },
            **{
                f"ndcg@{cutoff}": _ndcg_at(prediction, relevant, cutoff)
                for cutoff in ndcg_cutoffs
            },
            f"map@{map_cutoff}": _average_precision_at(
                prediction, relevant, map_cutoff
            ),
            f"mrr@{mrr_cutoff}": _reciprocal_rank_at(
                prediction, relevant, mrr_cutoff
            ),
        }
        per_query[qid] = {
            "raw_relevant_count": len(raw_relevant),
            "unique_relevant_count": len(relevant),
            "prediction_count": len(prediction),
            "metrics": metrics,
        }

    metric_names = tuple(next(iter(per_query.values()))["metrics"])
    return {
        "metric_semantics": "bsard_safe_canonical",
        "alignment": "query_id",
        "qrel_semantics": "unique_query_document_pairs",
        "binary_relevance": True,
        "query_count": len(per_query),
        "metrics": {
            name: sum(record["metrics"][name] for record in per_query.values())
            / len(per_query)
            for name in metric_names
        },
        "per_query": per_query,
    }


def paired_bootstrap_mean_delta(
    control: Sequence[float],
    treatment: Sequence[float],
    *,
    samples: int = 20000,
    seed: int = 42,
) -> dict[str, float]:
    """Deterministic paired bootstrap helper for later diagnostic use."""
    import numpy as np

    if len(control) != len(treatment) or not control:
        raise BSARDMetricError("paired bootstrap requires equal non-empty inputs")
    if samples <= 0:
        raise BSARDMetricError("bootstrap sample count must be positive")
    deltas = np.asarray(treatment, dtype=float) - np.asarray(control, dtype=float)
    if not np.all(np.isfinite(deltas)):
        raise BSARDMetricError("bootstrap inputs must be finite")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
    sampled_means = deltas[indices].mean(axis=1)
    return {
        "mean_delta": float(deltas.mean()),
        "ci_lower": float(np.percentile(sampled_means, 2.5)),
        "ci_upper": float(np.percentile(sampled_means, 97.5)),
        "samples": int(samples),
        "seed": int(seed),
    }


__all__ = [
    "BSARDMetricError",
    "evaluate_official_compat",
    "evaluate_safe_canonical",
    "paired_bootstrap_mean_delta",
]
