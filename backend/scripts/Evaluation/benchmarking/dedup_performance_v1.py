"""Semantics-preserving execution adapters for the frozen V1 dedup function.

The scientific implementation in :mod:`independent_benchmark` is protected by
an exact historical SHA256 and remains byte-for-byte unchanged.  R12 therefore
adds execution-only adapters here.  Both call the frozen function itself; the
optimized adapter temporarily supplies an output-equivalent memoized similarity
primitive under a process lock and always restores the historical primitive.
"""

from __future__ import annotations

import threading
from typing import Any

from . import independent_benchmark as frozen


_EXECUTION_LOCK = threading.RLock()


def _pool_counts(eligible: list[dict], report: dict) -> tuple[int, int]:
    noncanonical = sum(
        any(
            reason.startswith("canonical_representative:")
            for reason in exclusion["reasons"]
        )
        for exclusion in report["exclusions"]
    )
    new_candidate_count = len(eligible) + noncanonical
    return (
        new_candidate_count,
        new_candidate_count * (new_candidate_count - 1) // 2,
    )


def deduplicate_reference_profiled(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | list[tuple[str, str]] | None = None,
    *,
    performance_stats: dict[str, Any] | None = None,
) -> tuple[list[dict], dict]:
    """Run exact frozen V1 while counting its repeated similarity work."""

    stats = performance_stats if performance_stats is not None else {}
    stats.clear()
    stats.update(
        {
            "execution_adapter": "EXACT_FROZEN_V1_PROFILED",
            "precomputed_shingles": False,
            "shingle_computations": 0,
            "similarity_comparisons": 0,
            "length_bound_pruned_comparisons": 0,
        }
    )
    with _EXECUTION_LOCK:
        historical_similarity = frozen.dedup_text_similarity

        def counted(left: str, right: str) -> float:
            stats["similarity_comparisons"] += 1
            stats["shingle_computations"] += 2
            return historical_similarity(left, right)

        frozen.dedup_text_similarity = counted
        try:
            eligible, report = frozen.deduplicate_against_development(
                inventory,
                development_records,
                known_duplicate_links=known_duplicate_links,
            )
        finally:
            frozen.dedup_text_similarity = historical_similarity
    new_count, pair_count = _pool_counts(eligible, report)
    stats["new_candidate_count"] = new_count
    stats["pool_pair_comparisons"] = pair_count
    return eligible, report


def deduplicate_memoized(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | list[tuple[str, str]] | None = None,
    *,
    performance_stats: dict[str, Any] | None = None,
) -> tuple[list[dict], dict]:
    """Run frozen V1 with one shingle construction per distinct source text."""

    stats = performance_stats if performance_stats is not None else {}
    stats.clear()
    stats.update(
        {
            "execution_adapter": "R12_MEMOIZED_SHINGLES_V1",
            "precomputed_shingles": True,
            "shingle_computations": 0,
            "similarity_comparisons": 0,
            "length_bound_pruned_comparisons": 0,
        }
    )
    with _EXECUTION_LOCK:
        historical_similarity = frozen.dedup_text_similarity
        shingle_cache: dict[str, set[tuple[str, ...]]] = {}
        for record in inventory:
            text = record.get("_dedup_text", "")
            if text not in shingle_cache:
                shingle_cache[text] = frozen._dedup_shingles(text)
                stats["shingle_computations"] += 1

        def shingles(text: str) -> set[tuple[str, ...]]:
            if text not in shingle_cache:
                shingle_cache[text] = frozen._dedup_shingles(text)
                stats["shingle_computations"] += 1
            return shingle_cache[text]

        def memoized(left: str, right: str) -> float:
            stats["similarity_comparisons"] += 1
            left_shingles = shingles(left)
            right_shingles = shingles(right)
            if not left_shingles and not right_shingles:
                return 1.0
            larger = max(len(left_shingles), len(right_shingles))
            smaller = min(len(left_shingles), len(right_shingles))
            if larger and smaller / larger < frozen.NEAR_DUPLICATE_THRESHOLD:
                stats["length_bound_pruned_comparisons"] += 1
                return 0.0
            union = left_shingles | right_shingles
            return len(left_shingles & right_shingles) / len(union)

        frozen.dedup_text_similarity = memoized
        try:
            eligible, report = frozen.deduplicate_against_development(
                inventory,
                development_records,
                known_duplicate_links=known_duplicate_links,
            )
        finally:
            frozen.dedup_text_similarity = historical_similarity
    new_count, pair_count = _pool_counts(eligible, report)
    stats["new_candidate_count"] = new_count
    stats["pool_pair_comparisons"] = pair_count
    return eligible, report
