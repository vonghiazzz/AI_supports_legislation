"""Semantics-preserving execution adapters for the frozen V1 dedup function.

The scientific implementation in :mod:`independent_benchmark` is protected by
an exact historical SHA256 and remains byte-for-byte unchanged.  The adapters
therefore run it in a dedicated spawned process.  Any temporary similarity
primitive replacement is worker-local and cannot be observed by direct callers
in the parent process.
"""

from __future__ import annotations

import multiprocessing
import resource
import time
import traceback
from typing import Any

from . import independent_benchmark as frozen


WORKER_PROTOCOL_VERSION = "isolated_v1_dedup_worker_v1"
DEFAULT_WORKER_TIMEOUT_SECONDS = 15 * 60


class DedupPerformanceError(RuntimeError):
    """The isolated V1 execution worker failed closed."""


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


def _deduplicate_reference_profiled_local(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | list[tuple[str, str]] | None = None,
    *,
    performance_stats: dict[str, Any] | None = None,
    _delay_after_patch: float = 0.0,
    _inject_failure_after_patch: bool = False,
) -> tuple[list[dict], dict]:
    """Worker-local exact V1 execution with repeated-work counters."""

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
    historical_similarity = frozen.dedup_text_similarity

    def counted(left: str, right: str) -> float:
        stats["similarity_comparisons"] += 1
        stats["shingle_computations"] += 2
        return historical_similarity(left, right)

    frozen.dedup_text_similarity = counted
    try:
        if _delay_after_patch:
            time.sleep(_delay_after_patch)
        if _inject_failure_after_patch:
            raise RuntimeError("injected isolated reference failure")
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


def _deduplicate_memoized_local(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | list[tuple[str, str]] | None = None,
    *,
    performance_stats: dict[str, Any] | None = None,
    _delay_after_patch: float = 0.0,
    _inject_failure_after_patch: bool = False,
) -> tuple[list[dict], dict]:
    """Worker-local V1 execution with memoized shingle construction."""

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
        if _delay_after_patch:
            time.sleep(_delay_after_patch)
        if _inject_failure_after_patch:
            raise RuntimeError("injected isolated memoized failure")
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


def _worker_entry(
    connection,
    mode: str,
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links,
    delay_after_patch: float,
    inject_failure_after_patch: bool,
) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    try:
        stats: dict[str, Any] = {}
        evaluator = (
            _deduplicate_memoized_local
            if mode == "memoized"
            else _deduplicate_reference_profiled_local
        )
        eligible, report = evaluator(
            inventory,
            development_records,
            known_duplicate_links,
            performance_stats=stats,
            _delay_after_patch=delay_after_patch,
            _inject_failure_after_patch=inject_failure_after_patch,
        )
        stats["worker_wall_seconds"] = time.perf_counter() - started_wall
        stats["worker_cpu_seconds"] = time.process_time() - started_cpu
        stats["worker_peak_rss_platform_units"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        connection.send(
            {
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "ok": True,
                "eligible": eligible,
                "report": report,
                "statistics": stats,
            }
        )
    except BaseException as exc:
        connection.send(
            {
                "protocol_version": WORKER_PROTOCOL_VERSION,
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        connection.close()


def _execute_isolated(
    mode: str,
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links,
    *,
    performance_stats: dict[str, Any] | None,
    worker_timeout_seconds: float,
    worker_delay_after_patch: float,
    inject_worker_failure_after_patch: bool,
) -> tuple[list[dict], dict]:
    if mode not in {"reference", "memoized"}:
        raise DedupPerformanceError("Unsupported isolated worker mode")
    if worker_timeout_seconds <= 0 or worker_delay_after_patch < 0:
        raise DedupPerformanceError("Invalid isolated worker timing")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(
            child,
            mode,
            inventory,
            development_records,
            known_duplicate_links,
            worker_delay_after_patch,
            inject_worker_failure_after_patch,
        ),
        name=f"v1-dedup-{mode}-worker",
    )
    started = False
    try:
        process.start()
        started = True
        child.close()
        if not parent.poll(worker_timeout_seconds):
            process.terminate()
            process.join(timeout=10)
            raise DedupPerformanceError("Isolated V1 dedup worker timed out")
        payload = parent.recv()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            raise DedupPerformanceError("Isolated V1 dedup worker did not exit")
        if (
            not isinstance(payload, dict)
            or payload.get("protocol_version") != WORKER_PROTOCOL_VERSION
        ):
            raise DedupPerformanceError("Isolated V1 dedup worker response is invalid")
        if not payload.get("ok"):
            raise DedupPerformanceError(
                "Isolated V1 dedup worker failed: "
                f"{payload.get('error_type')}: {payload.get('error_message')}"
            )
        if process.exitcode != 0:
            raise DedupPerformanceError(
                f"Isolated V1 dedup worker exited with code {process.exitcode}"
            )
        if performance_stats is not None:
            performance_stats.clear()
            performance_stats.update(payload["statistics"])
        return payload["eligible"], payload["report"]
    except EOFError as exc:
        if started:
            process.join(timeout=10)
        raise DedupPerformanceError("Isolated V1 dedup worker ended without result") from exc
    finally:
        child.close()
        parent.close()
        if started and process.is_alive():
            process.terminate()
            process.join(timeout=10)


def deduplicate_reference_profiled(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | list[tuple[str, str]] | None = None,
    *,
    performance_stats: dict[str, Any] | None = None,
    _worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    _worker_delay_after_patch: float = 0.0,
    _inject_worker_failure_after_patch: bool = False,
) -> tuple[list[dict], dict]:
    """Run exact frozen V1 in a dedicated process with profiling counters."""

    return _execute_isolated(
        "reference",
        inventory,
        development_records,
        known_duplicate_links,
        performance_stats=performance_stats,
        worker_timeout_seconds=_worker_timeout_seconds,
        worker_delay_after_patch=_worker_delay_after_patch,
        inject_worker_failure_after_patch=_inject_worker_failure_after_patch,
    )


def deduplicate_memoized(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | list[tuple[str, str]] | None = None,
    *,
    performance_stats: dict[str, Any] | None = None,
    _worker_timeout_seconds: float = DEFAULT_WORKER_TIMEOUT_SECONDS,
    _worker_delay_after_patch: float = 0.0,
    _inject_worker_failure_after_patch: bool = False,
) -> tuple[list[dict], dict]:
    """Run memoized V1 in a dedicated process; parent globals stay untouched."""

    return _execute_isolated(
        "memoized",
        inventory,
        development_records,
        known_duplicate_links,
        performance_stats=performance_stats,
        worker_timeout_seconds=_worker_timeout_seconds,
        worker_delay_after_patch=_worker_delay_after_patch,
        inject_worker_failure_after_patch=_inject_worker_failure_after_patch,
    )
