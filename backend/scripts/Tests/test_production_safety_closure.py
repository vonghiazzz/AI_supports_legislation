import inspect
import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_performance_v1
from backend.scripts.Evaluation.benchmarking import independent_benchmark as frozen_v1
from backend.scripts.Evaluation.benchmarking import production_safety_closure as r13
from backend.scripts.Evaluation.benchmarking import source_intake


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_r13_baseline_and_r12_artifacts_are_exact():
    baseline = r13._baseline(PROJECT_ROOT)
    assert baseline["manifest_sha256"] == r13.SOURCE_MANIFEST_SHA256
    assert baseline["policy_fingerprint"] == r13.POLICY_FINGERPRINT
    assert baseline["r12_run_fingerprint"] == r13.R12_RUN_FINGERPRINT


def test_optimized_source_intake_path_is_enabled_by_default():
    source = inspect.getsource(source_intake._eligibility_dedup_report)
    assert "deduplicate_memoized" in source
    assert "if _precompute_shingles" in source


def test_parent_frozen_function_never_changes_during_concurrent_workers():
    result = r13.validate_shared_state_safety()
    assert result["parent_module_function_mutated"] is False
    assert result["concurrent_workers_equal"] is True
    assert result["direct_parent_caller_remained_original"] is True


def test_spawn_worker_does_not_observe_or_replace_parent_monkeypatch(monkeypatch):
    inventory, development, links, _ = r13.equivalence_fixture()
    sentinel_calls = []

    def sentinel(left, right):
        sentinel_calls.append((left, right))
        return 0.123456

    monkeypatch.setattr(frozen_v1, "dedup_text_similarity", sentinel)
    holder = {}

    def run():
        holder["result"] = dedup_performance_v1.deduplicate_memoized(
            inventory,
            development,
            links,
            _worker_delay_after_patch=0.5,
        )

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while thread.is_alive() and time.monotonic() < deadline:
        assert frozen_v1.dedup_text_similarity is sentinel
        assert frozen_v1.dedup_text_similarity("a", "b") == 0.123456
        time.sleep(0.01)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert "result" in holder
    assert frozen_v1.dedup_text_similarity is sentinel
    assert sentinel_calls


def test_worker_exception_and_timeout_leave_parent_unchanged():
    inventory, development, links, _ = r13.equivalence_fixture()
    original = frozen_v1.dedup_text_similarity
    with pytest.raises(dedup_performance_v1.DedupPerformanceError, match="injected"):
        dedup_performance_v1.deduplicate_memoized(
            inventory,
            development,
            links,
            _inject_worker_failure_after_patch=True,
        )
    assert frozen_v1.dedup_text_similarity is original
    with pytest.raises(dedup_performance_v1.DedupPerformanceError, match="timed out"):
        dedup_performance_v1.deduplicate_memoized(
            inventory,
            development,
            links,
            _worker_timeout_seconds=0.05,
            _worker_delay_after_patch=1.0,
        )
    assert frozen_v1.dedup_text_similarity is original
    time.sleep(0.05)
    assert not any(
        child.name.startswith("v1-dedup-")
        for child in multiprocessing.active_children()
    )


def test_targeted_v1_output_equivalence_covers_all_required_controls():
    result = r13.validate_fixture_equivalence()
    assert result["full_output_equal"] is True
    assert result["eligible_membership_equal"] is True
    assert result["source_order_equal"] is True
    assert result["development_findings_equal"] is True
    assert result["internal_clusters_equal"] is True
    assert result["canonical_representatives_equal"] is True
    assert result["exclusion_reasons_equal"] is True
    assert result["threshold_scores"]["above_threshold"] >= 0.72
    assert result["threshold_scores"]["below_threshold"] < 0.72


@pytest.fixture(scope="module")
def full_population_result():
    return r13.validate_full_population_against_r12(PROJECT_ROOT)


def test_full_population_isolated_worker_matches_r12_reference(full_population_result):
    assert full_population_result["population_record_count"] == 469
    assert full_population_result["matches_r12_exact_frozen_reference"] is True


@pytest.fixture(scope="module")
def nonempty_record_result():
    return r13.perform_nonempty_record_clone(PROJECT_ROOT)


def test_nonempty_v21_record_survives_real_population_v1_continuity(
    nonempty_record_result,
):
    assert nonempty_record_result["promotion_status"] == "eligible"
    assert nonempty_record_result["v2_review_applied"] is False
    assert nonempty_record_result["v2_evidence_used_for_promotion"] is False
    assert nonempty_record_result["live_mutation_executed"] is False
    assert all(nonempty_record_result["checks"].values())


def test_governance_release_candidate_keeps_all_live_authorities_false(
    full_population_result,
    nonempty_record_result,
):
    shared = r13.validate_shared_state_safety()
    fixture = r13.validate_fixture_equivalence()
    outputs = r13.build_outputs(
        PROJECT_ROOT,
        "f" * 40,
        r13.PROTOCOL_ID + "_fixture",
        shared,
        fixture,
        full_population_result,
        nonempty_record_result,
    )
    decision = outputs["final_go_no_go.json"]
    assert decision["ENGINEERING_RELEASE_CANDIDATE_READY"] is True
    assert decision["GOVERNANCE_APPROVAL_COMPLETE"] is False
    assert decision["PUBLIC_REGISTRY_PROVISIONED"] is False
    assert decision["LIVE_MIGRATION_AUTHORIZED"] is False
    summary = outputs["validation_summary.json"]
    assert summary["live_manifest_mutated"] is False
    assert summary["live_review_or_promotion_performed"] is False
