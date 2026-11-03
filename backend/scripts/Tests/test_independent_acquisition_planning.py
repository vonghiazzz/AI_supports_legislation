import ast
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.scripts.Evaluation.benchmarking import acquisition_planning as planner
from backend.scripts.Evaluation.benchmarking import sampling_policy


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / planner.DEFAULT_PLAN_RELATIVE
BENCHMARK_DIR = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"
INTAKE_MANIFEST = (
    PROJECT_ROOT / "backend/dataLaw/independent_intake/acquisition_manifest.json"
)


@pytest.fixture(scope="module")
def plan():
    value = planner.strict_yaml_load(PLAN_PATH)
    planner.validate_plan_config(value)
    return value


@pytest.fixture(scope="module")
def audit():
    return planner.build_acquisition_plan_audit(PROJECT_ROOT, PLAN_PATH)


def _close_metrics(**updates):
    value = {
        "final_eligible_count": 250,
        "land_2024_compatible": 35,
        "family_maximum_filled_minimum_slots": 185,
        "family_shortage": 0,
        "hall_bottleneck_shortage": 0,
        "stress_eligible": 87,
        "development_overlap": 0,
        "duplicate_source_or_case_identity": 0,
        "benchmark_state": "DRAFT",
        "selection_status": "NOT_STARTED",
        "complete_cohort_reviewed": True,
    }
    value.update(updates)
    return value


def _fake_git_runner(
    *,
    head="b" * 40,
    ancestor_returncode=0,
    unstaged_returncode=0,
    staged_returncode=0,
    calls=None,
):
    def run(command, **kwargs):
        if calls is not None:
            calls.append(list(command))
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{head}\n", stderr="")
        if command[:3] == ["git", "merge-base", "--is-ancestor"]:
            return SimpleNamespace(
                returncode=ancestor_returncode, stdout="", stderr=""
            )
        if command == ["git", "diff", "--quiet", "--"]:
            return SimpleNamespace(
                returncode=unstaged_returncode, stdout="", stderr=""
            )
        if command == ["git", "diff", "--cached", "--quiet", "--"]:
            return SimpleNamespace(
                returncode=staged_returncode, stdout="", stderr=""
            )
        raise AssertionError(f"Unexpected git command: {command}")

    return run


def test_plan_identity_and_parent_policy_fingerprint(plan):
    assert plan["plan_id"] == "independent_acquisition_plan_v1"
    assert plan["plan_status"] == "ACQUISITION_PLAN_FROZEN_NOT_EXECUTED"
    assert plan["batch_label"] == "BATCH_2D_B"
    parent = plan["parent_sampling_policy"]
    assert parent["policy_id"] == sampling_policy.POLICY_ID
    assert parent["policy_status"] == sampling_policy.POLICY_STATUS
    assert parent["checkpoint_commit"] == (
        "ef9c3fbefc45d76f0568fb0c3623270c77ed0d27"
    )
    assert parent["sha256"] == (
        "a9926eafe2ee4544f75c194e1de435eeeec5121076d91a6a819424e305b2c91f"
    )
    assert planner.sha256_file(PROJECT_ROOT / parent["path"]) == parent["sha256"]
    assert plan["plan_authoritative_base_commit"] == (
        "a5ebb4a03f35a4c33596faf93b8d5fa82d6d076f"
    )


def test_descendant_execution_may_differ_from_plan_authoring_base(monkeypatch, plan):
    descendant = "b" * 40
    monkeypatch.setattr(
        planner.subprocess, "run", _fake_git_runner(head=descendant)
    )
    result = planner._git_execution_provenance(
        PROJECT_ROOT, plan["parent_sampling_policy"]["sha256"]
    )
    assert descendant != planner.PLAN_AUTHORITATIVE_BASE_COMMIT
    assert result == {
        "execution_git_commit": descendant,
        "plan_authoritative_base_commit": (
            "a5ebb4a03f35a4c33596faf93b8d5fa82d6d076f"
        ),
        "plan_authoritative_base_is_ancestor": True,
        "tracked_worktree_clean": True,
        "parent_sampling_policy_checkpoint_commit": (
            "ef9c3fbefc45d76f0568fb0c3623270c77ed0d27"
        ),
        "parent_sampling_policy_sha256": (
            "a9926eafe2ee4544f75c194e1de435eeeec5121076d91a6a819424e305b2c91f"
        ),
    }


def test_non_descendant_execution_fails_closed(monkeypatch, plan):
    monkeypatch.setattr(
        planner.subprocess,
        "run",
        _fake_git_runner(ancestor_returncode=1),
    )
    with pytest.raises(planner.AcquisitionPlanningError, match="not an ancestor"):
        planner._git_execution_provenance(
            PROJECT_ROOT, plan["parent_sampling_policy"]["sha256"]
        )


@pytest.mark.parametrize(
    ("unstaged", "staged", "message"),
    [(1, 0, "unstaged tracked"), (0, 1, "staged tracked")],
)
def test_tracked_modification_fails_closed(
    monkeypatch, plan, unstaged, staged, message
):
    monkeypatch.setattr(
        planner.subprocess,
        "run",
        _fake_git_runner(
            unstaged_returncode=unstaged, staged_returncode=staged
        ),
    )
    with pytest.raises(planner.AcquisitionPlanningError, match=message):
        planner._git_execution_provenance(
            PROJECT_ROOT, plan["parent_sampling_policy"]["sha256"]
        )


def test_untracked_files_are_allowed_by_execution_gate(monkeypatch, plan):
    calls = []
    monkeypatch.setattr(
        planner.subprocess, "run", _fake_git_runner(calls=calls)
    )
    result = planner._git_execution_provenance(
        PROJECT_ROOT, plan["parent_sampling_policy"]["sha256"]
    )
    assert result["tracked_worktree_clean"] is True
    assert ["git", "diff", "--quiet", "--"] in calls
    assert ["git", "diff", "--cached", "--quiet", "--"] in calls
    assert all(command[:2] != ["git", "status"] for command in calls)


def test_current_or_future_descendant_publication_gate_is_not_exact_head(plan):
    result = planner._git_execution_provenance(
        PROJECT_ROOT, plan["parent_sampling_policy"]["sha256"]
    )
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert result["execution_git_commit"] == actual_head
    assert result["plan_authoritative_base_is_ancestor"] is True
    assert result["tracked_worktree_clean"] is True


def test_current_intake_count_and_statuses(plan):
    source = plan["authoritative_inputs"]["acquisition_manifest"]
    assert source["source_count"] == 271
    assert source["statuses"] == {
        "eligible": 223,
        "extracted": 47,
        "hash_verified": 1,
    }
    assert planner.sha256_file(INTAKE_MANIFEST) == source["sha256"]


def test_historical_batch_counts_and_yields(audit):
    rows = audit["yield_history.json"]["batches"]
    expected = {
        "pilot-20261025": (24, 24, 0, 0, 1.0),
        "scale-20261026": (50, 39, 11, 0, 0.78),
        "scale-20261027": (53, 41, 12, 0, 41 / 53),
        "scale-20261028": (53, 47, 6, 0, 47 / 53),
        "scale-20261029": (41, 35, 5, 1, 35 / 41),
        "scale-20261030": (50, 37, 13, 0, 0.74),
    }
    assert len(rows) == len(expected)
    for row in rows:
        acquired, eligible, extracted, hashed, expected_yield = expected[
            row["batch_id"]
        ]
        assert (
            row["acquired"],
            row["eligible"],
            row["extracted"],
            row["hash_verified"],
        ) == (acquired, eligible, extracted, hashed)
        assert row["eligible_yield"] == pytest.approx(expected_yield)


def test_scale_and_all_history_aggregates(audit):
    history = audit["yield_history.json"]
    scale = history["scale_aggregate"]
    overall = history["all_history_aggregate"]
    assert (scale["eligible"], scale["acquired"]) == (199, 247)
    assert scale["yield"] == pytest.approx(0.8056680162)
    assert (overall["eligible"], overall["acquired"]) == (223, 271)
    assert overall["yield"] == pytest.approx(0.8228782288)


def test_wilson_95_lower_bound_is_reproduced(audit):
    interval = audit["yield_history.json"]["scale_aggregate"]["wilson_95"]
    assert interval["confidence_level"] == 0.95
    assert interval["z"] == 1.96
    assert interval["lower"] == pytest.approx(0.7517961661, abs=1e-10)


def test_minimum_scale_and_conservative_yield_are_074(audit):
    history = audit["yield_history.json"]
    assert history["scale_aggregate"]["minimum_completed_batch_yield"] == 0.74
    assert history["conservative_planning_yield"] == 0.74


def test_pilot_is_excluded_from_primary_estimator(audit):
    history = audit["yield_history.json"]
    assert history["primary_yield_history_scope"] == "SCALE_BATCHES_ONLY"
    assert history["pilot_excluded_from_primary_estimator"] is True
    assert history["scale_aggregate"]["acquired"] == 271 - 24


def test_raw_count_hard_floor_is_37(audit):
    scenarios = audit["batch_size_scenarios.json"]
    assert scenarios["raw_count_hard_floor"] == 37
    assert planner.raw_count_hard_floor(27, 0.74) == 37


def test_deficits_remain_distinct_and_are_not_added(audit):
    scarcity = audit["scarcity_constraints.json"]
    assert scarcity["deficits"] == {
        "overall_eligible_deficit": 27,
        "land_2024_compatible_deficit": 10,
        "hall_union_deficit": 13,
        "family_max_flow_shortage": 13,
    }
    assert scarcity["deficits_overlap"] is True
    assert scarcity["deficits_added_for_raw_target"] is False
    assert audit["batch_size_scenarios.json"]["raw_count_hard_floor"] != 27 + 13


def test_scenario_calculations_are_deterministic_with_floor_rounding(audit):
    payload = audit["batch_size_scenarios.json"]
    assert payload["rounding_rule"] == "FLOOR"
    expected_conservative = {
        37: (27, 250, 0),
        40: (29, 252, 2),
        45: (33, 256, 6),
        50: (37, 260, 10),
    }
    for row in payload["scenarios"]:
        projection = row["projections"]["conservative"]
        assert (
            projection["projected_eligible_additions"],
            projection["projected_final_eligible_frame"],
            projection["selection_reserve_beyond_250"],
        ) == expected_conservative[row["raw_acquisition_count"]]
        assert projection["rounding"] == "FLOOR"
    rebuilt = planner.build_acquisition_plan_audit(PROJECT_ROOT, PLAN_PATH)
    assert rebuilt == audit


def test_operational_batch_history_and_decision_rule(audit):
    decision = audit["acquisition_plan.json"]["decision"]
    assert decision["operational_reference_rule"] == (
        "HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN"
    )
    assert decision["completed_scale_batch_sizes"] == [50, 53, 53, 41, 50]
    assert decision["historical_scale_median"] == 50
    assert decision["final_recommended_raw_batch_target"] == 50
    assert all(decision["operational_target_qualifications"].values())
    row41 = next(
        row for row in decision["historical_observed_batch_diagnostics"]
        if row["raw_acquisition_count"] == 41
    )
    assert row41 == {
        "raw_acquisition_count": 41,
        "conservative_expected_eligible_additions": 30,
        "projected_final_eligible": 253,
        "projected_selection_reserve": 3,
        "selected_under_operational_reference_rule": False,
        "selection_explanation": (
            "NOT_SELECTED_RULE_USES_COMPLETED_SCALE_MEDIAN_NOT_MINIMUM_"
            "OBSERVED_BATCH"
        ),
    }


def test_operational_target_is_derived_from_median_not_literal_50(audit):
    scenarios = audit["batch_size_scenarios.json"]["scenarios"]
    derived = planner.decide_raw_target(
        scenarios,
        37,
        [41, 46, 53],
        current_eligible=223,
        target_eligible=250,
        conservative_yield=0.74,
    )
    assert derived["historical_scale_median"] == 46
    assert derived["final_recommended_raw_batch_target"] == 46
    assert derived["conservative_expected_eligible_additions"] == 34
    assert derived["projected_final_eligible"] == 257
    assert derived["projected_selection_reserve"] == 7


def test_family_specific_raw_yield_is_uncalibrated(audit):
    calibration = audit["scarcity_constraints.json"][
        "family_specific_raw_yield_calibration"
    ]
    assert calibration["status"] == "FAMILY_SPECIFIC_RAW_YIELD_UNCALIBRATED"
    assert calibration["scale_source_count"] == 247
    assert calibration["sources_with_explicit_pre_acquisition_family_intent"] == 0
    assert calibration["reliable_denominator_available"] is False
    assert calibration["post_review_family_composition_is_not_causal_yield"] is True


def test_historical_land_and_hall_composition_is_descriptive(audit):
    rows = audit["scarcity_constraints.json"][
        "historical_eligible_post_review_composition"
    ]
    expected = {
        "pilot-20261025": (3, 12),
        "scale-20261026": (2, 22),
        "scale-20261027": (8, 18),
        "scale-20261028": (1, 17),
        "scale-20261029": (1, 12),
        "scale-20261030": (10, 31),
    }
    for row in rows:
        assert (
            row["land_2024_compatible"], row["hall_union_compatible"]
        ) == expected[row["batch_id"]]
        assert row["role"].endswith("NOT_ACQUISITION_YIELD")


def test_recommended_batch_target_and_conservative_reserve(audit):
    plan = audit["acquisition_plan.json"]
    assert plan["raw_count_hard_floor"] == 37
    assert plan["final_recommended_raw_batch_target"] == 50
    assert plan["conservative_expected_eligible_additions"] == 37
    assert plan["projected_final_eligible"] == 260
    assert plan["projected_selection_reserve"] == 10


def test_machine_readable_outputs_do_not_use_ambiguous_land3_label(audit, plan):
    assert "LAND3" not in json.dumps(audit, sort_keys=True)
    assert "LAND3" not in json.dumps(plan, sort_keys=True)


def test_targeting_priorities_do_not_claim_confirmed_family_or_raw_quota(audit):
    targeting = audit["acquisition_plan.json"]["targeting"]
    assert targeting["priority_order"] == [
        "LAND_2024_TARGETED_DISCOVERY",
        "LAND_HALL_TARGETED_DISCOVERY",
    ]
    assert targeting["confirmed_family_claim_before_review"] is False
    assert targeting["raw_family_quota"] is None


def test_all_post_review_close_gates_can_pass(plan):
    result = planner.evaluate_close_gates(
        _close_metrics(), plan["post_review_close_gates"]
    )
    assert result["all_close_gates_pass"] is True
    assert result["raw_target_reached_is_not_a_close_gate"] is True


def test_total_eligible_249_keeps_acquisition_open(plan):
    result = planner.evaluate_close_gates(
        _close_metrics(final_eligible_count=249), plan["post_review_close_gates"]
    )
    assert result["checks"]["final_eligible_count"] is False
    assert result["all_close_gates_pass"] is False


def test_land_2024_34_keeps_acquisition_open(plan):
    result = planner.evaluate_close_gates(
        _close_metrics(land_2024_compatible=34),
        plan["post_review_close_gates"],
    )
    assert result["checks"]["land_2024_compatible"] is False
    assert result["all_close_gates_pass"] is False


def test_max_flow_below_185_keeps_acquisition_open(plan):
    result = planner.evaluate_close_gates(
        _close_metrics(family_maximum_filled_minimum_slots=184),
        plan["post_review_close_gates"],
    )
    assert result["checks"]["family_max_flow"] is False
    assert result["all_close_gates_pass"] is False


def test_remaining_hall_shortage_keeps_acquisition_open(plan):
    result = planner.evaluate_close_gates(
        _close_metrics(hall_bottleneck_shortage=1),
        plan["post_review_close_gates"],
    )
    assert result["checks"]["hall_bottleneck_shortage"] is False
    assert result["all_close_gates_pass"] is False


def test_raw_target_alone_cannot_close_acquisition(plan):
    metrics = _close_metrics(complete_cohort_reviewed=False)
    metrics["raw_target_reached"] = True
    result = planner.evaluate_close_gates(metrics, plan["post_review_close_gates"])
    assert result["checks"]["complete_cohort_reviewed"] is False
    assert result["all_close_gates_pass"] is False
    assert result["raw_target_reached_is_not_a_close_gate"] is True


def test_complete_cohort_before_review_rule(plan):
    cohort = plan["cohort_review"]
    assert cohort["identify_complete_cohort_before_processing"] is True
    assert cohort["review_complete_processable_cohort_before_promotion"] is True
    assert cohort["within_batch_outcome_adaptive_downloads_forbidden"] is True


def test_top_up_requires_separate_identity_and_new_plan(plan):
    top_up = plan["top_up"]
    assert top_up["precreate_now"] is False
    assert top_up["silently_enlarge_batch_2d_b"] is False
    assert top_up["separate_batch_identity_required"] is True
    assert top_up["new_plan_must_use_residual_post_review_deficits"] is True


def test_sampling_seed_and_selection_remain_untouched(audit):
    state = audit["acquisition_plan.json"]["protected_state"]
    execution = audit["acquisition_plan.json"]["execution_state"]
    assert state["sampling_seed"] is None
    assert state["selection"] == "NOT_STARTED"
    assert state["selected_query_gt"] == [0, 0, 0]
    assert execution["selection_performed"] is False
    assert execution["sampling_seed_generated"] is False


def test_no_acquisition_network_model_retrieval_or_ocr_execution(audit):
    execution = audit["acquisition_plan.json"]["execution_state"]
    assert execution == {
        "acquisition_executed": False,
        "sources_downloaded": 0,
        "sources_registered": 0,
        "extraction_or_ocr_executed": False,
        "review_or_promotion_executed": False,
        "selection_performed": False,
        "sampling_seed_generated": False,
        "query_or_gt_generated": False,
        "models_executed": [],
        "retrievers_executed": [],
        "network_accessed": False,
    }


def test_no_network_model_retrieval_or_ocr_imports():
    source = Path(planner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports.isdisjoint(
        {
            "faiss",
            "google",
            "pdfplumber",
            "pytesseract",
            "rank_bm25",
            "requests",
            "sentence_transformers",
            "socket",
            "torch",
            "transformers",
            "urllib",
        }
    )


def test_build_audit_does_not_mutate_authoritative_inputs():
    paths = [
        INTAKE_MANIFEST,
        BENCHMARK_DIR / "benchmark_manifest.json",
        BENCHMARK_DIR / "sampling_manifest.json",
        BENCHMARK_DIR / "preliminary_stratification.json",
        PROJECT_ROOT / "backend/scripts/Evaluation/configs/independent_sampling_policy_v1.yaml",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    planner.build_acquisition_plan_audit(PROJECT_ROOT, PLAN_PATH)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before


def test_plan_config_rejects_arbitrary_raw_family_quota(plan):
    changed = copy.deepcopy(plan)
    changed["targeting"]["raw_family_quota"] = {"LAND_2024": 13}
    with pytest.raises(planner.AcquisitionPlanningError, match="Targeting"):
        planner.validate_plan_config(changed)


def test_plan_config_rejects_added_deficits(plan):
    changed = copy.deepcopy(plan)
    changed["current_constraints"]["add_deficits_for_raw_target"] = True
    with pytest.raises(planner.AcquisitionPlanningError, match="constraints"):
        planner.validate_plan_config(changed)


def test_strict_json_is_deterministic_and_nonfinite_fails(tmp_path, audit):
    assert planner.canonical_json_fingerprint(audit) == planner.canonical_json_fingerprint(
        planner.build_acquisition_plan_audit(PROJECT_ROOT, PLAN_PATH)
    )
    json.dumps(audit, sort_keys=True, allow_nan=False)
    bad = tmp_path / "bad.json"
    bad.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(sampling_policy.SamplingPolicyError, match="Non-finite"):
        planner.strict_json_load(bad)


def test_atomic_audit_publication_contains_no_action_or_selection(audit):
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runs = PROJECT_ROOT / "backend/data/runs"
    with tempfile.TemporaryDirectory(prefix="phase2b-test-", dir=runs) as temp:
        result = planner.publish_audit(PROJECT_ROOT, PLAN_PATH, Path(temp), audit)
        assert result.is_dir()
        assert not result.with_name(result.name + ".partial").exists()
        assert {path.name for path in result.iterdir()} == {
            "yield_history.json",
            "batch_size_scenarios.json",
            "scarcity_constraints.json",
            "acquisition_plan.json",
            "run_manifest.json",
        }
        manifest = planner.strict_json_load(result / "run_manifest.json")
        assert manifest["git_commit"] == actual_head
        assert manifest["execution_git_commit"] == actual_head
        assert manifest["plan_authoritative_base_commit"] == (
            "a5ebb4a03f35a4c33596faf93b8d5fa82d6d076f"
        )
        assert manifest["plan_authoritative_base_is_ancestor"] is True
        assert manifest["tracked_worktree_clean"] is True
        assert manifest["parent_sampling_policy_checkpoint_commit"] == (
            "ef9c3fbefc45d76f0568fb0c3623270c77ed0d27"
        )
        assert manifest["parent_sampling_policy_sha256"] == (
            "a9926eafe2ee4544f75c194e1de435eeeec5121076d91a6a819424e305b2c91f"
        )
        assert manifest["acquisition_executed"] is False
        assert manifest["sources_downloaded"] == 0
        assert manifest["sources_registered"] == 0
        assert manifest["selection_performed"] is False
        assert manifest["sampling_seed_generated"] is False
        assert manifest["query_or_gt_generated"] is False
        assert manifest["models_executed"] == []
        assert manifest["retrievers_executed"] == []
        serialized = " ".join(path.read_text() for path in result.iterdir())
        assert "selected_case_ids" not in serialized
