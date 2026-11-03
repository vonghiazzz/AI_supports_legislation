"""Freeze a read-only Phase-2B acquisition plan for the independent benchmark.

The planner derives historical yield, scarcity constraints, and post-review
close gates.  It never downloads, registers, extracts, reviews, promotes, or
selects a source and never executes OCR, retrieval, models, query generation,
or ground-truth generation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .family_quota import compatible_families
from .sampling_policy import (
    POLICY_ID,
    POLICY_STATUS,
    build_policy_audit,
    canonical_json_fingerprint,
    sha256_file,
    strict_json_load,
    strict_json_write,
    strict_yaml_load,
    validate_policy,
)


PLAN_ID = "independent_acquisition_plan_v1"
PLAN_STATUS = "ACQUISITION_PLAN_FROZEN_NOT_EXECUTED"
PROTOCOL_ID = "independent_acquisition_plan_audit_v1"
BATCH_LABEL = "BATCH_2D_B"
PLAN_AUTHORITATIVE_BASE_COMMIT = "a5ebb4a03f35a4c33596faf93b8d5fa82d6d076f"
PARENT_SAMPLING_POLICY_CHECKPOINT_COMMIT = (
    "ef9c3fbefc45d76f0568fb0c3623270c77ed0d27"
)
DEFAULT_PLAN_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/independent_acquisition_plan_v1.yaml"
)
DEFAULT_RUNS_RELATIVE = Path("backend/data/runs")
FAMILY_YIELD_UNCALIBRATED = "FAMILY_SPECIFIC_RAW_YIELD_UNCALIBRATED"
BATCH_ID_PATTERN = re.compile(r"^(pilot|scale)-(\d{8})-\d+$")
INTENT_FIELDS = {
    "pre_acquisition_target_family_intent",
    "acquisition_target_family",
    "target_family_intent",
    "targeting_intent_recorded_before_acquisition",
}


class AcquisitionPlanningError(ValueError):
    """Raised when Phase-2B planning inputs violate the frozen protocol."""


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise AcquisitionPlanningError(f"{label} differs from frozen plan")


def validate_plan_config(plan: dict) -> None:
    if plan.get("plan_id") != PLAN_ID or plan.get("plan_status") != PLAN_STATUS:
        raise AcquisitionPlanningError("Wrong acquisition plan identity/status")
    if plan.get("batch_label") != BATCH_LABEL:
        raise AcquisitionPlanningError("Wrong batch label")
    if plan.get("benchmark_id") != "retrieval_independent_v1":
        raise AcquisitionPlanningError("Wrong benchmark identity")
    if (
        plan.get("plan_authoritative_base_commit")
        != PLAN_AUTHORITATIVE_BASE_COMMIT
    ):
        raise AcquisitionPlanningError("Wrong plan authoritative base commit")
    parent = plan.get("parent_sampling_policy", {})
    if (
        parent.get("policy_id") != POLICY_ID
        or parent.get("policy_status") != POLICY_STATUS
        or parent.get("checkpoint_commit")
        != PARENT_SAMPLING_POLICY_CHECKPOINT_COMMIT
        or parent.get("sha256")
        != "a9926eafe2ee4544f75c194e1de435eeeec5121076d91a6a819424e305b2c91f"
    ):
        raise AcquisitionPlanningError("Parent sampling policy identity changed")
    _exact(
        plan.get("required_state"),
        {
            "benchmark": "DRAFT",
            "sampling": "DRAFT",
            "selection": "NOT_STARTED",
            "sampling_seed": None,
            "selected_cases": 0,
            "query_records": 0,
            "ground_truth_records": 0,
        },
        "required state",
    )
    constraints = plan.get("current_constraints", {})
    expected_constraints = {
        "target_eligible_count": 250,
        "overall_eligible_deficit": 27,
        "required_family_slots": 185,
        "maximum_filled_family_slots": 172,
        "family_max_flow_shortage": 13,
        "land_2024_compatible": 25,
        "land_2024_minimum": 35,
        "land_2024_compatible_deficit": 10,
        "hall_bottleneck": {
            "families": ["BLDS_CORE", "LAND_2013", "LAND_2024"],
            "compatible_unique_sources": 112,
            "required_slots": 125,
            "hall_union_deficit": 13,
        },
        "stress_eligible": 87,
        "deficits_overlap": True,
        "add_deficits_for_raw_target": False,
    }
    _exact(constraints, expected_constraints, "current constraints")
    yield_policy = plan.get("yield_planning", {})
    if (
        yield_policy.get("primary_history_scope") != "SCALE_BATCHES_ONLY"
        or yield_policy.get("pilot_excluded_from_primary_estimator") is not True
        or yield_policy.get("confidence_level") != 0.95
        or yield_policy.get("wilson_z") != 1.96
        or yield_policy.get("scenario_eligible_rounding") != "FLOOR"
        or yield_policy.get("scenarios") != [37, 40, 45, 50]
    ):
        raise AcquisitionPlanningError("Yield-planning policy changed")
    decision = plan.get("batch_size_decision", {})
    expected_decision = {
        "operational_reference_rule": (
            "HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN"
        ),
        "qualifications": [
            "RAW_COUNT_EXCEEDS_HARD_FLOOR",
            "NONZERO_CONSERVATIVE_SELECTION_RESERVE",
            "RAW_COUNT_WITHIN_OBSERVED_COMPLETED_SCALE_RANGE",
        ],
        "raw_target_role": (
            "ACQUISITION_COHORT_SIZE_NOT_SCIENTIFIC_SUFFICIENCY"
        ),
    }
    if decision != expected_decision:
        raise AcquisitionPlanningError("Batch-size decision contract changed")
    targeting = plan.get("targeting", {})
    if (
        targeting.get("priority_order")
        != ["LAND_2024_TARGETED_DISCOVERY", "LAND_HALL_TARGETED_DISCOVERY"]
        or targeting.get("raw_family_quota") is not None
        or targeting.get("family_specific_raw_yield_calibration_status")
        != FAMILY_YIELD_UNCALIBRATED
        or targeting.get("confirmed_family_claim_before_review") is not False
    ):
        raise AcquisitionPlanningError("Targeting policy changed")
    gates = plan.get("post_review_close_gates", {})
    expected_gates = {
        "final_eligible_count_minimum": 250,
        "land_2024_compatible_minimum": 35,
        "family_maximum_filled_minimum_slots": 185,
        "family_shortage": 0,
        "hall_bottleneck_shortage": 0,
        "stress_eligible_minimum": 50,
        "development_overlap": 0,
        "duplicate_source_or_case_identity": 0,
        "benchmark_state": "DRAFT",
        "selection_status": "NOT_STARTED",
        "complete_cohort_reviewed": True,
        "all_required": True,
    }
    _exact(gates, expected_gates, "post-review close gates")
    cohort = plan.get("cohort_review", {})
    if not all(
        cohort.get(key) is True
        for key in (
            "identify_complete_cohort_before_processing",
            "review_complete_processable_cohort_before_promotion",
            "within_batch_outcome_adaptive_downloads_forbidden",
            "raw_target_reached_does_not_close_acquisition",
        )
    ):
        raise AcquisitionPlanningError("Complete-cohort rule changed")
    top_up = plan.get("top_up", {})
    if (
        top_up.get("precreate_now") is not False
        or top_up.get("silently_enlarge_batch_2d_b") is not False
        or top_up.get("separate_batch_identity_required") is not True
        or top_up.get("new_plan_must_use_residual_post_review_deficits") is not True
    ):
        raise AcquisitionPlanningError("Top-up rule changed")
    prohibitions = plan.get("execution_prohibitions", {})
    if not prohibitions or not all(value is True for value in prohibitions.values()):
        raise AcquisitionPlanningError("Execution prohibitions must all be active")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> dict[str, float]:
    if (
        not isinstance(successes, int)
        or isinstance(successes, bool)
        or not isinstance(trials, int)
        or isinstance(trials, bool)
        or successes < 0
        or trials <= 0
        or successes > trials
        or not isinstance(z, (int, float))
        or isinstance(z, bool)
        or not math.isfinite(z)
        or z <= 0
    ):
        raise AcquisitionPlanningError("Invalid Wilson interval inputs")
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return {
        "confidence_level": 0.95,
        "z": float(z),
        "lower": center - half_width,
        "upper": center + half_width,
    }


def acquisition_batch_id(acquisition_id: str) -> str:
    if not isinstance(acquisition_id, str):
        raise AcquisitionPlanningError("acquisition_id must be a string")
    match = BATCH_ID_PATTERN.fullmatch(acquisition_id)
    if match is None:
        raise AcquisitionPlanningError(f"Unsupported acquisition_id: {acquisition_id}")
    return f"{match.group(1)}-{match.group(2)}"


def derive_yield_history(sources: list[dict], z: float = 1.96) -> dict:
    if not isinstance(sources, list) or not sources:
        raise AcquisitionPlanningError("Acquisition history must be non-empty")
    source_ids = [source.get("source_id") for source in sources]
    acquisition_ids = [source.get("acquisition_id") for source in sources]
    if len(source_ids) != len(set(source_ids)) or len(acquisition_ids) != len(
        set(acquisition_ids)
    ):
        raise AcquisitionPlanningError("Acquisition history identities must be unique")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for source in sources:
        grouped[acquisition_batch_id(source.get("acquisition_id"))].append(source)
    rows = []
    for batch_id in sorted(grouped):
        records = grouped[batch_id]
        statuses = Counter(record.get("acquisition_status") for record in records)
        if set(statuses) - {"eligible", "extracted", "hash_verified"}:
            raise AcquisitionPlanningError("Unexpected acquisition status")
        acquired = len(records)
        eligible = statuses["eligible"]
        rows.append(
            {
                "batch_id": batch_id,
                "batch_type": batch_id.split("-", 1)[0],
                "acquired": acquired,
                "eligible": eligible,
                "extracted": statuses["extracted"],
                "hash_verified": statuses["hash_verified"],
                "eligible_yield": eligible / acquired,
            }
        )
    scale = [row for row in rows if row["batch_type"] == "scale"]
    pilot = [row for row in rows if row["batch_type"] == "pilot"]
    if len(pilot) != 1 or not scale:
        raise AcquisitionPlanningError("Expected one pilot and completed scale history")
    scale_acquired = sum(row["acquired"] for row in scale)
    scale_eligible = sum(row["eligible"] for row in scale)
    overall_acquired = sum(row["acquired"] for row in rows)
    overall_eligible = sum(row["eligible"] for row in rows)
    interval = wilson_interval(scale_eligible, scale_acquired, z)
    minimum_scale_yield = min(row["eligible_yield"] for row in scale)
    conservative = min(interval["lower"], minimum_scale_yield)
    return {
        "batches": rows,
        "primary_yield_history_scope": "SCALE_BATCHES_ONLY",
        "pilot_excluded_from_primary_estimator": True,
        "scale_aggregate": {
            "eligible": scale_eligible,
            "acquired": scale_acquired,
            "yield": scale_eligible / scale_acquired,
            "wilson_95": interval,
            "minimum_completed_batch_yield": minimum_scale_yield,
        },
        "all_history_aggregate": {
            "eligible": overall_eligible,
            "acquired": overall_acquired,
            "yield": overall_eligible / overall_acquired,
        },
        "conservative_yield_rule": (
            "MIN(SCALE_AGGREGATE_WILSON_95_LOWER, "
            "MIN_COMPLETED_SCALE_BATCH_YIELD)"
        ),
        "conservative_planning_yield": conservative,
    }


def raw_count_hard_floor(target_deficit: int, conservative_yield: float) -> int:
    if (
        not isinstance(target_deficit, int)
        or isinstance(target_deficit, bool)
        or target_deficit < 0
        or not isinstance(conservative_yield, (int, float))
        or isinstance(conservative_yield, bool)
        or not math.isfinite(conservative_yield)
        or conservative_yield <= 0
        or conservative_yield > 1
    ):
        raise AcquisitionPlanningError("Invalid raw hard-floor inputs")
    return math.ceil(target_deficit / conservative_yield)


def evaluate_scenarios(
    raw_counts: list[int],
    *,
    current_eligible: int,
    target_eligible: int,
    scale_point_yield: float,
    wilson_lower_yield: float,
    conservative_yield: float,
) -> list[dict]:
    if raw_counts != sorted(set(raw_counts)):
        raise AcquisitionPlanningError("Scenarios must be unique and ascending")
    estimators = {
        "scale_point_estimate": scale_point_yield,
        "wilson_95_lower": wilson_lower_yield,
        "conservative": conservative_yield,
    }
    rows = []
    for raw_count in raw_counts:
        projections = {}
        for name, value in estimators.items():
            additions = math.floor(raw_count * value)
            final = current_eligible + additions
            projections[name] = {
                "yield": value,
                "rounding": "FLOOR",
                "projected_eligible_additions": additions,
                "projected_final_eligible_frame": final,
                "selection_reserve_beyond_250": final - target_eligible,
            }
        rows.append({"raw_acquisition_count": raw_count, "projections": projections})
    return rows


def decide_raw_target(
    scenarios: list[dict],
    hard_floor: int,
    completed_scale_sizes: list[int],
    *,
    current_eligible: int,
    target_eligible: int,
    conservative_yield: float,
) -> dict:
    if (
        not completed_scale_sizes
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in completed_scale_sizes
        )
    ):
        raise AcquisitionPlanningError("Completed scale sizes are required")
    median_value = statistics.median(completed_scale_sizes)
    if not float(median_value).is_integer():
        raise AcquisitionPlanningError(
            "Completed-scale median must resolve to an integer raw count"
        )
    median_size = int(median_value)

    def conservative_projection(raw_count: int) -> dict[str, int]:
        additions = math.floor(raw_count * conservative_yield)
        projected_final = current_eligible + additions
        return {
            "raw_acquisition_count": raw_count,
            "conservative_expected_eligible_additions": additions,
            "projected_final_eligible": projected_final,
            "projected_selection_reserve": projected_final - target_eligible,
        }

    target_projection = conservative_projection(median_size)
    qualifications = {
        "raw_count_exceeds_hard_floor": median_size > hard_floor,
        "nonzero_conservative_selection_reserve": (
            target_projection["projected_selection_reserve"] > 0
        ),
        "raw_count_within_observed_completed_scale_range": (
            min(completed_scale_sizes) <= median_size <= max(completed_scale_sizes)
        ),
    }
    if not all(qualifications.values()):
        raise AcquisitionPlanningError(
            "Historical completed-scale median fails operational qualifications"
        )

    scenario_diagnostics = []
    for scenario in scenarios:
        raw = scenario["raw_acquisition_count"]
        reserve = scenario["projections"]["conservative"][
            "selection_reserve_beyond_250"
        ]
        scenario_diagnostics.append(
            {
                "raw_acquisition_count": raw,
                "exceeds_hard_floor": raw > hard_floor,
                "has_positive_conservative_reserve": reserve > 0,
                "equals_operational_reference_median": raw == median_size,
            }
        )

    historical_diagnostics = []
    for raw in sorted(set(completed_scale_sizes)):
        row = conservative_projection(raw)
        row["selected_under_operational_reference_rule"] = raw == median_size
        if raw < median_size:
            row["selection_explanation"] = (
                "NOT_SELECTED_RULE_USES_COMPLETED_SCALE_MEDIAN_NOT_MINIMUM_"
                "OBSERVED_BATCH"
            )
        elif raw > median_size:
            row["selection_explanation"] = (
                "NOT_SELECTED_RULE_USES_COMPLETED_SCALE_MEDIAN_NOT_LARGER_"
                "OBSERVED_BATCH"
            )
        else:
            row["selection_explanation"] = (
                "SELECTED_HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN"
            )
        historical_diagnostics.append(row)

    return {
        "operational_reference_rule": (
            "HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN"
        ),
        "completed_scale_batch_sizes": completed_scale_sizes,
        "historical_scale_median": median_size,
        "operational_target_qualifications": qualifications,
        "historical_observed_batch_diagnostics": historical_diagnostics,
        "planning_scenario_diagnostics": scenario_diagnostics,
        "final_recommended_raw_batch_target": median_size,
        "conservative_expected_eligible_additions": target_projection[
            "conservative_expected_eligible_additions"
        ],
        "projected_final_eligible": target_projection[
            "projected_final_eligible"
        ],
        "projected_selection_reserve": target_projection[
            "projected_selection_reserve"
        ],
    }


def family_yield_calibration(sources: list[dict]) -> dict:
    scale = [
        source
        for source in sources
        if acquisition_batch_id(source["acquisition_id"]).startswith("scale-")
    ]
    sources_with_any_intent_field = sum(
        bool(INTENT_FIELDS.intersection(source)) for source in scale
    )
    status = FAMILY_YIELD_UNCALIBRATED
    return {
        "status": status,
        "scale_source_count": len(scale),
        "sources_with_explicit_pre_acquisition_family_intent": (
            sources_with_any_intent_field
        ),
        "reliable_denominator_available": False,
        "post_review_family_composition_is_not_causal_yield": True,
        "planning_effect": "POST_REVIEW_SUCCESS_GATES_NOT_RAW_FAMILY_QUOTAS",
    }


def historical_family_composition(records: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[acquisition_batch_id(record["acquisition_id"])].append(record)
    rows = []
    hall = {"BLDS_CORE", "LAND_2013", "LAND_2024"}
    for batch_id in sorted(grouped):
        batch = grouped[batch_id]
        memberships = [set(compatible_families(record)) for record in batch]
        rows.append(
            {
                "batch_id": batch_id,
                "eligible_post_review_sources": len(batch),
                "land_2024_compatible": sum(
                    "LAND_2024" in values for values in memberships
                ),
                "hall_union_compatible": sum(
                    bool(hall.intersection(values)) for values in memberships
                ),
                "role": "DESCRIPTIVE_POST_REVIEW_COMPOSITION_NOT_ACQUISITION_YIELD",
            }
        )
    return rows


def evaluate_close_gates(metrics: Mapping[str, Any], gates: Mapping[str, Any]) -> dict:
    checks = {
        "final_eligible_count": (
            metrics.get("final_eligible_count", -1)
            >= gates["final_eligible_count_minimum"]
        ),
        "land_2024_compatible": (
            metrics.get("land_2024_compatible", -1)
            >= gates["land_2024_compatible_minimum"]
        ),
        "family_max_flow": (
            metrics.get("family_maximum_filled_minimum_slots")
            == gates["family_maximum_filled_minimum_slots"]
        ),
        "family_shortage": (
            metrics.get("family_shortage") == gates["family_shortage"]
        ),
        "hall_bottleneck_shortage": (
            metrics.get("hall_bottleneck_shortage")
            == gates["hall_bottleneck_shortage"]
        ),
        "stress_eligible": (
            metrics.get("stress_eligible", -1) >= gates["stress_eligible_minimum"]
        ),
        "development_overlap": (
            metrics.get("development_overlap") == gates["development_overlap"]
        ),
        "duplicate_source_or_case_identity": (
            metrics.get("duplicate_source_or_case_identity")
            == gates["duplicate_source_or_case_identity"]
        ),
        "benchmark_state": metrics.get("benchmark_state")
        == gates["benchmark_state"],
        "selection_status": metrics.get("selection_status")
        == gates["selection_status"],
        "complete_cohort_reviewed": metrics.get("complete_cohort_reviewed")
        is gates["complete_cohort_reviewed"],
    }
    return {
        "checks": checks,
        "all_close_gates_pass": all(checks.values()),
        "raw_target_reached_is_not_a_close_gate": True,
    }


def _load_and_validate_inputs(project_root: Path, plan: dict) -> dict[str, Any]:
    parent = plan["parent_sampling_policy"]
    policy_path = project_root / parent["path"]
    if sha256_file(policy_path) != parent["sha256"]:
        raise AcquisitionPlanningError("Parent sampling policy bytes changed")
    policy = strict_yaml_load(policy_path)
    validate_policy(policy)
    policy_audit = build_policy_audit(project_root, policy_path)
    constraints = policy_audit["acquisition_constraints.json"]
    family = policy_audit["family_feasibility.json"]
    stress = policy_audit["stress_eligibility.json"]
    expected = plan["current_constraints"]
    if (
        constraints["target_count_deficit"] != expected["overall_eligible_deficit"]
        or family["required_slots"] != expected["required_family_slots"]
        or family["maximum_filled_minimum_slots"]
        != expected["maximum_filled_family_slots"]
        or family["shortage"] != expected["family_max_flow_shortage"]
        or family["compatible_membership_counts"]["LAND_2024"]
        != expected["land_2024_compatible"]
        or family["strongest_hall_bottleneck"]
        != {
            "families": expected["hall_bottleneck"]["families"],
            "compatible_source_count": expected["hall_bottleneck"][
                "compatible_unique_sources"
            ],
            "required_slots": expected["hall_bottleneck"]["required_slots"],
            "shortage": expected["hall_bottleneck"]["hall_union_deficit"],
        }
        or stress["eligible_source_count"] != expected["stress_eligible"]
    ):
        raise AcquisitionPlanningError("Phase-2A constraint validation failed")
    source = plan["authoritative_inputs"]["acquisition_manifest"]
    acquisition_path = project_root / source["path"]
    if sha256_file(acquisition_path) != source["sha256"]:
        raise AcquisitionPlanningError("Acquisition manifest bytes changed")
    acquisition = strict_json_load(acquisition_path)
    sources = acquisition.get("sources")
    if not isinstance(sources, list) or len(sources) != source["source_count"]:
        raise AcquisitionPlanningError("Acquisition source count changed")
    statuses = Counter(item.get("acquisition_status") for item in sources)
    if dict(statuses) != source["statuses"]:
        raise AcquisitionPlanningError("Acquisition status counts changed")
    frame_source = plan["authoritative_inputs"]["preliminary_stratification"]
    preliminary = strict_json_load(project_root / frame_source["path"])
    if (
        preliminary.get("source_count") != frame_source["source_count"]
        or preliminary.get("source_inventory_fingerprint")
        != frame_source["source_inventory_fingerprint"]
        or preliminary.get("records_fingerprint")
        != frame_source["records_fingerprint"]
        or canonical_json_fingerprint(preliminary.get("records"))
        != frame_source["records_fingerprint"]
    ):
        raise AcquisitionPlanningError("Preliminary frame identity changed")
    benchmark_path = (
        project_root
        / "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
    )
    sampling_path = (
        project_root
        / "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
    )
    benchmark = strict_json_load(benchmark_path)
    sampling = strict_json_load(sampling_path)
    state = plan["required_state"]
    counts = benchmark.get("current_counts", {})
    if (
        benchmark.get("state") != state["benchmark"]
        or sampling.get("state") != state["sampling"]
        or sampling.get("selection_status") != state["selection"]
        or sampling.get("sampling_seed") is not state["sampling_seed"]
        or counts.get("selected_cases") != state["selected_cases"]
        or counts.get("query_records") != state["query_records"]
        or counts.get("ground_truth_records") != state["ground_truth_records"]
    ):
        raise AcquisitionPlanningError("Protected benchmark/sampling state changed")
    return {
        "policy": policy,
        "policy_audit": policy_audit,
        "acquisition": acquisition,
        "preliminary": preliminary,
        "benchmark": benchmark,
        "sampling": sampling,
        "source_paths": {
            "parent_sampling_policy": policy_path,
            "acquisition_manifest": acquisition_path,
            "preliminary_stratification": project_root / frame_source["path"],
            "benchmark_manifest": benchmark_path,
            "sampling_manifest": sampling_path,
        },
    }


def build_acquisition_plan_audit(project_root: Path, plan_path: Path) -> dict[str, dict]:
    project_root = Path(project_root).resolve()
    plan_path = Path(plan_path).resolve()
    plan = strict_yaml_load(plan_path)
    validate_plan_config(plan)
    inputs = _load_and_validate_inputs(project_root, plan)
    sources = inputs["acquisition"]["sources"]
    history = derive_yield_history(sources, plan["yield_planning"]["wilson_z"])
    conservative = history["conservative_planning_yield"]
    hard_floor = raw_count_hard_floor(
        plan["current_constraints"]["overall_eligible_deficit"], conservative
    )
    scenarios = evaluate_scenarios(
        plan["yield_planning"]["scenarios"],
        current_eligible=plan["authoritative_inputs"]["preliminary_stratification"][
            "source_count"
        ],
        target_eligible=plan["current_constraints"]["target_eligible_count"],
        scale_point_yield=history["scale_aggregate"]["yield"],
        wilson_lower_yield=history["scale_aggregate"]["wilson_95"]["lower"],
        conservative_yield=conservative,
    )
    scale_sizes = [
        row["acquired"]
        for row in history["batches"]
        if row["batch_type"] == "scale"
    ]
    current_eligible = plan["authoritative_inputs"]["preliminary_stratification"][
        "source_count"
    ]
    target_eligible = plan["current_constraints"]["target_eligible_count"]
    decision = decide_raw_target(
        scenarios,
        hard_floor,
        scale_sizes,
        current_eligible=current_eligible,
        target_eligible=target_eligible,
        conservative_yield=conservative,
    )
    if hard_floor != 37:
        raise AcquisitionPlanningError("Derived batch-size decision changed")
    calibration = family_yield_calibration(sources)
    if calibration["status"] != plan["targeting"][
        "family_specific_raw_yield_calibration_status"
    ]:
        raise AcquisitionPlanningError("Family-yield calibration status changed")
    composition = historical_family_composition(inputs["preliminary"]["records"])
    scarcity = {
        "deficits": {
            "overall_eligible_deficit": plan["current_constraints"][
                "overall_eligible_deficit"
            ],
            "land_2024_compatible_deficit": plan["current_constraints"][
                "land_2024_compatible_deficit"
            ],
            "hall_union_deficit": plan["current_constraints"]["hall_bottleneck"][
                "hall_union_deficit"
            ],
            "family_max_flow_shortage": plan["current_constraints"][
                "family_max_flow_shortage"
            ],
        },
        "deficits_overlap": True,
        "deficits_added_for_raw_target": False,
        "hall_bottleneck": plan["current_constraints"]["hall_bottleneck"],
        "stress_eligible": plan["current_constraints"]["stress_eligible"],
        "stress_raw_availability_is_current_blocker": False,
        "family_specific_raw_yield_calibration": calibration,
        "historical_eligible_post_review_composition": composition,
        "composition_role": "DESCRIPTIVE_NOT_CAUSAL_ACQUISITION_YIELD",
    }
    acquisition_plan = {
        "plan_id": PLAN_ID,
        "plan_status": PLAN_STATUS,
        "batch_label": BATCH_LABEL,
        "raw_count_hard_floor": hard_floor,
        "final_recommended_raw_batch_target": decision[
            "final_recommended_raw_batch_target"
        ],
        "conservative_expected_eligible_additions": decision[
            "conservative_expected_eligible_additions"
        ],
        "projected_final_eligible": decision["projected_final_eligible"],
        "projected_selection_reserve": decision[
            "projected_selection_reserve"
        ],
        "decision": decision,
        "targeting": plan["targeting"],
        "land_2024_post_review_requirement": (
            "AT_LEAST_35_COMPATIBLE_TOTAL_AND_10_NET_NEW_FROM_CURRENT_FRAME"
        ),
        "hall_post_review_requirement": (
            "MAX_FLOW_185_SHORTAGE_0_AND_NO_HALL_SHORTAGE"
        ),
        "post_review_close_gates": plan["post_review_close_gates"],
        "cohort_review": plan["cohort_review"],
        "top_up": plan["top_up"],
        "leakage_firewall": plan["leakage_firewall"],
        "execution_prohibitions": plan["execution_prohibitions"],
        "execution_state": {
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
        },
        "protected_state": {
            "intake_sources": 271,
            "eligible": 223,
            "extracted": 47,
            "hash_verified": 1,
            "benchmark": "DRAFT",
            "sampling": "DRAFT",
            "selection": "NOT_STARTED",
            "sampling_seed": None,
            "selected_query_gt": [0, 0, 0],
        },
    }
    return {
        "yield_history.json": history,
        "batch_size_scenarios.json": {
            "rounding_rule": "FLOOR",
            "current_eligible": 223,
            "target_eligible": 250,
            "raw_count_hard_floor": hard_floor,
            "scenarios": scenarios,
        },
        "scarcity_constraints.json": scarcity,
        "acquisition_plan.json": acquisition_plan,
    }


def _git_execution_provenance(
    project_root: Path, parent_policy_sha256: str
) -> dict[str, Any]:
    """Validate descendant execution provenance while allowing untracked files."""

    project_root = Path(project_root).resolve()
    execution_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PLAN_AUTHORITATIVE_BASE_COMMIT,
            execution_commit,
        ],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode == 1:
        raise AcquisitionPlanningError(
            "Plan authoritative base is not an ancestor of execution HEAD"
        )
    if ancestor.returncode != 0:
        raise AcquisitionPlanningError("Unable to validate plan execution lineage")
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged tracked"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged tracked"),
    ):
        result = subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 1:
            raise AcquisitionPlanningError(
                f"Acquisition-plan audit requires clean {label} repository state"
            )
        if result.returncode != 0:
            raise AcquisitionPlanningError(
                f"Unable to validate {label} repository state"
            )
    return {
        "execution_git_commit": execution_commit,
        "plan_authoritative_base_commit": PLAN_AUTHORITATIVE_BASE_COMMIT,
        "plan_authoritative_base_is_ancestor": True,
        "tracked_worktree_clean": True,
        "parent_sampling_policy_checkpoint_commit": (
            PARENT_SAMPLING_POLICY_CHECKPOINT_COMMIT
        ),
        "parent_sampling_policy_sha256": parent_policy_sha256,
    }


def publish_audit(
    project_root: Path,
    plan_path: Path,
    output_root: Path,
    structured: Mapping[str, dict],
) -> Path:
    project_root = Path(project_root).resolve()
    plan_path = Path(plan_path).resolve()
    output_root = Path(output_root).resolve()
    try:
        output_root.relative_to(project_root)
    except ValueError as exc:
        raise AcquisitionPlanningError("Audit output must remain inside project root") from exc
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"
    partial = output_root / f"{run_id}.partial"
    final = output_root / run_id
    if partial.exists() or final.exists():
        raise AcquisitionPlanningError("Unsafe audit output path")
    output_root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        for name, payload in structured.items():
            strict_json_write(partial / name, payload)
        output_fingerprints = {
            path.name: sha256_file(path)
            for path in sorted(partial.iterdir(), key=lambda item: item.name)
        }
        plan_config = strict_yaml_load(plan_path)
        parent_policy_sha256 = plan_config["parent_sampling_policy"]["sha256"]
        provenance = _git_execution_provenance(
            project_root, parent_policy_sha256
        )
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": provenance["execution_git_commit"],
            **provenance,
            "plan_id": PLAN_ID,
            "plan_status": PLAN_STATUS,
            "plan_config_sha256": sha256_file(plan_path),
            "planner_source_sha256": sha256_file(Path(__file__)),
            "audit_fingerprint": canonical_json_fingerprint(dict(structured)),
            "output_file_sha256": output_fingerprints,
            "acquisition_executed": False,
            "sources_downloaded": 0,
            "sources_registered": 0,
            "extraction_or_ocr_executed": False,
            "selection_performed": False,
            "sampling_seed_generated": False,
            "query_or_gt_generated": False,
            "models_executed": [],
            "retrievers_executed": [],
            "network_accessed": False,
        }
        strict_json_write(partial / "run_manifest.json", manifest)
        if {path.name for path in partial.iterdir()} != set(structured) | {
            "run_manifest.json"
        }:
            raise AcquisitionPlanningError("Incomplete acquisition-plan audit")
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def run(
    *,
    project_root: Path | None = None,
    plan_path: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    plan = (
        Path(plan_path).resolve()
        if plan_path is not None
        else root / DEFAULT_PLAN_RELATIVE
    )
    output = (
        Path(output_root).resolve()
        if output_root is not None
        else root / DEFAULT_RUNS_RELATIVE
    )
    plan_config = strict_yaml_load(plan)
    validate_plan_config(plan_config)
    _git_execution_provenance(
        root, plan_config["parent_sampling_policy"]["sha256"]
    )
    structured = build_acquisition_plan_audit(root, plan)
    return publish_audit(root, plan, output, structured)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Freeze a read-only independent acquisition plan"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = run(
        project_root=args.project_root,
        plan_path=args.plan,
        output_root=args.output_root,
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
