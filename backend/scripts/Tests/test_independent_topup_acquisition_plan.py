import ast
import copy
import json
import tempfile
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import topup_acquisition_planning as planner


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_PATH = PROJECT_ROOT / planner.DEFAULT_PLAN_RELATIVE


@pytest.fixture(scope="module")
def plan():
    value = planner.strict_yaml_load(PLAN_PATH)
    planner.validate_plan_config(value)
    return value


def mutated(plan, path, value):
    result = copy.deepcopy(plan)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return result


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("parent_provenance", "promotion_checkpoint_commit"), "0" * 40),
        (("parent_provenance", "acquisition_manifest_sha256"), "0" * 64),
        (("parent_provenance", "b3_run_fingerprint"), "0" * 64),
        (("residual_constraints", "eligible_count", "deficit"), 1),
        (("residual_constraints", "land_2024_compatible", "deficit"), 4),
        (("residual_constraints", "family_minimum_slots", "shortage"), 4),
        (("residual_constraints", "hall_bottleneck", "shortage"), 4),
        (("residual_interpretation", "additive_deficit_forbidden"), False),
        (("batch_size", "raw_target"), 49),
        (("batch_size", "role"), "SCIENTIFIC_CLOSE_GUARANTEE"),
        (("family_specific_yield", "calibration_sufficient"), True),
        (("batch_identity", "prefix"), "batch2db"),
        (("targeting", "priority_order"), ["LAND_2013_TARGETED_DISCOVERY"]),
        (("required_state", "sampling_seed"), 42),
        (("required_state", "selection"), "STARTED"),
    ],
)
def test_frozen_contract_rejects_mutation(plan, path, value):
    with pytest.raises(planner.TopupPlanningError):
        planner.validate_plan_config(mutated(plan, path, value))


def test_plan_identity_and_separate_batch(plan):
    assert plan["plan_id"] == planner.PLAN_ID
    assert plan["plan_status"] == planner.PLAN_STATUS
    assert plan["batch_label"] == "BATCH_2D_C_TOPUP"
    assert plan["batch_id_prefix"] == "batch2dc"
    assert plan["batch_identity"] == {
        "separate_from_batch2db": True,
        "prefix": "batch2dc",
        "silent_extension_of_batch2db_forbidden": True,
    }


def test_deficits_are_one_land_bottleneck_not_additive(plan):
    residual = plan["residual_constraints"]
    assert residual["land_2024_compatible"]["deficit"] == 5
    assert residual["family_minimum_slots"]["shortage"] == 5
    assert residual["hall_bottleneck"]["shortage"] == 5
    interpretation = plan["residual_interpretation"]
    assert interpretation["deficits_overlap"] is True
    assert interpretation["additive_deficit_forbidden"] is True
    assert interpretation["scientific_compatible_source_deficit"] == 5


def test_raw_target_is_operational_not_scientific_yield_claim(plan):
    assert plan["batch_size"]["raw_target"] == 50
    assert plan["batch_size"]["historical_completed_scale_sizes"] == [41, 50, 50, 53, 53]
    assert plan["batch_size"]["rationale"] == "HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN_OPERATIONAL_REFERENCE"
    assert plan["batch_size"]["role"] == "OPERATIONAL_COHORT_SIZE_NOT_SCIENTIFIC_CLOSE_GUARANTEE"
    assert plan["family_specific_yield"]["calibration_sufficient"] is False
    assert plan["family_specific_yield"]["general_eligible_yield_is_land_yield"] is False


def test_only_land_2024_is_targeting_priority(plan):
    assert plan["targeting"]["priority_order"] == ["LAND_2024_TARGETED_DISCOVERY"]
    assert plan["targeting"]["confirmed_family_claim_before_review"] is False
    assert plan["targeting"]["raw_family_quota"] is None


def test_protected_state_remains_unstarted(plan):
    assert plan["required_state"] == {
        "benchmark": "DRAFT", "sampling": "DRAFT", "selection": "NOT_STARTED",
        "sampling_seed": None, "selected_cases": 0, "query_records": 0,
        "ground_truth_records": 0,
    }


def test_authoritative_b3_and_live_state_validate(plan):
    result = planner.validate_authoritative_inputs(PROJECT_ROOT, plan)
    assert result["intake_counts"] == {
        "total": 317, "eligible": 257, "extracted": 59, "hash_verified": 1,
    }
    assert result["protected_state"]["sampling_seed"] is None


def test_canonical_audit_is_deterministic_and_action_free(plan):
    first = planner.build_audit(PROJECT_ROOT, PLAN_PATH)
    second = planner.build_audit(PROJECT_ROOT, PLAN_PATH)
    assert first == second
    assert first["plan_fingerprint"] == planner.canonical_json_fingerprint(plan)
    assert all(
        value in (False, 0, True)
        for value in first["execution_state"].values()
    )
    assert first["execution_state"]["plan_only"] is True
    assert not any(value for key, value in first["execution_state"].items() if key != "plan_only")
    json.dumps(first, sort_keys=True, allow_nan=False)


def test_planner_has_no_network_or_acquisition_execution_calls():
    source_path = Path(planner.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imports.isdisjoint({"requests", "urllib", "httpx", "aiohttp", "socket"})
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint({"urlopen", "post", "download", "register_source", "extract", "promote"})


def test_publish_audit_is_atomic_and_contains_required_fingerprints():
    audit = planner.build_audit(PROJECT_ROOT, PLAN_PATH)
    runs_root = PROJECT_ROOT / "backend/data/runs"
    with tempfile.TemporaryDirectory(prefix="topup-plan-test-", dir=runs_root) as directory:
        output = Path(directory)
        final = planner.publish_audit(PROJECT_ROOT, PLAN_PATH, output, audit)
        manifest = json.loads((final / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["parent_b3_run_fingerprint"] == planner.B3_RUN_FINGERPRINT
        assert manifest["current_acquisition_manifest_sha256"] == planner.MANIFEST_SHA256
        assert manifest["plan_fingerprint"] == audit["plan_fingerprint"]
        assert manifest["run_fingerprint"] == planner.canonical_json_fingerprint(
            {key: value for key, value in manifest.items() if key != "run_fingerprint"}
        )
        assert not list(output.glob("*.partial"))
