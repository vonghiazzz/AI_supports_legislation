import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking.family_quota import (
    FamilyQuotaError,
    compatible_families,
    family_quota_status_report,
    solve_family_quota_assignment,
)
from backend.scripts.Evaluation.benchmarking.independent_benchmark import (
    validate_benchmark_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"
TARGETS = (
    "BLDS_CORE",
    "BLTTDS_PRIMARY",
    "LAND_2013",
    "LAND_2024",
    "HNGD_2014",
)
MINIMUMS = dict(zip(TARGETS, (55, 30, 35, 35, 30)))
MAXIMUMS = dict(zip(TARGETS, (65, 40, 45, 45, 40)))


def _bounds(**updates):
    values = {family: 0 for family in TARGETS}
    values.update(updates)
    return values


def _record(index, primary, secondary=None, source_prefix="independent:test"):
    return {
        "source_id": f"{source_prefix}-{index:03d}",
        "case_id": f"{index:03d}/2026/DS-ST",
        "primary_family_preliminary": primary,
        "secondary_families_preliminary": list(secondary or []),
        "classification_status": (
            "MULTI_FAMILY"
            if primary == "MULTI_FAMILY"
            else "UNKNOWN"
            if primary == "UNKNOWN"
            else "CLASSIFIED"
        ),
    }


def _checked_in_records():
    artifact = json.loads(
        (BENCHMARK_DIR / "preliminary_stratification.json").read_text(
            encoding="utf-8"
        )
    )
    return artifact["records"]


def test_compatible_families_preserves_multi_label_membership():
    record = _record(
        1,
        "MULTI_FAMILY",
        ["LAND_2013", "BLTTDS_PRIMARY", "BLDS_CORE"],
    )
    assert compatible_families(record) == (
        "BLDS_CORE",
        "BLTTDS_PRIMARY",
        "LAND_2013",
    )
    assert record["primary_family_preliminary"] == "MULTI_FAMILY"
    assert record["secondary_families_preliminary"] == [
        "LAND_2013",
        "BLTTDS_PRIMARY",
        "BLDS_CORE",
    ]


def test_one_source_cannot_fill_two_family_slots():
    record = _record(1, "MULTI_FAMILY", ["BLDS_CORE", "LAND_2013"])
    result = solve_family_quota_assignment(
        [record],
        _bounds(BLDS_CORE=1, LAND_2013=1),
        candidate_scope="synthetic_no_double_count",
    )
    assert result["feasible"] is False
    assert result["maximum_filled_minimum_slots"] == 1
    assert result["shortage"] == 1
    assert sum(result["family_counts"].values()) == 1
    assert result["quota_family_assignment"][record["source_id"]] in {
        "BLDS_CORE",
        "LAND_2013",
    }


def test_assignment_is_deterministic_across_repeated_runs():
    records = [
        _record(2, "MULTI_FAMILY", ["BLDS_CORE", "LAND_2013"]),
        _record(1, "MULTI_FAMILY", ["BLDS_CORE", "LAND_2013"]),
        _record(3, "BLDS_CORE"),
    ]
    kwargs = {
        "records": records,
        "lower_bounds": _bounds(BLDS_CORE=1, LAND_2013=1),
        "candidate_scope": "determinism_fixture",
    }
    first = solve_family_quota_assignment(**kwargs)
    second = solve_family_quota_assignment(**copy.deepcopy(kwargs))
    assert first == second
    assert "fewer compatible families first" in first["tie_break_rule"]


def test_minimum_and_optional_maximum_bounds_are_supported():
    records = [
        _record(1, "BLDS_CORE"),
        _record(2, "LAND_2013"),
        _record(3, "LAND_2024"),
    ]
    lower = _bounds(BLDS_CORE=1, LAND_2013=1, LAND_2024=1)
    upper = _bounds(BLDS_CORE=2, LAND_2013=2, LAND_2024=2)
    result = solve_family_quota_assignment(
        records,
        lower,
        upper_bounds=upper,
        candidate_scope="min_max_fixture",
    )
    assert result["feasible"] is True
    assert result["family_counts"] == lower
    assert all(
        result["family_counts"][family] <= upper[family]
        for family in TARGETS
    )


def test_current_223_case_frame_reproduces_membership_and_max_flow():
    result = solve_family_quota_assignment(
        _checked_in_records(),
        MINIMUMS,
        upper_bounds=MAXIMUMS,
        candidate_scope="reconciled_223_eligible_independent",
    )
    assert result["candidate_count"] == 223
    assert result["compatible_membership_counts"] == {
        "BLDS_CORE": 101,
        "BLTTDS_PRIMARY": 53,
        "LAND_2013": 46,
        "LAND_2024": 25,
        "HNGD_2014": 38,
    }
    assert result["feasible"] is False
    assert result["required_slots"] == 185
    assert result["maximum_filled_minimum_slots"] == 172
    assert result["shortage"] == 13


def test_current_frame_reproduces_strongest_hall_bottleneck():
    result = solve_family_quota_assignment(
        _checked_in_records(),
        MINIMUMS,
        candidate_scope="reconciled_223_eligible_independent",
    )
    assert result["hall_bottlenecks"][0] == {
        "families": ["BLDS_CORE", "LAND_2013", "LAND_2024"],
        "compatible_source_count": 112,
        "required_slots": 125,
        "shortage": 13,
    }


def test_thirteen_synthetic_land_2024_sources_make_minimums_feasible():
    records = copy.deepcopy(_checked_in_records())
    records.extend(
        _record(
            1000 + index,
            "LAND_2024",
            source_prefix="independent:synthetic-land-2024",
        )
        for index in range(13)
    )
    result = solve_family_quota_assignment(
        records,
        MINIMUMS,
        upper_bounds=MAXIMUMS,
        candidate_scope="temporary_plus_13_land_2024",
    )
    assert result["feasible"] is True
    assert result["maximum_filled_minimum_slots"] == 185
    assert result["shortage"] == 0
    assert result["family_counts"] == MINIMUMS


@pytest.mark.parametrize("primary", ["UNKNOWN", "OTHER"])
def test_unknown_and_other_cannot_fill_target_slots(primary):
    record = _record(1, primary)
    result = solve_family_quota_assignment(
        [record],
        _bounds(BLDS_CORE=1),
        candidate_scope=f"{primary.lower()}_fixture",
    )
    assert compatible_families(record) == ()
    assert result["maximum_filled_minimum_slots"] == 0
    assert result["quota_family_assignment"][record["source_id"]] is None


def test_development_source_is_rejected():
    record = _record(1, "BLDS_CORE", source_prefix="judgment:development")
    with pytest.raises(FamilyQuotaError, match="independent candidates only"):
        solve_family_quota_assignment(
            [record],
            _bounds(),
            candidate_scope="invalid_development_scope",
        )


@pytest.mark.parametrize("duplicate_field", ["source_id", "case_id"])
def test_duplicate_candidate_identity_is_rejected(duplicate_field):
    records = [_record(1, "BLDS_CORE"), _record(2, "LAND_2013")]
    records[1][duplicate_field] = records[0][duplicate_field]
    with pytest.raises(FamilyQuotaError, match=f"Duplicate {duplicate_field}"):
        solve_family_quota_assignment(
            records,
            _bounds(),
            candidate_scope="duplicate_fixture",
        )


def test_quota_assignment_does_not_change_multi_law_gt_or_membership():
    record = _record(
        1,
        "MULTI_FAMILY",
        ["BLDS_CORE", "BLTTDS_PRIMARY", "LAND_2013"],
    )
    record["ground_truth"] = {
        "core": [{"law_code": "bo_luat_dan_su_2015"}],
        "supporting": [
            {"law_code": "luat_dat_dai_2013"},
            {"law_code": "bo_luat_to_tung_dan_su_2015"},
        ],
    }
    frozen = copy.deepcopy(record)
    result = solve_family_quota_assignment(
        [record],
        _bounds(LAND_2013=1),
        candidate_scope="gt_separation_fixture",
    )
    assert result["quota_family_assignment"][record["source_id"]] == "LAND_2013"
    assert compatible_families(record) == (
        "BLDS_CORE",
        "BLTTDS_PRIMARY",
        "LAND_2013",
    )
    assert record == frozen
    assert result["assignment_role"] == "SAMPLING_METADATA_NOT_GROUND_TRUTH"


def test_schema_v2_still_accepts_ground_truth_from_multiple_laws():
    schema = json.loads(
        (BENCHMARK_DIR / "gt_schema_v2.snapshot.json").read_text(encoding="utf-8")
    )

    def gt(law_code, importance, score, domain):
        return {
            "law_code": law_code,
            "article": "Điều 1",
            "reasoning": "Synthetic schema-only validation record.",
            "importance": importance,
            "importance_score": score,
            "source_type": "explicit",
            "domain": domain,
        }

    benchmark = [{
        "id": 1,
        "judgment_name": "synthetic.pdf",
        "query": "Synthetic schema-only query.",
        "ground_truth": {
            "core": [
                gt("bo_luat_dan_su_2015", "decisive", 3, "substantive"),
                gt("luat_dat_dai_2013", "decisive", 3, "substantive"),
            ],
            "supporting": [
                gt(
                    "bo_luat_to_tung_dan_su_2015",
                    "supporting",
                    1,
                    "procedural",
                )
            ],
        },
    }]
    validate_benchmark_records(benchmark, schema)


def test_blttds_family_membership_is_separate_from_main_guardrail():
    record = _record(1, "BLTTDS_PRIMARY")
    record["blttds_only_procedural"] = False
    first = solve_family_quota_assignment(
        [record],
        _bounds(BLTTDS_PRIMARY=1),
        candidate_scope="blttds_separation_fixture",
    )
    record["blttds_only_procedural"] = True
    second = solve_family_quota_assignment(
        [record],
        _bounds(BLTTDS_PRIMARY=1),
        candidate_scope="blttds_separation_fixture",
    )
    assert first == second
    assert first["blttds_guardrail_separate"] is True


def test_candidate_scope_is_explicit_and_not_inferred():
    record = _record(1, "BLDS_CORE")
    with pytest.raises(FamilyQuotaError, match="candidate_scope"):
        solve_family_quota_assignment(
            [record],
            _bounds(),
            candidate_scope="",
        )
    with pytest.raises(TypeError, match="candidate_scope"):
        solve_family_quota_assignment([record], _bounds())


def test_quota_status_report_keeps_three_quantities_distinct():
    records = [
        _record(1, "LAND_2013"),
        _record(2, "MULTI_FAMILY", ["BLDS_CORE", "LAND_2013"]),
    ]
    report = family_quota_status_report(
        records,
        _bounds(LAND_2013=1),
        candidate_scope="reporting_fixture",
    )
    land = report["families"]["LAND_2013"]
    assert land == {
        "SINGLE_FAMILY_COUNT": 1,
        "COMPATIBLE_MEMBERSHIP_COUNT": 2,
        "UNIQUE_ASSIGNMENT_CAPACITY": 1,
    }
