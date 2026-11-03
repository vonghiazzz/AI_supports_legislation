import ast
import copy
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.scripts.Evaluation.benchmarking import sampling_policy as policy_module
from backend.scripts.Evaluation.benchmarking.family_quota import (
    compatible_families,
    solve_family_quota_assignment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / policy_module.DEFAULT_POLICY_RELATIVE
BENCHMARK_DIR = (
    PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1"
)
TARGET_FAMILIES = (
    "BLDS_CORE",
    "BLTTDS_PRIMARY",
    "LAND_2013",
    "LAND_2024",
    "HNGD_2014",
)


@pytest.fixture(scope="module")
def policy():
    value = policy_module.strict_yaml_load(POLICY_PATH)
    policy_module.validate_policy(value)
    return value


@pytest.fixture(scope="module")
def audit():
    return policy_module.build_policy_audit(PROJECT_ROOT, POLICY_PATH)


def _record(primary, secondary=None, index=1):
    return {
        "source_id": f"independent:test-{index}",
        "case_id": f"TEST-{index}/2026/DS-ST",
        "primary_family_preliminary": primary,
        "secondary_families_preliminary": list(secondary or []),
        "classification_status": (
            "MULTI_FAMILY"
            if primary == "MULTI_FAMILY"
            else "UNKNOWN"
            if primary == "UNKNOWN"
            else "CLASSIFIED"
        ),
        "stress_descriptors": [],
    }


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


def _repository_tracked_clean():
    return (
        subprocess.run(
            ["git", "diff", "--quiet", "--"], cwd=PROJECT_ROOT, check=False
        ).returncode
        == 0
        and subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--"],
            cwd=PROJECT_ROOT,
            check=False,
        ).returncode
        == 0
    )


def test_policy_target_and_status_are_phase_2a_only(policy):
    assert policy["policy_status"] == "POLICY_FROZEN_FOR_ACQUISITION_PLANNING"
    assert policy["policy_status"] != "SAMPLING_FROZEN"
    assert policy["target"]["total"] == 250
    assert policy["target"]["main"] == 200
    assert policy["target"]["stress"] == 50
    assert policy["target"]["slices_disjoint"] is True


def test_policy_yaml_keeps_authoritative_authoring_base(policy):
    assert policy["authoritative_base_commit"] == (
        "299b45317648219d4f780531ac3d2f3c6196094e"
    )
    assert policy_module.AUTHORITATIVE_BASE_COMMIT == policy[
        "authoritative_base_commit"
    ]


def test_descendant_execution_head_may_differ_from_policy_base(monkeypatch):
    descendant = "b" * 40
    monkeypatch.setattr(
        policy_module.subprocess,
        "run",
        _fake_git_runner(head=descendant),
    )
    result = policy_module._git_execution_provenance(PROJECT_ROOT)
    assert descendant != policy_module.AUTHORITATIVE_BASE_COMMIT
    assert result == {
        "execution_git_commit": descendant,
        "policy_authoritative_base_commit": (
            "299b45317648219d4f780531ac3d2f3c6196094e"
        ),
        "authoritative_base_is_ancestor": True,
        "tracked_worktree_clean": True,
    }


def test_non_descendant_execution_fails_closed(monkeypatch):
    monkeypatch.setattr(
        policy_module.subprocess,
        "run",
        _fake_git_runner(ancestor_returncode=1),
    )
    with pytest.raises(policy_module.SamplingPolicyError, match="not an ancestor"):
        policy_module._git_execution_provenance(PROJECT_ROOT)


@pytest.mark.parametrize(
    ("unstaged", "staged", "message"),
    [(1, 0, "unstaged tracked"), (0, 1, "staged tracked")],
)
def test_tracked_modifications_fail_closed(
    monkeypatch, unstaged, staged, message
):
    monkeypatch.setattr(
        policy_module.subprocess,
        "run",
        _fake_git_runner(
            unstaged_returncode=unstaged,
            staged_returncode=staged,
        ),
    )
    with pytest.raises(policy_module.SamplingPolicyError, match=message):
        policy_module._git_execution_provenance(PROJECT_ROOT)


def test_untracked_files_are_not_part_of_provenance_gate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        policy_module.subprocess,
        "run",
        _fake_git_runner(calls=calls),
    )
    result = policy_module._git_execution_provenance(PROJECT_ROOT)
    assert result["tracked_worktree_clean"] is True
    assert ["git", "status"] not in calls
    assert all("--untracked-files" not in command for command in calls)
    assert ["git", "diff", "--quiet", "--"] in calls
    assert ["git", "diff", "--cached", "--quiet", "--"] in calls


def test_current_clean_repository_accepts_descendant_execution_with_untracked_files():
    if not _repository_tracked_clean():
        pytest.skip("actual provenance integration requires post-commit tracked clean state")
    result = policy_module._git_execution_provenance(PROJECT_ROOT)
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert result["execution_git_commit"] == actual_head
    assert result["policy_authoritative_base_commit"] == (
        "299b45317648219d4f780531ac3d2f3c6196094e"
    )
    assert result["authoritative_base_is_ancestor"] is True
    assert result["tracked_worktree_clean"] is True


def test_benchmark_and_sampling_remain_draft(audit):
    state = audit["current_frame_audit.json"]["state"]
    assert state["benchmark_state"] == "DRAFT"
    assert state["sampling_state"] == "DRAFT"
    assert state["selection_status"] == "NOT_STARTED"
    assert state["sampling_seed"] is None
    assert state["sampling_frozen"] is False


def test_hard_family_minima_are_exact(policy):
    assert policy["family_quota"]["hard_minima"] == {
        "BLDS_CORE": 55,
        "BLTTDS_PRIMARY": 30,
        "LAND_2013": 35,
        "LAND_2024": 35,
        "HNGD_2014": 30,
    }
    assert sum(policy["family_quota"]["hard_minima"].values()) == 185


def test_soft_family_maxima_are_exact_and_not_hard(policy, audit):
    maxima = policy["family_quota"]["soft_maxima"]
    assert maxima == {
        "BLDS_CORE": 65,
        "BLTTDS_PRIMARY": 40,
        "LAND_2013": 45,
        "LAND_2024": 45,
        "HNGD_2014": 40,
    }
    family = audit["family_feasibility.json"]
    assert family["upper_bounds_used_for_hard_feasibility"] is None
    assert family["soft_maxima_role"] == "REPORT_AND_PENALIZE_NOT_INVALIDATE"
    counts = dict(maxima)
    counts["BLDS_CORE"] += 1
    assert policy_module.report_soft_max_overflow(counts, maxima) == {
        "BLDS_CORE": 1,
        "BLTTDS_PRIMARY": 0,
        "LAND_2013": 0,
        "LAND_2024": 0,
        "HNGD_2014": 0,
    }


def test_frame_above_soft_max_remains_hard_feasible_and_reports_overflow(policy):
    records = []
    index = 0
    for family, count in {
        "BLDS_CORE": 66,
        "BLTTDS_PRIMARY": 30,
        "LAND_2013": 35,
        "LAND_2024": 35,
        "HNGD_2014": 30,
    }.items():
        for _ in range(count):
            index += 1
            records.append(_record(family, index=index))
    result = solve_family_quota_assignment(
        records,
        policy["family_quota"]["hard_minima"],
        candidate_scope="soft_max_is_not_hard_fixture",
        upper_bounds=None,
    )
    availability = result["compatible_membership_counts"]
    overflow = policy_module.report_soft_max_overflow(
        availability, policy["family_quota"]["soft_maxima"]
    )
    assert result["feasible"] is True
    assert result["upper_bounds"] is None
    assert overflow["BLDS_CORE"] == 1
    assert sum(overflow.values()) == 1


def test_quota_applies_to_main_only_and_stress_contributes_zero(policy):
    record = _record("BLDS_CORE")
    assert policy["family_quota"]["scope"] == "MAIN_ONLY"
    assert policy_module.quota_memberships_for_slice(record, "MAIN") == (
        "BLDS_CORE",
    )
    assert policy_module.quota_memberships_for_slice(record, "STRESS") == ()
    assert policy["family_quota"]["stress_quota_contribution_to_main"] == 0


def test_multifamily_case_fills_at_most_one_slot(policy):
    record = _record(
        "MULTI_FAMILY", ["BLDS_CORE", "LAND_2013", "LAND_2024"]
    )
    minima = {family: 0 for family in TARGET_FAMILIES}
    minima.update({"BLDS_CORE": 1, "LAND_2013": 1})
    result = solve_family_quota_assignment(
        [record], minima, candidate_scope="phase_2a_no_double_count_test"
    )
    assert result["maximum_filled_minimum_slots"] == 1
    assert sum(result["family_counts"].values()) == 1
    assert policy["family_quota"]["max_quota_slots_per_case"] == 1


@pytest.mark.parametrize("family", ["UNKNOWN", "OTHER"])
def test_unknown_and_other_fill_zero_quota_slots(policy, family):
    assert compatible_families(_record(family)) == ()
    assert policy["family_quota"]["unknown_other_quota_slots"] == 0


def test_guardrail_proxy_derivation_is_membership_based():
    procedural = _record("BLTTDS_PRIMARY")
    mixed = _record(
        "MULTI_FAMILY", ["BLTTDS_PRIMARY", "LAND_2013"], index=2
    )
    unknown = _record("UNKNOWN", index=3)
    assert policy_module.derive_guardrail_proxies(procedural) == {
        "blttds_only_procedural": True,
        "anticipated_substantive_core": False,
    }
    assert policy_module.derive_guardrail_proxies(mixed) == {
        "blttds_only_procedural": False,
        "anticipated_substantive_core": True,
    }
    assert policy_module.derive_guardrail_proxies(unknown) == {
        "blttds_only_procedural": False,
        "anticipated_substantive_core": False,
    }


def test_main_guardrail_ratios_and_target_counts(policy):
    procedural = policy["main_guardrails"]["blttds_only_procedural"]
    substantive = policy["main_guardrails"]["anticipated_substantive_core"]
    assert procedural["maximum_ratio"] == 0.20
    assert procedural["maximum_count_at_target"] == 40
    assert substantive["minimum_ratio"] == 0.65
    assert substantive["minimum_count_at_target"] == 130


def test_current_guardrail_proxy_counts(audit):
    assert audit["current_frame_audit.json"]["guardrail_proxy_frame_counts"] == {
        "blttds_only_procedural": 45,
        "anticipated_substantive_core": 149,
    }


def test_stress_requires_descriptor_without_hard_descriptor_quotas(policy):
    record = _record("BLDS_CORE")
    assert policy_module.stress_eligible(record) is False
    record["stress_descriptors"] = ["OCR_HEAVY"]
    assert policy_module.stress_eligible(record) is True
    assert policy["stress"]["hard_descriptor_quotas"] is False
    assert (
        policy["stress"]["descriptor_distribution"]
        == "REPORT_ONLY_SOFT_DIVERSITY_OBJECTIVE"
    )


def test_current_stress_eligibility_and_descriptor_counts(audit):
    stress = audit["stress_eligibility.json"]
    assert stress["eligible_source_count"] == 87
    assert stress["frame_source_count"] == 223
    assert stress["descriptor_occurrence_counts"] == {
        "BLANK_PAGE_HEAVY": 15,
        "LONG_DOCUMENT": 7,
        "OCR_HEAVY": 47,
        "RARE_PROCEDURE_STAGE": 1,
        "SHORT_DOCUMENT": 35,
    }
    assert stress["multi_descriptor_source_count"] == 18


def test_joint_selection_is_required_and_lexicographic(policy):
    joint = policy["joint_selection"]
    assert joint["required"] is True
    assert joint["case_capacity_across_slices"] == 1
    assert joint["independent_slice_sampling_forbidden"] is True
    assert joint["objective_order"][0] == "FROZEN_ELIGIBLE_INDEPENDENT_FRAME_ONLY"
    assert joint["objective_order"][-1] == "FROZEN_SEEDED_TIE_BREAK_LAST"


def test_legacy_minimum_and_fallback_are_unauthorized(policy):
    target = policy["target"]
    assert target["selection_mode"] == "TARGET_ONLY"
    assert target["fallback_authorized"] is False
    assert target["legacy_minimum_mode_authorized"] is False
    assert target["silent_quota_reduction_authorized"] is False
    assert target["proportional_quota_scaling_authorized"] is False


def test_family_minima_cannot_be_silently_scaled(policy):
    changed = copy.deepcopy(policy)
    changed["family_quota"]["hard_minima"]["BLDS_CORE"] = 54
    with pytest.raises(policy_module.SamplingPolicyError, match="hard family minima"):
        policy_module.validate_policy(changed)


@pytest.mark.parametrize(
    "field",
    [
        "fallback_authorized",
        "legacy_minimum_mode_authorized",
        "silent_quota_reduction_authorized",
        "proportional_quota_scaling_authorized",
    ],
)
def test_legacy_fallback_modes_require_a_new_policy_version(policy, field):
    changed = copy.deepcopy(policy)
    changed["target"][field] = True
    with pytest.raises(policy_module.SamplingPolicyError, match="Target/fallback"):
        policy_module.validate_policy(changed)


def test_authoritative_current_frame_identity(audit):
    frame = audit["current_frame_audit.json"]
    assert frame["source_count"] == 223
    assert (
        frame["source_inventory_fingerprint"]
        == "8957b1a617da34cd63b4c88095504d339523574c999a085a377a6847a118bc55"
    )
    assert (
        frame["records_fingerprint"]
        == "1af8b33ad2be7dcd7745716102ad846a5f3cbb02071bdbc74f229b876824b54b"
    )
    assert frame["scope"] == {
        "eligible_independent_only": True,
        "development_sources_included": False,
        "blocked_sources_included": False,
        "hash_verified_sources_included": False,
    }


def test_current_primary_and_classification_distributions(audit):
    frame = audit["current_frame_audit.json"]
    assert frame["primary_family_distribution"] == {
        "BLDS_CORE": 52,
        "BLTTDS_PRIMARY": 45,
        "LAND_2013": 5,
        "LAND_2024": 1,
        "HNGD_2014": 30,
        "MULTI_FAMILY": 61,
        "OTHER": 1,
        "UNKNOWN": 28,
    }
    assert frame["classification_status_distribution"] == {
        "CLASSIFIED": 134,
        "MULTI_FAMILY": 61,
        "UNKNOWN": 28,
    }
    assert frame["classification_coverage"] == 0.874439461883408


def test_current_compatible_membership_counts(audit):
    assert audit["current_frame_audit.json"]["compatible_membership_counts"] == {
        "BLDS_CORE": 101,
        "BLTTDS_PRIMARY": 53,
        "LAND_2013": 46,
        "LAND_2024": 25,
        "HNGD_2014": 38,
    }


def test_current_family_max_flow_is_172_of_185(audit):
    family = audit["family_feasibility.json"]
    assert family["required_slots"] == 185
    assert family["maximum_filled_minimum_slots"] == 172
    assert family["shortage"] == 13
    assert family["feasible"] is False


def test_current_strongest_hall_bottleneck(audit):
    assert audit["family_feasibility.json"]["strongest_hall_bottleneck"] == {
        "families": ["BLDS_CORE", "LAND_2013", "LAND_2024"],
        "compatible_source_count": 112,
        "required_slots": 125,
        "shortage": 13,
    }


def test_land_2024_and_total_count_deficits_remain_distinct(audit):
    constraints = audit["acquisition_constraints.json"]
    assert constraints["target_count_deficit"] == 27
    assert constraints["family_max_flow_deficit"] == 13
    assert constraints["land_2024_individual_compatibility_deficit"] == 10
    assert constraints["requirements_overlap"] is True
    assert constraints["do_not_add_deficits"] is True


def test_plus_13_land_2024_is_only_a_feasibility_construction(audit):
    proof = audit["family_feasibility.json"]["synthetic_feasibility_proof"]
    assert proof["construction"] == "+13 synthetic LAND_2024-only sources"
    assert proof["role"] == "FEASIBILITY_CONSTRUCTION_NOT_ACQUISITION_PRESCRIPTION"
    assert proof["feasible"] is True
    assert proof["maximum_filled_minimum_slots"] == 185
    assert proof["shortage"] == 0


def test_seed_derivation_is_deterministic_distinct_and_signed_63_bit(policy):
    first = policy_module.derive_sampling_seed(policy["policy_id"], "a" * 64)
    repeated = policy_module.derive_sampling_seed(policy["policy_id"], "a" * 64)
    different = policy_module.derive_sampling_seed(policy["policy_id"], "b" * 64)
    assert first == repeated == 6079059479579410068
    assert different != first
    assert 0 <= first < 2**63
    assert 0 <= different < 2**63


def test_seed_cannot_be_instantiated_while_frame_is_open(policy, audit):
    seed = policy["seed_freeze"]
    assert seed["sampling_seed"] is None
    assert seed["generate_in_phase_2a"] is False
    assert seed["freeze_after_acquisition_closed"] is True
    assert seed["freeze_after_final_frame_fingerprinted"] is True
    assert seed["freeze_before_selection"] is True
    assert seed["commit_before_selected_ids_observed"] is True
    assert audit["current_frame_audit.json"]["state"]["sampling_seed"] is None
    assert audit["current_frame_audit.json"]["state"]["selection_performed"] is False


def test_stale_sampling_manifest_is_detected_but_not_authoritative(audit):
    stale = audit["current_frame_audit.json"]["legacy_sampling_manifest"]
    assert stale["classification"] == "LEGACY_DRAFT_QUOTA_FEASIBILITY_STALE"
    assert set(stale["zero_availability_values"].values()) == {0}
    assert stale["authoritative_for_phase_2a"] is False
    assert stale["mutated"] is False
    assert stale["regenerate_before_sampling_frozen"] is True


def test_build_audit_does_not_mutate_historical_artifacts():
    paths = [
        BENCHMARK_DIR / "benchmark_manifest.json",
        BENCHMARK_DIR / "sampling_manifest.json",
        BENCHMARK_DIR / "preliminary_stratification.json",
    ]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    policy_module.build_policy_audit(PROJECT_ROOT, POLICY_PATH)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before


def test_no_selection_query_or_gt_is_produced(audit):
    state = audit["current_frame_audit.json"]["state"]
    assert state["selection_performed"] is False
    assert state["selected_cases"] == 0
    assert state["query_records"] == 0
    assert state["ground_truth_records"] == 0
    serialized = json.dumps(audit, sort_keys=True)
    assert "quota_family_assignment" not in serialized
    assert "selected_case_ids" not in serialized


def test_forbidden_retrieval_fields_fail_closed():
    with pytest.raises(policy_module.SamplingPolicyError, match="Forbidden"):
        policy_module._inspect_forbidden_record_keys({"retrieval_score": 1.0})


def test_no_model_retrieval_query_gt_or_network_imports():
    source = Path(policy_module.__file__).read_text(encoding="utf-8")
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


def test_strict_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(policy_module.SamplingPolicyError, match="Non-finite"):
        policy_module.strict_json_load(path)


def test_policy_audit_is_strictly_deterministic(audit):
    second = policy_module.build_policy_audit(PROJECT_ROOT, POLICY_PATH)
    assert second == audit
    assert policy_module.canonical_json_fingerprint(second) == (
        policy_module.canonical_json_fingerprint(audit)
    )
    assert policy_module.canonical_json_fingerprint(audit) == (
        "147522f58db30e2e9c80fd534839e234995ab3b66d426c3b2234e00b2b5dc0d2"
    )
    json.dumps(audit, allow_nan=False, sort_keys=True)


def test_atomic_audit_publication_contains_no_selection_output(audit, monkeypatch):
    actual_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not _repository_tracked_clean():
        monkeypatch.setattr(
            policy_module,
            "_git_execution_provenance",
            lambda project_root: {
                "execution_git_commit": actual_head,
                "policy_authoritative_base_commit": (
                    "299b45317648219d4f780531ac3d2f3c6196094e"
                ),
                "authoritative_base_is_ancestor": True,
                "tracked_worktree_clean": True,
            },
        )
    runs = PROJECT_ROOT / "backend/data/runs"
    with tempfile.TemporaryDirectory(prefix="phase2a-test-", dir=runs) as temp:
        result = policy_module.publish_audit(
            PROJECT_ROOT, POLICY_PATH, Path(temp), audit
        )
        assert result.is_dir()
        assert not result.with_name(result.name + ".partial").exists()
        assert {path.name for path in result.iterdir()} == {
            "policy_snapshot.yaml",
            "current_frame_audit.json",
            "family_feasibility.json",
            "stress_eligibility.json",
            "acquisition_constraints.json",
            "run_manifest.json",
        }
        manifest = policy_module.strict_json_load(result / "run_manifest.json")
        assert manifest["git_commit"] == actual_head
        assert manifest["execution_git_commit"] == actual_head
        assert manifest["policy_authoritative_base_commit"] == (
            "299b45317648219d4f780531ac3d2f3c6196094e"
        )
        assert manifest["authoritative_base_is_ancestor"] is True
        assert manifest["tracked_worktree_clean"] is True
        assert manifest["selection_performed"] is False
        assert manifest["sampling_seed_generated"] is False
        assert manifest["acquisition_performed"] is False
        assert manifest["models_executed"] == []
        assert manifest["retrievers_executed"] == []
        assert manifest["query_or_gt_generated"] is False
