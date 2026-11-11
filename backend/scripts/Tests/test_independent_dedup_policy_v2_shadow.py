import ast
import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as v2
from backend.scripts.Evaluation.benchmarking import dedup_policy_v2_shadow as shadow


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TEST_RUN_ID = shadow.SHADOW_PROTOCOL_ID + "_test"


@pytest.fixture(scope="module")
def inputs():
    return shadow.load_shadow_inputs(PROJECT_ROOT)


@pytest.fixture(scope="module")
def outputs():
    return shadow.build_shadow_evaluation(PROJECT_ROOT, "a" * 40, TEST_RUN_ID)


def _sources(outputs):
    return {
        item["acquisition_id"]: item
        for item in outputs["source_level_shadow_decisions.json"]["sources"]
    }


def test_frozen_v1_and_manifest_compatibility():
    assert v2.sha256_file(
        PROJECT_ROOT / "backend/scripts/Evaluation/benchmarking/independent_benchmark.py"
    ) == "eba62bdb5c484f050db3f9caff7999d85a205a42b39d868d5f8ee79f92526901"
    assert v2.sha256_file(PROJECT_ROOT / shadow.MANIFEST_RELATIVE) == shadow.MANIFEST_SHA256
    assert shadow.EXPECTED_PARENT_GIT_SHA == "30e7d307dc692cefa49c9b257ba80e18fc737970"


def test_frozen_v2_policy_validation(inputs):
    assert inputs["policy"]["policy_fingerprint"] == (
        "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"
    )
    assert inputs["policy"]["activation"]["active"] is False
    assert inputs["policy"]["activation"]["replaces_v1"] is False


def test_real_evidence_bindings_cover_exact_batch(inputs):
    assert len(inputs["batch_sources"]) == 46
    assert {item["acquisition_status"] for item in inputs["batch_sources"]} == {"extracted"}
    assert len(inputs["source_artifact_evidence"]) == 46
    assert inputs["pair_evidence"]["pair_count"] == 23
    assert inputs["policy_adjudication_report"]["report_fingerprint"] == v2.PARENT_R3_REPORT_FINGERPRINT


def test_candidate_pair_generation_is_evidence_triggered_and_complete(inputs):
    controls = shadow.select_candidate_controls(inputs["fixture"]["pair_controls"])
    assert len(controls) == 23
    assert len({tuple(sorted((item["left_source_id"], item["right_source_id"]))) for item in controls}) == 23
    assert any(item["raw_sha_equal"] is True for item in controls)
    flagged = {
        item["acquisition_id"]
        for item in inputs["fixture"]["source_adjudications"]
    }
    covered = {
        source_id.removeprefix("independent:")
        for item in controls
        for source_id in (item["left_source_id"], item["right_source_id"])
        if source_id.startswith("independent:batch2dc-")
    }
    assert covered == flagged


def test_unrelated_development_pair_is_not_manufactured(inputs):
    unrelated = copy.deepcopy(inputs["fixture"]["pair_controls"][0])
    unrelated["control_id"] = "UNRELATED"
    unrelated["scope"] = "DEVELOPMENT"
    unrelated["finding_reasons"] = []
    with pytest.raises(shadow.ShadowEvaluationError, match="lacks attributable"):
        shadow.select_candidate_controls([unrelated])


def test_confirmed_development_overlap_blocks(inputs):
    control = copy.deepcopy(inputs["fixture"]["pair_controls"][0])
    control.update(
        {
            "scope": "DEVELOPMENT",
            "raw_sha_equal": True,
            "filename_equal": False,
            "finding_reasons": ["development_benchmark_exact_sha256"],
        }
    )
    result = v2.evaluate_pair(inputs["policy"], v2.pair_input_from_fixture(control))
    assert result["development_relation"] == "CONFIRMED_DEVELOPMENT_SAME_DOCUMENT"
    assert result["overall_blocking"] is True


def test_six_filename_only_development_false_positives_clear(outputs):
    ledger = outputs["candidate_pair_ledger.json"]["candidate_pairs"]
    development = [item for item in ledger if item["scope"] == "DEVELOPMENT"]
    assert len(development) == 6
    assert all(
        item["decision"]["development_relation"] == "WEAK_DEVELOPMENT_COLLISION_CLEARED"
        and item["decision"]["overall_blocking"] is False
        for item in development
    )


def test_judgment_number_collision_controls_remain_weak(outputs):
    rows = [
        item
        for item in outputs["candidate_pair_ledger.json"]["candidate_pairs"]
        if "new_pool_duplicate_normalized_identity" in item["candidate_generation_basis"]
        and item["control_id"] != "R3-PAIR-015"
    ]
    assert rows
    assert all(item["decision"]["same_document_relation"] == "WEAK_ASSOCIATION_DISTINCT_CONTROL" for item in rows)


def test_strong_duplicate_and_canonical_025_026(outputs):
    ledger = outputs["candidate_pair_ledger.json"]["candidate_pairs"]
    strong = [item for item in ledger if item["decision"]["same_document_relation"] == "CONFIRMED_EXACT_DUPLICATE"]
    assert [item["control_id"] for item in strong] == ["R3-PAIR-015"]
    canonical = outputs["source_level_shadow_decisions.json"]["canonical_retention"]
    assert canonical["canonical_source_id"] == "independent:batch2dc-20261105-025"
    assert canonical["noncanonical_rejected_source_ids"] == ["independent:batch2dc-20261105-026"]


def test_independent_blocker_survives_canonical_retention():
    sha = "f" * 64
    result = v2.canonicalize_exact_duplicate_cluster(
        [
            {
                "source_id": "A",
                "case_id": "1",
                "raw_sha256": sha,
                "other_blockers": ["DISPUTE_RELATION_UNRESOLVED"],
                "findings_provenance": ["duplicate"],
            },
            {
                "source_id": "B",
                "case_id": "2",
                "raw_sha256": sha,
                "other_blockers": [],
                "findings_provenance": ["duplicate"],
            },
        ]
    )
    assert result["canonical_blocked_by_other_findings"] is True
    assert result["canonical_other_blockers"] == ["DISPUTE_RELATION_UNRESOLVED"]


def test_weak_associations_do_not_create_strong_components(outputs):
    ledger = outputs["candidate_pair_ledger.json"]["candidate_pairs"]
    results = [item["decision"] for item in ledger]
    views = v2.relation_views(results)
    assert len(views["weak_associations_nontransitive"]) == 16
    assert views["strong_same_document_components"] == [[
        "independent:batch2dc-20261105-025",
        "independent:batch2dc-20261105-026",
    ]]


def test_five_underlying_dispute_sources_fail_closed(outputs):
    sources = _sources(outputs)
    expected = {
        "batch2dc-20261105-008",
        "batch2dc-20261105-014",
        "batch2dc-20261105-022",
        "batch2dc-20261105-024",
        "batch2dc-20261105-034",
    }
    observed = {
        acquisition_id
        for acquisition_id, item in sources.items()
        if item["shadow_eligibility_projection"] == "UNRESOLVED_FAIL_CLOSED"
    }
    assert observed == expected
    assert all(sources[item]["independent_blockers"] == ["DISPUTE_RELATION_UNRESOLVED"] for item in expected)


def test_missing_candidate_evidence_fails_closed(inputs):
    invalid = copy.deepcopy(inputs["fixture"]["pair_controls"][0])
    invalid["control_id"] = "MISSING"
    invalid["finding_reasons"] = ["UNKNOWN_UNVERIFIED_SIGNAL"]
    with pytest.raises(shadow.ShadowEvaluationError, match="unknown finding"):
        shadow.select_candidate_controls([invalid])


def test_duplicate_candidate_pair_is_rejected(inputs):
    control = copy.deepcopy(inputs["fixture"]["pair_controls"][0])
    duplicate = copy.deepcopy(control)
    duplicate["control_id"] = "DUPLICATE-ID-DIFFERENT"
    with pytest.raises(shadow.ShadowEvaluationError, match="Duplicate candidate pair"):
        shadow.select_candidate_controls([control, duplicate])


def test_candidate_order_and_shadow_result_are_reproducible(inputs, outputs):
    controls = inputs["fixture"]["pair_controls"]
    assert shadow.select_candidate_controls(controls) == shadow.select_candidate_controls(list(reversed(controls)))
    second = shadow.build_shadow_evaluation(PROJECT_ROOT, "a" * 40, TEST_RUN_ID)
    assert second == outputs


def test_source_level_decisions_cover_expected_control_groups(outputs):
    sources = _sources(outputs)
    counts = {}
    for item in sources.values():
        classification = item["analytical_classification"]
        counts[classification] = counts.get(classification, 0) + 1
    assert counts == {
        "PREVIOUSLY_CLEAN": 23,
        "DISTINCT_DOCUMENT_WITH_SHARED_METADATA": 16,
        "CONFIRMED_SAME_DOCUMENT": 2,
        "GRAPH_RELATED_BUT_DOCUMENT_IDENTITY_UNPROVEN": 5,
    }
    projections = [item["shadow_eligibility_projection"] for item in sources.values()]
    assert projections.count("POTENTIALLY_ELIGIBLE_UNDER_S2_SHADOW") == 40
    assert projections.count("UNRESOLVED_FAIL_CLOSED") == 5
    assert projections.count("REJECTED_NONCANONICAL") == 1


def test_counterfactuals_and_land_projection_match_frozen_solver(outputs):
    rows = outputs["v1_v2_comparison.json"]["counterfactuals"]
    assert [(item["scenario"], item["batch2dc_potential_eligible_count"], item["difference"]) for item in rows] == [
        ("S0", 23, 0),
        ("S1", 39, 0),
        ("S2", 40, 0),
    ]
    assert [(item["land_2024_compatible_count"], item["family_shortage"]) for item in rows] == [(31, 4), (31, 4), (32, 3)]


def test_migration_requires_new_versioned_contract(outputs):
    report = outputs["migration_readiness_report.json"]
    assert report["readiness"] == "NEW_VERSIONED_REVIEW_CONTRACT_REQUIRED"
    assert report["safe_to_replace_v1_in_place"] is False
    assert report["v2_activation_authorized"] is False
    assert len(report["required_changes"]) == 7


def test_all_output_artifacts_are_self_fingerprinted(outputs):
    for payload in outputs.values():
        assert payload["artifact_fingerprint"] == v2.canonical_fingerprint(
            payload, excluded_key="artifact_fingerprint"
        )


def test_shadow_has_no_review_or_promotion_dependency():
    path = PROJECT_ROOT / "backend/scripts/Evaluation/benchmarking/dedup_policy_v2_shadow.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert "record_source_review" not in names
    assert "promote_source_to_eligible" not in names
    assert "_eligibility_dedup_report" not in names


def test_build_does_not_mutate_protected_state():
    protected = [
        shadow.MANIFEST_RELATIVE,
        Path("backend/scripts/Evaluation/benchmarking/independent_benchmark.py"),
        Path("backend/scripts/Evaluation/configs/independent_dedup_policy_v2.yaml"),
        Path("backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"),
        Path("backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"),
    ]
    before = {path.as_posix(): v2.sha256_file(PROJECT_ROOT / path) for path in protected}
    shadow.build_shadow_evaluation(PROJECT_ROOT, "b" * 40, TEST_RUN_ID)
    after = {path.as_posix(): v2.sha256_file(PROJECT_ROOT / path) for path in protected}
    assert after == before


def test_validation_summary_explicitly_records_zero_mutations(outputs):
    summary = outputs["validation_summary.json"]
    assert summary["result"] == "PASS"
    assert summary["source_review_calls"] == 0
    assert summary["promotion_calls"] == 0
    assert summary["manifest_mutated"] is False
    assert summary["v1_modified"] is False
    assert summary["v2_activated"] is False
    assert summary["benchmark_or_sampling_mutated"] is False
    assert summary["network_used"] is False
    assert summary["retrieval_or_model_used"] is False
