import ast
import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import independent_benchmark as v1
from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as v2


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = PROJECT_ROOT / v2.DEFAULT_POLICY_RELATIVE
FIXTURE_PATH = PROJECT_ROOT / v2.EVIDENCE_FIXTURE_RELATIVE
MIGRATION_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/configs/independent_dedup_policy_v2_migration_report.json"
)


@pytest.fixture(scope="module")
def policy():
    return v2.load_and_validate_policy(PROJECT_ROOT)


@pytest.fixture(scope="module")
def evidence_fixture():
    return v2.validate_evidence_fixture(v2.load_evidence_fixture(FIXTURE_PATH))


def _pair(**updates):
    value = {
        "left_source_id": "independent:left",
        "right_source_id": "independent:right",
        "scope": "INTERNAL",
        "raw_sha_equal": False,
        "structured_sha_equal": False,
        "case_id_equal": False,
        "normalized_judgment_equal": False,
        "filename_equal": False,
        "court_equal": False,
        "decision_date_equal": False,
        "decision_year_equal": True,
        "document_type_equal": True,
        "procedure_stage_equal": True,
        "text_similarity": 0.01,
        "related_decision": False,
        "same_underlying_dispute": None,
        "findings_provenance": ["TEST_CONTROL"],
    }
    value.update(updates)
    return value


def _cluster_record(source_id, case_id, raw_sha, *, blockers=None, provenance=None):
    return {
        "source_id": source_id,
        "case_id": case_id,
        "raw_sha256": raw_sha,
        "other_blockers": blockers or [],
        "findings_provenance": provenance or ["new_pool_duplicate_sha256"],
    }


def _v1_record(source_id, case_id, source_name, payload):
    return {
        "source_id": source_id,
        "case_id": case_id,
        "source_name": source_name,
        "source_path": source_name,
        "source_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "structured_sha256": hashlib.sha256((payload + "-structured").encode()).hexdigest(),
        "judgment_number": case_id,
        "related_decision_ids": [],
        "underlying_dispute_id": None,
        "_dedup_text": payload,
    }


def test_policy_is_self_fingerprinted_bound_and_inactive(policy):
    assert policy["policy_fingerprint"] == v2.canonical_fingerprint(
        policy, excluded_key="policy_fingerprint"
    )
    assert policy["activation"] == {
        "active": False,
        "replaces_v1": False,
        "source_review_authorized": False,
        "eligibility_promotion_authorized": False,
        "migration_authorized": False,
    }
    assert policy["bindings"]["parent_git_sha"] == v2.PARENT_GIT_SHA
    assert policy["bindings"]["parent_manifest_sha256"] == v2.PARENT_MANIFEST_SHA256
    assert policy["bindings"]["v1_dedup_report_fingerprint"] == v2.V1_DEDUP_REPORT_FINGERPRINT


def test_policy_tampering_and_duplicate_yaml_keys_fail_closed(tmp_path):
    tampered = copy.deepcopy(v2.load_policy(POLICY_PATH))
    tampered["activation"]["active"] = True
    with pytest.raises(v2.DedupPolicyV2Error, match="fingerprint"):
        v2.validate_policy(tampered)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("policy_id: one\npolicy_id: two\n", encoding="utf-8")
    with pytest.raises(v2.DedupPolicyV2Error, match="Duplicate YAML key"):
        v2.load_policy(duplicate)


def test_frozen_fixture_identity_and_classification_counts(evidence_fixture):
    assert v2.sha256_file(FIXTURE_PATH) == v2.EVIDENCE_FIXTURE_SHA256
    assert evidence_fixture["fixture_fingerprint"] == v2.EVIDENCE_FIXTURE_FINGERPRINT
    assert len(evidence_fixture["clean_batch2dc_acquisition_ids"]) == 23
    assert len(evidence_fixture["pair_controls"]) == 23
    classifications = [
        item["analytical_classification"]
        for item in evidence_fixture["source_adjudications"]
    ]
    assert classifications.count("DISTINCT_DOCUMENT_WITH_SHARED_METADATA") == 16
    assert classifications.count("GRAPH_RELATED_BUT_DOCUMENT_IDENTITY_UNPROVEN") == 5
    assert classifications.count("CONFIRMED_SAME_DOCUMENT") == 2


def test_all_r3_pair_controls_reproduce_expected_v2_relation(policy, evidence_fixture):
    for control in evidence_fixture["pair_controls"]:
        result = v2.evaluate_pair(policy, v2.pair_input_from_fixture(control))
        expected = control["expected_v2_relation"]
        observed = (
            result["dispute_relation"]
            if expected == "DISPUTE_RELATION_UNRESOLVED"
            else result["same_document_relation"]
        )
        assert observed == expected, control["control_id"]
        assert result["findings_provenance"] == control["finding_reasons"]


def test_six_development_filename_only_controls_clear_with_complete_evidence(
    policy, evidence_fixture
):
    controls = [
        item
        for item in evidence_fixture["pair_controls"]
        if item["scope"] == "DEVELOPMENT"
    ]
    assert len(controls) == 6
    for control in controls:
        result = v2.evaluate_pair(policy, v2.pair_input_from_fixture(control))
        assert result["same_document_relation"] == "WEAK_ASSOCIATION_DISTINCT_CONTROL"
        assert result["development_relation"] == "WEAK_DEVELOPMENT_COLLISION_CLEARED"
        assert result["overall_blocking"] is False


@pytest.mark.parametrize("weak_field", ["case_id_equal", "normalized_judgment_equal", "filename_equal"])
def test_weak_metadata_collision_alone_is_not_document_equivalence(policy, weak_field):
    result = v2.evaluate_pair(policy, _pair(**{weak_field: True}))
    assert result["same_document_relation"] == "WEAK_ASSOCIATION_DISTINCT_CONTROL"
    assert result["overall_blocking"] is False


@pytest.mark.parametrize("strong_field", ["raw_sha_equal", "structured_sha_equal"])
def test_strong_positive_duplicate_evidence_cannot_be_cancelled(policy, strong_field):
    result = v2.evaluate_pair(
        policy,
        _pair(
            **{
                strong_field: True,
                "court_equal": False,
                "decision_date_equal": False,
                "decision_year_equal": False,
                "document_type_equal": False,
                "procedure_stage_equal": False,
            }
        ),
    )
    assert result["same_document_relation"] == "CONFIRMED_EXACT_DUPLICATE"
    assert result["overall_blocking"] is True


@pytest.mark.parametrize(
    "updates",
    [
        {"filename_equal": True, "structured_sha_equal": None},
        {"case_id_equal": True, "raw_sha_equal": None},
        {
            "normalized_judgment_equal": True,
            "court_equal": None,
            "decision_date_equal": None,
            "decision_year_equal": None,
            "document_type_equal": None,
            "procedure_stage_equal": None,
        },
    ],
)
def test_missing_required_weak_collision_evidence_fails_closed(policy, updates):
    result = v2.evaluate_pair(policy, _pair(scope="DEVELOPMENT", **updates))
    assert result["complete_negative_evidence"] is False
    assert result["development_relation"] == "AMBIGUOUS_DEVELOPMENT_OVERLAP"
    assert result["overall_blocking"] is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_text_similarity_is_rejected(policy, value):
    with pytest.raises(v2.DedupPolicyV2Error, match="finite"):
        v2.evaluate_pair(policy, _pair(text_similarity=value))


def test_related_decision_is_separate_from_document_identity_and_blocks_unresolved(policy):
    result = v2.evaluate_pair(policy, _pair(related_decision=True))
    assert result["same_document_relation"] == "NO_SAME_DOCUMENT_RELATION"
    assert result["dispute_relation"] == "DISPUTE_RELATION_UNRESOLVED"
    assert result["dispute_blocking"] is True


def test_reviewed_distinct_dispute_relation_can_clear(policy):
    result = v2.evaluate_pair(
        policy, _pair(related_decision=True, same_underlying_dispute=False)
    )
    assert result["dispute_relation"] == "REVIEWED_DISTINCT_DISPUTE_RELATION"
    assert result["dispute_blocking"] is False


def test_exact_duplicate_pair_local_related_edge_is_not_double_counted(policy):
    result = v2.evaluate_pair(
        policy, _pair(raw_sha_equal=True, related_decision=True)
    )
    assert result["same_document_relation"] == "CONFIRMED_EXACT_DUPLICATE"
    assert result["dispute_relation"] == "DUPLICATE_INTERNAL_RELATION_NOT_INDEPENDENT_BLOCKER"
    assert result["same_document_blocking"] is True
    assert result["dispute_blocking"] is False


def test_never_auto_promote_controls_are_frozen(policy):
    assert policy["dispute_leakage"]["never_auto_promote_acquisition_ids"] == [
        "batch2dc-20261105-008",
        "batch2dc-20261105-014",
        "batch2dc-20261105-022",
        "batch2dc-20261105-024",
        "batch2dc-20261105-034",
    ]


def test_weak_edges_are_nontransitive(policy):
    first = v2.evaluate_pair(policy, _pair(left_source_id="A", right_source_id="B", filename_equal=True))
    second = v2.evaluate_pair(policy, _pair(left_source_id="B", right_source_id="C", filename_equal=True))
    views = v2.relation_views([first, second])
    assert views["strong_same_document_components"] == []
    assert [(item["left_source_id"], item["right_source_id"]) for item in views["weak_associations_nontransitive"]] == [("A", "B"), ("B", "C")]


def test_strong_and_dispute_graphs_have_separate_transitivity(policy):
    strong = [
        v2.evaluate_pair(policy, _pair(left_source_id="A", right_source_id="B", raw_sha_equal=True)),
        v2.evaluate_pair(policy, _pair(left_source_id="B", right_source_id="C", raw_sha_equal=True)),
    ]
    dispute = [
        v2.evaluate_pair(policy, _pair(left_source_id="D", right_source_id="E", related_decision=True)),
        v2.evaluate_pair(policy, _pair(left_source_id="E", right_source_id="F", related_decision=True)),
    ]
    views = v2.relation_views(strong + dispute)
    assert views["strong_same_document_components"] == [["A", "B", "C"]]
    assert views["underlying_dispute_review_components"] == [["D", "E", "F"]]
    assert ["D", "E", "F"] not in views["strong_same_document_components"]


def test_canonical_exact_duplicate_retention_is_permutation_invariant():
    sha = "7" * 64
    records = [
        _cluster_record("independent:batch2dc-20261105-026", "1775/2023/HNGĐ-ST", sha),
        _cluster_record("independent:batch2dc-20261105-025", "1687/2023/HNGĐ-ST", sha),
    ]
    outputs = [v2.canonicalize_exact_duplicate_cluster(order) for order in itertools.permutations(records)]
    assert outputs[0] == outputs[1]
    assert outputs[0]["canonical_source_id"] == "independent:batch2dc-20261105-025"
    assert outputs[0]["noncanonical_rejected_source_ids"] == ["independent:batch2dc-20261105-026"]


def test_canonical_retention_preserves_other_blockers_and_provenance():
    sha = "8" * 64
    result = v2.canonicalize_exact_duplicate_cluster(
        [
            _cluster_record("A", "001", sha, blockers=["development_overlap"], provenance=["p1", "p2"]),
            _cluster_record("B", "002", sha, provenance=["p3"]),
        ]
    )
    assert result["canonical_source_id"] == "A"
    assert result["canonical_blocked_by_other_findings"] is True
    assert result["canonical_other_blockers"] == ["development_overlap"]
    assert result["findings_provenance_by_source"] == {"A": ["p1", "p2"], "B": ["p3"]}


def test_canonical_retention_rejects_nonexact_cluster():
    with pytest.raises(v2.DedupPolicyV2Error, match="exact raw-SHA"):
        v2.canonicalize_exact_duplicate_cluster(
            [_cluster_record("A", "001", "a" * 64), _cluster_record("B", "002", "b" * 64)]
        )


def test_counterfactual_membership_reproduces_r3_without_becoming_target(evidence_fixture):
    membership = v2.counterfactual_membership(evidence_fixture)
    assert {name: len(value["included"]) for name, value in membership.items()} == {
        "S0": 23,
        "S1": 39,
        "S2": 40,
    }
    assert membership["S2"]["rejected"] == ["batch2dc-20261105-026"]
    assert set(membership["S2"]["unresolved"]) == {
        "batch2dc-20261105-008",
        "batch2dc-20261105-014",
        "batch2dc-20261105-022",
        "batch2dc-20261105-024",
        "batch2dc-20261105-034",
    }


def test_migration_report_is_self_fingerprinted_and_noop():
    report = json.loads(MIGRATION_PATH.read_text(encoding="utf-8"))
    assert report["migration_report_fingerprint"] == v2.canonical_fingerprint(
        report, excluded_key="migration_report_fingerprint"
    )
    assert report["activation_status"] == "ANALYSIS_ONLY_NO_MIGRATION_AUTHORIZED"
    assert all(report["counterfactual_comparison"][name]["difference_from_r3"] == 0 for name in ("S0", "S1", "S2"))
    assert report["v1_compatibility"] == {
        "historical_artifacts_rewritten": False,
        "historical_expected_values_changed": False,
        "v1_implementation_modified": False,
        "v2_is_additive_and_inactive": True,
    }


def test_v1_historical_fingerprints_and_protected_files_are_unchanged():
    expected = {
        "backend/scripts/Evaluation/benchmarking/independent_benchmark.py": "eba62bdb5c484f050db3f9caff7999d85a205a42b39d868d5f8ee79f92526901",
        "backend/dataLaw/independent_intake/acquisition_manifest.json": "411a5ad54b26e640d190bc04b76fbbbfb39896c2d2b6b3fd96450347a8fec08a",
        "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json": "f3d69ee36653da42256c7c94158414a144ab863bd561112fa390c9c0cec99046",
        "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json": "dcbda00191779bcbdd9682ee75a570792bee071e38a7f4604cf97fab14e5c06c",
        "backend/data/benchmarks/retrieval_independent_v1/source_dedup_report.json": "a2c94bf0adc491fd37054ab834b5c13bfaca27442ddf9357c230da093259e5c2",
    }
    assert {path: v2.sha256_file(PROJECT_ROOT / path) for path in expected} == expected
    report = json.loads((PROJECT_ROOT / "backend/data/benchmarks/retrieval_independent_v1/source_dedup_report.json").read_text(encoding="utf-8"))
    assert report["algorithm"] == v2.V1_ALGORITHM
    assert v2.load_policy(POLICY_PATH)["bindings"]["v1_dedup_report_fingerprint"] == v2.V1_DEDUP_REPORT_FINGERPRINT


def test_original_v1_filename_and_text_duplicate_behavior_is_unchanged():
    left = _v1_record("independent:A", "A", "same.pdf", "alpha beta gamma delta " * 20)
    right = _v1_record("independent:B", "B", "same.pdf", "entirely different text")
    eligible, report = v1.deduplicate_against_development([right, left], [])
    assert [item["source_id"] for item in eligible] == ["independent:A"]
    assert report["algorithm"] == v2.V1_ALGORITHM
    assert report["near_duplicate_threshold"] == v2.NEAR_DUPLICATE_THRESHOLD
    assert any("new_pool_duplicate_source_identity" in item["reasons"] for item in report["exclusions"])


def test_cached_similarity_is_exact_v1_and_bounded():
    texts = {
        "A": "Tranh chấp hợp đồng vay tài sản " * 20,
        "B": "TRANH CHẤP HỢP ĐỒNG VAY TÀI SẢN " * 20,
        "C": "Quyết định giám đốc thẩm độc lập " * 20,
    }
    pairs = [("A", "B"), ("A", "C"), ("B", "C")]
    result = v2.cached_text_similarities(texts, pairs)
    for left, right in pairs:
        assert result["scores"][tuple(sorted((left, right)))] == v1.dedup_text_similarity(texts[left], texts[right])
    assert result["shingle_build_count"] == 3
    assert result["naive_shingle_build_count"] == 6
    with pytest.raises(v2.DedupPolicyV2Error, match="source limit"):
        v2.cached_text_similarities(texts, pairs, max_sources=2)
    with pytest.raises(v2.DedupPolicyV2Error, match="pair limit"):
        v2.cached_text_similarities(texts, pairs, max_pairs=2)


def test_validation_failure_does_not_mutate_inputs_or_frozen_files(policy):
    before_policy = POLICY_PATH.read_bytes()
    before_fixture = FIXTURE_PATH.read_bytes()
    invalid = _pair(filename_equal=True, structured_sha_equal=None)
    snapshot = copy.deepcopy(invalid)
    v2.evaluate_pair(policy, invalid)
    assert invalid == snapshot
    assert POLICY_PATH.read_bytes() == before_policy
    assert FIXTURE_PATH.read_bytes() == before_fixture


def test_v2_module_has_no_network_source_review_or_promotion_calls():
    source = (PROJECT_ROOT / "backend/scripts/Evaluation/benchmarking/dedup_policy_v2.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_roots.isdisjoint({"requests", "httpx", "urllib", "socket", "subprocess"})
    forbidden_calls = {"record_source_review", "promote_source", "transition_manifest", "write_text", "write_bytes"}
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert calls.isdisjoint(forbidden_calls)
