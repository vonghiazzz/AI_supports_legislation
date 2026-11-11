import copy
import json
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as v2
from backend.scripts.Evaluation.benchmarking import dedup_policy_v2_independent as r6
from backend.scripts.Evaluation.benchmarking import independent_benchmark as v1
from backend.scripts.Evaluation.benchmarking.source_intake import (
    SourceIntakeError,
    validate_source_review_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = r6.PROTOCOL_ID + "_test"


@pytest.fixture(scope="module")
def inventory_context():
    return r6._load_inventory(PROJECT_ROOT)


@pytest.fixture(scope="module")
def independent_outputs():
    return r6.build_independent_outputs(PROJECT_ROOT, "a" * 40, RUN_ID)


def _record(source_id, text, **updates):
    value = {
        "source_id": source_id,
        "acquisition_id": source_id.split(":")[-1],
        "population": "INDEPENDENT",
        "source_name": source_id.split(":")[-1] + ".pdf",
        "source_path": "raw.pdf",
        "structured_path": "structured.json",
        "raw_sha256": (source_id[-1] * 64)[:64],
        "structured_sha256": ((source_id[-1] + "f") * 64)[:64],
        "case_id": "1/2026/DS-ST-" + source_id[-1],
        "judgment_number": "1/2026/DS-ST-" + source_id[-1],
        "court": "court-" + source_id[-1],
        "decision_date": "2026-01-0" + str((ord(source_id[-1]) % 8) + 1),
        "decision_year": 2026,
        "document_type": None,
        "procedure_stage": "so_tham",
        "related_decision_ids": [],
        "underlying_dispute_id": None,
        "text": text,
    }
    value.update(updates)
    return value


def _generated_pair_keys(generated):
    return {
        tuple(sorted((item["evidence"]["left_source_id"], item["evidence"]["right_source_id"])))
        for item in generated["pairs"]
    }


def test_baseline_and_frozen_policy_bindings():
    assert v2.sha256_file(PROJECT_ROOT / r6.MANIFEST_RELATIVE) == r6.MANIFEST_SHA256
    policy = v2.load_and_validate_policy(PROJECT_ROOT)
    assert policy["policy_fingerprint"] == "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"
    assert policy["activation"]["active"] is False


def test_live_inventory_coverage_is_independent_and_complete(inventory_context):
    records, _ = inventory_context
    assert len(records) == 469
    assert sum(item["population"] == "DEVELOPMENT" for item in records) == 107
    assert sum(item["population"] == "INDEPENDENT" for item in records) == 362
    assert len({item["source_id"] for item in records}) == 469


def test_real_candidate_generation_covers_full_comparison_inventory(independent_outputs):
    coverage = independent_outputs["independent_candidate_coverage.json"]
    assert coverage["generation_source"] == "LIVE_VERIFIED_INVENTORY_NOT_FROZEN_CONTROLS"
    assert coverage["inventory_count"] == 469
    assert coverage["target_count"] == 46
    assert coverage["comparison_count"] == 20493
    assert coverage["shingle_build_count"] == 469
    assert coverage["candidate_pair_count"] == 26
    assert coverage["candidate_generation_protocol"]["fixture_used_during_generation"] is False


def test_bounded_candidate_recall_against_independent_exhaustive_oracle():
    shared = "tranh chấp hợp đồng vay tài sản " * 30
    records = [
        _record("independent:A", shared),
        _record("independent:B", shared.upper()),
        _record("independent:C", "nội dung riêng", source_name="A.pdf"),
        _record("development:D", "hoàn toàn khác", population="DEVELOPMENT"),
    ]
    generated = r6.generate_candidates(records, {"independent:A"})
    observed = _generated_pair_keys(generated)
    oracle = set()
    left = records[0]
    for right in records[1:]:
        signals = r6._metadata_signals(left, right)
        similarity = v1.dedup_text_similarity(left["text"], right["text"])
        if signals or similarity >= v2.NEAR_DUPLICATE_THRESHOLD:
            oracle.add(tuple(sorted((left["source_id"], right["source_id"]))))
    assert observed == oracle


def test_unrelated_development_pair_is_not_candidate():
    records = [
        _record("independent:A", "alpha unique content"),
        _record(
            "development:B",
            "beta unrelated material",
            population="DEVELOPMENT",
            acquisition_id=None,
            source_name="different.pdf",
        ),
    ]
    assert r6.generate_candidates(records, {"independent:A"})["pairs"] == []


@pytest.mark.parametrize(
    ("field", "signal"),
    [("raw_sha256", "RAW_SHA_EQUAL"), ("structured_sha256", "STRUCTURED_SHA_EQUAL")],
)
def test_positive_sha_duplicates_are_never_pruned(field, signal):
    left = _record("independent:A", "left")
    right = _record("independent:B", "right", **{field: left[field]})
    generated = r6.generate_candidates([left, right], {left["source_id"]})
    assert generated["pairs"][0]["signals"] == [signal]


def test_high_similarity_candidate_is_retained():
    text = "quyết định công nhận sự thỏa thuận của các đương sự " * 30
    left = _record("independent:A", text)
    right = _record("independent:B", text.upper())
    generated = r6.generate_candidates([left, right], {left["source_id"]})
    assert "TEXT_SIMILARITY_GE_0_72" in generated["pairs"][0]["signals"]


def test_weak_metadata_collision_is_generated_but_not_forced_duplicate(inventory_context):
    policy = inventory_context[1]["policy"]
    left = _record("independent:A", "left", source_name="same.pdf")
    right = _record("independent:B", "right", source_name="same.pdf")
    generated = r6.generate_candidates([left, right], {left["source_id"]})
    result = r6.evaluate_candidates(policy, generated)[0]["decision"]
    assert result["same_document_relation"] == "WEAK_ASSOCIATION_DISTINCT_CONTROL"
    assert result["overall_blocking"] is False


def test_related_decision_leakage_fails_closed(inventory_context):
    policy = inventory_context[1]["policy"]
    left = _record("independent:A", "left", judgment_number="1/2026/DS-ST")
    right = _record("independent:B", "right", related_decision_ids=["1/2026/DS-ST"])
    generated = r6.generate_candidates([left, right], {left["source_id"]})
    result = r6.evaluate_candidates(policy, generated)[0]["decision"]
    assert result["dispute_relation"] == "DISPUTE_RELATION_UNRESOLVED"
    assert result["overall_blocking"] is True


def test_missing_context_evidence_fails_closed(inventory_context):
    policy = inventory_context[1]["policy"]
    left = _record("independent:A", "left", source_name="same.pdf", court=None, decision_date=None, decision_year=None, procedure_stage=None)
    right = _record("development:B", "right", population="DEVELOPMENT", acquisition_id=None, source_name="same.pdf", court=None, decision_date=None, decision_year=None, procedure_stage=None)
    result = r6.evaluate_candidates(policy, r6.generate_candidates([left, right], {left["source_id"]}))[0]["decision"]
    assert result["development_relation"] == "AMBIGUOUS_DEVELOPMENT_OVERLAP"
    assert result["overall_blocking"] is True


def test_r5_fixture_is_only_posthoc_oracle(independent_outputs):
    comparison = independent_outputs["candidate_pair_comparison.json"]
    assert comparison["fixture_role"] == "POST_HOC_REGRESSION_ORACLE_ONLY"
    assert comparison["fixture_recall"] == 1.0
    assert comparison["missing_from_independent"] == []
    assert comparison["fixture_pair_count"] == 23
    assert comparison["independent_pair_count"] == 26
    assert len(comparison["extra_in_independent"]) == 3


def test_independent_source_decisions_are_not_fixture_membership(independent_outputs):
    result = independent_outputs["source_level_v2_recomputation.json"]
    assert result["membership_derived_from_fixture"] is False
    assert result["independent_counterfactuals"] == {
        "S0_current_strict_v1_reference": 23,
        "S1_weak_signal_reclassification": 37,
        "S2_plus_exact_duplicate_canonical_retention": 38,
    }
    assert result["outcome_counts"] == {
        "ready": 38,
        "blocked_fail_closed": 7,
        "noncanonical_rejected": 1,
    }


def test_new_relations_block_027_and_029(independent_outputs):
    sources = {
        item["acquisition_id"]: item
        for item in independent_outputs["source_level_v2_recomputation.json"]["sources"]
    }
    for acquisition_id in ("batch2dc-20261105-027", "batch2dc-20261105-029"):
        assert sources[acquisition_id]["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED"
        assert "DISPUTE_RELATION_UNRESOLVED" in sources[acquisition_id]["independent_blockers"]


def test_exact_duplicate_canonical_order_is_independently_derived(independent_outputs):
    sources = {
        item["acquisition_id"]: item
        for item in independent_outputs["source_level_v2_recomputation.json"]["sources"]
    }
    assert sources["batch2dc-20261105-025"]["canonical_source_id"] == "independent:batch2dc-20261105-025"
    assert sources["batch2dc-20261105-025"]["prospective_review_readiness"] == "READY_FOR_FUTURE_V2_REVIEW"
    assert sources["batch2dc-20261105-026"]["noncanonical_exact_duplicate"] is True


def test_review_v2_payload_is_incompatible_with_v1_validator(inventory_context):
    records, _ = inventory_context
    source = next(item for item in records if item["source_id"] == "independent:batch2dc-20261105-005")
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    r6.validate_review_payload_v2(payload)
    with pytest.raises(SourceIntakeError, match="invalid root fields"):
        validate_source_review_payload(payload)


def test_policy_fingerprint_and_unknown_review_versions_fail_closed(inventory_context):
    source = next(item for item in inventory_context[0] if item["source_id"] == "independent:batch2dc-20261105-005")
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    wrong_policy = copy.deepcopy(payload)
    wrong_policy["policy_binding"]["policy_fingerprint"] = "0" * 64
    wrong_policy["decision_fingerprint"] = v2.canonical_fingerprint(wrong_policy, excluded_key="decision_fingerprint")
    with pytest.raises(r6.IndependentCoverageError, match="policy binding"):
        r6.validate_review_payload_v2(wrong_policy)
    unknown = copy.deepcopy(payload)
    unknown["review_version"] = "unknown"
    unknown["decision_fingerprint"] = v2.canonical_fingerprint(unknown, excluded_key="decision_fingerprint")
    with pytest.raises(r6.IndependentCoverageError, match="Unsupported"):
        r6.validate_review_payload_v2(unknown)


def test_gate_preserves_rejections_and_cross_version_blockers(inventory_context):
    source = next(item for item in inventory_context[0] if item["source_id"] == "independent:batch2dc-20261105-005")
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    gate = r6.evaluate_gate_v2(payload, payload["source_binding"], ["historical-rejection"], ["v1-overlap"])
    assert gate["ready_for_separately_authorized_promotion"] is False
    assert gate["failures"] == [
        "CROSS_VERSION_OVERLAP_OR_DISPUTE",
        "EXISTING_REJECTION_REASON_DISCARDED",
    ]
    assert gate["mutation_performed"] is False


def test_gate_can_only_report_readiness_not_promote(inventory_context):
    source = next(item for item in inventory_context[0] if item["source_id"] == "independent:batch2dc-20261105-005")
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    gate = r6.evaluate_gate_v2(payload, payload["source_binding"], [], [])
    assert gate == {
        "gate_version": r6.GATE_VERSION,
        "ready_for_separately_authorized_promotion": True,
        "failures": [],
        "mutation_performed": False,
    }
    assert payload["eligibility_authorization"]["authorized"] is False


def test_all_outputs_are_self_fingerprinted(independent_outputs):
    for value in independent_outputs.values():
        assert value["artifact_fingerprint"] == v2.canonical_fingerprint(
            value, excluded_key="artifact_fingerprint"
        )


def test_no_protected_state_mutation(independent_outputs):
    assert independent_outputs["validation_summary.json"] == {
        **{key: independent_outputs["validation_summary.json"][key] for key in ("protocol_id", "run_id", "execution_git_sha", "parent_git_sha", "manifest_sha256", "policy_fingerprint", "artifact_fingerprint")},
        "result": "PASS_WITH_R5_COVERAGE_DIFFERENCE",
        "fixture_used_as_generator": False,
        "source_review_calls": 0,
        "promotion_calls": 0,
        "manifest_mutated": False,
        "v1_modified": False,
        "v2_policy_modified": False,
        "benchmark_or_sampling_mutated": False,
        "network_used": False,
        "retrieval_or_model_used": False,
    }
    assert v2.sha256_file(PROJECT_ROOT / r6.MANIFEST_RELATIVE) == r6.MANIFEST_SHA256
