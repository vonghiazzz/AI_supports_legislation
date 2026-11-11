import copy
import inspect
from pathlib import Path

import pytest

from backend.scripts.Evaluation.benchmarking import dedup_policy_v2 as v2
from backend.scripts.Evaluation.benchmarking import dedup_policy_v2_independent as r6
from backend.scripts.Evaluation.benchmarking import dedup_policy_v2_safety as r7
from backend.scripts.Evaluation.benchmarking.source_intake import (
    SourceIntakeError,
    validate_source_review_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_ID = r7.PROTOCOL_ID + "_test"


@pytest.fixture(scope="module")
def real_recomputation():
    records, context = r6._load_inventory(PROJECT_ROOT)
    targets = {
        item["source_id"]
        for item in records
        if isinstance(item["acquisition_id"], str)
        and item["acquisition_id"].startswith("batch2dc-")
    }
    generated = r6.generate_candidates(records, targets)
    results = r6.evaluate_candidates(context["policy"], generated)
    return records, context, generated, results


@pytest.fixture(scope="module")
def outputs():
    return r7.build_outputs(PROJECT_ROOT, "a" * 40, RUN_ID)


def _record(source_id, text, **updates):
    value = {
        "source_id": source_id,
        "acquisition_id": source_id.split(":")[-1],
        "population": "INDEPENDENT",
        "source_name": source_id[-1] + ".pdf",
        "source_path": "raw.pdf",
        "structured_path": "structured.json",
        "raw_sha256": source_id[-1] * 64,
        "structured_sha256": (source_id[-1].lower() or "a") * 64,
        "case_id": source_id[-1] + "/2026/DS-ST",
        "judgment_number": source_id[-1] + "/2026/DS-ST",
        "court": "court-" + source_id[-1],
        "decision_date": "2026-01-01",
        "decision_year": 2026,
        "document_type": "CIVIL",
        "procedure_stage": "so_tham",
        "related_decision_ids": [],
        "underlying_dispute_id": None,
        "identity_verified": True,
        "text": text,
    }
    value.update(updates)
    return value


def test_baseline_provenance_is_frozen():
    assert r7.PARENT_GIT_SHA == "4f64f160d6810f971824f1be254454ff5c0ad1a3"
    assert v2.sha256_file(PROJECT_ROOT / r6.MANIFEST_RELATIVE) == r6.MANIFEST_SHA256
    assert v2.load_and_validate_policy(PROJECT_ROOT)["policy_fingerprint"] == (
        "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"
    )


def test_targeted_027_relation_is_independent_dispute(outputs):
    rows = outputs["targeted_dispute_adjudication.json"]["adjudications"]
    row = next(item for item in rows if item["left"]["source_id"].endswith("027"))
    assert row["classification"] == "DIFFERENT_DOCUMENTS_INDEPENDENT_DISPUTES"
    assert row["same_legal_document"] is False
    assert row["same_underlying_dispute"] is False
    assert row["shared_identifiers"] == ["104/2026/QĐ-PT"]
    assert row["v2_reviewed_pair_decision"]["dispute_relation"] == "REVIEWED_DISTINCT_DISPUTE_RELATION"


def test_targeted_029_relations_are_independent_disputes(outputs):
    rows = outputs["targeted_dispute_adjudication.json"]["adjudications"]
    selected = [item for item in rows if item["left"]["source_id"].endswith("029")]
    assert len(selected) == 2
    assert all(item["classification"] == "DIFFERENT_DOCUMENTS_INDEPENDENT_DISPUTES" for item in selected)
    assert {item["relationship_origin"] for item in selected} == {
        "SHARED_LOCAL_DECISION_NUMBER_90_2026_QD_PT",
        "NORMALIZED_JUDGMENT_NUMBER_COLLISION_182_2026_DS_PT",
    }


def test_target_evidence_hashes_and_direct_provenance_are_retained(outputs):
    for row in outputs["targeted_dispute_adjudication.json"]["adjudications"]:
        assert row["direct_or_propagated"] == "DIRECT_TEXT_AND_IDENTITY_EVIDENCE"
        assert row["left"]["raw_pdf"]["sha256"] != row["right"]["raw_pdf"]["sha256"]
        assert row["left"]["structured"]["sha256"] != row["right"]["structured"]["sha256"]
        assert row["left"]["header_excerpt"]
        assert row["right"]["header_excerpt"]


def test_original_five_dispute_blockers_remain_protected(real_recomputation):
    records, _, _, results = real_recomputation
    targets = {item["source_id"] for item in records if isinstance(item["acquisition_id"], str) and item["acquisition_id"].startswith("batch2dc-")}
    decisions = r6.derive_source_decisions(records, targets, results)
    blocked = {
        item["acquisition_id"]
        for item in decisions
        if item["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED"
    }
    assert r7.ORIGINAL_UNRESOLVED <= blocked


def test_reference_oracle_is_independent_of_production_generator():
    source = inspect.getsource(r6.exhaustive_oracle_candidates)
    assert "generate_candidates(" not in source
    assert "dedup_text_similarity" in source


def test_independent_oracle_matches_generator_for_all_signal_controls():
    shared = "tranh chấp quyền sử dụng đất " * 30
    records = [
        _record("independent:A", shared),
        _record("independent:B", shared.upper()),
        _record("independent:C", "c", raw_sha256="A" * 64),
        _record("independent:D", "d", source_name="A.pdf"),
        _record("independent:E", "e", related_decision_ids=["A/2026/DS-ST"]),
    ]
    generated = r6.generate_candidates(records, {"independent:A"})
    observed = {
        tuple(sorted((item["evidence"]["left_source_id"], item["evidence"]["right_source_id"])))
        for item in generated["pairs"]
    }
    oracle = r6.exhaustive_oracle_candidates(records, {"independent:A"})
    assert observed == oracle
    assert len(observed) == len(generated["pairs"])


def test_oracle_and_generator_are_order_invariant():
    records = [_record("independent:A", "a"), _record("independent:B", "b", source_name="A.pdf"), _record("independent:C", "c")]
    forward = r6.exhaustive_oracle_candidates(records, {"independent:A"})
    reverse = r6.exhaustive_oracle_candidates(list(reversed(records)), {"independent:A"})
    assert forward == reverse
    assert r6.generate_candidates(records, {"independent:A"}) == r6.generate_candidates(list(reversed(records)), {"independent:A"})


def test_fabricated_review_completion_is_rejected(real_recomputation):
    records, _, _, _ = real_recomputation
    source = next(item for item in records if item["source_id"].endswith("005"))
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    forged = copy.deepcopy(payload)
    forged["completion_status"]["authoritative_review_completed"] = True
    forged["review_decisions"]["review_completed"] = True
    forged["decision_fingerprint"] = v2.canonical_fingerprint(forged, excluded_key="decision_fingerprint")
    with pytest.raises(r6.IndependentCoverageError, match="Authoritative review completion"):
        r6.validate_review_payload_v2(forged)


def test_missing_document_type_prevents_evidence_completion(real_recomputation):
    records, _, _, results = real_recomputation
    source = next(item for item in records if item["source_id"] == "independent:batch2dc-20261105-027")
    related = [item for item in results if source["source_id"] in (item["evidence"]["left_source_id"], item["evidence"]["right_source_id"])]
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, related)
    assert payload["completion_status"]["evidence_collection_complete"] is False
    assert "EVIDENCE_COLLECTION_INCOMPLETE" in payload["completion_status"]["completion_blockers"]


def test_unresolved_related_decision_prevents_adjudication(real_recomputation):
    records, _, _, results = real_recomputation
    source = next(item for item in records if item["source_id"] == "independent:batch2dc-20261105-008")
    related = [item for item in results if source["source_id"] in (item["evidence"]["left_source_id"], item["evidence"]["right_source_id"])]
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, related)
    assert payload["completion_status"]["evidence_adjudicated"] is False
    assert "PAIR_ADJUDICATION_UNRESOLVED" in payload["completion_status"]["completion_blockers"]


def test_unverified_identity_prevents_collection_completion():
    source = _record("independent:A", "a", identity_verified=False)
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    assert payload["completion_status"]["evidence_collection_complete"] is False


def test_missing_provenance_and_unsupported_policy_are_rejected(real_recomputation):
    source = next(item for item in real_recomputation[0] if item["source_id"].endswith("005"))
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    missing = copy.deepcopy(payload)
    del missing["decision_provenance"]
    with pytest.raises(r6.IndependentCoverageError, match="root fields"):
        r6.validate_review_payload_v2(missing)
    wrong = copy.deepcopy(payload)
    wrong["policy_binding"]["policy_fingerprint"] = "0" * 64
    wrong["decision_fingerprint"] = v2.canonical_fingerprint(wrong, excluded_key="decision_fingerprint")
    with pytest.raises(r6.IndependentCoverageError, match="policy binding"):
        r6.validate_review_payload_v2(wrong)


def test_v1_payload_contract_remains_separate(real_recomputation):
    source = next(item for item in real_recomputation[0] if item["source_id"].endswith("005"))
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    with pytest.raises(SourceIntakeError, match="invalid root fields"):
        validate_source_review_payload(payload)


def test_authoritative_gate_rejects_stale_source_and_manifest(real_recomputation):
    records, _, _, _ = real_recomputation
    source = next(item for item in records if item["source_id"].endswith("005"))
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    stale = copy.deepcopy(source)
    stale["raw_sha256"] = "0" * 64
    result = r6.verify_gate_v2_authoritatively(payload, stale, "0" * 64, [], [], [])
    assert "CALLER_FINDINGS_MISMATCH" in result["failures"]
    assert "STALE_MANIFEST_BINDING" in result["failures"]
    assert result["mutation_performed"] is False


def test_authoritative_gate_rejects_caller_findings_mismatch(real_recomputation):
    source = next(item for item in real_recomputation[0] if item["source_id"].endswith("005"))
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    forged = copy.deepcopy(payload)
    forged["source_findings"]["nonblocking_audit_signals"] = ["FABRICATED"]
    forged["decision_fingerprint"] = v2.canonical_fingerprint(forged, excluded_key="decision_fingerprint")
    result = r6.verify_gate_v2_authoritatively(forged, source, r6.MANIFEST_SHA256, [], [], [])
    assert result["mismatched_fields"] == ["source_findings"]
    assert "CALLER_FINDINGS_MISMATCH" in result["failures"]


def test_canonical_025_and_noncanonical_026_are_consistent(outputs):
    report = outputs["canonical_retention_consistency.json"]
    assert report["consistency"] == "PASS"
    assert report["canonical_source_id"].endswith("025")
    assert report["noncanonical_source_id"].endswith("026")
    assert "CONFIRMED_EXACT_DUPLICATE" not in report["canonical_blocking_reasons"]
    assert "CONFIRMED_EXACT_DUPLICATE" in report["noncanonical_blocking_reasons"]
    assert report["canonical_provenance_retained"] is True
    assert report["canonical_gate"]["ready_for_separately_authorized_promotion"] is False


def test_cross_version_and_existing_rejections_remain_gate_failures(real_recomputation):
    source = next(item for item in real_recomputation[0] if item["source_id"].endswith("005"))
    payload = r6.build_review_payload_v2(source, r6.MANIFEST_SHA256, [])
    result = r6.verify_gate_v2_authoritatively(payload, source, r6.MANIFEST_SHA256, [], ["old"], ["v1-overlap"])
    assert "EXISTING_REJECTION_REASON_DISCARDED" in result["failures"]
    assert "CROSS_VERSION_OVERLAP_OR_DISPUTE" in result["failures"]


def test_output_artifacts_are_self_fingerprinted(outputs):
    for payload in outputs.values():
        assert payload["artifact_fingerprint"] == v2.canonical_fingerprint(payload, excluded_key="artifact_fingerprint")


def test_no_source_or_benchmark_mutation(outputs):
    summary = outputs["validation_summary.json"]
    assert summary["manifest_mutated"] is False
    assert summary["source_review_calls"] == 0
    assert summary["promotion_calls"] == 0
    assert summary["v1_modified"] is False
    assert summary["v2_policy_modified"] is False
    assert v2.sha256_file(PROJECT_ROOT / r6.MANIFEST_RELATIVE) == r6.MANIFEST_SHA256
