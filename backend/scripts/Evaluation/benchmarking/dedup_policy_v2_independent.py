"""Independent Batch2DC evidence coverage and review-contract V2 scaffold.

This module is read-only with respect to source state.  Candidate pairs are
generated from the live verified inventory; frozen R3/R4 controls are accepted
only by the comparison helper after recomputation.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import dedup_policy_v2 as policy_v2
from . import independent_benchmark as v1


PROTOCOL_ID = "independent_batch2dc_v2_evidence_coverage_review_contract_v1"
PARENT_GIT_SHA = "1face92fba4d4e3deaeb052c444319cef6e83311"
MANIFEST_RELATIVE = Path("backend/dataLaw/independent_intake/acquisition_manifest.json")
MANIFEST_SHA256 = "411a5ad54b26e640d190bc04b76fbbbfb39896c2d2b6b3fd96450347a8fec08a"
REVIEW_VERSION = "independent_source_review_v2"
FINDINGS_PRODUCER_VERSION = "independent_dedup_findings_v2"
GATE_VERSION = "independent_eligibility_gate_v2"
REVIEW_ROOT_FIELDS = {
    "review_version",
    "policy_binding",
    "source_binding",
    "evidence_observations",
    "pair_classifications",
    "source_findings",
    "review_decisions",
    "completion_status",
    "eligibility_authorization",
    "decision_provenance",
    "decision_fingerprint",
}


class IndependentCoverageError(ValueError):
    """Independent evidence, V2 contract, or gate validation failed."""


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise IndependentCoverageError(f"Invalid JSON: {path}") from exc


def _full_text(payload: Mapping[str, Any]) -> str:
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise IndependentCoverageError("Structured evidence has no pages")
    texts = []
    for page in pages:
        if not isinstance(page, dict) or not isinstance(page.get("text"), str):
            raise IndependentCoverageError("Structured page text is invalid")
        texts.append(page["text"])
    return "\n".join(texts)


def _load_inventory(project_root: Path) -> tuple[list[dict], dict]:
    root = Path(project_root).resolve()
    manifest_path = root / MANIFEST_RELATIVE
    if policy_v2.sha256_file(manifest_path) != MANIFEST_SHA256:
        raise IndependentCoverageError("Acquisition manifest binding changed")
    manifest = _json(manifest_path)
    policy = policy_v2.load_and_validate_policy(root)
    development = v1.inventory_judgment_sources(root, include_private_text=True)
    records = []
    for item in development:
        records.append(
            {
                "source_id": item["source_id"],
                "acquisition_id": None,
                "population": "DEVELOPMENT",
                "source_name": item["source_name"],
                "source_path": item["source_path"],
                "structured_path": item["structured_path"],
                "raw_sha256": item["source_sha256"],
                "structured_sha256": item["structured_sha256"],
                "case_id": item["case_id"],
                "judgment_number": None,
                "court": item["court"],
                "decision_date": None,
                "decision_year": item["year"],
                "document_type": None,
                "procedure_stage": item["procedure_stage"],
                "related_decision_ids": [],
                "underlying_dispute_id": None,
                "identity_verified": False,
                "text": item["_dedup_text"],
            }
        )
    candidate_statuses = {"extracted", "metadata_verified", "dedup_checked", "eligible"}
    independent_count = 0
    for source in sorted(manifest["sources"], key=lambda value: value["source_id"]):
        if source["acquisition_status"] not in candidate_statuses:
            continue
        for field in ("judgment_number", "case_id", "court", "decision_date", "procedure_stage"):
            if not source.get(field) or source["metadata_verification"].get(field) is not True:
                raise IndependentCoverageError(f"Independent identity incomplete: {source['source_id']}")
        raw_path = root / source["stored_pdf_path"]
        structured_path = root / source["structured_path"]
        if raw_path.is_symlink() or structured_path.is_symlink():
            raise IndependentCoverageError("Symlink evidence is forbidden")
        if policy_v2.sha256_file(raw_path) != source["sha256"]:
            raise IndependentCoverageError(f"Raw evidence changed: {source['source_id']}")
        if policy_v2.sha256_file(structured_path) != source["structured_sha256"]:
            raise IndependentCoverageError(f"Structured evidence changed: {source['source_id']}")
        structured = _json(structured_path)
        records.append(
            {
                "source_id": source["source_id"],
                "acquisition_id": source["acquisition_id"],
                "population": "INDEPENDENT",
                "source_name": source["original_filename"],
                "source_path": source["stored_pdf_path"],
                "structured_path": source["structured_path"],
                "raw_sha256": source["sha256"],
                "structured_sha256": source["structured_sha256"],
                "case_id": source["case_id"],
                "judgment_number": source["judgment_number"],
                "court": source["court"],
                "decision_date": source["decision_date"],
                "decision_year": int(source["decision_date"][:4]),
                "document_type": None,
                "procedure_stage": source["procedure_stage"],
                "related_decision_ids": list(source["related_decision_ids"]),
                "underlying_dispute_id": source["underlying_dispute_id"],
                "identity_verified": True,
                "text": _full_text(structured),
            }
        )
        independent_count += 1
    if len(development) != 107 or independent_count != 362 or len(records) != 469:
        raise IndependentCoverageError("Relevant inventory must be 107 DEV + 362 independent")
    if len({item["source_id"] for item in records}) != len(records):
        raise IndependentCoverageError("Relevant inventory source IDs are not unique")
    return records, {"manifest": manifest, "policy": policy}


def _optional_equal(left: Any, right: Any) -> bool | None:
    if left is None or right is None or left == "" or right == "":
        return None
    return left == right


def _normalized_optional_equal(left: Any, right: Any) -> bool | None:
    if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
        return None
    return v1._normalize_text(left) == v1._normalize_text(right)


def _metadata_signals(left: Mapping[str, Any], right: Mapping[str, Any]) -> set[str]:
    signals = set()
    if left["raw_sha256"] == right["raw_sha256"]:
        signals.add("RAW_SHA_EQUAL")
    if left["structured_sha256"] and left["structured_sha256"] == right["structured_sha256"]:
        signals.add("STRUCTURED_SHA_EQUAL")
    if v1.canonical_source_name(left["source_name"]) == v1.canonical_source_name(right["source_name"]):
        signals.add("FILENAME_EQUAL")
    if _normalized_optional_equal(left["judgment_number"], right["judgment_number"]) is True:
        signals.add("NORMALIZED_JUDGMENT_EQUAL")
    if _optional_equal(left["case_id"], right["case_id"]) is True:
        signals.add("CASE_ID_EQUAL")
    if (
        left["underlying_dispute_id"]
        and left["underlying_dispute_id"] == right["underlying_dispute_id"]
    ):
        signals.add("SAME_UNDERLYING_DISPUTE")
    left_related = set(left["related_decision_ids"])
    right_related = set(right["related_decision_ids"])
    if (
        left_related & right_related
        or left["judgment_number"] in right_related
        or right["judgment_number"] in left_related
    ):
        signals.add("RELATED_DECISION")
    return signals


def _pair_key(left: str, right: str) -> tuple[str, str]:
    if left == right:
        raise IndependentCoverageError("Self-pairs are forbidden")
    return tuple(sorted((left, right)))


def generate_candidates(
    records: Sequence[Mapping[str, Any]],
    target_source_ids: set[str],
    known_duplicate_links: Iterable[tuple[str, str]] = (),
) -> dict:
    by_id = {item["source_id"]: dict(item) for item in records}
    if len(by_id) != len(records) or not target_source_ids <= set(by_id):
        raise IndependentCoverageError("Candidate inventory/targets are invalid")
    shingles = {source_id: v1._dedup_shingles(item["text"]) for source_id, item in by_id.items()}
    known = {_pair_key(left, right) for left, right in known_duplicate_links}
    pairs = []
    comparisons = 0
    for left_id in sorted(target_source_ids):
        for right_id in sorted(by_id):
            if left_id >= right_id and right_id in target_source_ids:
                continue
            if left_id == right_id:
                continue
            key = _pair_key(left_id, right_id)
            if key[0] != left_id and key[0] in target_source_ids:
                continue
            left, right = by_id[key[0]], by_id[key[1]]
            comparisons += 1
            signals = _metadata_signals(left, right)
            if key in known:
                signals.add("AUTHORITATIVE_DUPLICATE_LINK")
            union = shingles[key[0]] | shingles[key[1]]
            similarity = 1.0 if not union else len(shingles[key[0]] & shingles[key[1]]) / len(union)
            if similarity >= policy_v2.NEAR_DUPLICATE_THRESHOLD:
                signals.add("TEXT_SIMILARITY_GE_0_72")
            if not signals:
                continue
            scope = "DEVELOPMENT" if "DEVELOPMENT" in {left["population"], right["population"]} else "INTERNAL"
            evidence = {
                "left_source_id": key[0],
                "right_source_id": key[1],
                "scope": scope,
                "raw_sha_equal": left["raw_sha256"] == right["raw_sha256"],
                "structured_sha_equal": _optional_equal(left["structured_sha256"], right["structured_sha256"]),
                "case_id_equal": _optional_equal(left["case_id"], right["case_id"]),
                "normalized_judgment_equal": _normalized_optional_equal(left["judgment_number"], right["judgment_number"]),
                "filename_equal": v1.canonical_source_name(left["source_name"]) == v1.canonical_source_name(right["source_name"]),
                "court_equal": _optional_equal(left["court"], right["court"]),
                "decision_date_equal": _optional_equal(left["decision_date"], right["decision_date"]),
                "decision_year_equal": _optional_equal(left["decision_year"], right["decision_year"]),
                "document_type_equal": _optional_equal(left["document_type"], right["document_type"]),
                "procedure_stage_equal": _optional_equal(left["procedure_stage"], right["procedure_stage"]),
                "text_similarity": similarity,
                "related_decision": bool({"RELATED_DECISION", "AUTHORITATIVE_DUPLICATE_LINK"} & signals),
                "same_underlying_dispute": (
                    True
                    if "SAME_UNDERLYING_DISPUTE" in signals
                    else False
                    if left["underlying_dispute_id"] and right["underlying_dispute_id"]
                    else None
                ),
                "findings_provenance": sorted(signals),
            }
            pairs.append({"pair_id": f"PAIR-{len(pairs)+1:05d}", "signals": sorted(signals), "evidence": evidence})
    pairs.sort(key=lambda item: (item["evidence"]["left_source_id"], item["evidence"]["right_source_id"]))
    for index, item in enumerate(pairs, start=1):
        item["pair_id"] = f"PAIR-{index:05d}"
    return {
        "inventory_count": len(records),
        "target_count": len(target_source_ids),
        "comparison_count": comparisons,
        "shingle_build_count": len(shingles),
        "candidate_pair_count": len(pairs),
        "pairs": pairs,
    }


def exhaustive_oracle_candidates(
    records: Sequence[Mapping[str, Any]],
    target_source_ids: set[str],
    known_duplicate_links: Iterable[tuple[str, str]] = (),
) -> set[tuple[str, str]]:
    """Independent bounded oracle; deliberately does not call the generator."""
    if len(records) > 64:
        raise IndependentCoverageError("Exhaustive oracle is restricted to <=64 records")
    by_id = {item["source_id"]: dict(item) for item in records}
    if len(by_id) != len(records) or not target_source_ids <= set(by_id):
        raise IndependentCoverageError("Oracle inventory/targets are invalid")
    known = {_pair_key(left, right) for left, right in known_duplicate_links}
    result = set()
    for index, left_id in enumerate(sorted(by_id)):
        for right_id in sorted(by_id)[index + 1 :]:
            if not {left_id, right_id} & target_source_ids:
                continue
            left, right = by_id[left_id], by_id[right_id]
            candidate = False
            candidate |= left["raw_sha256"] == right["raw_sha256"]
            candidate |= bool(
                left["structured_sha256"]
                and left["structured_sha256"] == right["structured_sha256"]
            )
            candidate |= (
                v1.canonical_source_name(left["source_name"])
                == v1.canonical_source_name(right["source_name"])
            )
            candidate |= _normalized_optional_equal(
                left["judgment_number"], right["judgment_number"]
            ) is True
            candidate |= _optional_equal(left["case_id"], right["case_id"]) is True
            candidate |= bool(
                left["underlying_dispute_id"]
                and left["underlying_dispute_id"] == right["underlying_dispute_id"]
            )
            left_related = set(left["related_decision_ids"])
            right_related = set(right["related_decision_ids"])
            candidate |= bool(
                left_related & right_related
                or left["judgment_number"] in right_related
                or right["judgment_number"] in left_related
            )
            candidate |= _pair_key(left_id, right_id) in known
            candidate |= (
                v1.dedup_text_similarity(left["text"], right["text"])
                >= policy_v2.NEAR_DUPLICATE_THRESHOLD
            )
            if candidate:
                result.add(_pair_key(left_id, right_id))
    return result


def evaluate_candidates(policy: Mapping[str, Any], generated: Mapping[str, Any]) -> list[dict]:
    results = []
    seen = set()
    for pair in generated["pairs"]:
        evidence = pair["evidence"]
        key = _pair_key(evidence["left_source_id"], evidence["right_source_id"])
        if key in seen:
            raise IndependentCoverageError("Duplicate generated pair")
        seen.add(key)
        results.append({**copy.deepcopy(pair), "decision": policy_v2.evaluate_pair(policy, evidence)})
    return results


def derive_source_decisions(
    records: Sequence[Mapping[str, Any]], target_source_ids: set[str], results: Sequence[Mapping[str, Any]]
) -> list[dict]:
    by_id = {item["source_id"]: item for item in records}
    exact_raw_groups: dict[str, list[str]] = {}
    for item in records:
        exact_raw_groups.setdefault(item["raw_sha256"], []).append(item["source_id"])
    decisions = []
    for source_id in sorted(target_source_ids):
        related = [item for item in results if source_id in (item["evidence"]["left_source_id"], item["evidence"]["right_source_id"])]
        blockers = sorted(
            {
                item["decision"]["development_relation"]
                if item["evidence"]["scope"] == "DEVELOPMENT"
                else item["decision"]["dispute_relation"]
                if item["decision"]["dispute_blocking"]
                else item["decision"]["same_document_relation"]
                for item in related
                if item["decision"]["overall_blocking"]
            }
        )
        exact_group = sorted(exact_raw_groups[by_id[source_id]["raw_sha256"]])
        canonical = None
        noncanonical = False
        if len(exact_group) > 1:
            ordered = sorted(exact_group, key=lambda value: (by_id[value]["case_id"] or "", value, by_id[value]["raw_sha256"]))
            canonical = ordered[0]
            noncanonical = source_id != canonical
            if not noncanonical:
                blockers = [item for item in blockers if item != "CONFIRMED_EXACT_DUPLICATE"]
        readiness = (
            "REJECT_NONCANONICAL_EXACT_DUPLICATE"
            if noncanonical
            else "BLOCKED_FAIL_CLOSED"
            if blockers
            else "READY_FOR_FUTURE_V2_REVIEW"
        )
        decisions.append(
            {
                "source_id": source_id,
                "acquisition_id": by_id[source_id]["acquisition_id"],
                "pair_ids": [item["pair_id"] for item in related],
                "exact_duplicate_source_ids": exact_group if len(exact_group) > 1 else [],
                "weak_association_pair_ids": [item["pair_id"] for item in related if item["decision"]["weak_signal"]],
                "development_overlap_pair_ids": [item["pair_id"] for item in related if item["evidence"]["scope"] == "DEVELOPMENT"],
                "dispute_pair_ids": [item["pair_id"] for item in related if item["decision"]["dispute_relation"] not in {"NO_DISPUTE_RELATION", "DUPLICATE_INTERNAL_RELATION_NOT_INDEPENDENT_BLOCKER"}],
                "canonical_source_id": canonical,
                "noncanonical_exact_duplicate": noncanonical,
                "independent_blockers": blockers,
                "evidence_gap": (
                    ["DOCUMENT_TYPE_NOT_AUTHORITATIVELY_AVAILABLE"]
                    if any(item["evidence"]["document_type_equal"] is None for item in related)
                    else []
                ),
                "prospective_review_readiness": readiness,
                "decision_provenance": [
                    {"pair_id": item["pair_id"], "signals": item["signals"]}
                    for item in related
                ],
            }
        )
    return decisions


def review_contract_spec() -> dict:
    return {
        "review_version": REVIEW_VERSION,
        "policy_binding_required": ["policy_id", "policy_version", "policy_fingerprint"],
        "source_binding_required": ["source_id", "acquisition_id", "manifest_sha256", "raw_sha256", "structured_sha256", "identity_fingerprint", "identity_verified"],
        "separation": ["evidence_observations", "pair_classifications", "source_findings", "review_decisions", "completion_status", "eligibility_authorization"],
        "typed_findings": ["document_equivalence", "development_overlap", "underlying_dispute", "canonical_duplicate", "nonblocking_audit_signal", "blocking_rejection_reason"],
        "generic_override_allowed": False,
        "missing_evidence_behavior": "FAIL_CLOSED",
        "deterministic_ordering": "pair_id ASC; finding_type ASC then counterpart_source_id ASC",
        "automatic_eligibility": False,
    }


def gate_spec() -> dict:
    return {
        "gate_version": GATE_VERSION,
        "selection": {"V1": "existing V1 gate", "V2": "valid V2 contract required", "UNKNOWN": "REJECT"},
        "requirements": ["exact live source binding", "exact policy fingerprint", "reproducible findings", "all review decisions complete", "no blocking findings", "existing rejection reasons preserved", "cross-version findings clear", "development leakage clear"],
        "promotion_is_separate_authorization": True,
        "manifest_migration_required": True,
        "minimum_manifest_extension": ["review_contract_version", "dedup_policy_id", "dedup_policy_fingerprint", "review_decision_fingerprint", "review_evidence_artifact_binding"],
    }


def validate_review_payload_v2(payload: Mapping[str, Any]) -> dict:
    if not isinstance(payload, Mapping) or set(payload) != REVIEW_ROOT_FIELDS:
        raise IndependentCoverageError("V2 review payload root fields are invalid")
    checked = copy.deepcopy(dict(payload))
    if checked["review_version"] != REVIEW_VERSION:
        raise IndependentCoverageError("Unsupported V2 review version")
    expected_policy = {"policy_id": policy_v2.POLICY_ID, "policy_version": policy_v2.POLICY_VERSION, "policy_fingerprint": "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"}
    if checked["policy_binding"] != expected_policy:
        raise IndependentCoverageError("V2 policy binding mismatch")
    source_fields = set(review_contract_spec()["source_binding_required"])
    if not isinstance(checked["source_binding"], dict) or set(checked["source_binding"]) != source_fields:
        raise IndependentCoverageError("V2 source binding fields are invalid")
    for field in ("evidence_observations", "pair_classifications"):
        if not isinstance(checked[field], list):
            raise IndependentCoverageError(f"{field} must be a list")
    required_findings = {"document_equivalence", "development_overlap", "underlying_dispute", "canonical_duplicate", "nonblocking_audit_signals", "blocking_rejection_reasons"}
    if not isinstance(checked["source_findings"], dict) or set(checked["source_findings"]) != required_findings:
        raise IndependentCoverageError("V2 source findings fields are invalid")
    observations = checked["evidence_observations"]
    classifications = checked["pair_classifications"]
    if any(
        not isinstance(item, dict)
        or set(item) != {"pair_id", "signals", "counterpart_source_id", "evidence_fingerprint", "evidence_complete"}
        for item in observations
    ):
        raise IndependentCoverageError("V2 evidence observations are invalid")
    if any(
        not isinstance(item, dict)
        or set(item) != {"pair_id", "same_document_relation", "development_relation", "dispute_relation", "overall_blocking", "adjudication_status"}
        for item in classifications
    ):
        raise IndependentCoverageError("V2 pair classifications are invalid")
    collection_complete = (
        checked["source_binding"]["identity_verified"] is True
        and all(item["evidence_complete"] is True for item in observations)
    )
    adjudicated = all(item["adjudication_status"] == "ADJUDICATED" for item in classifications)
    expected_completion_blockers = []
    if not collection_complete:
        expected_completion_blockers.append("EVIDENCE_COLLECTION_INCOMPLETE")
    if not adjudicated:
        expected_completion_blockers.append("PAIR_ADJUDICATION_UNRESOLVED")
    completion = checked["completion_status"]
    if not isinstance(completion, dict) or set(completion) != {
        "evidence_collection_complete",
        "evidence_adjudicated",
        "authoritative_review_completed",
        "eligibility_approved",
        "completion_blockers",
    }:
        raise IndependentCoverageError("V2 completion status fields are invalid")
    if (
        completion["evidence_collection_complete"] is not collection_complete
        or completion["evidence_adjudicated"] is not adjudicated
        or completion["eligibility_approved"] is not False
        or completion["completion_blockers"] != expected_completion_blockers
    ):
        raise IndependentCoverageError("Fabricated or inconsistent review completion")
    provenance = checked["decision_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != {
        "findings_producer_version",
        "evidence_fingerprint",
        "review_authority",
    } or provenance["findings_producer_version"] != FINDINGS_PRODUCER_VERSION:
        raise IndependentCoverageError("V2 decision provenance is invalid")
    authoritative = provenance["review_authority"] == "AUTHORIZED_VERSIONED_REVIEW_PRODUCER"
    if completion["authoritative_review_completed"] is not (
        collection_complete and adjudicated and authoritative
    ):
        raise IndependentCoverageError("Authoritative review completion is not supported by evidence/provenance")
    decisions = checked["review_decisions"]
    if not isinstance(decisions, dict) or set(decisions) != {"evidence_collection_attested", "pair_adjudication_attested", "review_completed", "review_disposition"}:
        raise IndependentCoverageError("V2 review decisions fields are invalid")
    if (
        decisions["evidence_collection_attested"] is not collection_complete
        or decisions["pair_adjudication_attested"] is not adjudicated
        or decisions["review_completed"] is not completion["authoritative_review_completed"]
    ):
        raise IndependentCoverageError("V2 review decision attestations are inconsistent")
    if checked["eligibility_authorization"] != {"authorized": False, "gate_version": GATE_VERSION, "reasons": ["SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED"]}:
        raise IndependentCoverageError("Review payload cannot authorize eligibility")
    if checked["decision_fingerprint"] != policy_v2.canonical_fingerprint(checked, excluded_key="decision_fingerprint"):
        raise IndependentCoverageError("V2 decision fingerprint mismatch")
    return checked


def evaluate_gate_v2(
    payload: Mapping[str, Any], live_source_binding: Mapping[str, Any], existing_rejection_reasons: Sequence[str], cross_version_blockers: Sequence[str]
) -> dict:
    checked = validate_review_payload_v2(payload)
    failures = []
    if checked["source_binding"] != dict(live_source_binding):
        failures.append("STALE_SOURCE_BINDING")
    blocking = checked["source_findings"]["blocking_rejection_reasons"]
    if not isinstance(blocking, list) or any(not isinstance(item, str) for item in blocking):
        failures.append("INVALID_BLOCKING_FINDINGS")
    missing_existing = sorted(set(existing_rejection_reasons) - set(blocking))
    if missing_existing:
        failures.append("EXISTING_REJECTION_REASON_DISCARDED")
    if blocking:
        failures.append("BLOCKING_V2_FINDINGS")
    if cross_version_blockers:
        failures.append("CROSS_VERSION_OVERLAP_OR_DISPUTE")
    if not checked["completion_status"]["authoritative_review_completed"]:
        failures.append("REVIEW_NOT_AUTHORITATIVELY_COMPLETE")
    return {"gate_version": GATE_VERSION, "ready_for_separately_authorized_promotion": not failures, "failures": sorted(set(failures)), "mutation_performed": False}


def verify_gate_v2_authoritatively(
    payload: Mapping[str, Any],
    live_source: Mapping[str, Any],
    live_manifest_sha256: str,
    authoritative_pair_results: Sequence[Mapping[str, Any]],
    existing_rejection_reasons: Sequence[str],
    cross_version_blockers: Sequence[str],
    *,
    canonical_disposition: str | None = None,
) -> dict:
    """Pure future-gate verifier; recomputes findings and never promotes."""
    checked = validate_review_payload_v2(payload)
    recomputed = build_review_payload_v2(
        live_source,
        live_manifest_sha256,
        authoritative_pair_results,
        canonical_disposition=canonical_disposition,
        review_authority=checked["decision_provenance"]["review_authority"],
    )
    comparable_fields = {
        "policy_binding",
        "source_binding",
        "evidence_observations",
        "pair_classifications",
        "source_findings",
        "completion_status",
    }
    mismatches = sorted(
        field
        for field in comparable_fields
        if checked[field] != recomputed[field]
    )
    base = evaluate_gate_v2(
        checked,
        recomputed["source_binding"],
        existing_rejection_reasons,
        cross_version_blockers,
    )
    failures = list(base["failures"])
    if mismatches:
        failures.append("CALLER_FINDINGS_MISMATCH")
    if live_manifest_sha256 != MANIFEST_SHA256:
        failures.append("STALE_MANIFEST_BINDING")
    return {
        "gate_version": GATE_VERSION,
        "ready_for_separately_authorized_promotion": not failures,
        "failures": sorted(set(failures)),
        "mismatched_fields": mismatches,
        "authoritative_recomputation_performed": True,
        "complete_inventory_recheck_required_at_mutation_time": True,
        "mutation_performed": False,
    }


def build_review_payload_v2(
    source: Mapping[str, Any],
    manifest_sha256: str,
    pair_results: Sequence[Mapping[str, Any]],
    *,
    canonical_disposition: str | None = None,
    review_authority: str = "SHADOW_STRUCTURAL_PRODUCER",
) -> dict:
    observations = [
        {
            "pair_id": item["pair_id"],
            "signals": item["signals"],
            "counterpart_source_id": item["evidence"]["right_source_id"] if item["evidence"]["left_source_id"] == source["source_id"] else item["evidence"]["left_source_id"],
            "evidence_fingerprint": policy_v2.canonical_fingerprint(item["evidence"]),
            "evidence_complete": all(
                item["evidence"][field] is not None
                for field in (
                    "raw_sha_equal",
                    "structured_sha_equal",
                    "case_id_equal",
                    "court_equal",
                    "decision_date_equal",
                    "decision_year_equal",
                    "document_type_equal",
                    "procedure_stage_equal",
                    "text_similarity",
                )
            ),
        }
        for item in pair_results
    ]
    unresolved_relations = {
        "AMBIGUOUS_STRONG_TEXT_MATCH",
        "AMBIGUOUS_WEAK_ASSOCIATION",
        "AMBIGUOUS_DEVELOPMENT_OVERLAP",
        "DISPUTE_RELATION_UNRESOLVED",
    }
    classifications = [
        {
            "pair_id": item["pair_id"],
            "same_document_relation": item["decision"]["same_document_relation"],
            "development_relation": item["decision"]["development_relation"],
            "dispute_relation": item["decision"]["dispute_relation"],
            "overall_blocking": item["decision"]["overall_blocking"],
            "adjudication_status": (
                "UNRESOLVED"
                if unresolved_relations
                & {
                    item["decision"]["same_document_relation"],
                    item["decision"]["development_relation"],
                    item["decision"]["dispute_relation"],
                }
                else "ADJUDICATED"
            ),
        }
        for item in pair_results
    ]
    blockers = sorted(
        {
            relation
            for item in pair_results
            for enabled, relation in (
                (
                    item["decision"]["same_document_blocking"],
                    item["decision"]["same_document_relation"],
                ),
                (
                    item["decision"]["development_blocking"],
                    item["decision"]["development_relation"],
                ),
                (
                    item["decision"]["dispute_blocking"],
                    item["decision"]["dispute_relation"],
                ),
            )
            if enabled
        }
    )
    if canonical_disposition == "CANONICAL_RETAINED":
        blockers = [item for item in blockers if item != "CONFIRMED_EXACT_DUPLICATE"]
    if canonical_disposition == "NONCANONICAL_REJECTED" and "CONFIRMED_EXACT_DUPLICATE" not in blockers:
        blockers.append("CONFIRMED_EXACT_DUPLICATE")
    source_binding = {"source_id": source["source_id"], "acquisition_id": source["acquisition_id"], "manifest_sha256": manifest_sha256, "raw_sha256": source["raw_sha256"], "structured_sha256": source["structured_sha256"], "identity_fingerprint": policy_v2.canonical_fingerprint({key: source[key] for key in ("case_id", "judgment_number", "court", "decision_date", "procedure_stage")}), "identity_verified": source.get("identity_verified") is True}
    collection_complete = source_binding["identity_verified"] and all(item["evidence_complete"] for item in observations)
    adjudicated = all(item["adjudication_status"] == "ADJUDICATED" for item in classifications)
    completion_blockers = []
    if not collection_complete:
        completion_blockers.append("EVIDENCE_COLLECTION_INCOMPLETE")
    if not adjudicated:
        completion_blockers.append("PAIR_ADJUDICATION_UNRESOLVED")
    authoritative_completed = collection_complete and adjudicated and review_authority == "AUTHORIZED_VERSIONED_REVIEW_PRODUCER"
    payload = {
        "review_version": REVIEW_VERSION,
        "policy_binding": {"policy_id": policy_v2.POLICY_ID, "policy_version": policy_v2.POLICY_VERSION, "policy_fingerprint": "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914"},
        "source_binding": source_binding,
        "evidence_observations": observations,
        "pair_classifications": classifications,
        "source_findings": {"document_equivalence": sorted({item["same_document_relation"] for item in classifications}), "development_overlap": sorted({item["development_relation"] for item in classifications}), "underlying_dispute": sorted({item["dispute_relation"] for item in classifications}), "canonical_duplicate": [canonical_disposition] if canonical_disposition else [], "nonblocking_audit_signals": sorted({signal for item in observations for signal in item["signals"]}), "blocking_rejection_reasons": sorted(blockers)},
        "review_decisions": {"evidence_collection_attested": collection_complete, "pair_adjudication_attested": adjudicated, "review_completed": authoritative_completed, "review_disposition": "STRUCTURALLY_VALID_NOT_AUTHORITATIVE" if not authoritative_completed else "BLOCKED" if blockers else "REVIEW_COMPLETE_NO_BLOCKERS"},
        "completion_status": {"evidence_collection_complete": collection_complete, "evidence_adjudicated": adjudicated, "authoritative_review_completed": authoritative_completed, "eligibility_approved": False, "completion_blockers": completion_blockers},
        "eligibility_authorization": {"authorized": False, "gate_version": GATE_VERSION, "reasons": ["SEPARATE_PROMOTION_AUTHORIZATION_REQUIRED"]},
        "decision_provenance": {"findings_producer_version": FINDINGS_PRODUCER_VERSION, "evidence_fingerprint": policy_v2.canonical_fingerprint(observations), "review_authority": review_authority},
        "decision_fingerprint": "PENDING",
    }
    payload["decision_fingerprint"] = policy_v2.canonical_fingerprint(payload, excluded_key="decision_fingerprint")
    return validate_review_payload_v2(payload)


def compare_with_frozen_fixture(project_root: Path, generated: Mapping[str, Any]) -> dict:
    """Post-hoc oracle comparison; never participates in generation."""
    fixture = policy_v2.validate_evidence_fixture(
        policy_v2.load_evidence_fixture(Path(project_root) / policy_v2.EVIDENCE_FIXTURE_RELATIVE)
    )
    fixture_pairs = {
        _pair_key(item["left_source_id"], item["right_source_id"])
        for item in fixture["pair_controls"]
    }
    generated_pairs = {
        _pair_key(item["evidence"]["left_source_id"], item["evidence"]["right_source_id"])
        for item in generated["pairs"]
    }
    return {
        "fixture_role": "POST_HOC_REGRESSION_ORACLE_ONLY",
        "fixture_pair_count": len(fixture_pairs),
        "independent_pair_count": len(generated_pairs),
        "missing_from_independent": [list(item) for item in sorted(fixture_pairs - generated_pairs)],
        "extra_in_independent": [list(item) for item in sorted(generated_pairs - fixture_pairs)],
        "fixture_recall": len(fixture_pairs & generated_pairs) / len(fixture_pairs),
    }


def build_independent_outputs(project_root: Path, execution_git_sha: str, run_id: str) -> dict[str, dict]:
    root = Path(project_root).resolve()
    if len(execution_git_sha) != 40 or not run_id.startswith(PROTOCOL_ID + "_"):
        raise IndependentCoverageError("Execution identity is invalid")
    records, context = _load_inventory(root)
    targets = {
        item["source_id"]
        for item in records
        if isinstance(item["acquisition_id"], str)
        and item["acquisition_id"].startswith("batch2dc-20261105-")
    }
    links = [
        (
            f"independent:{item['acquisition_id']}",
            f"independent:{item['related_acquisition_id']}",
        )
        for item in context["manifest"]["known_duplicate_links"]
    ]
    generated = generate_candidates(records, targets, links)
    results = evaluate_candidates(context["policy"], generated)
    decisions = derive_source_decisions(records, targets, results)
    comparison = compare_with_frozen_fixture(root, generated)
    if comparison["missing_from_independent"]:
        raise IndependentCoverageError("Independent generator missed frozen positive controls")
    ready = sum(item["prospective_review_readiness"] == "READY_FOR_FUTURE_V2_REVIEW" for item in decisions)
    dispute_blocked = sum(item["prospective_review_readiness"] == "BLOCKED_FAIL_CLOSED" for item in decisions)
    rejected = sum(item["prospective_review_readiness"] == "REJECT_NONCANONICAL_EXACT_DUPLICATE" for item in decisions)
    exact_targets = {
        item["source_id"]
        for item in decisions
        if item["exact_duplicate_source_ids"]
    }
    s1 = 46 - dispute_blocked - len(exact_targets)
    s2 = ready
    signal_counts = {}
    for item in generated["pairs"]:
        for signal in item["signals"]:
            signal_counts[signal] = signal_counts.get(signal, 0) + 1
    evidence_gaps = sorted(
        {
            gap
            for item in decisions
            for gap in item["evidence_gap"]
        }
    )
    by_id = {item["source_id"]: item for item in records}
    review_examples = []
    for decision in decisions:
        related = [item for item in results if item["pair_id"] in decision["pair_ids"]]
        review_examples.append(
            build_review_payload_v2(
                by_id[decision["source_id"]],
                MANIFEST_SHA256,
                related,
                canonical_disposition=(
                    "NONCANONICAL_REJECTED"
                    if decision["noncanonical_exact_duplicate"]
                    else "CANONICAL_RETAINED"
                    if decision["canonical_source_id"] == decision["source_id"]
                    else None
                ),
            )
        )
    common = {
        "protocol_id": PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": PARENT_GIT_SHA,
        "manifest_sha256": MANIFEST_SHA256,
        "policy_fingerprint": context["policy"]["policy_fingerprint"],
    }
    coverage = {
        **common,
        "generation_source": "LIVE_VERIFIED_INVENTORY_NOT_FROZEN_CONTROLS",
        "scope": "46_BATCH2DC_TARGETS_VS_107_DEVELOPMENT_PLUS_362_INDEPENDENT",
        "inventory_count": generated["inventory_count"],
        "target_count": generated["target_count"],
        "comparison_count": generated["comparison_count"],
        "shingle_build_count": generated["shingle_build_count"],
        "candidate_pair_count": generated["candidate_pair_count"],
        "signal_counts": dict(sorted(signal_counts.items())),
        "known_duplicate_link_count": len(links),
        "coverage_status": "COMPLETE_FOR_SUPPORTED_FROZEN_POLICY_SIGNALS",
        "unverified_coverage": [],
        "evidence_gaps_for_adjudication": evidence_gaps,
        "candidate_generation_protocol": {
            "exact_and_metadata": "deterministic complete comparison for every target against full inventory",
            "text": "one cached shingle set per source; exhaustive target-to-inventory Jaccard",
            "threshold": policy_v2.NEAR_DUPLICATE_THRESHOLD,
            "fixture_used_during_generation": False,
        },
    }
    pair_comparison = {
        **common,
        **comparison,
        "extra_pair_investigation": [
            {
                "pair": item,
                "interpretation": "independent live-evidence relation absent from R5 frozen controls",
            }
            for item in comparison["extra_in_independent"]
        ],
    }
    recomputation = {
        **common,
        "candidate_pairs": results,
        "source_count": len(decisions),
        "sources": decisions,
        "independent_counterfactuals": {
            "S0_current_strict_v1_reference": 23,
            "S1_weak_signal_reclassification": s1,
            "S2_plus_exact_duplicate_canonical_retention": s2,
        },
        "outcome_counts": {
            "ready": ready,
            "blocked_fail_closed": dispute_blocked,
            "noncanonical_rejected": rejected,
        },
        "membership_derived_from_fixture": False,
    }
    contract = {
        **common,
        "specification": review_contract_spec(),
        "example_payload_count": len(review_examples),
        "example_payload_fingerprints": [item["decision_fingerprint"] for item in review_examples],
        "activation_authorized": False,
    }
    gate = {
        **common,
        "specification": gate_spec(),
        "activation_authorized": False,
    }
    migration = {
        **common,
        "v1_behavior_changed": False,
        "v1_review_payloads_remain_v1": True,
        "v2_requires_explicit_version": True,
        "unknown_version_behavior": "REJECT",
        "cross_version_comparison_inventory": "ALL_RELEVANT_V1_AND_V2_SOURCES",
        "cohort_atomic_future_execution": [
            "bind exact cohort and pre-mutation manifest SHA",
            "prevalidate every payload and evidence binding",
            "acquire one manifest mutation lock",
            "recheck manifest and source artifacts",
            "apply all V2 review records in memory",
            "validate full manifest and cross-version gates",
            "single atomic manifest replacement",
            "authoritative reload and byte equality check",
            "rollback on post-publication failure",
            "idempotent whole-cohort retry",
            "never mix eligibility promotion with review publication",
        ],
        "minimum_manifest_migration": gate_spec()["minimum_manifest_extension"],
        "migration_executed": False,
    }
    validation = {
        **common,
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
    outputs = {
        "independent_candidate_coverage.json": coverage,
        "candidate_pair_comparison.json": pair_comparison,
        "source_level_v2_recomputation.json": recomputation,
        "review_contract_v2_spec.json": contract,
        "eligibility_gate_v2_spec.json": gate,
        "migration_compatibility_report.json": migration,
        "validation_summary.json": validation,
    }
    for value in outputs.values():
        value["artifact_fingerprint"] = policy_v2.canonical_fingerprint(value)
    return outputs


def write_run(project_root: Path, output_dir: Path, execution_git_sha: str, run_id: str) -> dict:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    if destination.parent != (root / "backend/data/runs").resolve() or destination.name != run_id:
        raise IndependentCoverageError("Run directory must be a direct backend/data/runs child")
    if destination.exists() or destination.is_symlink():
        raise IndependentCoverageError("Run directory already exists")
    before = policy_v2.sha256_file(root / MANIFEST_RELATIVE)
    outputs = build_independent_outputs(root, execution_git_sha, run_id)
    destination.mkdir(mode=0o755)
    try:
        hashes = {}
        for name, payload in outputs.items():
            partial = destination / f".{name}.partial"
            final = destination / name
            partial.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            os.replace(partial, final)
            hashes[name] = policy_v2.sha256_file(final)
        if policy_v2.sha256_file(root / MANIFEST_RELATIVE) != before:
            raise IndependentCoverageError("Manifest changed during read-only run")
        run_manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": PARENT_GIT_SHA,
            "manifest_sha256": before,
            "policy_fingerprint": outputs["independent_candidate_coverage.json"]["policy_fingerprint"],
            "artifact_sha256": dict(sorted(hashes.items())),
            "run_status": "INDEPENDENT_COVERAGE_AND_CONTRACT_COMPLETE_NOT_ACTIVATED",
        }
        run_manifest["run_fingerprint"] = policy_v2.canonical_fingerprint(run_manifest)
        (destination / "run_manifest.json").write_text(json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return run_manifest
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-git-sha", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = write_run(args.project_root, args.output_dir, args.execution_git_sha, args.run_id)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
