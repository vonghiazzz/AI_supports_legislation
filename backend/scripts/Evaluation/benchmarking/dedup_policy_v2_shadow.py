"""Read-only Batch2DC shadow adapter for frozen Dedup Policy V2.

The adapter consumes only frozen local evidence, produces analytical run
artifacts, and has no dependency on source-review or promotion mutation APIs.
It does not activate V2 or alter the V1 eligibility contract.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import dedup_policy_v2 as v2


SHADOW_PROTOCOL_ID = "independent_batch2dc_dedup_policy_v2_shadow_v1"
EXPECTED_PARENT_GIT_SHA = "30e7d307dc692cefa49c9b257ba80e18fc737970"
MANIFEST_RELATIVE = Path("backend/dataLaw/independent_intake/acquisition_manifest.json")
MANIFEST_SHA256 = "411a5ad54b26e640d190bc04b76fbbbfb39896c2d2b6b3fd96450347a8fec08a"
R3_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_dedup_policy_adjudication_v1_20261111_163542_13a283e1"
)
R2_RUN_RELATIVE = Path(
    "backend/data/runs/"
    "independent_batch2dc_review_adjudication_plan_draft_v1_20261111_161740_13a283e1"
)
INPUT_BINDINGS = {
    "pair_evidence": {
        "path": (R3_RUN_RELATIVE / "pair_evidence.json").as_posix(),
        "sha256": "122a00327d02582a2149bf4ba427de4e818de0b93b133ac6561c590f968889e9",
    },
    "policy_adjudication_report": {
        "path": (R3_RUN_RELATIVE / "policy_adjudication_report.json").as_posix(),
        "sha256": "ce9ccc107f3edfa802934c7c4751834971b9f1f592c4489c650173c4569b00c5",
    },
    "counterfactual_analysis": {
        "path": (R3_RUN_RELATIVE / "counterfactual_analysis.json").as_posix(),
        "sha256": "25bcc28e5c9fe6f16f0b1918c7dbc6500a8981ebe69c25ad6ac58b296df3211c",
    },
    "review_adjudication_plan": {
        "path": (R2_RUN_RELATIVE / "review_adjudication_plan.json").as_posix(),
        "sha256": "c8429c0f9feb553a665c14900e1161b184e8b7f07c143082576f9341f113168d",
    },
    "land_2024_potential": {
        "path": (R2_RUN_RELATIVE / "land_2024_potential.json").as_posix(),
        "sha256": "982a306cb65695356dcf6f86ab49098e4b21f1d64b7189e837dee355e67e5026",
    },
}
RECOGNIZED_CANDIDATE_REASONS = {
    "AUDIT_ONLY_SHARED_FILENAME_AND_NORMALIZED_JUDGMENT_IDENTITY",
    "development_benchmark_exact_source",
    "development_benchmark_exact_sha256",
    "development_benchmark_same_case_id",
    "new_pool_duplicate_source_identity",
    "new_pool_duplicate_sha256",
    "new_pool_duplicate_case_id",
    "new_pool_duplicate_structured_sha256",
    "new_pool_duplicate_normalized_identity",
    "new_pool_same_underlying_dispute",
    "new_pool_related_decision_ids",
    "new_pool_known_duplicate_link",
}


class ShadowEvaluationError(ValueError):
    """Frozen shadow inputs or deterministic evaluation are invalid."""


def _strict_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ShadowEvaluationError(f"Invalid strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ShadowEvaluationError(f"JSON root must be an object: {path}")
    return value


def _load_bound_json(project_root: Path, binding: Mapping[str, str]) -> dict:
    path = project_root / binding["path"]
    if path.is_symlink() or not path.is_file():
        raise ShadowEvaluationError(f"Bound input is not a regular file: {path}")
    if v2.sha256_file(path) != binding["sha256"]:
        raise ShadowEvaluationError(f"Bound input SHA256 mismatch: {path}")
    return _strict_json(path)


def _verify_self_fingerprint(payload: Mapping[str, Any], key: str, label: str) -> None:
    if payload.get(key) != v2.canonical_fingerprint(payload, excluded_key=key):
        raise ShadowEvaluationError(f"{label} self-fingerprint mismatch")


def _candidate_reason_recognized(reason: str) -> bool:
    return (
        reason in RECOGNIZED_CANDIDATE_REASONS
        or reason.startswith("near_duplicate_source_text:")
        or reason.startswith("new_pool_near_duplicate_source_text:")
        or reason.startswith("known_underlying_dispute:")
    )


def select_candidate_controls(controls: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Validate evidence-triggered candidates without expanding to all pairs."""

    selected = []
    seen_pairs = set()
    seen_control_ids = set()
    for control in controls:
        reasons = control.get("finding_reasons")
        if not isinstance(reasons, list) or not reasons:
            raise ShadowEvaluationError("Candidate control lacks attributable findings")
        if not all(isinstance(reason, str) and _candidate_reason_recognized(reason) for reason in reasons):
            raise ShadowEvaluationError("Candidate control contains an unknown finding")
        control_id = control.get("control_id")
        if not isinstance(control_id, str) or control_id in seen_control_ids:
            raise ShadowEvaluationError("Candidate control ID is missing or duplicated")
        pair = tuple(sorted((control.get("left_source_id"), control.get("right_source_id"))))
        if len(pair) != 2 or not all(isinstance(item, str) and item for item in pair):
            raise ShadowEvaluationError("Candidate source IDs are invalid")
        if pair in seen_pairs:
            raise ShadowEvaluationError("Duplicate candidate pair")
        seen_control_ids.add(control_id)
        seen_pairs.add(pair)
        selected.append(copy.deepcopy(dict(control)))
    return sorted(selected, key=lambda item: item["control_id"])


def _verify_source_artifacts(project_root: Path, source: Mapping[str, Any]) -> dict:
    raw_path = project_root / source["stored_pdf_path"]
    structured_path = project_root / source["structured_path"]
    for path, expected, label in (
        (raw_path, source["sha256"], "raw PDF"),
        (structured_path, source["structured_sha256"], "structured artifact"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ShadowEvaluationError(f"{label} is unavailable: {path}")
        if v2.sha256_file(path) != expected:
            raise ShadowEvaluationError(f"{label} SHA256 mismatch: {path}")
    return {
        "raw_pdf": {"path": source["stored_pdf_path"], "sha256": source["sha256"]},
        "structured": {
            "path": source["structured_path"],
            "sha256": source["structured_sha256"],
        },
    }


def _validate_fixture_against_r3(fixture: Mapping[str, Any], pair_artifact: Mapping[str, Any]) -> None:
    pairs = pair_artifact.get("pairs")
    controls = fixture.get("pair_controls")
    if pair_artifact.get("pair_count") != 23 or not isinstance(pairs, list) or len(pairs) != 23:
        raise ShadowEvaluationError("R3 pair evidence must contain exactly 23 pairs")
    if not isinstance(controls, list) or len(controls) != 23:
        raise ShadowEvaluationError("V2 fixture must contain exactly 23 controls")
    for index, (control, evidence) in enumerate(zip(controls, pairs), start=1):
        left = evidence["left"]
        right = evidence["right"]
        expected = {
            "control_id": f"R3-PAIR-{index:03d}",
            "left_source_id": left["source_id"],
            "right_source_id": right["source_id"],
            "scope": (
                "DEVELOPMENT"
                if left["source_id"].startswith("judgment:")
                or right["source_id"].startswith("judgment:")
                else "INTERNAL"
            ),
            "raw_sha_equal": evidence["raw_sha_equal"],
            "structured_sha_equal": evidence["structured_sha_equal"],
            "case_id_equal": evidence["case_id_equal"],
            "normalized_judgment_equal": evidence["judgment_number_equal"],
            "filename_equal": left["source_name"].casefold() == right["source_name"].casefold(),
            "court_equal": evidence["court_equal"],
            "decision_date_equal": (
                None
                if left["decision_date"] is None or right["decision_date"] is None
                else left["decision_date"] == right["decision_date"]
            ),
            "decision_year_equal": (
                None
                if left["decision_year"] is None or right["decision_year"] is None
                else left["decision_year"] == right["decision_year"]
            ),
            "document_type_equal": (
                None
                if left["document_type"] is None or right["document_type"] is None
                else left["document_type"] == right["document_type"]
            ),
            "procedure_stage_equal": evidence["procedure_stage_equal"],
            "text_similarity": evidence["text_shingle_similarity"],
            "finding_reasons": evidence["finding_reasons"],
        }
        for field, value in expected.items():
            if control.get(field) != value:
                raise ShadowEvaluationError(
                    f"Fixture/R3 mismatch for {expected['control_id']} field {field}"
                )
        expected_paths = sorted({left["source_path"], right["source_path"]})
        expected_hashes = sorted({left["raw_sha256"], right["raw_sha256"]})
        if sorted(control["evidence_paths"]) != expected_paths:
            raise ShadowEvaluationError("Fixture/R3 evidence paths differ")
        if sorted(control["evidence_sha256"]) != expected_hashes:
            raise ShadowEvaluationError("Fixture/R3 evidence hashes differ")


def load_shadow_inputs(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    policy = v2.load_and_validate_policy(root)
    if policy["policy_fingerprint"] != "bf148f9421d27603f83328e6f98362d7b8d668263cdc4cb51735613c5e82b914":
        raise ShadowEvaluationError("Frozen V2 policy fingerprint changed")
    manifest_path = root / MANIFEST_RELATIVE
    if v2.sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ShadowEvaluationError("Acquisition manifest changed")
    manifest = _strict_json(manifest_path)
    bound = {name: _load_bound_json(root, binding) for name, binding in INPUT_BINDINGS.items()}
    _verify_self_fingerprint(bound["pair_evidence"], "artifact_fingerprint", "R3 pair evidence")
    _verify_self_fingerprint(bound["policy_adjudication_report"], "report_fingerprint", "R3 report")
    _verify_self_fingerprint(bound["counterfactual_analysis"], "artifact_fingerprint", "R3 counterfactual")
    fixture = v2.validate_evidence_fixture(
        v2.load_evidence_fixture(root / v2.EVIDENCE_FIXTURE_RELATIVE)
    )
    _validate_fixture_against_r3(fixture, bound["pair_evidence"])
    for evidence in bound["pair_evidence"]["pairs"]:
        for side in (evidence["left"], evidence["right"]):
            path = root / side["source_path"]
            if path.is_symlink() or not path.is_file():
                raise ShadowEvaluationError(f"Pair evidence source is unavailable: {path}")
            if v2.sha256_file(path) != side["raw_sha256"]:
                raise ShadowEvaluationError(f"Pair evidence source SHA256 mismatch: {path}")

    batch = sorted(
        (
            source
            for source in manifest["sources"]
            if source["acquisition_id"].startswith("batch2dc-20261105-")
        ),
        key=lambda item: item["acquisition_id"],
    )
    if len(batch) != 46 or any(item["acquisition_status"] != "extracted" for item in batch):
        raise ShadowEvaluationError("Batch2DC must remain exactly 46 extracted sources")
    evidence_by_id = {}
    for source in batch:
        verification = source["metadata_verification"]
        if not all(
            verification[field] is True
            for field in ("judgment_number", "case_id", "court", "decision_date", "procedure_stage")
        ):
            raise ShadowEvaluationError(f"Identity is not ready: {source['acquisition_id']}")
        if any(
            verification[field] is True
            for field in (
                "development_overlap_check",
                "internal_dedup_check",
                "underlying_dispute_review",
            )
        ):
            raise ShadowEvaluationError("R5 requires unreviewed Batch2DC sources")
        evidence_by_id[source["acquisition_id"]] = _verify_source_artifacts(root, source)

    plan_sources = bound["review_adjudication_plan"].get("sources")
    if not isinstance(plan_sources, list) or len(plan_sources) != 46:
        raise ShadowEvaluationError("Frozen R2 plan does not contain 46 sources")
    plan_ids = {item["acquisition_id"] for item in plan_sources}
    batch_ids = {item["acquisition_id"] for item in batch}
    if plan_ids != batch_ids:
        raise ShadowEvaluationError("R2 plan and live Batch2DC identities differ")
    for plan_source in plan_sources:
        source = next(item for item in batch if item["acquisition_id"] == plan_source["acquisition_id"])
        bindings = plan_source["evidence_bindings"]
        if (
            bindings["pdf"]["path"] != source["stored_pdf_path"]
            or bindings["pdf"]["sha256"] != source["sha256"]
            or bindings["structured"]["path"] != source["structured_path"]
            or bindings["structured"]["sha256"] != source["structured_sha256"]
        ):
            raise ShadowEvaluationError("R2 source evidence no longer matches manifest")

    report_classes = {
        item["acquisition_id"]: item["analytical_classification"]
        for item in bound["policy_adjudication_report"]["source_classifications"]
    }
    fixture_classes = {
        item["acquisition_id"]: item["analytical_classification"]
        for item in fixture["source_adjudications"]
    }
    if report_classes != fixture_classes:
        raise ShadowEvaluationError("Fixture/R3 source adjudications differ")

    return {
        "root": root,
        "policy": policy,
        "manifest": manifest,
        "batch_sources": batch,
        "source_artifact_evidence": evidence_by_id,
        "fixture": fixture,
        **bound,
    }


def _compatibility_matrix() -> list[dict]:
    rows = [
        ("evidence_collection", "V1 live inventory + private text", "Frozen verified R2/R3 evidence", "VERSIONED_PRODUCER_REQUIRED"),
        ("candidate_pair_generation", "All V1 signal edges in one pass", "Evidence-triggered ledger, separate from adjudication", "NEW_VERSION_REQUIRED"),
        ("same_document_detection", "Union of heterogeneous signals", "Strong/weak relations separated", "INCOMPATIBLE_WITH_V1_PAYLOAD"),
        ("development_overlap", "Any V1 development finding blocks", "Confirmed blocks; verified weak collision may clear", "NEW_VERSION_REQUIRED"),
        ("underlying_dispute_review", "Shares union graph with duplicate edges", "Separate fail-closed dispute graph", "NEW_VERSION_REQUIRED"),
        ("findings_serialization", "Three unversioned review fields", "Policy-bound typed relations + provenance", "VERSIONED_PAYLOAD_REQUIRED"),
        ("rejection_reasons", "Every finding becomes rejection", "Only V2 blocking dispositions reject", "VERSIONED_GATE_REQUIRED"),
        ("canonical_duplicate_retention", "Cluster canonical may inherit all findings", "Exact raw-SHA canonical only; independent blockers preserved", "NEW_VERSION_REQUIRED"),
        ("eligibility_gates", "Recomputes V1 findings at review and promotion", "Must recompute selected policy/version", "EXPLICIT_POLICY_SELECTION_REQUIRED"),
    ]
    return [
        {"contract_area": area, "v1": old, "v2_shadow": new, "migration": migration}
        for area, old, new, migration in rows
    ]


def build_shadow_evaluation(project_root: Path, execution_git_sha: str, run_id: str) -> dict[str, dict]:
    if not isinstance(execution_git_sha, str) or len(execution_git_sha) != 40:
        raise ShadowEvaluationError("execution_git_sha must be a full Git SHA")
    if not isinstance(run_id, str) or not run_id.startswith(SHADOW_PROTOCOL_ID + "_"):
        raise ShadowEvaluationError("run_id is invalid")
    inputs = load_shadow_inputs(project_root)
    policy = inputs["policy"]
    fixture = inputs["fixture"]
    controls = select_candidate_controls(fixture["pair_controls"])
    pair_results = []
    for control in controls:
        decision = v2.evaluate_pair(policy, v2.pair_input_from_fixture(control))
        pair_results.append(
            {
                "control_id": control["control_id"],
                "candidate_generation_basis": list(control["finding_reasons"]),
                "left_source_id": control["left_source_id"],
                "right_source_id": control["right_source_id"],
                "scope": control["scope"],
                "evidence_paths": list(control["evidence_paths"]),
                "evidence_sha256": list(control["evidence_sha256"]),
                "decision": decision,
            }
        )
    if len(pair_results) != 23:
        raise ShadowEvaluationError("Frozen candidate ledger must contain 23 pairs")

    classifications = {
        item["acquisition_id"]: item
        for item in inputs["policy_adjudication_report"]["source_classifications"]
    }
    clean = set(fixture["clean_batch2dc_acquisition_ids"])
    if len(classifications) != 23 or len(clean) != 23 or set(classifications) & clean:
        raise ShadowEvaluationError("Frozen clean/flagged controls are invalid")
    all_ids = {source["acquisition_id"] for source in inputs["batch_sources"]}
    if set(classifications) | clean != all_ids:
        raise ShadowEvaluationError("Frozen controls do not cover all 46 sources")

    membership = v2.counterfactual_membership(fixture)
    s2_included = set(membership["S2"]["included"])
    s2_unresolved = set(membership["S2"]["unresolved"])
    s2_rejected = set(membership["S2"]["rejected"])
    plan_by_id = {
        item["acquisition_id"]: item
        for item in inputs["review_adjudication_plan"]["sources"]
    }

    exact_pair = next(
        item
        for item in pair_results
        if item["decision"]["same_document_relation"] == "CONFIRMED_EXACT_DUPLICATE"
    )
    duplicate_ids = {
        source_id.split("independent:", 1)[1]
        for source_id in (exact_pair["left_source_id"], exact_pair["right_source_id"])
    }
    duplicate_records = []
    for acquisition_id in sorted(duplicate_ids):
        source = next(item for item in inputs["batch_sources"] if item["acquisition_id"] == acquisition_id)
        other_blockers = sorted(
            {
                item["decision"]["dispute_relation"]
                for item in pair_results
                if acquisition_id in {
                    item["left_source_id"].replace("independent:", ""),
                    item["right_source_id"].replace("independent:", ""),
                }
                and item["control_id"] != exact_pair["control_id"]
                and item["decision"]["overall_blocking"]
            }
        )
        duplicate_records.append(
            {
                "source_id": source["source_id"],
                "case_id": source["case_id"],
                "raw_sha256": source["sha256"],
                "other_blockers": other_blockers,
                "findings_provenance": list(exact_pair["candidate_generation_basis"]),
            }
        )
    canonical = v2.canonicalize_exact_duplicate_cluster(duplicate_records)

    decisions = []
    for source in inputs["batch_sources"]:
        acquisition_id = source["acquisition_id"]
        source_id = source["source_id"]
        related_pairs = [
            item
            for item in pair_results
            if source_id in (item["left_source_id"], item["right_source_id"])
        ]
        analytical = classifications.get(acquisition_id)
        plan = plan_by_id[acquisition_id]
        weak = [item["control_id"] for item in related_pairs if item["decision"]["weak_signal"]]
        disputes = [
            {"control_id": item["control_id"], "relation": item["decision"]["dispute_relation"]}
            for item in related_pairs
            if item["decision"]["dispute_relation"] != "NO_DISPUTE_RELATION"
            and item["decision"]["dispute_relation"] != "DUPLICATE_INTERNAL_RELATION_NOT_INDEPENDENT_BLOCKER"
        ]
        development = [
            {"control_id": item["control_id"], "relation": item["decision"]["development_relation"]}
            for item in related_pairs
            if item["scope"] == "DEVELOPMENT"
        ]
        independent_blockers = sorted(
            {
                relation["relation"]
                for relation in disputes
                if relation["relation"] in {
                    "CONFIRMED_SAME_UNDERLYING_DISPUTE",
                    "DISPUTE_RELATION_UNRESOLVED",
                }
            }
        )
        if acquisition_id in s2_rejected:
            disposition = "NONCANONICAL_EXACT_DUPLICATE_REJECTED"
            readiness = "DETERMINISTIC_REJECTION"
            projection = "REJECTED_NONCANONICAL"
        elif acquisition_id in s2_unresolved:
            disposition = "NO_CANONICAL_DISPOSITION"
            readiness = "BLOCKED_UNRESOLVED_DISPUTE"
            projection = "UNRESOLVED_FAIL_CLOSED"
        else:
            disposition = (
                "CANONICAL_EXACT_DUPLICATE_RETAINED"
                if source_id == canonical["canonical_source_id"]
                else "NOT_IN_EXACT_DUPLICATE_CLUSTER"
            )
            readiness = "READY_FOR_FUTURE_VERSIONED_REVIEW_CONTRACT"
            projection = "POTENTIALLY_ELIGIBLE_UNDER_S2_SHADOW"
        decisions.append(
            {
                "acquisition_id": acquisition_id,
                "source_id": source_id,
                "exact_source_identity": {
                    field: source[field]
                    for field in (
                        "judgment_number",
                        "case_id",
                        "court",
                        "decision_date",
                        "procedure_stage",
                    )
                },
                "evidence_hashes": inputs["source_artifact_evidence"][acquisition_id],
                "original_v1_findings": copy.deepcopy(plan["findings"]),
                "finding_origin": copy.deepcopy(plan["finding_origin"]),
                "analytical_classification": (
                    analytical["analytical_classification"] if analytical else "PREVIOUSLY_CLEAN"
                ),
                "v2_pair_decisions": [item["control_id"] for item in related_pairs],
                "strong_same_document_cluster": (
                    sorted(duplicate_ids) if acquisition_id in duplicate_ids else []
                ),
                "weak_metadata_associations": weak,
                "underlying_dispute_relations": disputes,
                "development_overlap_decisions": development,
                "canonical_disposition": disposition,
                "independent_blockers": independent_blockers,
                "shadow_review_readiness": readiness,
                "shadow_eligibility_projection": projection,
                "unresolved_evidence": independent_blockers,
                "decision_provenance": {
                    "policy_fingerprint": policy["policy_fingerprint"],
                    "pair_control_ids": [item["control_id"] for item in related_pairs],
                    "r2_plan_fingerprint": inputs["review_adjudication_plan"]["plan_fingerprint"],
                    "r3_report_fingerprint": inputs["policy_adjudication_report"]["report_fingerprint"],
                },
            }
        )
    decisions.sort(key=lambda item: item["acquisition_id"])
    if len(decisions) != 46 or len(s2_included) != 40:
        raise ShadowEvaluationError("Source-level shadow aggregation is incomplete")

    scenario_rows = []
    r3_scenarios = {
        item["scenario"].split("_", 1)[0]: item
        for item in inputs["counterfactual_analysis"]["scenarios"]
    }
    for name in ("S0", "S1", "S2"):
        included_count = len(membership[name]["included"])
        reference = r3_scenarios[name]
        scenario_rows.append(
            {
                "scenario": name,
                "batch2dc_potential_eligible_count": included_count,
                "r3_reference_count": reference["batch2dc_potential_eligible_count"],
                "difference": included_count - reference["batch2dc_potential_eligible_count"],
                "land_2024_compatible_count": reference["land_2024_compatible_count"],
                "maximum_filled_minimum_slots": reference["maximum_filled_minimum_slots"],
                "family_shortage": reference["family_shortage"],
            }
        )

    common = {
        "protocol_id": SHADOW_PROTOCOL_ID,
        "run_id": run_id,
        "execution_git_sha": execution_git_sha,
        "parent_git_sha": EXPECTED_PARENT_GIT_SHA,
        "manifest_sha256": MANIFEST_SHA256,
        "policy_fingerprint": policy["policy_fingerprint"],
    }
    protocol = {
        **common,
        "mode": "READ_ONLY_SHADOW_NOT_ACTIVATED",
        "input_bindings": copy.deepcopy(INPUT_BINDINGS),
        "fixture_sha256": v2.EVIDENCE_FIXTURE_SHA256,
        "fixture_fingerprint": v2.EVIDENCE_FIXTURE_FINGERPRINT,
        "candidate_generation": "FROZEN_EVIDENCE_TRIGGERED_PAIRS_ONLY",
        "pair_adjudication": "FROZEN_DEDUP_POLICY_V2",
        "source_review_calls": 0,
        "promotion_calls": 0,
    }
    ledger = {
        **common,
        "candidate_pair_count": len(pair_results),
        "candidate_pairs": pair_results,
        "duplicate_pair_count": 0,
        "all_batch2dc_flagged_sources_covered": True,
    }
    comparison = {
        **common,
        "compatibility_matrix": _compatibility_matrix(),
        "v1_modified": False,
        "v2_activated": False,
        "source_classification_counts": {
            "PREVIOUSLY_CLEAN": 23,
            "DISTINCT_DOCUMENT_WITH_SHARED_METADATA": 16,
            "CONFIRMED_SAME_DOCUMENT": 2,
            "GRAPH_RELATED_BUT_DOCUMENT_IDENTITY_UNPROVEN": 5,
        },
        "counterfactuals": scenario_rows,
    }
    source_output = {
        **common,
        "source_count": 46,
        "canonical_retention": canonical,
        "sources": decisions,
    }
    migration = {
        **common,
        "readiness": "NEW_VERSIONED_REVIEW_CONTRACT_REQUIRED",
        "required_changes": [
            "versioned review payload",
            "versioned findings producer",
            "versioned eligibility gate",
            "policy identity and fingerprint bindings",
            "explicit review-policy selection",
            "backward-compatible V1-reviewed-source handling",
            "source-level evidence and disposition persistence",
        ],
        "safe_to_replace_v1_in_place": False,
        "v2_activation_authorized": False,
        "review_application_authorized": False,
        "promotion_authorized": False,
        "counterfactuals": scenario_rows,
        "land_2024_role": "DESCRIPTIVE_FROZEN_PROJECTION_ONLY",
    }
    validation = {
        **common,
        "result": "PASS",
        "batch2dc_source_count": 46,
        "candidate_pair_count": 23,
        "source_review_calls": 0,
        "promotion_calls": 0,
        "manifest_mutated": False,
        "v1_modified": False,
        "v2_activated": False,
        "benchmark_or_sampling_mutated": False,
        "network_used": False,
        "retrieval_or_model_used": False,
    }
    outputs = {
        "shadow_protocol.json": protocol,
        "candidate_pair_ledger.json": ledger,
        "v1_v2_comparison.json": comparison,
        "source_level_shadow_decisions.json": source_output,
        "migration_readiness_report.json": migration,
        "validation_summary.json": validation,
    }
    for payload in outputs.values():
        payload["artifact_fingerprint"] = v2.canonical_fingerprint(payload)
    return outputs


def write_shadow_run(project_root: Path, output_dir: Path, execution_git_sha: str, run_id: str) -> dict:
    root = Path(project_root).resolve()
    destination = Path(output_dir).resolve()
    runs_root = (root / "backend/data/runs").resolve()
    if destination.parent != runs_root or destination.name != run_id:
        raise ShadowEvaluationError("Shadow output must be one direct child of backend/data/runs")
    if destination.exists() or destination.is_symlink():
        raise ShadowEvaluationError("Shadow run directory already exists")
    before_manifest = v2.sha256_file(root / MANIFEST_RELATIVE)
    outputs = build_shadow_evaluation(root, execution_git_sha, run_id)
    destination.mkdir(mode=0o755)
    try:
        artifact_hashes = {}
        for name, payload in outputs.items():
            temporary = destination / f".{name}.partial"
            final = destination / name
            raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
            temporary.write_bytes(raw)
            os.replace(temporary, final)
            artifact_hashes[name] = v2.sha256_file(final)
        after_manifest = v2.sha256_file(root / MANIFEST_RELATIVE)
        if before_manifest != MANIFEST_SHA256 or after_manifest != before_manifest:
            raise ShadowEvaluationError("Protected acquisition manifest changed during shadow run")
        run_manifest = {
            "protocol_id": SHADOW_PROTOCOL_ID,
            "run_id": run_id,
            "execution_git_sha": execution_git_sha,
            "parent_git_sha": EXPECTED_PARENT_GIT_SHA,
            "policy_fingerprint": outputs["shadow_protocol.json"]["policy_fingerprint"],
            "input_manifest_sha256": before_manifest,
            "output_artifact_sha256": dict(sorted(artifact_hashes.items())),
            "run_status": "SHADOW_COMPLETE_NOT_ACTIVATED",
        }
        run_manifest["run_fingerprint"] = v2.canonical_fingerprint(run_manifest)
        path = destination / "run_manifest.json"
        path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
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
    result = write_shadow_run(
        args.project_root, args.output_dir, args.execution_git_sha, args.run_id
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
