"""Validate and audit the frozen Phase-2D-B3.1 LAND_2024 top-up plan.

This module is deliberately read-only with respect to intake and benchmark
state.  Its only write is an optional, ignored audit run below data/runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .sampling_policy import (
    canonical_json_fingerprint,
    sha256_file,
    strict_json_load,
    strict_json_write,
    strict_yaml_load,
)


PLAN_ID = "independent_topup_acquisition_plan_v1"
PLAN_STATUS = "TOPUP_PLAN_FROZEN_NOT_EXECUTED"
PROTOCOL_ID = "independent_topup_acquisition_plan_audit_v1"
BENCHMARK_ID = "retrieval_independent_v1"
BATCH_LABEL = "BATCH_2D_C_TOPUP"
BATCH_PREFIX = "batch2dc"
PARENT_COMMIT = "3573aedcfc455ec2e2b4be5b33fdb8396cc15aab"
MANIFEST_SHA256 = "4716e61118ce77dee60597940f68247423be685a27b6dde1c3c479393bbc5094"
B3_RUN_ID = "independent_post_acquisition_close_gate_v1_20261104_212020_3e1d9a91"
B3_RUN_FINGERPRINT = "d024f30111929abb126df27b35fe862e6f0dd849b8d01c47b32c32f6a09da928"
LIVE_FRAME_FINGERPRINT = "1a2e3330df020a60acdd13b27bda47ddc7c0da511e0d68f795a2a1ff283d3a23"
FAMILY_QUOTA_FINGERPRINT = "e5f8cabd558d56d316ec2e724d0ba902e7180ee0447e3dc81b202ee95294f4a5"
CLOSE_GATE_FINGERPRINT = "a183d98b9e0655fd653f2e4d8f458864ff47e1e033d18d4413fd476e75886901"

DEFAULT_PLAN_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/independent_topup_acquisition_plan_v1.yaml"
)
DEFAULT_RUNS_RELATIVE = Path("backend/data/runs")
B3_RUN_RELATIVE = Path("backend/data/runs") / B3_RUN_ID
ACQUISITION_MANIFEST_RELATIVE = Path(
    "backend/dataLaw/independent_intake/acquisition_manifest.json"
)
BENCHMARK_RELATIVE = Path(
    "backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json"
)
SAMPLING_RELATIVE = Path(
    "backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json"
)
HISTORICAL_STRATIFICATION_RELATIVE = Path(
    "backend/data/benchmarks/retrieval_independent_v1/preliminary_stratification.json"
)
PROTECTED_SHA256 = {
    str(BENCHMARK_RELATIVE): "f3d69ee36653da42256c7c94158414a144ab863bd561112fa390c9c0cec99046",
    str(SAMPLING_RELATIVE): "dcbda00191779bcbdd9682ee75a570792bee071e38a7f4604cf97fab14e5c06c",
    str(HISTORICAL_STRATIFICATION_RELATIVE): "d59becbb09c4048df59fa22cd5c4c748513b6c7730aefdad491d403587a75068",
}


class TopupPlanningError(ValueError):
    """Raised when the frozen top-up protocol or its evidence changes."""


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise TopupPlanningError(f"{label} differs from frozen top-up plan")


def _canonical_without(value: Mapping[str, Any], key: str) -> str:
    return canonical_json_fingerprint({k: v for k, v in value.items() if k != key})


def validate_plan_config(plan: dict) -> None:
    expected_keys = {
        "plan_id", "plan_status", "benchmark_id", "batch_label",
        "batch_id_prefix", "parent_provenance", "residual_constraints",
        "residual_interpretation", "topup_objective", "family_specific_yield",
        "batch_size", "batch_identity", "targeting", "duplicate_firewall",
        "complete_cohort_rule", "failure_replacement_rule",
        "post_review_close_gates", "required_state", "leakage_firewall",
        "execution_prohibitions",
    }
    _exact(set(plan), expected_keys, "top-level keys")
    _exact(plan["plan_id"], PLAN_ID, "plan_id")
    _exact(plan["plan_status"], PLAN_STATUS, "plan_status")
    _exact(plan["benchmark_id"], BENCHMARK_ID, "benchmark_id")
    _exact(plan["batch_label"], BATCH_LABEL, "batch_label")
    _exact(plan["batch_id_prefix"], BATCH_PREFIX, "batch_id_prefix")
    _exact(
        plan["parent_provenance"],
        {
            "promotion_checkpoint_commit": PARENT_COMMIT,
            "acquisition_manifest_sha256": MANIFEST_SHA256,
            "b3_audit_id": B3_RUN_ID,
            "b3_run_fingerprint": B3_RUN_FINGERPRINT,
            "live_eligible_frame_fingerprint": LIVE_FRAME_FINGERPRINT,
            "family_quota_fingerprint": FAMILY_QUOTA_FINGERPRINT,
            "close_gate_summary_fingerprint": CLOSE_GATE_FINGERPRINT,
        },
        "parent provenance",
    )
    _exact(
        plan["residual_constraints"],
        {
            "eligible_count": {"observed": 257, "minimum": 250, "deficit": 0},
            "land_2024_compatible": {"observed": 30, "minimum": 35, "deficit": 5},
            "family_minimum_slots": {"required": 185, "maximum_filled": 180, "shortage": 5},
            "hall_bottleneck": {
                "families": ["LAND_2024"],
                "compatible_unique_sources": 30,
                "required_slots": 35,
                "shortage": 5,
            },
            "stress_eligible": {"observed": 113, "minimum": 50, "deficit": 0},
            "development_overlap": 0,
            "duplicate_identity": 0,
        },
        "residual constraints",
    )
    interpretation = plan["residual_interpretation"]
    _exact(
        interpretation,
        {
            "deficits_overlap": True,
            "shared_bottleneck": "LAND_2024",
            "additive_deficit_forbidden": True,
            "scientific_compatible_source_deficit": 5,
            "explicit_rule": "LAND deficit, max-flow shortage, and Hall shortage describe one bottleneck and MUST_NOT_BE_SUMMED.",
        },
        "residual interpretation",
    )
    _exact(
        plan["topup_objective"],
        {
            "primary": {"close_land_2024_compatible_deficit": 5},
            "required_post_review_condition": {"land_2024_compatible_minimum": 35},
            "secondary_effect_expected_but_not_assumed": {
                "family_shortage_zero": True,
                "hall_shortage_zero": True,
            },
        },
        "top-up objective",
    )
    yield_policy = plan["family_specific_yield"]
    _exact(yield_policy["status"], "SINGLE_TARGETED_BATCH_OBSERVED_NOT_CALIBRATED", "family yield status")
    _exact(yield_policy["calibration_sufficient"], False, "family yield calibration")
    _exact(yield_policy["evidence_role"], "DESCRIPTIVE_SINGLE_BATCH_EVIDENCE", "family yield evidence role")
    _exact(
        yield_policy["batch2db_descriptive_outcome"],
        {"frozen_raw_cohort": 50, "registered_extracted": 46, "promoted_eligible": 34, "land_2024_compatible_increase": 5},
        "Batch2DB descriptive outcome",
    )
    _exact(yield_policy["general_eligible_yield_is_land_yield"], False, "general yield use")
    _exact(yield_policy["forbidden_scientific_floor_formula"], "CEIL(5 / 0.74)", "forbidden yield formula")
    _exact(
        plan["batch_size"],
        {
            "raw_target": 50,
            "rationale": "HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN_OPERATIONAL_REFERENCE",
            "historical_completed_scale_sizes": [41, 50, 50, 53, 53],
            "historical_median": 50,
            "role": "OPERATIONAL_COHORT_SIZE_NOT_SCIENTIFIC_CLOSE_GUARANTEE",
        },
        "batch size",
    )
    _exact(
        plan["batch_identity"],
        {"separate_from_batch2db": True, "prefix": BATCH_PREFIX, "silent_extension_of_batch2db_forbidden": True},
        "batch identity",
    )
    _exact(
        plan["targeting"],
        {
            "mode": "TARGETED_DOCUMENTARY_DISCOVERY",
            "priority_order": ["LAND_2024_TARGETED_DISCOVERY"],
            "confirmed_family_claim_before_review": False,
            "raw_family_quota": None,
            "authoritative_domain": "congbobanan.toaan.gov.vn",
            "official_source_listing_urls_only": True,
            "third_party_mirrors_forbidden": True,
            "allowed_pre_acquisition_metadata": [
                "official_source_metadata", "official_case_document_identity", "court",
                "procedure_stage", "decision_date", "documentary_dispute_case_metadata",
                "source_provenance", "quality_metadata",
            ],
            "pdf_body_discovery_forbidden": True,
        },
        "targeting",
    )
    _exact(
        plan["duplicate_firewall"],
        {
            "compare_all_existing_317_source_urls": True,
            "compare_known_source_identities_where_metadata_permits": True,
            "compare_development_metadata_where_available": True,
            "compare_within_new_cohort": True,
        },
        "duplicate firewall",
    )
    for section in ("complete_cohort_rule", "failure_replacement_rule", "execution_prohibitions"):
        if not plan[section] or not all(value is True for value in plan[section].values()):
            raise TopupPlanningError(f"{section} must remain fully enabled")
    _exact(
        plan["post_review_close_gates"],
        {
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
        },
        "post-review close gates",
    )
    _exact(
        plan["required_state"],
        {
            "benchmark": "DRAFT", "sampling": "DRAFT", "selection": "NOT_STARTED",
            "sampling_seed": None, "selected_cases": 0, "query_records": 0,
            "ground_truth_records": 0,
        },
        "required state",
    )
    _exact(
        plan["leakage_firewall"]["forbidden_inputs"],
        [
            "DEV107_RETRIEVAL_RESULTS", "BSARD_RESULTS", "BM25_RANKS", "DENSE_RANKS",
            "RRF_RANKS", "RERANKER_SCORES", "GENERATED_QUERIES", "GROUND_TRUTH",
            "ANNOTATIONS", "MODEL_OUTPUT", "LLM_GEMINI_FAMILY_CLASSIFICATION",
        ],
        "leakage firewall",
    )


def _validate_self_fingerprint(payload: Mapping[str, Any], key: str, expected: str, label: str) -> None:
    _exact(payload.get(key), expected, f"{label} declared fingerprint")
    _exact(_canonical_without(payload, key), expected, f"{label} canonical fingerprint")


def validate_authoritative_inputs(project_root: Path, plan: dict) -> dict[str, Any]:
    root = Path(project_root).resolve()
    acquisition_path = root / ACQUISITION_MANIFEST_RELATIVE
    _exact(sha256_file(acquisition_path), MANIFEST_SHA256, "acquisition manifest SHA256")
    acquisition = strict_json_load(acquisition_path)
    statuses = Counter(source.get("acquisition_status") for source in acquisition["sources"])
    _exact(len(acquisition["sources"]), 317, "intake source count")
    _exact(dict(statuses), {"eligible": 257, "extracted": 59, "hash_verified": 1}, "intake statuses")

    b3_dir = root / B3_RUN_RELATIVE
    run_manifest = strict_json_load(b3_dir / "run_manifest.json")
    close_summary = strict_json_load(b3_dir / "close_gate_summary.json")
    family_report = strict_json_load(b3_dir / "family_quota_report.json")
    live_frame = strict_json_load(b3_dir / "live_eligible_frame.json")
    _validate_self_fingerprint(run_manifest, "run_fingerprint", B3_RUN_FINGERPRINT, "B3 run")
    _validate_self_fingerprint(close_summary, "summary_fingerprint", CLOSE_GATE_FINGERPRINT, "close-gate summary")
    _validate_self_fingerprint(family_report, "report_fingerprint", FAMILY_QUOTA_FINGERPRINT, "family quota report")
    _exact(canonical_json_fingerprint(live_frame), LIVE_FRAME_FINGERPRINT, "live eligible frame fingerprint")
    _exact(run_manifest.get("head"), PARENT_COMMIT, "B3 HEAD")
    _exact(run_manifest.get("classification"), "ACQUISITION_TOPUP_REQUIRED", "B3 classification")
    _exact(run_manifest.get("acquisition_manifest_sha256"), MANIFEST_SHA256, "B3 manifest SHA")
    _exact(run_manifest.get("live_frame_fingerprint"), LIVE_FRAME_FINGERPRINT, "B3 live frame")
    _exact(run_manifest.get("family_quota_report_fingerprint"), FAMILY_QUOTA_FINGERPRINT, "B3 family report")
    _exact(run_manifest.get("close_gate_summary_fingerprint"), CLOSE_GATE_FINGERPRINT, "B3 close summary")
    _exact(close_summary.get("classification"), "ACQUISITION_TOPUP_REQUIRED", "close classification")
    _exact(
        close_summary.get("residual_deficits"),
        {
            "development_overlap_count": 0, "duplicate_identity_count": 0,
            "eligible_deficit": 0, "hall_shortage": 5,
            "land_2024_compatible_deficit": 5, "max_flow_shortage": 5,
            "stress_deficit": 0,
        },
        "B3 residual deficits",
    )
    solver = family_report["solver"]
    _exact(family_report["compatible_membership_counts"]["LAND_2024"], 30, "LAND_2024 compatible count")
    _exact(solver["maximum_filled_minimum_slots"], 180, "maximum filled slots")
    _exact(solver["required_slots"], 185, "required slots")
    _exact(solver["shortage"], 5, "family shortage")
    _exact(
        solver["hall_bottlenecks"],
        [{"families": ["LAND_2024"], "compatible_source_count": 30, "required_slots": 35, "shortage": 5}],
        "Hall bottleneck",
    )

    benchmark = strict_json_load(root / BENCHMARK_RELATIVE)
    sampling = strict_json_load(root / SAMPLING_RELATIVE)
    state = plan["required_state"]
    _exact(benchmark.get("state"), state["benchmark"], "benchmark state")
    _exact(sampling.get("state"), state["sampling"], "sampling state")
    _exact(sampling.get("selection_status"), state["selection"], "selection status")
    _exact(sampling.get("sampling_seed"), state["sampling_seed"], "sampling seed")
    counts = benchmark.get("current_counts", {})
    for key in ("selected_cases", "query_records", "ground_truth_records"):
        _exact(counts.get(key), state[key], key)
    for relative, expected in PROTECTED_SHA256.items():
        _exact(sha256_file(root / relative), expected, f"protected SHA256 {relative}")
    return {
        "intake_counts": {"total": 317, "eligible": 257, "extracted": 59, "hash_verified": 1},
        "protected_state": state,
        "protected_artifact_sha256": dict(PROTECTED_SHA256),
    }


def build_audit(project_root: Path, plan_path: Path) -> dict[str, Any]:
    plan = strict_yaml_load(plan_path)
    validate_plan_config(plan)
    inputs = validate_authoritative_inputs(project_root, plan)
    return {
        "plan_id": PLAN_ID,
        "plan_status": PLAN_STATUS,
        "parent_b3_run_id": B3_RUN_ID,
        "parent_b3_run_fingerprint": B3_RUN_FINGERPRINT,
        "current_acquisition_manifest_sha256": MANIFEST_SHA256,
        "live_eligible_frame_fingerprint": LIVE_FRAME_FINGERPRINT,
        "residual_bottleneck": {
            "family": "LAND_2024", "compatible": 30, "required": 35,
            "scientific_deficit": 5, "family_shortage": 5, "hall_shortage": 5,
            "deficits_overlap": True, "additive_deficit_forbidden": True,
        },
        "raw_target": 50,
        "raw_target_role": "OPERATIONAL_COHORT_SIZE_NOT_SCIENTIFIC_CLOSE_GUARANTEE",
        "targeting_priority": ["LAND_2024_TARGETED_DISCOVERY"],
        "plan_fingerprint": canonical_json_fingerprint(plan),
        "execution_state": {
            "plan_only": True, "network_used": False, "discovery_executed": False,
            "cohort_created": False, "sources_downloaded": 0, "sources_registered": 0,
            "extraction_or_ocr_executed": False, "identity_review_promotion_mutated": False,
            "sampling_or_selection_mutated": False, "seed_generated": False,
            "query_or_gt_generated": False, "retrieval_or_model_executed": False,
        },
        **inputs,
    }


def _git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
        capture_output=True, text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40:
        raise TopupPlanningError("Invalid execution HEAD")
    return head


def publish_audit(project_root: Path, plan_path: Path, output_root: Path, audit: dict[str, Any]) -> Path:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise TopupPlanningError("Audit output must remain inside project root") from exc
    run_id = f"{PROTOCOL_ID}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    partial = output / f"{run_id}.partial"
    final = output / run_id
    output.mkdir(parents=True, exist_ok=True)
    if partial.exists() or final.exists():
        raise TopupPlanningError("Unsafe audit output path")
    partial.mkdir()
    try:
        strict_json_write(partial / "topup_plan_audit.json", audit)
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "execution_head": _git_head(root),
            "config_path": str(Path(plan_path).resolve().relative_to(root)),
            "config_sha256": sha256_file(plan_path),
            "validator_sha256": sha256_file(Path(__file__)),
            "parent_b3_run_id": B3_RUN_ID,
            "parent_b3_run_fingerprint": B3_RUN_FINGERPRINT,
            "current_acquisition_manifest_sha256": MANIFEST_SHA256,
            "live_eligible_frame_fingerprint": LIVE_FRAME_FINGERPRINT,
            "residual_bottleneck": audit["residual_bottleneck"],
            "raw_target": audit["raw_target"],
            "targeting_priority": audit["targeting_priority"],
            "plan_fingerprint": audit["plan_fingerprint"],
            "audit_sha256": sha256_file(partial / "topup_plan_audit.json"),
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        manifest["run_fingerprint"] = canonical_json_fingerprint(manifest)
        strict_json_write(partial / "run_manifest.json", manifest)
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def run(*, project_root: Path | None = None, plan_path: Path | None = None, output_root: Path | None = None) -> Path:
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[4]
    plan = Path(plan_path).resolve() if plan_path else root / DEFAULT_PLAN_RELATIVE
    output = Path(output_root).resolve() if output_root else root / DEFAULT_RUNS_RELATIVE
    audit = build_audit(root, plan)
    return publish_audit(root, plan, output, audit)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Audit the frozen LAND_2024 top-up acquisition plan")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = run(project_root=args.project_root, plan_path=args.plan, output_root=args.output_root)
    print(result)
    return result


if __name__ == "__main__":
    main()
