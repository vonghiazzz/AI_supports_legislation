"""Audit the frozen Phase-2A independent-benchmark sampling policy.

This module consumes only frozen documentary/provenance planning metadata.  It
does not select cases, derive the production sampling seed, acquire sources,
generate queries or ground truth, or execute retrieval/model code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from .family_quota import compatible_families, solve_family_quota_assignment
from .independent_benchmark import (
    BENCHMARK_ID,
    FORBIDDEN_SAMPLING_KEY_TOKENS,
    canonical_json_fingerprint,
    validate_sampling_records,
)
from .preliminary_stratification import (
    PRELIMINARY_FAMILIES,
    TARGET_FAMILIES,
    validate_preliminary_stratification,
)


POLICY_ID = "independent_sampling_policy_v1"
POLICY_STATUS = "POLICY_FROZEN_FOR_ACQUISITION_PLANNING"
PROTOCOL_ID = "independent_sampling_policy_audit_v1"
AUTHORITATIVE_BASE_COMMIT = "299b45317648219d4f780531ac3d2f3c6196094e"
DEFAULT_POLICY_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/independent_sampling_policy_v1.yaml"
)
DEFAULT_RUNS_RELATIVE = Path("backend/data/runs")
STALE_CLASSIFICATION = "LEGACY_DRAFT_QUOTA_FEASIBILITY_STALE"


class SamplingPolicyError(ValueError):
    """Raised when policy or audit inputs violate Phase-2A invariants."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise SamplingPolicyError(f"Duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_nonfinite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SamplingPolicyError(f"Non-finite value at {path}")
    if isinstance(value, dict):
        for key, nested in value.items():
            _reject_nonfinite(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_nonfinite(nested, f"{path}[{index}]")


def strict_json_load(path: Path) -> Any:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise SamplingPolicyError(f"Missing regular JSON artifact: {path}")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            SamplingPolicyError(f"Non-finite JSON constant: {token}")
        ),
    )
    _reject_nonfinite(payload)
    return payload


def strict_json_write(path: Path, payload: Any) -> None:
    _reject_nonfinite(payload)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def strict_yaml_load(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise SamplingPolicyError(f"Missing regular policy config: {path}")
    payload = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise SamplingPolicyError("Policy config must be a mapping")
    _reject_nonfinite(payload)
    return payload


def _exact_mapping(value: Any, expected: Mapping[str, Any], label: str) -> None:
    if value != dict(expected):
        raise SamplingPolicyError(f"{label} differs from frozen policy")


def validate_policy(policy: dict) -> None:
    if policy.get("policy_id") != POLICY_ID:
        raise SamplingPolicyError("Wrong policy_id")
    if policy.get("policy_status") != POLICY_STATUS:
        raise SamplingPolicyError("Wrong policy status")
    if policy.get("policy_status") == "SAMPLING_FROZEN":
        raise SamplingPolicyError("Phase 2A cannot freeze sampling")
    if policy.get("benchmark_id") != BENCHMARK_ID:
        raise SamplingPolicyError("Wrong benchmark_id")
    if policy.get("authoritative_base_commit") != AUTHORITATIVE_BASE_COMMIT:
        raise SamplingPolicyError("Wrong authoritative base commit")

    _exact_mapping(
        policy.get("required_state"),
        {
            "benchmark": "DRAFT",
            "sampling": "DRAFT",
            "selection": "NOT_STARTED",
            "sampling_seed": None,
        },
        "required_state",
    )
    target = policy.get("target")
    if not isinstance(target, dict) or {
        "selection_mode": target.get("selection_mode"),
        "total": target.get("total"),
        "main": target.get("main"),
        "stress": target.get("stress"),
        "slices_disjoint": target.get("slices_disjoint"),
        "fallback_authorized": target.get("fallback_authorized"),
        "legacy_minimum_mode_authorized": target.get(
            "legacy_minimum_mode_authorized"
        ),
        "silent_quota_reduction_authorized": target.get(
            "silent_quota_reduction_authorized"
        ),
        "proportional_quota_scaling_authorized": target.get(
            "proportional_quota_scaling_authorized"
        ),
    } != {
        "selection_mode": "TARGET_ONLY",
        "total": 250,
        "main": 200,
        "stress": 50,
        "slices_disjoint": True,
        "fallback_authorized": False,
        "legacy_minimum_mode_authorized": False,
        "silent_quota_reduction_authorized": False,
        "proportional_quota_scaling_authorized": False,
    }:
        raise SamplingPolicyError("Target/fallback policy differs from frozen values")
    if target["main"] + target["stress"] != target["total"]:
        raise SamplingPolicyError("MAIN and STRESS targets must sum to total")

    quota = policy.get("family_quota")
    if not isinstance(quota, dict):
        raise SamplingPolicyError("family_quota must be a mapping")
    if tuple(quota.get("target_families", ())) != TARGET_FAMILIES:
        raise SamplingPolicyError("Target family order differs from solver")
    _exact_mapping(
        quota.get("hard_minima"),
        dict(zip(TARGET_FAMILIES, (55, 30, 35, 35, 30))),
        "hard family minima",
    )
    _exact_mapping(
        quota.get("soft_maxima"),
        dict(zip(TARGET_FAMILIES, (65, 40, 45, 45, 40))),
        "soft family maxima",
    )
    if sum(quota["hard_minima"].values()) != 185:
        raise SamplingPolicyError("Hard family minima must sum to 185")
    expected_quota_controls = {
        "scope": "MAIN_ONLY",
        "soft_maxima_enforcement": "REPORT_AND_PENALIZE_NOT_INVALIDATE",
        "hard_feasibility_upper_bounds": None,
        "max_quota_slots_per_case": 1,
        "stress_quota_contribution_to_main": 0,
        "unknown_other_quota_slots": 0,
        "assignment_role": "SAMPLING_METADATA_NOT_GROUND_TRUTH",
    }
    if any(quota.get(key) != value for key, value in expected_quota_controls.items()):
        raise SamplingPolicyError("Family quota accounting differs from policy")

    guardrails = policy.get("main_guardrails", {})
    procedural = guardrails.get("blttds_only_procedural", {})
    substantive = guardrails.get("anticipated_substantive_core", {})
    if (
        procedural.get("maximum_ratio") != 0.20
        or procedural.get("maximum_count_at_target") != 40
        or substantive.get("minimum_ratio") != 0.65
        or substantive.get("minimum_count_at_target") != 130
        or set(substantive.get("compatible_family_intersection", ()))
        != {"BLDS_CORE", "LAND_2013", "LAND_2024", "HNGD_2014"}
        or guardrails.get("role")
        != "SAMPLING_TIME_ANTICIPATORY_METADATA_NOT_GROUND_TRUTH"
    ):
        raise SamplingPolicyError("MAIN guardrails differ from frozen policy")

    stress = policy.get("stress", {})
    if stress != {
        "target": 50,
        "eligibility": "AT_LEAST_ONE_STRESS_DESCRIPTOR",
        "family_quota": None,
        "main_quota_contribution": 0,
        "hard_descriptor_quotas": False,
        "descriptor_distribution": "REPORT_ONLY_SOFT_DIVERSITY_OBJECTIVE",
    }:
        raise SamplingPolicyError("STRESS policy differs from frozen policy")
    joint = policy.get("joint_selection", {})
    if (
        joint.get("required") is not True
        or joint.get("case_capacity_across_slices") != 1
        or joint.get("independent_slice_sampling_forbidden") is not True
        or len(joint.get("objective_order", ())) != 9
    ):
        raise SamplingPolicyError("Joint-selection policy is incomplete")
    seed = policy.get("seed_freeze", {})
    if (
        seed.get("sampling_seed") is not None
        or seed.get("generate_in_phase_2a") is not False
        or seed.get("freeze_before_selection") is not True
        or seed.get("commit_before_selected_ids_observed") is not True
    ):
        raise SamplingPolicyError("Seed policy would permit Phase-2A selection")
    frame = policy.get("authoritative_frame", {})
    if (
        frame.get("source_count") != 223
        or frame.get("source_inventory_fingerprint")
        != "8957b1a617da34cd63b4c88095504d339523574c999a085a377a6847a118bc55"
        or frame.get("records_fingerprint")
        != "1af8b33ad2be7dcd7745716102ad846a5f3cbb02071bdbc74f229b876824b54b"
    ):
        raise SamplingPolicyError("Authoritative frame identity differs")
    if policy.get("legacy_manifest", {}).get("stale_classification") != STALE_CLASSIFICATION:
        raise SamplingPolicyError("Legacy stale classification differs")
    validate_leakage_firewall(policy)


def validate_leakage_firewall(policy: dict) -> None:
    firewall = policy.get("leakage_firewall", {})
    required_forbidden = {
        "bm25_output",
        "dense_output",
        "rrf_output",
        "reranker_output",
        "retrieval_score_or_rank",
        "evaluation_metric",
        "generated_query",
        "ground_truth",
        "annotation_outcome",
        "llm_or_gemini_output",
    }
    if set(firewall.get("forbidden_inputs", ())) != required_forbidden:
        raise SamplingPolicyError("Leakage firewall forbidden inputs changed")
    if not firewall.get("allowed_metadata"):
        raise SamplingPolicyError("Leakage firewall requires an allowlist")


def derive_sampling_seed(policy_id: str, final_frame_fingerprint: str) -> int:
    """Implement the frozen future rule without deriving the production seed."""

    if policy_id != POLICY_ID:
        raise SamplingPolicyError("Seed derivation requires the frozen policy_id")
    if not isinstance(final_frame_fingerprint, str) or len(final_frame_fingerprint) != 64:
        raise SamplingPolicyError("Seed derivation requires a SHA-256 frame fingerprint")
    try:
        int(final_frame_fingerprint, 16)
    except ValueError as exc:
        raise SamplingPolicyError("Frame fingerprint is not hexadecimal") from exc
    digest = hashlib.sha256(
        f"{policy_id}\n{final_frame_fingerprint}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) & ((1 << 63) - 1)


def quota_memberships_for_slice(record: dict, slice_name: str) -> tuple[str, ...]:
    if slice_name == "MAIN":
        return compatible_families(record)
    if slice_name == "STRESS":
        return ()
    raise SamplingPolicyError(f"Unknown slice: {slice_name}")


def report_soft_max_overflow(
    family_counts: Mapping[str, int], soft_maxima: Mapping[str, int]
) -> dict[str, int]:
    """Report, but never invalidate, counts above planning soft maxima."""

    if set(family_counts) != set(TARGET_FAMILIES) or set(soft_maxima) != set(
        TARGET_FAMILIES
    ):
        raise SamplingPolicyError("Soft-max reporting requires all target families")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (*family_counts.values(), *soft_maxima.values())
    ):
        raise SamplingPolicyError("Soft-max reporting requires non-negative integers")
    return {
        family: max(0, family_counts[family] - soft_maxima[family])
        for family in TARGET_FAMILIES
    }


def derive_guardrail_proxies(record: dict) -> dict[str, bool]:
    families = set(compatible_families(record))
    return {
        "blttds_only_procedural": families == {"BLTTDS_PRIMARY"},
        "anticipated_substantive_core": bool(
            families & {"BLDS_CORE", "LAND_2013", "LAND_2024", "HNGD_2014"}
        ),
    }


def stress_descriptors(record: dict) -> tuple[str, ...]:
    values = record.get("stress_descriptors")
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value for value in values
    ):
        raise SamplingPolicyError("stress_descriptors must be a string list")
    if len(values) != len(set(values)):
        raise SamplingPolicyError("Duplicate stress descriptor")
    return tuple(values)


def stress_eligible(record: dict) -> bool:
    return bool(stress_descriptors(record))


def _inspect_forbidden_record_keys(value: Any, path: str = "record") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).casefold()
            if any(token in normalized for token in FORBIDDEN_SAMPLING_KEY_TOKENS):
                raise SamplingPolicyError(
                    f"Forbidden retrieval/model field in preliminary metadata: {path}.{key}"
                )
            _inspect_forbidden_record_keys(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _inspect_forbidden_record_keys(nested, f"{path}[{index}]")


def detect_stale_legacy_quota(sampling: dict, policy: dict) -> dict:
    if (
        sampling.get("state") != "DRAFT"
        or sampling.get("selection_status") != "NOT_STARTED"
        or sampling.get("sampling_seed") is not None
    ):
        raise SamplingPolicyError("Sampling manifest is not an untouched DRAFT")
    feasibility = sampling.get("quota_feasibility")
    targets = feasibility.get("targets") if isinstance(feasibility, dict) else None
    if not isinstance(targets, dict) or not targets:
        raise SamplingPolicyError("Legacy quota feasibility is missing")
    availability = {
        name: value.get("available_preliminary")
        for name, value in targets.items()
        if isinstance(value, dict)
    }
    if len(availability) != 5 or set(availability.values()) != {0}:
        raise SamplingPolicyError("Legacy quota feasibility is not the known stale draft")
    return {
        "classification": STALE_CLASSIFICATION,
        "zero_availability_values": availability,
        "authoritative_for_phase_2a": False,
        "mutated": False,
        "regenerate_before_sampling_frozen": policy["legacy_manifest"][
            "regenerate_before_sampling_frozen"
        ],
    }


def _load_inputs(project_root: Path, policy: dict) -> tuple[dict, dict, dict]:
    frame = policy["authoritative_frame"]
    artifact = strict_json_load(project_root / frame["artifact_path"])
    sampling = strict_json_load(project_root / frame["sampling_manifest_path"])
    benchmark = strict_json_load(project_root / frame["benchmark_manifest_path"])
    checkpoint = sampling.get("eligible_sources")
    if not isinstance(checkpoint, list):
        raise SamplingPolicyError("Sampling manifest eligible frame is missing")
    validate_sampling_records(checkpoint)
    if canonical_json_fingerprint(checkpoint) != frame["source_inventory_fingerprint"]:
        raise SamplingPolicyError("Sampling checkpoint fingerprint mismatch")
    validate_preliminary_stratification(artifact, checkpoint)
    if (
        artifact.get("source_count") != frame["source_count"]
        or artifact.get("source_inventory_fingerprint")
        != frame["source_inventory_fingerprint"]
        or artifact.get("records_fingerprint") != frame["records_fingerprint"]
    ):
        raise SamplingPolicyError("Preliminary-stratification identity mismatch")
    expected_scope = {
        "eligible_independent_only": True,
        "development_sources_included": False,
        "blocked_sources_included": False,
        "hash_verified_sources_included": False,
    }
    if artifact.get("scope") != expected_scope:
        raise SamplingPolicyError("Preliminary frame scope changed")
    required = policy["required_state"]
    if (
        benchmark.get("state") != required["benchmark"]
        or sampling.get("state") != required["sampling"]
        or sampling.get("selection_status") != required["selection"]
        or sampling.get("sampling_seed") is not required["sampling_seed"]
    ):
        raise SamplingPolicyError("Benchmark/sampling state changed")
    counts = benchmark.get("current_counts", {})
    if {
        "selected_cases": counts.get("selected_cases"),
        "query_records": counts.get("query_records"),
        "ground_truth_records": counts.get("ground_truth_records"),
    } != {"selected_cases": 0, "query_records": 0, "ground_truth_records": 0}:
        raise SamplingPolicyError("Selection/query/GT must remain empty")
    return artifact, sampling, benchmark


def _frame_summary(records: list[dict], artifact: dict) -> dict:
    for record in records:
        _inspect_forbidden_record_keys(record)
    primary = Counter(record["primary_family_preliminary"] for record in records)
    status = Counter(record["classification_status"] for record in records)
    compatible = Counter(
        family for record in records for family in compatible_families(record)
    )
    proxies = [derive_guardrail_proxies(record) for record in records]
    summary = {
        "source_count": len(records),
        "source_inventory_fingerprint": artifact["source_inventory_fingerprint"],
        "records_fingerprint": canonical_json_fingerprint(records),
        "scope": artifact["scope"],
        "primary_family_distribution": {
            family: primary[family] for family in PRELIMINARY_FAMILIES
        },
        "classification_status_distribution": {
            key: status[key] for key in ("CLASSIFIED", "MULTI_FAMILY", "UNKNOWN")
        },
        "classification_coverage": (
            status["CLASSIFIED"] + status["MULTI_FAMILY"]
        )
        / len(records),
        "compatible_membership_counts": {
            family: compatible[family] for family in TARGET_FAMILIES
        },
        "guardrail_proxy_frame_counts": {
            "blttds_only_procedural": sum(
                value["blttds_only_procedural"] for value in proxies
            ),
            "anticipated_substantive_core": sum(
                value["anticipated_substantive_core"] for value in proxies
            ),
        },
        "proxy_role": "SAMPLING_TIME_ANTICIPATORY_METADATA_NOT_GROUND_TRUTH",
    }
    if summary["primary_family_distribution"] != artifact["family_distribution"]:
        raise SamplingPolicyError("Recomputed primary family distribution mismatch")
    if summary["classification_status_distribution"] != artifact[
        "classification_status_distribution"
    ]:
        raise SamplingPolicyError("Recomputed classification distribution mismatch")
    if summary["classification_coverage"] != artifact["classification_coverage"]:
        raise SamplingPolicyError("Recomputed classification coverage mismatch")
    return summary


def _stress_summary(records: list[dict], policy: dict) -> dict:
    eligible = [record for record in records if stress_eligible(record)]
    occurrences = Counter(
        descriptor for record in records for descriptor in stress_descriptors(record)
    )
    return {
        "target": policy["stress"]["target"],
        "eligible_source_count": len(eligible),
        "frame_source_count": len(records),
        "descriptor_occurrence_counts": dict(sorted(occurrences.items())),
        "multi_descriptor_source_count": sum(
            len(stress_descriptors(record)) > 1 for record in records
        ),
        "eligibility_rule": policy["stress"]["eligibility"],
        "hard_descriptor_quotas": False,
        "descriptor_distribution": "REPORT_ONLY_SOFT_DIVERSITY_OBJECTIVE",
        "family_quota_contribution_to_main": 0,
    }


def _family_feasibility(records: list[dict], policy: dict) -> dict:
    lower = policy["family_quota"]["hard_minima"]
    result = solve_family_quota_assignment(
        records,
        lower,
        candidate_scope="phase_2a_current_223_eligible_independent",
        upper_bounds=None,
    )
    synthetic = list(records) + [
        {
            "source_id": f"independent:synthetic-land-2024-{index:02d}",
            "case_id": f"SYNTHETIC-{index:02d}/2026/DS-ST",
            "primary_family_preliminary": "LAND_2024",
            "secondary_families_preliminary": [],
            "classification_status": "CLASSIFIED",
        }
        for index in range(1, 14)
    ]
    proof = solve_family_quota_assignment(
        synthetic,
        lower,
        candidate_scope="phase_2a_feasibility_construction_plus_13_land_2024_only",
        upper_bounds=None,
    )
    strongest = result["hall_bottlenecks"][0] if result["hall_bottlenecks"] else None
    return {
        "candidate_scope": result["candidate_scope"],
        "source_count": result["candidate_count"],
        "hard_lower_bounds": result["lower_bounds"],
        "upper_bounds_used_for_hard_feasibility": result["upper_bounds"],
        "soft_maxima": policy["family_quota"]["soft_maxima"],
        "soft_maxima_role": "REPORT_AND_PENALIZE_NOT_INVALIDATE",
        "required_slots": result["required_slots"],
        "maximum_filled_minimum_slots": result["maximum_filled_minimum_slots"],
        "shortage": result["shortage"],
        "feasible": result["feasible"],
        "compatible_membership_counts": result["compatible_membership_counts"],
        "strongest_hall_bottleneck": strongest,
        "all_hall_bottlenecks": result["hall_bottlenecks"],
        "max_quota_slots_per_case": 1,
        "unknown_other_quota_slots": 0,
        "synthetic_feasibility_proof": {
            "construction": "+13 synthetic LAND_2024-only sources",
            "role": "FEASIBILITY_CONSTRUCTION_NOT_ACQUISITION_PRESCRIPTION",
            "feasible": proof["feasible"],
            "required_slots": proof["required_slots"],
            "maximum_filled_minimum_slots": proof[
                "maximum_filled_minimum_slots"
            ],
            "shortage": proof["shortage"],
        },
    }


def build_policy_audit(project_root: Path, policy_path: Path) -> dict[str, dict]:
    project_root = Path(project_root).resolve()
    policy_path = Path(policy_path).resolve()
    policy = strict_yaml_load(policy_path)
    validate_policy(policy)
    artifact, sampling, benchmark = _load_inputs(project_root, policy)
    records = artifact["records"]
    frame = _frame_summary(records, artifact)
    stress = _stress_summary(records, policy)
    family = _family_feasibility(records, policy)
    stale = detect_stale_legacy_quota(sampling, policy)
    target_deficit = max(0, policy["target"]["total"] - len(records))
    land_deficit = max(
        0,
        policy["family_quota"]["hard_minima"]["LAND_2024"]
        - family["compatible_membership_counts"]["LAND_2024"],
    )
    acquisition = {
        "phase": "PHASE_2B_INPUT_CONSTRAINTS_ONLY",
        "acquisition_performed": False,
        "reserve_target_frozen": False,
        "target_count_deficit": target_deficit,
        "family_max_flow_deficit": family["shortage"],
        "land_2024_individual_compatibility_deficit": land_deficit,
        "combined_hall_constraint": {
            "families": ["BLDS_CORE", "LAND_2013", "LAND_2024"],
            "minimum_additional_unique_compatible_sources": 13,
        },
        "net_eligible_total_minimum_addition": 27,
        "requirements_overlap": True,
        "do_not_add_deficits": True,
        "future_frame_if_target_is_impossible": (
            "STOP_AND_CREATE_SEPARATELY_REVIEWED_POLICY_VERSION"
        ),
    }
    state = {
        "benchmark_state": benchmark["state"],
        "sampling_state": sampling["state"],
        "selection_status": sampling["selection_status"],
        "sampling_seed": sampling["sampling_seed"],
        "selected_cases": benchmark["current_counts"]["selected_cases"],
        "query_records": benchmark["current_counts"]["query_records"],
        "ground_truth_records": benchmark["current_counts"][
            "ground_truth_records"
        ],
        "policy_status": policy["policy_status"],
        "sampling_frozen": False,
        "selection_performed": False,
    }
    return {
        "current_frame_audit.json": {
            **frame,
            "state": state,
            "legacy_sampling_manifest": stale,
            "target": policy["target"],
            "main_guardrails": policy["main_guardrails"],
            "joint_selection": policy["joint_selection"],
            "seed_freeze": policy["seed_freeze"],
            "leakage_firewall": policy["leakage_firewall"],
        },
        "family_feasibility.json": family,
        "stress_eligibility.json": stress,
        "acquisition_constraints.json": acquisition,
    }


def _git_head(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if result != AUTHORITATIVE_BASE_COMMIT:
        raise SamplingPolicyError("Audit must run from the authoritative base commit")
    return result


def publish_audit(
    project_root: Path,
    policy_path: Path,
    output_root: Path,
    structured: Mapping[str, dict],
) -> Path:
    project_root = Path(project_root).resolve()
    policy_path = Path(policy_path).resolve()
    output_root = Path(output_root).resolve()
    try:
        output_root.relative_to(project_root)
    except ValueError as exc:
        raise SamplingPolicyError("Audit output must remain inside project root") from exc
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"
    final = output_root / run_id
    partial = output_root / f"{run_id}.partial"
    if final.exists() or partial.exists():
        raise SamplingPolicyError("Unsafe audit publication path")
    output_root.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        shutil.copyfile(policy_path, partial / "policy_snapshot.yaml")
        for name, payload in structured.items():
            strict_json_write(partial / name, payload)
        output_fingerprints = {
            path.name: sha256_file(path)
            for path in sorted(partial.iterdir(), key=lambda item: item.name)
        }
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_commit": _git_head(project_root),
            "policy_id": POLICY_ID,
            "policy_status": POLICY_STATUS,
            "policy_config_sha256": sha256_file(policy_path),
            "runner_source_sha256": sha256_file(Path(__file__)),
            "audit_fingerprint": canonical_json_fingerprint(dict(structured)),
            "source_artifact_sha256": {
                "preliminary_stratification.json": sha256_file(
                    project_root
                    / "backend/data/benchmarks/retrieval_independent_v1/"
                    "preliminary_stratification.json"
                ),
                "sampling_manifest.json": sha256_file(
                    project_root
                    / "backend/data/benchmarks/retrieval_independent_v1/"
                    "sampling_manifest.json"
                ),
                "benchmark_manifest.json": sha256_file(
                    project_root
                    / "backend/data/benchmarks/retrieval_independent_v1/"
                    "benchmark_manifest.json"
                ),
            },
            "output_file_sha256": output_fingerprints,
            "selection_performed": False,
            "sampling_seed_generated": False,
            "acquisition_performed": False,
            "models_executed": [],
            "retrievers_executed": [],
            "query_or_gt_generated": False,
        }
        strict_json_write(partial / "run_manifest.json", manifest)
        expected = set(structured) | {"policy_snapshot.yaml", "run_manifest.json"}
        if {path.name for path in partial.iterdir()} != expected:
            raise SamplingPolicyError("Incomplete policy audit publication")
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def run(
    *,
    project_root: Path | None = None,
    policy_path: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    policy = (
        Path(policy_path).resolve()
        if policy_path is not None
        else root / DEFAULT_POLICY_RELATIVE
    )
    output = (
        Path(output_root).resolve()
        if output_root is not None
        else root / DEFAULT_RUNS_RELATIVE
    )
    _git_head(root)
    structured = build_policy_audit(root, policy)
    return publish_audit(root, policy, output, structured)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="Audit the frozen independent sampling policy without selection"
    )
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = run(
        project_root=args.project_root,
        policy_path=args.policy,
        output_root=args.output_root,
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
