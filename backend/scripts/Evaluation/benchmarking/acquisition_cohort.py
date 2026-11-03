"""Validate and audit the frozen Phase-2C Batch 2D-B cohort declaration.

This module is deliberately offline.  It validates documentary discovery
metadata and cohort identity only; it never downloads, registers, extracts,
reviews, promotes, classifies, samples, or retrieves any source.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import unicodedata
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from .sampling_policy import (
    canonical_json_fingerprint,
    sha256_file,
    strict_json_load,
    strict_json_write,
    strict_yaml_load,
)
from .source_intake import stable_source_id


COHORT_VERSION = "independent_batch2db_cohort_v1"
COHORT_ID = "batch2db-20261103"
COHORT_STATUS = "COHORT_DECLARED_NOT_ACQUIRED"
TARGET_RAW_COUNT = 50
OFFICIAL_DOMAIN = "congbobanan.toaan.gov.vn"
PARENT_PLAN_COMMIT = "a00a08b727db2ee55f347f9e09b8893ccb972e00"
PARENT_PLAN_SHA256 = (
    "9379d96f99e3291e38d2bcda0fbe1a9566e2b781d3093b14e5fa91ad9da0658e"
)
PLAN_AUTHORING_BASE_COMMIT = "a5ebb4a03f35a4c33596faf93b8d5fa82d6d076f"
PARENT_PLAN_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/independent_acquisition_plan_v1.yaml"
)
DEFAULT_COHORT_RELATIVE = Path(
    "backend/dataLaw/independent_intake/cohorts/batch2db-20261103.json"
)
DEFAULT_INTAKE_RELATIVE = Path(
    "backend/dataLaw/independent_intake/acquisition_manifest.json"
)
DEFAULT_RUNS_RELATIVE = Path("backend/data/runs")
PROTOCOL_ID = "independent_batch2db_cohort_audit_v1"
DISCOVERY_PRIORITIES = [
    "LAND_2024_TARGETED_DISCOVERY",
    "LAND_HALL_TARGETED_DISCOVERY",
]

ROOT_KEYS = {
    "cohort_version",
    "cohort_id",
    "status",
    "parent_acquisition_plan_commit",
    "parent_acquisition_plan_path",
    "parent_acquisition_plan_sha256",
    "plan_authoring_base_commit",
    "declaration_execution_commit",
    "declared_at",
    "target_raw_count",
    "source_domain",
    "discovery_priority_order",
    "outcome_adaptive_expansion_forbidden",
    "discovery_summary",
    "entries",
    "cohort_fingerprint",
}
ENTRY_KEYS = {
    "acquisition_id",
    "source_domain",
    "source_url",
    "official_listing_url",
    "discovery_priority",
    "documentary_dispute_type",
    "court",
    "decision_date",
    "procedure_stage",
    "official_document_title_or_filename",
    "official_case_or_judgment_id",
    "official_document_kind",
    "discovery_basis",
    "discovery_metadata_source",
    "pre_download_duplicate_check",
}
PRECHECK_KEYS = {
    "existing_intake_url_overlap",
    "new_cohort_url_overlap",
    "exact_known_intake_case_overlap",
    "exact_development_case_overlap",
    "case_identity_comparison_limited_by_metadata",
    "result",
}
FORBIDDEN_FIELD_NAMES = {
    "annotation",
    "confirmed_legal_family",
    "family_membership",
    "ground_truth",
    "gt",
    "legal_family",
    "model_output",
    "model_score",
    "query",
    "query_text",
    "raw_family_quota",
    "retrieval_rank",
    "retrieval_score",
    "sampling_seed",
    "selected_case_ids",
}


class AcquisitionCohortError(ValueError):
    """Raised when the frozen cohort violates the Phase-2C protocol."""


def _canonical_text(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        raise AcquisitionCohortError(f"{field} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value):
        raise AcquisitionCohortError(f"{field} must use Unicode NFC")
    if value != " ".join(value.split()):
        raise AcquisitionCohortError(f"{field} must be whitespace-normalized")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise AcquisitionCohortError(f"{field} contains control characters")
    return value


def canonical_url(value: Any, field: str) -> str:
    checked = _canonical_text(value, field)
    assert checked is not None
    if any(character.isspace() for character in checked):
        raise AcquisitionCohortError(f"{field} contains unescaped whitespace")
    parsed = urlsplit(checked)
    if (
        parsed.scheme != "https"
        or parsed.hostname != OFFICIAL_DOMAIN
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise AcquisitionCohortError(f"{field} must be an official HTTPS URL")
    normalized = urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            quote(
                unquote(parsed.path),
                safe="/:@-._~!$&()*+,;=",
            ),
            parsed.query,
            "",
        )
    )
    if checked != normalized:
        raise AcquisitionCohortError(f"{field} is not canonically encoded")
    return normalized


def normalize_existing_url(value: Any, field: str = "existing source_url") -> str:
    """Canonicalize authoritative legacy intake URLs without rewriting them."""

    if not isinstance(value, str) or not value:
        raise AcquisitionCohortError(f"{field} must be a non-empty string")
    checked = unicodedata.normalize("NFC", value)
    parsed = urlsplit(checked)
    if (
        parsed.scheme != "https"
        or parsed.hostname != OFFICIAL_DOMAIN
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise AcquisitionCohortError(f"{field} must be an official HTTPS URL")
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            quote(
                unicodedata.normalize("NFC", unquote(parsed.path)),
                safe="/:@-._~!$&()*+,;=",
            ),
            parsed.query,
            "",
        )
    )


def _reject_forbidden_fields(value: Any, location: str = "cohort") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise AcquisitionCohortError(f"{location} contains a non-string key")
            if key.casefold() in FORBIDDEN_FIELD_NAMES:
                raise AcquisitionCohortError(
                    f"Forbidden cohort field at {location}.{key}"
                )
            _reject_forbidden_fields(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{location}[{index}]")


def fingerprint_payload(cohort: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(cohort))
    payload.pop("declared_at", None)
    payload.pop("cohort_fingerprint", None)
    return payload


def cohort_fingerprint(cohort: Mapping[str, Any]) -> str:
    return canonical_json_fingerprint(fingerprint_payload(cohort))


def _expected_ids() -> list[str]:
    return [f"{COHORT_ID}-{index:03d}" for index in range(1, TARGET_RAW_COUNT + 1)]


def _validate_discovery_summary(summary: Any) -> None:
    expected_keys = {
        "candidate_links_inspected",
        "unique_candidate_listing_links",
        "candidate_detail_pages_inspected",
        "pre_download_duplicates_rejected",
        "development_exact_duplicates_rejected",
        "rejection_reason_counts",
        "query_metadata_summary",
        "priority_counts_are_descriptive_not_quotas",
    }
    if not isinstance(summary, dict) or set(summary) != expected_keys:
        raise AcquisitionCohortError("Discovery summary schema changed")
    for key in (
        "candidate_links_inspected",
        "unique_candidate_listing_links",
        "candidate_detail_pages_inspected",
        "pre_download_duplicates_rejected",
        "development_exact_duplicates_rejected",
    ):
        if not isinstance(summary[key], int) or isinstance(summary[key], bool) or summary[key] < 0:
            raise AcquisitionCohortError(f"Invalid discovery summary count: {key}")
    if summary["priority_counts_are_descriptive_not_quotas"] is not True:
        raise AcquisitionCohortError("Discovery priorities must not become raw quotas")
    if not isinstance(summary["rejection_reason_counts"], dict):
        raise AcquisitionCohortError("Invalid discovery rejection summary")
    if not isinstance(summary["query_metadata_summary"], list):
        raise AcquisitionCohortError("Invalid query metadata summary")


def validate_cohort(cohort: dict, intake: dict) -> dict[str, Any]:
    if not isinstance(cohort, dict) or set(cohort) != ROOT_KEYS:
        raise AcquisitionCohortError("Cohort root schema changed")
    _reject_forbidden_fields(cohort)
    expected_root = {
        "cohort_version": COHORT_VERSION,
        "cohort_id": COHORT_ID,
        "status": COHORT_STATUS,
        "parent_acquisition_plan_commit": PARENT_PLAN_COMMIT,
        "parent_acquisition_plan_path": PARENT_PLAN_RELATIVE.as_posix(),
        "parent_acquisition_plan_sha256": PARENT_PLAN_SHA256,
        "plan_authoring_base_commit": PLAN_AUTHORING_BASE_COMMIT,
        "target_raw_count": TARGET_RAW_COUNT,
        "source_domain": OFFICIAL_DOMAIN,
        "discovery_priority_order": DISCOVERY_PRIORITIES,
        "outcome_adaptive_expansion_forbidden": True,
    }
    for key, expected in expected_root.items():
        if cohort.get(key) != expected:
            raise AcquisitionCohortError(f"Frozen cohort root field changed: {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", cohort["declaration_execution_commit"]):
        raise AcquisitionCohortError("Invalid declaration execution commit")
    _canonical_text(cohort["declared_at"], "declared_at")
    _validate_discovery_summary(cohort["discovery_summary"])

    entries = cohort.get("entries")
    if not isinstance(entries, list) or len(entries) != TARGET_RAW_COUNT:
        raise AcquisitionCohortError("Cohort must contain exactly 50 entries")
    ids = [entry.get("acquisition_id") for entry in entries if isinstance(entry, dict)]
    if ids != _expected_ids():
        raise AcquisitionCohortError("Cohort acquisition IDs/order changed")
    if len(set(ids)) != TARGET_RAW_COUNT:
        raise AcquisitionCohortError("Duplicate acquisition_id in cohort")

    source_urls: list[str] = []
    listing_urls: list[str] = []
    priority_ranks: list[int] = []
    optional_fields = (
        "documentary_dispute_type",
        "court",
        "decision_date",
        "procedure_stage",
        "official_document_title_or_filename",
        "official_case_or_judgment_id",
    )
    missing_optional = Counter()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise AcquisitionCohortError(f"Entry schema changed at index {index}")
        acquisition_id = entry["acquisition_id"]
        stable_source_id(acquisition_id)
        if entry["source_domain"] != OFFICIAL_DOMAIN:
            raise AcquisitionCohortError("Non-official source_domain")
        source_url = canonical_url(entry["source_url"], "source_url")
        listing_url = canonical_url(entry["official_listing_url"], "official_listing_url")
        if not urlsplit(source_url).path.casefold().endswith(".pdf"):
            raise AcquisitionCohortError("source_url must identify official PDF metadata")
        if not re.fullmatch(r"/5ta\d+t1cvn/.+\.pdf", urlsplit(source_url).path, re.I):
            raise AcquisitionCohortError("Unexpected official PDF URL shape")
        if not re.fullmatch(
            r"/2ta\d+t1cvn/chi-tiet-ban-an", urlsplit(listing_url).path
        ):
            raise AcquisitionCohortError("Unexpected official listing URL shape")
        if entry["discovery_metadata_source"] != listing_url:
            raise AcquisitionCohortError("Discovery metadata source must equal listing URL")
        source_urls.append(source_url)
        listing_urls.append(listing_url)
        priority = entry["discovery_priority"]
        if priority not in DISCOVERY_PRIORITIES:
            raise AcquisitionCohortError("Unknown discovery priority")
        priority_ranks.append(DISCOVERY_PRIORITIES.index(priority))
        for field in optional_fields:
            checked = _canonical_text(entry[field], field, nullable=True)
            if checked is None:
                missing_optional[field] += 1
        _canonical_text(entry["official_document_kind"], "official_document_kind")
        _canonical_text(entry["discovery_basis"], "discovery_basis")
        precheck = entry["pre_download_duplicate_check"]
        if not isinstance(precheck, dict) or set(precheck) != PRECHECK_KEYS:
            raise AcquisitionCohortError("Duplicate pre-screen schema changed")
        if (
            precheck["existing_intake_url_overlap"] is not False
            or precheck["new_cohort_url_overlap"] is not False
            or precheck["result"] != "NO_OBVIOUS_DUPLICATE_IN_AVAILABLE_METADATA"
        ):
            raise AcquisitionCohortError("Entry did not pass duplicate pre-screen")
    if len(set(source_urls)) != TARGET_RAW_COUNT:
        raise AcquisitionCohortError("Duplicate source_url in cohort")
    if len(set(listing_urls)) != TARGET_RAW_COUNT:
        raise AcquisitionCohortError("Duplicate official_listing_url in cohort")
    if priority_ranks != sorted(priority_ranks):
        raise AcquisitionCohortError("Discovery priority order is not deterministic")

    sources = intake.get("sources")
    if not isinstance(sources, list):
        raise AcquisitionCohortError("Invalid existing acquisition manifest")
    existing_ids = {source.get("acquisition_id") for source in sources}
    if existing_ids.intersection(ids):
        raise AcquisitionCohortError("Cohort acquisition_id already exists in intake")
    existing_urls = {
        normalize_existing_url(source["source_url"])
        for source in sources
        if source.get("source_url")
    }
    if existing_urls.intersection(source_urls):
        raise AcquisitionCohortError("Cohort source_url already exists in intake")
    existing_case_ids = {
        str(source.get("case_id", "")).casefold()
        for source in sources
        if source.get("case_id")
    }
    for entry in entries:
        case_id = entry["official_case_or_judgment_id"]
        if case_id and "/" in case_id and case_id.casefold() in existing_case_ids:
            raise AcquisitionCohortError("Cohort case identity already exists in intake")

    calculated = cohort_fingerprint(cohort)
    if cohort["cohort_fingerprint"] != calculated:
        raise AcquisitionCohortError("Cohort fingerprint mismatch")
    return {
        "source_urls": source_urls,
        "listing_urls": listing_urls,
        "priority_counts": dict(Counter(entry["discovery_priority"] for entry in entries)),
        "missing_optional_metadata": dict(sorted(missing_optional.items())),
        "cohort_fingerprint": calculated,
    }


def _git_provenance(project_root: Path, cohort: dict) -> dict[str, Any]:
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )

    head = run(["git", "rev-parse", "HEAD"])
    if head.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}\n?", head.stdout):
        raise AcquisitionCohortError("A valid execution Git commit is required")
    execution_commit = head.stdout.strip()
    for command, label in (
        (["git", "diff", "--quiet", "--"], "unstaged tracked"),
        (["git", "diff", "--cached", "--quiet", "--"], "staged tracked"),
    ):
        result = run(command)
        if result.returncode != 0:
            raise AcquisitionCohortError(f"Dirty {label} state is forbidden")
    for ancestor, descendant, label in (
        (PARENT_PLAN_COMMIT, execution_commit, "execution HEAD"),
        (PARENT_PLAN_COMMIT, cohort["declaration_execution_commit"], "declaration commit"),
    ):
        result = run(["git", "merge-base", "--is-ancestor", ancestor, descendant])
        if result.returncode != 0:
            raise AcquisitionCohortError(f"Parent plan is not ancestor of {label}")
    return {
        "execution_git_commit": execution_commit,
        "parent_acquisition_plan_commit": PARENT_PLAN_COMMIT,
        "plan_authoring_base_commit": PLAN_AUTHORING_BASE_COMMIT,
        "tracked_worktree_clean": True,
    }


def load_and_validate_cohort(
    project_root: Path, cohort_path: Path
) -> tuple[dict, dict[str, Any]]:
    root = Path(project_root).resolve()
    path = Path(cohort_path).resolve()
    cohort = strict_json_load(path)
    intake = strict_json_load(root / DEFAULT_INTAKE_RELATIVE)
    plan_path = root / PARENT_PLAN_RELATIVE
    if sha256_file(plan_path) != PARENT_PLAN_SHA256:
        raise AcquisitionCohortError("Parent acquisition-plan bytes changed")
    plan = strict_yaml_load(plan_path)
    if (
        plan.get("plan_status") != "ACQUISITION_PLAN_FROZEN_NOT_EXECUTED"
        or plan.get("batch_size_decision", {}).get("operational_reference_rule")
        != "HISTORICAL_COMPLETED_SCALE_BATCH_MEDIAN"
        or plan.get("targeting", {}).get("raw_family_quota") is not None
    ):
        raise AcquisitionCohortError("Parent acquisition-plan contract changed")
    return cohort, validate_cohort(cohort, intake)


def build_cohort_audit(project_root: Path, cohort_path: Path) -> dict[str, dict]:
    root = Path(project_root).resolve()
    cohort, validated = load_and_validate_cohort(root, cohort_path)
    entries = cohort["entries"]
    summary = cohort["discovery_summary"]
    priority_counts = {
        priority: validated["priority_counts"].get(priority, 0)
        for priority in DISCOVERY_PRIORITIES
    }
    return {
        "cohort_summary.json": {
            "cohort_id": COHORT_ID,
            "status": COHORT_STATUS,
            "declared_cohort_size": len(entries),
            "unique_source_urls": len(set(validated["source_urls"])),
            "official_domain_count": sum(
                entry["source_domain"] == OFFICIAL_DOMAIN for entry in entries
            ),
            "first_acquisition_id": entries[0]["acquisition_id"],
            "last_acquisition_id": entries[-1]["acquisition_id"],
            "missing_optional_metadata": validated["missing_optional_metadata"],
            "cohort_fingerprint": validated["cohort_fingerprint"],
            "pdf_downloaded_or_stored": False,
            "source_registered": False,
            "selection_performed": False,
        },
        "duplicate_prescreen.json": {
            "existing_intake_source_count": 271,
            "candidate_links_inspected": summary["candidate_links_inspected"],
            "unique_candidate_listing_links": summary[
                "unique_candidate_listing_links"
            ],
            "candidate_detail_pages_inspected": summary[
                "candidate_detail_pages_inspected"
            ],
            "pre_download_duplicates_rejected": summary[
                "pre_download_duplicates_rejected"
            ],
            "development_exact_duplicates_rejected": summary[
                "development_exact_duplicates_rejected"
            ],
            "rejection_reason_counts": summary["rejection_reason_counts"],
            "accepted_source_url_overlap_with_intake": 0,
            "accepted_duplicate_source_urls": 0,
        },
        "discovery_priority_summary.json": {
            "priority_order": DISCOVERY_PRIORITIES,
            "priority_counts": priority_counts,
            "counts_are_descriptive_not_raw_family_quotas": True,
            "family_specific_raw_yield_status": (
                "FAMILY_SPECIFIC_RAW_YIELD_UNCALIBRATED"
            ),
            "query_metadata_summary": summary["query_metadata_summary"],
        },
    }


def publish_cohort_audit(
    project_root: Path,
    cohort_path: Path,
    output_root: Path,
    structured: Mapping[str, dict],
) -> Path:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise AcquisitionCohortError("Audit output must remain inside repository") from exc
    cohort = strict_json_load(cohort_path)
    provenance = _git_provenance(root, cohort)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"
    partial = output / f"{run_id}.partial"
    final = output / run_id
    if partial.exists() or final.exists():
        raise AcquisitionCohortError("Unsafe cohort-audit output path")
    output.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    try:
        for name, payload in structured.items():
            strict_json_write(partial / name, payload)
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(partial.iterdir(), key=lambda item: item.name)
        }
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "run_id": run_id,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            **provenance,
            "cohort_id": COHORT_ID,
            "cohort_status": COHORT_STATUS,
            "cohort_fingerprint": cohort["cohort_fingerprint"],
            "cohort_file_sha256": sha256_file(cohort_path),
            "validator_source_sha256": sha256_file(Path(__file__)),
            "audit_fingerprint": canonical_json_fingerprint(dict(structured)),
            "output_file_sha256": output_hashes,
            "network_accessed_by_validator": False,
            "pdf_downloaded": False,
            "sources_registered": 0,
            "extraction_or_ocr_executed": False,
            "review_or_promotion_executed": False,
            "selection_performed": False,
            "sampling_seed_generated": False,
            "query_or_gt_generated": False,
            "models_or_retrievers_executed": False,
        }
        strict_json_write(partial / "run_manifest.json", manifest)
        expected = set(structured) | {"run_manifest.json"}
        if {path.name for path in partial.iterdir()} != expected:
            raise AcquisitionCohortError("Incomplete cohort audit")
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def run(
    *,
    project_root: Path | None = None,
    cohort_path: Path | None = None,
    output_root: Path | None = None,
) -> Path:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    cohort = (
        Path(cohort_path).resolve()
        if cohort_path is not None
        else root / DEFAULT_COHORT_RELATIVE
    )
    output = (
        Path(output_root).resolve()
        if output_root is not None
        else root / DEFAULT_RUNS_RELATIVE
    )
    structured = build_cohort_audit(root, cohort)
    return publish_cohort_audit(root, cohort, output, structured)


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Audit frozen Batch 2D-B cohort")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    result = run(
        project_root=args.project_root,
        cohort_path=args.cohort,
        output_root=args.output_root,
    )
    print(result)
    return result


if __name__ == "__main__":
    main()
