"""Offline validator for the frozen Phase-2D-B3.2-R2 Batch 2D-C cohort."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import re
import shutil
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


COHORT_VERSION = "independent_batch2dc_topup_cohort_v1"
COHORT_ID = "batch2dc-20261105"
COHORT_STATUS = "COHORT_DECLARED_NOT_ACQUIRED"
TARGET_RAW_COUNT = 50
OFFICIAL_DOMAIN = "congbobanan.toaan.gov.vn"
DISCOVERY_PRIORITY = "LAND_2024_TARGETED_DISCOVERY"
PARENT_PLAN_COMMIT = "aa91c488b73df8fd7ceaa6091b29e9f1af31f6ae"
PARENT_PLAN_SHA256 = "f7aa4f80b2d922fea860eace1472543290f66cc816dcda54cc50f8916f265167"
PARENT_PLAN_FINGERPRINT = "617faeb3c04f731cb1bc3442c13b2304337b01837ea644d96e1a6773bbe199c4"
PARENT_POOL_RUN_ID = "independent_topup_listing_pool_v1_20261105_213507_5f20aeec"
PARENT_POOL_FINGERPRINT = "029a0075f2c1edc2ecb74b980d746de5abf77410752a6ea1c57f70c8fa0552cb"
PARENT_UNIQUE_POOL_FINGERPRINT = "e61ebdf554985a8c020798c91afd0e4a59b65764727f8211b213bf7059511c6a"
DETAIL_RUN_ID = "independent_topup_detail_validation_v1_20261105_145220_da4b6e0a"
DETAIL_PROTOCOL_FINGERPRINT = "a0804e03ae52d13924d684fa70ce8d28d8ed05eb64c6c5986f2194369fb5325a"
DETAIL_LEDGER_FINGERPRINT = "0621debdd8e706bd7639a9cd7a8d4f3b0327bf4c4ac82b4aa0be6ff5508c1975"
OUTCOMES_FINGERPRINT = "95c00c1950e87a837036f03b90fd14a8e244004a9286b0e25ddcfd487e37f326"
ACCEPTED_POOL_FINGERPRINT = "38e8c34f7867c85d88aa8586bab7049be4c14fdb1809185b24992e30b8067535"
SELECTION_PROOF_FINGERPRINT = "48b637de6b149e3372459e34142bb6eeb7bc074698db4a151105443e9344252c"
ACQUISITION_MANIFEST_SHA256 = "4716e61118ce77dee60597940f68247423be685a27b6dde1c3c479393bbc5094"

DEFAULT_COHORT_RELATIVE = Path("backend/dataLaw/independent_intake/cohorts/batch2dc-20261105.json")
DEFAULT_PLAN_RELATIVE = Path("backend/scripts/Evaluation/configs/independent_topup_acquisition_plan_v1.yaml")
DEFAULT_INTAKE_RELATIVE = Path("backend/dataLaw/independent_intake/acquisition_manifest.json")
DEFAULT_DEVELOPMENT_DEDUP_RELATIVE = Path("backend/data/benchmarks/retrieval_independent_v1/source_dedup_report.json")
DEFAULT_BENCHMARK_RELATIVE = Path("backend/data/benchmarks/retrieval_independent_v1/benchmark_manifest.json")
DEFAULT_SAMPLING_RELATIVE = Path("backend/data/benchmarks/retrieval_independent_v1/sampling_manifest.json")
DEFAULT_DETAIL_RUN_RELATIVE = Path("backend/data/runs") / DETAIL_RUN_ID
DEFAULT_TEST_RELATIVE = Path("backend/scripts/Tests/test_independent_topup_acquisition_cohort.py")
DEFAULT_RUNS_RELATIVE = Path("backend/data/runs")
AUDIT_PROTOCOL_ID = "independent_batch2dc_cohort_audit_v1"

ROOT_KEYS = {
    "cohort_version", "cohort_id", "status", "parent_topup_plan_commit",
    "parent_topup_plan_path", "parent_topup_plan_sha256",
    "parent_topup_plan_fingerprint", "declaration_execution_commit",
    "declared_at", "target_raw_count", "source_domain",
    "discovery_priority_order", "outcome_adaptive_expansion_forbidden",
    "parent_listing_pool", "detail_validation", "full_pool_selection_proof",
    "discovery_summary", "entries", "cohort_fingerprint",
}
ENTRY_KEYS = {
    "acquisition_id", "source_domain", "source_url", "official_listing_url",
    "discovery_priority", "documentary_dispute_type", "court", "decision_date",
    "procedure_stage", "official_document_title_or_filename",
    "official_case_or_judgment_id", "official_document_kind", "discovery_basis",
    "discovery_metadata_source", "pre_download_duplicate_check",
}
PRECHECK_KEYS = {
    "existing_intake_url_overlap", "new_cohort_url_overlap",
    "exact_known_intake_case_overlap", "exact_development_case_overlap",
    "case_identity_comparison_limited_by_metadata", "result",
}
FORBIDDEN_FIELD_NAMES = {
    "annotation", "confirmed_legal_family", "family_membership", "ground_truth",
    "gt", "legal_family", "model_output", "model_score", "query", "query_text",
    "raw_family_quota", "retrieval_rank", "retrieval_result", "retrieval_score",
    "sampling_seed", "selected_benchmark_ids",
}


class TopupAcquisitionCohortError(ValueError):
    """Raised when the Batch 2D-C declaration violates its frozen protocol."""


def _text(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        raise TopupAcquisitionCohortError(f"{field} must be a non-empty string")
    if value != unicodedata.normalize("NFC", value) or value != " ".join(value.split()):
        raise TopupAcquisitionCohortError(f"{field} is not canonical text")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise TopupAcquisitionCohortError(f"{field} contains control characters")
    return value


def canonical_url(value: Any, field: str) -> str:
    checked = _text(value, field)
    assert checked is not None
    if any(character.isspace() for character in checked):
        raise TopupAcquisitionCohortError(f"{field} contains unescaped whitespace")
    parsed = urlsplit(checked)
    if (
        parsed.scheme != "https" or parsed.hostname != OFFICIAL_DOMAIN
        or parsed.username is not None or parsed.password is not None or parsed.fragment
    ):
        raise TopupAcquisitionCohortError(f"{field} must be an official HTTPS URL")
    normalized = urlunsplit((
        "https", parsed.netloc.lower(),
        quote(unicodedata.normalize("NFC", unquote(parsed.path)), safe="/:@-._~!$&()*+,;="),
        parsed.query, "",
    ))
    if normalized != checked:
        raise TopupAcquisitionCohortError(f"{field} is not canonically encoded")
    return normalized


def normalize_existing_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TopupAcquisitionCohortError("existing source_url must be a string")
    parsed = urlsplit(unicodedata.normalize("NFC", value))
    if parsed.scheme != "https" or parsed.hostname != OFFICIAL_DOMAIN or parsed.fragment:
        raise TopupAcquisitionCohortError("existing source_url must be official HTTPS")
    return urlunsplit((
        "https", parsed.netloc.lower(),
        quote(unicodedata.normalize("NFC", unquote(parsed.path)), safe="/:@-._~!$&()*+,;="),
        parsed.query, "",
    ))


def _identity(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", value)).casefold()


def _reject_forbidden_fields(value: Any, location: str = "cohort") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TopupAcquisitionCohortError(f"{location} has a non-string key")
            if key.casefold() in FORBIDDEN_FIELD_NAMES:
                raise TopupAcquisitionCohortError(f"Forbidden field at {location}.{key}")
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


def _validate_bindings(cohort: Mapping[str, Any]) -> None:
    expected_pool = {
        "run_id": PARENT_POOL_RUN_ID,
        "listing_pool_fingerprint": PARENT_POOL_FINGERPRINT,
        "unique_pool_fingerprint": PARENT_UNIQUE_POOL_FINGERPRINT,
        "listing_position_fingerprint": "f877b1231d68e7cadea40e4043ccdedb81d3a01ebc50f487a7ff41fe60fe2431",
        "duplicate_mapping_fingerprint": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "position_count": 160,
        "unique_canonical_listing_url_count": 160,
        "duplicate_url_group_count": 0,
    }
    if cohort.get("parent_listing_pool") != expected_pool:
        raise TopupAcquisitionCohortError("R1.2 listing-pool binding changed")
    expected_detail = {
        "run_id": DETAIL_RUN_ID,
        "protocol_fingerprint": DETAIL_PROTOCOL_FINGERPRINT,
        "detail_fetch_ledger_fingerprint": DETAIL_LEDGER_FINGERPRINT,
        "all_160_outcome_fingerprint": OUTCOMES_FINGERPRINT,
        "accepted_pool_fingerprint": ACCEPTED_POOL_FINGERPRINT,
        "selection_proof_fingerprint": SELECTION_PROOF_FINGERPRINT,
        "detail_requests_attempted": 160,
        "detail_pages_completed": 160,
        "pdf_requests": 0,
    }
    if cohort.get("detail_validation") != expected_detail:
        raise TopupAcquisitionCohortError("R2 detail-validation binding changed")


def _validate_selection_proof(cohort: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    proof = cohort.get("full_pool_selection_proof")
    expected_keys = {
        "frozen_listing_urls", "accepted_sorted_listing_urls", "selected_listing_urls",
        "candidate_51_listing_url", "candidate_validation_complete", "accepted_count",
        "rejected_count", "selected_count", "selection_rule",
        "selected_is_exact_accepted_prefix", "earlier_accepted_url_skipped",
    }
    if not isinstance(proof, dict) or set(proof) != expected_keys:
        raise TopupAcquisitionCohortError("Full-pool selection proof schema changed")
    frozen = proof["frozen_listing_urls"]
    accepted = proof["accepted_sorted_listing_urls"]
    selected = proof["selected_listing_urls"]
    for label, values, expected_count in (
        ("frozen", frozen, 160), ("accepted", accepted, 140), ("selected", selected, 50)
    ):
        if not isinstance(values, list) or len(values) != expected_count:
            raise TopupAcquisitionCohortError(f"{label} URL count changed")
        checked = [canonical_url(value, f"{label} URL") for value in values]
        if checked != sorted(checked) or len(set(checked)) != len(checked):
            raise TopupAcquisitionCohortError(f"{label} URLs are not sorted and unique")
    if canonical_json_fingerprint(frozen) != PARENT_UNIQUE_POOL_FINGERPRINT:
        raise TopupAcquisitionCohortError("Frozen 160 URL fingerprint mismatch")
    if not set(accepted).issubset(frozen):
        raise TopupAcquisitionCohortError("Accepted URL is absent from frozen 160 pool")
    if selected != accepted[:50] or proof["candidate_51_listing_url"] != accepted[50]:
        raise TopupAcquisitionCohortError("Selection is not the exact first 50 accepted")
    if proof != {
        **proof,
        "candidate_validation_complete": "160/160",
        "accepted_count": 140,
        "rejected_count": 20,
        "selected_count": 50,
        "selection_rule": "first 50 accepted candidates by canonical_listing_url ASC",
        "selected_is_exact_accepted_prefix": True,
        "earlier_accepted_url_skipped": False,
    }:
        raise TopupAcquisitionCohortError("Full-pool completion/selection contract changed")
    return frozen, accepted


def _validate_summary(summary: Any) -> None:
    expected = {
        "frozen_listing_candidates": 160,
        "detail_pages_inspected": 160,
        "detail_fetch_failures": 0,
        "metadata_qualified_candidates": 160,
        "missing_pdf_link_count": 0,
        "non_land_intent_metadata_rejections": 0,
        "existing_intake_url_duplicates": 19,
        "exact_intake_identity_duplicates": 0,
        "exact_development_duplicates": 0,
        "within_pool_pdf_duplicates": 0,
        "within_pool_exact_identity_duplicates": 1,
        "accepted_total": 140,
        "rejected_total": 20,
        "selected_total": 50,
        "rejection_reason_counts": {
            "DUPLICATE_EXACT_IDENTITY_WITHIN_FROZEN_POOL": 1,
            "EXISTING_INTAKE_SOURCE_URL": 19,
        },
    }
    if summary != expected:
        raise TopupAcquisitionCohortError("Discovery summary changed")


def validate_cohort(
    cohort: Mapping[str, Any], intake: Mapping[str, Any], development_dedup: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(cohort, dict) or set(cohort) != ROOT_KEYS:
        raise TopupAcquisitionCohortError("Cohort root schema changed")
    _reject_forbidden_fields(cohort)
    expected_scalars = {
        "cohort_version": COHORT_VERSION,
        "cohort_id": COHORT_ID,
        "status": COHORT_STATUS,
        "parent_topup_plan_commit": PARENT_PLAN_COMMIT,
        "parent_topup_plan_path": str(DEFAULT_PLAN_RELATIVE),
        "parent_topup_plan_sha256": PARENT_PLAN_SHA256,
        "parent_topup_plan_fingerprint": PARENT_PLAN_FINGERPRINT,
        "declaration_execution_commit": PARENT_PLAN_COMMIT,
        "target_raw_count": TARGET_RAW_COUNT,
        "source_domain": OFFICIAL_DOMAIN,
        "discovery_priority_order": [DISCOVERY_PRIORITY],
        "outcome_adaptive_expansion_forbidden": True,
    }
    for key, expected in expected_scalars.items():
        if cohort.get(key) != expected:
            raise TopupAcquisitionCohortError(f"Frozen cohort field changed: {key}")
    _text(cohort["declared_at"], "declared_at")
    _validate_bindings(cohort)
    frozen_urls, accepted_urls = _validate_selection_proof(cohort)
    _validate_summary(cohort["discovery_summary"])

    entries = cohort.get("entries")
    if not isinstance(entries, list) or len(entries) != TARGET_RAW_COUNT:
        raise TopupAcquisitionCohortError("Cohort must contain exactly 50 entries")
    ids = [entry.get("acquisition_id") for entry in entries if isinstance(entry, dict)]
    if ids != _expected_ids():
        raise TopupAcquisitionCohortError("Batch2DC acquisition IDs/order changed")
    if any(stable_source_id(identifier) != f"independent:{identifier}" for identifier in ids):
        raise TopupAcquisitionCohortError("Stable source ID derivation changed")

    existing_sources = intake.get("sources")
    if not isinstance(existing_sources, list) or len(existing_sources) != 317:
        raise TopupAcquisitionCohortError("Authoritative intake count changed")
    statuses = Counter(source.get("acquisition_status") for source in existing_sources)
    if statuses != Counter({"eligible": 257, "extracted": 59, "hash_verified": 1}):
        raise TopupAcquisitionCohortError("Authoritative intake statuses changed")
    existing_ids = {source.get("acquisition_id") for source in existing_sources}
    existing_urls = {normalize_existing_url(source["source_url"]) for source in existing_sources}
    existing_identities = {
        identity for source in existing_sources
        for identity in (_identity(source.get("case_id")), _identity(source.get("judgment_number")))
        if identity
    }
    development_identities = {
        _identity(item.get("case_id"))
        for item in development_dedup.get("development_source_mapping", [])
        if _identity(item.get("case_id"))
    }
    if development_dedup.get("development_record_count") != 107:
        raise TopupAcquisitionCohortError("Development source count changed")

    source_urls: list[str] = []
    listing_urls: list[str] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != ENTRY_KEYS:
            raise TopupAcquisitionCohortError(f"Entry schema changed at {index}")
        if entry["source_domain"] != OFFICIAL_DOMAIN or entry["discovery_priority"] != DISCOVERY_PRIORITY:
            raise TopupAcquisitionCohortError("Entry domain/priority changed")
        source_url = canonical_url(entry["source_url"], "source_url")
        listing_url = canonical_url(entry["official_listing_url"], "official_listing_url")
        if not re.fullmatch(r"/5ta\d+t1cvn/.+\.pdf", urlsplit(source_url).path, flags=re.IGNORECASE):
            raise TopupAcquisitionCohortError("Unexpected official PDF URL shape")
        if not re.fullmatch(r"/2ta\d+t1cvn/chi-tiet-ban-an", urlsplit(listing_url).path):
            raise TopupAcquisitionCohortError("Unexpected official listing URL shape")
        if entry["discovery_metadata_source"] != listing_url:
            raise TopupAcquisitionCohortError("Discovery metadata source differs from listing URL")
        if listing_url != cohort["full_pool_selection_proof"]["selected_listing_urls"][index]:
            raise TopupAcquisitionCohortError("Entry order differs from deterministic selection")
        for field in (
            "documentary_dispute_type", "court", "procedure_stage",
            "official_document_title_or_filename", "official_case_or_judgment_id",
            "official_document_kind", "discovery_basis",
        ):
            _text(entry[field], field)
        if not re.fullmatch(r"\d{2}/\d{2}/\d{4}", entry["decision_date"]):
            raise TopupAcquisitionCohortError("Decision date must use DD/MM/YYYY")
        if "land-related legal relation" not in entry["discovery_basis"]:
            raise TopupAcquisitionCohortError("Discovery basis is not deterministic LAND targeting")
        precheck = entry["pre_download_duplicate_check"]
        if not isinstance(precheck, dict) or set(precheck) != PRECHECK_KEYS:
            raise TopupAcquisitionCohortError("Duplicate pre-screen schema changed")
        if (
            precheck["existing_intake_url_overlap"] is not False
            or precheck["new_cohort_url_overlap"] is not False
            or precheck["result"] != "NO_OBVIOUS_DUPLICATE_IN_AVAILABLE_METADATA"
        ):
            raise TopupAcquisitionCohortError("Entry did not pass duplicate pre-screen")
        limited = precheck["case_identity_comparison_limited_by_metadata"]
        if not isinstance(limited, bool):
            raise TopupAcquisitionCohortError("Identity limitation must be boolean")
        expected_overlap = None if limited else False
        if (
            precheck["exact_known_intake_case_overlap"] is not expected_overlap
            or precheck["exact_development_case_overlap"] is not expected_overlap
        ):
            raise TopupAcquisitionCohortError("Identity overlap/limitation semantics changed")
        identity = _identity(entry["official_case_or_judgment_id"])
        if not limited and identity in existing_identities:
            raise TopupAcquisitionCohortError("Exact known intake identity overlap")
        if not limited and identity in development_identities:
            raise TopupAcquisitionCohortError("Exact development identity overlap")
        source_urls.append(source_url)
        listing_urls.append(listing_url)
    if len(set(source_urls)) != 50:
        raise TopupAcquisitionCohortError("Duplicate source_url in cohort")
    if len(set(listing_urls)) != 50:
        raise TopupAcquisitionCohortError("Duplicate listing URL in cohort")
    if set(ids) & existing_ids:
        raise TopupAcquisitionCohortError("Batch2DC source is already registered")
    if set(source_urls) & existing_urls:
        raise TopupAcquisitionCohortError("Existing intake source URL overlap")
    if listing_urls != accepted_urls[:50] or not set(listing_urls).issubset(frozen_urls):
        raise TopupAcquisitionCohortError("Selected entries do not prove exact frozen-pool prefix")
    calculated = cohort_fingerprint(cohort)
    if cohort["cohort_fingerprint"] != calculated:
        raise TopupAcquisitionCohortError("Cohort fingerprint mismatch")
    return {
        "cohort_fingerprint": calculated,
        "source_urls": source_urls,
        "listing_urls": listing_urls,
        "priority_counts": dict(Counter(entry["discovery_priority"] for entry in entries)),
        "intake_counts": {"total": 317, "eligible": 257, "extracted": 59, "hash_verified": 1},
    }


def _validate_protected_state(root: Path, acquisition_ids: set[str]) -> dict[str, Any]:
    benchmark = strict_json_load(root / DEFAULT_BENCHMARK_RELATIVE)
    sampling = strict_json_load(root / DEFAULT_SAMPLING_RELATIVE)
    counts = benchmark.get("current_counts", {})
    expected = {
        "benchmark": "DRAFT", "sampling": "DRAFT", "selection": "NOT_STARTED",
        "sampling_seed": None, "selected_cases": 0, "query_records": 0,
        "ground_truth_records": 0,
    }
    observed = {
        "benchmark": benchmark.get("state"), "sampling": sampling.get("state"),
        "selection": sampling.get("selection_status"),
        "sampling_seed": sampling.get("sampling_seed"),
        "selected_cases": counts.get("selected_cases"),
        "query_records": counts.get("query_records"),
        "ground_truth_records": counts.get("ground_truth_records"),
    }
    if observed != expected:
        raise TopupAcquisitionCohortError("Protected benchmark state changed")
    intake_root = root / "backend/dataLaw/independent_intake"
    unexpected: list[str] = []
    for area in ("raw", "structured", "provenance", "transactions", "quarantine"):
        base = intake_root / area
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if any(identifier in path.name for identifier in acquisition_ids):
                unexpected.append(str(path.relative_to(root)))
    if unexpected:
        raise TopupAcquisitionCohortError("Batch2DC acquisition artifacts already exist")
    return {**expected, "batch2dc_acquisition_artifacts": 0}


def load_and_validate_cohort(project_root: Path, cohort_path: Path) -> tuple[dict, dict[str, Any]]:
    root = Path(project_root).resolve()
    cohort = strict_json_load(Path(cohort_path).resolve())
    plan_path = root / DEFAULT_PLAN_RELATIVE
    if sha256_file(plan_path) != PARENT_PLAN_SHA256:
        raise TopupAcquisitionCohortError("Parent top-up plan bytes changed")
    if canonical_json_fingerprint(strict_yaml_load(plan_path)) != PARENT_PLAN_FINGERPRINT:
        raise TopupAcquisitionCohortError("Parent top-up plan fingerprint changed")
    intake_path = root / DEFAULT_INTAKE_RELATIVE
    if sha256_file(intake_path) != ACQUISITION_MANIFEST_SHA256:
        raise TopupAcquisitionCohortError("Acquisition manifest bytes changed")
    intake = strict_json_load(intake_path)
    development = strict_json_load(root / DEFAULT_DEVELOPMENT_DEDUP_RELATIVE)
    validated = validate_cohort(cohort, intake, development)
    validated["protected_state"] = _validate_protected_state(root, set(_expected_ids()))
    return cohort, validated


def build_audit(project_root: Path, cohort_path: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    cohort, validated = load_and_validate_cohort(root, cohort_path)
    detail_run = root / DEFAULT_DETAIL_RUN_RELATIVE
    request_state = strict_json_load(detail_run / "request_state.json")
    ledger = strict_json_load(detail_run / "detail_fetch_ledger.json")
    if (
        request_state.get("detail_requests_attempted") != 160
        or request_state.get("detail_responses_completed") != 160
        or request_state.get("pdf_requests") != 0
        or len(ledger.get("records", [])) != 160
        or any(record.get("pdf_request") is not False for record in ledger["records"])
        or ledger.get("records_fingerprint") != DETAIL_LEDGER_FINGERPRINT
    ):
        raise TopupAcquisitionCohortError("Detail request/PDF firewall evidence changed")
    entries = cohort["entries"]
    proof = cohort["full_pool_selection_proof"]
    audit = {
        "protocol_id": AUDIT_PROTOCOL_ID,
        "cohort_id": COHORT_ID,
        "cohort_status": COHORT_STATUS,
        "detail_validation_run_id": DETAIL_RUN_ID,
        "detail_validation_protocol_fingerprint": DETAIL_PROTOCOL_FINGERPRINT,
        "all_160_candidate_outcome_fingerprint": OUTCOMES_FINGERPRINT,
        "rejection_summary": cohort["discovery_summary"],
        "accepted_candidate_fingerprint": ACCEPTED_POOL_FINGERPRINT,
        "deterministic_selection_proof": proof,
        "selected_50_ids": [entry["acquisition_id"] for entry in entries],
        "selected_50_listing_urls": [entry["official_listing_url"] for entry in entries],
        "selected_50_pdf_urls": [entry["source_url"] for entry in entries],
        "cohort_fingerprint": validated["cohort_fingerprint"],
        "cohort_file_sha256": sha256_file(cohort_path),
        "validator_sha256": sha256_file(Path(__file__)),
        "test_sha256": sha256_file(root / DEFAULT_TEST_RELATIVE),
        "topup_plan_sha256": PARENT_PLAN_SHA256,
        "topup_plan_fingerprint": PARENT_PLAN_FINGERPRINT,
        "listing_pool_fingerprint": PARENT_POOL_FINGERPRINT,
        "acquisition_manifest_sha256": ACQUISITION_MANIFEST_SHA256,
        "detail_requests": 160,
        "pdf_requests": 0,
        "pdf_body_fetched": False,
        "sources_registered": 0,
        "selection_benchmark_performed": False,
        "protected_state": validated["protected_state"],
    }
    audit["audit_fingerprint"] = canonical_json_fingerprint(audit)
    return audit


def publish_audit(project_root: Path, cohort_path: Path, output_root: Path) -> Path:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise TopupAcquisitionCohortError("Audit output must remain inside repository") from exc
    audit = build_audit(root, cohort_path)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run_id = f"{AUDIT_PROTOCOL_ID}_{stamp}_{uuid.uuid4().hex[:8]}"
    partial = output / f"{run_id}.partial"
    final = output / run_id
    output.mkdir(parents=True, exist_ok=True)
    if partial.exists() or final.exists():
        raise TopupAcquisitionCohortError("Unsafe audit output path")
    partial.mkdir()
    try:
        strict_json_write(partial / "audit.json", audit)
        manifest = {
            "protocol_id": AUDIT_PROTOCOL_ID,
            "run_id": run_id,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "git_head": PARENT_PLAN_COMMIT,
            "audit_fingerprint": audit["audit_fingerprint"],
            "audit_sha256": sha256_file(partial / "audit.json"),
            "network_accessed_by_validator": False,
            "pdf_requests": 0,
            "sources_registered": 0,
            "selection_benchmark_performed": False,
        }
        manifest["run_fingerprint"] = canonical_json_fingerprint(manifest)
        strict_json_write(partial / "run_manifest.json", manifest)
        os.replace(partial, final)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return final


def main(argv: list[str] | None = None) -> Path | None:
    parser = argparse.ArgumentParser(description="Validate frozen Batch 2D-C cohort")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--publish-audit", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    root = args.project_root.resolve() if args.project_root else Path(__file__).resolve().parents[4]
    cohort_path = args.cohort.resolve() if args.cohort else root / DEFAULT_COHORT_RELATIVE
    if args.publish_audit:
        output = args.output_root.resolve() if args.output_root else root / DEFAULT_RUNS_RELATIVE
        result = publish_audit(root, cohort_path, output)
        print(result)
        return result
    cohort, validated = load_and_validate_cohort(root, cohort_path)
    print(json.dumps({
        "cohort_id": cohort["cohort_id"],
        "cohort_fingerprint": validated["cohort_fingerprint"],
        "entries": len(cohort["entries"]),
        "priority_counts": validated["priority_counts"],
        "protected_state": validated["protected_state"],
    }, ensure_ascii=False, sort_keys=True))
    return None


if __name__ == "__main__":
    main()
