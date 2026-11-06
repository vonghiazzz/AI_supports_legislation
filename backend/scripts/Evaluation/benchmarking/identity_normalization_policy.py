"""Offline validation for one frozen source-identity separator exception.

This module defines an exact allowlist. It never mutates the acquisition
manifest and is intentionally not integrated into record_source_identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .source_intake import (
    REVIEW_METADATA_FIELDS,
    load_acquisition_manifest,
    validate_source_identity_payload,
)


POLICY_ID = "independent_identity_normalization_policy_v1"
POLICY_VERSION = POLICY_ID
POLICY_STATUS = "FROZEN_FOR_IDENTITY_ADJUDICATION"
POLICY_RESULT = "SEPARATOR_NORMALIZATION_POLICY_APPROVED"
DEFAULT_POLICY_RELATIVE = Path(
    "backend/scripts/Evaluation/configs/"
    "independent_identity_normalization_policy_v1.yaml"
)
EXPECTED_HEAD = "93b5f260b34974da02ed291423ceef0a3cd3a330"
EXPECTED_MANIFEST_SHA256 = (
    "6933ef27c489ba51da1d3f2d157b60ca93a6ae57a51eb8e67570fda0d68add5d"
)
EXPECTED_SOURCE_INTAKE_SHA256 = (
    "f0c093378723e916006d19887facbea44b0df0798ec0be87c8685288cf76c8c9"
)
EXPECTED_BLOCKED_RUN_ID = (
    "independent_batch2dc_identity_v1_20261106_033252_694479_944f4fda"
)
EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT = (
    "3d308a9665c163c0d20eca39dfb57ebe2958e4b7473e5e1ba2cf2bdaa0bb5f9c"
)
EXPECTED_BLOCKED_RUN_FINGERPRINT = (
    "791c3838e852c35dbee391260b2b93a9d97193cb770c22c94444f5e1193c252f"
)
EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT = (
    "b9f44e22a1536bd1a4929544909c85170a25a3ec99e68ef960793f884025f9c0"
)
EXPECTED_ACQUISITION_ID = "batch2dc-20261105-039"
EXPECTED_FIELD = "judgment_number"
EXPECTED_RAW_VALUE = "25 2026/QĐST-DS"
EXPECTED_CANONICAL_VALUE = "25/2026/QĐST-DS"
EXPECTED_RAW_SHA256 = (
    "cb62d2943a67975c40bffcac11c18ab57920ba23c8dc2f0900df463276a0da7d"
)
EXPECTED_STRUCTURED_SHA256 = (
    "5138c691effa572801ce96e0a7db214fd1a107f5d09a6212a6b02e1b49106acf"
)
EXPECTED_CORROBORATION_SHA256 = (
    "98f8a1208d2f9ee2f40bac5d905348cfcbcc9543fd9fbde8845e005de03e6ec3"
)

MALFORMED_SHAPE = re.compile(
    r"^(\d{1,4})\s+((?:19|20)\d{2}/[A-ZĐ0-9-]+)$"
)
CANONICAL_SHAPE = re.compile(
    r"^\d{1,4}/(?:19|20)\d{2}/[A-ZĐ0-9-]+$"
)

ROOT_FIELDS = {
    "policy_id",
    "policy_version",
    "status",
    "policy_fingerprint",
    "bindings",
    "scope",
    "transformation",
    "corroboration",
    "prohibitions",
    "protected_state",
}
EXPECTED_BINDINGS = {
    "parent_technical_checkpoint_commit": EXPECTED_HEAD,
    "acquisition_manifest_path":
        "backend/dataLaw/independent_intake/acquisition_manifest.json",
    "acquisition_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "source_intake_path":
        "backend/scripts/Evaluation/benchmarking/source_intake.py",
    "source_intake_sha256": EXPECTED_SOURCE_INTAKE_SHA256,
    "blocked_b3_4_run_id": EXPECTED_BLOCKED_RUN_ID,
    "blocked_b3_4_protocol_fingerprint":
        EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT,
    "blocked_b3_4_run_fingerprint": EXPECTED_BLOCKED_RUN_FINGERPRINT,
    "blocked_b3_4_resolved_count": 45,
    "blocked_b3_4_unresolved_count": 1,
    "blocked_b3_4_identity_application_count": 0,
    "provisional_45_payload_fingerprint":
        EXPECTED_PROVISIONAL_PAYLOAD_FINGERPRINT,
    "raw_pdf_path":
        "backend/dataLaw/independent_intake/raw/batch2dc-20261105-039/25.pdf",
    "raw_pdf_sha256": EXPECTED_RAW_SHA256,
    "structured_path":
        "backend/dataLaw/independent_intake/structured/"
        "batch2dc-20261105-039/25_structured.json",
    "structured_sha256": EXPECTED_STRUCTURED_SHA256,
}
EXPECTED_SCOPE = {
    "acquisition_id": EXPECTED_ACQUISITION_ID,
    "field": EXPECTED_FIELD,
    "raw_observed_value": EXPECTED_RAW_VALUE,
    "canonical_identity_value": EXPECTED_CANONICAL_VALUE,
    "current_document_heading": "Số:",
    "current_document_kind": "Quyết định",
    "current_document_procedure_stage": "Sơ thẩm",
    "current_document_case_category": "Dân sự",
    "competing_current_document_identity_count": 0,
}
EXPECTED_TRANSFORMATION = {
    "rule": "SINGLE_MISSING_STRUCTURAL_SLASH_NORMALIZATION",
    "malformed_shape_regex":
        r"^(\d{1,4})\s+((?:19|20)\d{2}/[A-ZĐ0-9-]+)$",
    "canonical_shape_regex":
        r"^\d{1,4}/(?:19|20)\d{2}/[A-ZĐ0-9-]+$",
    "sequence_group": "25",
    "remainder_group": "2026/QĐST-DS",
    "replacement": "replace exactly one ASCII-space boundary with one slash",
    "changed_boundary_count": 1,
    "whitespace_semantics": "exactly_one_ascii_space",
    "alter_digits": False,
    "alter_year": False,
    "alter_suffix": False,
    "alter_other_punctuation": False,
}
EXPECTED_CORROBORATION = {
    "result": "PASS",
    "run_id":
        "independent_topup_detail_validation_v1_20261105_145220_da4b6e0a",
    "artifact_path":
        "backend/data/runs/"
        "independent_topup_detail_validation_v1_20261105_145220_da4b6e0a/"
        "detail_html/040_2ta2270343t1cvn.html",
    "artifact_sha256": EXPECTED_CORROBORATION_SHA256,
    "frozen_before_b3_4": True,
    "official_detail_url":
        "https://congbobanan.toaan.gov.vn/"
        "2ta2270343t1cvn/chi-tiet-ban-an",
    "official_sequence_number": "25",
    "official_decision_date": "31/07/2026",
    "official_document_kind": "Quyết định",
    "official_procedure_stage": "Sơ thẩm",
    "official_case_category": "Dân sự",
    "independently_represented_identity": EXPECTED_CANONICAL_VALUE,
    "exact_required_fragments": [
        "<label>Quyết định số: </label>",
        "<span>25<br/> ngày 31/07/2026</span>",
        "<label>Cấp xét xử: </label><span>Sơ thẩm</span>",
        "<label>Loại vụ/việc:</label><span>Dân sự</span>",
    ],
    "filename_support_only": "25.pdf",
}
EXPECTED_PROHIBITIONS = {
    "no_network",
    "no_generic_typo_correction",
    "no_wildcard_or_source_family_rule",
    "no_other_acquisition_id",
    "no_other_field",
    "no_case_id_normalization",
    "no_related_decision_id_normalization",
    "no_identity_mutation_in_r1",
    "no_source_review",
    "no_promotion",
    "future_exception_requires_separate_policy_amendment",
}
EXPECTED_PROTECTED_STATE = {
    "global_total": 363,
    "global_eligible": 257,
    "global_extracted": 105,
    "global_hash_verified": 1,
    "batch2dc_extracted": 46,
    "batch2dc_identity_ready": 0,
    "batch2dc_eligible": 0,
    "benchmark_state": "DRAFT",
    "sampling_state": "DRAFT",
    "selection_status": "NOT_STARTED",
    "sampling_seed": None,
    "selected_count": 0,
    "query_count": 0,
    "ground_truth_count": 0,
}


class IdentityNormalizationPolicyError(ValueError):
    """The frozen normalization policy or one of its bindings is invalid."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise IdentityNormalizationPolicyError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


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


def policy_fingerprint(policy: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(policy))
    payload.pop("policy_fingerprint", None)
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_policy(path: Path) -> dict:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise IdentityNormalizationPolicyError(
            f"Policy must be a regular non-symlink file: {path}"
        )
    value = yaml.load(
        path.read_text(encoding="utf-8"), Loader=_UniqueKeySafeLoader
    )
    if not isinstance(value, dict):
        raise IdentityNormalizationPolicyError("Policy root must be a mapping")
    return value


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise IdentityNormalizationPolicyError(
            f"{label} differs from frozen policy"
        )


def _validate_shape(policy: Mapping[str, Any]) -> None:
    if set(policy) != ROOT_FIELDS:
        raise IdentityNormalizationPolicyError(
            "policy must contain exact frozen fields"
        )
    _require_equal(policy["policy_id"], POLICY_ID, "policy_id")
    _require_equal(policy["policy_version"], POLICY_VERSION, "policy_version")
    _require_equal(policy["status"], POLICY_STATUS, "status")
    fingerprint = policy["policy_fingerprint"]
    if not isinstance(fingerprint, str) or not re.fullmatch(
        r"[0-9a-f]{64}", fingerprint
    ):
        raise IdentityNormalizationPolicyError("policy_fingerprint is invalid")
    _require_equal(fingerprint, policy_fingerprint(policy), "policy_fingerprint")
    _require_equal(policy["bindings"], EXPECTED_BINDINGS, "bindings")
    _require_equal(policy["scope"], EXPECTED_SCOPE, "scope")
    _require_equal(
        policy["transformation"], EXPECTED_TRANSFORMATION, "transformation"
    )
    if not isinstance(policy["corroboration"], Mapping):
        raise IdentityNormalizationPolicyError("corroboration must be a mapping")
    if policy["corroboration"].get("result") != "PASS":
        raise IdentityNormalizationPolicyError(
            "mandatory same-source corroboration must be PASS"
        )
    _require_equal(
        policy["corroboration"], EXPECTED_CORROBORATION, "corroboration"
    )
    corroboration = policy["corroboration"]
    date_match = re.fullmatch(
        r"\d{2}/\d{2}/((?:19|20)\d{2})",
        corroboration["official_decision_date"],
    )
    if date_match is None:
        raise IdentityNormalizationPolicyError(
            "official corroboration date is not canonical"
        )
    component_codes = {
        "official_document_kind": {"Quyết định": "QĐ"},
        "official_procedure_stage": {"Sơ thẩm": "ST"},
        "official_case_category": {"Dân sự": "DS"},
    }
    try:
        suffix = (
            component_codes["official_document_kind"][
                corroboration["official_document_kind"]
            ]
            + component_codes["official_procedure_stage"][
                corroboration["official_procedure_stage"]
            ]
            + "-"
            + component_codes["official_case_category"][
                corroboration["official_case_category"]
            ]
        )
    except KeyError as exc:
        raise IdentityNormalizationPolicyError(
            "official corroboration components are unsupported"
        ) from exc
    represented = (
        f"{corroboration['official_sequence_number']}/"
        f"{date_match.group(1)}/{suffix}"
    )
    _require_equal(
        represented,
        corroboration["independently_represented_identity"],
        "independently represented identity",
    )
    prohibitions = policy["prohibitions"]
    if (
        not isinstance(prohibitions, list)
        or len(prohibitions) != len(EXPECTED_PROHIBITIONS)
        or set(prohibitions) != EXPECTED_PROHIBITIONS
    ):
        raise IdentityNormalizationPolicyError("prohibitions are not exact")
    _require_equal(
        policy["protected_state"], EXPECTED_PROTECTED_STATE, "protected_state"
    )

    match = MALFORMED_SHAPE.fullmatch(EXPECTED_RAW_VALUE)
    if match is None or match.groups() != ("25", "2026/QĐST-DS"):
        raise IdentityNormalizationPolicyError("frozen raw shape does not match")
    if EXPECTED_RAW_VALUE.count(" ") != 1:
        raise IdentityNormalizationPolicyError(
            "raw value must contain one ASCII-space boundary"
        )
    repaired = f"{match.group(1)}/{match.group(2)}"
    _require_equal(repaired, EXPECTED_CANONICAL_VALUE, "repaired value")
    if CANONICAL_SHAPE.fullmatch(repaired) is None:
        raise IdentityNormalizationPolicyError("repaired value is not canonical")
    changes = [
        (before, after)
        for before, after in zip(EXPECTED_RAW_VALUE, repaired)
        if before != after
    ]
    if len(repaired) != len(EXPECTED_RAW_VALUE) or changes != [(" ", "/")]:
        raise IdentityNormalizationPolicyError(
            "transformation is not one separator replacement"
        )


def _load_json(path: Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise IdentityNormalizationPolicyError(
            f"Required artifact must be a regular file: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise IdentityNormalizationPolicyError(f"Invalid JSON root: {path}")
    return value


def _validate_artifacts(policy: Mapping[str, Any], root: Path) -> None:
    bindings = policy["bindings"]
    for path_key, hash_key in (
        ("acquisition_manifest_path", "acquisition_manifest_sha256"),
        ("source_intake_path", "source_intake_sha256"),
        ("raw_pdf_path", "raw_pdf_sha256"),
        ("structured_path", "structured_sha256"),
    ):
        path = root / bindings[path_key]
        if not path.is_file() or path.is_symlink():
            raise IdentityNormalizationPolicyError(
                f"Bound artifact is invalid: {path_key}"
            )
        _require_equal(sha256_file(path), bindings[hash_key], hash_key)

    corroboration = policy["corroboration"]
    corroboration_path = root / corroboration["artifact_path"]
    if not corroboration_path.is_file() or corroboration_path.is_symlink():
        raise IdentityNormalizationPolicyError(
            "corroboration artifact is not a regular file"
        )
    _require_equal(
        sha256_file(corroboration_path),
        corroboration["artifact_sha256"],
        "corroboration artifact SHA256",
    )
    html = corroboration_path.read_text(encoding="utf-8")
    if any(
        fragment not in html
        for fragment in corroboration["exact_required_fragments"]
    ):
        raise IdentityNormalizationPolicyError(
            "corroboration artifact lacks an exact official fragment"
        )

    structured = _load_json(root / bindings["structured_path"])
    pages = structured.get("pages")
    if not isinstance(pages, list) or not pages:
        raise IdentityNormalizationPolicyError("structured source has no pages")
    page = pages[0]
    if not isinstance(page, dict) or page.get("page") != 1:
        raise IdentityNormalizationPolicyError(
            "structured first page is invalid"
        )
    text = page.get("text")
    if not isinstance(text, str) or "QUYẾT ĐỊNH" not in text.upper():
        raise IdentityNormalizationPolicyError(
            "current-document decision context is absent"
        )
    matches = re.findall(
        r"(?im)^\s*Số:\s*(\d{1,4}(?:\s+|/)(?:19|20)\d{2}/"
        r"[A-ZĐ0-9-]+)\s*$",
        text,
    )
    if matches != [EXPECTED_RAW_VALUE]:
        raise IdentityNormalizationPolicyError(
            "current-document heading is absent or competing"
        )

    blocked = root / "backend/data/runs" / EXPECTED_BLOCKED_RUN_ID
    run_manifest = _load_json(blocked / "run_manifest.json")
    _require_equal(
        run_manifest.get("canonical_run_fingerprint"),
        EXPECTED_BLOCKED_RUN_FINGERPRINT,
        "blocked run fingerprint",
    )
    protocol = _load_json(blocked / "identity_protocol.json")
    _require_equal(
        protocol.get("protocol_fingerprint"),
        EXPECTED_BLOCKED_PROTOCOL_FINGERPRINT,
        "blocked protocol fingerprint",
    )
    ledger = _load_json(blocked / "source_evidence_ledger.json")
    records = ledger.get("records")
    if not isinstance(records, list):
        raise IdentityNormalizationPolicyError("blocked ledger is invalid")
    source039 = [
        record
        for record in records
        if record.get("acquisition_id") == EXPECTED_ACQUISITION_ID
    ]
    if len(source039) != 1:
        raise IdentityNormalizationPolicyError(
            "blocked ledger must contain source039 exactly once"
        )
    _require_equal(
        source039[0].get("observed_judgment_header"),
        f"Số: {EXPECTED_RAW_VALUE}",
        "blocked raw observation",
    )


def _validate_protected_state(policy: Mapping[str, Any], root: Path) -> None:
    expected = policy["protected_state"]
    manifest = load_acquisition_manifest(root)
    statuses = Counter(
        source["acquisition_status"] for source in manifest["sources"]
    )
    _require_equal(len(manifest["sources"]), expected["global_total"], "total")
    for status in ("eligible", "extracted", "hash_verified"):
        _require_equal(
            statuses[status], expected[f"global_{status}"], f"global_{status}"
        )
    batch = [
        source
        for source in manifest["sources"]
        if source["acquisition_id"].startswith("batch2dc-20261105-")
    ]
    _require_equal(len(batch), expected["batch2dc_extracted"], "batch count")
    _require_equal(
        sum(source["acquisition_status"] == "extracted" for source in batch),
        expected["batch2dc_extracted"],
        "batch2dc_extracted",
    )
    _require_equal(
        sum(source["acquisition_status"] == "eligible" for source in batch),
        expected["batch2dc_eligible"],
        "batch2dc_eligible",
    )
    ready = sum(
        all(
            source["metadata_verification"][field]
            for field in REVIEW_METADATA_FIELDS
        )
        for source in batch
    )
    _require_equal(ready, expected["batch2dc_identity_ready"], "identity ready")

    benchmark = _load_json(
        root
        / "backend/data/benchmarks/retrieval_independent_v1/"
        "benchmark_manifest.json"
    )
    sampling = _load_json(
        root
        / "backend/data/benchmarks/retrieval_independent_v1/"
        "sampling_manifest.json"
    )
    _require_equal(benchmark.get("state"), expected["benchmark_state"], "benchmark")
    _require_equal(sampling.get("state"), expected["sampling_state"], "sampling")
    _require_equal(
        sampling.get("selection_status"),
        expected["selection_status"],
        "selection",
    )
    _require_equal(
        sampling.get("sampling_seed"), expected["sampling_seed"], "seed"
    )
    counts = benchmark.get("current_counts")
    if not isinstance(counts, dict):
        raise IdentityNormalizationPolicyError("benchmark counts are invalid")
    _require_equal(
        counts.get("selected_cases"), expected["selected_count"], "selected"
    )
    _require_equal(counts.get("query_records"), expected["query_count"], "queries")
    _require_equal(
        counts.get("ground_truth_records"),
        expected["ground_truth_count"],
        "ground truth",
    )


def validate_policy(policy: Mapping[str, Any], project_root: Path) -> dict:
    """Validate exact policy semantics, evidence bindings and protected state."""

    if not isinstance(policy, Mapping):
        raise IdentityNormalizationPolicyError("policy must be a mapping")
    root = Path(project_root).resolve()
    _validate_shape(policy)
    _validate_artifacts(policy, root)
    _validate_protected_state(policy, root)
    validate_source_identity_payload(
        {
            "identity_version": "independent_source_identity_v1",
            "verified_metadata": {
                "judgment_number": EXPECTED_CANONICAL_VALUE,
                "case_id": "26/2025/TLST-DS",
                "court": "Tòa án nhân dân Khu vực 2 - Nghệ An",
                "decision_date": "2026-07-31",
                "procedure_stage": "so_tham",
            },
            "verified_fields": {
                "judgment_number": True,
                "case_id": True,
                "court": True,
                "decision_date": True,
                "procedure_stage": True,
            },
            "related_decision_ids": [],
            "underlying_dispute_id": None,
        }
    )
    return deepcopy(dict(policy))


def normalize_allowed_identity_value(
    policy: Mapping[str, Any],
    acquisition_id: str,
    field: str,
    raw_value: str,
) -> str:
    """Purely normalize the exact frozen tuple; reject every other input."""

    _validate_shape(policy)
    scope = policy["scope"]
    if acquisition_id != scope["acquisition_id"]:
        raise IdentityNormalizationPolicyError(
            "acquisition_id is not allowlisted"
        )
    if field != scope["field"]:
        raise IdentityNormalizationPolicyError("field is not allowlisted")
    if raw_value != scope["raw_observed_value"]:
        raise IdentityNormalizationPolicyError(
            "raw value is not exactly allowlisted"
        )
    if raw_value.count(" ") != 1:
        raise IdentityNormalizationPolicyError(
            "raw value whitespace is not exact"
        )
    match = MALFORMED_SHAPE.fullmatch(raw_value)
    if match is None:
        raise IdentityNormalizationPolicyError("raw value shape is not allowed")
    canonical = f"{match.group(1)}/{match.group(2)}"
    if canonical != scope["canonical_identity_value"]:
        raise IdentityNormalizationPolicyError(
            "normalization exceeds frozen edit"
        )
    return canonical


def load_and_validate_policy(
    project_root: Path, policy_path: Path | None = None
) -> dict:
    root = Path(project_root).resolve()
    path = (
        Path(policy_path)
        if policy_path is not None
        else root / DEFAULT_POLICY_RELATIVE
    )
    return validate_policy(load_policy(path), root)
