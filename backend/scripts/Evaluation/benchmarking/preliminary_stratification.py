"""Label-blind preliminary metadata for independent-benchmark planning.

The output of this module is documentary planning metadata, not legal ground
truth.  It uses no retrieval output, model inference, generated query, or
annotation label.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path

from .independent_benchmark import (
    BENCHMARK_ID,
    canonical_json_fingerprint,
    inventory_independent_sources,
    validate_sampling_records,
)
from .source_intake import load_acquisition_manifest


ARTIFACT_VERSION = "preliminary_stratification_v1"
DERIVATION_VERSION = "documentary_rules_v1"
DEFAULT_ARTIFACT_RELATIVE = Path(
    "backend/data/benchmarks/retrieval_independent_v1/"
    "preliminary_stratification.json"
)
PRELIMINARY_FAMILIES = (
    "BLDS_CORE",
    "BLTTDS_PRIMARY",
    "LAND_2013",
    "LAND_2024",
    "HNGD_2014",
    "MULTI_FAMILY",
    "OTHER",
    "UNKNOWN",
)
TARGET_FAMILIES = (
    "BLDS_CORE",
    "BLTTDS_PRIMARY",
    "LAND_2013",
    "LAND_2024",
    "HNGD_2014",
)

RULE_DEFINITIONS = (
    {
        "rule_id": "PF-BLDS-01",
        "output": "BLDS_CORE",
        "evidence": (
            "Explicit 'Bộ luật Dân sự' citation plus a civil/commercial "
            "judgment code that is not a documentary decision code."
        ),
        "evidence_class": "explicit_law_name+documentary_case_type",
    },
    {
        "rule_id": "PF-BLTTDS-01",
        "output": "BLTTDS_PRIMARY",
        "evidence": (
            "Explicit 'Bộ luật Tố tụng dân sự' citation plus a documentary "
            "decision code; ordinary judgments with incidental procedural "
            "citations do not satisfy this rule."
        ),
        "evidence_class": "explicit_law_name+documentary_case_type",
    },
    {
        "rule_id": "PF-LAND-2013-01",
        "output": "LAND_2013",
        "evidence": "Explicit Land Law reference documentary-linked to 2013.",
        "evidence_class": "explicit_law_name+explicit_version",
    },
    {
        "rule_id": "PF-LAND-2024-01",
        "output": "LAND_2024",
        "evidence": "Explicit Land Law reference documentary-linked to 2024.",
        "evidence_class": "explicit_law_name+explicit_version",
    },
    {
        "rule_id": "PF-LAND-UNRESOLVED-01",
        "output": "UNKNOWN",
        "evidence": (
            "Land Law is explicit but no defensible 2013/2024 version is "
            "documentary-linked; decision year is never used as a substitute."
        ),
        "evidence_class": "ambiguous_version",
    },
    {
        "rule_id": "PF-HNGD-01",
        "output": "HNGD_2014",
        "evidence": "Authoritative HNGĐ/HN documentary judgment code.",
        "evidence_class": "documentary_case_type",
    },
    {
        "rule_id": "PF-OTHER-01",
        "output": "OTHER",
        "evidence": "Authoritative non-target HC/HS documentary judgment code.",
        "evidence_class": "documentary_case_type",
    },
    {
        "rule_id": "PF-MULTI-01",
        "output": "MULTI_FAMILY",
        "evidence": "Two or more target-family rules are independently satisfied.",
        "evidence_class": "multiple_documentary_rules",
    },
    {
        "rule_id": "PF-UNKNOWN-01",
        "output": "UNKNOWN",
        "evidence": "No rule has sufficient unambiguous documentary evidence.",
        "evidence_class": "insufficient_evidence",
    },
)


class PreliminaryStratificationError(ValueError):
    """Raised when planning metadata cannot be derived fail-closed."""


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFC", str(value or "")).casefold()
    return " ".join(value.split())


def _ascii_code(value: str) -> str:
    value = str(value or "").upper().replace("Đ", "D")
    return unicodedata.normalize("NFD", value).encode("ascii", "ignore").decode()


def documentary_case_type(judgment_number: str) -> str:
    tokens = set(re.findall(r"[A-Z0-9]+", _ascii_code(judgment_number)))
    if "HNGD" in tokens or "HN" in tokens:
        return "MARRIAGE_FAMILY"
    if "KDTM" in tokens:
        return "COMMERCIAL_BUSINESS"
    if tokens & {"DS", "VDS", "TCDS", "DSPT", "DSST", "QDDS"}:
        return "CIVIL"
    if tokens & {"HC", "HS"}:
        return "OTHER_SUPPORTED"
    return "UNKNOWN"


def _documentary_decision_code(judgment_number: str) -> bool:
    tokens = re.findall(r"[A-Z0-9]+", _ascii_code(judgment_number))
    return any(token.startswith("QD") for token in tokens)


def _pages_with_phrase(pages: list[dict], phrase: str) -> list[int]:
    normalized_phrase = _normalized_text(phrase)
    return [
        int(page["page"])
        for page in pages
        if normalized_phrase in _normalized_text(page.get("text", ""))
    ]


def _land_version_pages(pages: list[dict], year: int) -> list[int]:
    matches = []
    for page in pages:
        text = _normalized_text(page.get("text", ""))
        law_positions = [match.start() for match in re.finditer("luật đất đai", text)]
        year_positions = [match.start() for match in re.finditer(str(year), text)]
        if any(abs(left - right) <= 120 for left in law_positions for right in year_positions):
            matches.append(int(page["page"]))
    return matches


def _article_law_pages(pages: list[dict], law_name: str) -> list[int]:
    matches = []
    law = re.escape(_normalized_text(law_name))
    pattern = re.compile(
        rf"(?:điều\s+\d+.{{0,120}}{law}|{law}.{{0,120}}điều\s+\d+)"
    )
    for page in pages:
        if pattern.search(_normalized_text(page.get("text", ""))):
            matches.append(int(page["page"]))
    return matches


def _evidence(
    rule_id: str,
    evidence_class: str,
    field: str,
    canonical_value: str,
    page_numbers: list[int] | None = None,
) -> dict:
    result = {
        "rule_id": rule_id,
        "evidence_class": evidence_class,
        "field": field,
        "canonical_value": canonical_value,
    }
    if page_numbers is not None:
        result["page_numbers"] = sorted(set(page_numbers))
    return result


def _quality_metadata(structured: dict, procedure_stage: str) -> tuple[list[str], list[str]]:
    pages = structured["pages"]
    statuses = [page.get("extraction_status") for page in pages]
    ocr_pages = sum(status == "ocr_text" for status in statuses)
    blank_pages = sum(status == "empty_after_native_and_ocr" for status in statuses)
    unknown_pages = sum(status is None for status in statuses)
    page_count = len(pages)
    ocr_share = None if unknown_pages else ocr_pages / page_count
    blank_share = blank_pages / page_count

    quality_flags = []
    if unknown_pages:
        quality_flags.append("LEGACY_PAGE_STATUS_UNKNOWN")
    if ocr_share == 1:
        quality_flags.append("FULL_OCR")
    elif ocr_share is not None and ocr_share >= 0.75:
        quality_flags.append("HIGH_OCR_SHARE")
    if blank_share >= 0.25:
        quality_flags.append("HIGH_BLANK_PAGE_SHARE")

    stress = []
    if procedure_stage in {"giam_doc_tham", "tai_tham"}:
        stress.append("RARE_PROCEDURE_STAGE")
    if ocr_share is not None and ocr_share >= 0.75:
        stress.append("OCR_HEAVY")
    if page_count <= 3:
        stress.append("SHORT_DOCUMENT")
    if page_count > 20:
        stress.append("LONG_DOCUMENT")
    if blank_share >= 0.25:
        stress.append("BLANK_PAGE_HEAVY")
    return sorted(stress), sorted(quality_flags)


def classify_preliminary_source(source: dict, structured: dict) -> dict:
    """Classify one eligible source using only deterministic documentary evidence."""

    pages = structured["pages"]
    judgment_number = source["judgment_number"]
    doc_type = documentary_case_type(judgment_number)
    evidence = []
    candidates = set()
    ambiguity_flags = []

    blds_pages = _pages_with_phrase(pages, "Bộ luật Dân sự")
    blttds_pages = _pages_with_phrase(pages, "Bộ luật Tố tụng dân sự")
    land_pages = _pages_with_phrase(pages, "Luật Đất đai")
    land_2013_pages = _land_version_pages(pages, 2013)
    land_2024_pages = _land_version_pages(pages, 2024)
    hngd_pages = _pages_with_phrase(pages, "Luật Hôn nhân và gia đình")

    if blds_pages:
        evidence.append(
            _evidence(
                "EVIDENCE-BLDS-NAME",
                "explicit_law_name",
                "structured.pages[].text",
                "Bộ luật Dân sự",
                blds_pages,
            )
        )
        article_pages = _article_law_pages(pages, "Bộ luật Dân sự")
        if article_pages:
            evidence.append(
                _evidence(
                    "EVIDENCE-BLDS-ARTICLE",
                    "article_law_mapping",
                    "structured.pages[].text",
                    "Điều … Bộ luật Dân sự",
                    article_pages,
                )
            )
        if doc_type in {"CIVIL", "COMMERCIAL_BUSINESS"} and not (
            _documentary_decision_code(judgment_number)
        ):
            candidates.add("BLDS_CORE")
            evidence.append(
                _evidence(
                    "PF-BLDS-01",
                    "explicit_law_name+documentary_case_type",
                    "judgment_number",
                    doc_type,
                )
            )

    if blttds_pages:
        evidence.append(
            _evidence(
                "EVIDENCE-BLTTDS-NAME",
                "explicit_law_name",
                "structured.pages[].text",
                "Bộ luật Tố tụng dân sự",
                blttds_pages,
            )
        )
        article_pages = _article_law_pages(pages, "Bộ luật Tố tụng dân sự")
        if article_pages:
            evidence.append(
                _evidence(
                    "EVIDENCE-BLTTDS-ARTICLE",
                    "article_law_mapping",
                    "structured.pages[].text",
                    "Điều … Bộ luật Tố tụng dân sự",
                    article_pages,
                )
            )
        if _documentary_decision_code(judgment_number):
            candidates.add("BLTTDS_PRIMARY")
            evidence.append(
                _evidence(
                    "PF-BLTTDS-01",
                    "explicit_law_name+documentary_case_type",
                    "judgment_number",
                    "DOCUMENTARY_DECISION_CODE",
                )
            )

    if land_pages:
        evidence.append(
            _evidence(
                "EVIDENCE-LAND-NAME",
                "explicit_law_name",
                "structured.pages[].text",
                "Luật Đất đai",
                land_pages,
            )
        )
        if land_2013_pages:
            candidates.add("LAND_2013")
            evidence.append(
                _evidence(
                    "PF-LAND-2013-01",
                    "explicit_law_name+explicit_version",
                    "structured.pages[].text",
                    "Luật Đất đai … 2013",
                    land_2013_pages,
                )
            )
        if land_2024_pages:
            candidates.add("LAND_2024")
            evidence.append(
                _evidence(
                    "PF-LAND-2024-01",
                    "explicit_law_name+explicit_version",
                    "structured.pages[].text",
                    "Luật Đất đai … 2024",
                    land_2024_pages,
                )
            )
        if not land_2013_pages and not land_2024_pages:
            ambiguity_flags.append("LAND_VERSION_UNRESOLVED")
            evidence.append(
                _evidence(
                    "PF-LAND-UNRESOLVED-01",
                    "ambiguous_version",
                    "structured.pages[].text",
                    "Luật Đất đai without documentary-linked version",
                    land_pages,
                )
            )

    if doc_type == "MARRIAGE_FAMILY":
        candidates.add("HNGD_2014")
        evidence.append(
            _evidence(
                "PF-HNGD-01",
                "documentary_case_type",
                "judgment_number",
                "HNGĐ/HN",
            )
        )
    if hngd_pages:
        evidence.append(
            _evidence(
                "EVIDENCE-HNGD-NAME",
                "explicit_law_name",
                "structured.pages[].text",
                "Luật Hôn nhân và gia đình",
                hngd_pages,
            )
        )

    if doc_type == "OTHER_SUPPORTED" and not candidates:
        primary = "OTHER"
        status = "CLASSIFIED"
        rule_ids = ["PF-OTHER-01"]
        evidence.append(
            _evidence(
                "PF-OTHER-01",
                "documentary_case_type",
                "judgment_number",
                "HC/HS",
            )
        )
        secondary = []
    elif ambiguity_flags:
        primary = "UNKNOWN"
        status = "UNKNOWN"
        rule_ids = sorted(
            {item["rule_id"] for item in evidence if item["rule_id"].startswith("PF-")}
            | {"PF-UNKNOWN-01"}
        )
        secondary = sorted(candidates)
    elif len(candidates) > 1:
        primary = "MULTI_FAMILY"
        status = "MULTI_FAMILY"
        rule_ids = sorted(
            {item["rule_id"] for item in evidence if item["rule_id"].startswith("PF-")}
            | {"PF-MULTI-01"}
        )
        secondary = sorted(candidates)
    elif len(candidates) == 1:
        primary = next(iter(candidates))
        status = "CLASSIFIED"
        rule_ids = sorted(
            {item["rule_id"] for item in evidence if item["rule_id"].startswith("PF-")}
        )
        secondary = []
    else:
        primary = "UNKNOWN"
        status = "UNKNOWN"
        rule_ids = ["PF-UNKNOWN-01"]
        secondary = []

    stress, quality = _quality_metadata(structured, source["procedure_stage"])
    evidence.sort(
        key=lambda item: (
            item["rule_id"],
            item["evidence_class"],
            item["canonical_value"],
        )
    )
    return {
        "acquisition_id": source["acquisition_id"],
        "source_id": source["source_id"],
        "case_id": source["case_id"],
        "judgment_number": judgment_number,
        "primary_family_preliminary": primary,
        "secondary_families_preliminary": secondary,
        "classification_status": status,
        "rule_ids": rule_ids,
        "evidence": evidence,
        "ambiguity_flags": sorted(ambiguity_flags),
        "procedure_stage": source["procedure_stage"],
        "documentary_case_type": doc_type,
        "decision_year": int(source["decision_date"][:4]),
        "stress_descriptors": stress,
        "quality_watchlist_flags": quality,
    }


def _quota_observability(records: list[dict]) -> dict:
    targets = {
        "BLDS_CORE": {"minimum": 55, "maximum": 65},
        "BLTTDS_PRIMARY": {"minimum": 30, "maximum": 40},
        "LAND_2013": {"minimum": 35, "maximum": 45},
        "LAND_2024": {"minimum": 35, "maximum": 45},
        "HNGD_2014": {"minimum": 30, "maximum": 40},
    }
    unknown = sum(record["classification_status"] == "UNKNOWN" for record in records)
    for family, target in targets.items():
        classified = sum(
            record["primary_family_preliminary"] == family for record in records
        )
        multi = sum(
            record["classification_status"] == "MULTI_FAMILY"
            and family in record["secondary_families_preliminary"]
            for record in records
        )
        if classified >= target["minimum"]:
            status = "PASS"
        elif classified + multi >= target["minimum"]:
            status = "TIGHT"
        else:
            status = "FAIL"
        target.update(
            {
                "classified_available": classified,
                "multi_family_potential": multi,
                "unknown_global": unknown,
                "status": status,
                "multi_family_counted_toward_minimum": False,
            }
        )
    return {
        "families": targets,
        "main_guardrails": {
            "BLTTDS_only_procedural_max_ratio": 0.20,
            "substantive_core_min_ratio": 0.65,
            "status": "NOT_ASSESSABLE_BEFORE_SELECTION_AND_ANNOTATION",
        },
    }


def _status_distribution(records: list[dict]) -> dict:
    return {
        status: sum(record["classification_status"] == status for record in records)
        for status in ("CLASSIFIED", "MULTI_FAMILY", "UNKNOWN")
    }


def _family_distribution(records: list[dict]) -> dict:
    return {
        family: sum(record["primary_family_preliminary"] == family for record in records)
        for family in PRELIMINARY_FAMILIES
    }


def validate_checkpoint_frame_against_live(
    checkpoint_records: list[dict],
    live_eligible_records: list[dict],
) -> None:
    """Require an exact, unchanged checkpoint subset of the live eligible pool."""

    validate_sampling_records(checkpoint_records)
    validate_sampling_records(live_eligible_records)
    if any(
        not record["source_id"].startswith("independent:")
        for record in checkpoint_records
    ):
        raise PreliminaryStratificationError(
            "Development source contaminated the independent checkpoint frame"
        )
    live_by_source = {
        record["source_id"]: record for record in live_eligible_records
    }
    for checkpoint in checkpoint_records:
        source_id = checkpoint["source_id"]
        live = live_by_source.get(source_id)
        if live is None:
            raise PreliminaryStratificationError(
                f"Checkpoint source is no longer live eligible: {source_id}"
            )
        if live != checkpoint:
            raise PreliminaryStratificationError(
                f"Checkpoint source changed in live production: {source_id}"
            )


def classify_preliminary_records(
    source_records: list[dict],
    structured_by_source_id: dict[str, dict],
) -> list[dict]:
    """Classify an explicit eligible collection without loading any inventory."""

    if not isinstance(source_records, list):
        raise PreliminaryStratificationError("Explicit source records must be a list")
    source_ids = [record.get("source_id") for record in source_records]
    if any(not isinstance(source_id, str) or not source_id for source_id in source_ids):
        raise PreliminaryStratificationError("Explicit source_id is required")
    if len(source_ids) != len(set(source_ids)):
        raise PreliminaryStratificationError("Duplicate explicit source_id")
    if any(not source_id.startswith("independent:") for source_id in source_ids):
        raise PreliminaryStratificationError(
            "Development source cannot enter preliminary stratification"
        )
    if any(record.get("acquisition_status") != "eligible" for record in source_records):
        raise PreliminaryStratificationError(
            "Explicit preliminary classification requires eligible sources"
        )
    if set(structured_by_source_id) != set(source_ids):
        raise PreliminaryStratificationError(
            "Explicit structured-source mapping does not match source records"
        )
    records = [
        classify_preliminary_source(
            source,
            structured_by_source_id[source["source_id"]],
        )
        for source in source_records
    ]
    records.sort(key=lambda item: (item["case_id"], item["source_id"]))
    return records


def assemble_preliminary_stratification(
    checkpoint_frame: list[dict],
    classified_records: list[dict],
) -> dict:
    """Assemble one deterministic artifact for an explicitly supplied frame."""

    validate_sampling_records(checkpoint_frame)
    status_distribution = _status_distribution(classified_records)
    artifact = {
        "artifact_version": ARTIFACT_VERSION,
        "derivation_version": DERIVATION_VERSION,
        "benchmark_id": BENCHMARK_ID,
        "artifact_role": "PRELIMINARY_STRATIFICATION_METADATA_NOT_GROUND_TRUTH",
        "source_inventory_fingerprint": canonical_json_fingerprint(checkpoint_frame),
        "source_count": len(classified_records),
        "scope": {
            "eligible_independent_only": True,
            "development_sources_included": False,
            "blocked_sources_included": False,
            "hash_verified_sources_included": False,
        },
        "rules": list(RULE_DEFINITIONS),
        "family_distribution": _family_distribution(classified_records),
        "classification_status_distribution": status_distribution,
        "classification_coverage": (
            (status_distribution["CLASSIFIED"] + status_distribution["MULTI_FAMILY"])
            / len(classified_records)
            if classified_records
            else 0.0
        ),
        "quota_observability": _quota_observability(classified_records),
        "records_fingerprint": canonical_json_fingerprint(classified_records),
        "records": classified_records,
    }
    validate_preliminary_stratification(artifact, checkpoint_frame)
    return artifact


def validate_preliminary_stratification(
    artifact: dict,
    checkpoint_frame: list[dict] | None = None,
) -> None:
    if artifact.get("artifact_version") != ARTIFACT_VERSION:
        raise PreliminaryStratificationError("Unknown stratification artifact version")
    if artifact.get("derivation_version") != DERIVATION_VERSION:
        raise PreliminaryStratificationError("Unknown stratification derivation version")
    if artifact.get("benchmark_id") != BENCHMARK_ID:
        raise PreliminaryStratificationError("Wrong stratification benchmark")
    if artifact.get("artifact_role") != (
        "PRELIMINARY_STRATIFICATION_METADATA_NOT_GROUND_TRUTH"
    ):
        raise PreliminaryStratificationError("Wrong stratification artifact role")
    records = artifact.get("records")
    if not isinstance(records, list) or len(records) != artifact.get("source_count"):
        raise PreliminaryStratificationError("Stratification source count mismatch")
    if records != sorted(records, key=lambda item: (item["case_id"], item["source_id"])):
        raise PreliminaryStratificationError("Stratification records are not deterministic")
    if len({record["source_id"] for record in records}) != len(records):
        raise PreliminaryStratificationError("Duplicate stratification source_id")
    if len({record["case_id"] for record in records}) != len(records):
        raise PreliminaryStratificationError("Duplicate stratification case_id")
    for record in records:
        if record["primary_family_preliminary"] not in PRELIMINARY_FAMILIES:
            raise PreliminaryStratificationError("Unknown preliminary family")
        if record["classification_status"] not in {
            "CLASSIFIED",
            "MULTI_FAMILY",
            "UNKNOWN",
        }:
            raise PreliminaryStratificationError("Unknown classification status")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id.startswith("independent:"):
            raise PreliminaryStratificationError(
                "Development source contaminated preliminary stratification"
            )
    expected = canonical_json_fingerprint(records)
    if artifact.get("records_fingerprint") != expected:
        raise PreliminaryStratificationError("Stratification fingerprint mismatch")

    expected_status = _status_distribution(records)
    if artifact.get("classification_status_distribution") != expected_status:
        raise PreliminaryStratificationError(
            "Stratification classification summary mismatch"
        )
    if sum(expected_status.values()) != len(records):
        raise PreliminaryStratificationError(
            "Stratification classification statuses do not cover the frame"
        )
    expected_coverage = (
        (expected_status["CLASSIFIED"] + expected_status["MULTI_FAMILY"])
        / len(records)
        if records
        else 0.0
    )
    if artifact.get("classification_coverage") != expected_coverage:
        raise PreliminaryStratificationError("Stratification coverage mismatch")
    if artifact.get("family_distribution") != _family_distribution(records):
        raise PreliminaryStratificationError("Stratification family summary mismatch")
    if artifact.get("quota_observability") != _quota_observability(records):
        raise PreliminaryStratificationError("Stratification quota summary mismatch")

    if checkpoint_frame is not None:
        validate_sampling_records(checkpoint_frame)
        expected_frame_fingerprint = canonical_json_fingerprint(checkpoint_frame)
        if artifact.get("source_inventory_fingerprint") != expected_frame_fingerprint:
            raise PreliminaryStratificationError(
                "Stratification source inventory fingerprint mismatch"
            )
        frame_by_source = {
            record["source_id"]: record for record in checkpoint_frame
        }
        record_by_source = {record["source_id"]: record for record in records}
        if set(record_by_source) != set(frame_by_source):
            raise PreliminaryStratificationError(
                "Stratification source IDs do not match checkpoint frame"
            )
        for source_id, record in record_by_source.items():
            frame = frame_by_source[source_id]
            expected_identity = {
                "acquisition_id": frame.get("acquisition_id"),
                "case_id": frame.get("case_id"),
                "judgment_number": frame.get("judgment_number"),
                "procedure_stage": frame.get("procedure_stage"),
                "decision_year": frame.get("year"),
            }
            if any(record.get(key) != value for key, value in expected_identity.items()):
                raise PreliminaryStratificationError(
                    f"Stratification identity differs from checkpoint frame: {source_id}"
                )


def _load_sampling_checkpoint(project_root: Path) -> tuple[list[dict], str]:
    benchmark_dir = project_root / DEFAULT_ARTIFACT_RELATIVE.parent
    sampling = json.loads(
        (benchmark_dir / "sampling_manifest.json").read_text(encoding="utf-8")
    )
    checkpoint = sampling.get("eligible_sources")
    fingerprint = sampling.get("eligible_fingerprint")
    if not isinstance(checkpoint, list) or (
        canonical_json_fingerprint(checkpoint) != fingerprint
    ):
        raise PreliminaryStratificationError(
            "Reconciled sampling-frame fingerprint is invalid"
        )
    return checkpoint, fingerprint


def classify_live_preliminary_sources(
    project_root: Path,
    source_records: list[dict],
) -> list[dict]:
    """Classify an explicit live-eligible subset without changing checkpoints."""

    project_root = Path(project_root).resolve()
    live_eligible, _, _ = inventory_independent_sources(
        project_root, include_private_text=False
    )
    validate_checkpoint_frame_against_live(source_records, live_eligible)
    manifest = load_acquisition_manifest(project_root)
    manifest_by_source = {
        source["source_id"]: source for source in manifest["sources"]
    }
    sources = []
    structured_by_source = {}
    for frame_record in source_records:
        source_id = frame_record["source_id"]
        source = manifest_by_source.get(source_id)
        if source is None or source["acquisition_status"] != "eligible":
            raise PreliminaryStratificationError(
                f"Non-eligible source entered stratification: {source_id}"
            )
        sources.append(source)
        structured_by_source[source_id] = json.loads(
            (project_root / source["structured_path"]).read_text(encoding="utf-8")
        )
    return classify_preliminary_records(sources, structured_by_source)


def build_preliminary_stratification_for_frame(
    project_root: Path,
    checkpoint_frame: list[dict],
) -> dict:
    """Build from one explicit checkpoint frame; live eligible may be a superset."""

    project_root = Path(project_root).resolve()
    classified = classify_live_preliminary_sources(project_root, checkpoint_frame)
    return assemble_preliminary_stratification(checkpoint_frame, classified)


def build_preliminary_stratification(project_root: Path) -> dict:
    project_root = Path(project_root).resolve()
    checkpoint_frame, _ = _load_sampling_checkpoint(project_root)
    return build_preliminary_stratification_for_frame(
        project_root,
        checkpoint_frame,
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate label-blind preliminary benchmark stratification"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output or (root / DEFAULT_ARTIFACT_RELATIVE)
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise PreliminaryStratificationError(
            "Stratification output must remain inside the project root"
        ) from exc
    artifact = build_preliminary_stratification(root)
    _atomic_write_json(output, artifact)
    print(
        json.dumps(
            {
                "output": str(output.relative_to(root)),
                "source_count": artifact["source_count"],
                "records_fingerprint": artifact["records_fingerprint"],
                "classification_status_distribution": artifact[
                    "classification_status_distribution"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
