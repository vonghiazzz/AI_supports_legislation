"""Phase-1 infrastructure for a sealed independent retrieval benchmark.

This module inventories judgment sources, excludes development cases, validates
sampling metadata, and enforces benchmark manifest transitions. It deliberately
contains no retrieval, model inference, query generation, or GT generation.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..core.normalization import canonical_retrieval_id


BENCHMARK_ID = "retrieval_independent_v1"
SCHEMA_VERSION = "gt_schema_v2"
CORPUS_FINGERPRINT = (
    "66fb0b20e0505993ed616f864ece99ecfad801de98b4ba3a37d20a3a7bf55e58"
)
TARGET_TOTAL = 250
TARGET_MAIN = 200
TARGET_STRESS = 50
MINIMUM_TOTAL = 180
MINIMUM_MAIN = 150
NEAR_DUPLICATE_THRESHOLD = 0.72

MANIFEST_STATES = (
    "DRAFT",
    "SAMPLING_FROZEN",
    "ANNOTATION_IN_PROGRESS",
    "ANNOTATION_FROZEN",
    "SEALED",
    "OPENED_FOR_CONFIRMATORY_EVALUATION",
)

STATE_TRANSITIONS = {
    "DRAFT": "SAMPLING_FROZEN",
    "SAMPLING_FROZEN": "ANNOTATION_IN_PROGRESS",
    "ANNOTATION_IN_PROGRESS": "ANNOTATION_FROZEN",
    "ANNOTATION_FROZEN": "SEALED",
    "SEALED": "OPENED_FOR_CONFIRMATORY_EVALUATION",
}

FORBIDDEN_SAMPLING_KEY_TOKENS = {
    "bm25",
    "candidate",
    "dense",
    "hybrid",
    "metric",
    "rank",
    "rerank",
    "retrieval",
    "score",
}

SCHEMA_V2_REQUIRED_GT_FIELDS = {
    "law_code",
    "article",
    "reasoning",
    "importance",
    "importance_score",
    "source_type",
    "domain",
}


class BenchmarkProtocolError(ValueError):
    """Raised when independent-benchmark protocol invariants are violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_fingerprint(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_sha256(value, field_name: str) -> None:
    if not _is_sha256(value):
        raise BenchmarkProtocolError(
            f"{field_name} must be exactly 64 lowercase hexadecimal characters"
        )


def selected_records_fingerprint(records: list[dict]) -> str:
    ordered = sorted(
        (_public_record(record) for record in records),
        key=lambda record: (record["case_id"], record["source_id"]),
    )
    return canonical_json_fingerprint(ordered)


def _relative(path: Path, project_root: Path) -> str:
    return Path(path).resolve().relative_to(project_root.resolve()).as_posix()


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value or "")).lower()
    normalized = re.sub(r"[^0-9a-zà-ỹđ]+", " ", normalized)
    return " ".join(normalized.split())


def canonical_source_name(value: str) -> str:
    return unicodedata.normalize("NFC", Path(str(value)).name).casefold()


def _dedup_shingles(text: str, size: int = 5) -> set[tuple[str, ...]]:
    tokens = _normalize_text(text).split()
    if len(tokens) < size:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)}


def dedup_text_similarity(left: str, right: str) -> float:
    left_shingles = _dedup_shingles(left)
    right_shingles = _dedup_shingles(right)
    union = left_shingles | right_shingles
    if not union:
        return 1.0
    return len(left_shingles & right_shingles) / len(union)


def _structured_text(payload: dict) -> tuple[str, str]:
    pages = payload.get("pages", [])
    texts = [str(page.get("text", "")) for page in pages if isinstance(page, dict)]
    return (texts[0] if texts else "", "\n".join(texts))


def _extract_case_id(first_page: str, filename: str) -> str:
    match = re.search(
        r"s[ốôo]\s*:?\s*(\d{1,4}\s*/\s*(?:19|20)\d{2}\s*/\s*[A-ZĐa-zđ\-]+)",
        first_page,
        flags=re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", "", match.group(1)).upper()
    stem = Path(filename).stem
    year_match = re.search(r"(?:19|20)\d{2}", first_page)
    if ":" in stem:
        number, year = stem.split(":", 1)
        return f"{number}/{year}".upper()
    if year_match:
        return f"{stem}/{year_match.group(0)}".upper()
    return stem.upper()


def _extract_year(case_id: str, first_page: str) -> int | None:
    match = re.search(r"(?:19|20)\d{2}", case_id)
    if not match:
        match = re.search(r"ngày\s+\d{1,2}\D+\d{1,2}\D+((?:19|20)\d{2})", first_page, re.I)
    return int(match.group(0 if len(match.groups()) == 0 else 1)) if match else None


def _extract_procedure_stage(first_page: str) -> str | None:
    normalized = _normalize_text(first_page)
    if re.search(r"tái th[ẩâ]m", normalized):
        return "tai_tham"
    if re.search(r"giám đốc th[ẩâ]m", normalized):
        return "giam_doc_tham"
    if "phúc thẩm" in normalized:
        return "phuc_tham"
    if "sơ thẩm" in normalized:
        return "so_tham"
    return None


def _extract_court(first_page: str) -> str | None:
    normalized = _normalize_text(first_page)
    if re.search(r"tòa án nhân dân (cấp|cáp) cao", normalized):
        return "toa_an_cap_cao"
    if "tòa án nhân dân tối cao" in normalized:
        return "toa_an_nhan_dan_toi_cao"
    if "tòa án nhân dân" in normalized:
        return "toa_an_nhan_dan"
    return None


def _extract_case_type(first_page: str) -> str | None:
    lines = [" ".join(line.split()) for line in first_page.splitlines()]
    start = None
    for index, line in enumerate(lines):
        compact = _normalize_text(line)
        if re.match(r"^(v v|vw|về việc)\b", compact):
            start = index
            break
    if start is None:
        return None
    collected = []
    for line in lines[start : start + 5]:
        if re.search(r"nhân\s+danh", _normalize_text(line)):
            break
        collected.append(line)
    value = " ".join(collected)
    value = re.sub(r"^(V\s*/?\s*v|VW|Về việc)\s*:? ?", "", value, flags=re.I)
    value = value.strip(" “\"”':.-_")
    return value or None


def _preliminary_law_families(full_text: str) -> list[str]:
    normalized = _normalize_text(full_text)
    families = []
    if "bộ luật dân sự" in normalized:
        families.append("BLDS")
    if "bộ luật tố tụng dân sự" in normalized:
        families.append("BLTTDS")
    if "luật đất đai" in normalized:
        if re.search(r"luật đất đai.{0,40}2013|2013.{0,40}luật đất đai", normalized):
            families.append("LAND_2013")
        if re.search(r"luật đất đai.{0,40}2024|2024.{0,40}luật đất đai", normalized):
            families.append("LAND_2024")
        if not any(value.startswith("LAND_") for value in families):
            families.append("LAND_UNSPECIFIED")
    if "luật hôn nhân và gia đình" in normalized:
        families.append("HNGD_2014")
    return sorted(set(families))


def _preliminary_legal_functions(families: list[str]) -> list[str]:
    functions = []
    if "BLTTDS" in families:
        functions.append("procedural")
    if any(value != "BLTTDS" for value in families):
        functions.append("substantive")
    return functions or ["unknown"]


def inventory_judgment_sources(project_root: Path, include_private_text: bool = False) -> list[dict]:
    project_root = Path(project_root).resolve()
    source_dir = project_root / "backend/dataLaw/ban_an"
    structured_dir = project_root / "backend/data/structured_judgments"
    records = []
    for source_path in sorted(source_dir.glob("*.pdf"), key=lambda path: path.name):
        structured_path = structured_dir / f"{source_path.stem}_structured.json"
        structured_payload = {}
        first_page = ""
        full_text = ""
        if structured_path.exists():
            structured_payload = json.loads(structured_path.read_text(encoding="utf-8"))
            first_page, full_text = _structured_text(structured_payload)
        families = _preliminary_law_families(full_text)
        case_id = _extract_case_id(first_page, source_path.name)
        record = {
            "source_id": f"judgment:{canonical_source_name(source_path.name)}",
            "case_id": case_id,
            "source_name": source_path.name,
            "source_path": _relative(source_path, project_root),
            "structured_path": _relative(structured_path, project_root) if structured_path.exists() else None,
            "court": _extract_court(first_page),
            "procedure_stage": _extract_procedure_stage(first_page),
            "year": _extract_year(case_id, first_page),
            "case_type": _extract_case_type(first_page),
            "preliminary_law_families": families,
            "preliminary_legal_functions": _preliminary_legal_functions(families),
            "source_sha256": sha256_file(source_path),
            "structured_sha256": sha256_file(structured_path) if structured_path.exists() else None,
            "metadata_source": "structured_judgment_text_regex_v1",
        }
        if include_private_text:
            record["_dedup_text"] = full_text
        records.append(record)
    validate_sampling_records(records)
    return records


def validate_sampling_records(records: list[dict]) -> None:
    if not isinstance(records, list):
        raise BenchmarkProtocolError("Sampling records must be a list")
    source_ids = []
    case_ids = []

    def inspect_keys(value, path="record"):
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).casefold()
                if not normalized.startswith("_") and any(
                    token in normalized for token in FORBIDDEN_SAMPLING_KEY_TOKENS
                ):
                    raise BenchmarkProtocolError(
                        f"Retrieval-output field is forbidden in sampling metadata: {path}.{key}"
                    )
                inspect_keys(nested, f"{path}.{key}")
        elif isinstance(value, list):
            for index, nested in enumerate(value):
                inspect_keys(nested, f"{path}[{index}]")

    for record in records:
        if not isinstance(record, dict):
            raise BenchmarkProtocolError("Each sampling record must be an object")
        inspect_keys(record)
        for required in ("source_id", "case_id", "source_path", "source_sha256"):
            if not record.get(required):
                raise BenchmarkProtocolError(f"Sampling record is missing {required}")
        source_ids.append(record["source_id"])
        case_ids.append(record["case_id"])
    if len(source_ids) != len(set(source_ids)):
        raise BenchmarkProtocolError("Duplicate source_id values are forbidden")
    if len(case_ids) != len(set(case_ids)):
        raise BenchmarkProtocolError("Duplicate case_id values are forbidden")


def _public_record(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def deduplicate_against_development(
    inventory: list[dict],
    development_records: list[dict],
    known_duplicate_links: dict[str, str] | None = None,
) -> tuple[list[dict], dict]:
    known_duplicate_links = known_duplicate_links or {}
    ordered_inventory = sorted(
        inventory,
        key=lambda item: (
            item.get("case_id", ""),
            item.get("source_id", ""),
            item.get("source_sha256", ""),
        ),
    )
    inventory_by_name = {
        canonical_source_name(record["source_name"]): record
        for record in ordered_inventory
    }
    consumed = []
    for development_record in development_records:
        name = canonical_source_name(development_record.get("judgment_name", ""))
        source = inventory_by_name.get(name)
        consumed.append({
            "development_query_id": development_record.get("id"),
            "judgment_name": development_record.get("judgment_name"),
            "source_id": source.get("source_id") if source else None,
            "case_id": source.get("case_id") if source else None,
            "source_sha256": source.get("source_sha256") if source else None,
            "match_basis": "exact_normalized_source_name" if source else "source_not_found",
        })

    consumed_names = {canonical_source_name(item["judgment_name"]) for item in consumed}
    consumed_source_ids = {item["source_id"] for item in consumed if item.get("source_id")}
    consumed_case_ids = {item["case_id"] for item in consumed if item.get("case_id")}
    consumed_hashes = {item["source_sha256"] for item in consumed if item.get("source_sha256")}
    consumed_sources = [record for record in inventory if record["source_id"] in consumed_source_ids]

    exclusions = []
    new_candidates = []
    for record in ordered_inventory:
        reasons = []
        if canonical_source_name(record["source_name"]) in consumed_names:
            reasons.append("development_benchmark_exact_source")
        if record["source_sha256"] in consumed_hashes:
            reasons.append("development_benchmark_exact_sha256")
        if record["case_id"] in consumed_case_ids:
            reasons.append("development_benchmark_same_case_id")
        linked_case = known_duplicate_links.get(record["source_id"])
        if linked_case:
            reasons.append(f"known_underlying_dispute:{linked_case}")
        if not reasons and record.get("_dedup_text"):
            for consumed_source in consumed_sources:
                similarity = dedup_text_similarity(
                    record["_dedup_text"], consumed_source.get("_dedup_text", "")
                )
                if similarity >= NEAR_DUPLICATE_THRESHOLD:
                    reasons.append(
                        "near_duplicate_source_text:"
                        f"{consumed_source['case_id']}:{similarity:.6f}"
                    )
                    break
        if reasons:
            exclusions.append({
                "source_id": record["source_id"],
                "case_id": record["case_id"],
                "reasons": sorted(set(reasons)),
            })
        else:
            new_candidates.append(record)

    parent = list(range(len(new_candidates)))
    edges: dict[tuple[int, int], list[str]] = {}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left_index, left in enumerate(new_candidates):
        for right_index in range(left_index + 1, len(new_candidates)):
            right = new_candidates[right_index]
            pair_reasons = []
            if (
                left.get("source_id") == right.get("source_id")
                or canonical_source_name(left.get("source_name", ""))
                == canonical_source_name(right.get("source_name", ""))
            ):
                pair_reasons.append("new_pool_duplicate_source_identity")
            if left.get("source_sha256") == right.get("source_sha256"):
                pair_reasons.append("new_pool_duplicate_sha256")
            if left.get("case_id") == right.get("case_id"):
                pair_reasons.append("new_pool_duplicate_case_id")
            if left.get("_dedup_text") and right.get("_dedup_text"):
                similarity = dedup_text_similarity(
                    left["_dedup_text"], right["_dedup_text"]
                )
                if similarity >= NEAR_DUPLICATE_THRESHOLD:
                    pair_reasons.append(
                        f"new_pool_near_duplicate_source_text:{similarity:.6f}"
                    )
            if pair_reasons:
                edges[(left_index, right_index)] = sorted(set(pair_reasons))
                union(left_index, right_index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(new_candidates)):
        components[find(index)].append(index)

    eligible = []
    internal_duplicate_clusters = []
    for members in sorted(components.values(), key=lambda values: values[0]):
        canonical_index = min(members)
        canonical = new_candidates[canonical_index]
        eligible.append(_public_record(canonical))
        if len(members) == 1:
            continue
        cluster_members = []
        for member_index in members:
            member = new_candidates[member_index]
            cluster_members.append(member["source_id"])
            if member_index == canonical_index:
                continue
            member_reasons = []
            for (left_index, right_index), pair_reasons in edges.items():
                if member_index in (left_index, right_index):
                    member_reasons.extend(pair_reasons)
            member_reasons.append(
                f"canonical_representative:{canonical['source_id']}"
            )
            exclusions.append({
                "source_id": member["source_id"],
                "case_id": member["case_id"],
                "reasons": sorted(set(member_reasons)),
            })
        internal_duplicate_clusters.append({
            "canonical_source_id": canonical["source_id"],
            "member_source_ids": sorted(cluster_members),
        })

    eligible.sort(key=lambda item: (item["case_id"], item["source_id"]))
    exclusions.sort(key=lambda item: (item["case_id"], item["source_id"]))

    reason_counts = Counter(reason for item in exclusions for reason in item["reasons"])
    report = {
        "algorithm": "source_name_sha_case_id_text_shingle_v2",
        "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
        "inventory_count": len(inventory),
        "development_record_count": len(development_records),
        "development_sources_matched": sum(item["source_id"] is not None for item in consumed),
        "development_sources_unmatched": sum(item["source_id"] is None for item in consumed),
        "excluded_source_count": len(exclusions),
        "eligible_source_count": len(eligible),
        "reason_counts": dict(sorted(reason_counts.items())),
        "development_source_mapping": consumed,
        "exclusions": exclusions,
        "internal_duplicate_clusters": internal_duplicate_clusters,
        "known_duplicate_links": dict(sorted(known_duplicate_links.items())),
        "ordering_rule": (
            "case_id ASC, source_id ASC, source_sha256 ASC; "
            "first duplicate-cluster member is canonical"
        ),
    }
    return eligible, report


def sampling_breakdown(records: list[dict]) -> dict:
    def count_scalar(key):
        counts = Counter(str(record.get(key) or "UNKNOWN") for record in records)
        return dict(sorted(counts.items()))

    def count_multi(key):
        return dict(sorted(Counter(value for record in records for value in record.get(key, [])).items()))

    return {
        "total": len(records),
        "by_law_family_preliminary": count_multi("preliminary_law_families"),
        "by_legal_function_preliminary": count_multi("preliminary_legal_functions"),
        "by_case_type": count_scalar("case_type"),
        "by_court": count_scalar("court"),
        "by_procedure_stage": count_scalar("procedure_stage"),
        "by_year": count_scalar("year"),
    }


def assess_quota_feasibility(records: list[dict]) -> dict:
    family_counts = Counter(value for record in records for value in record.get("preliminary_law_families", []))
    targets = {
        "BLDS_core_substantive": {"minimum": 55, "maximum": 65, "available_preliminary": family_counts["BLDS"]},
        "BLTTDS_primary_procedural": {"minimum": 30, "maximum": 40, "available_preliminary": family_counts["BLTTDS"]},
        "LAND_2013": {"minimum": 35, "maximum": 45, "available_preliminary": family_counts["LAND_2013"]},
        "LAND_2024": {"minimum": 35, "maximum": 45, "available_preliminary": family_counts["LAND_2024"]},
        "HNGD_2014": {"minimum": 30, "maximum": 40, "available_preliminary": family_counts["HNGD_2014"]},
    }
    for value in targets.values():
        value["preliminary_minimum_feasible"] = value["available_preliminary"] >= value["minimum"]
        value["requires_annotation_confirmation"] = True
    total_feasible = len(records) >= MINIMUM_TOTAL
    blockers = []
    if not total_feasible:
        blockers.append(
            f"eligible source count {len(records)} is below minimum total {MINIMUM_TOTAL}"
        )
    for name, value in targets.items():
        if not value["preliminary_minimum_feasible"]:
            blockers.append(
                f"{name} preliminary availability {value['available_preliminary']} "
                f"is below minimum {value['minimum']}"
            )
    return {
        "status": "PRELIMINARILY_FEASIBLE" if not blockers else "INFEASIBLE",
        "eligible_count": len(records),
        "target_total": TARGET_TOTAL,
        "minimum_total": MINIMUM_TOTAL,
        "targets": targets,
        "main_guardrails": {
            "BLTTDS_only_procedural_max_ratio": 0.20,
            "substantive_core_min_ratio": 0.65,
            "status": "NOT_ASSESSABLE_BEFORE_ANNOTATION",
        },
        "blockers": blockers,
    }


def validate_schema_v2(schema: dict) -> None:
    if schema.get("type") != "array":
        raise BenchmarkProtocolError("Schema V2 root must be an array")
    item = schema.get("items", {})
    gt = item.get("properties", {}).get("ground_truth", {})
    if gt.get("required") != ["core", "supporting"]:
        raise BenchmarkProtocolError("Schema must keep core/supporting partitions")
    if gt.get("additionalProperties") is not False:
        raise BenchmarkProtocolError("Ground-truth schema must fail closed")
    for partition, importance, score in (
        ("core", "decisive", 3),
        ("supporting", "supporting", 1),
    ):
        partition_schema = gt.get("properties", {}).get(partition, {}).get("items", {})
        required = set(partition_schema.get("required", []))
        if required != SCHEMA_V2_REQUIRED_GT_FIELDS:
            raise BenchmarkProtocolError(f"{partition} GT fields drifted")
        properties = partition_schema.get("properties", {})
        if properties.get("importance", {}).get("const") != importance:
            raise BenchmarkProtocolError(f"{partition} importance semantics drifted")
        if properties.get("importance_score", {}).get("const") != score:
            raise BenchmarkProtocolError(f"{partition} weight semantics drifted")
        if properties.get("domain", {}).get("enum") != ["substantive", "procedural"]:
            raise BenchmarkProtocolError(f"{partition} legal-function semantics drifted")


def validate_benchmark_records(records: list[dict], schema: dict) -> None:
    validate_schema_v2(schema)
    try:
        import jsonschema

        jsonschema.validate(instance=records, schema=schema)
    except ImportError as exc:
        raise BenchmarkProtocolError(
            "jsonschema is required to validate an independent benchmark"
        ) from exc
    except jsonschema.exceptions.ValidationError as exc:
        raise BenchmarkProtocolError(
            f"Independent benchmark schema validation failed: {exc.message}"
        ) from exc


def validate_unique_selected_cases(records: list[dict], development_case_ids: set[str]) -> None:
    validate_sampling_records(records)
    selected_case_ids = [record["case_id"] for record in records]
    overlap = sorted(set(selected_case_ids) & set(development_case_ids))
    if overlap:
        raise BenchmarkProtocolError(f"Selected cases overlap development cases: {overlap}")


def validate_selection_guardrails(
    records: list[dict],
    mode: str = "target",
    sampling_frame_limitation_reason: str | None = None,
) -> dict:
    validate_sampling_records(records)
    if mode not in {"target", "minimum"}:
        raise BenchmarkProtocolError(f"Unknown selection validation mode: {mode}")
    main = [record for record in records if record.get("slice") == "main"]
    stress = [record for record in records if record.get("slice") == "stress"]
    if len(main) + len(stress) != len(records):
        raise BenchmarkProtocolError("Every selected case must declare slice=main or stress")
    if mode == "target":
        if len(records) != TARGET_TOTAL or len(main) != TARGET_MAIN or len(stress) != TARGET_STRESS:
            raise BenchmarkProtocolError(
                "Target selection requires exactly 250 total / 200 main / 50 stress cases"
            )
    else:
        if not sampling_frame_limitation_reason:
            raise BenchmarkProtocolError(
                "Minimum-size mode requires an explicit sampling-frame limitation reason"
            )
        if len(records) < MINIMUM_TOTAL or len(main) < MINIMUM_MAIN:
            raise BenchmarkProtocolError(
                "Minimum selection requires at least 180 total and 150 main cases"
            )
    blttds_only = sum(bool(record.get("blttds_only_procedural")) for record in main)
    substantive_core = sum(bool(record.get("anticipated_substantive_core")) for record in main)
    blttds_ratio = blttds_only / len(main) if main else 0.0
    substantive_core_ratio = substantive_core / len(main) if main else 0.0
    if blttds_ratio > 0.20:
        raise BenchmarkProtocolError("BLTTDS-only/procedural main-set ratio exceeds 20%")
    if substantive_core_ratio < 0.65:
        raise BenchmarkProtocolError("Substantive-core main-set ratio is below 65%")
    return {
        "mode": mode,
        "total": len(records),
        "main": len(main),
        "stress": len(stress),
        "BLTTDS_only_procedural_ratio": blttds_ratio,
        "substantive_core_ratio": substantive_core_ratio,
        "sampling_frame_limitation_reason": sampling_frame_limitation_reason,
    }


def _validate_artifact_entry(entry: dict, collection_name: str) -> tuple[str, str]:
    if not isinstance(entry, dict):
        raise BenchmarkProtocolError(f"{collection_name} entries must be objects")
    path = entry.get("path")
    digest = entry.get("sha256")
    if not isinstance(path, str) or not path:
        raise BenchmarkProtocolError(f"{collection_name} artifact path is missing")
    _require_sha256(digest, f"{collection_name}[{path}].sha256")
    return path, digest


def _declared_artifact_map(manifest: dict) -> dict[str, str]:
    artifacts = {}

    def add(path: str, digest: str, source: str) -> None:
        existing = artifacts.get(path)
        if existing is not None and existing != digest:
            raise BenchmarkProtocolError(
                f"Conflicting hashes declared for artifact {path} ({source})"
            )
        artifacts[path] = digest

    for collection_name in ("phase1_artifacts", "sealed_artifacts"):
        collection = manifest.get(collection_name, [])
        if not isinstance(collection, list):
            raise BenchmarkProtocolError(f"{collection_name} must be a list")
        for entry in collection:
            path, digest = _validate_artifact_entry(entry, collection_name)
            add(path, digest, collection_name)

    required_pairs = [
        ("benchmark_path", "benchmark_sha256"),
        ("gt_schema_path", "gt_schema_sha256"),
        ("annotation_guidelines_path", "annotation_guidelines_sha256"),
        ("annotation_agreement_path", "annotation_agreement_sha256"),
    ]
    if manifest.get("adjudication_required") or any(
        manifest.get(field) for field in ("adjudication_path", "adjudication_sha256")
    ):
        required_pairs.append(("adjudication_path", "adjudication_sha256"))
    if manifest.get("state") == "OPENED_FOR_CONFIRMATORY_EVALUATION":
        required_pairs.append(("preregistration_path", "preregistration_sha256"))

    for path_field, digest_field in required_pairs:
        path = manifest.get(path_field)
        digest = manifest.get(digest_field)
        if not path or not digest:
            raise BenchmarkProtocolError(
                f"Sealed manifest is missing {path_field}/{digest_field}"
            )
        _require_sha256(digest, digest_field)
        add(path, digest, path_field)
    return artifacts


def _verify_artifact_map(artifacts: dict[str, str], project_root: Path) -> None:
    project_root = Path(project_root).resolve()
    for relative_path, expected in sorted(artifacts.items()):
        relative = Path(relative_path)
        if relative.is_absolute():
            raise BenchmarkProtocolError(
                f"Artifact path must be repository-relative: {relative_path}"
            )
        path = (project_root / relative).resolve()
        try:
            path.relative_to(project_root)
        except ValueError as exc:
            raise BenchmarkProtocolError(
                f"Artifact path escapes project root: {relative_path}"
            ) from exc
        if path.is_symlink() or not path.is_file():
            raise BenchmarkProtocolError(f"Sealed artifact is missing: {relative_path}")
        if sha256_file(path) != expected:
            raise BenchmarkProtocolError(f"Sealed artifact changed: {relative_path}")


def _phase1_artifact_hash(manifest: dict, filename: str) -> str:
    matches = []
    for entry in manifest.get("phase1_artifacts", []):
        path, digest = _validate_artifact_entry(entry, "phase1_artifacts")
        if Path(path).name == filename:
            matches.append(digest)
    if len(matches) != 1:
        raise BenchmarkProtocolError(
            f"phase1_artifacts must declare exactly one {filename}"
        )
    return matches[0]


def _verify_manifest_artifact_pairs(
    manifest: dict,
    pairs: tuple[tuple[str, str], ...],
    project_root: Path | None,
    target_state: str,
) -> None:
    if project_root is None:
        raise BenchmarkProtocolError(
            f"Transition to {target_state} requires project_root verification"
        )
    artifacts = {}
    for path_field, digest_field in pairs:
        path = manifest.get(path_field)
        digest = manifest.get(digest_field)
        if not path or not digest:
            raise BenchmarkProtocolError(
                f"Transition to {target_state} requires {path_field}/{digest_field}"
            )
        _require_sha256(digest, digest_field)
        artifacts[path] = digest
    _verify_artifact_map(artifacts, project_root)


def _validate_sampling_freeze(
    manifest: dict,
    selected_records: list[dict] | None,
    eligible_records: list[dict] | None,
    project_root: Path | None,
) -> None:
    if not selected_records:
        raise BenchmarkProtocolError("Sampling freeze requires selected records")
    if eligible_records is None:
        raise BenchmarkProtocolError("Sampling freeze requires the eligible source pool")
    validate_sampling_records(selected_records)
    validate_sampling_records(eligible_records)

    selected_hashes = [record["source_sha256"] for record in selected_records]
    if len(selected_hashes) != len(set(selected_hashes)):
        raise BenchmarkProtocolError("Selected records contain duplicate source SHA256 values")
    selected_source_names = [
        canonical_source_name(record.get("source_name", ""))
        for record in selected_records
    ]
    if len(selected_source_names) != len(set(selected_source_names)):
        raise BenchmarkProtocolError(
            "Selected records contain duplicate normalized source identities"
        )
    for index, digest in enumerate(selected_hashes):
        _require_sha256(digest, f"selected_records[{index}].source_sha256")

    selected_count = manifest.get("selected_case_count")
    if not isinstance(selected_count, int) or isinstance(selected_count, bool):
        raise BenchmarkProtocolError("selected_case_count must be an integer")
    if selected_count != len(selected_records):
        raise BenchmarkProtocolError(
            "selected_case_count must equal the number of selected records"
        )

    selection_mode = manifest.get("selection_mode")
    validate_selection_guardrails(
        selected_records,
        mode=selection_mode,
        sampling_frame_limitation_reason=manifest.get(
            "sampling_frame_limitation_reason"
        ),
    )

    eligible_by_source = {record["source_id"]: record for record in eligible_records}
    if len(selected_records) > len(eligible_records):
        raise BenchmarkProtocolError("Eligible source pool cannot support the selection")
    for selected in selected_records:
        eligible = eligible_by_source.get(selected["source_id"])
        if eligible is None or any(
            selected.get(field) != eligible.get(field)
            for field in ("case_id", "source_path", "source_sha256")
        ):
            raise BenchmarkProtocolError(
                f"Selected source is absent from the eligible pool: {selected['source_id']}"
            )

    current_counts = manifest.get("current_counts", {})
    if current_counts.get("eligible_sources") != len(eligible_records):
        raise BenchmarkProtocolError(
            "Manifest eligible source count does not match the eligible pool"
        )

    stored_fingerprint = manifest.get("selected_case_fingerprint")
    _require_sha256(stored_fingerprint, "selected_case_fingerprint")
    recomputed_fingerprint = selected_records_fingerprint(selected_records)
    if stored_fingerprint != recomputed_fingerprint:
        raise BenchmarkProtocolError("Selected-source fingerprint mismatch")

    if not isinstance(manifest.get("sampling_seed"), int) or isinstance(
        manifest.get("sampling_seed"), bool
    ):
        raise BenchmarkProtocolError("sampling_seed must be an integer")
    for field in ("sampling_manifest_sha256", "source_dedup_report_sha256"):
        _require_sha256(manifest.get(field), field)
    if manifest["sampling_manifest_sha256"] != _phase1_artifact_hash(
        manifest, "sampling_manifest.json"
    ):
        raise BenchmarkProtocolError("Sampling manifest provenance hash mismatch")
    if manifest["source_dedup_report_sha256"] != _phase1_artifact_hash(
        manifest, "source_dedup_report.json"
    ):
        raise BenchmarkProtocolError("Source dedup report provenance hash mismatch")
    if project_root is None:
        raise BenchmarkProtocolError("Sampling freeze requires project_root verification")
    phase1_artifacts = {
        path: digest
        for path, digest in (
            _validate_artifact_entry(entry, "phase1_artifacts")
            for entry in manifest.get("phase1_artifacts", [])
        )
    }
    _verify_artifact_map(phase1_artifacts, project_root)


def _required_manifest_fields(state: str) -> tuple[str, ...]:
    requirements = {
        "DRAFT": (
            "benchmark_id",
            "schema_version",
            "state",
            "git_commit",
            "generation_timestamp",
            "corpus_fingerprint",
            "deterministic_ordering",
            "targets",
        ),
        "SAMPLING_FROZEN": (
            "sampling_seed",
            "selection_mode",
            "selected_case_count",
            "selected_case_fingerprint",
            "sampling_manifest_sha256",
            "source_dedup_report_sha256",
            "phase1_artifacts",
        ),
        "ANNOTATION_IN_PROGRESS": (
            "annotation_started_at",
            "annotation_guidelines_path",
            "annotation_guidelines_sha256",
            "annotator_role_assignments_fingerprint",
        ),
        "ANNOTATION_FROZEN": (
            "annotation_frozen_at",
            "benchmark_path",
            "benchmark_sha256",
            "gt_schema_path",
            "gt_schema_sha256",
            "annotation_agreement_path",
            "annotation_agreement_sha256",
        ),
        "SEALED": ("sealed_at",),
        "OPENED_FOR_CONFIRMATORY_EVALUATION": (
            "opened_at",
            "preregistration_path",
            "preregistration_sha256",
        ),
    }
    fields = []
    for candidate in MANIFEST_STATES[: MANIFEST_STATES.index(state) + 1]:
        fields.extend(requirements[candidate])
    return tuple(fields)


def validate_manifest_state(manifest: dict) -> None:
    state = manifest.get("state")
    if state not in MANIFEST_STATES:
        raise BenchmarkProtocolError(f"Unknown manifest state: {state}")
    missing = [field for field in _required_manifest_fields(state) if manifest.get(field) in (None, "")]
    if missing:
        raise BenchmarkProtocolError(f"Manifest state {state} is missing fields: {missing}")
    if manifest.get("corpus_fingerprint") != CORPUS_FINGERPRINT:
        raise BenchmarkProtocolError("Frozen corpus fingerprint mismatch")

    for field in _required_manifest_fields(state):
        value = manifest.get(field)
        if field.endswith("_sha256") or field.endswith("_fingerprint"):
            _require_sha256(value, field)
    for collection_name in ("source_files", "phase1_artifacts", "sealed_artifacts"):
        if collection_name in manifest:
            collection = manifest[collection_name]
            if not isinstance(collection, list):
                raise BenchmarkProtocolError(f"{collection_name} must be a list")
            for entry in collection:
                _validate_artifact_entry(entry, collection_name)


def transition_manifest(
    manifest: dict,
    new_state: str,
    updates: dict | None = None,
    *,
    selected_records: list[dict] | None = None,
    eligible_records: list[dict] | None = None,
    project_root: Path | None = None,
) -> dict:
    current = manifest.get("state")
    if STATE_TRANSITIONS.get(current) != new_state:
        raise BenchmarkProtocolError(f"Invalid state transition: {current} -> {new_state}")
    transition_label = f"{current} -> {new_state}"
    if transition_label in manifest.get("blocked_transitions", []):
        raise BenchmarkProtocolError(f"Manifest transition is blocked: {transition_label}")
    transitioned = dict(manifest)
    transitioned.update(updates or {})
    transitioned["state"] = new_state
    validate_manifest_state(transitioned)
    if new_state == "SAMPLING_FROZEN":
        _validate_sampling_freeze(
            transitioned,
            selected_records,
            eligible_records,
            project_root,
        )
    if new_state == "ANNOTATION_IN_PROGRESS":
        _verify_manifest_artifact_pairs(
            transitioned,
            (("annotation_guidelines_path", "annotation_guidelines_sha256"),),
            project_root,
            new_state,
        )
    if new_state == "ANNOTATION_FROZEN":
        _verify_manifest_artifact_pairs(
            transitioned,
            (
                ("benchmark_path", "benchmark_sha256"),
                ("gt_schema_path", "gt_schema_sha256"),
                ("annotation_guidelines_path", "annotation_guidelines_sha256"),
                ("annotation_agreement_path", "annotation_agreement_sha256"),
            ),
            project_root,
            new_state,
        )
    if new_state in {"SEALED", "OPENED_FOR_CONFIRMATORY_EVALUATION"}:
        if project_root is None:
            raise BenchmarkProtocolError(
                f"Transition to {new_state} requires project_root verification"
            )
        verify_sealed_artifacts(transitioned, project_root)
    return transitioned


def verify_sealed_artifacts(manifest: dict, project_root: Path) -> None:
    if manifest.get("state") not in {"SEALED", "OPENED_FOR_CONFIRMATORY_EVALUATION"}:
        raise BenchmarkProtocolError("Artifact verification requires a sealed benchmark")
    validate_manifest_state(manifest)
    artifacts = _declared_artifact_map(manifest)
    sampling_hash = _phase1_artifact_hash(manifest, "sampling_manifest.json")
    dedup_hash = _phase1_artifact_hash(manifest, "source_dedup_report.json")
    if sampling_hash != manifest.get("sampling_manifest_sha256"):
        raise BenchmarkProtocolError("Sampling manifest provenance hash mismatch")
    if dedup_hash != manifest.get("source_dedup_report_sha256"):
        raise BenchmarkProtocolError("Source dedup report provenance hash mismatch")
    _verify_artifact_map(artifacts, project_root)


def git_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def phase1_payloads(project_root: Path, generation_timestamp: str | None = None) -> tuple[dict, dict]:
    project_root = Path(project_root).resolve()
    timestamp = generation_timestamp or datetime.now(timezone.utc).isoformat()
    development_path = project_root / "backend/data/judgments_evaluation.json"
    development_records = json.loads(development_path.read_text(encoding="utf-8"))
    inventory_private = inventory_judgment_sources(project_root, include_private_text=True)
    eligible, dedup = deduplicate_against_development(inventory_private, development_records)
    public_inventory = [_public_record(record) for record in inventory_private]
    inventory_fingerprint = canonical_json_fingerprint(public_inventory)
    eligible_fingerprint = canonical_json_fingerprint(eligible)
    dedup.update({
        "benchmark_id": BENCHMARK_ID,
        "generated_at": timestamp,
        "git_commit": git_commit(project_root),
        "development_benchmark_path": _relative(development_path, project_root),
        "development_benchmark_sha256": sha256_file(development_path),
        "inventory_fingerprint": inventory_fingerprint,
        "eligible_fingerprint": eligible_fingerprint,
    })
    sampling = {
        "benchmark_id": BENCHMARK_ID,
        "phase": 1,
        "state": "DRAFT",
        "generated_at": timestamp,
        "git_commit": git_commit(project_root),
        "corpus_fingerprint": CORPUS_FINGERPRINT,
        "targets": {
            "total": TARGET_TOTAL,
            "main": TARGET_MAIN,
            "stress": TARGET_STRESS,
            "minimum_total": MINIMUM_TOTAL,
            "minimum_main": MINIMUM_MAIN,
        },
        "sampling_seed": None,
        "selection_status": "NOT_STARTED",
        "ordering_rule": "case_id ASC, then source_id ASC",
        "inventory_fingerprint": inventory_fingerprint,
        "eligible_fingerprint": eligible_fingerprint,
        "available_source_inventory": public_inventory,
        "eligible_sources": eligible,
        "available_breakdown": sampling_breakdown(public_inventory),
        "eligible_breakdown": sampling_breakdown(eligible),
        "quota_feasibility": assess_quota_feasibility(eligible),
        "blockers": [
            "No unconsumed judgment source remains after excluding the 107 development cases."
        ] if not eligible else [],
        "prohibited_content": [
            "retrieval outputs",
            "candidate rankings",
            "retrieval metrics",
            "generated queries",
            "ground-truth labels",
        ],
    }
    validate_sampling_records(public_inventory)
    validate_sampling_records(eligible)
    return sampling, dedup


def draft_benchmark_manifest(
    project_root: Path,
    generation_timestamp: str | None = None,
) -> dict:
    project_root = Path(project_root).resolve()
    timestamp = generation_timestamp or datetime.now(timezone.utc).isoformat()
    benchmark_dir = project_root / "backend/data/benchmarks/retrieval_independent_v1"
    artifact_names = (
        "sampling_manifest.json",
        "source_dedup_report.json",
        "gt_schema_v2.snapshot.json",
        "annotation_guidelines.md",
    )
    artifacts = []
    for name in artifact_names:
        path = benchmark_dir / name
        if not path.exists():
            raise BenchmarkProtocolError(f"Phase-1 artifact is missing: {path}")
        artifacts.append({
            "path": _relative(path, project_root),
            "sha256": sha256_file(path),
        })
    source_paths = (
        project_root / "backend/data/judgments_evaluation.json",
        project_root / "backend/data/gt_schema_v2.json",
        project_root / "backend/scripts/Evaluation/benchmarking/independent_benchmark.py",
        project_root / "backend/scripts/Tests/test_independent_benchmark_protocol.py",
    )
    manifest = {
        "benchmark_id": BENCHMARK_ID,
        "manifest_version": "independent_benchmark_manifest_v1",
        "schema_version": SCHEMA_VERSION,
        "phase": 1,
        "state": "DRAFT",
        "benchmark_role": "independent_confirmatory_evaluation",
        "development_benchmark_role": "development_exploratory_only",
        "git_commit": git_commit(project_root),
        "generation_timestamp": timestamp,
        "corpus_fingerprint": CORPUS_FINGERPRINT,
        "sampling_seed": None,
        "deterministic_ordering": "case_id ASC, then source_id ASC",
        "targets": {
            "total": TARGET_TOTAL,
            "main": TARGET_MAIN,
            "stress": TARGET_STRESS,
            "minimum_total": MINIMUM_TOTAL,
            "minimum_main": MINIMUM_MAIN,
            "silent_quota_reduction_permitted": False,
        },
        "current_counts": {
            "available_sources": 107,
            "development_sources_consumed": 107,
            "eligible_sources": 0,
            "selected_cases": 0,
            "query_records": 0,
            "ground_truth_records": 0,
        },
        "source_files": [
            {"path": _relative(path, project_root), "sha256": sha256_file(path)}
            for path in source_paths
        ],
        "phase1_artifacts": artifacts,
        "gt_schema_path": _relative(
            benchmark_dir / "gt_schema_v2.snapshot.json", project_root
        ),
        "annotation_guidelines_path": _relative(
            benchmark_dir / "annotation_guidelines.md", project_root
        ),
        "state_machine": {
            "states": list(MANIFEST_STATES),
            "transitions": STATE_TRANSITIONS,
            "no_skipped_or_backward_transitions": True,
        },
        "sealing_contract": {
            "sampling_frozen_requires": [
                "sampling_seed",
                "selection_mode",
                "selected_case_count",
                "selected_case_fingerprint",
                "sampling_manifest_sha256",
                "source_dedup_report_sha256",
                "phase1_artifacts",
            ],
            "annotation_in_progress_requires": [
                "annotation_started_at",
                "annotation_guidelines_path",
                "annotation_guidelines_sha256",
                "annotator_role_assignments_fingerprint",
            ],
            "annotation_frozen_requires": [
                "annotation_frozen_at",
                "benchmark_path",
                "benchmark_sha256",
                "gt_schema_path",
                "gt_schema_sha256",
                "annotation_agreement_path",
                "annotation_agreement_sha256",
            ],
            "sealed_requires": [
                "selected case/source fingerprint",
                "benchmark JSON SHA256",
                "GT schema SHA256",
                "annotation guideline SHA256",
                "corpus fingerprint",
                "sealed_at",
            ],
            "opened_requires": [
                "opened_at",
                "preregistration_path",
                "preregistration_sha256",
            ],
            "sealed_content_use": (
                "Query and GT content cannot be used for method, tokenizer, "
                "representation, fusion, or candidate-depth selection."
            ),
        },
        "blocked_transitions": [
            "DRAFT -> SAMPLING_FROZEN"
        ],
        "blockers": [
            "No eligible unconsumed judgment source remains in the repository.",
            "Target and minimum sampling quotas cannot be met from the current source inventory.",
        ],
        "explicit_absences": {
            "final_benchmark_json": True,
            "generated_queries": True,
            "ground_truth_records": True,
            "retrieval_outputs": True,
            "retrieval_metrics": True,
        },
    }
    validate_manifest_state(manifest)
    return manifest

