"""Strict, model-free adapter for the canonical Zenodo BSARD v1.0 release.

Raw BSARD data is intentionally external to this repository.  Callers must
provide an explicit raw root.  The primary loader reads only the human train
and test files; synthetic questions and mined negatives require a different,
explicit future loader.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


BENCHMARK_ID = "bsard_v1"
SCHEMA_VERSION = "bsard_canonical_v1"
CANONICAL_SOURCE = "Zenodo 5217310"
CANONICAL_VERSION = "1.0"
DATASET_LICENSE = "CC-BY-NC-SA-4.0"
CODE_REFERENCE_LICENSE = "MIT"
CURRENT_REFERENCE_REVISION = "f3ca6a396a47c4a3afd26b766b5abc0a56bb4205"

EXPECTED_COUNTS = {"documents": 22633, "train": 886, "test": 222}
EXPECTED_RAW_SHA256 = {
    "articles_fr.csv": "7886d16812fdc3daa1546dc2d9a7f5501b0ada2a616a7136c309fb3b7d16f2e7",
    "questions_fr_train.csv": "ddb76885f9be9d277e08a26498e8f218727bf282ef89bdba706159467cc2ddd3",
    "questions_fr_test.csv": "2607fff3deab14a2b915366b49674cb9fde35fabdbc9d41be0aa193dfaef37c8",
    "bsard_v1.zip": "769380ab00aa472bf25226f9052df96c8d6e6472edc584ef4e2554a68c5685bc",
}
EXPECTED_FINGERPRINTS = {
    "corpus_ordered": "13192650b6a488221b5d98654f6274354910f509ac83b49b9e9bbaff8cccaaf3",
    "corpus_unordered": "13192650b6a488221b5d98654f6274354910f509ac83b49b9e9bbaff8cccaaf3",
    "logical_corpus": "ab1682c5d20f5999dbca5039b63e561f4c69d878b08859ee3ab73752d42dec08",
    "train_split": "e9d180f99f05ab0398e18898d962828c631af5a951fb5643a34fc3055e28c357",
    "test_split": "38865923e39d24b23f140be7a3bd90618aed483c3da7def248186abec7c97c21",
    "train_unique_qrels": "d7445489a340975dc8bb5cfc1655353b76de7ebc6c9baa0a721d1c917d3c11a4",
    "test_unique_qrels": "c1fbd3251a5c1a806bb85d66c8e8a73fd84b864ca6ee5ee5bc072fc3dc2cab0a",
}

ARTICLE_FIELDS = ("id", "law_type", "code", "article_no", "article", "description")
QUESTION_FIELDS = (
    "id",
    "category",
    "subcategory",
    "question",
    "extra_description",
    "article_ids",
)


class BSARDProtocolError(ValueError):
    """Raised when raw BSARD input violates the frozen protocol."""


@dataclass(frozen=True)
class BSARDDataset:
    documents: tuple[dict[str, Any], ...]
    human_train: tuple[dict[str, Any], ...]
    human_test: tuple[dict[str, Any], ...]
    provenance: Mapping[str, Any]
    qrel_statistics: Mapping[str, Mapping[str, int]]
    fingerprints: Mapping[str, str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_csv(path: Path, expected_fields: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file() or path.is_symlink():
        raise BSARDProtocolError(f"required regular raw file is missing: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_fields):
            raise BSARDProtocolError(
                f"unexpected CSV schema for {path.name}: {reader.fieldnames!r}"
            )
        rows = [dict(row) for row in reader]
    return rows


def _resolve_paths(raw_root: str | Path) -> dict[str, Path]:
    root = Path(raw_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise BSARDProtocolError(f"raw root is not a directory: {root}")
    data_root = root / "bsard_v1" if (root / "bsard_v1").is_dir() else root
    paths = {
        "articles_fr.csv": data_root / "articles_fr.csv",
        "questions_fr_train.csv": data_root / "questions_fr_train.csv",
        "questions_fr_test.csv": data_root / "questions_fr_test.csv",
    }
    zip_candidates = (root / "bsard_v1.zip", data_root.parent / "bsard_v1.zip")
    paths["bsard_v1.zip"] = next(
        (candidate for candidate in zip_candidates if candidate.is_file()),
        root / "bsard_v1.zip",
    )
    return paths


def validate_raw_hashes(
    paths: Mapping[str, Path],
    expected: Mapping[str, str] = EXPECTED_RAW_SHA256,
) -> dict[str, str]:
    """Hash every frozen raw artifact and fail closed on absence or mismatch."""
    actual: dict[str, str] = {}
    for name, expected_hash in expected.items():
        path = paths.get(name)
        if path is None or not path.is_file() or path.is_symlink():
            raise BSARDProtocolError(f"missing canonical raw artifact: {name}")
        actual[name] = _sha256_file(path)
        if actual[name] != expected_hash:
            raise BSARDProtocolError(
                f"raw SHA-256 mismatch for {name}: {actual[name]} != {expected_hash}"
            )
    return actual


def _parse_positive_ids(raw_value: str, *, query_id: str) -> list[str]:
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise BSARDProtocolError(f"query {query_id} has malformed article_ids")
    values = [value.strip() for value in raw_value.split(",")]
    if any(not value or not value.isdecimal() for value in values):
        raise BSARDProtocolError(f"query {query_id} has malformed article_ids")
    return values


def _unique_in_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    return [value for value in values if not (value in seen or seen.add(value))]


def _load_documents(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        document_id = row["id"].strip()
        if not document_id or not document_id.isdecimal():
            raise BSARDProtocolError("missing or malformed document ID")
        if document_id in seen:
            raise BSARDProtocolError(f"duplicate document ID: {document_id}")
        seen.add(document_id)
        article = row["article"]
        if not article.strip():
            raise BSARDProtocolError(f"empty article for document {document_id}")
        documents.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "schema_version": SCHEMA_VERSION,
                "document_id": document_id,
                "original_document_id": int(document_id),
                "document_text": article,
                "source_metadata": {
                    "law_type": row["law_type"],
                    "code": row["code"],
                    "article_no": row["article_no"],
                    "description": row["description"],
                },
            }
        )
    return documents


def _load_queries(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    known_document_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if split not in {"train", "test"}:
        raise BSARDProtocolError(f"missing or unsupported split: {split!r}")
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_count = 0
    unique_count = 0
    for row in rows:
        query_id = row["id"].strip()
        if not query_id or not query_id.isdecimal():
            raise BSARDProtocolError("missing or malformed query ID")
        if query_id in seen:
            raise BSARDProtocolError(f"duplicate query ID in {split}: {query_id}")
        seen.add(query_id)
        if not row["question"].strip():
            raise BSARDProtocolError(f"empty question: {query_id}")
        raw_ids = _parse_positive_ids(row["article_ids"], query_id=query_id)
        unknown = [value for value in raw_ids if value not in known_document_ids]
        if unknown:
            raise BSARDProtocolError(
                f"query {query_id} references unknown article IDs: {unknown}"
            )
        unique_ids = _unique_in_order(raw_ids)
        raw_count += len(raw_ids)
        unique_count += len(unique_ids)
        queries.append(
            {
                "benchmark_id": BENCHMARK_ID,
                "schema_version": SCHEMA_VERSION,
                "query_id": query_id,
                "original_query_id": int(query_id),
                "query_text": row["question"],
                "split": split,
                "relevant_document_ids": unique_ids,
                "raw_relevant_document_ids": raw_ids,
                "source_metadata": {
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "extra_description": row["extra_description"],
                },
            }
        )
    return queries, {
        "raw_qrel_occurrence_count": raw_count,
        "unique_qrel_count": unique_count,
        "duplicate_qrel_count": raw_count - unique_count,
    }


def _fingerprints(
    article_rows: Sequence[Mapping[str, str]],
    train_rows: Sequence[Mapping[str, str]],
    test_rows: Sequence[Mapping[str, str]],
) -> dict[str, str]:
    corpus_records = [
        {
            "id": row["id"],
            "law_type": row["law_type"],
            "code": row["code"],
            "article_no": row["article_no"],
            "article_sha256": hashlib.sha256(row["article"].encode("utf-8")).hexdigest(),
            "description": row["description"],
        }
        for row in article_rows
    ]
    logical_records = [
        {key: value for key, value in record.items() if key != "article_no"}
        for record in corpus_records
    ]

    def split_records(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
        return [
            {
                **dict(row),
                "article_ids": _parse_positive_ids(
                    row["article_ids"], query_id=row["id"]
                ),
            }
            for row in rows
        ]

    def unique_qrels(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
        pairs = {
            (row["id"], document_id)
            for row in rows
            for document_id in _parse_positive_ids(
                row["article_ids"], query_id=row["id"]
            )
        }
        return sorted(pairs, key=lambda pair: (int(pair[0]), int(pair[1])))

    return {
        "corpus_ordered": _canonical_hash(corpus_records),
        "corpus_unordered": _canonical_hash(
            sorted(corpus_records, key=lambda record: int(record["id"]))
        ),
        "logical_corpus": _canonical_hash(logical_records),
        "train_split": _canonical_hash(split_records(train_rows)),
        "test_split": _canonical_hash(split_records(test_rows)),
        "train_unique_qrels": _canonical_hash(unique_qrels(train_rows)),
        "test_unique_qrels": _canonical_hash(unique_qrels(test_rows)),
    }


def load_bsard(
    raw_root: str | Path,
    *,
    strict_identity: bool = True,
) -> BSARDDataset:
    """Load the canonical human BSARD benchmark without model or network access."""
    paths = _resolve_paths(raw_root)
    raw_hashes = validate_raw_hashes(paths) if strict_identity else {
        name: _sha256_file(path)
        for name, path in paths.items()
        if path.is_file() and not path.is_symlink()
    }
    article_rows = _read_csv(paths["articles_fr.csv"], ARTICLE_FIELDS)
    train_rows = _read_csv(paths["questions_fr_train.csv"], QUESTION_FIELDS)
    test_rows = _read_csv(paths["questions_fr_test.csv"], QUESTION_FIELDS)

    documents = _load_documents(article_rows)
    document_ids = {document["document_id"] for document in documents}
    train, train_stats = _load_queries(
        train_rows, split="train", known_document_ids=document_ids
    )
    test, test_stats = _load_queries(
        test_rows, split="test", known_document_ids=document_ids
    )
    overlap = {query["query_id"] for query in train} & {
        query["query_id"] for query in test
    }
    if overlap:
        raise BSARDProtocolError(f"cross-split query-ID overlap: {sorted(overlap)}")

    fingerprints = _fingerprints(article_rows, train_rows, test_rows)
    if strict_identity:
        actual_counts = {
            "documents": len(documents), "train": len(train), "test": len(test)
        }
        if actual_counts != EXPECTED_COUNTS:
            raise BSARDProtocolError(
                f"strict canonical count mismatch: {actual_counts} != {EXPECTED_COUNTS}"
            )
        for name, expected in EXPECTED_FINGERPRINTS.items():
            if fingerprints[name] != expected:
                raise BSARDProtocolError(
                    f"strict canonical fingerprint mismatch for {name}: "
                    f"{fingerprints[name]} != {expected}"
                )

    return BSARDDataset(
        documents=tuple(documents),
        human_train=tuple(train),
        human_test=tuple(test),
        provenance={
            "canonical_source": CANONICAL_SOURCE,
            "canonical_version": CANONICAL_VERSION,
            "dataset_license": DATASET_LICENSE,
            "code_reference_license": CODE_REFERENCE_LICENSE,
            "current_reference_revision": CURRENT_REFERENCE_REVISION,
            "raw_sha256": raw_hashes,
            "synthetic_and_negative_pools_loaded": False,
        },
        qrel_statistics={"train": train_stats, "test": test_stats},
        fingerprints=fingerprints,
    )


__all__ = [
    "BENCHMARK_ID",
    "SCHEMA_VERSION",
    "BSARDDataset",
    "BSARDProtocolError",
    "EXPECTED_COUNTS",
    "EXPECTED_FINGERPRINTS",
    "EXPECTED_RAW_SHA256",
    "load_bsard",
    "validate_raw_hashes",
]
