import csv
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from scripts.Evaluation.benchmarking import bsard_adapter as adapter


ARTICLE_FIELDS = adapter.ARTICLE_FIELDS
QUESTION_FIELDS = adapter.QUESTION_FIELDS
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/configs/bsard_external_zero_shot_v1.yaml"
)
MANIFEST_PATH = PROJECT_ROOT / "backend/data/benchmarks/bsard/benchmark_manifest.json"


def _write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _raw_root(tmp_path, *, articles=None, train=None, test=None):
    data = tmp_path / "bsard_v1"
    data.mkdir()
    articles = articles or [
        {
            "id": "1",
            "law_type": "federal",
            "code": "Code A",
            "article_no": "1",
            "article": "Texte premier.",
            "description": "Description A",
        },
        {
            "id": "2",
            "law_type": "regional",
            "code": "Code B",
            "article_no": "2",
            "article": "Texte deuxième.",
            "description": "",
        },
    ]
    train = train or [
        {
            "id": "10",
            "category": "A",
            "subcategory": "B",
            "question": "Question train ?",
            "extra_description": "Contexte",
            "article_ids": "1,2",
        }
    ]
    test = test or [
        {
            "id": "20",
            "category": "C",
            "subcategory": "D",
            "question": "Question test ?",
            "extra_description": "",
            "article_ids": "2,2,1",
        }
    ]
    _write_csv(data / "articles_fr.csv", ARTICLE_FIELDS, articles)
    _write_csv(data / "questions_fr_train.csv", QUESTION_FIELDS, train)
    _write_csv(data / "questions_fr_test.csv", QUESTION_FIELDS, test)
    (tmp_path / "bsard_v1.zip").write_bytes(b"synthetic archive")
    return tmp_path


def test_primary_adapter_builds_exact_canonical_schema(tmp_path):
    dataset = adapter.load_bsard(_raw_root(tmp_path), strict_identity=False)
    assert len(dataset.documents) == 2
    assert dataset.documents[0] == {
        "benchmark_id": "bsard_v1",
        "schema_version": "bsard_canonical_v1",
        "document_id": "1",
        "original_document_id": 1,
        "document_text": "Texte premier.",
        "source_metadata": {
            "law_type": "federal",
            "code": "Code A",
            "article_no": "1",
            "description": "Description A",
        },
    }
    assert dataset.human_train[0]["query_text"] == "Question train ?"
    assert dataset.human_train[0]["original_query_id"] == 10
    assert dataset.human_train[0]["split"] == "train"


def test_document_text_is_article_only_without_metadata_prefix(tmp_path):
    dataset = adapter.load_bsard(_raw_root(tmp_path), strict_identity=False)
    assert dataset.documents[0]["document_text"] == "Texte premier."
    assert "Code A" not in dataset.documents[0]["document_text"]


def test_duplicate_qrels_preserve_raw_and_unique_first_occurrence(tmp_path):
    dataset = adapter.load_bsard(_raw_root(tmp_path), strict_identity=False)
    query = dataset.human_test[0]
    assert query["raw_relevant_document_ids"] == ["2", "2", "1"]
    assert query["relevant_document_ids"] == ["2", "1"]
    assert dataset.qrel_statistics["test"] == {
        "raw_qrel_occurrence_count": 3,
        "unique_qrel_count": 2,
        "duplicate_qrel_count": 1,
    }


def test_duplicate_qrel_detection_is_generic_not_qid_specific(tmp_path):
    dataset = adapter.load_bsard(_raw_root(tmp_path), strict_identity=False)
    assert dataset.human_test[0]["query_id"] == "20"
    assert dataset.qrel_statistics["test"]["duplicate_qrel_count"] == 1


def test_synthetic_and_negative_files_are_excluded_by_default(tmp_path):
    root = _raw_root(tmp_path)
    data = root / "bsard_v1"
    (data / "questions_synthetic.csv").write_text("not,parsed\n", encoding="utf-8")
    (data / "negatives.json").write_text("not json", encoding="utf-8")
    dataset = adapter.load_bsard(root, strict_identity=False)
    assert dataset.provenance["synthetic_and_negative_pools_loaded"] is False
    assert len(dataset.human_train) == 1
    assert len(dataset.human_test) == 1


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("id", "", "document ID"),
        ("article", " ", "empty article"),
    ],
)
def test_invalid_document_fields_fail_closed(tmp_path, field, value, match):
    root = _raw_root(tmp_path)
    path = root / "bsard_v1/articles_fr.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0][field] = value
    _write_csv(path, ARTICLE_FIELDS, rows)
    with pytest.raises(adapter.BSARDProtocolError, match=match):
        adapter.load_bsard(root, strict_identity=False)


def test_duplicate_document_id_fails_closed(tmp_path):
    root = _raw_root(tmp_path)
    path = root / "bsard_v1/articles_fr.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[1]["id"] = rows[0]["id"]
    _write_csv(path, ARTICLE_FIELDS, rows)
    with pytest.raises(adapter.BSARDProtocolError, match="duplicate document ID"):
        adapter.load_bsard(root, strict_identity=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("question", " ", "empty question"),
        ("article_ids", "", "malformed article_ids"),
        ("article_ids", "1,nope", "malformed article_ids"),
        ("article_ids", "999", "unknown article IDs"),
    ],
)
def test_invalid_query_fields_fail_closed(tmp_path, field, value, match):
    root = _raw_root(tmp_path)
    path = root / "bsard_v1/questions_fr_train.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0][field] = value
    _write_csv(path, QUESTION_FIELDS, rows)
    with pytest.raises(adapter.BSARDProtocolError, match=match):
        adapter.load_bsard(root, strict_identity=False)


def test_duplicate_query_id_within_split_fails_closed(tmp_path):
    train = [
        {
            "id": "10",
            "category": "A",
            "subcategory": "B",
            "question": f"Question {index}",
            "extra_description": "",
            "article_ids": "1",
        }
        for index in range(2)
    ]
    with pytest.raises(adapter.BSARDProtocolError, match="duplicate query ID"):
        adapter.load_bsard(
            _raw_root(tmp_path, train=train), strict_identity=False
        )


def test_cross_split_query_id_overlap_fails_closed(tmp_path):
    root = _raw_root(tmp_path)
    path = root / "bsard_v1/questions_fr_test.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["id"] = "10"
    _write_csv(path, QUESTION_FIELDS, rows)
    with pytest.raises(adapter.BSARDProtocolError, match="cross-split"):
        adapter.load_bsard(root, strict_identity=False)


def test_missing_split_fails_closed():
    with pytest.raises(adapter.BSARDProtocolError, match="split"):
        adapter._load_queries([], split="", known_document_ids=set())


def test_csv_schema_drift_fails_closed(tmp_path):
    root = _raw_root(tmp_path)
    path = root / "bsard_v1/articles_fr.csv"
    path.write_text("id,article\n1,text\n", encoding="utf-8")
    with pytest.raises(adapter.BSARDProtocolError, match="unexpected CSV schema"):
        adapter.load_bsard(root, strict_identity=False)


def test_raw_hash_validator_accepts_exact_hash_and_rejects_mutation(tmp_path):
    artifact = tmp_path / "raw.csv"
    artifact.write_bytes(b"canonical")
    expected = hashlib.sha256(b"canonical").hexdigest()
    assert adapter.validate_raw_hashes(
        {"raw.csv": artifact}, {"raw.csv": expected}
    ) == {"raw.csv": expected}
    artifact.write_bytes(b"mutated")
    with pytest.raises(adapter.BSARDProtocolError, match="SHA-256 mismatch"):
        adapter.validate_raw_hashes({"raw.csv": artifact}, {"raw.csv": expected})


def test_strict_mode_checks_hashes_before_parsing(tmp_path):
    root = _raw_root(tmp_path)
    with pytest.raises(adapter.BSARDProtocolError, match="raw SHA-256 mismatch"):
        adapter.load_bsard(root, strict_identity=True)


def test_expected_identity_constants_are_frozen():
    assert adapter.EXPECTED_COUNTS == {"documents": 22633, "train": 886, "test": 222}
    assert adapter.EXPECTED_FINGERPRINTS["corpus_ordered"] == (
        "13192650b6a488221b5d98654f6274354910f509ac83b49b9e9bbaff8cccaaf3"
    )
    assert adapter.EXPECTED_RAW_SHA256["bsard_v1.zip"] == (
        "769380ab00aa472bf25226f9052df96c8d6e6472edc584ef4e2554a68c5685bc"
    )
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    exposed_sha256_values = (
        list(adapter.EXPECTED_RAW_SHA256.values())
        + list(adapter.EXPECTED_FINGERPRINTS.values())
        + list(config["identity"]["expected_raw_sha256"].values())
        + list(config["identity"]["expected_fingerprints"].values())
        + list(manifest["raw_sha256"].values())
        + list(manifest["fingerprints"].values())
    )
    assert exposed_sha256_values
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in exposed_sha256_values)


def test_provenance_metadata_contains_licenses_and_version(tmp_path):
    dataset = adapter.load_bsard(_raw_root(tmp_path), strict_identity=False)
    assert dataset.provenance["canonical_source"] == "Zenodo 5217310"
    assert dataset.provenance["canonical_version"] == "1.0"
    assert dataset.provenance["dataset_license"] == "CC-BY-NC-SA-4.0"
    assert dataset.provenance["code_reference_license"] == "MIT"


def test_zero_shot_config_is_non_executing_and_has_no_user_path():
    raw = CONFIG_PATH.read_text(encoding="utf-8")
    config = yaml.safe_load(raw)
    assert config["protocol"]["execute_models"] is False
    assert config["dataset"]["raw_root"] is None
    assert config["dataset"]["raw_root_env"] == "BSARD_RAW_ROOT"
    assert "/Users/" not in raw


def test_zero_shot_config_freezes_models_depths_and_research_questions():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["evaluation"]["candidate_diagnostic_depths"] == [
        20, 30, 50, 100, 200, 500
    ]
    assert config["systems"]["bge_m3_dense"]["role"] == (
        "primary_multilingual_dense_baseline"
    )
    assert config["systems"]["vietnamese_sbert"]["role"] == (
        "not_primary_for_bsard"
    )
    assert set(config["research_questions"]) == {
        "RQ-BSARD-1", "RQ-BSARD-2", "RQ-BSARD-3", "RQ-BSARD-4", "RQ-BSARD-5"
    }


def test_manifest_is_metadata_only_scaffold_with_frozen_licenses():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["state"] == "SCAFFOLD"
    assert manifest["raw_data_in_repository"] is False
    assert manifest["dataset_license"] == "CC-BY-NC-SA-4.0"
    assert manifest["official_reference_code"]["license"] == "MIT"
    assert all(manifest["explicit_absences"].values())
