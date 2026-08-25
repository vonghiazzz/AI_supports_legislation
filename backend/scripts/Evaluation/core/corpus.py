import hashlib
import json
import os
import re
from pathlib import Path

from .representations import build_document_text, create_standard_id


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_law_config(laws_config_path):
    if not laws_config_path or not Path(laws_config_path).exists():
        return []
    with open(laws_config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_documents(
    law_files,
    document_representation=None,
    bm25_tokenizer="legacy_optional_pyvi",
    laws_config_path=None,
):
    docs = []
    use_pyvi = False
    if bm25_tokenizer == "legacy_optional_pyvi":
        try:
            from pyvi import ViTokenizer

            use_pyvi = True
            print("   -> Đã import pyvi thành công.")
        except ImportError:
            print("   -> [CẢNH BÁO] Chưa cài đặt pyvi.")
    elif bm25_tokenizer != "whitespace":
        raise ValueError(f"Unsupported BM25 tokenizer: {bm25_tokenizer}")

    if laws_config_path is None:
        laws_config_path = Path(__file__).resolve().parents[3] / "dataLaw" / "laws_config.json"
    configured_entries = _load_law_config(laws_config_path)
    active_ranges = {
        entry["key"]: {
            "start": entry.get("active_start_year"),
            "end": entry.get("active_end_year"),
        }
        for entry in configured_entries
        if entry.get("key")
    }
    key_to_name = {
        entry["key"]: entry["law_name"]
        for entry in configured_entries
        if entry.get("key") and entry.get("law_name")
    }

    for file_path in law_files:
        if not os.path.exists(file_path):
            continue
        file_name = Path(file_path).name
        doc_law_code = file_name.replace("standardized_", "").replace(".json", "")
        year_match = re.search(r"\b(19|20)\d{2}\b", doc_law_code)
        law_year = int(year_match.group(0)) if year_match else None
        with open(file_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        for item in data:
            title = item.get("title", "")
            law_name = item.get("law_name", "") or key_to_name.get(doc_law_code, doc_law_code)
            law_domain = "procedural" if "tố tụng" in law_name.lower() else "substance"
            standard_id = create_standard_id(item.get("law_id"), title, doc_law_code)

            if document_representation in {"d0", "d1"}:
                shared_text = build_document_text(item, document_representation)
                bm25_text = shared_text
                vector_text = shared_text
            else:
                bm25_text = build_document_text(item, "legacy_enriched_weighted")
                vector_text = build_document_text(item, "legacy_dense")
            if use_pyvi:
                bm25_text = ViTokenizer.tokenize(bm25_text)

            active_range = active_ranges.get(doc_law_code, {})
            active_start = active_range.get("start")
            if active_start is None:
                active_start = law_year if law_year else 1900
            active_end = active_range.get("end")
            if active_end is None:
                active_end = 2100

            docs.append({
                "standard_law_id": standard_id,
                "original_id": item.get("law_id"),
                "bm25_text": bm25_text,
                "vector_text": vector_text,
                "title": title,
                "law_domain": law_domain,
                "law_year": law_year,
                "active_start": active_start,
                "active_end": active_end,
            })
    return docs


def load_frozen_baseline_corpus(laws_config_path, data_dir, corpus_config):
    with open(laws_config_path, "r", encoding="utf-8") as handle:
        configured_entries = json.load(handle)
    configured_keys = {entry.get("key") for entry in configured_entries}

    required_keys = list(corpus_config.get("required_keys", []))
    expected_counts = corpus_config.get("expected_counts", {})
    expected_total = int(corpus_config.get("expected_total", 0))
    if not required_keys or set(required_keys) != set(expected_counts):
        raise ValueError("Baseline corpus config must define matching required_keys and expected_counts")

    law_files = []
    inventory = []
    total = 0
    for key in required_keys:
        if key not in configured_keys:
            raise FileNotFoundError(
                f"Required baseline corpus key is absent from laws_config.json: {key}"
            )
        path = Path(data_dir) / f"standardized_{key}.json"
        if not path.exists():
            raise FileNotFoundError(f"Required baseline corpus artifact is missing: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise ValueError(f"Baseline corpus must be a JSON list: {path}")
        actual_count = len(records)
        expected_count = int(expected_counts[key])
        if actual_count != expected_count:
            raise ValueError(
                f"Corpus count mismatch for {key}: expected {expected_count}, got {actual_count}"
            )
        total += actual_count
        law_files.append(path)
        inventory.append({
            "key": key,
            "path": str(path),
            "count": actual_count,
            "sha256": sha256_file(path),
        })

    if total != expected_total:
        raise ValueError(f"Baseline corpus total mismatch: expected {expected_total}, got {total}")
    return law_files, inventory, total
