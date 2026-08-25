import json
import re
import time
import os
import sys
import math
from pathlib import Path
import torch
import numpy as np
import yaml
import shutil
import argparse
import datetime
import hashlib
import importlib.metadata
import platform
import random
import subprocess
import uuid

# Default config loading at module-level
DEFAULT_CONFIG_PATH = Path(__file__).parent / "eval_config.yaml"
try:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        _eval_config = yaml.safe_load(f)
except Exception:
    _eval_config = {
        'evaluation': {'k': 20, 'pool_k': 30, 'alpha_core': 3, 'alpha_support': 1},
        'fusion': {'temperature_bm25': 5.0, 'temperature_faiss': 0.5}
    }

_eval_section = _eval_config.get('evaluation', {})
ALPHA_CORE = _eval_section.get('alpha_core', 3)
ALPHA_SUPPORT = _eval_section.get('alpha_support', 1)
print(f"Loaded alpha_core = {ALPHA_CORE}")
print(f"Loaded alpha_support = {ALPHA_SUPPORT}")

# ==========================================
# 1. DATA PREPARATION & NORMALIZATION
# ==========================================

def normalize_text(text):
    if not text:
        return ""
    return " ".join(str(text).lower().split())

def softmax_normalize(scores, temperature=1.0):
    if not scores:
        return []
    max_val = max(scores)
    exp_scores = [math.exp((s - max_val) / temperature) for s in scores]
    sum_exp = sum(exp_scores)
    if sum_exp == 0:
        return [1.0 / len(scores) for _ in scores]
    return [s / sum_exp for s in exp_scores]

BASELINE_MODE = "baseline"
BASELINE_CONFIGURATIONS = ["BM25", "FAISS", "Hybrid_Static_Fixed"]
BASELINE_REPORTED_K = (5, 10, 20)
BASELINE_PRIMARY_K = 10
BASELINE_PRIMARY_WORLD = "closed_world"
SUPPORTED_TEMPORAL_POLICIES = {"none", "judgment_year", "active_range"}
LAW_CODE_ALIASES = {
    "luat_hon_nhan_va_gia_dinh_2014": "luat_hon_nhan_gia_dinh_2014",
}


def normalize_query_q0(query):
    """Baseline Q0: the benchmark query with whitespace normalization only."""
    if query is None:
        return ""
    return " ".join(str(query).strip().split())


def clean_query_legacy(query):
    """Historical query cleaner retained only for legacy-run reproduction."""
    if not query:
        return ""
    text = str(query)
    match_vv = re.search(r"v/v:?\s*(.*)", text, re.IGNORECASE)
    cleaned = match_vv.group(1) if match_vv else text
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["giám đốc thẩm", "giám đốc thầm", "giám đốc thấm"]):
        cleaned = "giám đốc thẩm " + cleaned
    elif any(kw in text_lower for kw in ["phúc thẩm", "phúc thầm", "phúc thấm"]):
        cleaned = "phúc thẩm " + cleaned
    elif any(kw in text_lower for kw in ["sơ thẩm", "sơ thầm", "sơ thấm"]):
        cleaned = "sơ thẩm " + cleaned
    return " ".join(cleaned.split())


# Backwards-compatible name for legacy helpers and historical reporting code.
clean_query = clean_query_legacy


def build_query_text(record, representation):
    if representation == "q0":
        return normalize_query_q0(record.get("query", ""))
    if representation == "legacy":
        return clean_query_legacy(record.get("query", ""))
    raise ValueError(f"Unsupported query representation: {representation}")


def canonicalize_law_code(law_code):
    normalized = normalize_text(law_code).replace(" ", "_")
    return LAW_CODE_ALIASES.get(normalized, normalized)


def build_document_text(document, representation):
    title = " ".join(str(document.get("title", "")).split())
    content = " ".join(str(document.get("content", "")).split())
    summary = " ".join(str(document.get("summary", "")).split())
    lexical = " ".join(str(value).strip() for value in document.get("lexical_keywords", []) if str(value).strip())
    keywords = " ".join(str(value).strip() for value in document.get("keywords", []) if str(value).strip())

    if representation == "d0":
        return f"Tiêu đề: {title}. Nội dung: {content}".strip()
    if representation == "d1":
        terms = " ".join(value for value in [lexical, keywords] if value)
        return f"Tiêu đề: {title}. Nội dung: {content}. Tóm tắt: {summary}. Từ khóa: {terms}".strip()
    if representation == "legacy_enriched_weighted":
        tags = " ".join(str(value).strip() for value in document.get("tags", []) if str(value).strip())
        parts = [title, title, title, keywords, keywords, summary, summary, tags, lexical, content]
        return " ".join(value for value in parts if value)
    if representation == "legacy_dense":
        return " ".join(f"Tiêu đề: {title}. Tóm tắt: {summary}. Từ khóa: {lexical} {keywords}".split())
    raise ValueError(f"Unsupported document representation: {representation}")


def get_dynamic_weight(route, dynamic_weights):
    if route not in {"exact_citation", "procedural", "substance", "mixed"}:
        route = "mixed"
    return float(dynamic_weights[route])


def should_include_document(document, temporal_policy, judgment_year=None, target_year=None):
    if temporal_policy not in SUPPORTED_TEMPORAL_POLICIES:
        raise ValueError(f"Unsupported temporal policy: {temporal_policy}")
    if temporal_policy == "none":
        return True
    if temporal_policy == "judgment_year":
        if judgment_year is None or document.get("law_year") is None:
            return True
        return int(document["law_year"]) <= int(judgment_year)
    effective_year = target_year if target_year is not None else judgment_year
    if effective_year is None:
        return True
    return int(document.get("active_start", 1900)) <= int(effective_year) <= int(document.get("active_end", 2100))

def extract_article_number(text):
    match = re.search(r"(điều\s+\d+[a-z]?)", str(text), re.IGNORECASE)
    return match.group(1).lower() if match else normalize_text(text)

def get_base_law_name(law_name):
    name = normalize_text(law_name)
    if "tố tụng dân sự" in name or "to_tung_dan_su" in name: return "bộ luật tố tụng dân sự"
    if "dân sự" in name or "dan_su" in name: return "bộ luật dân sự"
    return name

def create_standard_id(law_id, title, law_name):
    if law_id:
        clean_id = str(law_id).strip().lower().replace("điều", "").strip()
        article = f"điều {clean_id}"
    else:
        article = extract_article_number(title)
    return f"{article} {canonicalize_law_code(law_name)}"

def is_match(retrieved_id, ground_truth):
    ret_norm = normalize_text(retrieved_id)
    gt_norm = normalize_text(ground_truth)
    ret_article = extract_article_number(ret_norm)
    gt_article = extract_article_number(gt_norm)
    
    ret_parts = ret_norm.split()
    gt_parts = gt_norm.split()
    
    if len(ret_parts) < 2 or len(gt_parts) < 2:
        return False
        
    ret_law = canonicalize_law_code(ret_parts[-1])
    gt_law = canonicalize_law_code(gt_parts[-1])
    
    if ret_article and gt_article and ret_law and gt_law:
        return ret_article == gt_article and ret_law == gt_law
    return False


def canonical_retrieval_id(value):
    """Canonical key using the evaluator's existing article-plus-law matching contract."""
    normalized = normalize_text(value)
    parts = normalized.split()
    if len(parts) < 2:
        return normalized
    article = extract_article_number(normalized)
    law_code = canonicalize_law_code(parts[-1])
    return normalize_text(f"{article} {law_code}")

def extract_gt_sub_proc(gt_obj):
    if not gt_obj:
        return [], []
    core_obj = gt_obj.get('core', [])
    supp_obj = gt_obj.get('supporting', [])
    
    if isinstance(core_obj, list) and isinstance(supp_obj, list):
        sub = []
        proc = []
        for item in core_obj + supp_obj:
            domain = item.get('domain')
            if domain == 'substantive':
                sub.append(item)
            elif domain == 'procedural':
                proc.append(item)
            else:
                sub.append(item)
        return sub, proc
    else:
        # Legacy fallback
        core_dict = core_obj if isinstance(core_obj, dict) else {}
        supp_dict = supp_obj if isinstance(supp_obj, dict) else {}
        sub = core_dict.get('substantive', []) + supp_dict.get('substantive', [])
        proc = core_dict.get('procedural', []) + supp_dict.get('procedural', [])
        return sub, proc

def get_gt_weights(gt_obj):
    weights = {}
    if not gt_obj:
        return weights
    
    core_obj = gt_obj.get('core', [])
    supp_obj = gt_obj.get('supporting', [])
    
    if isinstance(core_obj, list) and isinstance(supp_obj, list):
        for item in core_obj:
            std_val = f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()
            if std_val:
                weights[normalize_text(std_val)] = ALPHA_CORE
        for item in supp_obj:
            std_val = f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()
            if std_val:
                norm_val = normalize_text(std_val)
                if norm_val not in weights:
                    weights[norm_val] = ALPHA_SUPPORT
    else:
        # Legacy fallback
        core_dict = core_obj if isinstance(core_obj, dict) else {}
        for item in core_dict.get('substantive', []) + core_dict.get('procedural', []):
            std_val = f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()
            if std_val:
                weights[normalize_text(std_val)] = ALPHA_CORE
                
        supp_dict = supp_obj if isinstance(supp_obj, dict) else {}
        for item in supp_dict.get('substantive', []) + supp_dict.get('procedural', []):
            std_val = f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()
            if std_val:
                norm_val = normalize_text(std_val)
                if norm_val not in weights:
                    weights[norm_val] = ALPHA_SUPPORT
    return weights

def get_query_intent(query_clean):
    q = query_clean.lower()
    if re.search(r"điều\s*\d+", q) or re.search(r"khoản\s*\d+", q):
        return "exact_citation"
        
    procedural_keywords = ["giám đốc thẩm", "tái thẩm", "phúc thẩm", "kháng nghị", "đình chỉ", "hủy án", "tố tụng"]
    substance_keywords = ["tranh chấp", "bồi thường", "hợp đồng", "quyền sử dụng đất", "tài sản", "vô hiệu", "ly hôn", "thừa kế", "chia di sản"]
    
    has_procedural = any(kw in q for kw in procedural_keywords)
    has_substance = any(kw in q for kw in substance_keywords)
    
    if has_procedural and has_substance: return "mixed"
    elif has_procedural: return "procedural"
    elif has_substance: return "substance"
    else: return "mixed"

def load_documents(law_files, document_representation=None, bm25_tokenizer="legacy_optional_pyvi"):
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

    config_path = Path(__file__).parent.parent.parent / "dataLaw" / "laws_config.json"
    active_ranges = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for entry in config:
                    key = entry.get("key")
                    if key:
                        active_ranges[key] = {
                            "start": entry.get("active_start_year"),
                            "end": entry.get("active_end_year")
                        }
        except Exception as e:
            print(f"⚠️ Lỗi đọc laws_config.json trong load_documents: {e}")

    for file_path in law_files:
        if not os.path.exists(file_path): continue
        
        file_name = Path(file_path).name
        key = file_name.replace("standardized_", "").replace(".json", "")
        doc_law_code = key
        
        # Trích xuất năm của bộ luật từ key (ví dụ: bo_luat_dan_su_2015 -> 2015)
        year_match = re.search(r'\b(19|20)\d{2}\b', doc_law_code)
        law_year = int(year_match.group(0)) if year_match else None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            title = item.get('title', '')
            law_name = item.get('law_name', '')
            if not law_name:
                law_name = KEY_TO_NAME_MAP.get(doc_law_code, doc_law_code)
            
            # Domain classification
            law_domain = "procedural" if "tố tụng" in law_name.lower() else "substance"
            
            std_id = create_standard_id(item.get('law_id'), title, doc_law_code)
            
            if document_representation in {"d0", "d1"}:
                shared_text = build_document_text(item, document_representation)
                bm25_text = shared_text
                vector_text = shared_text
            else:
                bm25_text = build_document_text(item, "legacy_enriched_weighted")
                vector_text = build_document_text(item, "legacy_dense")
            if use_pyvi: bm25_text = ViTokenizer.tokenize(bm25_text)
            
            active_range = active_ranges.get(doc_law_code, {})
            active_start = active_range.get("start")
            if active_start is None:
                active_start = law_year if law_year else 1900
                
            active_end = active_range.get("end")
            if active_end is None:
                active_end = 2100
            
            docs.append({
                "standard_law_id": std_id,
                "original_id": item.get('law_id'),
                "bm25_text": bm25_text,
                "vector_text": vector_text,
                "title": title,
                "law_domain": law_domain,
                "law_year": law_year,
                "active_start": active_start,
                "active_end": active_end
            })
    return docs


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_records(records):
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frozen_baseline_corpus(laws_config_path, data_dir, corpus_config):
    """Resolve and validate only the explicitly frozen research corpus."""
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
            raise FileNotFoundError(f"Required baseline corpus key is absent from laws_config.json: {key}")
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
            raise ValueError(f"Corpus count mismatch for {key}: expected {expected_count}, got {actual_count}")
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


def build_canonical_ground_truth(record, corpus_document_ids):
    """Build raw/open-world and exact corpus-membership/closed-world GT views."""
    gt_obj = record.get("ground_truth", {})
    core = list(gt_obj.get("core", []))
    supporting = list(gt_obj.get("supporting", []))
    corpus_id_keys = {canonical_retrieval_id(value) for value in corpus_document_ids}

    def standard_id(item):
        return f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()

    def in_closed_world(item):
        return canonical_retrieval_id(standard_id(item)) in corpus_id_keys

    core_raw = [standard_id(item) for item in core]
    supporting_raw = [standard_id(item) for item in supporting]
    core_filtered = [standard_id(item) for item in core if in_closed_world(item)]
    supporting_filtered = [standard_id(item) for item in supporting if in_closed_world(item)]
    substantive_raw = [standard_id(item) for item in core + supporting if item.get("domain") == "substantive"]
    procedural_raw = [standard_id(item) for item in core + supporting if item.get("domain") == "procedural"]
    substantive_filtered = [standard_id(item) for item in core + supporting if item.get("domain") == "substantive" and in_closed_world(item)]
    procedural_filtered = [standard_id(item) for item in core + supporting if item.get("domain") == "procedural" and in_closed_world(item)]
    explicit_filtered = [standard_id(item) for item in core + supporting if item.get("source_type") == "explicit" and in_closed_world(item)]
    inferred_filtered = [standard_id(item) for item in core + supporting if item.get("source_type") != "explicit" and in_closed_world(item)]
    return {
        "core_gt_raw": core_raw,
        "supporting_gt_raw": supporting_raw,
        "core_gt_filtered": core_filtered,
        "supporting_gt_filtered": supporting_filtered,
        "substance_gt_raw": substantive_raw,
        "procedural_gt_raw": procedural_raw,
        "substance_gt_filtered": substantive_filtered,
        "procedural_gt_filtered": procedural_filtered,
        "explicit_gts": explicit_filtered,
        "inferred_gts": inferred_filtered,
        "gt_weights": get_gt_weights(gt_obj),
    }


def compute_ground_truth_coverage(benchmark, corpus_document_ids):
    """Count benchmark label occurrences after alias normalization and exact corpus lookup."""
    total_labels = 0
    in_corpus_labels = 0
    queries_with_in_corpus = 0
    corpus_id_keys = {canonical_retrieval_id(value) for value in corpus_document_ids}

    for record in benchmark:
        gt_obj = record.get("ground_truth", {})
        labels = list(gt_obj.get("core", [])) + list(gt_obj.get("supporting", []))
        query_in_corpus = 0
        for item in labels:
            canonical_id = canonical_retrieval_id(
                f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()
            )
            total_labels += 1
            if canonical_id in corpus_id_keys:
                in_corpus_labels += 1
                query_in_corpus += 1
        if query_in_corpus:
            queries_with_in_corpus += 1

    return {
        "total_labels": total_labels,
        "in_corpus_labels": in_corpus_labels,
        "out_of_corpus_labels": total_labels - in_corpus_labels,
        "coverage_ratio": in_corpus_labels / total_labels if total_labels else 0.0,
        "queries_with_at_least_one_in_corpus_gt": queries_with_in_corpus,
        "queries_without_in_corpus_gt": len(benchmark) - queries_with_in_corpus,
    }


def get_enabled_configurations(mode):
    if mode == BASELINE_MODE:
        return list(BASELINE_CONFIGURATIONS)
    return ["BM25", "FAISS", "Hybrid_Static_Fixed", "Hybrid_Static_Opt", "Hybrid_Dynamic", "Hybrid_Dynamic_Rerank"]


def should_initialize_reranker(mode):
    return mode != BASELINE_MODE


def should_run_tuning(mode):
    return mode != BASELINE_MODE


def _safe_package_version(package_name):
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_provenance(project_root, baseline_source_paths):
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=project_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        source_status = subprocess.run(
            ["git", "status", "--porcelain", "--", *baseline_source_paths],
            cwd=project_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
        return {
            "git_commit": commit,
            "git_dirty": bool(status),
            "tracked_code_dirty": bool(tracked_status),
            "baseline_source_dirty": bool(source_status),
            "baseline_source_status": source_status.splitlines(),
        }
    except Exception:
        return {
            "git_commit": None,
            "git_dirty": None,
            "tracked_code_dirty": None,
            "baseline_source_dirty": None,
            "baseline_source_status": [],
        }


def _resolved_model_revision(embedder):
    try:
        config = embedder._first_module().auto_model.config
        return getattr(config, "_commit_hash", None)
    except Exception:
        return None


def verify_model_revision(requested_revision, resolved_revision):
    if not requested_revision:
        raise ValueError("Baseline embedding_revision must be explicitly pinned")
    if not resolved_revision:
        raise RuntimeError("Could not resolve the loaded embedding model revision")
    if requested_revision != resolved_revision:
        raise RuntimeError(
            "Embedding revision mismatch: "
            f"requested {requested_revision}, resolved {resolved_revision}"
        )


def aggregate_latency_ms(values):
    values = [float(value) for value in values]
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def build_latency_aggregates(results):
    latency_rows = [item["latency_ms"] for item in results]
    bm25 = aggregate_latency_ms([row["bm25_retrieval_ms"] for row in latency_rows])
    dense = aggregate_latency_ms([row["dense_total_ms"] for row in latency_rows])
    hybrid_serial = aggregate_latency_ms([row["hybrid_serial_ms"] for row in latency_rows])
    hybrid_parallel = aggregate_latency_ms([row["hybrid_parallel_estimate_ms"] for row in latency_rows])
    return {
        "BM25": bm25,
        "FAISS": {
            **dense,
            "components": {
                "dense_encode": aggregate_latency_ms([row["dense_encode_ms"] for row in latency_rows]),
                "faiss_search": aggregate_latency_ms([row["faiss_search_ms"] for row in latency_rows]),
            },
        },
        "Hybrid_Static_Fixed": {
            **hybrid_serial,
            "measurement": "serial_wall_clock_components",
            "fusion": aggregate_latency_ms([row["fusion_ms"] for row in latency_rows]),
            "parallel_estimate": {
                **hybrid_parallel,
                "estimated": True,
                "formula": "max(bm25_retrieval_ms, dense_total_ms) + fusion_ms",
            },
        },
    }


def assemble_baseline_summary(metadata, metric_worlds, gt_coverage, latency_aggregates):
    if metadata.get("primary_world") != BASELINE_PRIMARY_WORLD:
        raise ValueError("Baseline summary primary_world must be closed_world")
    if int(metadata.get("primary_k", 0)) != BASELINE_PRIMARY_K:
        raise ValueError("Baseline summary primary_k must be 10")
    return {
        **metadata,
        "ground_truth_coverage": gt_coverage,
        "closed_world": metric_worlds["closed_world"],
        "open_world": metric_worlds["open_world"],
        "latency_ms": latency_aggregates,
    }


def enforce_full_run_provenance(max_queries, benchmark_count, provenance):
    is_full_run = max_queries is None or int(max_queries) >= int(benchmark_count)
    if is_full_run and provenance.get("baseline_source_dirty") is not False:
        status = provenance.get("baseline_source_status") or ["source status unavailable"]
        raise RuntimeError(
            "Full baseline requires committed evaluator/config/test/experiment-lock files. "
            f"Current baseline source status: {status}"
        )


def run_baseline_evaluation_pipeline(config_path, max_queries=None):
    """Run the frozen B0/B1/E1 protocol without reranking, routing, or tuning."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    evaluation = config["evaluation"]
    if evaluation.get("mode") != BASELINE_MODE:
        raise ValueError("run_baseline_evaluation_pipeline requires evaluation.mode=baseline")
    query_representation = evaluation.get("query_representation")
    document_representation = evaluation.get("document_representation")
    temporal_policy = evaluation.get("temporal_filter")
    if query_representation != "q0":
        raise ValueError("Baseline mode requires query_representation=q0")
    if document_representation not in {"d0", "d1"}:
        raise ValueError("Baseline mode requires document_representation=d0 or d1")
    if temporal_policy != "none":
        raise ValueError("Baseline B0/B1/E1 requires temporal_filter=none")

    bm25_tokenizer = config.get("bm25", {}).get("tokenizer")
    if bm25_tokenizer != "whitespace":
        raise ValueError("Baseline mode requires deterministic bm25.tokenizer=whitespace")
    fusion_method = config["fusion"].get("method")
    static_w_bm25 = float(config["fusion"].get("static_w_bm25"))
    if fusion_method != "minmax" or static_w_bm25 != 0.5:
        raise ValueError("Baseline E1 requires minmax fusion with static_w_bm25=0.5")

    metric_protocol = config.get("metric_protocol", {})
    primary_world = metric_protocol.get("primary_world")
    primary_k = int(metric_protocol.get("primary_k", 0))
    reported_k = tuple(int(value) for value in metric_protocol.get("reported_k", []))
    if (
        primary_world != BASELINE_PRIMARY_WORLD
        or primary_k != BASELINE_PRIMARY_K
        or reported_k != BASELINE_REPORTED_K
    ):
        raise ValueError(
            "Baseline metric protocol requires primary_world=closed_world, "
            "primary_k=10, reported_k=[5, 10, 20]"
        )
    if int(evaluation["k"]) < max(reported_k):
        raise ValueError("Stored retrieval depth must cover every reported metric K")

    random_seed = int(evaluation.get("random_seed", 42))
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    project_root = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = project_root / config["paths"]["output_dir"]
    laws_config_path = project_root / config["paths"]["laws_config"]
    benchmark_path = project_root / config["paths"]["eval_file"]
    schema_path = output_dir / "gt_schema_v2.json"

    sys.path.append(str(Path(__file__).parent.resolve()))
    from validate_benchmark import validate_benchmark_file
    validate_benchmark_file(benchmark_path)

    law_files, corpus_inventory, corpus_count = load_frozen_baseline_corpus(
        laws_config_path, output_dir, config["corpus"]
    )
    documents = load_documents(
        law_files,
        document_representation=document_representation,
        bm25_tokenizer=bm25_tokenizer,
    )
    if len(documents) != corpus_count:
        raise ValueError(f"Document count changed during representation build: {len(documents)} != {corpus_count}")
    if any(doc["bm25_text"] != doc["vector_text"] for doc in documents):
        raise AssertionError("Baseline BM25 and dense must use the same document representation")

    with open(benchmark_path, "r", encoding="utf-8") as handle:
        benchmark = json.load(handle)
    selected_queries = benchmark[:max_queries] if max_queries is not None else benchmark
    if not selected_queries:
        raise ValueError("No benchmark queries selected")

    baseline_source_paths = [
        "backend/scripts/Evaluation/run_rag_evaluation.py",
        "backend/scripts/Evaluation/eval_config.yaml",
        "backend/scripts/Evaluation/requirements-baseline.txt",
        "backend/scripts/Tests/test_baseline_protocol.py",
    ]
    provenance = _git_provenance(project_root, baseline_source_paths)
    enforce_full_run_provenance(max_queries, len(benchmark), provenance)

    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import faiss

    tokenized_corpus = [doc["bm25_text"].lower().split() for doc in documents]
    bm25_index = BM25Okapi(tokenized_corpus)
    embedding_model = config["models"]["embedding_model"]
    embedding_revision = config["models"].get("embedding_revision")
    local_files_only = bool(config["models"].get("local_files_only", True))
    embedder = SentenceTransformer(
        embedding_model,
        revision=embedding_revision,
        device="cpu",
        local_files_only=local_files_only,
    )
    resolved_embedding_revision = _resolved_model_revision(embedder)
    verify_model_revision(embedding_revision, resolved_embedding_revision)
    corpus_embeddings = embedder.encode(
        [doc["vector_text"] for doc in documents],
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(corpus_embeddings)
    embedding_dimension = int(corpus_embeddings.shape[1])
    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    faiss_index.add(corpus_embeddings)

    global ALPHA_CORE, ALPHA_SUPPORT
    ALPHA_CORE = int(evaluation.get("alpha_core", 3))
    ALPHA_SUPPORT = int(evaluation.get("alpha_support", 1))
    k = int(evaluation["k"])
    pool_k = int(evaluation["pool_k"])
    corpus_document_ids = {item["standard_law_id"] for item in documents}
    gt_coverage = compute_ground_truth_coverage(benchmark, corpus_document_ids)
    final_results = []

    warmup_queries = int(config.get("latency", {}).get("warmup_queries", 1))
    if warmup_queries < 1:
        raise ValueError("Baseline latency protocol requires at least one warm-up query")
    for warmup_index in range(warmup_queries):
        warmup_record = selected_queries[warmup_index % len(selected_queries)]
        warmup_text = build_query_text(warmup_record, query_representation)
        warmup_bm25_scores = bm25_index.get_scores(warmup_text.lower().split())
        warmup_bm25_indices = sorted(
            range(len(documents)),
            key=lambda idx: (-float(warmup_bm25_scores[idx]), idx),
        )[:pool_k]
        warmup_bm25_pool = [
            {
                "rank": rank,
                "standard_law_id": documents[idx]["standard_law_id"],
                "score": float(warmup_bm25_scores[idx]),
            }
            for rank, idx in enumerate(warmup_bm25_indices, 1)
        ]
        warmup_embedding = embedder.encode(
            [warmup_text], show_progress_bar=False, convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(warmup_embedding)
        warmup_scores, warmup_indices = faiss_index.search(warmup_embedding, pool_k)
        warmup_faiss_pool = [
            {
                "rank": rank,
                "standard_law_id": documents[int(idx)]["standard_law_id"],
                "score": float(score),
            }
            for rank, (score, idx) in enumerate(zip(warmup_scores[0], warmup_indices[0]), 1)
            if idx >= 0
        ]
        fuse_pools(warmup_bm25_pool, warmup_faiss_pool, static_w_bm25, fusion_method)

    for index, record in enumerate(selected_queries):
        query_text = build_query_text(record, query_representation)

        started = time.perf_counter_ns()
        bm25_scores = bm25_index.get_scores(query_text.lower().split())
        bm25_indices = sorted(range(len(documents)), key=lambda idx: (-float(bm25_scores[idx]), idx))[:pool_k]
        bm25_pool = [
            {"rank": rank, "standard_law_id": documents[idx]["standard_law_id"], "score": float(bm25_scores[idx])}
            for rank, idx in enumerate(bm25_indices, 1)
        ]
        bm25_retrieval_ms = (time.perf_counter_ns() - started) / 1_000_000.0

        started = time.perf_counter_ns()
        query_embedding = embedder.encode([query_text], show_progress_bar=False, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)
        dense_encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0

        started = time.perf_counter_ns()
        dense_scores, dense_indices = faiss_index.search(query_embedding, pool_k)
        faiss_pool = [
            {"rank": rank, "standard_law_id": documents[int(idx)]["standard_law_id"], "score": float(score)}
            for rank, (score, idx) in enumerate(zip(dense_scores[0], dense_indices[0]), 1)
            if idx >= 0
        ]
        faiss_search_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        dense_total_ms = dense_encode_ms + faiss_search_ms

        started = time.perf_counter_ns()
        hybrid = fuse_pools(bm25_pool, faiss_pool, static_w_bm25, fusion_method)
        hybrid_pool = [
            {"rank": rank, "standard_law_id": standard_id, "score": float(score)}
            for rank, (standard_id, score) in enumerate(hybrid[:k], 1)
        ]
        fusion_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        hybrid_serial_ms = bm25_retrieval_ms + dense_total_ms + fusion_ms
        hybrid_parallel_estimate_ms = max(bm25_retrieval_ms, dense_total_ms) + fusion_ms
        gt_contract = build_canonical_ground_truth(record, corpus_document_ids)
        final_results.append({
            "query_id": record.get("id", index),
            "query": query_text,
            **gt_contract,
            "latency_ms": {
                "bm25_retrieval_ms": bm25_retrieval_ms,
                "dense_encode_ms": dense_encode_ms,
                "faiss_search_ms": faiss_search_ms,
                "dense_total_ms": dense_total_ms,
                "fusion_ms": fusion_ms,
                "hybrid_serial_ms": hybrid_serial_ms,
                "hybrid_parallel_estimate_ms": hybrid_parallel_estimate_ms,
            },
            "metrics": {
                "BM25": {"top_k": bm25_pool[:k]},
                "FAISS": {"top_k": faiss_pool[:k]},
                "Hybrid_Static_Fixed": {"top_k": hybrid_pool},
            },
        })

    configurations = get_enabled_configurations(BASELINE_MODE)
    metric_worlds = build_metric_world_report(final_results, configurations, reported_k)
    latency_aggregates = build_latency_aggregates(final_results)

    now = datetime.datetime.now(datetime.timezone.utc)
    run_id = f"baseline_{now.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:8]}"
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    summary_metadata = {
        "run_id": run_id,
        "evaluation_mode": BASELINE_MODE,
        "query_representation": query_representation,
        "document_representation": document_representation,
        "temporal_filter": temporal_policy,
        "corpus_count": corpus_count,
        "benchmark_query_count": len(benchmark),
        "evaluated_query_count": len(selected_queries),
        "primary_world": primary_world,
        "primary_k": primary_k,
        "metric_protocol": {
            "primary_world": primary_world,
            "primary_k": primary_k,
            "reported_k": list(reported_k),
            "stored_ranking_k": k,
        },
        "embedding_model": embedding_model,
        "embedding_revision": resolved_embedding_revision,
        "embedding_requested_revision": embedding_revision,
        "fusion_method": fusion_method,
        "static_w_bm25": static_w_bm25,
    }
    summary = assemble_baseline_summary(
        summary_metadata, metric_worlds, gt_coverage, latency_aggregates
    )
    results_output = {
        "run_id": run_id,
        "experiment_id": evaluation["experiment_id"],
        "evaluation_mode": BASELINE_MODE,
        "configurations": configurations,
        "results": final_results,
    }
    results_path = run_dir / "pipeline_results.json"
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results_output, handle, ensure_ascii=False, indent=2)
    with open(run_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    shutil.copy2(config_path, run_dir / "eval_config.snapshot.yaml")

    query_fingerprint = fingerprint_records([
        {"id": record.get("id"), "text": build_query_text(record, query_representation)}
        for record in selected_queries
    ])
    combined_corpus_fingerprint = fingerprint_records([
        {"key": item["key"], "count": item["count"], "sha256": item["sha256"]}
        for item in corpus_inventory
    ])
    manifest = {
        "run_id": run_id,
        "timestamp": now.isoformat(),
        "evaluation_mode": BASELINE_MODE,
        "experiment_id": evaluation["experiment_id"],
        "query_count": len(selected_queries),
        "benchmark_query_count": len(benchmark),
        "corpus_count": corpus_count,
        "query_representation": query_representation,
        "document_representation": document_representation,
        "temporal_filter": temporal_policy,
        "embedding_model": embedding_model,
        "embedding_requested_revision": embedding_revision,
        "embedding_resolved_revision": resolved_embedding_revision,
        "embedding_revision_verified": embedding_revision == resolved_embedding_revision,
        "embedding_local_files_only": local_files_only,
        "embedding_dimension": embedding_dimension,
        "embedding_normalized": True,
        "faiss_index_type": type(faiss_index).__name__,
        "bm25_tokenizer": bm25_tokenizer,
        "bm25_tokenizer_version": None,
        "fusion_method": fusion_method,
        "static_w_bm25": static_w_bm25,
        "alpha_core": ALPHA_CORE,
        "alpha_support": ALPHA_SUPPORT,
        "metric_protocol": summary["metric_protocol"],
        "ground_truth_coverage": gt_coverage,
        "latency_ms": latency_aggregates,
        "warmup_policy": {
            "enabled": True,
            "query_count": warmup_queries,
            "query_selection": "first_selected_queries_cycled_if_needed",
            "components": ["bm25_retrieval", "dense_encode", "faiss_search", "fusion"],
            "excluded_from_per_query_latency": True,
            "model_initialization_excluded": True,
            "corpus_embedding_build_excluded": True,
        },
        "benchmark": {"path": str(benchmark_path), "sha256": sha256_file(benchmark_path)},
        "gt_schema": {"path": str(schema_path), "sha256": sha256_file(schema_path)},
        "corpora": corpus_inventory,
        "combined_corpus_fingerprint": combined_corpus_fingerprint,
        "query_representation_fingerprint": query_fingerprint,
        **provenance,
        "python_version": platform.python_version(),
        "package_versions": {
            name: _safe_package_version(name)
            for name in ["torch", "numpy", "faiss-cpu", "rank-bm25", "sentence-transformers", "PyYAML"]
        },
        "random_seed": random_seed,
        "retrieval_configurations": configurations,
        "graded_relevance_preserved": all("gt_weights" in item for item in final_results),
    }
    with open(run_dir / "run_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"BASELINE RUN: {run_id}")
    print(f"query_count={len(selected_queries)} corpus_count={corpus_count}")
    print(f"query_representation={query_representation} document_representation={document_representation}")
    print(f"temporal_filter={temporal_policy} embedding_dimension={embedding_dimension}")
    print(f"retrieval_configurations={configurations}")
    print(f"manifest={run_dir / 'run_manifest.json'}")
    return results_path

# ==========================================
# 2. EVALUATION PIPELINE HELPERS & AUDIT
# ==========================================

def load_gt_to_key_map():
    config_path = Path(__file__).parent.parent.parent / "dataLaw" / "laws_config.json"
    mapping = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for entry in config:
                    key = entry.get("key")
                    if key:
                        mapping[key] = key
                        if "luat_hon_nhan_va_gia_dinh" in key:
                            mapping[key.replace("_va_gia_dinh", "_gia_dinh")] = key
                        elif "luat_hon_nhan_gia_dinh" in key:
                            mapping[key.replace("_gia_dinh", "_va_gia_dinh")] = key
        except Exception as e:
            print(f"⚠️ Lỗi đọc laws_config.json khi dựng GT_TO_KEY_MAP: {e}")
    return mapping

GT_TO_KEY_MAP = load_gt_to_key_map()

def load_key_to_name_map():
    config_path = Path(__file__).parent.parent.parent / "dataLaw" / "laws_config.json"
    mapping = {}
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for entry in config:
                    key = entry.get("key")
                    law_name = entry.get("law_name")
                    if key and law_name:
                        mapping[key] = law_name
        except Exception as e:
            print(f"⚠️ Lỗi đọc laws_config.json khi dựng KEY_TO_NAME_MAP: {e}")
    return mapping

KEY_TO_NAME_MAP = load_key_to_name_map()

def map_law_code_to_family(law_code):
    lc = str(law_code).lower().strip()
    if "bo_luat_dan_su" in lc or "luat_dan_su" in lc:
        return "Civil Code"
    elif "to_tung_dan_su" in lc:
        return "Civil Procedure Code"
    elif "dat_dai" in lc:
        return "Land Law"
    elif "hon_nhan" in lc or "gia_dinh" in lc:
        return "Marriage & Family Law"
    else:
        return "Others"

def run_ground_truth_audit(queries):
    print("\n" + "="*90)
    print(" SECTION 1: GROUND-TRUTH COVERAGE AUDIT & STATISTICS")
    print("="*90)
    
    lengths = []
    gts_counts = []
    total_explicit = 0
    total_inferred = 0
    total_substantive = 0
    total_procedural = 0
    
    law_code_counts = {}
    
    for q in queries:
        text = q.get("query", "")
        lengths.append(len(text.split()))
        
        gt = q.get("ground_truth", {})
        sub, proc = extract_gt_sub_proc(gt)
        
        total_substantive += len(sub)
        total_procedural += len(proc)
        gts_counts.append(len(sub) + len(proc))
        
        for item in sub + proc:
            s_type = item.get("source_type", "explicit")
            if s_type == "explicit":
                total_explicit += 1
            else:
                total_inferred += 1
                
            code = item.get("law_code", "unknown")
            law_code_counts[code] = law_code_counts.get(code, 0) + 1
            
    total_items = total_substantive + total_procedural
    explicit_ratio = total_explicit / total_items if total_items else 0.0
    inferred_ratio = total_inferred / total_items if total_items else 0.0
    
    len_p30 = float(np.percentile(lengths, 30))
    len_p70 = float(np.percentile(lengths, 70))
    gts_p30 = float(np.percentile(gts_counts, 30))
    gts_p70 = float(np.percentile(gts_counts, 70))
    
    print(f" Tổng số câu hỏi (Queries): {len(queries)}")
    print(f" Độ dài câu hỏi (Số từ): Min={min(lengths)}, Max={max(lengths)}, Trung bình={np.mean(lengths):.2f}")
    print(f" Phân vị độ dài câu hỏi: P30={len_p30:.1f}, P70={len_p70:.1f}")
    print(f" Số lượng nhãn đúng/câu hỏi: Min={min(gts_counts)}, Max={max(gts_counts)}, Trung bình={np.mean(gts_counts):.2f}")
    print(f" Phân vị số lượng nhãn đúng: P30={gts_p30:.1f}, P70={gts_p70:.1f}")
    print(f" Tổng số điều luật nhãn đúng: {total_items}")
    print(f"   - Luật nội dung (Substantive): {total_substantive} ({total_substantive/total_items*100:.1f}%)")
    print(f"   - Luật tố tụng (Procedural): {total_procedural} ({total_procedural/total_items*100:.1f}%)")
    print(f"   - Trực tiếp (Explicit): {total_explicit} ({explicit_ratio*100:.1f}%)")
    print(f"   - Suy luận (Inferred): {total_inferred} ({inferred_ratio*100:.1f}%)")
    print(" Phân phối theo văn bản luật (Law Code distribution):")
    for code, count in sorted(law_code_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {code}: {count} điều ({count/total_items*100:.1f}%)")
    print("="*90 + "\n")
    
    return len_p30, len_p70, gts_p30, gts_p70

VIETNAMESE_STOPWORDS = {
    "và", "của", "tại", "là", "bị", "được", "tòa", "án", "nhân", "dân", "theo", "cho", "các", "những", 
    "trong", "có", "cơ", "quan", "nhà", "nước", "đã", "đang", "sẽ", "phải", "cần", "để", "ra", "với", 
    "về", "từ", "lên", "vào", "đối", "tự", "lại", "này", "kia", "đó", "nào", "nơi", "khi", "như", 
    "một", "hai", "ba", "bốn", "năm", "sau", "trước", "dưới", "trên", "ngoại", "trừ", "thì", "mà", 
    "sự", "việc", "vụ", "cứ", "căn", "bản", "bản_án", "quyết_định", "nghị_định", "thông_tư", 
    "pháp_luật", "quy_định", "hội_đồng", "xét_xử"
}

def clean_and_tokenize(text):
    if not text:
        return set()
    cleaned = re.sub(r'[^\w\s]', ' ', text.lower())
    tokens = cleaned.split()
    cleaned_tokens = set()
    for t in tokens:
        sub_tokens = t.split('_')
        for st in sub_tokens:
            if st and st not in VIETNAMESE_STOPWORDS and not st.isdigit():
                cleaned_tokens.add(st)
    return cleaned_tokens

def get_hierarchy_type(std_id):
    std_id_lower = std_id.lower()
    if any(x in std_id_lower for x in ["nghi_dinh", "nd_cp", "nd-cp", "decree"]):
        return "decree"
    elif any(x in std_id_lower for x in ["thong_tu", "circular"]):
        return "circular"
    elif any(x in std_id_lower for x in ["quyet_dinh", "decision"]):
        return "decision"
    elif "bo_luat" in std_id_lower:
        return "code"
    elif "luat" in std_id_lower:
        return "law"
    return "other"

def parse_standard_id(std_id):
    parts = std_id.lower().strip().split()
    if len(parts) < 3:
        return None, None, None
    article = ""
    law_code = ""
    if parts[0] == "điều":
        article = " ".join(parts[:2])
        law_code = " ".join(parts[2:])
    else:
        article = parts[0]
        law_code = " ".join(parts[1:])
        
    year_match = re.search(r'\b(19|20)\d{2}\b', law_code)
    year = int(year_match.group(0)) if year_match else None
    
    base_law = law_code
    if year:
        base_law = re.sub(r'\b' + str(year) + r'\b', '', base_law)
    base_law = base_law.replace("_", " ").strip()
    base_law = " ".join(base_law.split())
    
    return article, base_law, year

def map_law_code_to_family(law_code):
    lc = str(law_code).lower().strip()
    if "bo_luat_dan_su" in lc or "luat_dan_su" in lc or "bds" in lc:
        return "Civil Code"
    elif "to_tung_dan_su" in lc or "ttds" in lc:
        return "Civil Procedure Code"
    elif "dat_dai" in lc or "dd" in lc or "nghi_dinh_43" in lc or "nghi_dinh_01_2017" in lc:
        return "Land Law"
    elif "hon_nhan" in lc or "gia_dinh" in lc:
        return "Marriage & Family Law"
    elif "nghi_dinh_163" in lc or "nghi_dinh_17" in lc:
        return "Civil Code"
    else:
        return "Others"

def classify_query_profile(query_text, gt_sub, gt_proc, len_p70, gts_p70, max_w_len, max_total_gts):
    q = query_text.lower()
    words = q.split()
    w_len = len(words)
    total_gts = len(gt_sub) + len(gt_proc)
    
    has_citation = bool(re.search(r"\b(điều\s+\d+|khoản\s+\d+)\b", q))
    if has_citation:
        return {
            "profile": "citation_exact",
            "confidence": 1.0,
            "source": "heuristic"
        }
        
    if w_len > len_p70:
        conf = 0.5 + 0.5 * (w_len - len_p70) / (max_w_len - len_p70) if max_w_len > len_p70 else 1.0
        return {
            "profile": "long_narrative",
            "confidence": float(round(min(1.0, conf), 4)),
            "source": "heuristic"
        }
        
    proc_ratio = len(gt_proc) / total_gts if total_gts > 0 else 0.0
    if proc_ratio >= 0.5:
        return {
            "profile": "procedural_focus",
            "confidence": float(round(proc_ratio, 4)),
            "source": "heuristic"
        }
        
    if total_gts >= gts_p70:
        conf = 0.5 + 0.5 * (total_gts - gts_p70) / (max_total_gts - gts_p70) if max_total_gts > gts_p70 else 1.0
        return {
            "profile": "mixed_multi_issue",
            "confidence": float(round(min(1.0, conf), 4)),
            "source": "heuristic"
        }
        
    legal_terms = ["đương sự", "bị đơn", "nguyên đơn", "kháng nghị", "tòa án", "hội đồng xét xử", "viện kiểm sát", "án lệ", "hợp đồng", "bồi thường", "thừa kế", "ly hôn"]
    legal_term_matches = sum(1 for term in legal_terms if term in q)
    if legal_term_matches <= 3:
        conf = 1.0 - (legal_term_matches / 4.0)
        return {
            "profile": "factual_narrative",
            "confidence": float(round(conf, 4)),
            "source": "heuristic"
        }
        
    return {
        "profile": "doctrinal_semantic",
        "confidence": 0.6,
        "source": "heuristic"
    }

def classify_difficulty_level(ds, ds_p33, ds_p67):
    if ds <= ds_p33:
        return "easy"
    elif ds <= ds_p67:
        return "medium"
    else:
        return "hard"

def classify_retrieval_error(query_text, missed_gts, missed_gts_raw, top_k_docs, id_to_doc, route, active_law_codes, judgment_year):
    if not missed_gts:
        return None
        
    # 1. Check for temporal mismatch first: same article, same base law, but different year
    for gt in missed_gts:
        gt_article, gt_base_law, gt_year = parse_standard_id(gt)
        if not gt_article or not gt_base_law:
            continue
        for doc in top_k_docs:
            doc_article, doc_base_law, doc_year = parse_standard_id(doc["standard_law_id"])
            if doc_article == gt_article and doc_base_law == gt_base_law and doc_year != gt_year:
                return "temporal_mismatch"
                
    # 2. Check for hierarchy confusion: expected decree/circular but retrieved law/code of the same family (or vice versa)
    for gt in missed_gts_raw:
        gt_h = get_hierarchy_type(gt)
        gt_family = map_law_code_to_family(gt.split()[-1]) if gt.split() else "Others"
        for doc in top_k_docs:
            doc_id = doc["standard_law_id"]
            doc_h = get_hierarchy_type(doc_id)
            doc_family = map_law_code_to_family(doc_id.split()[-1]) if doc_id.split() else "Others"
            if gt_family == doc_family and gt_family != "Others":
                if (gt_h in ["decree", "circular", "decision"] and doc_h in ["code", "law"]) or \
                   (gt_h in ["code", "law"] and doc_h in ["decree", "circular", "decision"]):
                    return "hierarchy_confusion"
                
    # 3. Check for procedural confusion
    retrieved_domains = []
    for doc in top_k_docs:
        std_id = doc["standard_law_id"]
        if std_id in id_to_doc:
            retrieved_domains.append(id_to_doc[std_id]["law_domain"])
    if retrieved_domains:
        sub_ratio = retrieved_domains.count("substance") / len(retrieved_domains)
        proc_ratio = retrieved_domains.count("procedural") / len(retrieved_domains)
        if route == "procedural" and sub_ratio > 0.6:
            return "procedural_confusion"
        if route == "substance" and proc_ratio > 0.6:
            return "procedural_confusion"
            
    # 4. Check for multi-issue dilution
    missed_families = set(map_law_code_to_family(g.split()[-1]) for g in missed_gts if g.split())
    ret_families = set(map_law_code_to_family(d["standard_law_id"].split()[-1]) for d in top_k_docs if d["standard_law_id"].split())
    if len(missed_families) > 1 and len(ret_families) == 1:
        return "multi_issue_dilution"
        
    # 5. Check for lexical miss (using stopwords removal and jaccard < 0.03)
    query_tokens = clean_and_tokenize(query_text)
    for gt in missed_gts:
        gt_lower = gt.lower().strip()
        if gt_lower in id_to_doc:
            doc_tokens = clean_and_tokenize(id_to_doc[gt_lower]["bm25_text"])
            if doc_tokens:
                jaccard = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
                if jaccard < 0.03:
                    return "lexical_miss"
                    
    # 6. Default is semantic drift
    return "semantic_drift"


def fuse_pools(bm25_pool, faiss_pool, w_bm25, fusion_type="minmax", temp_bm25=None, temp_faiss=None):
    if temp_bm25 is None:
        temp_bm25 = _eval_config.get('fusion', {}).get('temperature_bm25', 5.0)
    if temp_faiss is None:
        temp_faiss = _eval_config.get('fusion', {}).get('temperature_faiss', 0.5)

    w_faiss = 1.0 - w_bm25
    union_ids = list(set([d["standard_law_id"] for d in bm25_pool] + [d["standard_law_id"] for d in faiss_pool]))
    if not union_ids:
        return []
        
    bm25_map = {d["standard_law_id"]: d["score"] for d in bm25_pool}
    faiss_map = {d["standard_law_id"]: d["score"] for d in faiss_pool}
    
    if fusion_type == "minmax":
        bm25_scores = [d["score"] for d in bm25_pool]
        faiss_scores = [d["score"] for d in faiss_pool]
        
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        
        faiss_min = min(faiss_scores) if faiss_scores else 0.0
        faiss_max = max(faiss_scores) if faiss_scores else 1.0
        faiss_range = faiss_max - faiss_min if faiss_max != faiss_min else 1.0
        
        combined = []
        for std_id in union_ids:
            s_bm25 = (bm25_map[std_id] - bm25_min) / bm25_range if std_id in bm25_map else 0.0
            s_faiss = (faiss_map[std_id] - faiss_min) / faiss_range if std_id in faiss_map else 0.0
            score = w_bm25 * s_bm25 + w_faiss * s_faiss
            combined.append((std_id, score))
            
    elif fusion_type == "softmax":
        bm25_min = min([d["score"] for d in bm25_pool]) if bm25_pool else 0.0
        faiss_min = min([d["score"] for d in faiss_pool]) if faiss_pool else 0.0
        
        bm25_scores = [bm25_map.get(std_id, bm25_min) for std_id in union_ids]
        faiss_scores = [faiss_map.get(std_id, faiss_min) for std_id in union_ids]
        
        bm25_norm = softmax_normalize(bm25_scores, temperature=temp_bm25)
        faiss_norm = softmax_normalize(faiss_scores, temperature=temp_faiss)
        
        combined = []
        for idx, std_id in enumerate(union_ids):
            score = w_bm25 * bm25_norm[idx] + w_faiss * faiss_norm[idx]
            combined.append((std_id, score))
            
    elif fusion_type == "rrf":
        bm25_ranks = {d["standard_law_id"]: r for r, d in enumerate(bm25_pool, 1)}
        faiss_ranks = {d["standard_law_id"]: r for r, d in enumerate(faiss_pool, 1)}
        
        combined = []
        for std_id in union_ids:
            r_bm25 = bm25_ranks.get(std_id, None)
            r_faiss = faiss_ranks.get(std_id, None)
            val_bm25 = 1.0 / (60.0 + r_bm25) if r_bm25 is not None else 0.0
            val_faiss = 1.0 / (60.0 + r_faiss) if r_faiss is not None else 0.0
            score = w_bm25 * val_bm25 + w_faiss * val_faiss
            combined.append((std_id, score))
    else:
        combined = []
        
    combined.sort(key=lambda x: x[1], reverse=True)
    return combined

def determine_failure_mode(bm25_recall, faiss_recall, static_opt_recall, dynamic_recall, oracle_recall):
    if oracle_recall == 0.0:
        return "retriever_failure"
    elif dynamic_recall < static_opt_recall:
        return "router_failure"
    elif static_opt_recall < max(bm25_recall, faiss_recall):
        return "fusion_failure"
    else:
        return "neutral"

def run_evaluation_pipeline(temporal_filter="on", config_path=None, max_queries=None):
    print("--- BẮT ĐẦU ĐÁNH GIÁ RAG (ADVANCED EXPERIMENTAL FRAMEWORK) ---")
    
    # 1. Load config file
    if config_path is None:
        config_path = Path(__file__).parent / "eval_config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        requested_config = yaml.safe_load(f)
    if requested_config.get("evaluation", {}).get("mode") == BASELINE_MODE:
        return run_baseline_evaluation_pipeline(config_path, max_queries=max_queries)
    
    print(f"[*] Loading evaluation config from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        eval_config = yaml.safe_load(f)
        
    global ALPHA_CORE, ALPHA_SUPPORT
    eval_section = eval_config.get('evaluation', {})
    ALPHA_CORE = eval_section.get('alpha_core', 3)
    ALPHA_SUPPORT = eval_section.get('alpha_support', 1)
    print(f"Loaded alpha_core = {ALPHA_CORE}")
    print(f"Loaded alpha_support = {ALPHA_SUPPORT}")
        
    K = eval_config['evaluation']['k']
    POOL_K = eval_config['evaluation']['pool_k']
    
    embedding_model_name = eval_config['models']['embedding_model']
    rerank_model_name = eval_config['models']['rerank_model']
    
    temp_bm25 = eval_config['fusion']['temperature_bm25']
    temp_faiss = eval_config['fusion']['temperature_faiss']
    
    dh_weights = eval_config['dynamic_hybrid']['weights']
    exact_citation_w = dh_weights.get('exact_citation', 0.9)
    procedural_w = dh_weights.get('procedural', 0.8)
    substance_w = dh_weights.get('substance', 0.2)
    mixed_w = dh_weights.get('mixed', 0.5)
    
    project_root = Path(__file__).parent.parent.parent.parent
    
    eval_file_rel = eval_config['paths']['eval_file']
    eval_file = project_root / eval_file_rel if not Path(eval_file_rel).is_absolute() else Path(eval_file_rel)
    
    laws_config_rel = eval_config['paths']['laws_config']
    laws_config_path = project_root / laws_config_rel if not Path(laws_config_rel).is_absolute() else Path(laws_config_rel)
    
    output_dir_rel = eval_config['paths']['output_dir']
    output_dir = project_root / output_dir_rel if not Path(output_dir_rel).is_absolute() else Path(output_dir_rel)
    BASE_DIR = output_dir


    # 2. Schema Validation check
    try:
        # Resolve imports relatively
        sys.path.append(str(Path(__file__).parent.resolve()))
        from validate_benchmark import validate_benchmark_file
        validate_benchmark_file(eval_file)
    except Exception as e:
        print(f"❌ LỖI VÀ XÁC THỰC BENCHMARK THẤT BẠI: {e}")
        sys.exit(1)
    
    law_files = []
    if laws_config_path.exists():
        try:
            with open(laws_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                for entry in config:
                    key = entry["key"]
                    f_path = output_dir / f"standardized_{key}.json"
                    if f_path.exists():
                        law_files.append(f_path)
                    else:
                        print(f"    ⚠️ Chưa có file chuẩn hóa: {entry['law_name']} ({f_path.name})")
        except Exception as e:
            print(f"⚠️ Lỗi đọc laws_config.json: {e}")
            
    if not law_files:
        law_files = [p for p in output_dir.glob("standardized_*.json") if "copy" not in p.name]

    
    documents = load_documents(law_files)
    print(f"   -> Tổng số điều luật: {len(documents)}")
    with open(eval_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)

    # Perform Ground-Truth Coverage Audit and get percentiles
    len_p30, len_p70, gts_p30, gts_p70 = run_ground_truth_audit(queries)

    # Precompute difficulty scores and find percentiles
    query_word_lengths = []
    query_gt_counts = []
    for q_item in queries:
        query_word_lengths.append(len(q_item.get('query', '').split()))
        gt_obj = q_item.get('ground_truth', {})
        sub_list, proc_list = extract_gt_sub_proc(gt_obj)
        query_gt_counts.append(len(sub_list) + len(proc_list))
        
    max_w_len = max(query_word_lengths) if query_word_lengths else 1
    max_total_gts = max(query_gt_counts) if query_gt_counts else 1
    
    difficulty_scores = []
    for w_len, total_gts in zip(query_word_lengths, query_gt_counts):
        ds = 0.5 * (w_len / max_w_len) + 0.5 * (total_gts / max_total_gts)
        difficulty_scores.append(ds)
        
    ds_p33 = float(np.percentile(difficulty_scores, 33.33)) if difficulty_scores else 0.0
    ds_p67 = float(np.percentile(difficulty_scores, 66.67)) if difficulty_scores else 0.0

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   -> Thiết bị phần cứng phát hiện: {device.upper()}")

    # Models init
    try:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [doc['bm25_text'].lower().split() for doc in documents]
        bm25_index = BM25Okapi(tokenized_corpus)
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo BM25: {e}")
        bm25_index = None

    try:
        from sentence_transformers import SentenceTransformer
        import faiss
        embedder = SentenceTransformer(embedding_model_name, device=device)
        corpus_embeddings = embedder.encode([doc['vector_text'] for doc in documents], show_progress_bar=False)
        faiss_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        faiss.normalize_L2(corpus_embeddings)
        faiss_index.add(corpus_embeddings)
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo FAISS: {e}")
        embedder = faiss_index = None

    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(rerank_model_name, device=device)
    except Exception as e:
        print(f"⚠️ Lỗi khởi tạo PhoRanker Reranker: {e}")
        reranker = None

    try:
        from pyvi import ViTokenizer
        use_pyvi = True
    except:
        use_pyvi = False

    id_to_doc = {doc['standard_law_id']: doc for doc in documents}

    
    ACTIVE_LAW_CODES = set()
    for file_path in law_files:
        key = Path(file_path).name.replace("standardized_", "").replace(".json", "")
        ACTIVE_LAW_CODES.add(key)

    # Warmup Phase (3 queries)
    if queries and (bm25_index or faiss_index or reranker):
        print("\n[*] Khởi động (Warmup) 3 query để ổn định RAM/GPU...")
        warmup_queries = queries[:3]
        for q_item in warmup_queries:
            q_clean = clean_query(q_item.get('query', ''))
            if bm25_index:
                q_seg = ViTokenizer.tokenize(q_clean) if use_pyvi else q_clean
                _ = bm25_index.get_scores(q_seg.lower().split())
            if embedder and faiss_index:
                q_emb = embedder.encode([q_clean], show_progress_bar=False)
                faiss.normalize_L2(q_emb)
                _, _ = faiss_index.search(q_emb, 10)
            if reranker and documents:
                _ = reranker.predict([(q_clean, documents[0]["vector_text"])], show_progress_bar=False)
        print("✅ Khởi động hoàn tất. Bắt đầu đánh giá đo đạc thực tế...")

    # Temporal Validation comparison counters
    total_future_laws_removed = 0
    total_affected_queries = 0
    total_mismatch_before = 0
    total_mismatch_after = 0
    
    on_recall_hits = 0.0
    off_recall_hits = 0.0
    total_recall_valid_queries = 0

    raw_query_records = []

    
    for i, q_item in enumerate(queries):
        query_text = q_item.get('query', '')
        
        gt_obj = q_item.get('ground_truth', {})
        sub_list, proc_list = extract_gt_sub_proc(gt_obj)
        gt_weights = get_gt_weights(gt_obj)
        
        # Core & Supporting splits
        core_obj = gt_obj.get('core', [])
        supp_obj = gt_obj.get('supporting', [])
        if isinstance(core_obj, list) and isinstance(supp_obj, list):
            core_list = core_obj
            supp_list = supp_obj
        else:
            core_dict = core_obj if isinstance(core_obj, dict) else {}
            supp_dict = supp_obj if isinstance(supp_obj, dict) else {}
            core_list = core_dict.get('substantive', []) + core_dict.get('procedural', [])
            supp_list = supp_dict.get('substantive', []) + supp_dict.get('procedural', [])
        
        def canonical_gt_id(item):
            return f"{item.get('article', '')} {canonicalize_law_code(item.get('law_code', ''))}".strip()

        def is_active_gt(item):
            return canonicalize_law_code(item.get('law_code', '')) in ACTIVE_LAW_CODES

        core_gt_raw = [canonical_gt_id(item) for item in core_list]
        supporting_gt_raw = [canonical_gt_id(item) for item in supp_list]
        core_gt_filtered = [canonical_gt_id(item) for item in core_list if is_active_gt(item)]
        supporting_gt_filtered = [canonical_gt_id(item) for item in supp_list if is_active_gt(item)]

        
        # Explicit vs Inferred splits (Unfiltered)
        explicit_gts_raw = []
        inferred_gts_raw = []
        for item in sub_list + proc_list:
            std_val = canonical_gt_id(item)
            if item.get('source_type') == 'explicit':
                explicit_gts_raw.append(std_val)
            else:
                inferred_gts_raw.append(std_val)

        # Raw (Unfiltered) Ground Truth
        sub_gt_raw = [canonical_gt_id(item) for item in sub_list]
        proc_gt_raw = [canonical_gt_id(item) for item in proc_list]
        
        # Filtered Ground Truth
        sub_list_filtered = [item for item in sub_list if is_active_gt(item)]
        proc_list_filtered = [item for item in proc_list if is_active_gt(item)]
        sub_gt_filtered = [canonical_gt_id(item) for item in sub_list_filtered]
        proc_gt_filtered = [canonical_gt_id(item) for item in proc_list_filtered]
        
        # Explicit vs Inferred splits (Filtered)
        explicit_gts_filtered = []
        inferred_gts_filtered = []
        for item in sub_list_filtered + proc_list_filtered:
            std_val = canonical_gt_id(item)
            if item.get('source_type') == 'explicit':
                explicit_gts_filtered.append(std_val)
            else:
                inferred_gts_filtered.append(std_val)
        
        if not sub_gt_raw and not proc_gt_raw: 
            continue
            
        # Classify query profile and difficulty dynamically using computed percentiles
        # Priority 1: Annotated source
        annotated_profile = q_item.get("query_profile")
        if not annotated_profile and "metadata" in q_item:
            annotated_profile = q_item["metadata"].get("query_profile")
            
        if annotated_profile:
            if isinstance(annotated_profile, dict):
                query_profile = {
                    "profile": annotated_profile.get("profile", "doctrinal_semantic"),
                    "confidence": float(annotated_profile.get("confidence", 1.0)),
                    "source": annotated_profile.get("source", "annotation")
                }
            else:
                query_profile = {
                    "profile": str(annotated_profile),
                    "confidence": 1.0,
                    "source": "annotation"
                }
        else:
            query_profile = classify_query_profile(query_text, sub_gt_raw, proc_gt_raw, len_p70, gts_p70, max_w_len, max_total_gts)
            
        annotated_difficulty = q_item.get("difficulty")
        if not annotated_difficulty and "metadata" in q_item:
            annotated_difficulty = q_item["metadata"].get("difficulty")
            
        if annotated_difficulty:
            if isinstance(annotated_difficulty, dict):
                difficulty = {
                    "level": annotated_difficulty.get("level", "medium"),
                    "score": float(annotated_difficulty.get("score", 0.5))
                }
            else:
                level_str = str(annotated_difficulty)
                score_val = 0.2 if level_str == "easy" else 0.8 if level_str == "hard" else 0.5
                difficulty = {
                    "level": level_str,
                    "score": score_val
                }
        else:
            ds = difficulty_scores[i]
            diff_level = classify_difficulty_level(ds, ds_p33, ds_p67)
            difficulty = {
                "level": diff_level,
                "score": float(round(ds, 4))
            }
        
        metadata = q_item.get('metadata', {})
        judgment_year = metadata.get('year')
        dispute_year = metadata.get('dispute_year')
        target_year = dispute_year if dispute_year is not None else judgment_year
        if target_year:
            try:
                target_year = int(target_year)
            except ValueError:
                target_year = None
                
        query_clean = clean_query(query_text)
        route = get_query_intent(query_clean)
        
        # Routing weight for Dynamic Hybrid
        w_bm25_dynamic = get_dynamic_weight(route, dh_weights)

        # Latency breakdown measurements & temporal stats
        encode_time_ms = 0.0
        bm25_retrieval_time_ms = 0.0
        faiss_retrieval_time_ms = 0.0
        fusion_time_ms = 0.0
        rerank_time_ms = 0.0
        
        # 1. BM25 Search (OFF mode - no temporal filtering)
        bm25_top_pool_off = []
        if bm25_index:
            q_segmented = ViTokenizer.tokenize(query_clean) if use_pyvi else query_clean
            doc_scores = bm25_index.get_scores(q_segmented.lower().split())
            valid_scores_off = []
            for idx, score in enumerate(doc_scores):
                doc = documents[idx]
                if judgment_year and doc["law_year"] and doc["law_year"] > judgment_year:
                    continue
                valid_scores_off.append((idx, score))
            valid_scores_off.sort(key=lambda x: x[1], reverse=True)
            for rank, (idx, score) in enumerate(valid_scores_off[:POOL_K], 1):
                bm25_top_pool_off.append({"rank": rank, "standard_law_id": documents[idx]['standard_law_id'], "score": float(score)})

        # 2. BM25 Search (ON mode - with temporal filtering)
        bm25_top_pool_on = []
        skipped_future_laws = set()
        if bm25_index:
            start_bm25_ret = time.perf_counter()
            q_segmented = ViTokenizer.tokenize(query_clean) if use_pyvi else query_clean
            doc_scores = bm25_index.get_scores(q_segmented.lower().split())
            valid_scores_on = []
            for idx, score in enumerate(doc_scores):
                doc = documents[idx]
                if target_year:
                    active_start = doc.get("active_start", 1900)
                    active_end = doc.get("active_end", 2100)
                    if not (active_start <= target_year <= active_end):
                        if doc["law_year"] and doc["law_year"] > target_year:
                            skipped_future_laws.add(doc["standard_law_id"])
                        continue
                elif judgment_year and doc["law_year"] and doc["law_year"] > judgment_year:
                    continue
                valid_scores_on.append((idx, score))
            valid_scores_on.sort(key=lambda x: x[1], reverse=True)
            for rank, (idx, score) in enumerate(valid_scores_on[:POOL_K], 1):
                bm25_top_pool_on.append({"rank": rank, "standard_law_id": documents[idx]['standard_law_id'], "score": float(score)})
            bm25_retrieval_time_ms = (time.perf_counter() - start_bm25_ret) * 1000

        # 3. FAISS Encoding (Runs once for both modes)
        q_emb = None
        if embedder and faiss_index:
            start_encode = time.perf_counter()
            q_emb = embedder.encode([query_clean], show_progress_bar=False)
            faiss.normalize_L2(q_emb)
            encode_time_ms = (time.perf_counter() - start_encode) * 1000

        # 4. FAISS Search (OFF mode - no temporal filtering)
        faiss_top_pool_off = []
        if embedder and faiss_index and q_emb is not None:
            scores, indices = faiss_index.search(q_emb, POOL_K * 10)
            rank = 1
            for score, idx in zip(scores[0], indices[0]):
                doc = documents[idx]
                if judgment_year and doc["law_year"] and doc["law_year"] > judgment_year:
                    continue
                faiss_top_pool_off.append({"rank": rank, "standard_law_id": doc['standard_law_id'], "score": float(score)})
                rank += 1
                if len(faiss_top_pool_off) == POOL_K:
                    break

        # 5. FAISS Search (ON mode - with temporal filtering)
        faiss_top_pool_on = []
        if embedder and faiss_index and q_emb is not None:
            start_faiss_ret = time.perf_counter()
            scores, indices = faiss_index.search(q_emb, POOL_K * 10)
            rank = 1
            for score, idx in zip(scores[0], indices[0]):
                doc = documents[idx]
                if target_year:
                    active_start = doc.get("active_start", 1900)
                    active_end = doc.get("active_end", 2100)
                    if not (active_start <= target_year <= active_end):
                        if doc["law_year"] and doc["law_year"] > target_year:
                            skipped_future_laws.add(doc["standard_law_id"])
                        continue
                elif judgment_year and doc["law_year"] and doc["law_year"] > judgment_year:
                    continue
                faiss_top_pool_on.append({"rank": rank, "standard_law_id": doc['standard_law_id'], "score": float(score)})
                rank += 1
                if len(faiss_top_pool_on) == POOL_K:
                    break
            faiss_retrieval_time_ms = (time.perf_counter() - start_faiss_ret) * 1000

        # Calculate future laws removed
        future_laws_removed_query = len(skipped_future_laws)

        # 6. Fusion OFF mode
        c2_top_pool_off = []
        if bm25_top_pool_off and faiss_top_pool_off:
            combined_c2_off = fuse_pools(bm25_top_pool_off, faiss_top_pool_off, w_bm25_dynamic, "minmax")
            for rank, (std_id, score) in enumerate(combined_c2_off[:POOL_K], 1):
                c2_top_pool_off.append({"rank": rank, "standard_law_id": std_id, "score": float(score)})

        # 7. Fusion ON mode
        c2_top_pool_on = []
        if bm25_top_pool_on and faiss_top_pool_on:
            start_fusion = time.perf_counter()
            combined_c2_on = fuse_pools(bm25_top_pool_on, faiss_top_pool_on, w_bm25_dynamic, "minmax")
            for rank, (std_id, score) in enumerate(combined_c2_on[:POOL_K], 1):
                c2_top_pool_on.append({"rank": rank, "standard_law_id": std_id, "score": float(score)})
            fusion_time_ms = (time.perf_counter() - start_fusion) * 1000

        # 8. Rerank OFF mode
        d_top_pool_off = []
        if c2_top_pool_off and reranker:
            pairs_off = [(query_clean, id_to_doc[doc["standard_law_id"]]["vector_text"]) for doc in c2_top_pool_off if doc["standard_law_id"] in id_to_doc]
            valid_docs_off = [doc["standard_law_id"] for doc in c2_top_pool_off if doc["standard_law_id"] in id_to_doc]
            if pairs_off:
                rerank_scores_off = [float(s) for s in reranker.predict(pairs_off, show_progress_bar=False)]
                c2_scores_off = [doc["score"] for doc in c2_top_pool_off if doc["standard_law_id"] in id_to_doc]
                c2_min = min(c2_scores_off) if c2_scores_off else 0.0
                c2_max = max(c2_scores_off) if c2_scores_off else 1.0
                c2_range = c2_max - c2_min if c2_max != c2_min else 1.0
                c2_norm = [(s - c2_min) / c2_range for s in c2_scores_off]
                
                rr_min = min(rerank_scores_off) if rerank_scores_off else 0.0
                rr_max = max(rerank_scores_off) if rerank_scores_off else 1.0
                rr_range = rr_max - rr_min if rr_max != rr_min else 1.0
                rr_norm = [(s - rr_min) / rr_range for s in rerank_scores_off]
                
                combined_scores_off = []
                for idx, std_id in enumerate(valid_docs_off):
                    score = 0.5 * c2_norm[idx] + 0.5 * rr_norm[idx]
                    combined_scores_off.append((std_id, score))
                combined_scores_off.sort(key=lambda x: x[1], reverse=True)
                for rank, (std_id, score) in enumerate(combined_scores_off[:K], 1):
                    d_top_pool_off.append({"rank": rank, "standard_law_id": std_id, "score": float(score)})

        # 9. Rerank ON mode (with caching optimization if pool is identical)
        d_top_pool_on = []
        if c2_top_pool_on and reranker:
            on_ids = [d["standard_law_id"] for d in c2_top_pool_on]
            off_ids = [d["standard_law_id"] for d in c2_top_pool_off]
            if on_ids == off_ids:
                d_top_pool_on = list(d_top_pool_off)
                rerank_time_ms = 0.0
            else:
                start_rerank = time.perf_counter()
                pairs_on = [(query_clean, id_to_doc[doc["standard_law_id"]]["vector_text"]) for doc in c2_top_pool_on if doc["standard_law_id"] in id_to_doc]
                valid_docs_on = [doc["standard_law_id"] for doc in c2_top_pool_on if doc["standard_law_id"] in id_to_doc]
                if pairs_on:
                    rerank_scores_on = [float(s) for s in reranker.predict(pairs_on, show_progress_bar=False)]
                    c2_scores_on = [doc["score"] for doc in c2_top_pool_on if doc["standard_law_id"] in id_to_doc]
                    c2_min = min(c2_scores_on) if c2_scores_on else 0.0
                    c2_max = max(c2_scores_on) if c2_scores_on else 1.0
                    c2_range = c2_max - c2_min if c2_max != c2_min else 1.0
                    c2_norm = [(s - c2_min) / c2_range for s in c2_scores_on]
                    
                    rr_min = min(rerank_scores_on) if rerank_scores_on else 0.0
                    rr_max = max(rerank_scores_on) if rerank_scores_on else 1.0
                    rr_range = rr_max - rr_min if rr_max != rr_min else 1.0
                    rr_norm = [(s - rr_min) / rr_range for s in rerank_scores_on]
                    
                    combined_scores_on = []
                    for idx, std_id in enumerate(valid_docs_on):
                        score = 0.5 * c2_norm[idx] + 0.5 * rr_norm[idx]
                        combined_scores_on.append((std_id, score))
                    combined_scores_on.sort(key=lambda x: x[1], reverse=True)
                    for rank, (std_id, score) in enumerate(combined_scores_on[:K], 1):
                        d_top_pool_on.append({"rank": rank, "standard_law_id": std_id, "score": float(score)})
                rerank_time_ms = (time.perf_counter() - start_rerank) * 1000
        else:
            d_top_pool_on = list(d_top_pool_off)

        # 10. Select pools based on requested temporal_filter mode
        if temporal_filter == "on":
            bm25_top_pool = bm25_top_pool_on
            faiss_top_pool = faiss_top_pool_on
            c2_top_pool = c2_top_pool_on
            d_top_pool = d_top_pool_on
        else:
            bm25_top_pool = bm25_top_pool_off
            faiss_top_pool = faiss_top_pool_off
            c2_top_pool = c2_top_pool_off
            d_top_pool = d_top_pool_off

        # 11. Calculate stats comparing ON vs OFF modes
        on_top_k_ids = [d["standard_law_id"] for d in d_top_pool_on[:10]]
        off_top_k_ids = [d["standard_law_id"] for d in d_top_pool_off[:10]]
        affected_query = 1 if set(on_top_k_ids) != set(off_top_k_ids) else 0

        # Calculate mismatch before vs after in the top-10 retrieved sets
        overall_gts_filtered = sub_gt_filtered + proc_gt_filtered
        mismatch_before_query = sum(1 for ret_id in off_top_k_ids if check_temporal_mismatch(ret_id, overall_gts_filtered))
        mismatch_after_query = sum(1 for ret_id in on_top_k_ids if check_temporal_mismatch(ret_id, overall_gts_filtered))

        # Calculate pipeline latencies (Serial vs Parallel)
        lat_bm25 = bm25_retrieval_time_ms
        lat_faiss = encode_time_ms + faiss_retrieval_time_ms
        lat_hybrid_serial = encode_time_ms + bm25_retrieval_time_ms + faiss_retrieval_time_ms + fusion_time_ms
        lat_hybrid_parallel = max(encode_time_ms + faiss_retrieval_time_ms, bm25_retrieval_time_ms) + fusion_time_ms
        
        lat_rerank_serial = lat_hybrid_serial + rerank_time_ms
        lat_rerank_parallel = lat_hybrid_parallel + rerank_time_ms


        latencies = {
            "BM25": lat_bm25,
            "FAISS": lat_faiss,
            "Hybrid_Static_Fixed": lat_hybrid_serial,
            "Hybrid_Static_Opt": lat_hybrid_serial,
            "Hybrid_Dynamic": lat_hybrid_serial,
            "Hybrid_Dynamic_Rerank": lat_rerank_serial,
            "Hybrid_Dynamic_Parallel": lat_hybrid_parallel,
            "Hybrid_Dynamic_Rerank_Parallel": lat_rerank_parallel
        }

        latency_breakdown = {
            "encode_time_ms": float(round(encode_time_ms, 4)),
            "bm25_retrieval_time_ms": float(round(bm25_retrieval_time_ms, 4)),
            "faiss_retrieval_time_ms": float(round(faiss_retrieval_time_ms, 4)),
            "fusion_time_ms": float(round(fusion_time_ms, 4)),
            "rerank_time_ms": float(round(rerank_time_ms, 4)),
            "total_serial_time_ms": float(round(lat_rerank_serial, 4)),
            "total_parallel_time_ms": float(round(lat_rerank_parallel, 4))
        }

        # Accumulate G3 temporal comparison metrics
        total_future_laws_removed += future_laws_removed_query
        total_affected_queries += affected_query
        total_mismatch_before += mismatch_before_query
        total_mismatch_after += mismatch_after_query
        
        if overall_gts_filtered:
            on_hits = sum(1 for gt in overall_gts_filtered if any(is_match(r, gt) for r in on_top_k_ids))
            off_hits = sum(1 for gt in overall_gts_filtered if any(is_match(r, gt) for r in off_top_k_ids))
            on_recall_hits += on_hits / len(overall_gts_filtered)
            off_recall_hits += off_hits / len(overall_gts_filtered)
            total_recall_valid_queries += 1

        raw_query_records.append({

            "query_id": q_item.get('id', i),
            "judgment_name": q_item.get('judgment_name', ''),
            "case_type": q_item.get('metadata', {}).get('case_type', ''),
            "query": query_text,
            "route": route,
            "query_profile": query_profile,
            "difficulty": difficulty,
            "judgment_year": judgment_year,
            "explicit_gts_raw": explicit_gts_raw,
            "inferred_gts_raw": inferred_gts_raw,
            "explicit_gts_filtered": explicit_gts_filtered,
            "inferred_gts_filtered": inferred_gts_filtered,
            "substance_gt_raw": sub_gt_raw,
            "procedural_gt_raw": proc_gt_raw,
            "substance_gt_filtered": sub_gt_filtered,
            "procedural_gt_filtered": proc_gt_filtered,
            "core_gt_raw": core_gt_raw,
            "supporting_gt_raw": supporting_gt_raw,
            "core_gt_filtered": core_gt_filtered,
            "supporting_gt_filtered": supporting_gt_filtered,
            "future_law_removed": future_laws_removed_query,
            "affected_query": affected_query,
            "mismatch_before": mismatch_before_query,
            "mismatch_after": mismatch_after_query,
            "bm25_pool": bm25_top_pool,
            "faiss_pool": faiss_top_pool,
            "c2_pool": c2_top_pool,
            "d_pool": d_top_pool,
            "latencies": latencies,
            "latency_breakdown": latency_breakdown,
            "w_bm25_dynamic": w_bm25_dynamic,
            "gt_weights": gt_weights
        })


        
        sys.stdout.write(f"\r   Đã xử lý hồ sơ {i+1}/{len(queries)} queries (Tốc độ GPU)")
        sys.stdout.flush()
        
    print("\n   -> Hoàn tất xử lý thô. Bắt đầu Quét tham số tìm TH C1-opt trong bộ nhớ...")
    
    # Grid Search for TH C1-opt
    best_method, best_w, ablation_grid = run_in_memory_tuning_study(raw_query_records, k=10)
    print(f"   -> Kết quả tối ưu TH C1-opt: Method={best_method.upper()}, w_bm25={best_w}")

    # Build final results dict with all configurations
    final_results = []
    explainability_logs = []
    
    for item in raw_query_records:
        bm25_pool = item["bm25_pool"]
        faiss_pool = item["faiss_pool"]
        w_bm25_dynamic = item["w_bm25_dynamic"]
        
        # TH C1: Static Hybrid (w=0.5)
        c1_combined = fuse_pools(bm25_pool, faiss_pool, 0.5, "minmax")
        c1_top_k = [{"rank": rank, "standard_law_id": std_id, "score": float(score)} for rank, (std_id, score) in enumerate(c1_combined[:K], 1)]
        
        # TH C1-opt: Tuned Static Hybrid
        c1_opt_combined = fuse_pools(bm25_pool, faiss_pool, best_w, best_method)
        c1_opt_top_k = [{"rank": rank, "standard_law_id": std_id, "score": float(score)} for rank, (std_id, score) in enumerate(c1_opt_combined[:K], 1)]
        
        # Compute static latencies
        combine_latency_fixed = 0.5
        item["latencies"]["Hybrid_Static_Fixed"] = item["latencies"]["BM25"] + item["latencies"]["FAISS"] + combine_latency_fixed
        item["latencies"]["Hybrid_Static_Opt"] = item["latencies"]["BM25"] + item["latencies"]["FAISS"] + combine_latency_fixed

        metrics_dict = {
            "BM25": {"latency_ms": item["latencies"]["BM25"], "top_k": bm25_pool[:K]},
            "FAISS": {"latency_ms": item["latencies"]["FAISS"], "top_k": faiss_pool[:K]},
            "Hybrid_Static_Fixed": {"latency_ms": item["latencies"]["Hybrid_Static_Fixed"], "top_k": c1_top_k},
            "Hybrid_Static_Opt": {"latency_ms": item["latencies"]["Hybrid_Static_Opt"], "top_k": c1_opt_top_k},
            "Hybrid_Dynamic": {
                "latency_ms": item["latencies"]["Hybrid_Dynamic"], 
                "latency_parallel_ms": item["latencies"]["Hybrid_Dynamic_Parallel"],
                "top_k": item["c2_pool"][:K]
            },
            "Hybrid_Dynamic_Rerank": {
                "latency_ms": item["latencies"]["Hybrid_Dynamic_Rerank"],
                "latency_parallel_ms": item["latencies"]["Hybrid_Dynamic_Rerank_Parallel"],
                "top_k": item["d_pool"]
            }
        }
        
        # For router analysis (attributions)
        dyn_rec_ids = [doc["standard_law_id"] for doc in item["c2_pool"][:10]]
        opt_rec_ids = [doc["standard_law_id"] for doc in c1_opt_top_k[:10]]
        overall_gts = item["substance_gt_filtered"] + item["procedural_gt_filtered"]
        overall_gts_raw = item["substance_gt_raw"] + item["procedural_gt_raw"]
        
        dyn_hits = sum(1 for gt in overall_gts if any(is_match(r, gt) for r in dyn_rec_ids))
        opt_hits = sum(1 for gt in overall_gts if any(is_match(r, gt) for r in opt_rec_ids))
        dyn_rec10 = dyn_hits / len(overall_gts) if overall_gts else 0.0
        opt_rec10 = opt_hits / len(overall_gts) if overall_gts else 0.0
        
        if dyn_rec10 > opt_rec10:
            attribution = "dynamic_helped"
        elif dyn_rec10 < opt_rec10:
            attribution = "dynamic_hurt"
        else:
            attribution = "neutral"

        # Find oracle best weight for this query
        oracle_best_w = 0.5
        max_recall = -1.0
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            comb = fuse_pools(bm25_pool, faiss_pool, w, "minmax")
            comb_ids = [x[0] for x in comb[:10]]
            hits = sum(1 for gt in overall_gts if any(is_match(r, gt) for r in comb_ids))
            rec = hits / len(overall_gts) if overall_gts else 0.0
            if rec > max_recall:
                max_recall = rec
                oracle_best_w = w

        # Calculate query-level recalls for determining failure mode
        bm25_rec_ids = [doc["standard_law_id"] for doc in bm25_pool[:10]]
        bm25_hits = sum(1 for gt in overall_gts if any(is_match(r, gt) for r in bm25_rec_ids))
        bm25_rec10 = bm25_hits / len(overall_gts) if overall_gts else 0.0

        faiss_rec_ids = [doc["standard_law_id"] for doc in faiss_pool[:10]]
        faiss_hits = sum(1 for gt in overall_gts if any(is_match(r, gt) for r in faiss_rec_ids))
        faiss_rec10 = faiss_hits / len(overall_gts) if overall_gts else 0.0

        oracle_hits = sum(1 for gt in overall_gts if any(is_match(r, gt) for r in bm25_rec_ids) or any(is_match(r, gt) for r in faiss_rec_ids))
        oracle_rec10 = oracle_hits / len(overall_gts) if overall_gts else 0.0

        failure_mode = determine_failure_mode(bm25_rec10, faiss_rec10, opt_rec10, dyn_rec10, oracle_rec10)

        final_results.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "route": item["route"],
            "query_profile": item["query_profile"],
            "difficulty": item["difficulty"],
            "explicit_gts": item["explicit_gts_filtered"],
            "inferred_gts": item["inferred_gts_filtered"],
            "substance_gt_raw": item["substance_gt_raw"],
            "procedural_gt_raw": item["procedural_gt_raw"],
            "substance_gt_filtered": item["substance_gt_filtered"],
            "procedural_gt_filtered": item["procedural_gt_filtered"],
            "core_gt_raw": item["core_gt_raw"],
            "supporting_gt_raw": item["supporting_gt_raw"],
            "core_gt_filtered": item["core_gt_filtered"],
            "supporting_gt_filtered": item["supporting_gt_filtered"],
            "gt_weights": item["gt_weights"],
            "metrics": metrics_dict,
            "latency_breakdown": item["latency_breakdown"]
        })

        
        # Build explainability logs with automatic error tags and diagnostic logs
        expl_configs = {}
        for cfg in ["BM25", "FAISS", "Hybrid_Dynamic", "Hybrid_Dynamic_Rerank"]:
            run_pool = metrics_dict[cfg]["top_k"]
            topk_res = []
            missed_gts = list(overall_gts)
            
            for doc in run_pool[:10]:
                std_id = doc["standard_law_id"]
                is_rel = any(is_match(std_id, gt) for gt in overall_gts)
                matched_gt = next((gt for gt in overall_gts if is_match(std_id, gt)), None)
                if is_rel and matched_gt in missed_gts:
                    missed_gts.remove(matched_gt)
                
                expl_doc = {
                    "rank": doc["rank"],
                    "doc_id": std_id,
                    "score": doc["score"],
                    "is_relevant": is_rel
                }
                if matched_gt:
                    expl_doc["matched_gt"] = matched_gt
                topk_res.append(expl_doc)
                
            # Calculate raw missed gts for hierarchy confusion
            missed_gts_raw = list(overall_gts_raw)
            for doc in run_pool[:10]:
                std_id = doc["standard_law_id"]
                matched_gt_raw = next((gt for gt in overall_gts_raw if is_match(std_id, gt)), None)
                if matched_gt_raw in missed_gts_raw:
                    missed_gts_raw.remove(matched_gt_raw)

            err_type = classify_retrieval_error(item["query"], missed_gts, missed_gts_raw, run_pool[:10], id_to_doc, item["route"], ACTIVE_LAW_CODES, item["judgment_year"]) if missed_gts else None
            
            expl_configs[cfg] = {
                "topk_results": topk_res,
                "error_type": err_type,
                "router_decision": item["route"] if "Dynamic" in cfg else None,
                "routing_weight": w_bm25_dynamic if "Dynamic" in cfg else None,
                "oracle_best_weight": oracle_best_w if "Dynamic" in cfg else None,
                "attribution": attribution if "Dynamic" in cfg else None
            }
            
        explainability_logs.append({
          "query_id": item["query_id"],
          "metadata": {
            "judgment_name": item["judgment_name"],
            "year": item["judgment_year"],
            "case_type": item["case_type"]
          },
          "query_profile": item["query_profile"],
          "difficulty_score": item["difficulty"].get("score") if isinstance(item["difficulty"], dict) else 0.0,
          "difficulty_level": item["difficulty"].get("level") if isinstance(item["difficulty"], dict) else item["difficulty"],
          "latency_breakdown": item["latency_breakdown"],
          "diagnostics": {
            "oracle_best_weight": oracle_best_w,
            "router_weight": w_bm25_dynamic,
            "failure_mode": failure_mode
          },
          "configurations": expl_configs
        })
        
    # Calculate G3 temporal comparison metrics summary
    avg_on_recall_at_10 = on_recall_hits / total_recall_valid_queries if total_recall_valid_queries else 0.0
    avg_off_recall_at_10 = off_recall_hits / total_recall_valid_queries if total_recall_valid_queries else 0.0
    
    temporal_comparison_stats = {
        "future_law_removed": total_future_laws_removed,
        "affected_queries": total_affected_queries,
        "mismatch_before": total_mismatch_before,
        "mismatch_after": total_mismatch_after,
        "on_recall_at_10": float(round(avg_on_recall_at_10, 4)),
        "off_recall_at_10": float(round(avg_off_recall_at_10, 4))
    }

    # Generate timestamped run directory under output_dir / "runs"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / "runs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "pipeline_results.json"
    results_output = {
        "experiment_id": eval_config['evaluation']['experiment_id'],
        "temporal_filter_mode": temporal_filter,
        "configurations": ["BM25", "FAISS", "Hybrid_Static_Fixed", "Hybrid_Static_Opt", "Hybrid_Dynamic", "Hybrid_Dynamic_Rerank"],
        "results": final_results,
        "ablation_grid": {f"{k[0]}_{k[1]}": v for k, v in ablation_grid.items()},
        "best_static_hybrid": {"method": best_method, "w_bm25": best_w},
        "temporal_comparison": temporal_comparison_stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_output, f, ensure_ascii=False, indent=2)
        
    expl_path = run_dir / "retrieval_explainability.json"
    with open(expl_path, 'w', encoding='utf-8') as f:
        json.dump(explainability_logs, f, ensure_ascii=False, indent=2)
        
    # Copy evaluation config to the run directory
    try:
        shutil.copy(config_path, run_dir / "eval_config.yaml")
        print(f"   -> Đã sao chép cấu hình vào: {run_dir / 'eval_config.yaml'}")
    except Exception as e:
        print(f"⚠️ Cảnh báo sao chép cấu hình: {e}")

    # Copy files to default output path for backward compatibility
    default_output_path = output_dir / "pipeline_results.json"
    default_expl_path = output_dir / "retrieval_explainability.json"
    try:
        shutil.copy(output_path, default_output_path)
        shutil.copy(expl_path, default_expl_path)
        print(f"   -> Đã đồng bộ kết quả mặc định tại: {default_output_path}")
    except Exception as e:
        print(f"⚠️ Cảnh báo đồng bộ kết quả mặc định: {e}")
        
    print(f"   -> Đã lưu explainability log tại: {expl_path}")
    print(f"   -> Đã lưu toàn bộ kết quả tại: {output_path}")
    return output_path


def run_in_memory_tuning_study(query_records, k=10):
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fusion_methods = ["minmax", "softmax", "rrf"]
    
    best_method = "minmax"
    best_w = 0.5
    best_ndcg = -1.0
    grid_results = {}
    
    for method in fusion_methods:
        for w in weights:
            recall_sum = 0.0
            ndcg_sum = 0.0
            valid_queries = 0
            
            for item in query_records:
                gts = item["substance_gt_filtered"] + item["procedural_gt_filtered"]
                if not gts: continue
                
                combined = fuse_pools(item["bm25_pool"], item["faiss_pool"], w, method)
                top_k_ids = [x[0] for x in combined[:k]]
                
                hits = sum(1 for gt in gts if any(is_match(ret_id, gt) for ret_id in top_k_ids))
                recall = hits / len(gts) if gts else 0.0
                ndcg = calculate_ndcg(top_k_ids, gts, k, gt_weights=item.get("gt_weights"))
                
                recall_sum += recall
                ndcg_sum += ndcg
                valid_queries += 1
                
            mean_recall = recall_sum / valid_queries if valid_queries else 0.0
            mean_ndcg = ndcg_sum / valid_queries if valid_queries else 0.0
            grid_results[(method, w)] = {"recall": mean_recall, "ndcg": mean_ndcg}
            
            if mean_ndcg > best_ndcg:
                best_ndcg = mean_ndcg
                best_method = method
                best_w = w
                
    return best_method, best_w, grid_results

# ==========================================
# 3. METRICS & ACADEMIC TABLES REPORTING
# ==========================================

def calculate_ndcg(ret_ids, gts, k, gt_weights=None):
    top_k_ids = ret_ids[:k]
    
    relevance = []
    for ret_id in top_k_ids:
        matched_gts = [gt for gt in gts if is_match(ret_id, gt)]
        if matched_gts:
            if gt_weights:
                max_w = 0
                for gt in matched_gts:
                    norm_gt = normalize_text(gt)
                    max_w = max(max_w, gt_weights.get(norm_gt, 1))
                relevance.append(max_w)
            else:
                relevance.append(1)
        else:
            relevance.append(0)
            
    dcg = 0.0
    for idx, rel in enumerate(relevance, 1):
        dcg += (2**rel - 1) / math.log2(idx + 1)
        
    if gt_weights:
        target_weights = [gt_weights.get(normalize_text(gt), 1) for gt in gts]
    else:
        target_weights = [1] * len(gts)
        
    ideal_relevance = sorted(target_weights, reverse=True)[:k]
    ideal_relevance += [0] * (k - len(ideal_relevance))
    
    idcg = 0.0
    for idx, rel in enumerate(ideal_relevance, 1):
        idcg += (2**rel - 1) / math.log2(idx + 1)
        
    if idcg == 0.0:
        return 0.0
    return dcg / idcg

def get_task_metrics(results, cfg, k, is_filtered=True, task_type="overall", filter_profile=None, filter_difficulty=None, filter_law_family=None):
    recall_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    valid_queries = 0
    
    for item in results:
        # Apply profile/difficulty filters
        qp = item.get("query_profile")
        qp_str = qp.get("profile") if isinstance(qp, dict) else qp
        if filter_profile and qp_str != filter_profile:
            continue
            
        diff = item.get("difficulty")
        diff_str = diff.get("level") if isinstance(diff, dict) else diff
        if filter_difficulty and diff_str != filter_difficulty:
            continue
            
        if is_filtered:
            sub = item.get("substance_gt_filtered", [])
            proc = item.get("procedural_gt_filtered", [])
            exp = item.get("explicit_gts", []) 
            inf = item.get("inferred_gts", [])
            core = item.get("core_gt_filtered", [])
            supp = item.get("supporting_gt_filtered", [])
        else:
            sub = item.get("substance_gt_raw", [])
            proc = item.get("procedural_gt_raw", [])
            exp = item.get("explicit_gts", []) 
            inf = item.get("inferred_gts", [])
            core = item.get("core_gt_raw", [])
            supp = item.get("supporting_gt_raw", [])
            
        if task_type == "substantive":
            gts = sub
        elif task_type == "procedural":
            gts = proc
        elif task_type == "explicit":
            gts = exp
        elif task_type == "inferred":
            gts = inf
        elif task_type == "core":
            gts = core
        elif task_type == "supporting":
            gts = supp
        else: # overall
            gts = sub + proc
            
        # Apply law family filter if requested
        if filter_law_family:
            gts = [gt for gt in gts if map_law_code_to_family(gt.split()[-1]) == filter_law_family]
            
        if not gts: 
            continue
            
        run_data = item['metrics'].get(cfg)
        if not run_data:
            continue
            
        pool_ids = [doc['standard_law_id'] for doc in run_data['top_k']]
        top_k_ids = pool_ids[:k]
        
        hits = sum(1 for gt in gts if any(is_match(ret_id, gt) for ret_id in top_k_ids))
        recall_sum += hits / len(gts) if gts else 0.0
        ndcg_sum += calculate_ndcg(top_k_ids, gts, k, gt_weights=item.get("gt_weights"))
        
        mrr = 0.0
        for rank, ret_id in enumerate(top_k_ids, 1):
            if any(is_match(ret_id, gt) for gt in gts):
                mrr = 1.0 / rank
                break
        mrr_sum += mrr
        valid_queries += 1
        
    if valid_queries == 0:
        return 0.0, 0.0, 0.0, 0
    return recall_sum / valid_queries, ndcg_sum / valid_queries, mrr_sum / valid_queries, valid_queries

def get_weighted_recall(results, cfg, k, is_filtered=True):
    weighted_recall_sum = 0.0
    valid_queries = 0
    for item in results:
        if is_filtered:
            core = item.get("core_gt_filtered", [])
            supp = item.get("supporting_gt_filtered", [])
        else:
            core = item.get("core_gt_raw", [])
            supp = item.get("supporting_gt_raw", [])
            
        if not core and not supp:
            continue
            
        run_data = item['metrics'].get(cfg)
        if not run_data:
            continue
            
        pool_ids = [doc['standard_law_id'] for doc in run_data['top_k']]
        top_k_ids = pool_ids[:k]
        
        core_hits = sum(1 for gt in core if any(is_match(ret_id, gt) for ret_id in top_k_ids))
        supp_hits = sum(1 for gt in supp if any(is_match(ret_id, gt) for ret_id in top_k_ids))
        
        num_core = len(core)
        num_supp = len(supp)
        
        denominator = ALPHA_CORE * num_core + ALPHA_SUPPORT * num_supp
        if denominator > 0:
            query_w_recall = (ALPHA_CORE * core_hits + ALPHA_SUPPORT * supp_hits) / denominator
            weighted_recall_sum += query_w_recall
            valid_queries += 1
            
    if valid_queries == 0:
        return 0.0
    return weighted_recall_sum / valid_queries


def build_metric_world_report(results, configurations, reported_k=BASELINE_REPORTED_K):
    """Evaluate every K by slicing each method's single stored top-20 ranking."""
    worlds = {}
    for world_name, is_filtered in (("closed_world", True), ("open_world", False)):
        method_metrics = {}
        for configuration in configurations:
            metrics_at_k = {}
            for k in reported_k:
                recall, ndcg, mrr, valid = get_task_metrics(
                    results,
                    configuration,
                    k=int(k),
                    is_filtered=is_filtered,
                    task_type="overall",
                )
                metrics_at_k[str(k)] = {
                    "recall": recall,
                    "ndcg": ndcg,
                    "mrr": mrr,
                    "weighted_recall": get_weighted_recall(
                        results, configuration, k=int(k), is_filtered=is_filtered
                    ),
                    "valid_queries": valid,
                }
            method_metrics[configuration] = metrics_at_k
        worlds[world_name] = method_metrics
    return worlds


def check_temporal_mismatch(ret_id, gts):
    ret_article, ret_base_law, ret_year = parse_standard_id(ret_id)
    if not ret_article or not ret_base_law:
        return False
    for gt in gts:
        gt_article, gt_base_law, gt_year = parse_standard_id(gt)
        if gt_article == ret_article and gt_base_law == ret_base_law and gt_year != ret_year:
            return True
    return False

def compute_table_8_oracle_overlap(results, k=10):

    print("\n" + "="*95)
    print(" TABLE 8: ORACLE UPPER BOUND & COMPLEMENTARITY OVERLAP ANALYSIS @K=10")
    print("="*95)
    
    total_gts_count = 0
    bm25_only_count = 0
    faiss_only_count = 0
    both_count = 0
    neither_count = 0
    
    oracle_recall_sum = 0.0
    oracle_ndcg_sum = 0.0
    valid_queries = 0
    
    # Overlap of retrieved sets
    jaccard_overlap_sum = 0.0
    for item in results:
        bm25_set = set(doc["standard_law_id"] for doc in item["metrics"]["BM25"]["top_k"][:k])
        faiss_set = set(doc["standard_law_id"] for doc in item["metrics"]["FAISS"]["top_k"][:k])
        union_len = len(bm25_set | faiss_set)
        if union_len > 0:
            jaccard_overlap_sum += len(bm25_set & faiss_set) / union_len
    avg_jaccard_overlap = jaccard_overlap_sum / len(results) if results else 0.0

    bm25_unique_hits = 0
    faiss_unique_hits = 0
    total_hits = 0

    for item in results:
        gts = item.get("substance_gt_filtered", []) + item.get("procedural_gt_filtered", [])
        if not gts:
            continue
            
        bm25_top = [doc["standard_law_id"] for doc in item["metrics"]["BM25"]["top_k"]][:k]
        faiss_top = [doc["standard_law_id"] for doc in item["metrics"]["FAISS"]["top_k"]][:k]
        
        # Overlap analysis
        for gt in gts:
            hit_bm25 = any(is_match(r, gt) for r in bm25_top)
            hit_faiss = any(is_match(r, gt) for r in faiss_top)
            
            if hit_bm25 or hit_faiss:
                total_hits += 1
            if hit_bm25 and not hit_faiss:
                bm25_only_count += 1
                bm25_unique_hits += 1
            elif hit_faiss and not hit_bm25:
                faiss_only_count += 1
                faiss_unique_hits += 1
            elif hit_bm25 and hit_faiss:
                both_count += 1
            else:
                neither_count += 1
            total_gts_count += 1
            
        # Oracle upper bound
        oracle_hits = []
        for gt in gts:
            if any(is_match(r, gt) for r in bm25_top) or any(is_match(r, gt) for r in faiss_top):
                oracle_hits.append(gt)
                
        oracle_recall = len(oracle_hits) / len(gts)
        oracle_recall_sum += oracle_recall
        
        # Oracle nDCG with graded relevance
        gt_weights = item.get("gt_weights", {})
        
        oracle_matched_gts = []
        for gt in gts:
            if any(is_match(r, gt) for r in bm25_top) or any(is_match(r, gt) for r in faiss_top):
                oracle_matched_gts.append(gt)
                
        matched_weights = sorted([gt_weights.get(normalize_text(gt), 1) for gt in oracle_matched_gts], reverse=True)[:k]
        matched_weights += [0] * (k - len(matched_weights))
        odcg = sum((2**rel - 1) / math.log2(idx + 1) for idx, rel in enumerate(matched_weights, 1))
        
        target_weights = sorted([gt_weights.get(normalize_text(gt), 1) for gt in gts], reverse=True)[:k]
        target_weights += [0] * (k - len(target_weights))
        idcg = sum((2**rel - 1) / math.log2(idx + 1) for idx, rel in enumerate(target_weights, 1))
        
        oracle_ndcg = odcg / idcg if idcg > 0.0 else 0.0
        oracle_ndcg_sum += oracle_ndcg
        
        valid_queries += 1
        
    avg_oracle_recall = oracle_recall_sum / valid_queries if valid_queries else 0.0
    avg_oracle_ndcg = oracle_ndcg_sum / valid_queries if valid_queries else 0.0
    
    p_bm25_only = bm25_only_count / total_gts_count if total_gts_count else 0.0
    p_faiss_only = faiss_only_count / total_gts_count if total_gts_count else 0.0
    p_both = both_count / total_gts_count if total_gts_count else 0.0
    p_neither = neither_count / total_gts_count if total_gts_count else 0.0
    
    bm25_unique_contrib = bm25_unique_hits / total_hits if total_hits else 0.0
    faiss_unique_contrib = faiss_unique_hits / total_hits if total_hits else 0.0
    
    # Calculate BM25 recall at 10
    bm25_recall_sum = 0.0
    valid_bm25_queries = 0
    for item in results:
        gts = item.get("substance_gt_filtered", []) + item.get("procedural_gt_filtered", [])
        if not gts:
            continue
        bm25_top = [doc["standard_law_id"] for doc in item["metrics"]["BM25"]["top_k"]][:k]
        hits = sum(1 for gt in gts if any(is_match(r, gt) for r in bm25_top))
        bm25_recall_sum += hits / len(gts)
        valid_bm25_queries += 1
    avg_bm25_recall = bm25_recall_sum / valid_bm25_queries if valid_bm25_queries else 0.0
    coverage_gain = avg_oracle_recall - avg_bm25_recall
    
    print(f" {'Metric':<15} | {'BM25 Only':<12} | {'FAISS Only':<12} | {'Both':<12} | {'Neither':<12} | {'Oracle Upper Bound'}")
    print("-"*97)
    print(f" {'Recall@10':<15} | {p_bm25_only:<12.4f} | {p_faiss_only:<12.4f} | {p_both:<12.4f} | {p_neither:<12.4f} | {avg_oracle_recall:<18.4f}")
    print(f" {'nDCG@10':<15} | {'-':<12} | {'-':<12} | {'-':<12} | {'-':<12} | {avg_oracle_ndcg:<18.4f}")
    print("-"*97)
    print(f" Jaccard Overlap of retrieved sets: {avg_jaccard_overlap:.4f}")
    print(f" BM25 Unique Hit Contribution: {bm25_unique_contrib:.4f}")
    print(f" FAISS Unique Hit Contribution: {faiss_unique_contrib:.4f}")
    print(f" Coverage Gain (Oracle - BM25): {coverage_gain:.4f}")
    print("="*95)

def print_table_9_law_families(results, configs):
    print("\n" + "="*95)
    print(" TABLE 9: PERFORMANCE STRATIFIED BY LAW FAMILY (RECALL@10)")
    print("="*95)
    
    families = ["Civil Code", "Civil Procedure Code", "Land Law", "Marriage & Family Law", "Others"]
    print(f" {'Configuration':<25} | {'Civil Code':<12} | {'Civil Proc.':<12} | {'Land Law':<12} | {'Marr. & Fam.':<13} | {'Others':<10}")
    print("-"*97)
    
    for cfg in configs:
        scores = []
        for fam in families:
            rec, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="overall", filter_law_family=fam)
            scores.append(rec)
        print(f" {cfg:<25} | {scores[0]:<12.4f} | {scores[1]:<12.4f} | {scores[2]:<12.4f} | {scores[3]:<13.4f} | {scores[4]:<10.4f}")
    print("="*95)

def print_router_attribution_analysis(results):
    print("\n" + "="*80)
    print(" SECTION 5: ROUTER CONTRIBUTION ATTRIBUTION ANALYSIS")
    print("="*80)
    
    helped = 0
    hurt = 0
    neutral = 0
    
    for item in results:
        dyn_rec = item["metrics"]["Hybrid_Dynamic"]["top_k"]
        stat_rec = item["metrics"]["Hybrid_Static_Opt"]["top_k"]
        
        gts = item.get("substance_gt_filtered", []) + item.get("procedural_gt_filtered", [])
        if not gts:
            continue
            
        def get_rec(top_k):
            ids = [d["standard_law_id"] for d in top_k][:10]
            hits = sum(1 for gt in gts if any(is_match(r, gt) for r in ids))
            return hits / len(gts)
            
        r_dyn = get_rec(dyn_rec)
        r_stat = get_rec(stat_rec)
        
        if r_dyn > r_stat:
            helped += 1
        elif r_dyn < r_stat:
            hurt += 1
        else:
            neutral += 1
            
    total = helped + hurt + neutral
    if total:
        print(f" Dynamic Helped: {helped} queries ({helped/total*100:.1f}%)")
        print(f" Dynamic Hurt:   {hurt} queries ({hurt/total*100:.1f}%)")
        print(f" Neutral:        {neutral} queries ({neutral/total*100:.1f}%)")
    print("="*80 + "\n")

def generate_paper_tables(results_file_path):
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    configs = data['configurations']
    results = data['results']
    
    # -------------------------------------------------------------
    # TABLE 1: Overall Performance (Task E - Overall) @ K=5 and K=10
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print(f" TABLE 1: OVERALL RETRIEVAL PERFORMANCE (TASK E - ALL GROUND TRUTHS COMBINED)")
    print("="*95)
    print(f" {'Configuration':<25} | {'F-Recall@5':<11} | {'F-nDCG@5':<9} | {'F-MRR@5':<8} | {'U-Recall@10':<11} | {'U-nDCG@10':<9} | {'U-MRR@10':<8}")
    print("-"*97)
    
    for cfg in configs:
        f_rec5, f_ndcg5, f_mrr5, _ = get_task_metrics(results, cfg, k=5, is_filtered=True, task_type="overall")
        u_rec10, u_ndcg10, u_mrr10, _ = get_task_metrics(results, cfg, k=10, is_filtered=False, task_type="overall")
        print(f" {cfg:<25} | {f_rec5:<11.4f} | {f_ndcg5:<9.4f} | {f_mrr5:<8.4f} | {u_rec10:<11.4f} | {u_ndcg10:<9.4f} | {u_mrr10:<8.4f}")
    print("="*95)

    # -------------------------------------------------------------
    # TABLE 1B: CORE VS SUPPORTING & WEIGHTED RECALL PERFORMANCE
    # -------------------------------------------------------------
    print("\n" + "="*115)
    print(" TABLE 1B: CORE VS SUPPORTING & WEIGHTED RECALL PERFORMANCE")
    print("="*115)
    print(f" {'Configuration':<25} | {'Core-Rec@5':<11} | {'Supp-Rec@5':<11} | {'W-Rec@5':<9} | {'Core-Rec@10':<11} | {'Supp-Rec@10':<12} | {'W-Rec@10':<9}")
    print("-"*117)
    for cfg in configs:
        core_rec5, _, _, _ = get_task_metrics(results, cfg, k=5, is_filtered=True, task_type="core")
        supp_rec5, _, _, _ = get_task_metrics(results, cfg, k=5, is_filtered=True, task_type="supporting")
        w_rec5 = get_weighted_recall(results, cfg, k=5, is_filtered=True)
        
        core_rec10, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="core")
        supp_rec10, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="supporting")
        w_rec10 = get_weighted_recall(results, cfg, k=10, is_filtered=True)
        print(f" {cfg:<25} | {core_rec5:<11.4f} | {supp_rec5:<11.4f} | {w_rec5:<9.4f} | {core_rec10:<11.4f} | {supp_rec10:<12.4f} | {w_rec10:<9.4f}")
    print("="*115)


    # -------------------------------------------------------------
    # TABLE 2: Performance by Explicit vs Inferred Law (Task A vs Task B) @ K=10
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print(" TABLE 2: EXPLICIT (TASK A) VS INFERRED (TASK B) RETRIEVAL @K=10")
    print("="*95)
    print(f" {'Configuration':<25} | {'Explicit Recall':<16} | {'Explicit nDCG':<14} | {'Inferred Recall':<16} | {'Inferred nDCG':<14}")
    print("-"*97)
    for cfg in configs:
        exp_rec, exp_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="explicit")
        inf_rec, inf_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="inferred")
        print(f" {cfg:<25} | {exp_rec:<16.4f} | {exp_ndcg:<14.4f} | {inf_rec:<16.4f} | {inf_ndcg:<14.4f}")
    print("="*95)

    # -------------------------------------------------------------
    # TABLE 3: Substantive (Task C) vs Procedural (Task D) @K=10
    # -------------------------------------------------------------
    print("\n" + "="*95)
    print(" TABLE 3: SUBSTANTIVE (TASK C) VS PROCEDURAL (TASK D) RETRIEVAL @K=10")
    print("="*95)
    print(f" {'Configuration':<25} | {'Substantive Rec.':<17} | {'Substantive nDCG':<16} | {'Procedural Rec.':<16} | {'Procedural nDCG':<15}")
    print("-"*97)
    for cfg in configs:
        sub_rec, sub_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="substantive")
        proc_rec, proc_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="procedural")
        print(f" {cfg:<25} | {sub_rec:<17.4f} | {sub_ndcg:<16.4f} | {proc_rec:<16.4f} | {proc_ndcg:<15.4f}")
    print("="*95)

    # -------------------------------------------------------------
    # TABLE 4: Performance by Query Profile (Recall@10)
    # -------------------------------------------------------------
    print("\n" + "="*110)
    print(" TABLE 4: RETRIEVAL RECALL@10 STRATIFIED BY LEGAL QUERY INTENT PROFILE")
    print("="*110)
    profiles = ["citation_exact", "factual_narrative", "doctrinal_semantic", "procedural_focus", "mixed_multi_issue", "long_narrative"]
    print(f" {'Configuration':<25} | {'Exact Cit.':<11} | {'Fact. Narr.':<12} | {'Doctr. Sem.':<12} | {'Proc. Focus':<12} | {'Mixed Multi':<12} | {'Long Narr.':<11}")
    print("-"*117)
    for cfg in configs:
        prof_recalls = []
        for prof in profiles:
            rec, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="overall", filter_profile=prof)
            prof_recalls.append(rec)
        print(f" {cfg:<25} | {prof_recalls[0]:<11.4f} | {prof_recalls[1]:<12.4f} | {prof_recalls[2]:<12.4f} | {prof_recalls[3]:<12.4f} | {prof_recalls[4]:<12.4f} | {prof_recalls[5]:<11.4f}")
    print("="*110)

    # -------------------------------------------------------------
    # TABLE 5: Performance by Difficulty Level (Recall@10)
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print(" TABLE 5: PERFORMANCE STRATIFIED BY QUERY DIFFICULTY LEVEL @K=10")
    print("="*70)
    difficulties = ["easy", "medium", "hard"]
    print(f" {'Configuration':<25} | {'Easy (Recall)':<14} | {'Medium (Recall)':<16} | {'Hard (Recall)':<14}")
    print("-"*67)
    for cfg in configs:
        diff_recalls = []
        for diff in difficulties:
            rec, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="overall", filter_difficulty=diff)
            diff_recalls.append(rec)
        print(f" {cfg:<25} | {diff_recalls[0]:<14.4f} | {diff_recalls[1]:<16.4f} | {diff_recalls[2]:<14.4f}")
    print("="*70)

    # -------------------------------------------------------------
    # TABLE 6: Error Taxonomy Distribution
    # -------------------------------------------------------------
    print("\n" + "="*110)
    print(" TABLE 6: DIAGNOSTIC RETRIEVAL ERROR TAXONOMY DISTRIBUTION (FAILURES @K=10)")
    print("="*110)
    error_types = ["lexical_miss", "semantic_drift", "temporal_mismatch", "procedural_confusion", "hierarchy_confusion", "multi_issue_dilution"]
    print(f" {'Configuration':<25} | {'Lexical Miss':<13} | {'Sem. Drift':<11} | {'Temp. Mism.':<12} | {'Proc. Conf.':<12} | {'Hier. Conf.':<12} | {'Multi Dilut.':<12}")
    print("-"*117)
    
    expl_path = Path(results_file_path).parent / "retrieval_explainability.json"
    if expl_path.exists():
        with open(expl_path, 'r', encoding='utf-8') as f:
            expl_data = json.load(f)
        for cfg in ["BM25", "FAISS", "Hybrid_Dynamic", "Hybrid_Dynamic_Rerank"]:
            counts = {err: 0 for err in error_types}
            for item in expl_data:
                err = item["configurations"].get(cfg, {}).get("error_type")
                if err in counts:
                    counts[err] += 1
            print(f" {cfg:<25} | {counts['lexical_miss']:<13} | {counts['semantic_drift']:<11} | {counts['temporal_mismatch']:<12} | {counts['procedural_confusion']:<12} | {counts['hierarchy_confusion']:<12} | {counts['multi_issue_dilution']:<12}")
    else:
        print("   [!] Explainability log not found. Cannot populate Table 6.")
    print("="*110)

    # Compute Table 8 (Oracle & Overlap)
    compute_table_8_oracle_overlap(results, k=10)

    # Compute Table 9 (Law families breakdown)
    print_table_9_law_families(results, configs)
    
    # Compute Router attribution analysis
    print_router_attribution_analysis(results)

    # -------------------------------------------------------------
    # TABLE 12: TEMPORAL FILTERING IMPACT COMPARISON
    # -------------------------------------------------------------
    comp = data.get("temporal_comparison", {})
    if comp:
        print("\n" + "="*95)
        print(" TABLE 12: TEMPORAL FILTERING IMPACT COMPARISON")
        print("="*95)
        print(f" {'Metric / Statistic':<45} | {'Value':<15}")
        print("-"*65)
        print(f" {'Total Future Laws Removed':<45} | {comp.get('future_law_removed', 0):<15}")
        print(f" {'Total Affected Queries (Retrieval Changed)':<45} | {comp.get('affected_queries', 0):<15}")
        print(f" {'Total Temporal Mismatches (Before Filter)':<45} | {comp.get('mismatch_before', 0):<15}")
        print(f" {'Total Temporal Mismatches (After Filter)':<45} | {comp.get('mismatch_after', 0):<15}")
        print(f" {'Dynamic Hybrid Rerank Recall@10 (Filter OFF)':<45} | {comp.get('off_recall_at_10', 0.0):<15.4f}")
        print(f" {'Dynamic Hybrid Rerank Recall@10 (Filter ON)':<45} | {comp.get('on_recall_at_10', 0.0):<15.4f}")
        print("="*95)

def profile_latency_tradeoff(results_file_path):

    print("\n" + "="*95)
    print(" TABLE 10: LATENCY (ms/query) VS RETRIEVAL ACCURACY (RECALL@10) TRADE-OFF ")
    print("="*95)
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    results = data['results']
    
    # Gather latencies
    latencies_map = {
        "BM25": [],
        "FAISS": [],
        "Hybrid_Static_Fixed": [],
        "Hybrid_Static_Opt": [],
        "Hybrid_Dynamic_Serial": [],
        "Hybrid_Dynamic_Parallel": [],
        "Hybrid_Dynamic_Rerank_Serial": [],
        "Hybrid_Dynamic_Rerank_Parallel": []
    }
    
    for item in results:
        metrics = item.get("metrics", {})
        latencies_map["BM25"].append(metrics.get("BM25", {}).get("latency_ms", 0.0))
        latencies_map["FAISS"].append(metrics.get("FAISS", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Static_Fixed"].append(metrics.get("Hybrid_Static_Fixed", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Static_Opt"].append(metrics.get("Hybrid_Static_Opt", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Serial"].append(metrics.get("Hybrid_Dynamic", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Parallel"].append(metrics.get("Hybrid_Dynamic", {}).get("latency_parallel_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Rerank_Serial"].append(metrics.get("Hybrid_Dynamic_Rerank", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Rerank_Parallel"].append(metrics.get("Hybrid_Dynamic_Rerank", {}).get("latency_parallel_ms", 0.0))
        
    print(f" {'Architecture':<35} | {'Mean Latency':<13} | {'P95 Latency':<12} | {'Overall Recall@10':<18} | {'Execution Type':<15}")
    print("-"*102)
    
    display_names = [
        ("BM25", "TH A (BM25 Only)", "BM25", "Tuần tự (Serial)"),
        ("FAISS", "TH B (FAISS Only)", "FAISS", "Tuần tự (Serial)"),
        ("Hybrid_Static_Fixed", "TH C1 (Static Hybrid)", "Hybrid_Static_Fixed", "Tuần tự (Serial)"),
        ("Hybrid_Static_Opt", "TH C1-opt (Tuned Static)", "Hybrid_Static_Opt", "Tuần tự (Serial)"),
        ("Hybrid_Dynamic_Serial", "TH C2 (Dynamic Hybrid)", "Hybrid_Dynamic", "Tuần tự (Serial)"),
        ("Hybrid_Dynamic_Parallel", "TH C2 (Dynamic Hybrid)", "Hybrid_Dynamic", "Song song (Parallel)"),
        ("Hybrid_Dynamic_Rerank_Serial", "TH D (Dynamic + Reranker)", "Hybrid_Dynamic_Rerank", "Tuần tự (Serial)"),
        ("Hybrid_Dynamic_Rerank_Parallel", "TH D (Dynamic + Reranker)", "Hybrid_Dynamic_Rerank", "Song song (Parallel)")
    ]
    
    for key, name, cfg_key, exec_type in display_names:
        times = latencies_map[key]
        if not times:
            continue
        mean_val = np.mean(times)
        p95_val = np.percentile(times, 95)
        recall, _, _, _ = get_task_metrics(results, cfg_key, k=10, is_filtered=True, task_type="overall")
        print(f" {name:<35} | {mean_val:<13.2f} | {p95_val:<12.2f} | {recall:<18.4f} | {exec_type:<15}")
    print("="*95)

def perform_pairwise_significance_tests(results_file_path, k=10):
    print("\n" + "="*95)
    print(" SECTION 4: PAIRED WILCOXON SIGNED-RANK SIGNIFICANCE TESTS (p-values)")
    print("="*95)
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    from scipy.stats import wilcoxon
    results = data['results']
    
    metric_vectors = {
        "BM25": {"recall": [], "ndcg": [], "mrr": []},
        "FAISS": {"recall": [], "ndcg": [], "mrr": []},
        "Hybrid_Static_Fixed": {"recall": [], "ndcg": [], "mrr": []},
        "Hybrid_Static_Opt": {"recall": [], "ndcg": [], "mrr": []},
        "Hybrid_Dynamic": {"recall": [], "ndcg": [], "mrr": []},
        "Hybrid_Dynamic_Rerank": {"recall": [], "ndcg": [], "mrr": []}
    }
    
    for item in results:
        gts = item.get("substance_gt_filtered", []) + item.get("procedural_gt_filtered", [])
        if not gts: 
            continue
            
        for cfg in metric_vectors.keys():
            run_data = item['metrics'].get(cfg)
            if not run_data:
                recall = ndcg = mrr = 0.0
            else:
                pool_ids = [doc['standard_law_id'] for doc in run_data['top_k']]
                top_k_ids = pool_ids[:k]
                
                hits = sum(1 for gt in gts if any(is_match(ret_id, gt) for ret_id in top_k_ids))
                recall = hits / len(gts) if gts else 0.0
                ndcg = calculate_ndcg(top_k_ids, gts, k, gt_weights=item.get("gt_weights"))
                
                mrr = 0.0
                for rank, ret_id in enumerate(top_k_ids, 1):
                    if any(is_match(ret_id, gt) for gt in gts):
                        mrr = 1.0 / rank
                        break
            
            metric_vectors[cfg]["recall"].append(recall)
            metric_vectors[cfg]["ndcg"].append(ndcg)
            metric_vectors[cfg]["mrr"].append(mrr)
            
    pairs_to_test = [
        ("Hybrid_Dynamic", "Hybrid_Static_Opt", "TH C2 (Dynamic) vs TH C1-opt (Tuned)"),
        ("Hybrid_Dynamic_Rerank", "Hybrid_Dynamic", "TH D (Rerank) vs TH C2 (Dynamic)"),
        ("Hybrid_Dynamic_Rerank", "BM25", "TH D (Rerank) vs TH A (BM25)"),
        ("Hybrid_Dynamic_Rerank", "FAISS", "TH D (Rerank) vs TH B (FAISS)")
    ]
    
    print(f" {'Paired Comparison':<38} | {'Recall@10 p-value':<17} | {'nDCG@10 p-value':<15} | {'MRR p-value':<12} | {'Significant?'}")
    print("-"*102)
    
    for cfg_a, cfg_b, label in pairs_to_test:
        row_p = {}
        for m_name in ["recall", "ndcg", "mrr"]:
            a = metric_vectors[cfg_a][m_name]
            b = metric_vectors[cfg_b][m_name]
            
            diffs = [x - y for x, y in zip(a, b)]
            if all(d == 0 for d in diffs):
                p_val = 1.0
            else:
                try:
                    stat, p_val = wilcoxon(a, b, alternative='greater')
                except Exception:
                    p_val = 1.0
            row_p[m_name] = p_val
            
        significant = any(p < 0.05 for p in row_p.values())
        sig_label = "Yes (*)" if significant else "No"
        print(f" {label:<38} | {row_p['recall']:<17.6f} | {row_p['ndcg']:<15.6f} | {row_p['mrr']:<12.6f} | {sig_label}")
    print("="*95)

def render_fusion_ablation_table(results_file_path):
    print("\n" + "="*95)
    print(" TABLE 7: FUSION TECHNIQUE & SENSITIVITY ABLATION STUDY (RECALL@10)")
    print("="*95)
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    grid = data.get("ablation_grid", {})
    if not grid:
        print("   [!] Ablation data not found.")
        return
        
    weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f" {'Weight w_bm25':<17} | {'Min-Max Score Fusion':<22} | {'Softmax Score Fusion':<22} | {'Reciprocal Rank Fusion (RRF)':<22}")
    print("-"*90)
    for w in weights:
        mm_val = grid.get(f"minmax_{w}", {}).get("recall", 0.0)
        sm_val = grid.get(f"softmax_{w}", {}).get("recall", 0.0)
        rrf_val = grid.get(f"rrf_{w}", {}).get("recall", 0.0)
        print(f" {w:<17.1f} | {mm_val:<22.4f} | {sm_val:<22.4f} | {rrf_val:<22.4f}")
    print("="*95)
    best_static = data.get("best_static_hybrid", {})
    print(f"   -> Tuned Static Hybrid Parameter (TH C1-opt): w_bm25={best_static.get('w_bm25')}, Method={best_static.get('method').upper()}\n")

def print_failure_mode_analysis(results_file_path):
    print("\n" + "="*80)
    print(" TABLE 11: DYNAMIC HYBRID FAILURE MODE DISTRIBUTION")
    print("="*80)
    
    expl_path = Path(results_file_path).parent / "retrieval_explainability.json"
    if not expl_path.exists():
        print("   [!] retrieval_explainability.json not found.")
        return
        
    with open(expl_path, 'r', encoding='utf-8') as f:
        explainability_logs = json.load(f)
        
    retriever_fail = 0
    router_fail = 0
    fusion_fail = 0
    neutral_success = 0
    
    for item in explainability_logs:
        mode = item.get("diagnostics", {}).get("failure_mode", "neutral")
        if mode == "retriever_failure":
            retriever_fail += 1
        elif mode == "router_failure":
            router_fail += 1
        elif mode == "fusion_failure":
            fusion_fail += 1
        else:
            neutral_success += 1
            
    total = len(explainability_logs)
    if total:
        print(f" {'Failure Mode':<35} | {'Count':<10} | {'Percentage':<12}")
        print("-"*65)
        print(f" {'Retriever Failure (Oracle Recall = 0)':<35} | {retriever_fail:<10} | {retriever_fail/total*100:<11.2f}%")
        print(f" {'Router Failure (Dynamic < Tuned Static)':<35} | {router_fail:<10} | {router_fail/total*100:<11.2f}%")
        print(f" {'Fusion Failure (Tuned Static < Max)':<35} | {fusion_fail:<10} | {fusion_fail/total*100:<11.2f}%")
        print(f" {'Success / Neutral (Dynamic >= Static)':<35} | {neutral_success:<10} | {neutral_success/total*100:<11.2f}%")
    print("="*80 + "\n")

def generate_latex_tables_source(results_file_path):
    print("\n" + "="*95)
    print(" LATEX TABLE GENERATOR (COPY-PASTE FOR SCIENTIFIC PAPERS)")
    print("="*95)
    
    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    configs = data['configurations']
    results = data['results']
    
    # 1. LaTeX Table 1 (Overall)
    print("\n% --- TABLE 1: OVERALL PERFORMANCE ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Overall Retrieval Performance (Task E - All Ground Truths Combined)}")
    print("\\label{tab:overall_perf}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{F-Recall@5} & \\textbf{F-nDCG@5} & \\textbf{F-MRR@5} & \\textbf{U-Recall@10} & \\textbf{U-nDCG@10} & \\textbf{U-MRR@10} \\\\")
    print("\\hline")
    for cfg in configs:
        f_rec5, f_ndcg5, f_mrr5, _ = get_task_metrics(results, cfg, k=5, is_filtered=True, task_type="overall")
        u_rec10, u_ndcg10, u_mrr10, _ = get_task_metrics(results, cfg, k=10, is_filtered=False, task_type="overall")
        name = cfg.replace("_", "\\_")
        print(f"  {name:<25} & {f_rec5:<10.4f} & {f_ndcg5:<9.4f} & {f_mrr5:<8.4f} & {u_rec10:<11.4f} & {u_ndcg10:<9.4f} & {u_mrr10:<8.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 1B. LaTeX Table 1B (Core vs Supporting & Weighted Recall)
    print("\n% --- TABLE 1B: CORE VS SUPPORTING & WEIGHTED RECALL ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Core vs Supporting \\& Weighted Recall Performance}")
    print("\\label{tab:core_supporting}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{Core-Rec@5} & \\textbf{Supp-Rec@5} & \\textbf{W-Rec@5} & \\textbf{Core-Rec@10} & \\textbf{Supp-Rec@10} & \\textbf{W-Rec@10} \\\\")
    print("\\hline")
    for cfg in configs:
        core_rec5, _, _, _ = get_task_metrics(results, cfg, k=5, is_filtered=True, task_type="core")
        supp_rec5, _, _, _ = get_task_metrics(results, cfg, k=5, is_filtered=True, task_type="supporting")
        w_rec5 = get_weighted_recall(results, cfg, k=5, is_filtered=True)
        
        core_rec10, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="core")
        supp_rec10, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="supporting")
        w_rec10 = get_weighted_recall(results, cfg, k=10, is_filtered=True)
        name = cfg.replace("_", "\\_")
        print(f"  {name:<25} & {core_rec5:<10.4f} & {supp_rec5:<10.4f} & {w_rec5:<8.4f} & {core_rec10:<11.4f} & {supp_rec10:<11.4f} & {w_rec10:<8.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 2. LaTeX Table 2 (Explicit vs Inferred)
    print("\n% --- TABLE 2: EXPLICIT VS INFERRED ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Explicit (Task A) vs Inferred (Task B) Retrieval @K=10}")
    print("\\label{tab:explicit_inferred}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{Explicit Recall} & \\textbf{Explicit nDCG} & \\textbf{Inferred Recall} & \\textbf{Inferred nDCG} \\\\")
    print("\\hline")
    for cfg in configs:
        exp_rec, exp_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="explicit")
        inf_rec, inf_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="inferred")
        name = cfg.replace("_", "\\_")
        print(f"  {name:<25} & {exp_rec:<15.4f} & {exp_ndcg:<13.4f} & {inf_rec:<15.4f} & {inf_ndcg:<13.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 3. LaTeX Table 3 (Substantive vs Procedural)
    print("\n% --- TABLE 3: SUBSTANTIVE VS PROCEDURAL ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Substantive (Task C) vs Procedural (Task D) Retrieval @K=10}")
    print("\\label{tab:substantive_procedural}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{Substantive Rec.} & \\textbf{Substantive nDCG} & \\textbf{Procedural Rec.} & \\textbf{Procedural nDCG} \\\\")
    print("\\hline")
    for cfg in configs:
        sub_rec, sub_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="substantive")
        proc_rec, proc_ndcg, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="procedural")
        name = cfg.replace("_", "\\_")
        print(f"  {name:<25} & {sub_rec:<16.4f} & {sub_ndcg:<16.4f} & {proc_rec:<15.4f} & {proc_ndcg:<15.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 4. LaTeX Table 4 (Query Profiles)
    print("\n% --- TABLE 4: STRATIFIED BY QUERY PROFILE ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Retrieval Recall@10 Stratified by Legal Query Intent Profile}")
    print("\\label{tab:query_profiles}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{Exact Cit.} & \\textbf{Fact. Narr.} & \\textbf{Doctr. Sem.} & \\textbf{Proc. Focus} & \\textbf{Mixed Multi} & \\textbf{Long Narr.} \\\\")
    print("\\hline")
    profiles = ["citation_exact", "factual_narrative", "doctrinal_semantic", "procedural_focus", "mixed_multi_issue", "long_narrative"]
    for cfg in configs:
        cfg_escaped = cfg.replace('_', '\\_')
        row_str = f"  {cfg_escaped:<25}"
        for prof in profiles:
            rec, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="overall", filter_profile=prof)
            row_str += f" & {rec:<10.4f}"
        print(row_str + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 5. LaTeX Table 5 (Difficulty)
    print("\n% --- TABLE 5: STRATIFIED BY QUERY DIFFICULTY ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Performance Stratified by Query Difficulty Level @K=10}")
    print("\\label{tab:difficulty_levels}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{Easy (Recall)} & \\textbf{Medium (Recall)} & \\textbf{Hard (Recall)} \\\\")
    print("\\hline")
    difficulties = ["easy", "medium", "hard"]
    for cfg in configs:
        cfg_escaped = cfg.replace('_', '\\_')
        row_str = f"  {cfg_escaped:<25}"
        for diff in difficulties:
            rec, _, _, _ = get_task_metrics(results, cfg, k=10, is_filtered=True, task_type="overall", filter_difficulty=diff)
            row_str += f" & {rec:<13.4f}"
        print(row_str + " \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 6. LaTeX Table 6 (Error Taxonomy)
    print("\n% --- TABLE 6: DIAGNOSTIC RETRIEVAL ERROR TAXONOMY ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Diagnostic Retrieval Error Taxonomy Distribution (Failures @K=10)}")
    print("\\label{tab:error_taxonomy}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("  \\textbf{Configuration} & \\textbf{Lexical Miss} & \\textbf{Sem. Drift} & \\textbf{Temp. Mism.} & \\textbf{Proc. Conf.} & \\textbf{Hier. Conf.} & \\textbf{Multi Dilut.} \\\\")
    print("\\hline")
    error_types = ["lexical_miss", "semantic_drift", "temporal_mismatch", "procedural_confusion", "hierarchy_confusion", "multi_issue_dilution"]
    expl_path = Path(results_file_path).parent / "retrieval_explainability.json"
    if expl_path.exists():
        with open(expl_path, 'r', encoding='utf-8') as f_expl:
            expl_data = json.load(f_expl)
        for cfg in ["BM25", "FAISS", "Hybrid_Dynamic", "Hybrid_Dynamic_Rerank"]:
            counts = {err: 0 for err in error_types}
            for item in expl_data:
                err = item["configurations"].get(cfg, {}).get("error_type")
                if err in counts:
                    counts[err] += 1
            name = cfg.replace("_", "\\_")
            print(f"  {name:<25} & {counts['lexical_miss']:<12} & {counts['semantic_drift']:<10} & {counts['temporal_mismatch']:<11} & {counts['procedural_confusion']:<11} & {counts['hierarchy_confusion']:<11} & {counts['multi_issue_dilution']:<12} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 7. LaTeX Table 10 (Latency trade-off)
    print("\n% --- TABLE 10: LATENCY VS ACCURACY ---")
    print("\\begin{table}[ht]")
    print("\\centering")
    print("\\caption{Latency (ms/query) vs Retrieval Accuracy (Recall@10) Trade-off}")
    print("\\label{tab:latency_tradeoff}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("  \\textbf{Architecture} & \\textbf{Mean Latency} & \\textbf{P95 Latency} & \\textbf{Overall Recall@10} & \\textbf{Execution Type} \\\\")
    print("\\hline")
    
    # Gather latencies again
    latencies_map = {
        "BM25": [], "FAISS": [], "Hybrid_Static_Fixed": [], "Hybrid_Static_Opt": [],
        "Hybrid_Dynamic_Serial": [], "Hybrid_Dynamic_Parallel": [],
        "Hybrid_Dynamic_Rerank_Serial": [], "Hybrid_Dynamic_Rerank_Parallel": []
    }
    for item in results:
        metrics = item.get("metrics", {})
        latencies_map["BM25"].append(metrics.get("BM25", {}).get("latency_ms", 0.0))
        latencies_map["FAISS"].append(metrics.get("FAISS", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Static_Fixed"].append(metrics.get("Hybrid_Static_Fixed", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Static_Opt"].append(metrics.get("Hybrid_Static_Opt", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Serial"].append(metrics.get("Hybrid_Dynamic", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Parallel"].append(metrics.get("Hybrid_Dynamic", {}).get("latency_parallel_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Rerank_Serial"].append(metrics.get("Hybrid_Dynamic_Rerank", {}).get("latency_ms", 0.0))
        latencies_map["Hybrid_Dynamic_Rerank_Parallel"].append(metrics.get("Hybrid_Dynamic_Rerank", {}).get("latency_parallel_ms", 0.0))
        
    display_names = [
        ("BM25", "TH A (BM25 Only)", "BM25", "Serial"),
        ("FAISS", "TH B (FAISS Only)", "FAISS", "Serial"),
        ("Hybrid_Static_Fixed", "TH C1 (Static Hybrid)", "Hybrid_Static_Fixed", "Serial"),
        ("Hybrid_Static_Opt", "TH C1-opt (Tuned Static)", "Hybrid_Static_Opt", "Serial"),
        ("Hybrid_Dynamic_Serial", "TH C2 (Dynamic Hybrid)", "Hybrid_Dynamic", "Serial"),
        ("Hybrid_Dynamic_Parallel", "TH C2 (Dynamic Hybrid)", "Hybrid_Dynamic", "Parallel"),
        ("Hybrid_Dynamic_Rerank_Serial", "TH D (Dynamic + Reranker)", "Hybrid_Dynamic_Rerank", "Serial"),
        ("Hybrid_Dynamic_Rerank_Parallel", "TH D (Dynamic + Reranker)", "Hybrid_Dynamic_Rerank", "Parallel")
    ]
    for key, name, cfg_key, exec_type in display_names:
        times = latencies_map[key]
        if not times: continue
        mean_val = np.mean(times)
        p95_val = np.percentile(times, 95)
        recall, _, _, _ = get_task_metrics(results, cfg_key, k=10, is_filtered=True, task_type="overall")
        print(f"  {name:<30} & {mean_val:<12.2f} & {p95_val:<11.2f} & {recall:<18.4f} & {exec_type:<10} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 8. LaTeX Table 12 (Temporal validation comparison)
    comp = data.get("temporal_comparison", {})
    if comp:
        print("\n% --- TABLE 12: TEMPORAL FILTERING IMPACT COMPARISON ---")
        print("\\begin{table}[ht]")
        print("\\centering")
        print("\\caption{Temporal Filtering Impact Comparison}")
        print("\\label{tab:temporal_impact}")
        print("\\begin{tabular}{lc}")
        print("\\hline")
        print("  \\textbf{Metric / Statistic} & \\textbf{Value} \\\\")
        print("\\hline")
        print(f"  Total Future Laws Removed & {comp.get('future_law_removed', 0)} \\\\")
        print(f"  Total Affected Queries (Retrieval Changed) & {comp.get('affected_queries', 0)} \\\\")
        print(f"  Total Temporal Mismatches (Before Filter) & {comp.get('mismatch_before', 0)} \\\\")
        print(f"  Total Temporal Mismatches (After Filter) & {comp.get('mismatch_after', 0)} \\\\")
        print(f"  Dynamic Hybrid Rerank Recall@10 (Filter OFF) & {comp.get('off_recall_at_10', 0.0):.4f} \\\\")
        print(f"  Dynamic Hybrid Rerank Recall@10 (Filter ON) & {comp.get('on_recall_at_10', 0.0):.4f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")

    print("="*95 + "\n")


# Compatibility exports: active implementations live in focused modules.
# The legacy/advanced functions above resolve these names at call time, so this
# preserves their behavior while existing imports continue to work.
if __package__:
    from .experiments.baseline import (
        get_enabled_configurations,
        run_baseline_evaluation_pipeline,
        should_initialize_reranker,
        should_run_tuning,
    )
    from .core.corpus import (
        load_documents,
        load_frozen_baseline_corpus,
        sha256_file,
    )
    from .core.ground_truth import (
        build_canonical_ground_truth as _build_canonical_ground_truth,
        extract_gt_sub_proc,
        get_gt_weights as _get_gt_weights,
    )
    from .core.normalization import (
        canonical_retrieval_id,
        canonicalize_law_code,
        extract_article_number,
        is_match,
        normalize_text,
    )
    from .core.query import build_query_text, clean_query_legacy, normalize_query_q0
    from .core.representations import build_document_text, create_standard_id
    from .evaluation.coverage import compute_ground_truth_coverage
    from .evaluation.latency import aggregate_latency_ms, build_latency_aggregates
    from .evaluation.metrics import (
        build_metric_world_report as _build_metric_world_report,
        calculate_ndcg,
        get_task_metrics,
        get_weighted_recall as _get_weighted_recall,
        map_law_code_to_family,
    )
    from .retrieval.fusion import fuse_pools as _fuse_pools
    from .retrieval.fusion import softmax_normalize
    from .retrieval.temporal import should_include_document
    from .reporting.manifest import (
        enforce_full_run_provenance,
        fingerprint_records,
        git_provenance as _git_provenance,
        resolved_model_revision as _resolved_model_revision,
        safe_package_version as _safe_package_version,
        verify_model_revision,
    )
    from .reporting.serialization import assemble_baseline_summary
else:
    _project_root_for_imports = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root_for_imports) not in sys.path:
        sys.path.insert(0, str(_project_root_for_imports))
    from backend.scripts.Evaluation.core.normalization import (
        canonical_retrieval_id,
        canonicalize_law_code,
        extract_article_number,
        is_match,
        normalize_text,
    )
    from backend.scripts.Evaluation.core.corpus import (
        load_documents,
        load_frozen_baseline_corpus,
        sha256_file,
    )
    from backend.scripts.Evaluation.core.ground_truth import (
        build_canonical_ground_truth as _build_canonical_ground_truth,
        extract_gt_sub_proc,
        get_gt_weights as _get_gt_weights,
    )
    from backend.scripts.Evaluation.core.query import (
        build_query_text,
        clean_query_legacy,
        normalize_query_q0,
    )
    from backend.scripts.Evaluation.core.representations import (
        build_document_text,
        create_standard_id,
    )
    from backend.scripts.Evaluation.evaluation.coverage import compute_ground_truth_coverage
    from backend.scripts.Evaluation.experiments.baseline import (
        get_enabled_configurations,
        run_baseline_evaluation_pipeline,
        should_initialize_reranker,
        should_run_tuning,
    )
    from backend.scripts.Evaluation.evaluation.latency import (
        aggregate_latency_ms,
        build_latency_aggregates,
    )
    from backend.scripts.Evaluation.evaluation.metrics import (
        build_metric_world_report as _build_metric_world_report,
        calculate_ndcg,
        get_task_metrics,
        get_weighted_recall as _get_weighted_recall,
        map_law_code_to_family,
    )
    from backend.scripts.Evaluation.retrieval.fusion import fuse_pools as _fuse_pools
    from backend.scripts.Evaluation.retrieval.fusion import softmax_normalize
    from backend.scripts.Evaluation.retrieval.temporal import should_include_document
    from backend.scripts.Evaluation.reporting.manifest import (
        enforce_full_run_provenance,
        fingerprint_records,
        git_provenance as _git_provenance,
        resolved_model_revision as _resolved_model_revision,
        safe_package_version as _safe_package_version,
        verify_model_revision,
    )
    from backend.scripts.Evaluation.reporting.serialization import assemble_baseline_summary

clean_query = clean_query_legacy


def get_gt_weights(gt_obj):
    return _get_gt_weights(gt_obj, ALPHA_CORE, ALPHA_SUPPORT)


def build_canonical_ground_truth(record, corpus_document_ids):
    return _build_canonical_ground_truth(
        record,
        corpus_document_ids,
        alpha_core=ALPHA_CORE,
        alpha_support=ALPHA_SUPPORT,
    )


def get_weighted_recall(results, cfg, k, is_filtered=True):
    return _get_weighted_recall(
        results,
        cfg,
        k,
        is_filtered=is_filtered,
        alpha_core=ALPHA_CORE,
        alpha_support=ALPHA_SUPPORT,
    )


def build_metric_world_report(results, configurations, reported_k=BASELINE_REPORTED_K):
    return _build_metric_world_report(
        results,
        configurations,
        reported_k=reported_k,
        alpha_core=ALPHA_CORE,
        alpha_support=ALPHA_SUPPORT,
    )


def fuse_pools(
    bm25_pool,
    faiss_pool,
    w_bm25,
    fusion_type="minmax",
    temp_bm25=None,
    temp_faiss=None,
):
    if temp_bm25 is None:
        temp_bm25 = _eval_config.get("fusion", {}).get("temperature_bm25", 5.0)
    if temp_faiss is None:
        temp_faiss = _eval_config.get("fusion", {}).get("temperature_faiss", 0.5)
    return _fuse_pools(
        bm25_pool,
        faiss_pool,
        w_bm25,
        fusion_type=fusion_type,
        temp_bm25=temp_bm25,
        temp_faiss=temp_faiss,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Pipeline")
    parser.add_argument(
        "--temporal-filter",
        choices=["on", "off"],
        default="on",
        help="Enable or disable temporal filtering of search pools (default: 'on')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom evaluation config yaml file (default: backend/scripts/Evaluation/eval_config.yaml)"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Run only the first N benchmark queries (intended for smoke tests; omit for a full run)"
    )
    args = parser.parse_args()

    selected_config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    with open(selected_config_path, "r", encoding="utf-8") as handle:
        selected_mode = yaml.safe_load(handle).get("evaluation", {}).get("mode", "legacy")
    output_file = run_evaluation_pipeline(
        temporal_filter=args.temporal_filter,
        config_path=selected_config_path,
        max_queries=args.max_queries,
    )
    if selected_mode != BASELINE_MODE:
        generate_paper_tables(output_file)
        perform_pairwise_significance_tests(output_file, k=10)
        render_fusion_ablation_table(output_file)
        profile_latency_tradeoff(output_file)
        print_failure_mode_analysis(output_file)
        generate_latex_tables_source(output_file)
    else:
        print("Baseline mode complete. Advanced/tuned reporting was intentionally skipped.")
