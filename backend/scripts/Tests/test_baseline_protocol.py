import json
import copy
from pathlib import Path

import pytest
import yaml

from backend.scripts.Evaluation import run_rag_evaluation as evaluator


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
LAWS_CONFIG_PATH = PROJECT_ROOT / "backend" / "dataLaw" / "laws_config.json"
EVAL_CONFIG_PATH = PROJECT_ROOT / "backend" / "scripts" / "Evaluation" / "eval_config.yaml"
BENCHMARK_PATH = DATA_DIR / "judgments_evaluation.json"
BASELINE_REQUIREMENTS_PATH = (
    PROJECT_ROOT / "backend" / "scripts" / "Evaluation" / "requirements-baseline.txt"
)


@pytest.fixture(scope="module")
def baseline_config():
    with open(EVAL_CONFIG_PATH, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def frozen_corpus(baseline_config):
    return evaluator.load_frozen_baseline_corpus(
        LAWS_CONFIG_PATH, DATA_DIR, baseline_config["corpus"]
    )


def test_q0_only_normalizes_whitespace():
    assert evaluator.normalize_query_q0("  Tranh chấp\n  hợp đồng\t vay  ") == "Tranh chấp hợp đồng vay"


def test_q0_does_not_prepend_procedural_stage():
    query = "Bản án giám đốc thẩm về tranh chấp hợp đồng"
    assert evaluator.normalize_query_q0(query) == query
    assert not evaluator.normalize_query_q0(query).startswith("giám đốc thẩm Bản án")


def test_q0_does_not_concatenate_metadata_or_ground_truth():
    record = {
        "query": "Tranh chấp hợp đồng",
        "metadata": {"case_type": "SECRET_CASE", "year": 2024},
        "query_intent": "SECRET_INTENT",
        "ground_truth": {"core": [{"article": "SECRET_ARTICLE"}]},
    }
    assert evaluator.build_query_text(record, "q0") == "Tranh chấp hợp đồng"


def test_d0_is_title_plus_original_content_only():
    document = {
        "title": "Điều mẫu",
        "content": "Nội dung gốc",
        "summary": "GEMINI_SUMMARY",
        "lexical_keywords": ["GEMINI_LEXICAL"],
        "keywords": ["GEMINI_KEYWORD"],
        "tags": ["GEMINI_TAG"],
    }
    text = evaluator.build_document_text(document, "d0")
    assert text == "Tiêu đề: Điều mẫu. Nội dung: Nội dung gốc"
    assert "GEMINI_" not in text


def test_d1_contains_declared_enrichment_fields():
    document = {
        "title": "Điều mẫu",
        "content": "Nội dung gốc",
        "summary": "Tóm tắt",
        "lexical_keywords": ["từ khóa lexical"],
        "keywords": ["từ khóa thường"],
        "tags": ["TAG_MUST_NOT_APPEAR"],
    }
    text = evaluator.build_document_text(document, "d1")
    assert text == (
        "Tiêu đề: Điều mẫu. Nội dung: Nội dung gốc. "
        "Tóm tắt: Tóm tắt. Từ khóa: từ khóa lexical từ khóa thường"
    )
    assert "TAG_MUST_NOT_APPEAR" not in text


def test_frozen_corpus_is_exactly_1811_articles(frozen_corpus):
    law_files, inventory, total = frozen_corpus
    assert total == 1811
    assert sum(item["count"] for item in inventory) == 1811
    assert len(law_files) == 5


def test_test_law_and_mau_so_52_are_not_loaded(frozen_corpus):
    _, inventory, _ = frozen_corpus
    loaded_keys = {item["key"] for item in inventory}
    assert "test_law" not in loaded_keys
    assert "mau_so_52" not in loaded_keys


def test_missing_corpus_fails_fast(tmp_path):
    laws_config = tmp_path / "laws_config.json"
    laws_config.write_text(
        json.dumps([{"key": "required_law"}]), encoding="utf-8"
    )
    corpus_config = {
        "required_keys": ["required_law"],
        "expected_counts": {"required_law": 1},
        "expected_total": 1,
    }
    with pytest.raises(FileNotFoundError, match="Required baseline corpus artifact is missing"):
        evaluator.load_frozen_baseline_corpus(laws_config, tmp_path, corpus_config)


def test_d0_d1_preserve_article_order(frozen_corpus):
    law_files, _, _ = frozen_corpus
    d0 = evaluator.load_documents(law_files, "d0", "whitespace")
    d1 = evaluator.load_documents(law_files, "d1", "whitespace")
    assert [item["standard_law_id"] for item in d0] == [item["standard_law_id"] for item in d1]


def test_bm25_and_dense_use_same_d0_source_text(frozen_corpus):
    law_files, _, _ = frozen_corpus
    documents = evaluator.load_documents(law_files, "d0", "whitespace")
    assert len(documents) == 1811
    assert all(item["bm25_text"] == item["vector_text"] for item in documents)


def test_marriage_law_alias_normalizes_and_matches():
    assert evaluator.canonicalize_law_code(
        "luat_hon_nhan_va_gia_dinh_2014"
    ) == "luat_hon_nhan_gia_dinh_2014"
    assert evaluator.is_match(
        "điều 38 luat_hon_nhan_gia_dinh_2014",
        "Điều 38 luat_hon_nhan_va_gia_dinh_2014",
    )


def test_gt_weights_survive_json_save_reload(tmp_path):
    evaluator.ALPHA_CORE = 3
    evaluator.ALPHA_SUPPORT = 1
    record = {
        "ground_truth": {
            "core": [{
                "article": "Điều 1",
                "law_code": "bo_luat_dan_su_2015",
                "domain": "substantive",
                "source_type": "explicit",
            }],
            "supporting": [{
                "article": "Điều 2",
                "law_code": "bo_luat_dan_su_2015",
                "domain": "substantive",
                "source_type": "inferred",
            }],
        }
    }
    result = evaluator.build_canonical_ground_truth(
        record,
        {
            "điều 1 bo_luat_dan_su_2015",
            "điều 2 bo_luat_dan_su_2015",
        },
    )
    output = tmp_path / "result.json"
    output.write_text(json.dumps([result]), encoding="utf-8")
    reloaded = json.loads(output.read_text(encoding="utf-8"))[0]
    assert reloaded["gt_weights"] == {
        "điều 1 bo_luat_dan_su_2015": 3,
        "điều 2 bo_luat_dan_su_2015": 1,
    }


def test_temporal_none_keeps_future_law_candidate():
    future_law = {
        "law_year": 2024,
        "active_start": 2024,
        "active_end": 2099,
    }
    assert evaluator.should_include_document(
        future_law, "none", judgment_year=2022, target_year=2022
    )
    assert not evaluator.should_include_document(
        future_law, "judgment_year", judgment_year=2022
    )


def test_dynamic_weight_comes_from_config():
    weights = {
        "exact_citation": 0.11,
        "procedural": 0.22,
        "substance": 0.33,
        "mixed": 0.44,
    }
    assert evaluator.get_dynamic_weight("procedural", weights) == 0.22
    assert evaluator.get_dynamic_weight("unknown", weights) == 0.44


def test_baseline_mode_disables_reranker_and_tuning():
    assert evaluator.get_enabled_configurations("baseline") == [
        "BM25", "FAISS", "Hybrid_Static_Fixed"
    ]
    assert evaluator.should_initialize_reranker("baseline") is False
    assert evaluator.should_run_tuning("baseline") is False
    assert "Hybrid_Static_Opt" not in evaluator.get_enabled_configurations("baseline")


def test_baseline_config_freezes_required_protocol(baseline_config):
    evaluation = baseline_config["evaluation"]
    assert evaluation["mode"] == "baseline"
    assert evaluation["query_representation"] == "q0"
    assert evaluation["document_representation"] == "d0"
    assert evaluation["temporal_filter"] == "none"
    assert baseline_config["models"]["embedding_model"] == "keepitreal/vietnamese-sbert"
    assert baseline_config["models"]["embedding_revision"] == (
        "a9467ef2ef47caa6448edeabfd8e5e5ce0fa2a23"
    )
    assert baseline_config["models"]["local_files_only"] is True
    assert baseline_config["bm25"]["tokenizer"] == "whitespace"
    assert baseline_config["fusion"]["method"] == "minmax"
    assert baseline_config["fusion"]["static_w_bm25"] == 0.5


def test_benchmark_query_count_is_107():
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as handle:
        benchmark = json.load(handle)
    assert len(benchmark) == 107


def test_metrics_at_5_10_20_slice_the_same_stored_ranking():
    evaluator.ALPHA_CORE = 3
    evaluator.ALPHA_SUPPORT = 1
    ranking = [
        {"rank": rank, "standard_law_id": f"điều {rank} luat_a", "score": 21 - rank}
        for rank in range(1, 21)
    ]
    results = [{
        "core_gt_raw": ["điều 3 luat_a", "điều 7 luat_a", "điều 15 luat_a"],
        "supporting_gt_raw": [],
        "core_gt_filtered": ["điều 3 luat_a", "điều 7 luat_a", "điều 15 luat_a"],
        "supporting_gt_filtered": [],
        "substance_gt_raw": ["điều 3 luat_a", "điều 7 luat_a", "điều 15 luat_a"],
        "procedural_gt_raw": [],
        "substance_gt_filtered": ["điều 3 luat_a", "điều 7 luat_a", "điều 15 luat_a"],
        "procedural_gt_filtered": [],
        "gt_weights": {
            "điều 3 luat_a": 3,
            "điều 7 luat_a": 3,
            "điều 15 luat_a": 3,
        },
        "metrics": {"BM25": {"top_k": ranking}},
    }]
    ranking_before = copy.deepcopy(ranking)

    report = evaluator.build_metric_world_report(results, ["BM25"], (5, 10, 20))

    assert ranking == ranking_before
    assert report["closed_world"]["BM25"]["5"]["recall"] == pytest.approx(1 / 3)
    assert report["closed_world"]["BM25"]["10"]["recall"] == pytest.approx(2 / 3)
    assert report["closed_world"]["BM25"]["20"]["recall"] == pytest.approx(1.0)


def test_closed_world_ndcg_idcg_excludes_out_of_corpus_core_weight():
    raw_weights = {
        "điều 999 law_outside": 3,
        "điều 2 law_inside": 1,
    }
    closed_world_gts = ["điều 2 law_inside"]

    ndcg = evaluator.calculate_ndcg(
        ["điều 2 law_inside"], closed_world_gts, k=10, gt_weights=raw_weights
    )

    assert ndcg == pytest.approx(1.0)


def test_closed_world_and_open_world_are_separate_metric_namespaces():
    evaluator.ALPHA_CORE = 3
    evaluator.ALPHA_SUPPORT = 1
    result = {
        "core_gt_raw": ["điều 999 law_outside"],
        "supporting_gt_raw": ["điều 2 law_inside"],
        "core_gt_filtered": [],
        "supporting_gt_filtered": ["điều 2 law_inside"],
        "substance_gt_raw": ["điều 999 law_outside", "điều 2 law_inside"],
        "procedural_gt_raw": [],
        "substance_gt_filtered": ["điều 2 law_inside"],
        "procedural_gt_filtered": [],
        "gt_weights": {
            "điều 999 law_outside": 3,
            "điều 2 law_inside": 1,
        },
        "metrics": {
            "BM25": {
                "top_k": [{"rank": 1, "standard_law_id": "điều 2 law_inside", "score": 1.0}]
            }
        },
    }

    report = evaluator.build_metric_world_report([result], ["BM25"], (10,))

    assert set(report) == {"closed_world", "open_world"}
    assert report["closed_world"]["BM25"]["10"]["recall"] == pytest.approx(1.0)
    assert report["open_world"]["BM25"]["10"]["recall"] == pytest.approx(0.5)
    assert report["closed_world"]["BM25"]["10"]["weighted_recall"] == pytest.approx(1.0)
    assert report["open_world"]["BM25"]["10"]["weighted_recall"] == pytest.approx(0.25)


def test_ground_truth_coverage_uses_alias_normalization_and_exact_article_membership():
    benchmark = [
        {
            "ground_truth": {
                "core": [{
                    "article": "Khoản 2 Điều 38",
                    "law_code": "luat_hon_nhan_va_gia_dinh_2014",
                }],
                "supporting": [{
                    "article": "Điều 999",
                    "law_code": "bo_luat_dan_su_2015",
                }],
            }
        },
        {
            "ground_truth": {
                "core": [{
                    "article": "Điều 1",
                    "law_code": "law_outside",
                }],
                "supporting": [],
            }
        },
    ]
    corpus_ids = {"điều 38 luat_hon_nhan_gia_dinh_2014"}

    coverage = evaluator.compute_ground_truth_coverage(benchmark, corpus_ids)

    assert coverage == {
        "total_labels": 3,
        "in_corpus_labels": 1,
        "out_of_corpus_labels": 2,
        "coverage_ratio": pytest.approx(1 / 3),
        "queries_with_at_least_one_in_corpus_gt": 1,
        "queries_without_in_corpus_gt": 1,
    }


def test_latency_fields_are_non_negative_and_aggregated_per_method():
    required_fields = {
        "bm25_retrieval_ms",
        "dense_encode_ms",
        "faiss_search_ms",
        "dense_total_ms",
        "fusion_ms",
        "hybrid_serial_ms",
        "hybrid_parallel_estimate_ms",
    }
    latency = {
        "bm25_retrieval_ms": 1.0,
        "dense_encode_ms": 2.0,
        "faiss_search_ms": 0.5,
        "dense_total_ms": 2.5,
        "fusion_ms": 0.25,
        "hybrid_serial_ms": 3.75,
        "hybrid_parallel_estimate_ms": 2.75,
    }
    assert set(latency) == required_fields
    assert all(value >= 0 for value in latency.values())

    aggregate = evaluator.build_latency_aggregates([{"latency_ms": latency}])

    assert aggregate["BM25"]["mean_ms"] == 1.0
    assert aggregate["FAISS"]["mean_ms"] == 2.5
    assert aggregate["Hybrid_Static_Fixed"]["mean_ms"] == 3.75
    assert aggregate["Hybrid_Static_Fixed"]["parallel_estimate"]["estimated"] is True


def test_model_revision_requested_and_resolved_must_match(baseline_config):
    revision = baseline_config["models"]["embedding_revision"]
    evaluator.verify_model_revision(revision, revision)
    with pytest.raises(RuntimeError, match="Embedding revision mismatch"):
        evaluator.verify_model_revision(revision, "different-revision")


def test_summary_contract_declares_closed_world_primary_at_10():
    metadata = {"run_id": "smoke", "primary_world": "closed_world", "primary_k": 10}
    worlds = {"closed_world": {"BM25": {}}, "open_world": {"BM25": {}}}
    coverage = {"total_labels": 1, "in_corpus_labels": 1}
    latency = {"BM25": {"mean_ms": 1.0, "median_ms": 1.0, "p95_ms": 1.0}}

    summary = evaluator.assemble_baseline_summary(metadata, worlds, coverage, latency)

    assert summary["primary_k"] == 10
    assert summary["primary_world"] == "closed_world"
    assert "closed_world" in summary and "open_world" in summary


def test_baseline_experiment_lock_pins_smoke_pass_environment():
    pins = set(BASELINE_REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines())
    assert {
        "torch==2.2.2",
        "numpy==1.26.4",
        "faiss-cpu==1.12.0",
        "rank-bm25==0.2.2",
        "sentence-transformers==5.5.0",
        "PyYAML==6.0.2",
    }.issubset(pins)
