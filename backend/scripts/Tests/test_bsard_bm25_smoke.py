import copy
import json
import math
from pathlib import Path

import pytest
import yaml

from scripts.Evaluation.benchmarking.bsard_adapter import BSARDDataset
from scripts.Evaluation.experiments import bsard_bm25_smoke as smoke


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/configs/bsard_external_zero_shot_v1.yaml"
)


def _queries():
    return [
        {
            "query_id": "274",
            "query_text": "Question française exacte ?",
            "raw_relevant_document_ids": ["1", "2", "2"],
            "relevant_document_ids": ["1", "2"],
        }
    ]


def _records():
    return [
        {
            "query_id": "274",
            "ranking": [
                {"rank": 1, "standard_law_id": "1", "score": 2.5},
                {"rank": 2, "standard_law_id": "2", "score": 1.5},
            ],
        }
    ]


def test_document_mapping_is_article_only_and_preserves_identity():
    mapped = smoke.map_documents(
        [{"document_id": "9", "document_text": "Article officiel.", "source_metadata": {"code": "Never prepend"}}]
    )
    assert mapped == [{"standard_law_id": "9", "bm25_text": "Article officiel."}]


def test_bm25_contract_is_frozen_to_shared_project_defaults():
    assert smoke.BM25_PARAMETERS == {
        "implementation": "rank_bm25.BM25Okapi",
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25,
    }
    assert smoke.TOKENIZATION_POLICY["tokenizer"] == "python_str_split_whitespace"
    assert smoke.TOKENIZATION_POLICY["case_handling"] == "str.lower"
    assert smoke.TOKENIZATION_POLICY["stemming"] is False
    assert smoke.TOKENIZATION_POLICY["lemmatization"] is False


def test_retrieval_calls_shared_bm25_once_per_query_at_requested_depth(monkeypatch):
    calls = []

    def fake_search(index, documents, query_text, pool_k):
        calls.append((index, documents, query_text, pool_k))
        return [{"rank": 1, "standard_law_id": "1", "score": 1.0}]

    monkeypatch.setattr(smoke, "search_bm25", fake_search)
    records, latencies = smoke.retrieve_rankings(
        "index", [{"standard_law_id": "1", "bm25_text": "a"}], _queries(), top_k=500
    )
    assert records[0]["query_id"] == "274"
    assert calls == [("index", [{"standard_law_id": "1", "bm25_text": "a"}], "Question française exacte ?", 500)]
    assert len(latencies) == 1 and latencies[0] >= 0.0


def test_valid_ranking_contract_passes():
    smoke.validate_rankings(_records(), _queries(), {"1", "2"})


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda records: records.append(copy.deepcopy(records[0])), "qid coverage"),
        (lambda records: records[0].update(query_id="999"), "qid coverage"),
        (lambda records: records[0]["ranking"].append(copy.deepcopy(records[0]["ranking"][0])), "duplicate"),
        (lambda records: records[0]["ranking"][0].update(standard_law_id="999"), "unknown"),
        (lambda records: records[0]["ranking"][0].update(rank=2), "rank"),
        (lambda records: records[0]["ranking"][0].update(score=math.nan), "finite"),
        (lambda records: records[0]["ranking"][0].update(score=math.inf), "finite"),
    ],
)
def test_invalid_rankings_fail_closed(mutation, match):
    records = copy.deepcopy(_records())
    mutation(records)
    with pytest.raises(smoke.BSARDBM25SmokeError, match=match):
        smoke.validate_rankings(records, _queries(), {"1", "2"})


def test_ranking_depth_above_limit_fails_closed():
    record = {"query_id": "274", "ranking": []}
    record["ranking"] = [
        {"rank": rank, "standard_law_id": str(rank), "score": 1.0}
        for rank in range(1, 4)
    ]
    with pytest.raises(smoke.BSARDBM25SmokeError, match="length"):
        smoke.validate_rankings([record], _queries(), {"1", "2", "3"}, maximum_depth=2)


def test_ranking_fingerprint_depends_on_ordered_ids_not_scores():
    records = _records()
    original = smoke.ranking_fingerprint(records)
    score_changed = copy.deepcopy(records)
    score_changed[0]["ranking"][0]["score"] = 999.0
    assert smoke.ranking_fingerprint(score_changed) == original
    order_changed = copy.deepcopy(records)
    order_changed[0]["ranking"].reverse()
    assert smoke.ranking_fingerprint(order_changed) != original


def test_official_and_safe_qid274_semantics_remain_distinct():
    official, safe = smoke.evaluate_records(_records(), _queries(), {"1", "2"})
    assert official["metrics"]["recall@100"] == pytest.approx(2 / 3)
    assert safe["metrics"]["recall@100"] == 1.0
    assert official["metric_semantics"] == "bsard_official_compat"
    assert safe["metric_semantics"] == "bsard_safe_canonical"


def test_coverage_diagnostics_use_unique_reference_and_query_denominators():
    coverage = smoke.coverage_diagnostics(_records(), _queries(), depths=(1, 2))
    assert coverage["1"] == {
        "unique_relevant_reference_denominator": 2,
        "retrieved_relevant_references": 1,
        "missed_relevant_references": 1,
        "reference_level_candidate_miss_rate": 0.5,
        "candidate_recall": 0.5,
        "query_denominator": 1,
        "zero_relevant_query_count": 0,
        "zero_relevant_query_rate": 0.0,
    }
    assert coverage["2"]["candidate_recall"] == 1.0


def test_zero_relevant_query_is_counted():
    records = copy.deepcopy(_records())
    records[0]["ranking"] = [{"rank": 1, "standard_law_id": "3", "score": 1.0}]
    coverage = smoke.coverage_diagnostics(records, _queries(), depths=(1,))
    assert coverage["1"]["zero_relevant_query_count"] == 1
    assert coverage["1"]["missed_relevant_references"] == 2


def test_paper_comparison_is_not_directly_comparable():
    metrics = {name: value + 0.01 for name, value in smoke.PAPER_BM25["metrics"].items()}
    comparison = smoke.compare_with_paper(metrics)
    assert comparison["classification"] == "NOT_DIRECTLY_COMPARABLE"
    assert all(value == pytest.approx(0.01) for value in comparison["absolute_delta_fresh_minus_paper"].values())
    assert comparison["paper"]["parameters"] == {"k1": 1.0, "b": 0.6, "tuned_on": "BSARD train"}


def test_latency_summary_reports_separate_build_query_and_replay_times():
    result = smoke.latency_summary(
        index_build_seconds=1.0,
        query_latencies_seconds=[0.001, 0.002, 0.004],
        total_retrieval_seconds=0.01,
        replay_seconds=0.02,
        peak_rss_mb=10.0,
    )
    assert result["index_build_seconds"] == 1.0
    assert result["query_latency_ms"]["mean"] == pytest.approx(7 / 3)
    assert result["query_latency_ms"]["p50"] == 2.0
    assert result["total_test_retrieval_seconds"] == 0.01
    assert result["deterministic_replay_seconds"] == 0.02


def test_config_gate_rejects_model_or_synthetic_enablement():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    smoke._validate_config(config)
    config["protocol"]["execute_models"] = True
    with pytest.raises(smoke.BSARDBM25SmokeError, match="model execution"):
        smoke._validate_config(config)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    config["dataset"]["include_synthetic_by_default"] = True
    with pytest.raises(smoke.BSARDBM25SmokeError, match="synthetic"):
        smoke._validate_config(config)


def test_atomic_publication_is_strict_json_and_complete(tmp_path):
    run_dir = tmp_path / "run"
    artifacts = {
        "retrieval_rankings.json": {"records": []},
        "summary_metrics.json": {"value": 1.0},
        "run_manifest.json": {"run_id": "run"},
    }
    smoke.atomic_publish(run_dir, CONFIG_PATH, artifacts)
    assert {path.name for path in run_dir.iterdir()} == {
        *artifacts,
        "eval_config.snapshot.yaml",
    }
    assert json.loads((run_dir / "summary_metrics.json").read_text())["value"] == 1.0


def test_failed_publication_cleans_partial_and_leaves_no_final(tmp_path):
    run_dir = tmp_path / "run"
    artifacts = {
        "retrieval_rankings.json": {"records": []},
        "summary_metrics.json": {"bad": math.nan},
        "run_manifest.json": {"run_id": "run"},
    }
    with pytest.raises(ValueError):
        smoke.atomic_publish(run_dir, CONFIG_PATH, artifacts)
    assert not run_dir.exists()
    assert not run_dir.with_name("run.partial").exists()


def test_full_mocked_runner_publishes_provenance_without_models(tmp_path, monkeypatch):
    documents = (
        {"document_id": "1", "document_text": "alpha"},
        {"document_id": "2", "document_text": "beta"},
    )
    dataset = BSARDDataset(
        documents=documents,
        human_train=({"query_id": "1"},),
        human_test=tuple(_queries()),
        provenance={"raw_sha256": {"articles_fr.csv": "a" * 64}},
        qrel_statistics={
            "test": {
                "raw_qrel_occurrence_count": 3,
                "unique_qrel_count": 2,
                "duplicate_qrel_count": 1,
            }
        },
        fingerprints={
            "corpus_ordered": "b" * 64,
            "test_split": "c" * 64,
            "test_unique_qrels": "d" * 64,
        },
    )
    monkeypatch.setattr(smoke, "EXPECTED_DOCUMENT_COUNT", 2)
    monkeypatch.setattr(smoke, "EXPECTED_TRAIN_QUERY_COUNT", 1)
    monkeypatch.setattr(smoke, "EXPECTED_TEST_QUERY_COUNT", 1)
    monkeypatch.setattr(smoke, "EXPECTED_TEST_RAW_QRELS", 3)
    monkeypatch.setattr(smoke, "EXPECTED_TEST_UNIQUE_QRELS", 2)
    monkeypatch.setattr(smoke, "load_bsard", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(smoke, "build_bm25_index", lambda documents: object())
    monkeypatch.setattr(
        smoke,
        "retrieve_rankings",
        lambda *args, **kwargs: (copy.deepcopy(_records()), [0.001]),
    )
    monkeypatch.setattr(smoke, "_git_head", lambda project_root: "e" * 40)

    final_dir = smoke.run(raw_root=tmp_path, config_path=CONFIG_PATH, output_root=tmp_path)
    manifest = json.loads((final_dir / "run_manifest.json").read_text())
    summary = json.loads((final_dir / "summary_metrics.json").read_text())
    rankings = json.loads((final_dir / "retrieval_rankings.json").read_text())
    assert manifest["git_commit"] == "e" * 40
    assert manifest["runner"]["sha256"] == smoke.sha256_file(Path(smoke.__file__))
    assert manifest["config"]["sha256"] == smoke.sha256_file(CONFIG_PATH)
    assert manifest["models_used"] == []
    assert manifest["train_queries_used_for_retrieval_or_tuning"] is False
    assert manifest["ranking_count"] == 1
    assert summary["determinism"]["identical"] is True
    assert rankings["query_count"] == 1
    assert not list(tmp_path.glob("*.partial"))


def test_main_requires_explicit_external_raw_root(monkeypatch):
    monkeypatch.delenv("BSARD_RAW_ROOT", raising=False)
    with pytest.raises(smoke.BSARDBM25SmokeError, match="explicit"):
        smoke.main([])
