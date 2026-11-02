import copy
import json
import math
from pathlib import Path

import pytest

from scripts.Evaluation.experiments import bsard_rrf_hybrid as rrf


KNOWN = {str(value) for value in range(1, 20)}


def item(document_id, score=999.0):
    return {"standard_law_id": str(document_id), "score": score}


def test_exact_one_based_formula_and_both_contributions():
    result = rrf.rrf_fuse([item(1), item(2)], [item(2), item(3)], KNOWN, expected_depth=2, output_depth=3)
    scores = {row["standard_law_id"]: row["score"] for row in result}
    assert scores["1"] == 1 / 61
    assert scores["2"] == 1 / 62 + 1 / 61
    assert scores["3"] == 1 / 62


def test_single_source_contribution_and_no_raw_score_normalization():
    first = rrf.rrf_fuse([item(1, -1000), item(2, 9999)], [item(3, math.pi), item(4, -5)], KNOWN, expected_depth=2, output_depth=4)
    second = rrf.rrf_fuse([item(1, 0), item(2, 0)], [item(3, 0), item(4, 0)], KNOWN, expected_depth=2, output_depth=4)
    assert first == second


def test_equal_weights_and_source_order_invariance():
    left, right = [item(1), item(2)], [item(2), item(3)]
    assert rrf.rrf_fuse(left, right, KNOWN, expected_depth=2, output_depth=3) == rrf.rrf_fuse(right, left, KNOWN, expected_depth=2, output_depth=3)
    assert rrf.SOURCE_WEIGHTS == {"bm25": 1.0, "bge_m3": 1.0}


def test_numeric_tie_break_not_string_order():
    result = rrf.rrf_fuse([item(10), item(2)], [item(3), item(1)], KNOWN, expected_depth=2, output_depth=4)
    assert [row["standard_law_id"] for row in result] == ["3", "10", "1", "2"]


@pytest.mark.parametrize("bad_k", [10, 20, 30, 100])
def test_rrf_k_is_frozen(bad_k):
    with pytest.raises(rrf.BSARDRRFError, match="k"):
        rrf.rrf_fuse([item(1)], [item(2)], KNOWN, expected_depth=1, rrf_k=bad_k)


def test_duplicate_source_prediction_rejected():
    with pytest.raises(rrf.BSARDRRFError, match="duplicate"):
        rrf.rrf_fuse([item(1), item(1)], [item(2), item(3)], KNOWN, expected_depth=2)


def test_unknown_source_prediction_rejected():
    with pytest.raises(rrf.BSARDRRFError, match="unknown"):
        rrf.rrf_fuse([item(99)], [item(1)], KNOWN, expected_depth=1)


@pytest.mark.parametrize("source", [[item(1)], [item(1), item(2), item(3)]])
def test_source_depth_mismatch_rejected(source):
    with pytest.raises(rrf.BSARDRRFError, match="depth"):
        rrf.rrf_fuse(source, [item(1), item(2)], KNOWN, expected_depth=2)


def test_fused_output_unique_finite_and_depth_limited():
    result = rrf.rrf_fuse([item(1), item(2)], [item(2), item(3)], KNOWN, expected_depth=2, output_depth=2)
    assert len(result) == len({row["standard_law_id"] for row in result}) == 2
    assert all(math.isfinite(row["score"]) for row in result)


def test_top500_fused_output_is_unique_and_finite():
    known = {str(value) for value in range(1, 1001)}
    left = [item(value) for value in range(1, 501)]
    right = [item(value) for value in range(501, 1001)]
    result = rrf.rrf_fuse(left, right, known)
    assert len(result) == len({row["standard_law_id"] for row in result}) == 500
    assert all(math.isfinite(row["score"]) for row in result)


def test_deterministic_replay_has_identical_order_and_fingerprint():
    left, right = [item(1), item(2)], [item(2), item(3)]
    first = rrf.rrf_fuse(left, right, KNOWN, expected_depth=2, output_depth=3)
    second = rrf.rrf_fuse(left, right, KNOWN, expected_depth=2, output_depth=3)
    assert first == second
    assert rrf.sha256_json(first) == rrf.sha256_json(second)


def test_missing_and_extra_qid_rejected():
    queries = [{"query_id": "1"}]
    source = [{"query_id": "1", "ranking": [item(1)]}]
    with pytest.raises(rrf.BSARDRRFError, match="qid"):
        rrf.fuse_records([], source, queries, KNOWN)
    extra = source + [{"query_id": "2", "ranking": [item(2)]}]
    with pytest.raises(rrf.BSARDRRFError, match="qid"):
        rrf.fuse_records(extra, source, queries, KNOWN)


def test_source_artifact_fingerprint_mismatch_rejected():
    source = {"manifest": {"run_id": rrf.BM25_RUN_ID, "retrieval_fingerprint": "bad"}, "summary": {}, "rankings": {}}
    with pytest.raises(rrf.BSARDRRFError, match="retrieval fingerprint"):
        rrf.validate_source(source, kind="bm25", expected_run=rrf.BM25_RUN_ID, expected_retrieval=rrf.BM25_RETRIEVAL_FP, expected_result=rrf.BM25_RESULT_FP, dataset=None)


def test_query_bound_qrel_pair_diagnostics_count_repeated_doc_twice(monkeypatch):
    monkeypatch.setattr(rrf, "EXPECTED_SOURCE_UNION", {"both": 2, "bm25_only": 0, "bge_only": 0, "missed_both": 0})
    monkeypatch.setattr(rrf, "EXPECTED_UNIQUE_QRELS", 2)
    monkeypatch.setattr(rrf, "EXPECTED_SOURCE_UNION_ZERO", 0)
    queries = [{"query_id": "1", "relevant_document_ids": ["5"]}, {"query_id": "2", "relevant_document_ids": ["5"]}]
    records = [{"query_id": q, "ranking": [item(5)]} for q in ("1", "2")]
    monkeypatch.setattr(rrf, "OUTPUT_DEPTH", 1)
    result = rrf.source_membership_analysis(records, records, records, queries)
    assert result["source_top500_qrel_partition"]["both"] == 2
    control = rrf.source_complementarity_control(records, records, queries)
    assert control == {"qrel_partition": {"both": 2, "bm25_only": 0, "bge_only": 0, "missed_both": 0}, "relevant_qrels": 2, "zero_relevant_queries": 0}


def test_strict_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"x": NaN}')
    with pytest.raises(rrf.BSARDRRFError, match="non-finite"):
        rrf.strict_load(path)
    with pytest.raises(ValueError):
        rrf.strict_write(tmp_path / "write.json", {"x": math.inf})


def test_publish_atomic_and_exact(tmp_path):
    config = tmp_path / "c.yaml"; config.write_text("x: 1\n")
    names = {"retrieval_rankings.json", "summary_metrics.json", "fusion_analysis.json", "run_manifest.json"}
    final = tmp_path / "run"; rrf.publish(final, config, {name: {"ok": True} for name in names})
    assert {path.name for path in final.iterdir()} == names | {"eval_config.snapshot.yaml"}


def test_classification_rules():
    base = {metric: {"mean_delta": 0.01, "ci_lower": 0.001, "ci_upper": 0.02} for metric in rrf.PAIRED_METRICS}
    assert rrf.classify(base) == "RRF_IMPROVES_BGE"
    base["mrr@100"]["mean_delta"] = -0.01
    assert rrf.classify(base) == "RRF_IMPROVES_DEEP_COVERAGE_BUT_NOT_RANKING"
