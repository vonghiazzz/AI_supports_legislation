import math

import pytest

from scripts.Evaluation.evaluation.bsard_metrics import (
    BSARDMetricError,
    evaluate_official_compat,
    evaluate_safe_canonical,
    paired_bootstrap_mean_delta,
)


KNOWN = {str(value) for value in range(1, 701)}


def _official(gt, prediction):
    return evaluate_official_compat([gt], [prediction], known_document_ids=KNOWN)


def _safe(gt, prediction, **kwargs):
    return evaluate_safe_canonical(
        {"10": gt}, {"10": prediction}, known_document_ids=KNOWN, **kwargs
    )


def test_official_compat_perfect_ranking():
    result = _official(["1", "2"], ["1", "2", "3"])
    assert result["metric_semantics"] == "bsard_official_compat"
    assert result["metrics"] == {
        "recall@100": 1.0,
        "recall@200": 1.0,
        "recall@500": 1.0,
        "map@100": 1.0,
        "mrr@100": 1.0,
    }


def test_official_compat_all_wrong():
    assert all(value == 0.0 for value in _official(["1"], ["2", "3"])["metrics"].values())


def test_official_compat_first_relevant_at_rank_one():
    assert _official(["1"], ["1", "2"])["metrics"]["mrr@100"] == 1.0


def test_official_compat_first_relevant_after_rank_one():
    result = _official(["1"], ["2", "1", "3"])
    assert result["metrics"]["mrr@100"] == 0.5
    assert result["metrics"]["map@100"] == 0.5


def test_official_compat_multiple_positives_and_partial_retrieval():
    result = _official(["1", "2", "3"], ["1", "9", "3"])
    assert result["metrics"]["recall@100"] == pytest.approx(2 / 3)
    assert result["metrics"]["map@100"] == pytest.approx((1 + 2 / 3) / 3)


@pytest.mark.parametrize("cutoff", [100, 200, 500])
def test_official_compat_cutoff_boundary_is_inclusive(cutoff):
    prediction = [str(value) for value in range(2, cutoff + 1)] + ["1"]
    result = _official(["1"], prediction)
    assert result["metrics"][f"recall@{cutoff}"] == 1.0


@pytest.mark.parametrize("cutoff", [100, 200, 500])
def test_official_compat_item_after_cutoff_is_excluded(cutoff):
    prediction = [str(value) for value in range(2, cutoff + 2)] + ["1"]
    result = _official(["1"], prediction)
    assert result["metrics"][f"recall@{cutoff}"] == 0.0


def test_official_compat_macro_average():
    result = evaluate_official_compat(
        [["1"], ["2"]],
        [["1"], ["3"]],
        known_document_ids=KNOWN,
        query_ids=["a", "b"],
    )
    assert result["query_count"] == 2
    assert result["metrics"]["recall@100"] == 0.5


def test_official_compat_rejects_legacy_zip_truncation_case():
    with pytest.raises(BSARDMetricError, match="length mismatch"):
        evaluate_official_compat([["1"]], [], known_document_ids=KNOWN)


def test_official_compat_rejects_duplicate_unknown_and_empty_predictions():
    with pytest.raises(BSARDMetricError, match="duplicate"):
        _official(["1"], ["1", "1"])
    with pytest.raises(BSARDMetricError, match="unknown"):
        _official(["1"], ["9999"])
    with pytest.raises(BSARDMetricError, match="empty"):
        _official(["1"], [])


def test_safe_alignment_is_qid_based_and_input_order_independent():
    first = evaluate_safe_canonical(
        {"2": ["2"], "1": ["1"]},
        {"1": ["1"], "2": ["3"]},
        known_document_ids=KNOWN,
    )
    second = evaluate_safe_canonical(
        {"1": ["1"], "2": ["2"]},
        {"2": ["3"], "1": ["1"]},
        known_document_ids=KNOWN,
    )
    assert first == second
    assert first["metrics"]["recall@100"] == 0.5


def test_safe_rejects_duplicate_prediction():
    with pytest.raises(BSARDMetricError, match="duplicate"):
        _safe(["1"], ["1", "1"])


def test_safe_rejects_unknown_prediction_document():
    with pytest.raises(BSARDMetricError, match="unknown"):
        _safe(["1"], ["9999"])


def test_safe_rejects_missing_and_extra_qids():
    with pytest.raises(BSARDMetricError, match=r"missing=\['10'\]"):
        evaluate_safe_canonical({"10": ["1"]}, {}, known_document_ids=KNOWN)
    with pytest.raises(BSARDMetricError, match=r"extra=\['11'\]"):
        evaluate_safe_canonical(
            {"10": ["1"]}, {"10": ["1"], "11": ["1"]}, known_document_ids=KNOWN
        )


def test_safe_empty_prediction_has_deterministic_zero_metrics():
    result = _safe(["1", "2"], [])
    assert result["metric_semantics"] == "bsard_safe_canonical"
    assert result["per_query"]["10"]["prediction_count"] == 0
    assert all(value == 0.0 for value in result["metrics"].values())


def test_safe_multiple_qrels_metrics_are_correct():
    result = _safe(
        ["1", "2", "3"],
        ["1", "9", "3"],
        recall_cutoffs=(3,),
        ndcg_cutoffs=(3,),
        map_cutoff=3,
        mrr_cutoff=3,
    )
    assert result["metrics"]["recall@3"] == pytest.approx(2 / 3)
    assert result["metrics"]["candidate_recall@3"] == pytest.approx(2 / 3)
    assert result["metrics"]["map@3"] == pytest.approx((1 + 2 / 3) / 3)
    expected_ndcg = (1 + 1 / math.log2(4)) / (
        1 + 1 / math.log2(3) + 1 / math.log2(4)
    )
    assert result["metrics"]["ndcg@3"] == pytest.approx(expected_ndcg)


def test_safe_duplicate_raw_qrel_uses_unique_denominator():
    result = _safe(["1", "2", "2"], ["1", "2"])
    assert result["per_query"]["10"]["raw_relevant_count"] == 3
    assert result["per_query"]["10"]["unique_relevant_count"] == 2
    assert result["metrics"]["recall@100"] == 1.0
    assert result["metrics"]["map@100"] == 1.0


def test_safe_rejects_zero_relevant_set():
    with pytest.raises(BSARDMetricError, match="empty"):
        _safe([], ["1"])


def test_safe_rejects_result_longer_than_corpus():
    with pytest.raises(BSARDMetricError, match="unknown|longer than the corpus"):
        evaluate_safe_canonical(
            {"10": ["1"]},
            {"10": ["1", "2"]},
            known_document_ids={"1"},
        )


def test_qid_274_shape_keeps_official_and_safe_denominators_distinct():
    raw = ["1", "2", "3", "4", "5", "6", "5"]
    prediction = ["1", "2", "3", "4", "5", "6"]
    official = evaluate_official_compat(
        [raw], [prediction], known_document_ids=KNOWN, query_ids=["274"]
    )
    safe = evaluate_safe_canonical(
        {"274": raw}, {"274": prediction}, known_document_ids=KNOWN
    )
    assert official["per_query"][0]["raw_relevant_count"] == 7
    assert official["per_query"][0]["unique_relevant_count"] == 6
    assert official["metrics"]["recall@100"] == pytest.approx(6 / 7)
    assert official["metrics"]["map@100"] == pytest.approx(6 / 7)
    assert safe["per_query"]["274"]["unique_relevant_count"] == 6
    assert safe["metrics"]["recall@100"] == 1.0
    assert safe["metrics"]["map@100"] == 1.0


def test_safe_rejects_unknown_ground_truth_and_zero_or_invalid_cutoff():
    with pytest.raises(BSARDMetricError, match="unknown"):
        _safe(["9999"], ["1"])
    with pytest.raises(BSARDMetricError, match="positive integers"):
        _safe(["1"], ["1"], recall_cutoffs=(0,))


def test_paired_bootstrap_is_deterministic_and_reports_delta():
    first = paired_bootstrap_mean_delta([0.0, 0.5], [0.5, 0.5], samples=100, seed=42)
    second = paired_bootstrap_mean_delta([0.0, 0.5], [0.5, 0.5], samples=100, seed=42)
    assert first == second
    assert first["mean_delta"] == 0.25
    assert first["samples"] == 100
    assert first["seed"] == 42


def test_metrics_consume_ranked_ids_without_reordering():
    first_rank = _safe(["1"], ["1", "2"])["metrics"]["mrr@100"]
    second_rank = _safe(["1"], ["2", "1"])["metrics"]["mrr@100"]
    assert first_rank == 1.0
    assert second_rank == 0.5
