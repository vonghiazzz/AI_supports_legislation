import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import backend.scripts.Evaluation.run_experiment as experiment_runner
from backend.scripts.Evaluation.experiments import exp03_static_fusion_cv as exp03
from backend.scripts.Evaluation.experiments import (
    exp04_selective_dynamic_hybrid_nested_cv as exp04,
)
from backend.scripts.Evaluation.reporting import serialization
from backend.scripts.Evaluation.reporting.serialization import write_exp04_artifacts


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/configs/exp04_selective_dynamic_hybrid_nested_cv.yaml"
)


def _load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _set_nested(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _pool(prefix, scores=None, shared=()):
    scores = scores or [31.0 - rank for rank in range(1, 31)]
    shared = list(shared)
    return [
        {
            "rank": rank,
            "standard_law_id": (
                shared[rank - 1] if rank <= len(shared) else f"{prefix}-{rank}"
            ),
            "score": float(scores[rank - 1]),
        }
        for rank in range(1, 31)
    ]


def _candidate(query_id=1, query="query 2024"):
    relevant = "bm25-1"
    return {
        "query_id": query_id,
        "query": query,
        "core_gt_raw": [relevant],
        "supporting_gt_raw": [],
        "core_gt_filtered": [relevant],
        "supporting_gt_filtered": [],
        "substance_gt_raw": [relevant],
        "procedural_gt_raw": [],
        "substance_gt_filtered": [relevant],
        "procedural_gt_filtered": [],
        "explicit_gts": [relevant],
        "inferred_gts": [],
        "gt_weights": {relevant: 3},
        "bm25_pool": _pool("bm25"),
        "dense_pool": _pool("dense"),
    }


def _view(query="query 2024", bm25=None, dense=None):
    return {
        "query": query,
        "bm25_pool": bm25 or _pool("bm25"),
        "dense_pool": dense or _pool("dense"),
    }


def _legal_pool(prefix):
    return [
        {
            "rank": rank,
            "standard_law_id": f"Điều {rank} {prefix}_law",
            "score": float(31 - rank),
        }
        for rank in range(1, 31)
    ]


def _set_synthetic_ground_truth(record, relevant):
    for key in (
        "core_gt_raw",
        "core_gt_filtered",
        "substance_gt_raw",
        "substance_gt_filtered",
        "explicit_gts",
    ):
        record[key] = [relevant]
    for key in (
        "supporting_gt_raw",
        "supporting_gt_filtered",
        "procedural_gt_raw",
        "procedural_gt_filtered",
        "inferred_gts",
    ):
        record[key] = []
    record["gt_weights"] = {relevant: 3}


def _nested_candidate(query_id, label_kind):
    relevant = {
        "dense": "Điều 10 dense_law",
        "bm25": "Điều 10 bm25_law",
        "stay": "Điều 1 bm25_law",
    }[label_kind]
    query = {
        "dense": "dense " * 40,
        "bm25": "bm25 ",
        "stay": "stay " * 10,
    }[label_kind]
    record = {
        "query_id": query_id,
        "query": f"{query}{query_id}",
        "bm25_pool": _legal_pool("bm25"),
        "dense_pool": _legal_pool("dense"),
    }
    _set_synthetic_ground_truth(record, relevant)
    return record


def _nested_fixture(fold_count):
    records = []
    mapping = {}
    query_id = 1
    for fold in range(fold_count):
        for label_kind in ("dense", "bm25", "stay", "stay"):
            records.append(_nested_candidate(query_id, label_kind))
            mapping[query_id] = fold
            query_id += 1
    return records, mapping


def test_exp04_config_validator_accepts_frozen_config():
    protocol = exp04._validate_exp04_protocol(_load_config())

    assert protocol == {
        "protocol_id": exp04.EXP04_PROTOCOL_ID,
        "experiment_id": exp04.EXP04_EXPERIMENT_ID,
        "mode": exp04.EXP04_MODE,
        "query_count": 107,
        "pool_k": 30,
        "stored_k": 20,
        "reported_k": (5, 10, 20),
        "fold_count": 5,
        "fold_sizes": (22, 22, 21, 21, 21),
        "c_grid": (0.01, 0.1, 1.0),
        "threshold_grid": (0.0, 0.5, 1.0),
        "action_pairs": exp04.EXP04_ACTION_PAIRS,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol", "id"), "other"),
        (("evaluation", "mode"), "baseline"),
        (("evaluation", "query_representation"), "q1"),
        (("evaluation", "document_representation"), "d1"),
        (("evaluation", "pool_k"), 20),
        (("candidate_source", "fingerprint"), "other"),
        (("fold_source", "mapping_fingerprint"), "other"),
        (("control", "w_bm25"), 0.4),
        (("policy", "action_pairs"), []),
        (("feature_contract", "version"), "v2"),
        (("feature_contract", "ordered_features"), ["other"]),
        (("feature_contract", "legal_ref_regex"), "other"),
        (("labeling", "abs_tol"), 1e-6),
        (("model", "solver"), "lbfgs"),
        (("model", "class_weight"), None),
        (("tuning", "c_grid"), [0.1, 1.0]),
        (("tuning", "threshold_grid"), [0.0, 1.0]),
        (("tuning", "tie_break"), ["other"]),
        (("metric_protocol", "primary_metric"), "recall"),
        (("statistics", "bootstrap_samples"), 1000),
        (("statistics", "sign_flip_samples"), 1000),
    ],
)
def test_exp04_config_validator_rejects_protocol_drift(path, value):
    config = _load_config()
    _set_nested(config, path, value)

    with pytest.raises(ValueError, match="EXP-04 requires"):
        exp04._validate_exp04_protocol(config)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda view: view["bm25_pool"].pop(), "exactly 30"),
        (lambda view: view["dense_pool"].__setitem__(1, {**view["dense_pool"][1], "rank": 3}), "ranks"),
        (lambda view: view["bm25_pool"].__setitem__(1, {**view["bm25_pool"][1], "standard_law_id": view["bm25_pool"][0]["standard_law_id"]}), "unique"),
        (lambda view: view["dense_pool"][0].__setitem__("score", math.nan), "finite"),
        (lambda view: view.pop("bm25_pool"), "inference-only"),
    ],
)
def test_candidate_input_validation_fails_closed(mutation, match):
    view = _view()
    mutation(view)

    with pytest.raises(ValueError, match=match):
        exp04.extract_feature_vector(view)


def test_feature_vector_exact_order_dtype_and_formulas():
    shared = [f"shared-{rank}" for rank in range(1, 6)]
    bm25 = _pool("bm25", shared=shared)
    dense = _pool("dense", shared=shared)
    query = "  Điều 12A theo BỘ LUẬT, khoản 2 năm 2024  "

    vector = exp04.extract_feature_vector(_view(query, bm25, dense))
    values = dict(zip(exp04.FEATURE_NAMES, vector))
    expected_sd = np.std(np.arange(30.0, 0.0, -1.0), ddof=0) / 29.0
    harmonic_10 = sum(1.0 / rank for rank in range(1, 11))
    expected_rro = (2.0 / harmonic_10) * sum(1.0 / (2 * rank) for rank in range(1, 6))

    assert vector.shape == (18,)
    assert vector.dtype == np.float64
    assert tuple(values) == exp04.FEATURE_NAMES
    assert values["query_log_char_count"] == pytest.approx(math.log1p(len(exp04.normalize_query_q0(query))))
    assert values["query_log_token_count"] == pytest.approx(math.log1p(9))
    assert values["query_type_token_ratio"] == 1.0
    assert values["query_digit_token_fraction"] == pytest.approx(3.0 / 9.0)
    assert values["query_log_legal_ref_count"] == pytest.approx(math.log1p(2))
    assert values["query_log_legal_doc_count"] == pytest.approx(math.log1p(1))
    assert values["bm25_top1_score"] == 30.0
    assert values["bm25_gap_1_2_norm"] == pytest.approx(1.0 / 29.0)
    assert values["bm25_gap_1_5_norm"] == pytest.approx(4.0 / 29.0)
    assert values["bm25_score_sd_norm"] == pytest.approx(expected_sd)
    assert values["top1_id_agreement"] == 1.0
    assert values["top10_id_jaccard"] == pytest.approx(5.0 / 15.0)
    assert values["top10_reciprocal_rank_overlap"] == pytest.approx(expected_rro)
    assert values["top30_union_fraction"] == pytest.approx(55.0 / 60.0)


def test_empty_query_and_zero_score_ranges_are_defined():
    constant = [2.0] * 30
    vector = exp04.extract_feature_vector(
        _view("", _pool("bm25", constant), _pool("dense", constant))
    )
    values = dict(zip(exp04.FEATURE_NAMES, vector))

    assert all(values[name] == 0.0 for name in exp04.FEATURE_NAMES[:6])
    assert values["bm25_gap_1_2_norm"] == 0.0
    assert values["bm25_gap_1_5_norm"] == 0.0
    assert values["bm25_score_sd_norm"] == 0.0
    assert values["dense_score_sd_norm"] == 0.0
    assert np.isfinite(vector).all()


def test_regex_contract_is_case_insensitive_unicode_bounded_and_non_overlapping():
    text = "ĐIỀU 12a, khoản 2; điểm 3. điềuX 4; Bộ LUẬT và nghị định, luật"

    refs = [match.group(0) for match in exp04.LEGAL_REF_RE.finditer(text)]
    docs = [match.group(0) for match in exp04.LEGAL_DOC_RE.finditer(text)]

    assert refs == ["ĐIỀU 12a", "khoản 2", "điểm 3"]
    assert docs == ["Bộ LUẬT", "nghị định", "luật"]


def test_gt_mutation_cannot_change_features_and_full_record_is_rejected():
    record = _candidate()
    before = exp04.extract_feature_vector(exp04.make_inference_view(record))
    for key in exp03.GT_RESULT_KEYS:
        record[key] = ["mutated"] if key != "gt_weights" else {"mutated": 99}
    record["metrics"] = {"leak": True}
    record["query_intent"] = "leak"
    after = exp04.extract_feature_vector(exp04.make_inference_view(record))

    np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match="unexpected"):
        exp04.extract_feature_vector(record)


@pytest.mark.parametrize(
    "winning_weights,oracle_label,training_label,ambiguous",
    [
        ({0.5}, "STAY", "STAY", False),
        ({0.2, 0.3}, "DENSE_HEAVY", "DENSE_HEAVY", False),
        ({0.7}, "BM25_HEAVY", "BM25_HEAVY", False),
        ({0.2, 0.8}, "AMBIGUOUS", "STAY", True),
    ],
)
def test_plateau_label_contract(monkeypatch, winning_weights, oracle_label, training_label, ambiguous):
    def fake_fuse(record, weight, stored_k):
        return [{"weight": weight}]

    def fake_wr(record, ranking, k=10):
        weight = ranking[0]["weight"]
        if weight in winning_weights:
            return 1.0
        if weight == 0.1 and winning_weights == {0.2, 0.3}:
            return 1.0 - 5e-13
        return 0.0

    monkeypatch.setattr(exp04, "fuse_cached_query", fake_fuse)
    monkeypatch.setattr(exp04, "query_weighted_recall", fake_wr)

    label = exp04.construct_training_label({"query_id": 7})

    assert label["oracle_label"] == oracle_label
    assert label["training_label"] == training_label
    assert label["ambiguous"] is ambiguous
    if winning_weights == {0.2, 0.3}:
        assert 0.1 in label["best_set"]  # exact frozen abs_tol includes this value


def test_best_set_excludes_value_outside_exact_tolerance(monkeypatch):
    monkeypatch.setattr(
        exp04,
        "fuse_cached_query",
        lambda record, weight, stored_k: [{"weight": weight}],
    )
    monkeypatch.setattr(
        exp04,
        "query_weighted_recall",
        lambda record, ranking, k=10: (
            1.0 if ranking[0]["weight"] == 0.2 else 1.0 - 2e-12
            if ranking[0]["weight"] == 0.1
            else 0.0
        ),
    )

    label = exp04.construct_training_label({"query_id": 1})

    assert label["best_set"] == [0.2]
    assert label["oracle_label"] == "DENSE_HEAVY"


@pytest.mark.parametrize(
    "dense,bm25,threshold,expected_route,expected_weight",
    [
        (1.0, -1.0, 0.5, "DENSE_DIRECTION", 0.4),
        (-1.0, 1.0, 0.5, "BM25_DIRECTION", 0.6),
        (1.0, 1.0, 0.5, "STAY", 0.5),
        (-1.0, -1.0, 0.5, "STAY", 0.5),
        (float("-inf"), 1.0, 0.5, "BM25_DIRECTION", 0.6),
        (float("-inf"), float("-inf"), 0.0, "STAY", 0.5),
    ],
)
def test_direction_decision_semantics(dense, bm25, threshold, expected_route, expected_weight):
    decision = exp04.route_from_scores(
        dense, bm25, threshold, exp04.EXP04_ACTION_PAIRS[0]
    )

    assert decision["route"] == expected_route
    assert decision["w_bm25"] == expected_weight


def test_zero_positive_disables_without_model_fit(monkeypatch):
    monkeypatch.setattr(
        exp04,
        "_new_logistic_regression",
        lambda c: pytest.fail("zero-positive direction must not construct a model"),
    )
    fitted = exp04.fit_directional_model(
        "DENSE_DIRECTION", np.zeros((4, 18)), [0, 0, 0, 0], 0.1
    )

    assert fitted.disabled
    assert fitted.model is None
    assert fitted.disable_reason == "zero_positive_training_examples"
    score = exp04._direction_score(fitted, np.zeros(18))
    assert math.isinf(score)
    assert score < 0
    for threshold in exp04.EXP04_THRESHOLD_GRID:
        decision = exp04.route_from_scores(
            score, 1.0, threshold, exp04.EXP04_ACTION_PAIRS[0]
        )
        assert decision["dense_pass"] is False
        assert decision["bm25_pass"] is True
        assert decision["route"] == "BM25_DIRECTION"


def test_zero_negative_is_protocol_error():
    with pytest.raises(ValueError, match="zero negatives"):
        exp04.fit_directional_model(
            "BM25_DIRECTION", np.zeros((3, 18)), [1, 1, 1], 0.1
        )


def test_model_and_scaler_use_frozen_parameters():
    model = exp04._new_logistic_regression(0.1)

    assert model.get_params()["penalty"] == "l2"
    assert model.get_params()["solver"] == "liblinear"
    assert model.get_params()["class_weight"] == "balanced"
    assert model.get_params()["random_state"] == 42
    assert model.get_params()["max_iter"] == 1000
    assert model.get_params()["fit_intercept"] is True
    assert not hasattr(model, "probability_calibration")


def test_router_fits_two_independent_models_and_never_refits_scaler_on_test():
    records = [_candidate(query_id, query=f"query {query_id} " * query_id) for query_id in range(1, 7)]
    labels = ["DENSE_HEAVY", "DENSE_HEAVY", "BM25_HEAVY", "BM25_HEAVY", "STAY", "STAY"]

    def label_builder(training_records):
        return [
            {
                "query_id": record["query_id"],
                "training_label": label,
                "oracle_label": label,
                "ambiguous": False,
                "best_set": [0.5],
                "maximum_wr10": 0.0,
            }
            for record, label in zip(training_records, labels)
        ]

    router = exp04.fit_router(records, 0.1, label_builder=label_builder)
    mean_before = router.scaler.mean_.copy()
    score = exp04.score_inference_view(
        router, exp04.make_inference_view(_candidate(99, query="unseen " * 100))
    )

    assert not router.dense.disabled and not router.bm25.disabled
    assert (router.dense.positive_count, router.dense.negative_count) == (2, 4)
    assert (router.bm25.positive_count, router.bm25.negative_count) == (2, 4)
    assert isinstance(score["dense_score"], float)
    assert isinstance(score["bm25_score"], float)
    np.testing.assert_array_equal(router.scaler.mean_, mean_before)


def test_exact_27_config_enumeration_and_ids():
    configs = exp04.enumerate_inner_configs()

    assert len(configs) == 27
    assert len({item["config_id"] for item in configs}) == 27
    assert {item["C"] for item in configs} == {0.01, 0.1, 1.0}
    assert {item["threshold"] for item in configs} == {0.0, 0.5, 1.0}
    assert {item["action_pair"]["id"] for item in configs} == {"A", "B", "C"}
    assert exp04.canonical_config_id(exp04.EXP04_ACTION_PAIRS[0], 0.01, 0.0) == "A__C001__T000"


def _surface_item(
    config_id,
    wr=1.0,
    overrides=1,
    closeness=0.2,
    c_value=0.1,
    threshold=0.5,
):
    return {
        "config_id": config_id,
        "pooled_inner_oof_wr10": wr,
        "override_count": overrides,
        "action_closeness": closeness,
        "C": c_value,
        "threshold": threshold,
    }


@pytest.mark.parametrize(
    "left,right,winner",
    [
        (_surface_item("left", wr=1.0), _surface_item("right", wr=0.9), "left"),
        (_surface_item("left", overrides=1), _surface_item("right", overrides=2), "left"),
        (_surface_item("left", closeness=0.2), _surface_item("right", closeness=0.4), "left"),
        (_surface_item("left", c_value=0.01), _surface_item("right", c_value=0.1), "left"),
        (_surface_item("left", threshold=1.0), _surface_item("right", threshold=0.5), "left"),
        (_surface_item("A"), _surface_item("B"), "A"),
    ],
)
def test_inner_config_tie_break_order(left, right, winner):
    assert exp04.choose_inner_configuration([right, left])["config_id"] == winner


def _disabled_router(label_records=None):
    disabled_dense = exp04.DirectionalModel(
        "DENSE_DIRECTION", None, True, 0, 3, "zero_positive_training_examples"
    )
    disabled_bm25 = exp04.DirectionalModel(
        "BM25_DIRECTION", None, True, 0, 3, "zero_positive_training_examples"
    )
    scaler = SimpleNamespace(mean_=np.zeros(18), scale_=np.ones(18))
    return exp04.RouterFit(
        scaler=scaler,
        dense=disabled_dense,
        bm25=disabled_bm25,
        c_value=0.1,
        label_records=label_records or [],
    )


def test_inner_orchestration_freezes_inference_before_validation_gt_evaluation(
    monkeypatch,
):
    records, mapping = _nested_fixture(4)
    expected_ids = set(mapping)
    validation_folds = sorted(set(mapping.values()))
    fit_calls = []
    feature_views = []
    real_fit_router = exp04.fit_router
    real_extract_feature_vector = exp04.extract_feature_vector

    def guarded_fit_router(training_records, c_value):
        fit_calls.append(
            (float(c_value), {_record["query_id"] for _record in training_records})
        )
        return real_fit_router(training_records, c_value)

    def guarded_extract_feature_vector(view):
        assert set(view) == exp04.INFERENCE_VIEW_KEYS
        assert not any(key in view for key in exp03.GT_RESULT_KEYS)
        feature_views.append(copy.deepcopy(view))
        return real_extract_feature_vector(view)

    monkeypatch.setattr(exp04, "fit_router", guarded_fit_router)
    monkeypatch.setattr(exp04, "extract_feature_vector", guarded_extract_feature_vector)

    scores_by_c = {}
    for c_value in exp04.EXP04_C_GRID:
        scores_by_c[c_value], diagnostics = exp04.build_inner_oof_scores(
            records, mapping, c_value
        )
        assert len(scores_by_c[c_value]) == len(records)
        assert len({row["query_id"] for row in scores_by_c[c_value]}) == len(records)
        assert len(diagnostics) == 4

    assert feature_views
    assert len(fit_calls) == len(exp04.EXP04_C_GRID) * 4
    for call_index, (_, fitted_ids) in enumerate(fit_calls):
        validation_fold = validation_folds[call_index % 4]
        validation_ids = {
            query_id for query_id, fold in mapping.items() if fold == validation_fold
        }
        assert fitted_ids.isdisjoint(validation_ids)
        assert fitted_ids | validation_ids == expected_ids

    frozen_scores = copy.deepcopy(scores_by_c)
    views_before = {
        record["query_id"]: exp04.make_inference_view(record) for record in records
    }
    real_fuse_cached_query = exp04.fuse_cached_query
    fusion_calls = []

    def guarded_fuse_cached_query(view, weight, stored_k):
        assert set(view) == exp04.INFERENCE_VIEW_KEYS
        assert not any(key in view for key in exp03.GT_RESULT_KEYS)
        assert len(view["bm25_pool"]) == len(view["dense_pool"]) == 30
        ranking = real_fuse_cached_query(view, weight, stored_k)
        fusion_calls.append((copy.deepcopy(view), float(weight), copy.deepcopy(ranking)))
        return ranking

    monkeypatch.setattr(exp04, "fuse_cached_query", guarded_fuse_cached_query)
    original_selection = exp04.select_from_frozen_inner_scores(records, scores_by_c)
    original_calls = copy.deepcopy(fusion_calls)

    mutated = copy.deepcopy(records)
    for index, record in enumerate(mutated):
        if index % 4 == 0:
            relevant = "Điều 10 bm25_law"
        elif index % 4 == 1:
            relevant = "Điều 10 dense_law"
        else:
            relevant = "Điều 1 dense_law"
        _set_synthetic_ground_truth(record, relevant)
        record["metrics"] = {"would_leak": True}
        record["oracle_label"] = "would-leak"
        record["outcome"] = "would-leak"

    views_after = {
        record["query_id"]: exp04.make_inference_view(record) for record in mutated
    }
    assert views_after == views_before
    assert scores_by_c == frozen_scores

    fusion_calls.clear()
    mutated_selection = exp04.select_from_frozen_inner_scores(mutated, scores_by_c)
    mutated_calls = copy.deepcopy(fusion_calls)

    original_routes = {
        item["config_id"]: item["routes"] for item in original_selection["surface"]
    }
    mutated_routes = {
        item["config_id"]: item["routes"] for item in mutated_selection["surface"]
    }
    assert original_routes == mutated_routes
    assert original_calls == mutated_calls
    assert original_selection["selected"]["config_id"] == "A__C010__T050"
    assert mutated_selection["selected"]["config_id"] == "A__C001__T100"


def test_outer_gt_mutation_cannot_change_real_outer_fold_orchestration():
    records, mapping = _nested_fixture(5)
    heldout_ids = {query_id for query_id, fold in mapping.items() if fold == 0}
    mutated = copy.deepcopy(records)
    for record in mutated:
        if record["query_id"] in heldout_ids:
            current = exp04.construct_training_label(record)["training_label"]
            replacement = (
                "Điều 10 bm25_law"
                if current != "BM25_HEAVY"
                else "Điều 10 dense_law"
            )
            _set_synthetic_ground_truth(record, replacement)

    before_by_id = {record["query_id"]: record for record in records}
    after_by_id = {record["query_id"]: record for record in mutated}
    assert {
        query_id: exp04.make_inference_view(before_by_id[query_id])
        for query_id in heldout_ids
    } == {
        query_id: exp04.make_inference_view(after_by_id[query_id])
        for query_id in heldout_ids
    }
    assert [
        exp04.construct_training_label(before_by_id[query_id])["training_label"]
        for query_id in sorted(heldout_ids)
    ] != [
        exp04.construct_training_label(after_by_id[query_id])["training_label"]
        for query_id in sorted(heldout_ids)
    ]

    before_predictions, before_evaluated, before_fold = exp04.build_outer_fold_result(
        before_by_id, mapping, 0
    )
    after_predictions, after_evaluated, after_fold = exp04.build_outer_fold_result(
        after_by_id, mapping, 0
    )

    assert before_fold["selected_config"] == after_fold["selected_config"]
    assert before_fold["scaler"] == after_fold["scaler"]
    assert before_fold["dense_model"] == after_fold["dense_model"]
    assert before_fold["bm25_model"] == after_fold["bm25_model"]
    assert before_fold["prediction_fingerprint"] == after_fold["prediction_fingerprint"]
    assert before_predictions == after_predictions
    assert before_evaluated != after_evaluated


def test_107_style_outer_assembly_has_exactly_one_prediction_per_query(monkeypatch):
    records = [{"query_id": query_id} for query_id in range(1, 108)]
    sizes = [22, 22, 21, 21, 21]
    mapping = {}
    cursor = 1
    for fold, size in enumerate(sizes):
        for query_id in range(cursor, cursor + size):
            mapping[query_id] = fold
        cursor += size

    selected = {
        "config_id": "A__C010__T100",
        "C": 0.1,
        "threshold": 1.0,
        "action_pair": exp04.EXP04_ACTION_PAIRS[0],
        "action_closeness": 0.2,
        "pooled_inner_oof_wr10": 0.0,
        "override_count": 0,
    }
    monkeypatch.setattr(
        exp04,
        "select_inner_configuration",
        lambda training, inner_mapping: {
            "selected": selected,
            "surface": [selected],
            "split_diagnostics_by_c": {},
        },
    )
    monkeypatch.setattr(exp04, "fit_router", lambda records, c: _disabled_router())
    monkeypatch.setattr(
        exp04,
        "make_inference_view",
        lambda record: {"query": "", "bm25_pool": [], "dense_pool": []},
    )

    def fake_freeze(router, inference, config, fold):
        predictions = [{"query_id": query_id, "outer_fold": fold} for query_id, _ in inference]
        return predictions, f"fingerprint-{fold}"

    monkeypatch.setattr(exp04, "freeze_outer_predictions", fake_freeze)
    monkeypatch.setattr(
        exp04,
        "_evaluation_result",
        lambda record, prediction: {"query_id": record["query_id"]},
    )
    monkeypatch.setattr(
        exp04,
        "_serialize_direction",
        lambda direction: {"disabled": True},
    )

    predictions, evaluated, folds = exp04.build_nested_oof_results(records, mapping)

    assert len(predictions) == len(evaluated) == 107
    assert len({item["query_id"] for item in predictions}) == 107
    assert len(folds) == 5


def test_full_top30_pools_are_passed_to_existing_fusion(monkeypatch):
    record = _candidate()
    score = {
        "query_id": 1,
        "dense_score": float("-inf"),
        "bm25_score": float("-inf"),
    }
    config = exp04.enumerate_inner_configs()[0]
    observed = []

    def fake_fuse(candidate, weight, stored_k):
        observed.append((len(candidate["bm25_pool"]), len(candidate["dense_pool"]), stored_k))
        return candidate["bm25_pool"][:stored_k]

    monkeypatch.setattr(exp04, "fuse_cached_query", fake_fuse)
    exp04.evaluate_inner_configuration([record], [score], config)

    assert observed == [(30, 30, 20)]


@pytest.mark.parametrize(
    "agree,bm25_scores,dense_scores,expected",
    [
        (True, None, None, 0.5),
        (False, [30.0, 29.9] + list(np.linspace(28, 0, 28)), [30.0, 20.0] + list(np.linspace(19, 0, 28)), 0.4),
        (False, [30.0, 20.0] + list(np.linspace(19, 0, 28)), [30.0, 29.9] + list(np.linspace(28, 0, 28)), 0.6),
        (False, None, None, 0.5),
    ],
)
def test_deterministic_rule_baseline(agree, bm25_scores, dense_scores, expected):
    shared = ["same"] if agree else []
    bm25 = _pool("bm25", bm25_scores, shared=shared)
    dense = _pool("dense", dense_scores, shared=shared)

    assert exp04.deterministic_rule_weight(_view("q", bm25, dense)) == expected


def _metric_world(value=0.1):
    return {
        world: {
            "Fixed_0_5": {
                str(k): {
                    "recall": value,
                    "ndcg": value,
                    "mrr": value,
                    "weighted_recall": value,
                    "valid_queries": 107,
                }
                for k in (5, 10, 20)
            }
        }
        for world in ("closed_world", "open_world")
    }


def _source_summary(value=0.1):
    worlds = _metric_world(value)
    return {
        world: {"Control_Fixed_0_5": worlds[world]["Fixed_0_5"]}
        for world in worlds
    }


def test_fixed_control_parity_exact_pass_and_fail():
    assert exp04.validate_fixed_control_parity(_metric_world(), _source_summary()) == {
        "passed": True,
        "maximum_metric_error": 0.0,
    }
    broken = _source_summary()
    broken["closed_world"]["Control_Fixed_0_5"]["10"]["weighted_recall"] += 1e-15

    with pytest.raises(RuntimeError, match="parity failed"):
        exp04.validate_fixed_control_parity(_metric_world(), broken)


def _evaluated_result(query_id, route, dynamic_relevant, fixed_relevant=True):
    relevant = "điều 1 law"
    irrelevant = "điều 2 law"
    record = {
        "query_id": query_id,
        "outer_fold": (query_id - 1) % 5,
        "route": route,
        "core_gt_filtered": [relevant],
        "supporting_gt_filtered": [],
        "substance_gt_filtered": [relevant],
        "procedural_gt_filtered": [],
        "gt_weights": {relevant: 3},
        "metrics": {
            "Dynamic_OOF": {
                "top_k": [
                    {"standard_law_id": relevant if dynamic_relevant else irrelevant}
                ]
            },
            "Fixed_0_5": {
                "top_k": [
                    {"standard_law_id": relevant if fixed_relevant else irrelevant}
                ]
            },
        },
    }
    return record


def test_false_override_classification_and_harm_ratio_edges():
    results = [
        _evaluated_result(1, "DENSE_DIRECTION", True, False),
        _evaluated_result(2, "BM25_DIRECTION", False, True),
        _evaluated_result(3, "DENSE_DIRECTION", True, True),
        _evaluated_result(4, "STAY", True, True),
    ]
    report = exp04.false_override_report(results)

    assert report["route_count"] == 3
    assert report["stay_count"] == 1
    assert report["dense_override_count"] == 2
    assert report["bm25_override_count"] == 1
    assert report["beneficial_overrides"] == 1
    assert report["harmful_overrides"] == 1
    assert report["tied_overrides"] == 1
    assert report["harm_benefit_ratio"] == 1.0

    zero = exp04.false_override_report([_evaluated_result(5, "STAY", True, True)])
    assert zero["harm_benefit_ratio"] == 0.0
    harm_only = exp04.false_override_report(
        [_evaluated_result(6, "BM25_DIRECTION", False, True)]
    )
    assert math.isinf(harm_only["harm_benefit_ratio"])


def test_bootstrap_and_sign_flip_are_deterministic_golden_vectors():
    deltas = [0.0, 0.1, -0.05, 0.2]

    bootstrap = exp04.paired_bootstrap(deltas, samples=200, seed=42)
    sign_flip = exp04.paired_sign_flip(deltas, samples=500, seed=42)

    assert bootstrap == exp04.paired_bootstrap(deltas, samples=200, seed=42)
    assert sign_flip == exp04.paired_sign_flip(deltas, samples=500, seed=42)
    assert bootstrap["mean_delta"] == pytest.approx(0.0625)
    assert bootstrap["ci_95_percentile"] == pytest.approx([-0.025, 0.175])
    assert sign_flip["observed_absolute_mean"] == pytest.approx(0.0625)
    assert sign_flip["extreme_count"] == 248
    assert sign_flip["p_value"] == pytest.approx(249 / 501)


def _successful_analysis():
    per_query = []
    for fold in range(5):
        for index in range(2):
            per_query.append(
                {
                    "query_id": fold * 2 + index,
                    "outer_fold": fold,
                    "deltas": {
                        "weighted_recall": 0.1,
                        "recall": 0.01,
                        "ndcg": 0.01,
                        "mrr": 0.01,
                    },
                }
            )
    metrics = {}
    for name in ("weighted_recall", "recall", "ndcg", "mrr"):
        metrics[name] = {
            "paired_bootstrap": {"ci_95_percentile": [0.01, 0.2]},
            "paired_sign_flip": {"p_value": 0.01},
        }
    return {"per_query": per_query, "metrics": metrics}


def _successful_override_report():
    return {
        "route_count": 5,
        "total_positive_gain": 1.0,
        "total_negative_harm": 0.1,
    }


@pytest.mark.parametrize(
    "mutation,failed_condition",
    [
        (lambda analysis, report: [row["deltas"].__setitem__("weighted_recall", -0.1) for row in analysis["per_query"]], "dynamic_wr10_above_fixed"),
        (lambda analysis, report: analysis["metrics"]["weighted_recall"]["paired_bootstrap"].__setitem__("ci_95_percentile", [0.0, 0.2]), "bootstrap_lower_above_zero"),
        (lambda analysis, report: analysis["metrics"]["weighted_recall"]["paired_sign_flip"].__setitem__("p_value", 0.05), "sign_flip_p_below_0_05"),
        (lambda analysis, report: report.__setitem__("route_count", 0), "nonzero_routing"),
        (lambda analysis, report: report.__setitem__("total_negative_harm", 1.0), "benefit_exceeds_harm"),
        (lambda analysis, report: [row["deltas"].__setitem__("weighted_recall", 0.0) for row in analysis["per_query"][1:]], "positive_after_removing_largest_gain"),
        (lambda analysis, report: [row["deltas"].__setitem__("weighted_recall", -0.01) for row in analysis["per_query"] if row["outer_fold"] >= 2], "positive_in_at_least_three_folds"),
        (lambda analysis, report: analysis["metrics"]["recall"]["paired_bootstrap"].__setitem__("ci_95_percentile", [-0.2, -0.01]), "no_serious_secondary_collapse"),
    ],
)
def test_success_contract_each_failure_condition(mutation, failed_condition):
    analysis = _successful_analysis()
    report = _successful_override_report()
    mutation(analysis, report)

    result = exp04.evaluate_success_contract(analysis, report)

    assert result["conditions"][failed_condition] is False
    assert result["all_conditions_hold"] is False
    assert result["decision"] == "DO_NOT_PROMOTE"


def test_success_contract_all_conditions_and_exploratory_scope():
    result = exp04.evaluate_success_contract(
        _successful_analysis(), _successful_override_report()
    )

    assert result["all_conditions_hold"] is True
    assert result["decision"] == "EXPLORATORY_SUCCESS"
    assert result["result_scope"] == "exploratory"


def _artifact_payload():
    predictions = [{"query_id": query_id} for query_id in range(1, 108)]
    manifest = {
        key: None
        for key in (
            "run_id",
            "timestamp",
            "experiment_id",
            "protocol_id",
            "evaluation_mode",
            "control_protocol",
            "candidate_source_run",
            "candidate_path",
            "candidate_fingerprint",
            "fold_path",
            "fold_mapping_fingerprint",
            "feature_contract_version",
            "source_files",
            "git_commit",
            "git_dirty",
            "tracked_code_dirty",
            "exp04_source_dirty",
            "package_versions",
        )
    }
    return {
        "feature_contract.json": exp04.feature_contract_payload(),
        "outer_fold_manifest.json": {},
        "inner_selection_results.json": {"outer_folds": [{} for _ in range(5)]},
        "oof_predictions.json": {"predictions": predictions},
        "pipeline_results.json": {},
        "summary_metrics.json": {},
        "statistical_analysis.json": {},
        "run_manifest.json": manifest,
    }


def test_artifact_schema_enforces_prediction_and_provenance_isolation():
    artifacts = _artifact_payload()
    exp04.validate_exp04_artifacts(artifacts)

    artifacts["oof_predictions.json"]["predictions"][0]["core_gt_filtered"] = []
    with pytest.raises(ValueError, match="must not contain ground truth"):
        exp04.validate_exp04_artifacts(artifacts)


def test_exp04_writer_is_atomic_and_run_scoped(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("protocol: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp04-run"
    artifacts = {"pipeline_results.json": {"results": []}, "run_manifest.json": {}}

    output = write_exp04_artifacts(run_dir, config_path, artifacts)

    assert output == run_dir / "pipeline_results.json"
    assert {path.name for path in run_dir.iterdir()} == {
        "pipeline_results.json",
        "run_manifest.json",
        "eval_config.snapshot.yaml",
    }
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def test_exp04_writer_removes_partial_directory_on_failure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("protocol: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp04-run"
    real_dump = serialization.json.dump
    calls = 0

    def fail_second_dump(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic EXP-04 artifact failure")
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(serialization.json, "dump", fail_second_dump)
    with pytest.raises(OSError, match="synthetic EXP-04 artifact failure"):
        write_exp04_artifacts(
            run_dir,
            config_path,
            {"one.json": {}, "two.json": {}},
        )

    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def test_cli_dispatches_exp04_explicitly_without_running_it(monkeypatch):
    calls = []

    def fake_exp04(path, max_queries=None):
        calls.append((path, max_queries))
        return "synthetic-exp04-result"

    monkeypatch.setattr(
        experiment_runner, "run_exp04_selective_dynamic_hybrid_nested_cv", fake_exp04
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_exp03_static_fusion_cv",
        lambda *args, **kwargs: pytest.fail("EXP-04 must not route to EXP-03"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_evaluation_pipeline",
        lambda *args, **kwargs: pytest.fail("EXP-04 must not route to baseline"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_legacy_advanced",
        lambda *args, **kwargs: pytest.fail("EXP-04 must not route to legacy"),
    )

    assert experiment_runner.main(["--config", str(CONFIG_PATH)]) == "synthetic-exp04-result"
    assert calls == [(CONFIG_PATH, None)]


def test_existing_dispatch_paths_remain_distinct():
    source = Path(experiment_runner.__file__).read_text(encoding="utf-8")

    assert "mode == BASELINE_MODE" in source
    assert "mode == EXP03_MODE" in source
    assert "mode == EXP04_MODE" in source
    assert "mode == LEGACY_MODE" in source
    assert "Unsupported evaluation dispatch combination" in source


def test_exp04_source_has_no_retrieval_or_external_model_entrypoints():
    source = Path(exp04.__file__).read_text(encoding="utf-8")

    forbidden = (
        "build_bm25_index",
        "build_dense_retriever",
        "search_bm25",
        "search_faiss",
        "SentenceTransformer",
        "google.generativeai",
        "Gemini",
    )
    assert not [name for name in forbidden if name in source]


def test_exp04_provenance_does_not_change_earlier_protocol_source_sets():
    baseline_source = Path(
        PROJECT_ROOT / "backend/scripts/Evaluation/experiments/baseline.py"
    ).read_text(encoding="utf-8")
    exp03_paths = exp03._source_paths(
        PROJECT_ROOT / "backend/scripts/Evaluation/configs/exp03_static_fusion_cv.yaml",
        PROJECT_ROOT,
    )

    assert "test_exp04_protocol.py" not in baseline_source
    assert not any("exp04" in path.lower() for path in exp03_paths)
    assert any("test_exp04_protocol.py" in path for path in exp04._source_paths(CONFIG_PATH, PROJECT_ROOT))


def test_outer_prediction_payload_fingerprint_is_sensitive_and_contains_no_gt(monkeypatch):
    router = _disabled_router()
    selected = {
        "config_id": "A__C001__T100",
        "C": 0.01,
        "threshold": 1.0,
        "action_pair": exp04.EXP04_ACTION_PAIRS[0],
    }
    view = _view()
    monkeypatch.setattr(
        exp04,
        "score_inference_view",
        lambda router, inference: {
            "feature_vector": [0.0] * 18,
            "dense_score": float("-inf"),
            "bm25_score": float("-inf"),
            "dense_disabled": True,
            "bm25_disabled": True,
        },
    )
    first, first_hash = exp04.freeze_outer_predictions(
        router, [(1, view)], selected, 0
    )
    assert first[0]["dense_score"] is None
    assert first[0]["bm25_score"] is None
    assert first[0]["dense_disabled"] is True
    assert first[0]["bm25_disabled"] is True
    changed = copy.deepcopy(view)
    changed["query"] = "changed inference query"
    second, second_hash = exp04.freeze_outer_predictions(
        router, [(1, changed)], selected, 0
    )

    assert not any(key in first[0] for key in exp03.GT_RESULT_KEYS)
    # The monkeypatched score is constant, so query text is intentionally absent
    # from the persisted payload; changing a frozen ranking must change the hash.
    second[0]["dynamic_top_k"][0]["score"] += 0.1
    changed_payload = json.dumps(second, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    changed_hash = exp04.hashlib.sha256(changed_payload.encode("utf-8")).hexdigest()
    assert changed_hash != first_hash
    assert second_hash == first_hash
