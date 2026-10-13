import copy
import json
from pathlib import Path

import pytest
import yaml

import backend.scripts.Evaluation.run_experiment as experiment_runner
from backend.scripts.Evaluation.experiments import exp03_static_fusion_cv as exp03
from backend.scripts.Evaluation.reporting import serialization
from backend.scripts.Evaluation.reporting.serialization import write_exp03_artifacts
from backend.scripts.Evaluation.retrieval.fusion import fuse_pools


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "backend" / "scripts" / "Evaluation" / "configs"
CONFIG_PATH = (
    CONFIG_DIR / "exp03_static_fusion_cv.yaml"
)
BASELINE_CONFIG_PATH = CONFIG_DIR / "baseline_d0.yaml"
EXP02_D0_CONFIG_PATH = CONFIG_DIR / "exp02_d0.yaml"
EXP02_D1_CONFIG_PATH = CONFIG_DIR / "exp02_d1.yaml"


def _load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _set_nested(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def _pool(first_id, second_id):
    return [
        {"rank": 1, "standard_law_id": first_id, "score": 2.0},
        {"rank": 2, "standard_law_id": second_id, "score": 0.0},
    ]


def _candidate(query_id, relevant_id, bm25_first="điều 1 law_a"):
    first = "điều 1 law_a"
    second = "điều 2 law_a"
    bm25_second = second if bm25_first == first else first
    dense_first = bm25_second
    dense_second = bm25_first
    return {
        "query_id": query_id,
        "query": f"query {query_id}",
        "core_gt_raw": [relevant_id],
        "supporting_gt_raw": [],
        "core_gt_filtered": [relevant_id],
        "supporting_gt_filtered": [],
        "substance_gt_raw": [relevant_id],
        "procedural_gt_raw": [],
        "substance_gt_filtered": [relevant_id],
        "procedural_gt_filtered": [],
        "explicit_gts": [relevant_id],
        "inferred_gts": [],
        "gt_weights": {relevant_id: 3},
        "bm25_pool": _pool(bm25_first, bm25_second),
        "dense_pool": _pool(dense_first, dense_second),
    }


def _golden_pipeline(candidate_records):
    return {
        "results": [
            {
                "query_id": record["query_id"],
                "metrics": {
                    "BM25": {"top_k": record["bm25_pool"][:2]},
                    "FAISS": {"top_k": record["dense_pool"][:2]},
                    "Hybrid_Static_Fixed": {
                        "top_k": exp03.fuse_cached_query(record, 0.5, stored_k=2)
                    }
                },
            }
            for record in candidate_records
        ]
    }


def test_exp03_config_validator_accepts_exact_config():
    protocol = exp03._validate_exp03_protocol(_load_config())

    assert protocol == {
        "protocol_id": "exp03_static_hybrid_5fold_cv_v1",
        "experiment_id": "exp03_static_hybrid_weight_v1",
        "control_protocol": "baseline_d0_v2",
        "query_representation": "q0",
        "document_representation": "d0",
        "temporal_policy": "none",
        "pool_k": 30,
        "stored_k": 20,
        "reported_k": (5, 10, 20),
        "primary_k": 10,
        "alpha_core": 3,
        "alpha_support": 1,
        "fold_count": 5,
        "fold_seed": 42,
        "fold_algorithm": "sha256_protocol_seed_query_id_contiguous_v1",
        "fold_sizes": (22, 22, 21, 21, 21),
        "weight_grid": tuple(value / 10 for value in range(11)),
        "weight_tie_break": (
            "highest_unrounded_objective",
            "smallest_distance_to_0.5",
            "smaller_w_bm25",
        ),
        "ranking_tie_break": "score_desc_then_standard_law_id_asc",
        "control_weight": 0.5,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol", "id"), "other"),
        (("protocol", "experiment_id"), "other"),
        (("protocol", "control_protocol"), "other"),
        (("evaluation", "mode"), "baseline"),
        (("evaluation", "experiment_id"), "other"),
        (("evaluation", "query_representation"), "q1"),
        (("evaluation", "document_representation"), "d1"),
        (("evaluation", "temporal_filter"), "judgment_year"),
        (("evaluation", "pool_k"), 20),
        (("evaluation", "k"), 30),
        (("evaluation", "alpha_core"), 2),
        (("evaluation", "alpha_support"), 2),
        (("evaluation", "random_seed"), 7),
        (("evaluation", "expected_query_count"), 106),
        (("evaluation", "query_fingerprint"), "other"),
        (("metric_protocol", "primary_world"), "open_world"),
        (("metric_protocol", "primary_metric"), "recall"),
        (("metric_protocol", "reported_k"), [5, 10]),
        (("metric_protocol", "primary_k"), 20),
        (("models", "embedding_model"), "other/model"),
        (("models", "embedding_revision"), "other-revision"),
        (("models", "local_files_only"), False),
        (("models", "embedding_dimension"), 384),
        (("models", "embedding_normalized"), False),
        (("corpus", "expected_total"), 1812),
        (("corpus", "fingerprint"), "other"),
        (("corpus", "required_keys"), ["other"]),
        (("corpus", "expected_counts"), {"other": 1811}),
        (("cross_validation", "fold_count"), 4),
        (("cross_validation", "seed"), 7),
        (("cross_validation", "algorithm"), "numpy_shuffle"),
        (("cross_validation", "fold_sizes"), [22, 22, 21, 21, 20]),
        (("weight_tuning", "grid"), [0.0, 0.5, 1.0]),
        (("weight_tuning", "objective", "world"), "open_world"),
        (("weight_tuning", "objective", "metric"), "ndcg"),
        (("weight_tuning", "objective", "k"), 20),
        (("weight_tuning", "tie_break"), ["secondary_metric"]),
        (("control", "static_w_bm25"), 0.6),
        (("golden_parity", "run_id"), "other"),
        (("golden_parity", "configurations"), ["Hybrid_Static_Fixed"]),
        (("golden_parity", "metric_tolerance"), 1.0e-6),
        (("paths", "golden_run_dir"), "backend/data/runs/other"),
        (("fusion", "method"), "rrf"),
        (("fusion", "ranking_tie_break"), "score_only"),
        (("faiss", "index_type"), "IndexIVFFlat"),
        (("bm25", "tokenizer"), "pyvi"),
    ],
)
def test_exp03_validator_rejects_protocol_drift(path, value):
    config = copy.deepcopy(_load_config())
    _set_nested(config, path, value)

    with pytest.raises(ValueError):
        exp03._validate_exp03_protocol(config)


def test_fold_assignment_is_deterministic_complete_and_order_independent():
    records = [{"id": query_id} for query_id in range(1, 108)]
    mapping = exp03.assign_folds(records)
    reversed_mapping = exp03.assign_folds(list(reversed(records)))

    assert mapping == reversed_mapping
    assert set(mapping) == set(range(1, 108))
    assert [sum(fold == index for fold in mapping.values()) for index in range(5)] == [
        22,
        22,
        21,
        21,
        21,
    ]
    assert exp03.fingerprint_fold_mapping(mapping) == exp03.fingerprint_fold_mapping(
        reversed_mapping
    )
    assert exp03.fingerprint_fold_mapping(mapping) == (
        "94e9f68e7113f21b24e4cfbbccb2faacac1bfb106ac35e5c2167e7f51eee7310"
    )


def test_fold_seed_changes_mapping_and_fingerprint():
    query_ids = list(range(1, 108))
    seed_42 = exp03.assign_folds(query_ids, seed=42)
    seed_43 = exp03.assign_folds(query_ids, seed=43)

    assert seed_42 != seed_43
    assert exp03.fingerprint_fold_mapping(seed_42) != exp03.fingerprint_fold_mapping(seed_43)


def test_weight_selection_cannot_see_heldout_labels():
    training = [_candidate(1, "điều 1 law_a")]
    heldout = [
        _candidate(2, "điều 2 law_a"),
        _candidate(3, "điều 2 law_a"),
    ]

    training_only = exp03.select_fold_weight(training, weight_grid=(0.0, 1.0), k=1)
    leaked = exp03.select_fold_weight(training + heldout, weight_grid=(0.0, 1.0), k=1)

    assert training_only["selected_w_bm25"] == 1.0
    assert leaked["selected_w_bm25"] == 0.0


def test_oof_builder_excludes_heldout_labels_from_weight_selection():
    first = "điều 1 law_a"
    second = "điều 2 law_a"
    candidates = [
        _candidate(1, second),
        _candidate(2, second),
        _candidate(3, first),
    ]
    mapping = {1: 0, 2: 0, 3: 1}

    leaked_selection = exp03.select_fold_weight(
        candidates, weight_grid=(0.0, 1.0), k=1
    )
    oof_results, fold_results = exp03.build_oof_results(
        candidates,
        mapping,
        weight_grid=(0.0, 1.0),
        fold_count=2,
        stored_k=2,
        objective_k=1,
    )

    fold_zero = next(item for item in fold_results if item["fold"] == 0)
    assert leaked_selection["selected_w_bm25"] == 0.0
    assert fold_zero["training_query_ids"] == [3]
    assert fold_zero["heldout_query_ids"] == [1, 2]
    assert fold_zero["selected_w_bm25"] == 1.0
    heldout = [item for item in oof_results if item["cv_fold"] == 0]
    assert {item["query_id"] for item in heldout} == {1, 2}
    assert all(item["selected_w_bm25"] == 1.0 for item in heldout)
    assert all(
        item["metrics"]["OOF_Selected_Policy"]["top_k"][0]["standard_law_id"]
        == first
        for item in heldout
    )
    assert [item["query_id"] for item in oof_results] == [1, 2, 3]


def test_weight_tie_chooses_nearest_to_control():
    record = _candidate(1, "điều 1 law_a")
    record["dense_pool"] = copy.deepcopy(record["bm25_pool"])

    selected = exp03.select_fold_weight([record], (0.0, 0.4, 0.5, 0.6, 1.0), k=1)

    assert selected["selected_w_bm25"] == 0.5


def test_symmetric_weight_tie_chooses_smaller_bm25_weight():
    record = _candidate(1, "điều 1 law_a")
    record["dense_pool"] = copy.deepcopy(record["bm25_pool"])

    selected = exp03.select_fold_weight([record], (0.4, 0.6), k=1)

    assert selected["selected_w_bm25"] == 0.4


@pytest.mark.parametrize("weight", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_cached_fusion_reuses_existing_fusion_semantics(weight):
    record = _candidate(1, "điều 1 law_a")
    expected = fuse_pools(record["bm25_pool"], record["dense_pool"], weight, "minmax")

    actual = exp03.fuse_cached_query(record, weight, stored_k=2)

    assert [(item["standard_law_id"], item["score"]) for item in actual] == expected


def test_fusion_normalizes_full_pool_before_stored_ranking_cut():
    full = {
        "query_id": 1,
        "bm25_pool": [
            {"rank": 1, "standard_law_id": "a", "score": 10.0},
            {"rank": 2, "standard_law_id": "b", "score": 9.0},
            {"rank": 3, "standard_law_id": "c", "score": 0.0},
        ],
        "dense_pool": [
            {"rank": 1, "standard_law_id": "b", "score": 10.0},
            {"rank": 2, "standard_law_id": "c", "score": 9.0},
            {"rank": 3, "standard_law_id": "a", "score": 0.0},
        ],
    }
    truncated = {
        "query_id": 1,
        "bm25_pool": full["bm25_pool"][:2],
        "dense_pool": full["dense_pool"][:2],
    }

    full_ranking = exp03.fuse_cached_query(full, 0.5, stored_k=2)
    truncated_ranking = exp03.fuse_cached_query(truncated, 0.5, stored_k=2)

    assert [item["standard_law_id"] for item in full_ranking] == ["b", "a"]
    assert [item["standard_law_id"] for item in truncated_ranking] == ["a", "b"]


def test_missing_modality_candidates_receive_zero_contribution():
    bm25 = [
        {"rank": 1, "standard_law_id": "a", "score": 10.0},
        {"rank": 2, "standard_law_id": "c", "score": 0.0},
    ]
    dense = [
        {"rank": 1, "standard_law_id": "b", "score": 10.0},
        {"rank": 2, "standard_law_id": "c", "score": 0.0},
    ]

    assert fuse_pools(bm25, dense, 0.5, "minmax") == [
        ("a", 0.5),
        ("b", 0.5),
        ("c", 0.0),
    ]


def test_constant_score_pool_matches_existing_zero_range_semantics():
    bm25 = [
        {"rank": 1, "standard_law_id": "a", "score": 5.0},
        {"rank": 2, "standard_law_id": "b", "score": 5.0},
    ]
    dense = [
        {"rank": 1, "standard_law_id": "a", "score": 2.0},
        {"rank": 2, "standard_law_id": "b", "score": 0.0},
    ]

    assert fuse_pools(bm25, dense, 0.5, "minmax") == [("a", 0.5), ("b", 0.0)]


def test_fusion_score_ties_keep_standard_law_id_ascending():
    record = _candidate(1, "điều 1 law_a")

    ranking = exp03.fuse_cached_query(record, 0.5, stored_k=2)

    assert [item["standard_law_id"] for item in ranking] == [
        "điều 1 law_a",
        "điều 2 law_a",
    ]


def test_oof_aggregation_has_one_traceable_prediction_per_query():
    candidates = [
        _candidate(1, "điều 1 law_a"),
        _candidate(2, "điều 2 law_a"),
        _candidate(3, "điều 1 law_a"),
        _candidate(4, "điều 2 law_a"),
    ]
    mapping = {1: 0, 2: 0, 3: 1, 4: 1}

    results, folds = exp03.build_oof_results(
        candidates,
        mapping,
        weight_grid=(0.0, 1.0),
        fold_count=2,
        stored_k=2,
        objective_k=1,
    )

    assert [item["query_id"] for item in results] == [1, 2, 3, 4]
    assert all("selected_w_bm25" in item and "cv_fold" in item for item in results)
    assert all(
        set(item["metrics"]) == {"OOF_Selected_Policy", "Control_Fixed_0_5"}
        for item in results
    )
    assert len(folds) == 2


def test_oof_coverage_rejects_duplicate_and_missing_queries():
    with pytest.raises(ValueError, match="duplicate"):
        exp03.validate_oof_coverage([{"query_id": 1}, {"query_id": 1}], [1, 2])
    with pytest.raises(ValueError, match="exactly"):
        exp03.validate_oof_coverage([{"query_id": 1}], [1, 2])


def test_train_heldout_split_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        exp03.validate_train_heldout_split([1, 2], [2, 3], [1, 2, 3])


def test_candidate_fingerprint_is_canonical_and_sensitive_to_scores():
    records = [_candidate(2, "điều 2 law_a"), _candidate(1, "điều 1 law_a")]
    reordered = []
    for record in reversed(records):
        reordered_record = dict(reversed(list(record.items())))
        reordered_record["bm25_pool"] = [
            dict(reversed(list(item.items()))) for item in record["bm25_pool"]
        ]
        reordered_record["dense_pool"] = [
            dict(reversed(list(item.items()))) for item in record["dense_pool"]
        ]
        reordered.append(reordered_record)
    changed = copy.deepcopy(records)
    changed[0]["bm25_pool"][0]["score"] += 0.001
    changed_id = copy.deepcopy(records)
    changed_id[0]["dense_pool"][0]["standard_law_id"] = "different"

    fingerprint = exp03.fingerprint_candidate_pools(records)

    assert fingerprint == exp03.fingerprint_candidate_pools(reordered)
    assert fingerprint != exp03.fingerprint_candidate_pools(changed)
    assert fingerprint != exp03.fingerprint_candidate_pools(changed_id)


def test_candidate_cache_requires_full_contiguous_pools():
    records = []
    for query_id in (1, 2):
        pool = [
            {"rank": rank, "standard_law_id": f"law-{rank}", "score": 31.0 - rank}
            for rank in range(1, 31)
        ]
        records.append(
            {"query_id": query_id, "bm25_pool": pool, "dense_pool": copy.deepcopy(pool)}
        )

    exp03.validate_candidate_pools(records, expected_query_count=2, pool_k=30)
    records[0]["dense_pool"].pop()
    with pytest.raises(ValueError, match="pool_k=30"):
        exp03.validate_candidate_pools(records, expected_query_count=2, pool_k=30)


@pytest.mark.parametrize("configuration", exp03.EXP03_GOLDEN_CONFIGURATIONS)
def test_golden_parity_rejects_each_retriever_ranking_drift(configuration):
    candidates = [
        _candidate(1, "điều 1 law_a"),
        _candidate(2, "điều 2 law_a"),
    ]
    golden = _golden_pipeline(candidates)

    assert exp03.validate_golden_parity(
        candidates, golden, stored_k=2, reported_k=(1,)
    )["passed"]

    broken = copy.deepcopy(golden)
    broken["results"][0]["metrics"][configuration]["top_k"][0][
        "standard_law_id"
    ] = "different"
    with pytest.raises(RuntimeError, match=rf"{configuration}.*ranking differs"):
        exp03.validate_golden_parity(candidates, broken, stored_k=2, reported_k=(1,))


@pytest.mark.parametrize("configuration", exp03.EXP03_GOLDEN_CONFIGURATIONS)
def test_golden_parity_rejects_each_retriever_metric_drift(configuration):
    candidates = [_candidate(1, "điều 1 law_a")]
    golden = _golden_pipeline(candidates)
    parity = exp03.validate_golden_parity(
        candidates, golden, stored_k=2, reported_k=(1,)
    )
    summary = copy.deepcopy(parity["metric_worlds"])
    exp03.validate_golden_parity(
        candidates, golden, summary, stored_k=2, reported_k=(1,)
    )
    summary["closed_world"][configuration]["1"]["weighted_recall"] += 0.01

    with pytest.raises(
        RuntimeError, match=rf"{configuration} weighted_recall@1"
    ):
        exp03.validate_golden_parity(
            candidates, golden, summary, stored_k=2, reported_k=(1,)
        )


def test_artifact_schema_requires_cv_traceability_and_provenance():
    candidates = [_candidate(1, "điều 1 law_a"), _candidate(2, "điều 2 law_a")]
    mapping = {1: 0, 2: 1}
    results, fold_results = exp03.build_oof_results(
        candidates,
        mapping,
        weight_grid=(0.0, 1.0),
        fold_count=2,
        stored_k=2,
        objective_k=1,
    )
    candidate_cache = {
        "schema_version": "v1",
        "protocol_id": exp03.EXP03_PROTOCOL_ID,
        "query_count": 2,
        "pool_k": 2,
        "fingerprint_algorithm": "sha256_canonical_json_v1",
        "candidate_cache_fingerprint": exp03.fingerprint_candidate_pools(candidates),
        "records": candidates,
    }
    fold_manifest = {
        "assignment_algorithm": exp03.EXP03_FOLD_ALGORITHM,
        "seed": 42,
        "fold_count": 2,
        "fold_sizes": [1, 1],
        "mapping": [{"query_id": 1, "fold": 0}, {"query_id": 2, "fold": 1}],
        "mapping_fingerprint": exp03.fingerprint_fold_mapping(mapping),
    }
    pipeline = {
        "run_id": "run",
        "protocol_id": exp03.EXP03_PROTOCOL_ID,
        "experiment_id": exp03.EXP03_EXPERIMENT_ID,
        "control_protocol": "baseline_d0_v2",
        "configurations": ["OOF_Selected_Policy", "Control_Fixed_0_5"],
        "fold_results": fold_results,
        "results": results,
    }
    summary = {
        "run_id": "run",
        "protocol_id": exp03.EXP03_PROTOCOL_ID,
        "experiment_id": exp03.EXP03_EXPERIMENT_ID,
        "primary_world": "closed_world",
        "primary_metric": "weighted_recall",
        "primary_k": 10,
        "reported_k": [5, 10, 20],
        "selected_weights_by_fold": [],
        "closed_world": {},
        "open_world": {},
    }
    manifest_fields = {
        "run_id",
        "timestamp",
        "protocol_id",
        "experiment_id",
        "control_protocol",
        "query_count",
        "corpus_count",
        "query_representation",
        "document_representation",
        "temporal_filter",
        "embedding_model",
        "embedding_requested_revision",
        "embedding_resolved_revision",
        "embedding_revision_verified",
        "embedding_local_files_only",
        "embedding_normalized",
        "embedding_dimension",
        "faiss_index_type",
        "bm25_tokenizer",
        "pool_k",
        "stored_k",
        "candidate_cache_fingerprint",
        "fold_algorithm",
        "fold_seed",
        "fold_count",
        "fold_sizes",
        "fold_mapping_fingerprint",
        "weight_grid",
        "objective",
        "weight_tie_break",
        "selected_weights_by_fold",
        "ranking_tie_break",
        "primary_k",
        "reported_k",
        "alpha_core",
        "alpha_support",
        "combined_corpus_fingerprint",
        "query_representation_fingerprint",
        "benchmark",
        "gt_schema",
        "config",
        "source_files",
        "corpora",
        "ground_truth_coverage",
        "golden_parity",
        "git_commit",
        "git_dirty",
        "tracked_code_dirty",
        "random_seed",
        "package_versions",
    }
    manifest = {field: None for field in manifest_fields}
    manifest.update(
        {
            "candidate_cache_fingerprint": candidate_cache[
                "candidate_cache_fingerprint"
            ],
            "fold_mapping_fingerprint": fold_manifest["mapping_fingerprint"],
        }
    )

    exp03.validate_exp03_artifact_schema(
        candidate_cache, fold_manifest, pipeline, summary, manifest
    )
    del manifest["candidate_cache_fingerprint"]
    with pytest.raises(ValueError, match="candidate_cache_fingerprint"):
        exp03.validate_exp03_artifact_schema(
            candidate_cache, fold_manifest, pipeline, summary, manifest
        )


def test_exp03_writer_creates_only_run_scoped_artifacts(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp03-run"

    output = write_exp03_artifacts(
        run_dir,
        config_path,
        {"candidate": True},
        {"fold": True},
        {"results": []},
        {"summary": True},
        {"manifest": True},
    )

    assert output == run_dir / "pipeline_results.json"
    assert {path.name for path in run_dir.iterdir()} == {
        "candidate_pools.json",
        "fold_manifest.json",
        "pipeline_results.json",
        "summary_metrics.json",
        "run_manifest.json",
        "eval_config.snapshot.yaml",
    }
    assert json.loads((run_dir / "candidate_pools.json").read_text(encoding="utf-8")) == {
        "candidate": True
    }
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def test_exp03_writer_removes_partial_directory_on_json_failure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp03-run"
    real_dump = serialization.json.dump
    calls = 0

    def fail_intermediate_json(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic intermediate JSON failure")
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(serialization.json, "dump", fail_intermediate_json)
    with pytest.raises(OSError, match="synthetic intermediate JSON failure"):
        write_exp03_artifacts(run_dir, config_path, {}, {}, {}, {}, {})

    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def test_exp03_writer_removes_partial_directory_on_snapshot_failure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("evaluation: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp03-run"

    def fail_snapshot(*args, **kwargs):
        raise OSError("synthetic snapshot failure")

    monkeypatch.setattr(serialization.shutil, "copy2", fail_snapshot)
    with pytest.raises(OSError, match="synthetic snapshot failure"):
        write_exp03_artifacts(run_dir, config_path, {}, {}, {}, {}, {})

    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


@pytest.mark.parametrize(
    "config_path",
    [BASELINE_CONFIG_PATH, EXP02_D0_CONFIG_PATH, EXP02_D1_CONFIG_PATH],
)
def test_cli_dispatches_baseline_and_exp02_to_existing_evaluator(
    config_path, monkeypatch
):
    calls = []

    def fake_baseline(path, max_queries=None):
        calls.append((path, max_queries))
        return "baseline-result"

    monkeypatch.setattr(experiment_runner, "run_baseline_evaluation_pipeline", fake_baseline)
    monkeypatch.setattr(
        experiment_runner,
        "run_legacy_advanced",
        lambda **kwargs: pytest.fail("controlled protocol must not route to legacy"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_exp03_static_fusion_cv",
        lambda *args, **kwargs: pytest.fail("baseline/EXP-02 must not route to EXP-03"),
    )

    assert experiment_runner.main(["--config", str(config_path)]) == "baseline-result"
    assert calls == [(config_path, None)]


def test_cli_dispatches_exp03_to_exp03_evaluator(monkeypatch):
    calls = []

    def fake_exp03(path, max_queries=None):
        calls.append((path, max_queries))
        return "exp03-result"

    monkeypatch.setattr(experiment_runner, "run_exp03_static_fusion_cv", fake_exp03)
    monkeypatch.setattr(
        experiment_runner,
        "run_legacy_advanced",
        lambda **kwargs: pytest.fail("EXP-03 must not route to legacy"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_evaluation_pipeline",
        lambda *args, **kwargs: pytest.fail("EXP-03 must not route to baseline"),
    )

    assert experiment_runner.main(["--config", str(CONFIG_PATH)]) == "exp03-result"
    assert calls == [(CONFIG_PATH, None)]


def test_cli_dispatches_recognized_legacy_mode(tmp_path, monkeypatch):
    config_path = tmp_path / "legacy.yaml"
    config_path.write_text("evaluation:\n  mode: legacy\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        experiment_runner,
        "run_exp03_static_fusion_cv",
        lambda *args, **kwargs: pytest.fail("legacy mode must not route to EXP-03"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_evaluation_pipeline",
        lambda *args, **kwargs: pytest.fail("legacy mode must not route to baseline"),
    )

    def fake_legacy(**kwargs):
        calls.append(kwargs)
        return "legacy-result"

    monkeypatch.setattr(experiment_runner, "run_legacy_advanced", fake_legacy)

    assert experiment_runner.main(
        ["--config", str(config_path), "--max-queries", "3", "--temporal-filter", "off"]
    ) == "legacy-result"
    assert calls == [
        {
            "temporal_filter": "off",
            "config_path": config_path,
            "max_queries": 3,
        }
    ]


@pytest.mark.parametrize(
    "contents",
    [
        "evaluation:\n  mode: another_experiment\n",
        "protocol:\n  id: unknown_protocol\nevaluation:\n  mode: baseline\n",
    ],
)
def test_cli_unknown_mode_or_protocol_fails_closed(tmp_path, monkeypatch, contents):
    config_path = tmp_path / "unknown.yaml"
    config_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(
        experiment_runner,
        "run_legacy_advanced",
        lambda **kwargs: pytest.fail("unsupported dispatch must not route to legacy"),
    )

    with pytest.raises(ValueError, match="Unsupported evaluation dispatch combination"):
        experiment_runner.main(["--config", str(config_path)])
