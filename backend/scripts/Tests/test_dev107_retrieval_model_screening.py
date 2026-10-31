import copy
import datetime
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.Evaluation.experiments import dev107_retrieval_model_screening as screening
from scripts.Evaluation.experiments.exp03_static_fusion_cv import fingerprint_candidate_pools


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/configs/dev107_retrieval_model_screening.yaml"
)


@pytest.fixture(scope="module")
def config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def candidate_cache(config):
    path = PROJECT_ROOT / config["frozen_inputs"]["exp03_candidate_path"]
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_path(config):
    return PROJECT_ROOT / config["frozen_inputs"]["exp03_candidate_path"]


def test_percentiles_returns_exact_latency_contract_for_non_empty_values():
    assert screening._percentiles([0.001, 0.002, 0.004, 0.008]) == {
        "mean_ms": 3.75,
        "p50_ms": 3.0,
        "p95_ms": 7.399999999999999,
    }


def test_percentiles_has_no_outer_scope_dependency(monkeypatch):
    monkeypatch.delattr(screening, "documents", raising=False)
    monkeypatch.delattr(screening, "records", raising=False)
    assert screening._percentiles([0.002]) == {
        "mean_ms": 2.0,
        "p50_ms": 2.0,
        "p95_ms": 2.0,
    }


def test_percentiles_preserves_empty_and_single_value_behavior():
    assert screening._percentiles([]) == {
        "mean_ms": None,
        "p50_ms": None,
        "p95_ms": None,
    }
    assert screening._percentiles([0.005]) == {
        "mean_ms": 5.0,
        "p50_ms": 5.0,
        "p95_ms": 5.0,
    }


def test_retrieve_all_aggregates_latency_without_changing_rankings(
    tmp_path, monkeypatch
):
    import sentence_transformers

    documents = [
        {"standard_law_id": "d1", "vector_text": "alpha"},
        {"standard_law_id": "d2", "vector_text": "beta"},
        {"standard_law_id": "d3", "vector_text": "gamma"},
    ]
    records = [{"query_id": 1, "query": "alpha"}]
    bm25 = [
        {"rank": 1, "standard_law_id": "d1", "score": 3.0},
        {"rank": 2, "standard_law_id": "d2", "score": 2.0},
    ]
    vietnamese_sbert = [
        {"rank": 1, "standard_law_id": "d2", "score": 0.9},
        {"rank": 2, "standard_law_id": "d3", "score": 0.8},
    ]
    bge_m3 = [
        {"rank": 1, "standard_law_id": "d3", "score": 0.7},
        {"rank": 2, "standard_law_id": "d1", "score": 0.6},
    ]
    config = {
        "evaluation": {"candidate_depths": [2]},
        "vietnamese_sbert": {
            "id": "synthetic/vietnamese-sbert",
            "revision": "a" * 40,
            "dimension": 768,
            "normalize_embeddings": True,
        },
        "bge_m3": {
            "id": "synthetic/bge-m3",
            "revision": "b" * 40,
            "dimension": 1024,
            "normalize_embeddings": True,
            "pooling": "cls",
            "query_instruction": None,
        },
    }
    for filename in ("vietnamese_sbert_embeddings.npz", "bge_m3_embeddings.npz"):
        (tmp_path / filename).touch()

    def fail_if_model_constructed(*args, **kwargs):
        raise AssertionError("synthetic retrieval test must not construct a model")

    def fake_load(path, expected):
        if path.name == "vietnamese_sbert_embeddings.npz":
            return (
                np.zeros((1811, 768), dtype=np.float32),
                np.zeros((107, 768), dtype=np.float32),
                [0.001],
                0.01,
                {"model_name": "Vietnamese_SBERT"},
            )
        return (
            np.zeros((1811, 1024), dtype=np.float32),
            np.zeros((107, 1024), dtype=np.float32),
            [0.002],
            0.02,
            {"model_name": "BGE_M3"},
        )

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fail_if_model_constructed)
    monkeypatch.setattr(screening, "_load_embedding_cache", fake_load)
    monkeypatch.setattr(screening, "build_bm25_index", lambda _documents: object())
    monkeypatch.setattr(screening, "search_bm25", lambda *args: copy.deepcopy(bm25))
    monkeypatch.setattr(
        screening,
        "_search_faiss",
        lambda index, *args: copy.deepcopy(
            vietnamese_sbert if index.d == 768 else bge_m3
        ),
    )
    monkeypatch.setattr(screening, "sha256_file", lambda path: path.name)

    output, timing, _ = screening._retrieve_all(
        config,
        records,
        documents,
        tmp_path,
        {},
        {"git_commit": "c" * 40},
    )

    assert output == [{
        "query_id": 1,
        "query": "alpha",
        "bm25": bm25,
        "vietnamese_sbert": vietnamese_sbert,
        "bge_m3": bge_m3,
        "candidate_pools": {"2": {
            "Current_Hybrid": [
                {"rank": 1, "standard_law_id": "d1", "score": 0.5},
                {"rank": 2, "standard_law_id": "d2", "score": 0.5},
            ],
            "BGE_M3_Hybrid_RRF": [
                {
                    "rank": 1,
                    "standard_law_id": "d1",
                    "score": 0.03252247488101534,
                },
                {
                    "rank": 2,
                    "standard_law_id": "d3",
                    "score": 0.01639344262295082,
                },
            ],
        }},
    }]
    assert timing["query_latency"]["Vietnamese_SBERT"] == {
        "mean_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
    }
    assert timing["query_latency"]["BGE_M3"] == {
        "mean_ms": 2.0,
        "p50_ms": 2.0,
        "p95_ms": 2.0,
    }
    assert set(timing["query_latency"]["BM25"]) == {
        "mean_ms", "p50_ms", "p95_ms"
    }


def test_valid_candidate_cache_accepted(config, candidate_cache):
    records, _, _ = screening._validate_candidate_and_source(
        candidate_cache, _candidate_path(config), config["frozen_inputs"]
    )
    assert len(records) == 107


def test_candidate_records_modified_with_dishonest_metadata_rejected(config, candidate_cache):
    changed = copy.deepcopy(candidate_cache)
    changed["records"][0]["bm25_pool"][0]["score"] += 1.0
    changed["candidate_cache_fingerprint"] = fingerprint_candidate_pools(changed["records"])
    with pytest.raises(Exception, match="candidate fingerprint|CANDIDATE_INTEGRITY_FAILURE"):
        screening._validate_candidate_and_source(
            changed, _candidate_path(config), config["frozen_inputs"]
        )


def test_candidate_stale_fingerprint_rejected(config, candidate_cache):
    changed = copy.deepcopy(candidate_cache)
    changed["candidate_cache_fingerprint"] = "0" * 64
    with pytest.raises(Exception, match="candidate fingerprint"):
        screening._validate_candidate_and_source(
            changed, _candidate_path(config), config["frozen_inputs"]
        )


def test_candidate_count_mismatch_rejected(config, candidate_cache):
    changed = copy.deepcopy(candidate_cache)
    changed["records"].pop()
    with pytest.raises(Exception, match="query|CANDIDATE_INTEGRITY_FAILURE"):
        screening._validate_candidate_and_source(
            changed, _candidate_path(config), config["frozen_inputs"]
        )


def test_candidate_record_order_change_rejected(config, candidate_cache):
    changed = copy.deepcopy(candidate_cache)
    changed["records"][0], changed["records"][1] = changed["records"][1], changed["records"][0]
    with pytest.raises(screening.Dev107IntegrityError, match="CANDIDATE_INTEGRITY_FAILURE"):
        screening._validate_candidate_and_source(
            changed, _candidate_path(config), config["frozen_inputs"]
        )


@pytest.mark.parametrize(
    "field",
    ["combined_corpus_fingerprint", "query_representation_fingerprint"],
)
def test_candidate_source_query_or_corpus_provenance_rejected(
    tmp_path, config, candidate_cache, field
):
    candidate_path = tmp_path / "candidate_pools.json"
    candidate_path.write_text(json.dumps(candidate_cache), encoding="utf-8")
    source = json.loads(
        _candidate_path(config).with_name("run_manifest.json").read_text(encoding="utf-8")
    )
    source[field] = "0" * 64
    candidate_path.with_name("run_manifest.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    with pytest.raises(screening.Dev107IntegrityError, match="CANDIDATE_INTEGRITY_FAILURE"):
        screening._validate_candidate_and_source(
            candidate_cache, candidate_path, config["frozen_inputs"]
        )


def _small_cache(tmp_path):
    path = tmp_path / "cache.npz"
    documents = np.arange(6, dtype=np.float32).reshape(3, 2)
    queries = np.arange(4, dtype=np.float32).reshape(2, 2)
    expected = {
        "cache_schema_version": screening.EMBEDDING_CACHE_SCHEMA,
        "model_name": "Synthetic",
        "model_id": "model/a",
        "model_revision": "a" * 40,
        "embedding_dimension": 2,
        "normalization": True,
        "pooling": "cls",
        "query_instruction": None,
        "document_count": 3,
        "query_count": 2,
        "corpus_fingerprint": "b" * 64,
        "document_representation_id": "d0",
        "document_representation_fingerprint": "c" * 64,
        "query_fingerprint": "d" * 64,
        "config_fingerprint": "e" * 64,
        "document_ids_fingerprint": "f" * 64,
        "query_ids_fingerprint": "1" * 64,
    }
    screening._write_embedding_cache(
        path,
        documents,
        queries,
        [0.1, 0.2],
        1.0,
        expected,
        {"runner_sha256": "2" * 64, "git_commit": "3" * 40},
    )
    return path, expected


def _rewrite_cache_metadata(path, mutate, *, recompute_fingerprint):
    with np.load(path, allow_pickle=False) as cached:
        values = {name: np.asarray(cached[name]) for name in cached.files}
    metadata = json.loads(str(values["metadata_json"].item()))
    mutate(metadata)
    if recompute_fingerprint:
        metadata.pop("metadata_fingerprint", None)
        metadata["metadata_fingerprint"] = screening._sha256_json(metadata)
    values["metadata_json"] = np.asarray(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
    )
    with open(path, "wb") as handle:
        np.savez(handle, **values)


def test_valid_embedding_cache_accepted(tmp_path):
    path, expected = _small_cache(tmp_path)
    documents, queries, _, _, metadata = screening._load_embedding_cache(path, expected)
    assert documents.shape == (3, 2)
    assert queries.shape == (2, 2)
    assert metadata["document_count"] == 3
    assert metadata["query_count"] == 2


def test_embedding_cache_context_derives_counts_from_ordered_inputs(config):
    documents = [
        {"standard_law_id": f"doc-{index}", "vector_text": f"document {index}"}
        for index in range(4)
    ]
    records = [
        {"query_id": f"query-{index}"}
        for index in range(3)
    ]
    before = screening._config_fingerprint(config)
    context = screening._embedding_cache_context(config, documents, records)
    assert context["document_count"] == 4
    assert context["query_count"] == 3
    assert context["document_ids_fingerprint"] == screening._sha256_json(
        [item["standard_law_id"] for item in documents]
    )
    assert context["query_ids_fingerprint"] == screening._sha256_json(
        [item["query_id"] for item in records]
    )
    assert screening._config_fingerprint(config) == before


@pytest.mark.parametrize("count_key", ["document_count", "query_count"])
def test_embedding_cache_missing_count_fails_explicitly(tmp_path, count_key):
    path, expected = _small_cache(tmp_path)
    _rewrite_cache_metadata(
        path,
        lambda metadata: metadata.pop(count_key),
        recompute_fingerprint=True,
    )
    with pytest.raises(
        screening.Dev107IntegrityError,
        match=f"EMBEDDING_PROVENANCE_FAILURE.*{count_key} mismatch",
    ):
        screening._load_embedding_cache(path, expected)


@pytest.mark.parametrize("count_key", ["document_count", "query_count"])
def test_embedding_cache_count_tampering_breaks_metadata_fingerprint(tmp_path, count_key):
    path, expected = _small_cache(tmp_path)
    _rewrite_cache_metadata(
        path,
        lambda metadata: metadata.__setitem__(count_key, metadata[count_key] + 1),
        recompute_fingerprint=False,
    )
    with pytest.raises(
        screening.Dev107IntegrityError,
        match="EMBEDDING_PROVENANCE_FAILURE.*metadata integrity mismatch",
    ):
        screening._load_embedding_cache(path, expected)


@pytest.mark.parametrize("count_key", ["document_count", "query_count"])
def test_coordinated_count_and_metadata_fingerprint_rewrite_rejected(tmp_path, count_key):
    path, expected = _small_cache(tmp_path)
    _rewrite_cache_metadata(
        path,
        lambda metadata: metadata.__setitem__(count_key, metadata[count_key] + 1),
        recompute_fingerprint=True,
    )
    with pytest.raises(
        screening.Dev107IntegrityError,
        match=f"EMBEDDING_PROVENANCE_FAILURE.*{count_key} mismatch",
    ):
        screening._load_embedding_cache(path, expected)


@pytest.mark.parametrize(
    ("matrix_name", "documents", "queries", "query_seconds", "expected_count"),
    [
        ("document", np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32), [0.1, 0.2], 3),
        ("query", np.zeros((3, 2), dtype=np.float32), np.zeros((3, 2), dtype=np.float32), [0.1, 0.2, 0.3], 4),
    ],
)
def test_embedding_matrix_row_count_mismatch_rejected(
    tmp_path, matrix_name, documents, queries, query_seconds, expected_count
):
    path = tmp_path / f"{matrix_name}-count-mismatch.npz"
    _, expected = _small_cache(tmp_path)
    expected = dict(expected)
    expected[f"{matrix_name}_count"] = expected_count
    screening._write_embedding_cache(
        path,
        documents,
        queries,
        query_seconds,
        1.0,
        expected,
        {"runner_sha256": "2" * 64, "git_commit": "3" * 40},
    )
    with pytest.raises(
        screening.Dev107IntegrityError,
        match="EMBEDDING_PROVENANCE_FAILURE.*matrix shape mismatch",
    ):
        screening._load_embedding_cache(path, expected)


def test_fresh_embedding_cache_immediate_self_validation_succeeds(tmp_path):
    path, expected = _small_cache(tmp_path)
    loaded = screening._load_embedding_cache(path, expected)
    assert loaded[0].shape[0] == expected["document_count"]
    assert loaded[1].shape[0] == expected["query_count"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "wrong/model"),
        ("model_revision", "9" * 40),
        ("corpus_fingerprint", "8" * 64),
        ("document_representation_id", "content_only"),
    ],
)
def test_embedding_semantic_provenance_mismatch_rejected(tmp_path, field, value):
    path, expected = _small_cache(tmp_path)
    changed = {**expected, field: value}
    with pytest.raises(screening.Dev107IntegrityError, match="EMBEDDING_PROVENANCE_FAILURE"):
        screening._load_embedding_cache(path, changed)


def test_embedding_matrix_tamper_rejected(tmp_path):
    path, expected = _small_cache(tmp_path)
    with np.load(path, allow_pickle=False) as cached:
        values = {name: np.asarray(cached[name]) for name in cached.files}
    values["documents"] = values["documents"].copy()
    values["documents"][0, 0] += 1.0
    with open(path, "wb") as handle:
        np.savez(handle, **values)
    with pytest.raises(screening.Dev107IntegrityError, match="matrix fingerprint"):
        screening._load_embedding_cache(path, expected)


def test_incomplete_legacy_embedding_cache_rejected(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, documents=np.zeros((3, 2)), queries=np.zeros((2, 2)))
    _, expected = _small_cache(tmp_path)
    with pytest.raises(screening.Dev107IntegrityError, match="CACHE_PROVENANCE_INCOMPLETE"):
        screening._load_embedding_cache(path, expected)


def test_valid_exp05_prediction_artifact_accepted(config, candidate_cache):
    records = screening.validate_candidate_source(candidate_cache)
    path = PROJECT_ROOT / config["frozen_inputs"]["exp05_prediction_path"]
    scores, provenance = screening._preseed_exp05(path, records, config)
    assert len(scores) == 3210
    assert len(provenance["prediction_fingerprint"]) == 64


def _copy_exp05_run(tmp_path, config):
    source = (PROJECT_ROOT / config["frozen_inputs"]["exp05_prediction_path"]).parent
    for name in (
        "run_manifest.json",
        "prediction_artifact.json",
        "summary_metrics.json",
        "statistical_analysis.json",
    ):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    return tmp_path / "prediction_artifact.json"


def test_exp05_wrong_prediction_fingerprint_rejected(tmp_path, config, candidate_cache):
    path = _copy_exp05_run(tmp_path, config)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["prediction_fingerprint"] = "0" * 64
    path.write_text(json.dumps(artifact), encoding="utf-8")
    records = screening.validate_candidate_source(candidate_cache)
    with pytest.raises(screening.Dev107IntegrityError, match="EXP05_ARTIFACT_INTEGRITY_FAILURE"):
        screening._preseed_exp05(path, records, config)


def test_exp05_candidate_source_mismatch_rejected(config, candidate_cache):
    path = PROJECT_ROOT / config["frozen_inputs"]["exp05_prediction_path"]
    records = copy.deepcopy(screening.validate_candidate_source(candidate_cache))
    records[0]["bm25_pool"][0]["score"] += 1000.0
    with pytest.raises(screening.Dev107IntegrityError, match="EXP05_ARTIFACT_INTEGRITY_FAILURE"):
        screening._preseed_exp05(path, records, config)


def test_exp05_parity_pass_allows_publication():
    parity = {
        name: {"absolute_error": screening.EXP05_PARITY_TOLERANCE}
        for name in screening.EXPECTED_EXP05_CONTROL
    }
    assert screening._enforce_exp05_parity(parity) == screening.EXP05_PARITY_TOLERANCE


def test_exp05_parity_failure_leaves_no_final_directory(tmp_path):
    partial = tmp_path / "run.partial"
    final = tmp_path / "run"
    partial.mkdir()
    parity = {"weighted_recall": {"absolute_error": 1e-6}}
    with pytest.raises(screening.Dev107IntegrityError, match="EXP05_PARITY_FAILURE"):
        screening._enforce_exp05_parity(parity)
    assert partial.is_dir()
    assert not final.exists()


def _synthetic_retrieval_artifact():
    documents = [
        {"standard_law_id": "d1", "vector_text": "alpha"},
        {"standard_law_id": "d2", "vector_text": "beta"},
        {"standard_law_id": "d3", "vector_text": "gamma"},
    ]
    source = [{"query_id": 1, "query": "alpha"}]
    bm25 = [
        {"rank": 1, "standard_law_id": "d1", "score": 3.0},
        {"rank": 2, "standard_law_id": "d2", "score": 2.0},
    ]
    dense = [
        {"rank": 1, "standard_law_id": "d2", "score": 0.9},
        {"rank": 2, "standard_law_id": "d3", "score": 0.8},
    ]
    bge = [
        {"rank": 1, "standard_law_id": "d3", "score": 0.7},
        {"rank": 2, "standard_law_id": "d1", "score": 0.6},
    ]
    current = screening._as_ranking(screening.fuse_pools(bm25, dense, 0.5, "minmax")[:2])
    modern = screening._as_ranking(screening._rrf(bm25, bge, 60)[:2])
    records = [{
        "query_id": 1,
        "query": "alpha",
        "bm25": bm25,
        "vietnamese_sbert": dense,
        "bge_m3": bge,
        "candidate_pools": {"2": {
            "Current_Hybrid": current,
            "BGE_M3_Hybrid_RRF": modern,
        }},
    }]
    provenance = {"frozen": True}
    artifact = {
        "schema_version": screening.RETRIEVAL_CACHE_SCHEMA,
        "protocol_id": screening.PROTOCOL_ID,
        "records": records,
        "timing": {},
        "provenance": provenance,
        "fingerprint": screening._sha256_json(records),
    }
    config = {"evaluation": {"candidate_depths": [2]}}
    return artifact, config, source, documents, provenance


def test_valid_resume_artifact_accepted():
    artifact, config, source, documents, provenance = _synthetic_retrieval_artifact()
    assert screening._validate_retrieval_artifact(
        artifact, config, source, documents, provenance
    ) == artifact["records"]


def test_resume_candidate_records_tampered_rejected():
    artifact, config, source, documents, provenance = _synthetic_retrieval_artifact()
    artifact["records"][0]["candidate_pools"]["2"]["Current_Hybrid"][0]["score"] += 1
    with pytest.raises(screening.Dev107IntegrityError, match="RESUME_INTEGRITY_FAILURE"):
        screening._validate_retrieval_artifact(artifact, config, source, documents, provenance)


def test_resume_coordinated_metadata_rewrite_still_rejected():
    artifact, config, source, documents, provenance = _synthetic_retrieval_artifact()
    artifact["records"][0]["candidate_pools"]["2"]["Current_Hybrid"][0]["score"] += 1
    artifact["fingerprint"] = screening._sha256_json(artifact["records"])
    artifact["provenance"] = provenance
    with pytest.raises(screening.Dev107IntegrityError, match="fail recomputation"):
        screening._validate_retrieval_artifact(artifact, config, source, documents, provenance)


def test_resume_directory_inside_run_root_accepted(tmp_path):
    run_root = tmp_path / "runs"
    resume = run_root / "safe.partial"
    resume.mkdir(parents=True)
    assert screening._resolve_resume_dir(resume, run_root) == resume.resolve()


def test_resume_directory_outside_run_root_rejected(tmp_path):
    run_root = tmp_path / "runs"
    outside = tmp_path / "outside.partial"
    run_root.mkdir()
    outside.mkdir()
    with pytest.raises(screening.Dev107IntegrityError, match="UNSAFE_RESUME_PATH"):
        screening._resolve_resume_dir(outside, run_root)


def test_resume_parent_traversal_rejected(tmp_path):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    with pytest.raises(screening.Dev107IntegrityError, match="UNSAFE_RESUME_PATH"):
        screening._resolve_resume_dir(Path("../escape.partial"), run_root)


def test_resume_symlink_escape_rejected(tmp_path):
    run_root = tmp_path / "runs"
    outside = tmp_path / "outside.partial"
    run_root.mkdir()
    outside.mkdir()
    link = run_root / "escape.partial"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(screening.Dev107IntegrityError, match="UNSAFE_RESUME_PATH"):
        screening._resolve_resume_dir(link, run_root)


def _clean_provenance(commit="a" * 40):
    return {
        "git_commit": commit,
        "git_dirty": False,
        "tracked_code_dirty": False,
        "baseline_source_dirty": False,
        "baseline_source_status": [],
    }


@pytest.mark.parametrize(
    "dirty_field", ["git_dirty", "tracked_code_dirty", "baseline_source_dirty"]
)
def test_dirty_or_uncommitted_pre_run_provenance_rejected(dirty_field):
    provenance = _clean_provenance()
    provenance[dirty_field] = True
    with pytest.raises(screening.Dev107IntegrityError, match="PRE_RUN_PROVENANCE_FAILURE"):
        screening._validate_pre_run_provenance(provenance)


def test_source_change_between_pre_run_and_publish_rejected(monkeypatch):
    expected = {
        "git_commit": "a" * 40,
        "source_paths": ["runner.py"],
        "source_files": [{"path": "runner.py", "sha256": "1" * 64}],
        "input_provenance": {"input": "same"},
    }
    monkeypatch.setattr(screening, "git_provenance", lambda *_: _clean_provenance())
    monkeypatch.setattr(
        screening,
        "_source_file_provenance",
        lambda *_: [{"path": "runner.py", "sha256": "2" * 64}],
    )
    with pytest.raises(screening.Dev107IntegrityError, match="TOCTOU_PUBLICATION_FAILURE"):
        screening._verify_prepublication_provenance(
            PROJECT_ROOT, expected, {"input": "same"}
        )


def test_head_change_between_pre_run_and_publish_rejected(monkeypatch):
    expected = {
        "git_commit": "a" * 40,
        "source_paths": [],
        "source_files": [],
        "input_provenance": {},
    }
    monkeypatch.setattr(
        screening, "git_provenance", lambda *_: _clean_provenance("b" * 40)
    )
    monkeypatch.setattr(screening, "_source_file_provenance", lambda *_: [])
    with pytest.raises(screening.Dev107IntegrityError, match="TOCTOU_PUBLICATION_FAILURE"):
        screening._verify_prepublication_provenance(PROJECT_ROOT, expected, {})


def test_successful_synthetic_publication_is_deterministic(tmp_path):
    manifests = []
    for number in (1, 2):
        partial = tmp_path / f"run-{number}.partial"
        final = tmp_path / f"run-{number}"
        partial.mkdir()
        retrieval = {"records": [], "fingerprint": screening._sha256_json([])}
        scores = {"pair_count": 0, "scores": {}, "fingerprint": screening._sha256_json({})}
        summary = {"scope": "DEVELOPMENT / EXPLORATORY"}
        statistics = {"seed": 42}
        manifest = {
            "final_result_fingerprint": screening._completed_result_fingerprint(
                retrieval["fingerprint"], scores["fingerprint"], summary, statistics
            )
        }
        for name, payload in (
            ("retrieval_candidates.json", retrieval),
            ("reranker_scores.json", scores),
            ("summary_metrics.json", summary),
            ("statistical_analysis.json", statistics),
            ("run_manifest.json", manifest),
        ):
            screening._strict_write(partial / name, payload)
        (partial / "eval_config.snapshot.yaml").write_bytes(CONFIG_PATH.read_bytes())
        screening._validate_partial_publication(partial, CONFIG_PATH)
        os.replace(partial, final)
        manifests.append(json.loads((final / "run_manifest.json").read_text()))
    assert manifests[0] == manifests[1]


@pytest.mark.parametrize("parity_failure", [False, True])
def test_full_mocked_runner_publication_contract(
    tmp_path, monkeypatch, config, parity_failure
):
    class FixedDateTime(datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 31, 0, 0, 0, tzinfo=tz)

    class FixedUuid:
        hex = "12345678" * 4

    monkeypatch.setattr(screening.dt, "datetime", FixedDateTime)
    monkeypatch.setattr(screening.uuid, "uuid4", lambda: FixedUuid())

    def fake_pre_run(project_root, config_path, config_fingerprint):
        return {
            "git_commit": "a" * 40,
            "git_state": _clean_provenance(),
            "source_paths": ["dev107_retrieval_model_screening.py"],
            "source_files": [{
                "path": "dev107_retrieval_model_screening.py",
                "sha256": "b" * 64,
            }],
            "config_fingerprint": config_fingerprint,
        }

    def fake_load_inputs(_config, project_root):
        candidate = project_root / "candidate_pools.json"
        candidate_manifest = project_root / "candidate_manifest.json"
        benchmark = project_root / "benchmark.json"
        for path in (candidate, candidate_manifest, benchmark):
            path.write_text("{}\n", encoding="utf-8")
        return [], [], [], candidate, candidate_manifest, benchmark, hashlib.sha256(b"{}\n").hexdigest()

    def fake_retrieve(_config, records, documents, work_dir, cache_context, creation_source):
        provenance = {}
        for name, filename in (
            ("Vietnamese_SBERT", "vietnamese_sbert_embeddings.npz"),
            ("BGE_M3", "bge_m3_embeddings.npz"),
        ):
            path = work_dir / filename
            path.write_bytes(name.encode("ascii"))
            provenance[name] = {
                "path": filename,
                "sha256": screening.sha256_file(path),
                "metadata": {"synthetic": True},
            }
        return [], {"synthetic": True}, provenance

    def fake_score(*args, **kwargs):
        return {}, {}, {
            "exp05_provenance": {"prediction_fingerprint": "c" * 64},
            "required": 0,
        }

    def fake_results(*args, **kwargs):
        output = {
            "retrieval_systems": {},
            "reranked_systems": {},
            "candidate_analysis": {},
            "statistics": {},
            "exp05_reproduction": {
                name: {"actual": value, "expected": value, "absolute_error": 0.0}
                for name, value in screening.EXPECTED_EXP05_CONTROL.items()
            },
            "timing": {},
        }
        if parity_failure:
            output["exp05_reproduction"]["weighted_recall"]["absolute_error"] = 1e-6
        return output

    monkeypatch.setattr(screening, "_capture_pre_run_provenance", fake_pre_run)
    monkeypatch.setattr(screening, "_load_inputs", fake_load_inputs)
    monkeypatch.setattr(screening, "_capture_input_provenance", lambda *args: {"stable": True})
    monkeypatch.setattr(screening, "_retrieve_all", fake_retrieve)
    monkeypatch.setattr(screening, "_validate_retrieval_artifact", lambda artifact, *args, **kwargs: artifact["records"])
    monkeypatch.setattr(screening, "_score_missing", fake_score)
    monkeypatch.setattr(screening, "_assemble_results", fake_results)
    monkeypatch.setattr(screening, "_verify_prepublication_provenance", lambda *args: None)

    manifests = []
    for number in ((1,) if parity_failure else (1, 2)):
        root = tmp_path / f"root-{number}"
        config_path = root / "backend/scripts/Evaluation/configs/dev107_retrieval_model_screening.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        (root / "backend/data/runs").mkdir(parents=True)
        if parity_failure:
            with pytest.raises(screening.Dev107IntegrityError, match="EXP05_PARITY_FAILURE"):
                screening.run(config_path, _project_root=root)
            final = root / "backend/data/runs/dev107_retrieval_model_screening_v1_20261031_000000_000000_12345678"
            assert not final.exists()
            assert len(list((root / "backend/data/runs").glob("*.partial"))) == 1
            continue
        final = screening.run(config_path, _project_root=root)
        manifests.append(json.loads((final / "run_manifest.json").read_text(encoding="utf-8")))
    if not parity_failure:
        assert manifests[0] == manifests[1]


def test_config_fingerprint_changes_for_semantic_option(config):
    original = screening._config_fingerprint(config)
    runtime_only = copy.deepcopy(config)
    runtime_only["_resolved_snapshot_paths"] = {"bge_m3": "/runtime-only"}
    assert screening._config_fingerprint(runtime_only) == original
    changed = copy.deepcopy(config)
    changed["fusion"]["bge_hybrid"]["rrf_k"] = 61
    assert screening._config_fingerprint(changed) != original
    with pytest.raises(ValueError, match="RRF"):
        screening._validate_config(changed)


def test_unknown_model_revision_is_reported_not_invented(config):
    changed = copy.deepcopy(config["bge_m3"])
    changed["revision"] = None
    with pytest.raises(screening.Dev107IntegrityError, match="revision is unknown"):
        screening._resolve_local_snapshot(changed)


def test_unknown_critical_config_field_rejected(config):
    changed = copy.deepcopy(config)
    changed["evaluation"]["unreviewed_option"] = True
    with pytest.raises(ValueError, match="unknown"):
        screening._validate_config(changed)
