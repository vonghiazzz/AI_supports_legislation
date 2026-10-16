import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import backend.scripts.Evaluation.run_experiment as experiment_runner
from backend.scripts.Evaluation.core.representations import build_document_text
from backend.scripts.Evaluation.evaluation.metrics import build_metric_world_report
from backend.scripts.Evaluation.experiments import (
    exp05_frozen_hybrid30_zero_shot_reranker as exp05,
)
from backend.scripts.Evaluation.reporting import serialization
from backend.scripts.Evaluation.reporting.serialization import write_exp05_artifacts
from backend.scripts.Evaluation.retrieval.fusion import fuse_pools


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = (
    PROJECT_ROOT
    / "backend/scripts/Evaluation/configs/exp05_frozen_hybrid30_zero_shot_reranker.yaml"
)
RUN_DIR = (
    PROJECT_ROOT
    / "backend/data/runs"
    / exp05.EXP05_SOURCE_RUN_ID
)


def _load_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _set_nested(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.fixture(scope="module")
def frozen_inputs():
    config = _load_config()
    candidate_cache = json.loads(
        (RUN_DIR / "candidate_pools.json").read_text(encoding="utf-8")
    )
    fold_manifest = json.loads(
        (RUN_DIR / "fold_manifest.json").read_text(encoding="utf-8")
    )
    records = exp05.validate_candidate_source(candidate_cache)
    mapping = exp05.validate_fold_source(
        fold_manifest, [record["query_id"] for record in records]
    )
    documents, _ = exp05.validate_frozen_corpus(config, PROJECT_ROOT)
    inference, controls = exp05.build_inference_records(
        records, documents, require_full=True
    )
    return {
        "records": records,
        "mapping": mapping,
        "documents": documents,
        "inference": inference,
        "controls": controls,
    }


@pytest.fixture(scope="module")
def full_prediction_artifact(frozen_inputs):
    scored = [
        {**record, "reranker_score": -float(record["original_hybrid_rank"])}
        for record in frozen_inputs["inference"]
    ]
    return exp05.freeze_prediction_artifact(scored, frozen_inputs["controls"])


def test_exp05_config_validator_accepts_exact_frozen_contract():
    assert exp05._validate_exp05_protocol(_load_config()) == {
        "protocol_id": exp05.EXP05_PROTOCOL_ID,
        "experiment_id": exp05.EXP05_EXPERIMENT_ID,
        "mode": exp05.EXP05_MODE,
        "query_count": 107,
        "candidate_count": 30,
        "inference_pair_count": 3210,
        "reported_k": (5, 10, 20),
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol", "id"), "other"),
        (("evaluation", "mode"), "baseline"),
        (("evaluation", "candidate_count"), 20),
        (("candidate_source", "fingerprint"), "other"),
        (("candidate_source", "w_bm25"), 0.4),
        (("frozen_inputs", "corpus_fingerprint"), "other"),
        (("model", "revision"), "other"),
        (("model", "local_files_only"), False),
        (("inference", "device"), "mps"),
        (("inference", "dtype"), "float16"),
        (("inference", "batch_size"), 2),
        (("inference", "max_length"), 512),
        (("inference", "truncation"), True),
        (("inference", "sigmoid"), True),
        (("inference", "blend_with_hybrid"), True),
        (("inference", "rerank_tie_break"), ["score"]),
        (("control", "closed_world_at_10", "weighted_recall"), 0.5),
        (("statistics", "bootstrap_samples"), 1000),
        (("statistics", "sign_flip_samples"), 1000),
        (("success_contract", "conditions"), []),
    ],
)
def test_exp05_config_validator_rejects_protocol_drift(path, value):
    config = _load_config()
    _set_nested(config, path, value)
    with pytest.raises(ValueError, match="EXP-05 requires"):
        exp05._validate_exp05_protocol(config)


def test_frozen_candidate_corpus_and_query_fingerprints_validate(frozen_inputs):
    assert len(frozen_inputs["records"]) == 107
    assert len(frozen_inputs["documents"]) == 1811
    assert len(frozen_inputs["inference"]) == 3210
    assert len(frozen_inputs["controls"]) == 107


def test_candidate_fingerprint_mutation_fails_closed():
    cache = json.loads((RUN_DIR / "candidate_pools.json").read_text(encoding="utf-8"))
    cache["records"][0]["bm25_pool"][0]["score"] += 1.0
    with pytest.raises(ValueError, match="computed candidate fingerprint"):
        exp05.validate_candidate_source(cache)


def test_exact_hybrid_top30_reconstruction_uses_full_pools(frozen_inputs):
    record = frozen_inputs["records"][0]
    expected = fuse_pools(record["bm25_pool"], record["dense_pool"], 0.5, "minmax")[:30]
    actual = exp05.build_frozen_hybrid_top30(record)
    assert [row["standard_law_id"] for row in actual] == [row[0] for row in expected]
    assert [row["score"] for row in actual] == [row[1] for row in expected]
    assert len(actual) == 30


def test_inference_view_is_exactly_gt_free_and_d0_is_frozen(frozen_inputs):
    assert all(set(record) == exp05.EXP05_INFERENCE_KEYS for record in frozen_inputs["inference"])
    assert not any(key in record for record in frozen_inputs["inference"] for key in exp05.GT_RESULT_KEYS)
    assert build_document_text(
        {"title": "  Điều   mẫu ", "content": " Nội dung   gốc "}, "d0"
    ) == "Tiêu đề: Điều mẫu. Nội dung: Nội dung gốc"


def test_inference_schema_rejects_gt_or_extra_fields(frozen_inputs):
    record = {**frozen_inputs["inference"][0], "core_gt_filtered": []}
    with pytest.raises(ValueError, match="exact GT-free"):
        exp05.validate_inference_records([record])


def test_rerank_uses_raw_scores_and_exact_tie_break(frozen_inputs):
    base = frozen_inputs["inference"][:4]
    scored = [
        {**base[0], "reranker_score": -2.0},
        {**base[1], "reranker_score": 3.0},
        {**base[2], "reranker_score": 3.0, "original_hybrid_rank": 7, "standard_law_id": "z"},
        {**base[3], "reranker_score": 3.0, "original_hybrid_rank": 7, "standard_law_id": "a"},
    ]
    ranked = exp05.rerank_scored_candidates(scored)
    assert [row["standard_law_id"] for row in ranked[:3]] == [
        base[1]["standard_law_id"],
        "a",
        "z",
    ]
    assert [row["reranker_score"] for row in ranked] == [3.0, 3.0, 3.0, -2.0]


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_rerank_rejects_nonfinite_raw_scores(frozen_inputs, score):
    with pytest.raises(ValueError, match="finite"):
        exp05.rerank_scored_candidates(
            [{**frozen_inputs["inference"][0], "reranker_score": score}]
        )


class _FakeTokenizer:
    load_kwargs = None

    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.load_kwargs = {"path": path, **kwargs}
        return cls()

    def __call__(self, queries, documents=None, **kwargs):
        self.calls.append(kwargs)
        batch = len(queries) if isinstance(queries, list) else 1
        return {"input_ids": torch.ones((batch, 3), dtype=torch.int64)}


class _FakeModel:
    load_kwargs = None

    def __init__(self):
        self.training = True
        self.to_kwargs = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.load_kwargs = {"path": path, **kwargs}
        return cls()

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self

    def eval(self):
        self.training = False
        return self

    def __call__(self, **kwargs):
        batch = int(kwargs["input_ids"].shape[0])
        return SimpleNamespace(logits=torch.arange(batch, dtype=torch.float32).reshape(-1, 1))


class _StaticLogitModel:
    def __init__(self, logits):
        self.logits = logits
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(logits=self.logits)


def test_local_loader_and_scorer_freeze_transformer_api_and_tokenization(frozen_inputs):
    tokenizer, model, torch_module, metadata = exp05.load_local_reranker(
        exp05.EXP05_MODEL_SNAPSHOT,
        tokenizer_class=_FakeTokenizer,
        model_class=_FakeModel,
        torch_module=torch,
    )
    assert _FakeTokenizer.load_kwargs["local_files_only"] is True
    assert _FakeModel.load_kwargs["local_files_only"] is True
    assert _FakeModel.load_kwargs["dtype"] is torch.float32
    assert _FakeModel.load_kwargs["use_safetensors"] is True
    assert model.training is False
    assert model.to_kwargs == {"device": "cpu", "dtype": torch.float32}
    assert metadata["device"] == "cpu"
    assert metadata["dtype"] == "float32"

    scored, timing = exp05.score_inference_records(
        frozen_inputs["inference"][:2], tokenizer, model, torch_module
    )
    assert len(scored) == 2
    assert all(math.isfinite(row["reranker_score"]) for row in scored)
    assert timing["pair_count"] == 2
    assert all(call["max_length"] == 1024 for call in tokenizer.calls)
    assert all(call["truncation"] == "only_second" for call in tokenizer.calls)
    assert all(call["padding"] is True for call in tokenizer.calls)
    assert all("return_tensors" in call for call in tokenizer.calls)


def test_scorer_accepts_exact_batch_by_one_logit_shape(frozen_inputs, monkeypatch):
    monkeypatch.setattr(exp05, "EXP05_BATCH_SIZE", 2)
    model = _StaticLogitModel(torch.tensor([[1.25], [-2.5]], dtype=torch.float32))
    scored, timing = exp05.score_inference_records(
        frozen_inputs["inference"][:2], _FakeTokenizer(), model, torch
    )
    assert [row["reranker_score"] for row in scored] == [1.25, -2.5]
    assert timing["pair_count"] == 2
    assert model.calls == 1


@pytest.mark.parametrize(
    "logits",
    [
        pytest.param(torch.zeros(2), id="batch"),
        pytest.param(torch.zeros((1, 2)), id="one-by-batch"),
        pytest.param(torch.zeros((2, 2)), id="batch-by-two"),
        pytest.param(torch.zeros((1, 1, 2)), id="three-dimensional"),
    ],
)
def test_scorer_rejects_noncanonical_logit_shapes(frozen_inputs, monkeypatch, logits):
    monkeypatch.setattr(exp05, "EXP05_BATCH_SIZE", 2)
    model = _StaticLogitModel(logits)
    with pytest.raises(RuntimeError, match="exact shape"):
        exp05.score_inference_records(
            frozen_inputs["inference"][:2], _FakeTokenizer(), model, torch
        )
    assert model.calls == 1


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_scorer_rejects_nonfinite_model_logits(frozen_inputs, score):
    model = _StaticLogitModel(torch.tensor([[score]], dtype=torch.float32))
    with pytest.raises(ValueError, match="NaN or Infinity"):
        exp05.score_inference_records(
            frozen_inputs["inference"][:1], _FakeTokenizer(), model, torch
        )
    assert model.calls == 1


def test_model_revision_and_hash_validation_logic(tmp_path):
    snapshot = tmp_path / exp05.EXP05_MODEL_REVISION
    snapshot.mkdir()
    expected = {}
    for filename in exp05.EXP05_MODEL_HASHES:
        payload = f"synthetic-{filename}".encode()
        (snapshot / filename).write_bytes(payload)
        expected[filename] = hashlib.sha256(payload).hexdigest()
    assert exp05.validate_model_artifacts(snapshot, expected) == expected
    (snapshot / "config.json").write_text("mutated", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        exp05.validate_model_artifacts(snapshot, expected)


def test_gt_mutation_cannot_change_inference_scores_order_or_fingerprint(frozen_inputs):
    original = frozen_inputs["records"][0]
    mutated = copy.deepcopy(original)
    for key in exp05.GT_RESULT_KEYS:
        mutated[key] = {"mutated": 1} if key == "gt_weights" else ["mutated"]

    original_view, original_control = exp05.build_inference_records(
        [original], frozen_inputs["documents"]
    )
    mutated_view, mutated_control = exp05.build_inference_records(
        [mutated], frozen_inputs["documents"]
    )
    original_scored = [
        {**row, "reranker_score": float(31 - row["original_hybrid_rank"])}
        for row in original_view
    ]
    mutated_scored = [
        {**row, "reranker_score": float(31 - row["original_hybrid_rank"])}
        for row in mutated_view
    ]
    first = exp05.freeze_prediction_artifact(original_scored, original_control)
    second = exp05.freeze_prediction_artifact(mutated_scored, mutated_control)
    assert original_view == mutated_view
    assert original_scored == mutated_scored
    assert first["predictions"] == second["predictions"]
    assert first["prediction_fingerprint"] == second["prediction_fingerprint"]


def test_evaluation_requires_valid_frozen_prediction(full_prediction_artifact, frozen_inputs):
    invalid = copy.deepcopy(full_prediction_artifact)
    invalid["frozen"] = False
    with pytest.raises(ValueError, match="requires a frozen"):
        exp05.attach_ground_truth_after_prediction_freeze(
            frozen_inputs["records"], invalid, frozen_inputs["mapping"]
        )
    invalid = copy.deepcopy(full_prediction_artifact)
    invalid["prediction_fingerprint"] = "other"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        exp05.attach_ground_truth_after_prediction_freeze(
            frozen_inputs["records"], invalid, frozen_inputs["mapping"]
        )


def test_evaluation_does_not_alias_frozen_prediction_rows(
    full_prediction_artifact, frozen_inputs
):
    before = copy.deepcopy(full_prediction_artifact)
    results = exp05.attach_ground_truth_after_prediction_freeze(
        frozen_inputs["records"], full_prediction_artifact, frozen_inputs["mapping"]
    )
    results[0]["metrics"][exp05.EXP05_CONTROL]["top_k"][0]["rank"] = 999
    results[0]["metrics"][exp05.EXP05_TREATMENT]["top_k"][0]["rank"] = 999
    assert full_prediction_artifact == before


def test_control_parity_gate_is_exact(full_prediction_artifact, frozen_inputs):
    results = exp05.attach_ground_truth_after_prediction_freeze(
        frozen_inputs["records"], full_prediction_artifact, frozen_inputs["mapping"]
    )
    worlds = build_metric_world_report(
        results, [exp05.EXP05_CONTROL], exp05.EXP05_REPORTED_K, 3, 1
    )
    parity = exp05.validate_control_parity(worlds)
    assert parity["passed"] is True
    assert parity["maximum_metric_error"] <= 1.0e-12


def test_bootstrap_and_sign_flip_are_deterministic():
    deltas = [0.0, 0.1, -0.05, 0.2, 0.0]
    assert exp05.paired_bootstrap(deltas, samples=1000, seed=42) == exp05.paired_bootstrap(
        deltas, samples=1000, seed=42
    )
    assert exp05.paired_sign_flip(deltas, samples=1000, seed=42) == exp05.paired_sign_flip(
        deltas, samples=1000, seed=42
    )


def _successful_statistical_analysis():
    metric_payload = {
        "paired_bootstrap": {"ci_95_percentile": [0.01, 0.2]},
        "paired_sign_flip": {"p_value": 0.01},
    }
    return {
        "metrics": {
            name: copy.deepcopy(metric_payload)
            for name in ("weighted_recall", "recall", "ndcg", "mrr")
        },
        "per_query": [
            {"query_id": fold + 1, "outer_fold": fold, "deltas": {"weighted_recall": 0.1}}
            for fold in range(5)
        ],
        "wr10_win_tie_loss": {"wins": 5, "ties": 0, "losses": 0},
        "fold_mean_wr10_deltas": {str(fold): 0.1 for fold in range(5)},
    }


def test_success_contract_requires_every_predeclared_condition():
    worlds = {
        "closed_world": {
            exp05.EXP05_CONTROL: {"10": {"weighted_recall": 0.1}},
            exp05.EXP05_TREATMENT: {"10": {"weighted_recall": 0.2}},
        }
    }
    statistical = _successful_statistical_analysis()
    result = exp05.evaluate_success_contract(worlds, statistical)
    assert result["all_conditions_hold"] is True
    assert set(result["conditions"]) == set(exp05.EXP05_SUCCESS_CONDITIONS)
    statistical["metrics"]["weighted_recall"]["paired_sign_flip"]["p_value"] = 0.05
    result = exp05.evaluate_success_contract(worlds, statistical)
    assert result["all_conditions_hold"] is False
    assert result["decision"] == "DO_NOT_PROMOTE"


def _minimal_manifest(prediction_artifact):
    return {
        "run_id": "run",
        "timestamp": "2026-10-16T00:00:00+00:00",
        "protocol_id": exp05.EXP05_PROTOCOL_ID,
        "experiment_id": exp05.EXP05_EXPERIMENT_ID,
        "evaluation_mode": exp05.EXP05_MODE,
        "candidate_fingerprint": exp05.EXP05_CANDIDATE_FINGERPRINT,
        "corpus_fingerprint": exp05.EXP05_CORPUS_FINGERPRINT,
        "query_fingerprint": exp05.EXP05_QUERY_FINGERPRINT,
        "fold_mapping_fingerprint": exp05.EXP05_FOLD_FINGERPRINT,
        "model_id": exp05.EXP05_MODEL_ID,
        "model_revision": exp05.EXP05_MODEL_REVISION,
        "model_file_sha256": exp05.EXP05_MODEL_HASHES,
        "python_version": "3.10.0",
        "package_versions": {},
        "device": "cpu",
        "dtype": "float32",
        "batch_size": 1,
        "max_length": 1024,
        "truncation": "only_second",
        "candidate_count": 30,
        "prediction_fingerprint": prediction_artifact["prediction_fingerprint"],
        "git_commit": "abc",
        "git_dirty": False,
        "source_files": [],
    }


def test_artifact_schema_is_gt_free_and_strict_json(full_prediction_artifact):
    artifacts = {
        "run_manifest.json": _minimal_manifest(full_prediction_artifact),
        "prediction_artifact.json": full_prediction_artifact,
        "summary_metrics.json": {},
        "statistical_analysis.json": {},
    }
    assert exp05.validate_exp05_artifacts(artifacts) is True
    invalid = copy.deepcopy(artifacts)
    invalid["prediction_artifact.json"]["predictions"][0]["core_gt_filtered"] = []
    invalid["prediction_artifact.json"]["prediction_fingerprint"] = exp05.fingerprint_predictions(
        invalid["prediction_artifact.json"]["predictions"]
    )
    with pytest.raises(ValueError, match="GT-free"):
        exp05.validate_exp05_artifacts(invalid)
    invalid = copy.deepcopy(artifacts)
    invalid["summary_metrics.json"] = {"bad": math.nan}
    with pytest.raises(ValueError):
        exp05.validate_exp05_artifacts(invalid)


def test_exp05_writer_is_atomic_strict_and_run_scoped(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("protocol: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp05-run"
    artifacts = {filename: {} for filename in exp05.EXP05_ARTIFACT_FILES}
    output = write_exp05_artifacts(run_dir, config_path, artifacts)
    assert output == run_dir / "summary_metrics.json"
    assert {path.name for path in run_dir.iterdir()} == set(exp05.EXP05_ARTIFACT_FILES) | {
        "eval_config.snapshot.yaml"
    }
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def test_exp05_writer_rejects_nan_and_removes_partial_directory(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("protocol: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp05-run"
    artifacts = {filename: {} for filename in exp05.EXP05_ARTIFACT_FILES}
    artifacts["summary_metrics.json"] = {"bad": math.nan}
    with pytest.raises(ValueError):
        write_exp05_artifacts(run_dir, config_path, artifacts)
    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def test_exp05_writer_removes_partial_directory_on_failure(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("protocol: {}\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "exp05-run"
    artifacts = {filename: {} for filename in exp05.EXP05_ARTIFACT_FILES}
    real_dump = serialization.json.dump
    calls = 0

    def fail_second_dump(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic EXP-05 artifact failure")
        return real_dump(*args, **kwargs)

    monkeypatch.setattr(serialization.json, "dump", fail_second_dump)
    with pytest.raises(OSError, match="synthetic EXP-05 artifact failure"):
        write_exp05_artifacts(run_dir, config_path, artifacts)
    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(f".{run_dir.name}.partial-*"))


def _clean_exp05_provenance(commit="a" * 40, **overrides):
    provenance = {
        "git_commit": commit,
        "git_dirty": False,
        "tracked_code_dirty": False,
        "baseline_source_dirty": False,
        "baseline_source_status": [],
    }
    provenance.update(overrides)
    return provenance


def _install_mocked_exp05_runner(
    monkeypatch,
    *,
    provenances=None,
    source_snapshots=None,
    token_audit=None,
    parity_failure=False,
):
    records = []
    inference = []
    controls = []
    mapping = {}
    for query_id in range(1, exp05.EXP05_QUERY_COUNT + 1):
        records.append(
            {
                "query_id": query_id,
                "query": f"query-{query_id:03d}",
                "core_gt_filtered": [],
                "support_gt_filtered": [],
                "core_gt_unfiltered": [],
                "support_gt_unfiltered": [],
                "gt_weights": {},
            }
        )
        mapping[query_id] = (query_id - 1) % exp05.EXP05_FOLD_COUNT
        control_rows = []
        for rank in range(1, exp05.EXP05_CANDIDATE_COUNT + 1):
            identifier = f"law-{query_id:03d}-{rank:02d}"
            inference.append(
                {
                    "query_id": query_id,
                    "query": f"query-{query_id:03d}",
                    "standard_law_id": identifier,
                    "document": f"document-{identifier}",
                    "original_hybrid_rank": rank,
                }
            )
            control_rows.append(
                {
                    "rank": rank,
                    "standard_law_id": identifier,
                    "score": float(exp05.EXP05_CANDIDATE_COUNT - rank),
                }
            )
        controls.append({"query_id": query_id, "top_30": control_rows})

    state = {
        "events": [],
        "tokenizer_constructions": 0,
        "model_loads": 0,
        "score_calls": 0,
        "scored_order": [],
        "writer_calls": 0,
        "written_artifacts": None,
    }
    provenances = list(provenances or [_clean_exp05_provenance()])
    source_snapshots = list(
        source_snapshots
        or [[{"path": "exp05-source.py", "sha256": "1" * 64}]]
    )
    provenance_index = 0
    source_index = 0

    def fake_git_provenance(*args, **kwargs):
        nonlocal provenance_index
        state["events"].append("provenance")
        value = provenances[min(provenance_index, len(provenances) - 1)]
        provenance_index += 1
        return copy.deepcopy(value)

    def fake_source_provenance(*args, **kwargs):
        nonlocal source_index
        state["events"].append("source_hashes")
        value = source_snapshots[min(source_index, len(source_snapshots) - 1)]
        source_index += 1
        return copy.deepcopy(value)

    monkeypatch.setattr(exp05, "validate_candidate_source", lambda cache: records)
    monkeypatch.setattr(exp05, "validate_fold_source", lambda manifest, ids: mapping)
    monkeypatch.setattr(exp05, "validate_frozen_corpus", lambda config, root: ({}, []))
    monkeypatch.setattr(
        exp05,
        "build_inference_records",
        lambda candidate_records, documents, require_full=False: (inference, controls),
    )
    monkeypatch.setattr(exp05, "git_provenance", fake_git_provenance)
    monkeypatch.setattr(exp05, "_source_file_provenance", fake_source_provenance)

    def fake_model_hashes(*args, **kwargs):
        state["events"].append("model_hashes")
        return copy.deepcopy(exp05.EXP05_MODEL_HASHES)

    def fake_load(*args, **kwargs):
        state["events"].append("model_load")
        state["tokenizer_constructions"] += 1
        state["model_loads"] += 1
        return object(), object(), object(), {"load_seconds": 0.0}

    default_audit = {
        "total_pairs": exp05.EXP05_INFERENCE_PAIR_COUNT,
        "query_budget_threats": 0,
        "query_preservation_failures": 0,
        "truncation_side": "document_only",
    }

    def fake_audit(inference_records, tokenizer):
        state["events"].append("token_audit")
        return copy.deepcopy(token_audit or default_audit)

    def fake_score(inference_records, tokenizer, model, torch_module):
        state["events"].append("score")
        state["score_calls"] += 1
        state["scored_order"] = [
            (row["query_id"], row["original_hybrid_rank"])
            for row in inference_records
        ]
        scored = [
            {**row, "reranker_score": -float(row["original_hybrid_rank"])}
            for row in inference_records
        ]
        return scored, {
            "inference_seconds": 1.0,
            "pairs_per_second": float(len(scored)),
            "pair_count": len(scored),
        }

    monkeypatch.setattr(exp05, "validate_model_artifacts", fake_model_hashes)
    monkeypatch.setattr(exp05, "load_local_reranker", fake_load)
    monkeypatch.setattr(exp05, "audit_token_lengths", fake_audit)
    monkeypatch.setattr(exp05, "score_inference_records", fake_score)

    real_freeze = exp05.freeze_prediction_artifact
    real_attach = exp05.attach_ground_truth_after_prediction_freeze

    def tracked_freeze(scored_records, control_records):
        state["events"].append("freeze")
        return real_freeze(scored_records, control_records)

    def tracked_attach(candidate_records, prediction_artifact, fold_mapping):
        assert "freeze" in state["events"]
        state["events"].append("attach_gt")
        frozen_before = copy.deepcopy(prediction_artifact)
        results = real_attach(candidate_records, prediction_artifact, fold_mapping)
        assert prediction_artifact == frozen_before
        return results

    monkeypatch.setattr(exp05, "freeze_prediction_artifact", tracked_freeze)
    monkeypatch.setattr(exp05, "attach_ground_truth_after_prediction_freeze", tracked_attach)

    def fake_metric_worlds(results, configurations, reported_k, **kwargs):
        state["events"].append(
            "control_metrics" if configurations == [exp05.EXP05_CONTROL] else "all_metrics"
        )
        worlds = {}
        for world in ("closed_world", "open_world"):
            worlds[world] = {
                configuration: {
                    str(k): copy.deepcopy(exp05.EXP05_CONTROL_AT_10)
                    for k in reported_k
                }
                for configuration in configurations
            }
        if parity_failure and configurations == [exp05.EXP05_CONTROL]:
            worlds["closed_world"][exp05.EXP05_CONTROL]["10"]["weighted_recall"] += 0.1
        return worlds

    monkeypatch.setattr(exp05, "build_metric_world_report", fake_metric_worlds)
    monkeypatch.setattr(
        exp05,
        "build_statistical_analysis",
        lambda results: {
            "metrics": {},
            "per_query": [],
            "wr10_win_tie_loss": {"wins": 0, "ties": 107, "losses": 0},
            "fold_mean_wr10_deltas": {str(fold): 0.0 for fold in range(5)},
            "harmful_queries": [],
        },
    )
    monkeypatch.setattr(
        exp05,
        "evaluate_success_contract",
        lambda worlds, statistics: {
            "conditions": {},
            "all_conditions_hold": False,
            "decision": "DO_NOT_PROMOTE",
        },
    )

    def fake_writer(run_dir, config_path, artifacts):
        state["events"].append("writer")
        state["writer_calls"] += 1
        state["written_artifacts"] = copy.deepcopy(artifacts)
        return Path(run_dir) / "summary_metrics.json"

    monkeypatch.setattr(exp05, "write_exp05_artifacts", fake_writer)
    return state


@pytest.mark.parametrize("dirty_field", ["git_dirty", "tracked_code_dirty", "baseline_source_dirty"])
def test_runner_rejects_dirty_state_before_model_load(monkeypatch, dirty_field):
    provenance = _clean_exp05_provenance(**{dirty_field: True})
    state = _install_mocked_exp05_runner(monkeypatch, provenances=[provenance])
    with pytest.raises(RuntimeError, match="clean worktree"):
        exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    assert "model_load" not in state["events"]
    assert state["writer_calls"] == 0


@pytest.mark.parametrize("commit", [None, "not-a-valid-commit"])
def test_runner_rejects_missing_or_invalid_commit_before_model_load(monkeypatch, commit):
    state = _install_mocked_exp05_runner(
        monkeypatch,
        provenances=[_clean_exp05_provenance(commit=commit)],
    )
    with pytest.raises(RuntimeError, match="valid committed git HEAD"):
        exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    assert "model_load" not in state["events"]
    assert state["writer_calls"] == 0


def test_runner_rejects_head_mutation_after_inference(monkeypatch):
    state = _install_mocked_exp05_runner(
        monkeypatch,
        provenances=[
            _clean_exp05_provenance("a" * 40),
            _clean_exp05_provenance("b" * 40),
        ],
    )
    with pytest.raises(RuntimeError, match="HEAD changed during inference"):
        exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    assert state["score_calls"] == 1
    assert len(state["scored_order"]) == exp05.EXP05_INFERENCE_PAIR_COUNT
    assert state["writer_calls"] == 0


def test_runner_rejects_source_mutation_after_inference(monkeypatch):
    state = _install_mocked_exp05_runner(
        monkeypatch,
        source_snapshots=[
            [{"path": "exp05-source.py", "sha256": "1" * 64}],
            [{"path": "exp05-source.py", "sha256": "2" * 64}],
        ],
    )
    with pytest.raises(RuntimeError, match="source/config files changed"):
        exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    assert state["score_calls"] == 1
    assert state["writer_calls"] == 0


@pytest.mark.parametrize(
    "failed_field",
    ["query_budget_threats", "query_preservation_failures"],
)
def test_query_preservation_failure_prevents_scoring(monkeypatch, failed_field):
    audit = {
        "total_pairs": exp05.EXP05_INFERENCE_PAIR_COUNT,
        "query_budget_threats": 0,
        "query_preservation_failures": 0,
        "truncation_side": "document_only",
    }
    audit[failed_field] = 1
    state = _install_mocked_exp05_runner(monkeypatch, token_audit=audit)
    with pytest.raises(RuntimeError, match=failed_field):
        exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    assert state["score_calls"] == 0
    assert state["writer_calls"] == 0


def test_full_mocked_runner_preserves_call_order_and_frozen_artifact(monkeypatch):
    state = _install_mocked_exp05_runner(monkeypatch)
    output = exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    assert output.name == "summary_metrics.json"
    assert state["tokenizer_constructions"] == 1
    assert state["model_loads"] == 1
    assert state["score_calls"] == 1
    assert state["scored_order"] == [
        (query_id, rank)
        for query_id in range(1, exp05.EXP05_QUERY_COUNT + 1)
        for rank in range(1, exp05.EXP05_CANDIDATE_COUNT + 1)
    ]
    assert state["events"].index("provenance") < state["events"].index("model_load")
    assert state["events"].index("freeze") < state["events"].index("attach_gt")
    assert state["writer_calls"] == 1
    manifest = state["written_artifacts"]["run_manifest.json"]
    assert manifest["git_commit"] == "a" * 40
    assert manifest["source_files"] == [
        {"path": "exp05-source.py", "sha256": "1" * 64}
    ]


def test_control_parity_failure_publishes_no_run_or_partial_directory(monkeypatch):
    runs_root = PROJECT_ROOT / "backend/data/runs"
    before = {path.name for path in runs_root.iterdir()}
    state = _install_mocked_exp05_runner(monkeypatch, parity_failure=True)
    with pytest.raises(RuntimeError, match="control parity failed"):
        exp05.run_exp05_frozen_hybrid30_zero_shot_reranker(CONFIG_PATH)
    after = {path.name for path in runs_root.iterdir()}
    assert after == before
    assert state["writer_calls"] == 0
    assert not any(name.startswith(f".{exp05.EXP05_PROTOCOL_ID}_") for name in after)


def test_cli_dispatches_exp05_without_running_real_model(monkeypatch):
    calls = []

    def fake_exp05(path, max_queries=None):
        calls.append((path, max_queries))
        return "synthetic-exp05-result"

    monkeypatch.setattr(
        experiment_runner,
        "run_exp05_frozen_hybrid30_zero_shot_reranker",
        fake_exp05,
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_exp04_selective_dynamic_hybrid_nested_cv",
        lambda *args, **kwargs: pytest.fail("EXP-05 must not route to EXP-04"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_exp03_static_fusion_cv",
        lambda *args, **kwargs: pytest.fail("EXP-05 must not route to EXP-03"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_baseline_evaluation_pipeline",
        lambda *args, **kwargs: pytest.fail("EXP-05 must not route to baseline"),
    )
    monkeypatch.setattr(
        experiment_runner,
        "run_legacy_advanced",
        lambda *args, **kwargs: pytest.fail("EXP-05 must not route to legacy"),
    )
    assert experiment_runner.main(["--config", str(CONFIG_PATH)]) == "synthetic-exp05-result"
    assert calls == [(CONFIG_PATH, None)]
