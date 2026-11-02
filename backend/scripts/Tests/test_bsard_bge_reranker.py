import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.Evaluation.experiments import bsard_bge_reranker as runner


def source_ranking(prefix=""):
    return [
        {"rank": rank, "standard_law_id": str(rank), "score": float(501 - rank)}
        for rank in range(1, 501)
    ]


def source_records(qids=("1", "2")):
    return [{"query_id": qid, "ranking": source_ranking()} for qid in qids]


def cache_fixture(tmp_path, pairs=None):
    pairs = pairs or [("1", "1"), ("1", "2")]
    contents = [
        {
            "query_id": qid,
            "document_id": did,
            "query_text_fingerprint": f"q{qid}",
            "document_text_fingerprint": f"d{did}",
            "pair_fingerprint": f"p{qid}-{did}",
        }
        for qid, did in pairs
    ]
    source = tmp_path / "runner.py"; source.write_text("pass\n")
    config = tmp_path / "config.yaml"; config.write_text("x: 1\n")
    context = runner.cache_context(
        pairs=pairs, contents=contents, model_hashes=runner.MODEL_HASHES,
        runner=source, config=config, commit="a" * 40,
    )
    cache = tmp_path / runner.CACHE_SCHEMA; cache.mkdir()
    scores = np.asarray([0.25, -0.5], dtype=np.float32)[:len(pairs)]
    metadata = runner.save_final_cache(cache / "scores.npz", scores, context, {"seconds": 1.0})
    return cache, context, scores, metadata, pairs, contents


def rewrite_cache(cache, scores, metadata):
    runner._atomic_npz(
        cache / "scores.npz", scores=np.asarray(scores),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    )


def test_exact_model_revision_and_contract_are_frozen():
    assert runner.MODEL_ID == "BAAI/bge-reranker-v2-m3"
    assert runner.MODEL_REVISION == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert (runner.MAX_LENGTH, runner.TRUNCATION, runner.BATCH_SIZE, runner.DEVICE, runner.DTYPE) == (1024, "only_second", 1, "cpu", "float32")


def test_candidate_extraction_n20_n50_and_prefix():
    pools = runner.extract_candidate_pools(source_records(), ["1", "2"])
    assert len(pools[20]["1"]) == 20
    assert len(pools[50]["1"]) == 50
    assert pools[20]["1"] == pools[50]["1"][:20]


def test_candidate_extraction_rejects_qid_mismatch():
    with pytest.raises(runner.BSARDRerankerError, match="qid mismatch"):
        runner.extract_candidate_pools(source_records(("1",)), ["1", "2"])


def test_candidate_extraction_rejects_duplicate():
    records = source_records(("1",)); records[0]["ranking"][1]["standard_law_id"] = "1"
    with pytest.raises(runner.BSARDRerankerError, match="uniqueness"):
        runner.extract_candidate_pools(records, ["1"])


def test_unique_pair_union_shared_pair_scored_once_and_ordered():
    bge = {"2": ["3", "1"], "1": ["2", "1"]}
    rrf = {"2": ["2", "1"], "1": ["3", "1"]}
    pairs, counts = runner.build_unique_pairs(bge, rrf, ["1", "2"])
    assert pairs == [("1", "1"), ("1", "2"), ("1", "3"), ("2", "1"), ("2", "2"), ("2", "3")]
    assert pairs.count(("1", "1")) == 1
    assert counts["unique_union"] == 6


def test_pair_order_fingerprint_detects_reordering():
    pairs = [("1", "1"), ("1", "2")]
    assert runner.ordered_pair_fingerprint(pairs) != runner.ordered_pair_fingerprint(list(reversed(pairs)))
    assert runner.pair_set_fingerprint(pairs) == runner.pair_set_fingerprint(list(reversed(pairs)))


@pytest.mark.parametrize("field,value", [
    ("model_revision", "wrong"),
    ("model_artifact_fingerprints", {"wrong": "hash"}),
    ("corpus_fingerprint", "wrong"),
    ("bge_source_retrieval_fingerprint", "wrong"),
    ("candidate_pair_set_fingerprint", "wrong"),
    ("ordered_pair_fingerprint", "wrong"),
    ("pair_content_fingerprint", "wrong"),
])
def test_cache_rejects_provenance_mismatch(tmp_path, field, value):
    cache, context, *_ = cache_fixture(tmp_path)
    changed = dict(context); changed[field] = value
    with pytest.raises(runner.BSARDRerankerError, match="cache context mismatch"):
        runner.load_final_cache(cache, changed)


def test_context_rejects_recomputed_field_without_context_fingerprint(tmp_path):
    _, context, *_ = cache_fixture(tmp_path)
    changed = dict(context); changed["git_commit"] = "b" * 40
    with pytest.raises(runner.BSARDRerankerError, match="provenance mismatch"):
        runner.validate_context(changed, changed)


def test_partial_cache_rejected_as_final(tmp_path):
    cache, context, *_ = cache_fixture(tmp_path)
    partial = cache.with_name(cache.name + ".partial"); cache.rename(partial)
    with pytest.raises(runner.BSARDRerankerError, match="partial cache"):
        runner.load_final_cache(partial, context)


def test_cache_rejects_wrong_score_count(tmp_path):
    cache, context, _, metadata, *_ = cache_fixture(tmp_path)
    rewrite_cache(cache, np.asarray([1.0], dtype=np.float32), metadata)
    with pytest.raises(runner.BSARDRerankerError, match="vector validation"):
        runner.load_final_cache(cache, context)


def test_cache_rejects_wrong_dtype(tmp_path):
    cache, context, scores, metadata, *_ = cache_fixture(tmp_path)
    rewrite_cache(cache, scores.astype(np.float64), metadata)
    with pytest.raises(runner.BSARDRerankerError, match="vector validation"):
        runner.load_final_cache(cache, context)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_cache_rejects_nonfinite_scores(tmp_path, bad):
    cache, context, scores, metadata, *_ = cache_fixture(tmp_path)
    scores = scores.copy(); scores[0] = bad
    rewrite_cache(cache, scores, metadata)
    with pytest.raises(runner.BSARDRerankerError, match="vector validation"):
        runner.load_final_cache(cache, context)


def test_cache_rejects_score_fingerprint_mismatch(tmp_path):
    cache, context, scores, metadata, *_ = cache_fixture(tmp_path)
    scores = scores.copy(); scores[0] += 1
    rewrite_cache(cache, scores, metadata)
    with pytest.raises(runner.BSARDRerankerError, match="vector validation"):
        runner.load_final_cache(cache, context)


def test_shard_resume_cannot_cross_provenance(tmp_path):
    _, context, scores, _, pairs, contents = cache_fixture(tmp_path)
    path = tmp_path / "shard.npz"
    metadata = runner._shard_metadata(context, pairs, contents, 0, len(pairs), scores)
    runner.save_shard(path, scores, metadata)
    changed = dict(context); changed["context_fingerprint"] = "wrong"
    expected = runner._shard_metadata(changed, pairs, contents, 0, len(pairs), scores)
    with pytest.raises(runner.BSARDRerankerError, match="provenance mismatch"):
        runner.load_shard(path, expected, len(pairs))


def test_reranked_head_set_tail_and_depth_invariants():
    source = source_ranking(); scores = {("1", str(i)): float(-i) for i in range(1, 51)}
    result = runner.rerank_one(source, "1", 20, scores)
    assert len(result) == len({item["standard_law_id"] for item in result}) == 500
    assert {item["standard_law_id"] for item in result[:20]} == {str(i) for i in range(1, 21)}
    assert [item["standard_law_id"] for item in result[20:]] == [str(i) for i in range(21, 501)]


def test_no_candidate_outside_head_can_enter():
    source = source_ranking(); scores = {("1", str(i)): float(i) for i in range(1, 21)}
    result = runner.rerank_one(source, "1", 20, scores)
    assert [item["standard_law_id"] for item in result[:3]] == ["20", "19", "18"]
    assert "21" not in {item["standard_law_id"] for item in result[:20]}


def test_deterministic_equal_score_tie_uses_original_rank():
    source = source_ranking(); scores = {("1", str(i)): 0.0 for i in range(1, 21)}
    result = runner.rerank_one(source, "1", 20, scores)
    assert [item["standard_law_id"] for item in result[:20]] == [str(i) for i in range(1, 21)]


def test_rerank_rejects_missing_authoritative_score():
    with pytest.raises(KeyError):
        runner.rerank_one(source_ranking(), "1", 20, {})


class MockTokenizer:
    def __call__(self, left, right, **kwargs):
        import torch
        return {"input_ids": torch.ones((len(left), 3), dtype=torch.long)}


class MockModel:
    def __init__(self, shape, values=None): self.shape, self.values = shape, values
    def __call__(self, **kwargs):
        import torch
        values = self.values if self.values is not None else torch.zeros(self.shape)
        return SimpleNamespace(logits=values)


@pytest.mark.parametrize("shape", [(1,), (1, 2), (1, 1, 1)])
def test_score_batch_rejects_wrong_logit_shape(shape):
    import torch
    with pytest.raises(runner.BSARDRerankerError, match="exact shape"):
        runner.score_batch([("1", "1")], {"1": "q"}, {"1": "d"}, MockTokenizer(), MockModel(shape), torch)


def test_score_batch_accepts_exact_batch_one_scalar():
    import torch
    result = runner.score_batch([("1", "1")], {"1": "q"}, {"1": "d"}, MockTokenizer(), MockModel((1, 1)), torch)
    assert result.shape == (1,) and result.dtype == np.float32


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_score_batch_rejects_nonfinite(bad):
    import torch
    with pytest.raises(runner.BSARDRerankerError, match="NaN or Infinity"):
        runner.score_batch([("1", "1")], {"1": "q"}, {"1": "d"}, MockTokenizer(), MockModel((1, 1), torch.tensor([[bad]])), torch)


def test_query_preservation_gate():
    runner.enforce_query_preservation({"query_budget_threats": 0, "query_preservation_failures": 0})
    with pytest.raises(runner.BSARDRerankerError):
        runner.enforce_query_preservation({"query_budget_threats": 1, "query_preservation_failures": 0})
    with pytest.raises(runner.BSARDRerankerError):
        runner.enforce_query_preservation({"query_budget_threats": 0, "query_preservation_failures": 1})


def test_failure_decomposition_distinguishes_all_three_query_outcomes():
    sources = source_records(("1", "2", "3"))
    reranked = source_records(("1", "2", "3"))
    queries = (
        {"query_id": "1", "relevant_document_ids": ["999"]},
        {"query_id": "2", "relevant_document_ids": ["20"]},
        {"query_id": "3", "relevant_document_ids": ["10"]},
    )

    result = runner.failure_and_promotion(sources, reranked, 20, queries)

    assert result["query_level"] == {
        "candidate_failure": 1,
        "ranking_failure": 1,
        "success": 1,
        "total": 3,
        "conditional_top10_success_rate": 0.5,
    }
    reference = result["reference_level"]
    assert reference["absent_candidate_pool"] == 1
    assert reference["present_below_top10"] == 1
    assert reference["present_top10"] == 1
    assert sum(reference[key] for key in (
        "absent_candidate_pool", "present_below_top10", "present_top10"
    )) == 3


def test_strict_json_rejects_nonfinite(tmp_path):
    path = tmp_path / "bad.json"; path.write_text('{"score": NaN}')
    with pytest.raises(runner.BSARDRerankerError, match="non-finite"):
        runner.strict_load(path)


def test_strict_json_writer_rejects_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        runner.strict_write(tmp_path / "bad.json", {"score": float("inf")})


def test_model_hash_set_covers_weights_config_and_tokenizer():
    assert {"model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json"} == set(runner.MODEL_HASHES)


def test_classifications_are_from_closed_vocabularies():
    row = {metric: {"mean_delta": 0.0, "ci_lower": -0.1, "ci_upper": 0.1} for metric in runner.SENSITIVE_METRICS}
    comparisons = {"bge_n50_vs_source": row, "rrf_n50_vs_source": row}
    assert runner.classify_reranker(comparisons) == "RERANKER_NO_RELIABLE_GAIN"
    assert runner.classify_generator(row) == "NO_CLEAR_GENERATOR_WINNER"
