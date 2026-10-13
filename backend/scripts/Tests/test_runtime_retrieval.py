from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError


BACKEND_ROOT = Path(__file__).resolve().parents[2]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


from app.config import settings
from app.models.schema import (
    AskRequest,
    SearchRequest,
    SearchResult,
)
from app.services.search_service import search_service


def valid_runtime_manifest():
    return {
        "runtime_contract_id": settings.RUNTIME_CONTRACT_ID,
        "baseline_protocol_id": settings.RUNTIME_PROTOCOL_ID,
        "query_representation": "q0",
        "document_representation": "d0",
        "retrieval_document_fields": {
            "bm25": "bm25_text",
            "dense": "vector_text",
        },
        "corpus_count": settings.EXPECTED_CORPUS_COUNT,
        "metadata_count": settings.EXPECTED_CORPUS_COUNT,
        "combined_corpus_fingerprint": (
            settings.EXPECTED_CORPUS_FINGERPRINT
        ),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_requested_revision": (
            settings.EMBEDDING_REVISION
        ),
        "embedding_resolved_revision": (
            settings.EMBEDDING_REVISION
        ),
        "embedding_dimension": (
            settings.EXPECTED_EMBEDDING_DIMENSION
        ),
        "embedding_normalized": True,
        "faiss_index_type": "IndexFlatIP",
        "faiss_ntotal": settings.EXPECTED_CORPUS_COUNT,
        "bm25_tokenizer": "whitespace",
        "fusion_method": settings.RUNTIME_FUSION_METHOD,
        "static_w_bm25": settings.RUNTIME_STATIC_W_BM25,
        "static_w_dense": (
            1.0 - settings.RUNTIME_STATIC_W_BM25
        ),
        "tie_break_policy": "standard_law_id_asc",
        "artifacts": {
            "index": {
                "filename": Path(
                    settings.RUNTIME_INDEX_PATH
                ).name,
                "sha256": "index-sha",
            },
            "metadata": {
                "filename": Path(
                    settings.RUNTIME_METADATA_PATH
                ).name,
                "sha256": "metadata-sha",
            },
        },
    }


def test_runtime_manifest_accepts_frozen_contract():
    search_service._validate_manifest_contract(
        valid_runtime_manifest()
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("runtime_contract_id", "runtime_retrieval_v2"),
        ("baseline_protocol_id", "baseline_d0_v1"),
        ("document_representation", "d1"),
        ("corpus_count", 1813),
        ("metadata_count", 1813),
        ("embedding_model", "BAAI/bge-m3"),
        ("embedding_dimension", 1024),
        ("faiss_index_type", "IndexFlatL2"),
        ("faiss_ntotal", 1813),
        ("bm25_tokenizer", "regex"),
        ("fusion_method", "rrf"),
        ("static_w_bm25", 0.4),
        ("static_w_dense", 0.6),
        ("tie_break_policy", "score_only"),
    ],
)
def test_runtime_manifest_rejects_contract_drift(
    field,
    invalid_value,
):
    manifest = deepcopy(
        valid_runtime_manifest()
    )
    manifest[field] = invalid_value

    with pytest.raises(RuntimeError):
        search_service._validate_manifest_contract(
            manifest
        )


def test_runtime_manifest_requires_d0_fields():
    manifest = deepcopy(
        valid_runtime_manifest()
    )
    manifest["retrieval_document_fields"][
        "bm25"
    ] = "content"

    with pytest.raises(RuntimeError):
        search_service._validate_manifest_contract(
            manifest
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, 21, 100],
)
def test_search_request_rejects_invalid_top_k(
    value,
):
    with pytest.raises(ValidationError):
        SearchRequest(
            query="test",
            top_k=value,
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, 21, 100],
)
def test_ask_request_rejects_invalid_top_k(
    value,
):
    with pytest.raises(ValidationError):
        AskRequest(
            query="test",
            top_k=value,
        )


def test_search_request_accepts_runtime_bounds():
    assert SearchRequest(
        query="test",
        top_k=1,
    ).top_k == 1

    assert SearchRequest(
        query="test",
        top_k=20,
    ).top_k == 20


def test_search_result_supports_canonical_id():
    result = SearchResult(
        standard_law_id=(
            "điều 1 bo_luat_dan_su_2015"
        ),
        law_id="1",
        title="Phạm vi điều chỉnh",
        content="Nội dung",
        score=0.5,
    )

    assert (
        result.standard_law_id
        == "điều 1 bo_luat_dan_su_2015"
    )


def test_search_hybrid_wrapper_uses_first_query(
    monkeypatch,
):
    calls = {}

    def fake_search(query, top_k=5):
        calls["query"] = query
        calls["top_k"] = top_k
        return [{"ok": True}]

    monkeypatch.setattr(
        search_service,
        "search",
        fake_search,
    )

    result = search_service.search_hybrid(
        [
            "original query",
            "expanded query",
        ],
        keywords={
            "keywords": ["ignored"],
        },
        top_k=7,
    )

    assert result == [{"ok": True}]
    assert calls == {
        "query": "original query",
        "top_k": 7,
    }
