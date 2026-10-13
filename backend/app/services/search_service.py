from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from app.config import settings
from app.services.embedding_service import embedding_service

from scripts.Evaluation.core.query import normalize_query_q0
from scripts.Evaluation.retrieval.bm25 import (
    build_bm25_index,
    search_bm25,
)
from scripts.Evaluation.retrieval.fusion import fuse_pools


class SearchService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print(
                "[*] Initializing frozen Runtime Retrieval Service..."
            )

            instance = super().__new__(cls)

            instance.index = None
            instance.metadata = []
            instance.documents = []
            instance.bm25 = None
            instance.manifest = None

            instance.metadata_by_standard_id = {}

            instance.data_loaded = False
            instance.ready = False
            instance.load_error = None

            instance._faiss = None

            cls._instance = instance

        return cls._instance

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()

        with path.open("rb") as handle:
            for chunk in iter(
                lambda: handle.read(1024 * 1024),
                b"",
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @staticmethod
    def _load_json(path: Path):
        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            return json.load(handle)

    def _validate_manifest_contract(
        self,
        manifest: dict,
    ) -> None:
        required_values = {
            "runtime_contract_id": settings.RUNTIME_CONTRACT_ID,
            "baseline_protocol_id": settings.RUNTIME_PROTOCOL_ID,
            "query_representation": "q0",
            "document_representation": "d0",
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
            "static_w_bm25": (
                settings.RUNTIME_STATIC_W_BM25
            ),
            "static_w_dense": (
                1.0 - settings.RUNTIME_STATIC_W_BM25
            ),
            "tie_break_policy": "standard_law_id_asc",
        }

        mismatches = []

        for key, expected in required_values.items():
            actual = manifest.get(key)

            if actual != expected:
                mismatches.append(
                    f"{key}: expected={expected!r}, "
                    f"actual={actual!r}"
                )

        if mismatches:
            raise RuntimeError(
                "Runtime retrieval manifest contract mismatch:\n- "
                + "\n- ".join(mismatches)
            )

        fields = manifest.get(
            "retrieval_document_fields",
            {},
        )

        if fields.get("bm25") != "bm25_text":
            raise RuntimeError(
                "Manifest BM25 document field must be bm25_text"
            )

        if fields.get("dense") != "vector_text":
            raise RuntimeError(
                "Manifest Dense document field must be vector_text"
            )

        artifacts = manifest.get("artifacts", {})

        expected_index_name = Path(
            settings.RUNTIME_INDEX_PATH
        ).name
        expected_metadata_name = Path(
            settings.RUNTIME_METADATA_PATH
        ).name

        if (
            artifacts.get("index", {}).get("filename")
            != expected_index_name
        ):
            raise RuntimeError(
                "Manifest index filename does not match "
                "runtime configuration"
            )

        if (
            artifacts.get("metadata", {}).get("filename")
            != expected_metadata_name
        ):
            raise RuntimeError(
                "Manifest metadata filename does not match "
                "runtime configuration"
            )

    def _validate_artifact_hashes(
        self,
        manifest: dict,
        index_path: Path,
        metadata_path: Path,
    ) -> None:
        artifacts = manifest["artifacts"]

        expected_index_sha = artifacts[
            "index"
        ].get("sha256")

        expected_metadata_sha = artifacts[
            "metadata"
        ].get("sha256")

        actual_index_sha = self._sha256_file(
            index_path
        )

        actual_metadata_sha = self._sha256_file(
            metadata_path
        )

        if actual_index_sha != expected_index_sha:
            raise RuntimeError(
                "Runtime FAISS SHA256 mismatch"
            )

        if actual_metadata_sha != expected_metadata_sha:
            raise RuntimeError(
                "Runtime metadata SHA256 mismatch"
            )

    def _validate_metadata(
        self,
        metadata: list,
    ) -> None:
        if len(metadata) != settings.EXPECTED_CORPUS_COUNT:
            raise RuntimeError(
                "Runtime metadata count mismatch: "
                f"expected "
                f"{settings.EXPECTED_CORPUS_COUNT}, "
                f"got {len(metadata)}"
            )

        standard_ids = []

        for index, item in enumerate(metadata):
            standard_id = item.get(
                "standard_law_id"
            )

            if not standard_id:
                raise RuntimeError(
                    "Missing standard_law_id at "
                    f"metadata index {index}"
                )

            standard_ids.append(standard_id)

            if "bm25_text" not in item:
                raise RuntimeError(
                    "Missing bm25_text at "
                    f"metadata index {index}"
                )

            if "vector_text" not in item:
                raise RuntimeError(
                    "Missing vector_text at "
                    f"metadata index {index}"
                )

            if (
                item["bm25_text"]
                != item["vector_text"]
            ):
                raise RuntimeError(
                    "D0 contract violation at "
                    f"metadata index {index}: "
                    "BM25 and Dense representation differ"
                )

        if len(set(standard_ids)) != len(standard_ids):
            raise RuntimeError(
                "Duplicate standard_law_id detected "
                "in runtime metadata"
            )

    def ensure_ready(self) -> None:
        if self.ready:
            return

        try:
            manifest_path = Path(
                settings.RUNTIME_MANIFEST_PATH
            )
            index_path = Path(
                settings.RUNTIME_INDEX_PATH
            )
            metadata_path = Path(
                settings.RUNTIME_METADATA_PATH
            )

            for path in (
                manifest_path,
                index_path,
                metadata_path,
            ):
                if not path.exists():
                    raise RuntimeError(
                        "Required runtime retrieval "
                        f"artifact is missing: {path}"
                    )

            print(
                "[*] Loading runtime manifest: "
                f"{manifest_path}"
            )

            manifest = self._load_json(
                manifest_path
            )

            self._validate_manifest_contract(
                manifest
            )

            self._validate_artifact_hashes(
                manifest,
                index_path,
                metadata_path,
            )

            print(
                "[*] Loading runtime metadata: "
                f"{metadata_path}"
            )

            metadata = self._load_json(
                metadata_path
            )

            self._validate_metadata(
                metadata
            )

            documents = [
                {
                    "standard_law_id": item[
                        "standard_law_id"
                    ],
                    "bm25_text": item[
                        "bm25_text"
                    ],
                    "vector_text": item[
                        "vector_text"
                    ],
                }
                for item in metadata
            ]

            print(
                "[*] Building frozen whitespace BM25 index..."
            )

            bm25 = build_bm25_index(
                documents
            )

            # IMPORTANT:
            # Load sentence-transformers / PyTorch before FAISS.
            # This ordering avoids the macOS native-library
            # segmentation fault reproduced during parity testing.
            embedding_service.ensure_loaded()

            import faiss

            faiss.omp_set_num_threads(1)

            print(
                "[*] Loading runtime FAISS index: "
                f"{index_path}"
            )

            index = faiss.read_index(
                str(index_path)
            )

            if type(index).__name__ != "IndexFlatIP":
                raise RuntimeError(
                    "Runtime FAISS type mismatch: "
                    f"{type(index).__name__}"
                )

            if (
                int(index.d)
                != settings.EXPECTED_EMBEDDING_DIMENSION
            ):
                raise RuntimeError(
                    "Runtime FAISS dimension mismatch: "
                    f"expected "
                    f"{settings.EXPECTED_EMBEDDING_DIMENSION}, "
                    f"got {index.d}"
                )

            if (
                int(index.ntotal)
                != settings.EXPECTED_CORPUS_COUNT
            ):
                raise RuntimeError(
                    "Runtime FAISS vector count mismatch: "
                    f"expected "
                    f"{settings.EXPECTED_CORPUS_COUNT}, "
                    f"got {index.ntotal}"
                )

            if int(index.ntotal) != len(metadata):
                raise RuntimeError(
                    "FAISS/metadata alignment mismatch: "
                    f"{index.ntotal} != {len(metadata)}"
                )

            self.manifest = manifest
            self.metadata = metadata
            self.documents = documents
            self.bm25 = bm25
            self.index = index
            self._faiss = faiss

            self.metadata_by_standard_id = {
                item["standard_law_id"]: item
                for item in metadata
            }

            self.data_loaded = True
            self.ready = True
            self.load_error = None

            print(
                "✅ Runtime retrieval ready: "
                f"{len(metadata)} articles, "
                f"d={index.d}, "
                f"ntotal={index.ntotal}"
            )

        except Exception as exc:
            self.ready = False
            self.data_loaded = False
            self.load_error = str(exc)

            print(
                "❌ Runtime retrieval initialization failed: "
                f"{exc}"
            )

            raise

    def status(self) -> dict:
        return {
            "status": (
                "ready"
                if self.ready
                else "not_ready"
            ),
            "ready": self.ready,
            "runtime_contract_id": (
                self.manifest.get(
                    "runtime_contract_id"
                )
                if self.manifest
                else settings.RUNTIME_CONTRACT_ID
            ),
            "baseline_protocol_id": (
                self.manifest.get(
                    "baseline_protocol_id"
                )
                if self.manifest
                else settings.RUNTIME_PROTOCOL_ID
            ),
            "articles_loaded": len(
                self.metadata
            ),
            "embedding_model": (
                settings.EMBEDDING_MODEL
            ),
            "embedding_revision": (
                settings.EMBEDDING_REVISION
            ),
            "embedding_dimension": (
                settings.EXPECTED_EMBEDDING_DIMENSION
            ),
            "error": self.load_error,
        }

    def _search_dense_pool(
        self,
        query_text: str,
        pool_k: int,
    ) -> list:
        query_vector = embedding_service.encode(
            query_text
        ).astype("float32").reshape(1, -1)

        self._faiss.normalize_L2(
            query_vector
        )

        scores, indices = self.index.search(
            query_vector,
            pool_k,
        )

        return [
            {
                "rank": rank,
                "standard_law_id": self.documents[
                    int(document_index)
                ]["standard_law_id"],
                "score": float(score),
            }
            for rank, (score, document_index)
            in enumerate(
                zip(
                    scores[0],
                    indices[0],
                ),
                1,
            )
            if document_index >= 0
        ]

    def _format_result(
        self,
        standard_id: str,
        hybrid_score: float,
        bm25_map: dict,
        dense_map: dict,
    ) -> Dict[str, Any]:
        item = self.metadata_by_standard_id[
            standard_id
        ]

        return {
            "standard_law_id": standard_id,
            "law_id": (
                item.get("law_id")
                or item.get("original_id")
            ),
            "title": item.get(
                "title",
                "Không có tiêu đề",
            ),
            "content": item.get(
                "content",
                "",
            ),
            "normalized_content": item.get(
                "normalized_content"
            ),
            "summary": item.get("summary"),
            "keywords": item.get(
                "keywords",
                [],
            ),
            "lexical_keywords": item.get(
                "lexical_keywords",
                [],
            ),
            "legal_concepts": item.get(
                "legal_concepts",
                [],
            ),
            "category": item.get("category"),
            "chapter": item.get("chapter"),
            "section": item.get("section"),
            "type": item.get("type"),
            "priority": item.get(
                "priority",
                0.5,
            ),
            "tags": item.get(
                "tags",
                [],
            ),
            "score": float(hybrid_score),
            "source": item.get(
                "source_key"
            ),
            "vector_score": (
                float(dense_map[standard_id])
                if standard_id in dense_map
                else None
            ),
            "bm25_score": (
                float(bm25_map[standard_id])
                if standard_id in bm25_map
                else None
            ),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Canonical runtime retrieval path.

        Contract:
          Q0 -> BM25(D0) + Dense(D0)
             -> min-max 0.5/0.5
             -> deterministic standard_law_id tie-break.
        """
        self.ensure_ready()

        if top_k < 1:
            raise ValueError(
                "top_k must be >= 1"
            )

        if top_k > 20:
            raise ValueError(
                "runtime_retrieval_v1 supports top_k <= 20"
            )

        query_text = normalize_query_q0(
            query
        )

        if not query_text:
            return []

        pool_k = settings.RUNTIME_POOL_K

        bm25_pool = search_bm25(
            self.bm25,
            self.documents,
            query_text,
            pool_k,
        )

        dense_pool = self._search_dense_pool(
            query_text,
            pool_k,
        )

        hybrid_pairs = fuse_pools(
            bm25_pool,
            dense_pool,
            settings.RUNTIME_STATIC_W_BM25,
            settings.RUNTIME_FUSION_METHOD,
        )

        bm25_map = {
            item["standard_law_id"]:
            item["score"]
            for item in bm25_pool
        }

        dense_map = {
            item["standard_law_id"]:
            item["score"]
            for item in dense_pool
        }

        return [
            self._format_result(
                standard_id,
                score,
                bm25_map,
                dense_map,
            )
            for standard_id, score
            in hybrid_pairs[:top_k]
        ]

    def search_hybrid(
        self,
        queries,
        keywords=None,
        top_k: int = 5,
    ):
        """
        Backward-compatible wrapper.

        Expanded LLM queries / keyword lists are intentionally ignored by
        the frozen runtime retrieval contract. The first query is used as
        Q0 input.
        """
        if isinstance(queries, str):
            query = queries
        elif queries:
            query = queries[0]
        else:
            query = ""

        return self.search(
            query,
            top_k=top_k,
        )


search_service = SearchService()
