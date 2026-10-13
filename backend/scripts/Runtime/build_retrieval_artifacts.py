from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path

import yaml

from backend.scripts.Evaluation.core.corpus import (
    load_documents,
    load_frozen_baseline_corpus,
    sha256_file,
)
from backend.scripts.Evaluation.experiments.baseline import (
    BASELINE_PROTOCOL_ID,
    _validate_baseline_protocol,
)
from backend.scripts.Evaluation.reporting.manifest import (
    fingerprint_records,
    resolved_model_revision,
    verify_model_revision,
)
from backend.scripts.Evaluation.retrieval.dense import build_dense_retriever


RUNTIME_CONTRACT_ID = "runtime_retrieval_v1"

EXPECTED_CORPUS_COUNT = 1811
EXPECTED_CORPUS_FINGERPRINT = (
    "66fb0b20e0505993ed616f864ece99ec"
    "fad801de98b4ba3a37d20a3a7bf55e58"
)
EXPECTED_EMBEDDING_DIMENSION = 768

DEFAULT_INDEX_NAME = "runtime_law_index.faiss"
DEFAULT_METADATA_NAME = "runtime_law_metadata.json"
DEFAULT_MANIFEST_NAME = "runtime_retrieval_manifest.json"


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_runtime_metadata(
    law_files: list[Path],
    documents: list[dict],
) -> list[dict]:
    metadata = []
    document_index = 0

    for law_file in law_files:
        source_key = (
            Path(law_file)
            .stem
            .replace("standardized_", "", 1)
        )

        with Path(law_file).open("r", encoding="utf-8") as handle:
            records = json.load(handle)

        for record in records:
            if document_index >= len(documents):
                raise ValueError(
                    "Metadata/document alignment overflow while "
                    f"processing {law_file}"
                )

            document = documents[document_index]

            row = dict(record)
            row.update(
                {
                    "standard_law_id": document["standard_law_id"],
                    "original_id": document["original_id"],
                    "bm25_text": document["bm25_text"],
                    "vector_text": document["vector_text"],
                    "law_domain": document["law_domain"],
                    "law_year": document["law_year"],
                    "active_start": document["active_start"],
                    "active_end": document["active_end"],
                    "source_key": source_key,
                }
            )

            metadata.append(row)
            document_index += 1

    if document_index != len(documents):
        raise ValueError(
            "Metadata/document alignment mismatch: "
            f"metadata={document_index}, documents={len(documents)}"
        )

    return metadata


def _validate_contract(
    config: dict,
    project_root: Path,
) -> tuple[
    dict,
    list[Path],
    list[dict],
    list[dict],
    list[dict],
    str,
]:
    protocol = _validate_baseline_protocol(config)

    if protocol["document_representation"] != "d0":
        raise ValueError(
            "Runtime recovery v1 requires document_representation=d0"
        )

    if protocol["query_representation"] != "q0":
        raise ValueError(
            "Runtime recovery v1 requires query_representation=q0"
        )

    output_dir = project_root / config["paths"]["output_dir"]
    laws_config_path = project_root / config["paths"]["laws_config"]

    law_files, corpus_inventory, corpus_count = (
        load_frozen_baseline_corpus(
            laws_config_path,
            output_dir,
            config["corpus"],
        )
    )

    if corpus_count != EXPECTED_CORPUS_COUNT:
        raise ValueError(
            f"Runtime corpus count mismatch: "
            f"expected {EXPECTED_CORPUS_COUNT}, got {corpus_count}"
        )

    documents = load_documents(
        law_files,
        document_representation=protocol[
            "document_representation"
        ],
        bm25_tokenizer=protocol["bm25_tokenizer"],
        laws_config_path=laws_config_path,
    )

    if len(documents) != corpus_count:
        raise ValueError(
            "Document count changed during representation build: "
            f"{len(documents)} != {corpus_count}"
        )

    if any(
        document["bm25_text"] != document["vector_text"]
        for document in documents
    ):
        raise AssertionError(
            "Runtime D0 requires BM25 and Dense to use "
            "the exact same document representation"
        )

    standard_ids = [
        document["standard_law_id"]
        for document in documents
    ]

    if len(set(standard_ids)) != len(standard_ids):
        raise ValueError(
            "Duplicate standard_law_id detected in frozen corpus"
        )

    metadata = _build_runtime_metadata(
        law_files,
        documents,
    )

    if len(metadata) != corpus_count:
        raise ValueError(
            f"Metadata count mismatch: "
            f"{len(metadata)} != {corpus_count}"
        )

    corpus_fingerprint = fingerprint_records(
        [
            {
                "key": item["key"],
                "count": item["count"],
                "sha256": item["sha256"],
            }
            for item in corpus_inventory
        ]
    )

    if corpus_fingerprint != EXPECTED_CORPUS_FINGERPRINT:
        raise ValueError(
            "Frozen corpus fingerprint mismatch:\n"
            f"expected={EXPECTED_CORPUS_FINGERPRINT}\n"
            f"actual={corpus_fingerprint}"
        )

    return (
        protocol,
        law_files,
        corpus_inventory,
        documents,
        metadata,
        corpus_fingerprint,
    )


def _write_json_atomic(path: Path, payload) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )
        handle.write("\n")

    os.replace(temp_path, path)


def build(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[3]

    config_path = (
        Path(args.config).resolve()
        if args.config
        else project_root
        / "backend"
        / "scripts"
        / "Evaluation"
        / "configs"
        / "baseline_d0.yaml"
    )

    config = _load_config(config_path)

    (
        protocol,
        law_files,
        corpus_inventory,
        documents,
        metadata,
        corpus_fingerprint,
    ) = _validate_contract(
        config,
        project_root,
    )

    embedding_model = config["models"]["embedding_model"]
    embedding_revision = config["models"]["embedding_revision"]
    local_files_only = bool(
        config["models"].get("local_files_only", True)
    )

    summary = {
        "runtime_contract_id": RUNTIME_CONTRACT_ID,
        "baseline_protocol_id": BASELINE_PROTOCOL_ID,
        "corpus_count": len(documents),
        "metadata_count": len(metadata),
        "unique_standard_law_ids": len(
            {
                document["standard_law_id"]
                for document in documents
            }
        ),
        "combined_corpus_fingerprint": corpus_fingerprint,
        "query_representation": protocol[
            "query_representation"
        ],
        "document_representation": protocol[
            "document_representation"
        ],
        "bm25_tokenizer": protocol["bm25_tokenizer"],
        "fusion_method": protocol["fusion_method"],
        "static_w_bm25": protocol["static_w_bm25"],
        "tie_break_policy": protocol["tie_break_policy"],
        "embedding_model": embedding_model,
        "embedding_requested_revision": embedding_revision,
        "embedding_local_files_only": local_files_only,
    }

    print("=== RUNTIME RETRIEVAL CONTRACT VALIDATION ===")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.validate_only:
        print()
        print("PASS: runtime retrieval contract validated")
        print("No artifacts were written.")
        return

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root / "backend" / "data"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / DEFAULT_INDEX_NAME
    metadata_path = output_dir / DEFAULT_METADATA_NAME
    manifest_path = output_dir / DEFAULT_MANIFEST_NAME

    print()
    print("=== BUILDING DENSE ARTIFACT ===")
    print(f"model={embedding_model}")
    print(f"revision={embedding_revision}")
    print(f"documents={len(documents)}")
    print(f"device={args.device}")

    embedder, faiss_index, embedding_dimension = (
        build_dense_retriever(
            documents,
            embedding_model,
            embedding_revision,
            local_files_only=local_files_only,
            device=args.device,
        )
    )

    resolved_revision = resolved_model_revision(embedder)
    verify_model_revision(
        embedding_revision,
        resolved_revision,
    )

    if embedding_dimension != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected {EXPECTED_EMBEDDING_DIMENSION}, "
            f"got {embedding_dimension}"
        )

    if int(faiss_index.ntotal) != EXPECTED_CORPUS_COUNT:
        raise ValueError(
            "FAISS vector count mismatch: "
            f"expected {EXPECTED_CORPUS_COUNT}, "
            f"got {faiss_index.ntotal}"
        )

    if type(faiss_index).__name__ != "IndexFlatIP":
        raise ValueError(
            "Unexpected FAISS type: "
            f"{type(faiss_index).__name__}"
        )

    import faiss

    index_tmp = index_path.with_name(
        f".{index_path.name}.tmp"
    )
    faiss.write_index(
        faiss_index,
        str(index_tmp),
    )
    os.replace(index_tmp, index_path)

    _write_json_atomic(
        metadata_path,
        metadata,
    )

    manifest = {
        "runtime_contract_id": RUNTIME_CONTRACT_ID,
        "baseline_protocol_id": BASELINE_PROTOCOL_ID,
        "built_at": dt.datetime.now(
            dt.timezone.utc
        ).isoformat(),
        "query_representation": protocol[
            "query_representation"
        ],
        "document_representation": protocol[
            "document_representation"
        ],
        "retrieval_document_fields": {
            "bm25": "bm25_text",
            "dense": "vector_text",
        },
        "corpus_count": len(documents),
        "metadata_count": len(metadata),
        "combined_corpus_fingerprint": corpus_fingerprint,
        "corpora": corpus_inventory,
        "embedding_model": embedding_model,
        "embedding_requested_revision": embedding_revision,
        "embedding_resolved_revision": resolved_revision,
        "embedding_revision_verified": (
            embedding_revision == resolved_revision
        ),
        "embedding_dimension": embedding_dimension,
        "embedding_normalized": True,
        "embedding_local_files_only": local_files_only,
        "faiss_index_type": type(
            faiss_index
        ).__name__,
        "faiss_ntotal": int(faiss_index.ntotal),
        "bm25_tokenizer": protocol[
            "bm25_tokenizer"
        ],
        "fusion_method": protocol["fusion_method"],
        "static_w_bm25": protocol[
            "static_w_bm25"
        ],
        "static_w_dense": (
            1.0 - protocol["static_w_bm25"]
        ),
        "tie_break_policy": protocol[
            "tie_break_policy"
        ],
        "artifacts": {
            "index": {
                "filename": index_path.name,
                "sha256": sha256_file(index_path),
            },
            "metadata": {
                "filename": metadata_path.name,
                "sha256": sha256_file(metadata_path),
            },
        },
    }

    _write_json_atomic(
        manifest_path,
        manifest,
    )

    print()
    print("=== BUILD COMPLETE ===")
    print(f"index={index_path}")
    print(f"metadata={metadata_path}")
    print(f"manifest={manifest_path}")
    print(f"dimension={embedding_dimension}")
    print(f"ntotal={faiss_index.ntotal}")
    print(f"metadata_count={len(metadata)}")
    print(f"fingerprint={corpus_fingerprint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic runtime retrieval artifacts "
            "from the frozen baseline_d0_v2 contract."
        )
    )

    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
