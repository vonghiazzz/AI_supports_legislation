from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Native-library safeguards for macOS/PyTorch/FAISS coexistence.
# These must be set before importing sentence-transformers, torch, or FAISS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import yaml

from backend.scripts.Evaluation.core.query import build_query_text
from backend.scripts.Evaluation.reporting.manifest import (
    resolved_model_revision,
    verify_model_revision,
)
from backend.scripts.Evaluation.retrieval.bm25 import (
    build_bm25_index,
    search_bm25,
)
from backend.scripts.Evaluation.retrieval.dense import (
    encode_dense_query,
    search_faiss,
)
from backend.scripts.Evaluation.retrieval.fusion import fuse_pools


DEFAULT_SAMPLE = (
    0,
    1,
    2,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    104,
    106,
)

DEFAULT_GOLDEN_RUN = (
    "baseline_d0_v2_20260825_081357_617763_02bf9e13"
)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _ranking_ids(ranking: list[dict]) -> list[str]:
    return [
        item["standard_law_id"]
        for item in ranking
    ]


def _first_difference(
    expected: list[str],
    actual: list[str],
):
    size = max(len(expected), len(actual))

    for index in range(size):
        expected_id = (
            expected[index]
            if index < len(expected)
            else "<missing>"
        )
        actual_id = (
            actual[index]
            if index < len(actual)
            else "<missing>"
        )

        if expected_id != actual_id:
            return index + 1, expected_id, actual_id

    return None


def _max_score_delta(
    expected: list[dict],
    actual: list[dict],
) -> float | None:
    if _ranking_ids(expected) != _ranking_ids(actual):
        return None

    if not expected:
        return 0.0

    return max(
        abs(
            float(expected_item["score"])
            - float(actual_item["score"])
        )
        for expected_item, actual_item
        in zip(expected, actual)
    )


def _compare_ranking(
    query_id,
    name: str,
    expected: list[dict],
    actual: list[dict],
) -> bool:
    expected_ids = _ranking_ids(expected)
    actual_ids = _ranking_ids(actual)

    if expected_ids == actual_ids:
        delta = _max_score_delta(
            expected,
            actual,
        )

        print(
            f"  PASS {name:<21} "
            f"top-{len(expected_ids)} IDs match "
            f"(max_score_delta={delta:.12g})"
        )
        return True

    mismatch = _first_difference(
        expected_ids,
        actual_ids,
    )

    rank, expected_id, actual_id = mismatch

    print(
        f"  FAIL {name:<21} "
        f"first mismatch rank={rank}"
    )
    print(
        f"       expected: {expected_id}"
    )
    print(
        f"       actual  : {actual_id}"
    )
    print(
        f"       query_id: {query_id}"
    )

    return False


def verify(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[3]

    artifact_dir = Path(
        args.artifact_dir
    ).resolve()

    index_path = (
        artifact_dir
        / "runtime_law_index.faiss"
    )
    metadata_path = (
        artifact_dir
        / "runtime_law_metadata.json"
    )

    golden_dir = (
        project_root
        / "backend"
        / "data"
        / "runs"
        / DEFAULT_GOLDEN_RUN
    )

    golden_results_path = (
        golden_dir
        / "pipeline_results.json"
    )
    config_path = (
        golden_dir
        / "eval_config.snapshot.yaml"
    )
    benchmark_path = (
        project_root
        / "backend"
        / "data"
        / "judgments_evaluation.json"
    )

    metadata = _load_json(metadata_path)
    golden = _load_json(golden_results_path)
    benchmark = _load_json(benchmark_path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)

    golden_rows = golden["results"]

    if len(metadata) != 1811:
        raise ValueError(
            f"Expected 1811 metadata rows, "
            f"got {len(metadata)}"
        )

    if len(golden_rows) != len(benchmark):
        raise ValueError(
            "Golden result count does not match "
            "benchmark count"
        )

    documents = [
        {
            "standard_law_id": row[
                "standard_law_id"
            ],
            "bm25_text": row["bm25_text"],
            "vector_text": row["vector_text"],
        }
        for row in metadata
    ]

    bm25_index = build_bm25_index(
        documents
    )

    embedding_model = config[
        "models"
    ]["embedding_model"]
    embedding_revision = config[
        "models"
    ]["embedding_revision"]
    local_files_only = bool(
        config["models"].get(
            "local_files_only",
            True,
        )
    )

    print("=== LOADING QUERY EMBEDDER ===")
    print(f"model={embedding_model}")
    print(f"revision={embedding_revision}")
    print(
        f"local_files_only="
        f"{local_files_only}"
    )

    # Load SentenceTransformer before importing FAISS.
    # This order avoids a reproducible macOS native-library segfault.
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(
        embedding_model,
        revision=embedding_revision,
        device="cpu",
        local_files_only=local_files_only,
    )

    resolved_revision = resolved_model_revision(
        embedder
    )
    verify_model_revision(
        embedding_revision,
        resolved_revision,
    )

    import faiss

    faiss.omp_set_num_threads(1)

    faiss_index = faiss.read_index(
        str(index_path)
    )

    if type(faiss_index).__name__ != "IndexFlatIP":
        raise ValueError(
            "Unexpected FAISS type: "
            f"{type(faiss_index).__name__}"
        )

    if faiss_index.d != 768:
        raise ValueError(
            f"Expected FAISS d=768, "
            f"got {faiss_index.d}"
        )

    if faiss_index.ntotal != 1811:
        raise ValueError(
            f"Expected FAISS ntotal=1811, "
            f"got {faiss_index.ntotal}"
        )

    query_representation = config[
        "evaluation"
    ]["query_representation"]

    pool_k = int(
        config["evaluation"]["pool_k"]
    )
    stored_k = int(
        config["evaluation"]["k"]
    )

    fusion_method = config[
        "fusion"
    ]["method"]

    static_w_bm25 = float(
        config["fusion"]["static_w_bm25"]
    )

    if args.all:
        selected_indices = list(
            range(len(golden_rows))
        )
    else:
        selected_indices = list(
            DEFAULT_SAMPLE
        )

    print()
    print("=== GOLDEN RETRIEVAL PARITY ===")
    print(
        f"queries={len(selected_indices)}"
    )
    print(f"pool_k={pool_k}")
    print(f"stored_k={stored_k}")
    print(
        f"fusion={fusion_method} "
        f"w_bm25={static_w_bm25}"
    )

    failures = []

    for index in selected_indices:
        benchmark_row = benchmark[index]
        golden_row = golden_rows[index]

        query_text = build_query_text(
            benchmark_row,
            query_representation,
        )

        if query_text != golden_row["query"]:
            raise AssertionError(
                "Q0 mismatch at benchmark index "
                f"{index}: historical query "
                "does not equal current "
                "build_query_text output"
            )

        query_id = golden_row.get(
            "query_id",
            index,
        )

        print()
        print(
            f"QUERY index={index} "
            f"id={query_id}"
        )

        bm25_pool = search_bm25(
            bm25_index,
            documents,
            query_text,
            pool_k,
        )

        query_embedding = encode_dense_query(
            embedder,
            query_text,
        )

        faiss_pool = search_faiss(
            faiss_index,
            query_embedding,
            documents,
            pool_k,
        )

        hybrid_pairs = fuse_pools(
            bm25_pool,
            faiss_pool,
            static_w_bm25,
            fusion_method,
        )

        hybrid_pool = [
            {
                "rank": rank,
                "standard_law_id": standard_id,
                "score": float(score),
            }
            for rank, (standard_id, score)
            in enumerate(
                hybrid_pairs[:stored_k],
                1,
            )
        ]

        actual = {
            "BM25": bm25_pool[:stored_k],
            "FAISS": faiss_pool[:stored_k],
            "Hybrid_Static_Fixed": hybrid_pool,
        }

        expected = golden_row["metrics"]

        query_ok = True

        for name in (
            "BM25",
            "FAISS",
            "Hybrid_Static_Fixed",
        ):
            ok = _compare_ranking(
                query_id,
                name,
                expected[name]["top_k"],
                actual[name],
            )

            if not ok:
                query_ok = False
                failures.append(
                    {
                        "index": index,
                        "query_id": query_id,
                        "retriever": name,
                    }
                )

        if query_ok:
            print(
                "  PASS query retrieval parity"
            )

    print()
    print("=== PARITY SUMMARY ===")
    print(
        f"queries_checked="
        f"{len(selected_indices)}"
    )
    print(
        f"retrieval_checks="
        f"{len(selected_indices) * 3}"
    )
    print(
        f"failures={len(failures)}"
    )

    if failures:
        print()
        print("FAILURES:")
        for failure in failures:
            print(failure)

        raise SystemExit(1)

    print()
    print(
        "PASS: new runtime artifacts reproduce "
        "golden BM25/FAISS/Hybrid rankings"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify runtime retrieval artifacts "
            "against the frozen baseline_d0_v2 "
            "golden retrieval rankings."
        )
    )

    parser.add_argument(
        "--artifact-dir",
        required=True,
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check all 107 benchmark queries.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    verify(parse_args())
