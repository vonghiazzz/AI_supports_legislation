import numpy as np


def aggregate_latency_ms(values):
    values = [float(value) for value in values]
    if not values:
        return {"mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def build_latency_aggregates(results):
    rows = [item["latency_ms"] for item in results]
    bm25 = aggregate_latency_ms([row["bm25_retrieval_ms"] for row in rows])
    dense = aggregate_latency_ms([row["dense_total_ms"] for row in rows])
    hybrid_serial = aggregate_latency_ms([row["hybrid_serial_ms"] for row in rows])
    hybrid_parallel = aggregate_latency_ms(
        [row["hybrid_parallel_estimate_ms"] for row in rows]
    )
    return {
        "BM25": bm25,
        "FAISS": {
            **dense,
            "components": {
                "dense_encode": aggregate_latency_ms(
                    [row["dense_encode_ms"] for row in rows]
                ),
                "faiss_search": aggregate_latency_ms(
                    [row["faiss_search_ms"] for row in rows]
                ),
            },
        },
        "Hybrid_Static_Fixed": {
            **hybrid_serial,
            "measurement": "serial_wall_clock_components",
            "fusion": aggregate_latency_ms([row["fusion_ms"] for row in rows]),
            "parallel_estimate": {
                **hybrid_parallel,
                "estimated": True,
                "formula": "max(bm25_retrieval_ms, dense_total_ms) + fusion_ms",
            },
        },
    }
