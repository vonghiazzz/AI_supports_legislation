import math


def softmax_normalize(scores, temperature=1.0):
    if not scores:
        return []
    max_value = max(scores)
    exp_scores = [math.exp((score - max_value) / temperature) for score in scores]
    total = sum(exp_scores)
    if total == 0:
        return [1.0 / len(scores) for _ in scores]
    return [score / total for score in exp_scores]


def fuse_pools(
    bm25_pool,
    faiss_pool,
    w_bm25,
    fusion_type="minmax",
    temp_bm25=5.0,
    temp_faiss=0.5,
):
    w_faiss = 1.0 - w_bm25
    union_ids = list(
        set(
            [document["standard_law_id"] for document in bm25_pool]
            + [document["standard_law_id"] for document in faiss_pool]
        )
    )
    if not union_ids:
        return []
    bm25_map = {document["standard_law_id"]: document["score"] for document in bm25_pool}
    faiss_map = {document["standard_law_id"]: document["score"] for document in faiss_pool}

    if fusion_type == "minmax":
        bm25_scores = [document["score"] for document in bm25_pool]
        faiss_scores = [document["score"] for document in faiss_pool]
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        faiss_min = min(faiss_scores) if faiss_scores else 0.0
        faiss_max = max(faiss_scores) if faiss_scores else 1.0
        faiss_range = faiss_max - faiss_min if faiss_max != faiss_min else 1.0
        combined = []
        for standard_id in union_ids:
            bm25_score = (
                (bm25_map[standard_id] - bm25_min) / bm25_range
                if standard_id in bm25_map
                else 0.0
            )
            faiss_score = (
                (faiss_map[standard_id] - faiss_min) / faiss_range
                if standard_id in faiss_map
                else 0.0
            )
            combined.append((standard_id, w_bm25 * bm25_score + w_faiss * faiss_score))
    elif fusion_type == "softmax":
        bm25_min = min([document["score"] for document in bm25_pool]) if bm25_pool else 0.0
        faiss_min = min([document["score"] for document in faiss_pool]) if faiss_pool else 0.0
        bm25_scores = [bm25_map.get(standard_id, bm25_min) for standard_id in union_ids]
        faiss_scores = [faiss_map.get(standard_id, faiss_min) for standard_id in union_ids]
        bm25_normalized = softmax_normalize(bm25_scores, temperature=temp_bm25)
        faiss_normalized = softmax_normalize(faiss_scores, temperature=temp_faiss)
        combined = [
            (
                standard_id,
                w_bm25 * bm25_normalized[index] + w_faiss * faiss_normalized[index],
            )
            for index, standard_id in enumerate(union_ids)
        ]
    elif fusion_type == "rrf":
        bm25_ranks = {
            document["standard_law_id"]: rank
            for rank, document in enumerate(bm25_pool, 1)
        }
        faiss_ranks = {
            document["standard_law_id"]: rank
            for rank, document in enumerate(faiss_pool, 1)
        }
        combined = []
        for standard_id in union_ids:
            bm25_rank = bm25_ranks.get(standard_id)
            faiss_rank = faiss_ranks.get(standard_id)
            bm25_score = 1.0 / (60.0 + bm25_rank) if bm25_rank is not None else 0.0
            faiss_score = 1.0 / (60.0 + faiss_rank) if faiss_rank is not None else 0.0
            combined.append((standard_id, w_bm25 * bm25_score + w_faiss * faiss_score))
    else:
        combined = []
    combined.sort(key=lambda item: item[1], reverse=True)
    return combined
