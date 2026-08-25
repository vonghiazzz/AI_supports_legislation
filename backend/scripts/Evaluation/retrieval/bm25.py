def build_bm25_index(documents):
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [document["bm25_text"].lower().split() for document in documents]
    return BM25Okapi(tokenized_corpus)


def search_bm25(index, documents, query_text, pool_k):
    scores = index.get_scores(query_text.lower().split())
    indices = sorted(
        range(len(documents)),
        key=lambda document_index: (-float(scores[document_index]), document_index),
    )[:pool_k]
    return [
        {
            "rank": rank,
            "standard_law_id": documents[document_index]["standard_law_id"],
            "score": float(scores[document_index]),
        }
        for rank, document_index in enumerate(indices, 1)
    ]
