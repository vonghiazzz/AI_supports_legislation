def build_dense_retriever(
    documents,
    embedding_model,
    embedding_revision,
    local_files_only=True,
    device="cpu",
):
    import faiss
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(
        embedding_model,
        revision=embedding_revision,
        device=device,
        local_files_only=local_files_only,
    )
    corpus_embeddings = embedder.encode(
        [document["vector_text"] for document in documents],
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")
    faiss.normalize_L2(corpus_embeddings)
    embedding_dimension = int(corpus_embeddings.shape[1])
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(corpus_embeddings)
    return embedder, index, embedding_dimension


def encode_dense_query(embedder, query_text):
    import faiss

    query_embedding = embedder.encode(
        [query_text], show_progress_bar=False, convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(query_embedding)
    return query_embedding


def search_faiss(index, query_embedding, documents, pool_k):
    scores, indices = index.search(query_embedding, pool_k)
    return [
        {
            "rank": rank,
            "standard_law_id": documents[int(document_index)]["standard_law_id"],
            "score": float(score),
        }
        for rank, (score, document_index) in enumerate(zip(scores[0], indices[0]), 1)
        if document_index >= 0
    ]
