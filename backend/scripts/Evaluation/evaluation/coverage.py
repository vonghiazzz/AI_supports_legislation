from ..core.normalization import canonical_retrieval_id, canonicalize_law_code


def compute_ground_truth_coverage(benchmark, corpus_document_ids):
    total_labels = 0
    in_corpus_labels = 0
    queries_with_in_corpus = 0
    corpus_id_keys = {canonical_retrieval_id(value) for value in corpus_document_ids}

    for record in benchmark:
        gt_obj = record.get("ground_truth", {})
        labels = list(gt_obj.get("core", [])) + list(gt_obj.get("supporting", []))
        query_in_corpus = 0
        for item in labels:
            canonical_id = canonical_retrieval_id(
                f"{item.get('article', '')} "
                f"{canonicalize_law_code(item.get('law_code', ''))}".strip()
            )
            total_labels += 1
            if canonical_id in corpus_id_keys:
                in_corpus_labels += 1
                query_in_corpus += 1
        if query_in_corpus:
            queries_with_in_corpus += 1

    return {
        "total_labels": total_labels,
        "in_corpus_labels": in_corpus_labels,
        "out_of_corpus_labels": total_labels - in_corpus_labels,
        "coverage_ratio": in_corpus_labels / total_labels if total_labels else 0.0,
        "queries_with_at_least_one_in_corpus_gt": queries_with_in_corpus,
        "queries_without_in_corpus_gt": len(benchmark) - queries_with_in_corpus,
    }
