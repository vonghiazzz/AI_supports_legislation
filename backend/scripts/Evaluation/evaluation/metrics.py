import math

from ..core.normalization import is_match, normalize_text


BASELINE_REPORTED_K = (5, 10, 20)


def map_law_code_to_family(law_code):
    law_code = str(law_code).lower().strip()
    if "bo_luat_dan_su" in law_code or "luat_dan_su" in law_code or "bds" in law_code:
        return "Civil Code"
    if "to_tung_dan_su" in law_code or "ttds" in law_code:
        return "Civil Procedure Code"
    if (
        "dat_dai" in law_code
        or "dd" in law_code
        or "nghi_dinh_43" in law_code
        or "nghi_dinh_01_2017" in law_code
    ):
        return "Land Law"
    if "hon_nhan" in law_code or "gia_dinh" in law_code:
        return "Marriage & Family Law"
    if "nghi_dinh_163" in law_code or "nghi_dinh_17" in law_code:
        return "Civil Code"
    return "Others"


def calculate_ndcg(ret_ids, gts, k, gt_weights=None):
    top_k_ids = ret_ids[:k]
    relevance = []
    for retrieved_id in top_k_ids:
        matched_gts = [gt for gt in gts if is_match(retrieved_id, gt)]
        if matched_gts:
            if gt_weights:
                max_weight = 0
                for gt in matched_gts:
                    max_weight = max(max_weight, gt_weights.get(normalize_text(gt), 1))
                relevance.append(max_weight)
            else:
                relevance.append(1)
        else:
            relevance.append(0)

    dcg = sum(
        (2**relevance_value - 1) / math.log2(index + 1)
        for index, relevance_value in enumerate(relevance, 1)
    )
    if gt_weights:
        target_weights = [gt_weights.get(normalize_text(gt), 1) for gt in gts]
    else:
        target_weights = [1] * len(gts)
    ideal_relevance = sorted(target_weights, reverse=True)[:k]
    ideal_relevance += [0] * (k - len(ideal_relevance))
    idcg = sum(
        (2**relevance_value - 1) / math.log2(index + 1)
        for index, relevance_value in enumerate(ideal_relevance, 1)
    )
    return dcg / idcg if idcg else 0.0


def get_task_metrics(
    results,
    configuration,
    k,
    is_filtered=True,
    task_type="overall",
    filter_profile=None,
    filter_difficulty=None,
    filter_law_family=None,
):
    recall_sum = 0.0
    ndcg_sum = 0.0
    mrr_sum = 0.0
    valid_queries = 0

    for item in results:
        query_profile = item.get("query_profile")
        query_profile_value = (
            query_profile.get("profile") if isinstance(query_profile, dict) else query_profile
        )
        if filter_profile and query_profile_value != filter_profile:
            continue
        difficulty = item.get("difficulty")
        difficulty_value = difficulty.get("level") if isinstance(difficulty, dict) else difficulty
        if filter_difficulty and difficulty_value != filter_difficulty:
            continue

        suffix = "filtered" if is_filtered else "raw"
        substantive = item.get(f"substance_gt_{suffix}", [])
        procedural = item.get(f"procedural_gt_{suffix}", [])
        core = item.get(f"core_gt_{suffix}", [])
        supporting = item.get(f"supporting_gt_{suffix}", [])
        explicit = item.get("explicit_gts", [])
        inferred = item.get("inferred_gts", [])

        if task_type == "substantive":
            ground_truths = substantive
        elif task_type == "procedural":
            ground_truths = procedural
        elif task_type == "explicit":
            ground_truths = explicit
        elif task_type == "inferred":
            ground_truths = inferred
        elif task_type == "core":
            ground_truths = core
        elif task_type == "supporting":
            ground_truths = supporting
        else:
            ground_truths = substantive + procedural

        if filter_law_family:
            ground_truths = [
                gt
                for gt in ground_truths
                if map_law_code_to_family(gt.split()[-1]) == filter_law_family
            ]
        if not ground_truths:
            continue
        run_data = item["metrics"].get(configuration)
        if not run_data:
            continue
        pool_ids = [document["standard_law_id"] for document in run_data["top_k"]]
        top_k_ids = pool_ids[:k]
        hits = sum(
            1
            for gt in ground_truths
            if any(is_match(retrieved_id, gt) for retrieved_id in top_k_ids)
        )
        recall_sum += hits / len(ground_truths)
        ndcg_sum += calculate_ndcg(
            top_k_ids,
            ground_truths,
            k,
            gt_weights=item.get("gt_weights"),
        )
        reciprocal_rank = 0.0
        for rank, retrieved_id in enumerate(top_k_ids, 1):
            if any(is_match(retrieved_id, gt) for gt in ground_truths):
                reciprocal_rank = 1.0 / rank
                break
        mrr_sum += reciprocal_rank
        valid_queries += 1

    if valid_queries == 0:
        return 0.0, 0.0, 0.0, 0
    return (
        recall_sum / valid_queries,
        ndcg_sum / valid_queries,
        mrr_sum / valid_queries,
        valid_queries,
    )


def get_weighted_recall(
    results,
    configuration,
    k,
    is_filtered=True,
    alpha_core=3,
    alpha_support=1,
):
    weighted_recall_sum = 0.0
    valid_queries = 0
    suffix = "filtered" if is_filtered else "raw"
    for item in results:
        core = item.get(f"core_gt_{suffix}", [])
        supporting = item.get(f"supporting_gt_{suffix}", [])
        if not core and not supporting:
            continue
        run_data = item["metrics"].get(configuration)
        if not run_data:
            continue
        top_k_ids = [
            document["standard_law_id"] for document in run_data["top_k"][:k]
        ]
        core_hits = sum(
            1 for gt in core if any(is_match(retrieved_id, gt) for retrieved_id in top_k_ids)
        )
        supporting_hits = sum(
            1
            for gt in supporting
            if any(is_match(retrieved_id, gt) for retrieved_id in top_k_ids)
        )
        denominator = alpha_core * len(core) + alpha_support * len(supporting)
        if denominator > 0:
            weighted_recall_sum += (
                alpha_core * core_hits + alpha_support * supporting_hits
            ) / denominator
            valid_queries += 1
    return weighted_recall_sum / valid_queries if valid_queries else 0.0


def build_metric_world_report(
    results,
    configurations,
    reported_k=BASELINE_REPORTED_K,
    alpha_core=3,
    alpha_support=1,
):
    worlds = {}
    for world_name, is_filtered in (("closed_world", True), ("open_world", False)):
        method_metrics = {}
        for configuration in configurations:
            metrics_at_k = {}
            for k in reported_k:
                recall, ndcg, mrr, valid = get_task_metrics(
                    results,
                    configuration,
                    k=int(k),
                    is_filtered=is_filtered,
                    task_type="overall",
                )
                metrics_at_k[str(k)] = {
                    "recall": recall,
                    "ndcg": ndcg,
                    "mrr": mrr,
                    "weighted_recall": get_weighted_recall(
                        results,
                        configuration,
                        k=int(k),
                        is_filtered=is_filtered,
                        alpha_core=alpha_core,
                        alpha_support=alpha_support,
                    ),
                    "valid_queries": valid,
                }
            method_metrics[configuration] = metrics_at_k
        worlds[world_name] = method_metrics
    return worlds
