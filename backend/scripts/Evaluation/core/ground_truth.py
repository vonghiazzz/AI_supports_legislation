from .normalization import canonical_retrieval_id, canonicalize_law_code, normalize_text


def extract_gt_sub_proc(gt_obj):
    if not gt_obj:
        return [], []
    core_obj = gt_obj.get("core", [])
    supporting_obj = gt_obj.get("supporting", [])
    if isinstance(core_obj, list) and isinstance(supporting_obj, list):
        substantive = []
        procedural = []
        for item in core_obj + supporting_obj:
            domain = item.get("domain")
            if domain == "procedural":
                procedural.append(item)
            else:
                substantive.append(item)
        return substantive, procedural
    core_dict = core_obj if isinstance(core_obj, dict) else {}
    supporting_dict = supporting_obj if isinstance(supporting_obj, dict) else {}
    substantive = core_dict.get("substantive", []) + supporting_dict.get("substantive", [])
    procedural = core_dict.get("procedural", []) + supporting_dict.get("procedural", [])
    return substantive, procedural


def get_gt_weights(gt_obj, alpha_core=3, alpha_support=1):
    weights = {}
    if not gt_obj:
        return weights
    core_obj = gt_obj.get("core", [])
    supporting_obj = gt_obj.get("supporting", [])
    if isinstance(core_obj, list) and isinstance(supporting_obj, list):
        core_items = core_obj
        supporting_items = supporting_obj
    else:
        core_dict = core_obj if isinstance(core_obj, dict) else {}
        supporting_dict = supporting_obj if isinstance(supporting_obj, dict) else {}
        core_items = core_dict.get("substantive", []) + core_dict.get("procedural", [])
        supporting_items = (
            supporting_dict.get("substantive", []) + supporting_dict.get("procedural", [])
        )

    for item in core_items:
        standard_id = (
            f"{item.get('article', '')} "
            f"{canonicalize_law_code(item.get('law_code', ''))}"
        ).strip()
        if standard_id:
            weights[normalize_text(standard_id)] = alpha_core
    for item in supporting_items:
        standard_id = (
            f"{item.get('article', '')} "
            f"{canonicalize_law_code(item.get('law_code', ''))}"
        ).strip()
        normalized = normalize_text(standard_id)
        if standard_id and normalized not in weights:
            weights[normalized] = alpha_support
    return weights


def build_canonical_ground_truth(
    record,
    corpus_document_ids,
    alpha_core=3,
    alpha_support=1,
):
    gt_obj = record.get("ground_truth", {})
    core = list(gt_obj.get("core", []))
    supporting = list(gt_obj.get("supporting", []))
    corpus_id_keys = {canonical_retrieval_id(value) for value in corpus_document_ids}

    def standard_id(item):
        return (
            f"{item.get('article', '')} "
            f"{canonicalize_law_code(item.get('law_code', ''))}"
        ).strip()

    def in_closed_world(item):
        return canonical_retrieval_id(standard_id(item)) in corpus_id_keys

    all_items = core + supporting
    return {
        "core_gt_raw": [standard_id(item) for item in core],
        "supporting_gt_raw": [standard_id(item) for item in supporting],
        "core_gt_filtered": [standard_id(item) for item in core if in_closed_world(item)],
        "supporting_gt_filtered": [
            standard_id(item) for item in supporting if in_closed_world(item)
        ],
        "substance_gt_raw": [
            standard_id(item) for item in all_items if item.get("domain") == "substantive"
        ],
        "procedural_gt_raw": [
            standard_id(item) for item in all_items if item.get("domain") == "procedural"
        ],
        "substance_gt_filtered": [
            standard_id(item)
            for item in all_items
            if item.get("domain") == "substantive" and in_closed_world(item)
        ],
        "procedural_gt_filtered": [
            standard_id(item)
            for item in all_items
            if item.get("domain") == "procedural" and in_closed_world(item)
        ],
        "explicit_gts": [
            standard_id(item)
            for item in all_items
            if item.get("source_type") == "explicit" and in_closed_world(item)
        ],
        "inferred_gts": [
            standard_id(item)
            for item in all_items
            if item.get("source_type") != "explicit" and in_closed_world(item)
        ],
        "gt_weights": get_gt_weights(gt_obj, alpha_core, alpha_support),
    }
