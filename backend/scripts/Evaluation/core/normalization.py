import re


LAW_CODE_ALIASES = {
    "luat_hon_nhan_va_gia_dinh_2014": "luat_hon_nhan_gia_dinh_2014",
}


def normalize_text(text):
    if not text:
        return ""
    return " ".join(str(text).lower().split())


def canonicalize_law_code(law_code):
    normalized = normalize_text(law_code).replace(" ", "_")
    return LAW_CODE_ALIASES.get(normalized, normalized)


def extract_article_number(text):
    match = re.search(r"(điều\s+\d+[a-z]?)", str(text), re.IGNORECASE)
    return match.group(1).lower() if match else normalize_text(text)


def is_match(retrieved_id, ground_truth):
    ret_norm = normalize_text(retrieved_id)
    gt_norm = normalize_text(ground_truth)
    ret_article = extract_article_number(ret_norm)
    gt_article = extract_article_number(gt_norm)
    ret_parts = ret_norm.split()
    gt_parts = gt_norm.split()
    if len(ret_parts) < 2 or len(gt_parts) < 2:
        return False
    ret_law = canonicalize_law_code(ret_parts[-1])
    gt_law = canonicalize_law_code(gt_parts[-1])
    if ret_article and gt_article and ret_law and gt_law:
        return ret_article == gt_article and ret_law == gt_law
    return False


def canonical_retrieval_id(value):
    normalized = normalize_text(value)
    parts = normalized.split()
    if len(parts) < 2:
        return normalized
    article = extract_article_number(normalized)
    law_code = canonicalize_law_code(parts[-1])
    return normalize_text(f"{article} {law_code}")
