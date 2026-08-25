import re


def normalize_query_q0(query):
    """Baseline Q0: benchmark query with whitespace normalization only."""
    if query is None:
        return ""
    return " ".join(str(query).strip().split())


def clean_query_legacy(query):
    """Historical query cleaner retained only for legacy-run reproduction."""
    if not query:
        return ""
    text = str(query)
    match_vv = re.search(r"v/v:?\s*(.*)", text, re.IGNORECASE)
    cleaned = match_vv.group(1) if match_vv else text
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["giám đốc thẩm", "giám đốc thầm", "giám đốc thấm"]):
        cleaned = "giám đốc thẩm " + cleaned
    elif any(kw in text_lower for kw in ["phúc thẩm", "phúc thầm", "phúc thấm"]):
        cleaned = "phúc thẩm " + cleaned
    elif any(kw in text_lower for kw in ["sơ thẩm", "sơ thầm", "sơ thấm"]):
        cleaned = "sơ thẩm " + cleaned
    return " ".join(cleaned.split())


def build_query_text(record, representation):
    if representation == "q0":
        return normalize_query_q0(record.get("query", ""))
    if representation == "legacy":
        return clean_query_legacy(record.get("query", ""))
    raise ValueError(f"Unsupported query representation: {representation}")
