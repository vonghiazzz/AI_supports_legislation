from .normalization import canonicalize_law_code, extract_article_number


def build_document_text(document, representation):
    title = " ".join(str(document.get("title", "")).split())
    content = " ".join(str(document.get("content", "")).split())
    summary = " ".join(str(document.get("summary", "")).split())
    lexical = " ".join(
        str(value).strip()
        for value in document.get("lexical_keywords", [])
        if str(value).strip()
    )
    keywords = " ".join(
        str(value).strip()
        for value in document.get("keywords", [])
        if str(value).strip()
    )

    if representation == "d0":
        return f"Tiêu đề: {title}. Nội dung: {content}".strip()
    if representation == "d1":
        terms = " ".join(value for value in [lexical, keywords] if value)
        return (
            f"Tiêu đề: {title}. Nội dung: {content}. "
            f"Tóm tắt: {summary}. Từ khóa: {terms}"
        ).strip()
    if representation == "legacy_enriched_weighted":
        tags = " ".join(
            str(value).strip()
            for value in document.get("tags", [])
            if str(value).strip()
        )
        parts = [title, title, title, keywords, keywords, summary, summary, tags, lexical, content]
        return " ".join(value for value in parts if value)
    if representation == "legacy_dense":
        return " ".join(
            f"Tiêu đề: {title}. Tóm tắt: {summary}. Từ khóa: {lexical} {keywords}".split()
        )
    raise ValueError(f"Unsupported document representation: {representation}")


def create_standard_id(law_id, title, law_name):
    if law_id:
        clean_id = str(law_id).strip().lower().replace("điều", "").strip()
        article = f"điều {clean_id}"
    else:
        article = extract_article_number(title)
    return f"{article} {canonicalize_law_code(law_name)}"
