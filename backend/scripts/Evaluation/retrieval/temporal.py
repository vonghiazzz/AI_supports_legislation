SUPPORTED_TEMPORAL_POLICIES = {"none", "judgment_year", "active_range"}


def should_include_document(
    document,
    temporal_policy,
    judgment_year=None,
    target_year=None,
):
    if temporal_policy not in SUPPORTED_TEMPORAL_POLICIES:
        raise ValueError(f"Unsupported temporal policy: {temporal_policy}")
    if temporal_policy == "none":
        return True
    if temporal_policy == "judgment_year":
        if judgment_year is None or document.get("law_year") is None:
            return True
        return int(document["law_year"]) <= int(judgment_year)
    effective_year = target_year if target_year is not None else judgment_year
    if effective_year is None:
        return True
    return int(document.get("active_start", 1900)) <= int(effective_year) <= int(
        document.get("active_end", 2100)
    )
