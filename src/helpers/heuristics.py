import re


def classify_extraction_unit(text: str) -> tuple[str, float]:
    text_lower = text.lower()

    # Calculate relevance score based on keyword density
    legal_keywords = {
        "shall",
        "must",
        "payment",
        "royalty",
        "termination",
        "indemnify",
        "confidential",
        "audit",
        "notice",
        "obligation",
        "restriction",
        "liability",
        "warranty",
        "breach",
        "covenant",
        "jurisdiction",
        "governing law",
        "intellectual property",
        "license",
        "fee",
        "invoice",
        "taxes",
        "assignment",
        "waiver",
        "severability",
    }
    words = set(re.findall(r"\w+", text_lower))
    matched_keywords = legal_keywords.intersection(words)
    relevance_score = min(1.0, len(matched_keywords) / 5.0)

    is_definition = False
    if "means" in text_lower or "has the meaning" in text_lower:
        if (
            re.search(r'(?i)"[^"]+"\s+means', text)
            or re.search(r"(?i)'[^']+'\s+means", text)
            or "has the meaning set forth" in text_lower
        ):
            is_definition = True

    if is_definition:
        duty_patterns = [
            r"\bshall\b",
            r"\bmust\b",
            r"\bwill\s+not\b",
            r"\bis\s+required\s+to\b",
            r"\bis\s+prohibited\s+from\b",
            r"\bis\s+entitled\s+to\b",
            r"\bagrees?\s+to\b",
            r"\bundertakes?\s+to\b",
        ]
        if any(re.search(pat, text_lower) for pat in duty_patterns):
            return "OPERATIVE_DEFINITION", relevance_score
        else:
            return "PURE_DEFINITION", 0.0
    return "SUBSTANTIVE", relevance_score


def contains_risk_trigger_terms(text: str) -> bool:
    text_lower = text.lower()
    triggers = [
        "shall",
        "must",
        "payment",
        "royalty",
        "termination",
        "indemnify",
        "confidential",
        "audit",
        "notice",
        "obligation",
        "restriction",
    ]
    return any(t in text_lower for t in triggers)
