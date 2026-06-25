import re
from typing import Any

from src.models import ObligationItem


def build_obligations_from_llm(obligations_data: list[dict[str, Any]]) -> list[ObligationItem]:
    obligations: list[ObligationItem] = []
    for obligation_obj in obligations_data:
        if not isinstance(obligation_obj, dict):
            continue

        obligation_text = obligation_obj.get("obligation")
        obligation_text = str(obligation_text).strip() if obligation_text is not None else ""
        if not obligation_text:
            continue

        party_val = obligation_obj.get("party")
        party = str(party_val).strip() if party_val is not None else None

        due_val = obligation_obj.get("due_date")
        due_date = str(due_val).strip() if due_val is not None else None

        freq_val = obligation_obj.get("frequency")
        frequency = str(freq_val).strip() if freq_val is not None else None

        cond_val = obligation_obj.get("condition")
        condition = str(cond_val).strip() if cond_val is not None else None

        otype_val = obligation_obj.get("obligation_type")
        obligation_type = str(otype_val).strip() if otype_val is not None else None

        source_val = obligation_obj.get("source_clause")
        source_clause = str(source_val).strip() if source_val is not None else None

        obligations.append(
            ObligationItem(
                party=party,
                obligation=obligation_text,
                due_date=due_date,
                frequency=frequency,
                condition=condition,
                obligation_type=obligation_type,
                source_clause=source_clause,
            )
        )
    return obligations


def infer_party(text: str) -> str | None:
    match = re.match(
        r"([A-Z][A-Za-z0-9&.,/\- ]{2,80}?)\s+(shall|must|will|may not|shall not|agrees to|agrees that)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return None


def classify_obligation(text: str) -> str:
    lower = text.lower()
    if any(
        token in lower for token in ("pay", "fee", "royalt", "price", "commission", "consideration")
    ):
        return "payment"
    if any(token in lower for token in ("notice", "notify", "written notice")):
        return "notice"
    if any(
        token in lower
        for token in (
            "not",
            "may not",
            "shall not",
            "prohibit",
            "restrict",
            "exclusive",
            "non-compete",
        )
    ):
        return "restriction"
    return "general"


def infer_frequency(text: str) -> str | None:
    lower = text.lower()
    for token in ("annually", "annual", "monthly", "quarterly", "daily", "weekly", "yearly"):
        if token in lower:
            return token
    return None


def infer_condition(text: str) -> str | None:
    lower = text.lower()
    if "provided that" in lower:
        return lower.split("provided that", 1)[1].strip()[:240]
    if "if " in lower:
        idx = lower.find("if ")
        return lower[idx : idx + 240]
    return None


def find_longest_common_prefix_words(s1: str, s2: str) -> tuple[str, str, str]:
    s1_clean = " ".join(s1.split())
    s2_clean = " ".join(s2.split())

    words1 = s1_clean.split()
    words2 = s2_clean.split()

    common_words = []
    min_len = min(len(words1), len(words2))
    for i in range(min_len):
        if words1[i].lower() == words2[i].lower():
            common_words.append(words1[i])
        else:
            break

    if len(common_words) >= 2:
        prefix = " ".join(words1[: len(common_words)])
        suffix1 = " ".join(words1[len(common_words) :])
        suffix2 = " ".join(words2[len(common_words) :])
        return prefix, suffix1, suffix2
    return "", s1, s2


def merge_similar_obligations(items: list[ObligationItem]) -> list[ObligationItem]:
    if not items:
        return []

    current_list = list(items)
    merged_any = True

    while merged_any:
        merged_any = False
        new_list = []
        skip_indices = set()

        for i in range(len(current_list)):
            if i in skip_indices:
                continue

            merged_item = current_list[i]
            for j in range(i + 1, len(current_list)):
                if j in skip_indices:
                    continue

                item2 = current_list[j]

                same_party = (merged_item.party or "").strip().lower() == (
                    item2.party or ""
                ).strip().lower()
                same_type = (merged_item.obligation_type or "").strip().lower() == (
                    item2.obligation_type or ""
                ).strip().lower()
                same_due = (merged_item.due_date or "").strip().lower() == (
                    item2.due_date or ""
                ).strip().lower()
                same_freq = (merged_item.frequency or "").strip().lower() == (
                    item2.frequency or ""
                ).strip().lower()
                same_cond = (merged_item.condition or "").strip().lower() == (
                    item2.condition or ""
                ).strip().lower()

                if same_party and same_type and same_due and same_freq and same_cond:
                    prefix, suffix1, suffix2 = find_longest_common_prefix_words(
                        merged_item.obligation or "", item2.obligation or ""
                    )
                    if prefix and suffix1.strip() and suffix2.strip():
                        s1 = suffix1.rstrip(".,; ")
                        s2 = suffix2.rstrip(".,; ")

                        if s1.lower().startswith("and "):
                            s1 = s1[4:]
                        if s2.lower().startswith("and "):
                            s2 = s2[4:]

                        new_text = f"{prefix} {s1}, and {s2}."
                        new_text = " ".join(new_text.split())
                        new_text = new_text.rstrip(".") + "."

                        merged_item = ObligationItem(
                            party=merged_item.party,
                            obligation=new_text,
                            due_date=merged_item.due_date,
                            frequency=merged_item.frequency,
                            condition=merged_item.condition,
                            obligation_type=merged_item.obligation_type,
                            source_clause=f"{merged_item.source_clause or ''}; {item2.source_clause or ''}".strip(
                                "; "
                            ),
                        )
                        skip_indices.add(j)
                        merged_any = True

            new_list.append(merged_item)
        current_list = new_list
        if not merged_any:
            break

    return current_list
