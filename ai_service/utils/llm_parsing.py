"""Centralized helper functions for parsing and processing LLM outputs."""

import json
import logging
import re
from typing import Any

from app import config

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences (e.g. ```json) from LLM responses."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = [line for line in lines[1:] if line.strip() != "```"]
        return "\n".join(inner).strip()
    return stripped


def parse_llm_json_response(response_text: str) -> dict[str, Any] | list[Any] | None:
    """Resiliently parse JSON from an LLM response."""
    if not response_text:
        return None

    text = strip_markdown_fences(response_text)

    # Try direct load
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Resilient boundary extraction
    first_obj = text.find("{")
    last_obj = text.rfind("}")
    first_list = text.find("[")
    last_list = text.rfind("]")

    # Try list first if it starts before object
    if first_list != -1 and last_list != -1 and (first_obj == -1 or first_list < first_obj):
        try:
            return json.loads(text[first_list : last_list + 1])
        except json.JSONDecodeError:
            pass

    if first_obj != -1 and last_obj != -1:
        try:
            return json.loads(text[first_obj : last_obj + 1])
        except json.JSONDecodeError:
            pass

    if first_list != -1 and last_list != -1:
        try:
            return json.loads(text[first_list : last_list + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_markdown_response(text: str) -> dict[str, Any] | None:
    """Parse Markdown output into the clause/metadata dict structure using permissive regex."""
    if not text:
        return None

    metadata = {}
    clauses = []

    # Strip out markdown code blocks if the LLM wraps the response in ```markdown
    text = re.sub(r"^```markdown\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    # 1. Parse Metadata
    meta_match = re.search(r"##\s*Metadata(.*?)##\s*Clauses", text, re.IGNORECASE | re.DOTALL)
    if meta_match:
        meta_text = meta_match.group(1)
        for line in meta_text.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].replace("-", "").strip().lower()
                    val = parts[1].strip()
                    if val.lower() not in ("null", "[string | null]", ""):
                        if "document name" in key:
                            metadata["document_name"] = val
                        elif "contract type" in key:
                            metadata["contract_type"] = val
                        elif "agreement date" in key:
                            metadata["agreement_date"] = val
                        elif "effective date" in key:
                            metadata["effective_date"] = val
                        elif "expiration date" in key:
                            metadata["expiration_date"] = val
                        elif "renewal term" in key:
                            metadata["renewal_term"] = val
                        elif "notice period" in key:
                            metadata["notice_period_to_terminate_renewal"] = val
                        elif "governing law" in key:
                            metadata["governing_law"] = val

    # 2. Parse Clauses
    clauses_section = text
    clauses_match = re.search(r"##\s*Clauses(.*)", text, re.IGNORECASE | re.DOTALL)
    if clauses_match:
        clauses_section = clauses_match.group(1)

    def parse_clause_body(body: str, ctype: str) -> dict[str, Any]:
        cat_match = re.search(r"-\s*\*\*Category:\*\*\s*(.*?)(?=\n|$)", body, re.IGNORECASE)
        ref_match = re.search(r"-\s*\*\*Reference:\*\*\s*(.*?)(?=\n|$)", body, re.IGNORECASE)
        conf_match = re.search(r"-\s*\*\*Confidence:\*\*\s*(.*?)(?=\n|$)", body, re.IGNORECASE)
        text_match = re.search(
            r"-\s*\*\*Text:\*\*\s*\n*(.*?)(?=\n-\s*\*\*(?:Category|Reference|Confidence|Text):\*\*|$)",
            body,
            re.IGNORECASE | re.DOTALL,
        )

        category = cat_match.group(1).strip() if cat_match else None
        reference = ref_match.group(1).strip() if ref_match else None
        conf_str = conf_match.group(1).strip() if conf_match else "0.5"
        raw_text = text_match.group(1).strip() if text_match else ""

        if category and category.lower() in ("null", "[cuad_category]"):
            category = None
        if reference and reference.lower() in ("null", "[section_reference]"):
            reference = None

        conf = 0.5
        try:
            conf = float(conf_str)
        except ValueError:
            pass

        return {
            "clause_type": ctype,
            "cuad_category": category,
            "section_reference": reference,
            "confidence": conf,
            "raw_text": raw_text,
            "subclauses": [],
        }

    # Split by ### [Clause Type]
    clause_blocks = re.split(r"(?=\n###\s+)", "\n" + clauses_section)
    for block in clause_blocks:
        block = block.strip()
        if not block.startswith("### "):
            continue

        lines = block.split("\n", 1)
        c_type = lines[0].replace("###", "").strip()
        c_body = lines[1] if len(lines) > 1 else ""

        sub_blocks = re.split(r"(?=\n####\s+)", "\n" + c_body)
        primary_body = sub_blocks[0].strip() if sub_blocks else ""

        primary_clause = parse_clause_body(primary_body, c_type)

        for sub in sub_blocks[1:]:
            sub = sub.strip()
            if not sub:
                continue
            s_lines = sub.split("\n", 1)
            s_type = s_lines[0].replace("#### Subclause:", "").replace("####", "").strip()
            s_body = s_lines[1] if len(s_lines) > 1 else ""

            sub_clause = parse_clause_body(s_body, s_type)
            primary_clause["subclauses"].append(sub_clause)

        clauses.append(primary_clause)

    if not clauses and not metadata:
        return None

    return {"metadata": metadata, "clauses": clauses}


def parse_llm_response(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response by trying Markdown first, falling back to JSON."""
    parsed = parse_markdown_response(response_text)
    if parsed is not None:
        return parsed

    logger.warning(
        "Markdown parsing failed or yielded no clauses/metadata, falling back to JSON parser"
    )
    return parse_json_fallback(response_text)


def parse_json_fallback(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response JSON, with truncation recovery fallback."""
    if not response_text:
        return None
    text = response_text.strip()

    # 1. Attempt standard JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Attempt substring parsing between first { and last }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first : last + 1])
        except json.JSONDecodeError:
            pass

    # 3. Fallback: resilient recovery of truncated/broken JSON
    try:
        clauses = []
        open_indices = [m.start() for m in re.finditer(r"\{", text)]
        close_indices = [m.start() for m in re.finditer(r"\}", text)]

        # Extract valid clauses
        for start in open_indices:
            for end in close_indices:
                if end > start:
                    if end - start > config.CLAUSE_EXTRACTION_MAX_TOKENS_LIMIT:
                        break  # Optimize search space by stopping if length exceeds limit
                    candidate = text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "clause_type" in obj and "raw_text" in obj:
                            clauses.append(obj)
                            break
                    except Exception:
                        pass

        # Additionally, scan the last portion of the response string specifically for truncated blocks
        last_portion_start = int(len(text) * config.TRUNCATION_SCAN_FACTOR)
        for start in [idx for idx in open_indices if idx >= last_portion_start]:
            for end in [idx for idx in close_indices if idx >= last_portion_start]:
                if end > start:
                    if end - start > config.CLAUSE_EXTRACTION_MAX_TOKENS_LIMIT:
                        break
                    candidate = text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "clause_type" in obj and "raw_text" in obj:
                            if obj not in clauses:
                                clauses.append(obj)
                            break
                    except Exception:
                        pass

        # Filter out nested clauses
        unique_clauses = []
        for c in clauses:
            is_nested = False
            for other in clauses:
                if (
                    other is not c
                    and other.get("raw_text")
                    and c.get("raw_text")
                    and c.get("raw_text") in other.get("raw_text")
                ):
                    if len(other.get("raw_text", "")) > len(c.get("raw_text", "")):
                        is_nested = True
                        break
            if not is_nested and c not in unique_clauses:
                unique_clauses.append(c)

        # Extract metadata
        metadata = None
        for start in open_indices:
            for end in close_indices:
                if end > start:
                    if end - start > 4000:
                        break
                    candidate = text[start : end + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict) and "parties" in obj:
                            metadata = obj
                            break
                    except Exception:
                        pass
            if metadata:
                break

        if unique_clauses or metadata:
            return {"clauses": unique_clauses, "metadata": metadata or {}}
    except Exception:
        pass

    return None
