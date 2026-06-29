import hashlib
import re
import uuid
import math
from typing import Any

from app import config
from ai_service.utils.contract_analysis import normalize_whitespace
from app.utils.text_utils import get_precise_token_count


def _split_by_pages(text: str) -> list[tuple[int, str]]:
    """Split contract text into individual pages based on --- PAGE {idx} --- markers."""
    page_pattern = re.compile(r"---\s*PAGE\s*(\d+)\s*---", re.IGNORECASE)
    parts = page_pattern.split(text)
    if len(parts) <= 1:
        return [(1, text)]

    pages = []
    initial_text = parts[0].strip()

    for i in range(1, len(parts), 2):
        page_num = int(parts[i])
        page_content = parts[i + 1]
        if i == 1 and initial_text:
            page_content = initial_text.strip() + "\n\n" + page_content.strip()
        pages.append((page_num, page_content.strip()))
    return pages


def _token_aware_chunk_plan(
    pages: list[tuple[int, str]], target_chunk_tokens: int = 2000
) -> list[str]:
    """Plan chunks based on page token count, maintaining a 1-page backward-only overlap."""
    chunks = []
    chunk_groups = []
    current_group = []
    current_tokens = 0

    for page_num, page_text in pages:
        page_tokens = get_precise_token_count(page_text)

        if current_tokens + page_tokens > target_chunk_tokens and current_group:
            chunk_groups.append(current_group)
            current_group = []
            current_tokens = 0

        current_group.append((page_num, page_text))
        current_tokens += page_tokens

    if current_group:
        chunk_groups.append(current_group)

    for idx, group in enumerate(chunk_groups):
        chunk_pages = []
        if idx > 0 and chunk_groups[idx - 1]:
            overlap_page_num, overlap_page_text = chunk_groups[idx - 1][-1]
            chunk_pages.append(
                f"--- PAGE {overlap_page_num} (CONTEXT OVERLAP) ---\n{overlap_page_text}"
            )

        for page_num, page_text in group:
            chunk_pages.append(f"--- PAGE {page_num} ---\n{page_text}")

        chunks.append("\n\n".join(chunk_pages))

    return chunks


def _split_by_sections(text: str) -> list[str]:
    """Split contract text into logical sections based on headings."""
    heading_pattern = re.compile(
        r"(?:\n|^)"
        r"(?:"
        r"\s*(?:ARTICLE|SECTION|SECT|CLAUSE|EXHIBIT|SCHEDULE|APPENDIX)"
        r"\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)[\.\:\-\s].*"
        r"|"
        r"\s*\d+(?:\.\d+){0,2}\.?\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r"|"
        r"\s*[IVXLCDM]+\.\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r"|"
        r"\s*\([a-z]\)\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r")",
        re.MULTILINE,
    )
    matches = list(heading_pattern.finditer(text))
    if not matches:
        return [text]

    sections = []
    prev_idx = 0
    for match in matches:
        start_idx = match.start()
        if start_idx > prev_idx:
            sections.append(text[prev_idx:start_idx])
        prev_idx = start_idx

    sections.append(text[prev_idx:])
    return [s.strip() for s in sections if s.strip()]


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;:])\s+(?=[A-Z(0-9])")


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences/clausal units on legal punctuation boundaries.

    Dependency-free; tuned for contract prose (splits on . ! ? ; : followed by a
    capital, digit, or opening paren). Used to break a single oversized paragraph
    without cutting mid-sentence.
    """
    parts = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    return parts or [text]


def _pack_units(units: list[str], max_tokens: int, joiner: str) -> list[str]:
    """Greedily pack text units into chunks under ``max_tokens``, preserving order."""
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for u in units:
        u_tokens = get_precise_token_count(u)
        if current_tokens + u_tokens > max_tokens and current:
            chunks.append(joiner.join(current))
            current = [u]
            current_tokens = u_tokens
        else:
            current.append(u)
            current_tokens += u_tokens
    if current:
        chunks.append(joiner.join(current))
    return chunks


def _get_tail_overlap(text: str, target_tokens: int) -> str:
    """Helper to extract up to target_tokens from the tail of text."""
    if target_tokens <= 0:
        return ""
    words = text.split()
    selected_words = []
    selected_tokens = 0
    for w in reversed(words):
        w_tokens = get_precise_token_count(w) + 1
        if selected_tokens + w_tokens > target_tokens:
            break
        selected_words.append(w)
        selected_tokens += w_tokens
    if not selected_words:
        return ""
    return " ".join(reversed(selected_words))


def _extract_protected_blocks(text: str) -> tuple[str, dict[str, str]]:
    """Scan text for markdown tables and fenced code blocks, replacing them with placeholders."""
    protected = {}
    
    # 1. Code Blocks (``` or ~~~)
    code_pattern = re.compile(r"(((?:^|\n)(?:```|~~~)[a-zA-Z]*\n[\s\S]*?\n(?:```|~~~)))")
    def replace_code(match):
        block = match.group(1)
        placeholder = f"[PROTECTED_BLOCK_{uuid.uuid4()}]"
        protected[placeholder] = block
        return f"\n{placeholder}\n"
    
    sanitized = code_pattern.sub(replace_code, text)

    # 2. Markdown Tables
    lines = sanitized.split("\n")
    new_lines = []
    table_lines = []
    for line in lines:
        if line.strip().startswith("|"):
            table_lines.append(line)
        else:
            if table_lines:
                table_block = "\n".join(table_lines)
                placeholder = f"[PROTECTED_BLOCK_{uuid.uuid4()}]"
                protected[placeholder] = table_block
                new_lines.append(placeholder)
                table_lines = []
            new_lines.append(line)
    if table_lines:
        table_block = "\n".join(table_lines)
        placeholder = f"[PROTECTED_BLOCK_{uuid.uuid4()}]"
        protected[placeholder] = table_block
        new_lines.append(placeholder)

    return "\n".join(new_lines), protected


def _restore_protected_blocks(text: str, protected: dict[str, str]) -> str:
    """Restore placeholders back to original protected blocks."""
    restored = text
    for placeholder, original in protected.items():
        restored = restored.replace(placeholder, original)
    return restored


def split_oversized_text(text: str, path: str, max_tokens: int | None = None) -> list[dict]:
    if max_tokens is None:
        max_tokens = config.CLAUSE_EXTRACTOR_CHUNK_SIZE
    est_tokens = get_precise_token_count(text)
    if est_tokens <= max_tokens:
        result = [{"text": text, "path": path}]
    else:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if config.ENABLE_SEMANTIC_SPLIT:
            units: list[str] = []
            for p in paragraphs:
                if get_precise_token_count(p) > max_tokens:
                    units.extend(_split_sentences(p))
                else:
                    units.append(p)
            chunks = _pack_units(units, max_tokens, joiner="\n\n")
        else:
            chunks = _pack_units(paragraphs, max_tokens, joiner="\n\n")

        result = []
        for idx, chunk in enumerate(chunks, 1):
            result.append({"text": chunk, "path": f"{path} (Part {idx})"})

    # Hard ceiling guard
    max_tokens_cap = getattr(config, "CLAUSE_EXTRACTOR_MAX_TOKENS", 4000)
    final_result = []
    for item in result:
        chunk_text = item["text"]
        chunk_path = item["path"]
        tokens = get_precise_token_count(chunk_text)
        if tokens > max_tokens_cap:
            num_splits = math.ceil(tokens / max_tokens_cap)
            char_step = math.floor(len(chunk_text) / num_splits)
            for s_idx in range(num_splits):
                start = s_idx * char_step
                end = (s_idx + 1) * char_step if s_idx + 1 < num_splits else len(chunk_text)
                sub_text = chunk_text[start:end].strip()
                if sub_text:
                    final_result.append({
                        "text": sub_text,
                        "path": f"{chunk_path} (Forced Part {s_idx + 1})"
                    })
        else:
            final_result.append(item)
    return final_result


def _build_raw_units(text: str, matches: list[Any]) -> list[dict]:
    raw_units = []
    if not matches:
        raw_units.append({"section_title": "Preamble", "section_path": "Preamble", "text": text})
    else:
        first_start = matches[0].start()
        if first_start > 0:
            preamble_text = text[:first_start].strip()
            if preamble_text:
                raw_units.append(
                    {"section_title": "Preamble", "section_path": "Preamble", "text": preamble_text}
                )

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            sec_text = text[start:end].strip()
            heading_text = match.group(0).strip()
            raw_units.append(
                {"section_title": heading_text, "section_path": heading_text, "text": sec_text}
            )
    return raw_units


def _resolve_section_paths(raw_units: list[dict]) -> None:
    current_parent = "Preamble"
    current_sub = ""

    for u in raw_units:
        title = u["section_title"]
        if title == "Preamble":
            u["section_path"] = "Preamble"
            continue

        is_parent = False
        if any(w in title.upper() for w in ["ARTICLE", "EXHIBIT", "SCHEDULE", "APPENDIX"]):
            is_parent = True
        elif re.match(r"^\d+\.\s+[A-Z]", title):
            is_parent = True

        if is_parent:
            current_parent = title
            current_sub = ""
            u["section_path"] = title
        else:
            current_sub = title
            u["section_path"] = f"{current_parent} > {current_sub}"


def _group_units(processed_raw_units: list[dict]) -> list[dict]:
    final_units = []
    current_group = []
    current_group_tokens = 0
    current_group_parent = None

    for u in processed_raw_units:
        parent = (
            u["section_path"].split(" > ")[0] if " > " in u["section_path"] else u["section_title"]
        )
        u_tokens = get_precise_token_count(u["text"])

        if (current_group_parent is not None and parent != current_group_parent) or (
            current_group_tokens + u_tokens > config.CLAUSE_EXTRACTOR_CHUNK_SIZE and current_group
        ):
            combined_text = "\n\n".join(item["text"] for item in current_group)
            combined_path = " & ".join(item["section_path"] for item in current_group)
            
            # Apply backward overlap if final_units exists
            if final_units and getattr(config, "CLAUSE_EXTRACTOR_CHUNK_OVERLAP", 0) > 0:
                overlap_text = _get_tail_overlap(final_units[-1]["text"], config.CLAUSE_EXTRACTOR_CHUNK_OVERLAP)
                if overlap_text:
                    combined_text = f"{overlap_text}\n\n[CONTEXT OVERLAP]\n\n" + combined_text
            
            for split_chunk in split_oversized_text(
                combined_text,
                combined_path,
                max_tokens=config.CLAUSE_EXTRACTOR_OVERSIZED_SPLIT_TOKENS,
            ):
                final_units.append(split_chunk)
            current_group = [u]
            current_group_tokens = u_tokens
            current_group_parent = parent
        else:
            current_group.append(u)
            current_group_tokens += u_tokens
            current_group_parent = parent

    if current_group:
        combined_text = "\n\n".join(item["text"] for item in current_group)
        combined_path = " & ".join(item["section_path"] for item in current_group)
        
        # Apply backward overlap if final_units exists
        if final_units and getattr(config, "CLAUSE_EXTRACTOR_CHUNK_OVERLAP", 0) > 0:
            overlap_text = _get_tail_overlap(final_units[-1]["text"], config.CLAUSE_EXTRACTOR_CHUNK_OVERLAP)
            if overlap_text:
                combined_text = f"{overlap_text}\n\n[CONTEXT OVERLAP]\n\n" + combined_text

        for split_chunk in split_oversized_text(
            combined_text,
            combined_path,
            max_tokens=config.CLAUSE_EXTRACTOR_OVERSIZED_SPLIT_TOKENS,
        ):
            final_units.append(split_chunk)

    return final_units


def _create_structured_units(
    final_units: list[dict], contract_type: str, parent_hash: str
) -> list[dict]:
    structured_units = []
    for idx, unit in enumerate(final_units):
        unit_text = unit["text"]
        unit_path = unit["path"]

        norm_text = normalize_whitespace(unit_text)
        chunk_hash = hashlib.sha1(
            f"{contract_type}:{unit_path}:{norm_text}".encode("utf-8")
        ).hexdigest()

        prev_title = final_units[idx - 1]["path"] if idx > 0 else "None"
        next_title = final_units[idx + 1]["path"] if idx + 1 < len(final_units) else "None"
        parent_title = unit_path.split(" > ")[0] if " > " in unit_path else "None"

        context_header = (
            f"Context Headers:\n"
            f"- Contract Type: {contract_type}\n"
            f"- Current Section: {unit_path}\n"
            f"- Parent Section: {parent_title}\n"
            f"- Previous Section: {prev_title}\n"
            f"- Next Section: {next_title}"
        )

        structured_units.append(
            {
                "id": chunk_hash,
                "section": unit_path,
                "path": unit_path,
                "text": unit_text,
                "token_count": get_precise_token_count(unit_text),
                "context_header": context_header,
                "parent_hash": parent_hash,
            }
        )
    return structured_units


def split_into_extraction_units(text: str, contract_type: str) -> list[dict]:
    # Phase 5 Step 1: Scan for and extract protected blocks
    sanitized_text, protected_blocks = _extract_protected_blocks(text)

    raw_sections = _split_by_sections(sanitized_text)

    if len(raw_sections) <= 1:
        pages = _split_by_pages(sanitized_text)
        chunks = _token_aware_chunk_plan(pages, target_chunk_tokens=config.CLAUSE_EXTRACTOR_CHUNK_SIZE)
        final_units = []
        parent_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        for idx, chunk in enumerate(chunks, 1):
            chunk_hash = hashlib.sha1(
                f"{contract_type}:Page Chunk {idx}:{normalize_whitespace(chunk)}".encode("utf-8")
            ).hexdigest()
            context_header = (
                f"Context Headers:\n"
                f"- Contract Type: {contract_type}\n"
                f"- Current Section: Page Chunk {idx}\n"
                f"- Previous Section: {'Page Chunk ' + str(idx-1) if idx > 1 else 'None'}\n"
                f"- Next Section: {'Page Chunk ' + str(idx+1) if idx < len(chunks) else 'None'}"
            )
            final_units.append(
                {
                    "id": chunk_hash,
                    "section": f"Page Chunk {idx}",
                    "path": f"Page Chunk {idx}",
                    "text": chunk,
                    "token_count": get_precise_token_count(chunk),
                    "context_header": context_header,
                    "parent_hash": parent_hash,
                }
            )
        # Restore protected blocks in final_units
        for unit in final_units:
            unit["text"] = _restore_protected_blocks(str(unit["text"]), protected_blocks)
            unit["token_count"] = get_precise_token_count(str(unit["text"]))
        return final_units

    heading_pattern = re.compile(
        r"(?:\n|^)"
        r"(?:"
        r"\s*(?:ARTICLE|SECTION|SECT|CLAUSE|EXHIBIT|SCHEDULE|APPENDIX)"
        r"\s+(?:[IVXLCDM]+|\d+(?:\.\d+)*)[\.\:\-\s].*"
        r"|"
        r"\s*\d+(?:\.\d+){0,2}\.?\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r"|"
        r"\s*[IVXLCDM]+\.\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r"|"
        r"\s*\([a-z]\)\s+[A-Z][A-Za-z0-9\s,\-\(\)]{2,60}"
        r")",
        re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(sanitized_text))
    raw_units = _build_raw_units(sanitized_text, matches)
    _resolve_section_paths(raw_units)

    processed_raw_units = []
    for u in raw_units:
        u_tokens = get_precise_token_count(u["text"])
        if u_tokens > config.CLAUSE_EXTRACTOR_CHUNK_SIZE + 1000:
            sub_chunks = split_oversized_text(u["text"], u["section_path"])
            for sub_chunk in sub_chunks:
                processed_raw_units.append(
                    {
                        "section_title": u["section_title"],
                        "section_path": sub_chunk["path"],
                        "text": sub_chunk["text"],
                    }
                )
        else:
            processed_raw_units.append(u)

    final_units = _group_units(processed_raw_units)
    parent_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
    structured_units = _create_structured_units(final_units, contract_type, parent_hash)

    # Restore protected blocks in structured_units
    for u in structured_units:
        u["text"] = _restore_protected_blocks(str(u["text"]), protected_blocks)
        u["token_count"] = get_precise_token_count(str(u["text"]))

    return structured_units
