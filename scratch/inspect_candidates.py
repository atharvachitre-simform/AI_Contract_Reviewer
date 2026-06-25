import sys

sys.path.insert(0, '.')
import re

from src.helpers.contract_analysis import normalize_whitespace

with open('scratch/contract_46ce978f8a9180f4.txt', 'r') as f:
    raw_text = f.read()

# Let's run the header/footer detection logic isolated so we can print the candidate set
cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n")
page_marker_re = re.compile(r"(--- page \d+ ---)", re.IGNORECASE)
parts = page_marker_re.split(cleaned)

pages = []
markers = []
i = 0
if len(parts) > 1:
    if not parts[0].strip() and page_marker_re.match(parts[1]):
        i = 1

while i < len(parts):
    if page_marker_re.match(parts[i]):
        markers.append(parts[i])
        if i + 1 < len(parts):
            pages.append(parts[i+1])
            i += 2
        else:
            pages.append("")
            i += 1
    else:
        pages.append(parts[i])
        markers.append("")
        i += 1

from collections import Counter

line_counts = Counter()
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")

for page in pages:
    seen_lines_in_page = set()
    for line in page.split("\n"):
        line_clean = _WHITESPACE_RE.sub(" ", line).strip()
        if len(line_clean) > 8:
            seen_lines_in_page.add(line_clean)
    for line_clean in seen_lines_in_page:
        line_counts[line_clean] += 1

header_footer_candidates = set()
for line_clean, count in line_counts.items():
    if count >= 3:
        lower_line = line_clean.lower()
        is_heading = any(lower_line.startswith(prefix) for prefix in [
            "section", "article", "clause", "para", "part", "schedule", "exhibit"
        ])
        is_heading = is_heading or bool(re.match(r"^\d+[\.\s]+[A-Z]", line_clean))
        if not is_heading:
            header_footer_candidates.add(line_clean)

print("Number of candidate header/footers identified:", len(header_footer_candidates))
print("Sample candidates (first 30):")
for c in sorted(list(header_footer_candidates))[:30]:
    print(f" - {c[:100]}")
