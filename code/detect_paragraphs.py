"""Detect §-paragraphs and their headlines in Slovak law PDFs.

Loads a PDF with PyPDFLoader, strips the per-page running header, then walks
every § marker and resolves its headline using three priority-ordered patterns.

Patterns (highest priority first):
    1. AFTER § :  \\n§ N\\n{headline}\\n(1)
    2. BEFORE §:  \\n{headline}\\n§ N\\n(1)
                  — headline must start with a capital letter and not end with "."
    3. NONE   :  \\n§ N\\n(1)
    4. AFTER § (no "(1)"):  \\n§ N\\n{headline}\\n
                  — headline must start with a capital letter and not end with "."
"""

import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


PDF_PATH = Path(__file__).parent / "assets" / "ZZ_2004_222_20260101.pdf"

# Slovak uppercase letters (incl. diacritics) — used to spot heading starts.
CAPITAL = "A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ"
# Slovak lowercase letters (incl. diacritics) — used to detect soft-wrap continuations.
LOWERCASE = "a-záäčďéíľĺňóôŕšťúýž"

# Per-page running header. Two layouts:
#   even pages: "Strana N   Zbierka zákonov Slovenskej republiky   222/2004 Z. z."
#   odd  pages: "222/2004 Z. z.   Zbierka zákonov Slovenskej republiky   Strana N"
PAGE_HEADER_RE = re.compile(
    r"(?:Strana\s+\d+\s+Zbierka\s+zákonov\s+Slovenskej\s+republiky\s+222/2004\s+Z\.\s*z\.)"
    r"|(?:222/2004\s+Z\.\s*z\.\s+Zbierka\s+zákonov\s+Slovenskej\s+republiky\s+Strana\s+\d+)"
)

# Every §-block opens with "\n§ <number><0-2 letters>\n", e.g. "§ 4", "§ 16a", "§ 68cb".
SECTION_RE = re.compile(r"\n§\s*(\d+[a-z]{0,2})\n")


def _glue_soft_wrap(body: str) -> str:
    """Join soft-wrapped lines from the start of `body` into one logical line.

    Stops at terminal punctuation (.?!), blank lines, or structural markers
    that cannot be part of a headline:
      - "(" — subsection markers like "(1)", "(18)"
      - "§" — references to other paragraphs
      - "a)", "b)" ... — letter list markers
      - "1.", "2." ... — number list markers
    A continuation line must start with a lowercase Slovak letter; anything
    else (capital, digit, punctuation) ends the headline.
    """
    lines = body.split("\n")
    i = next((k for k, line in enumerate(lines) if line.strip()), None)
    if i is None:
        return ""

    glued = lines[i].strip()
    for j in range(i + 1, len(lines)):
        nxt = lines[j].strip()
        if not nxt:
            break
        if glued and glued[-1] in ".?!":
            break
        if nxt[0] in "§(":
            break
        if re.match(r"[a-z]\)", nxt) or re.match(r"\d+\.", nxt):
            break
        if not re.match(rf"[{LOWERCASE}]", nxt):
            break
        glued = f"{glued} {nxt}"
    return glued


def _is_headline_valid(candidate: str) -> bool:
    """Tightened Pattern-4 validator — rejects body prose mistaken for headlines.

    Real Slovak legal headlines are noun phrases. Sentences that get captured
    instead carry telltale signs we can filter on.
    """
    if not candidate:
        return False
    if not re.match(rf"[{CAPITAL}]", candidate):
        return False
    if candidate[-1] in ".:;?!":
        return False
    if candidate.count(",") >= 2:                 # enumeration → sentence
        return False
    if re.search(r"§\s*\d", candidate):           # cites another §
        return False
    if re.search(r"\d+\)", candidate):            # footnote marker like "33)"
        return False
    return True


def load_pdf_text(pdf_path: str | Path) -> str:
    """Load PDF, strip running headers per page, return concatenated text."""
    pages = PyPDFLoader(str(pdf_path)).load()
    cleaned = [PAGE_HEADER_RE.sub("", page.page_content) for page in pages]
    return "\n".join(cleaned)


def detect_paragraphs(text: str) -> list[dict]:
    """Find every § paragraph and resolve its headline.

    Returns a list of dicts:
        {
            "marker":     "§ 16a",
            "number":     "16a",
            "headline":   str | None,
            "position":   "after" | "before" | "none",
            "body_start": int,   # char offset in `text` where body starts
            "body_end":   int,   # char offset where body ends (next § or EOF)
        }
    """
    results: list[dict] = []
    matches = list(SECTION_RE.finditer(text))

    for i, m in enumerate(matches):
        number = m.group(1)
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end]

        headline: str | None = None
        position = "none"
        body_starts_with_1 = body.lstrip().startswith("(1)")

        if not body_starts_with_1:
            # Priority 1 — headline AFTER § (between "§ N\n" and "\n(1)")
            # `re.DOTALL` so `.+?` can span newlines for multi-line headlines.
            after_m = re.match(r"(.+?)\n\(1\)", body, re.DOTALL)
            if after_m and after_m.group(1).strip():
                headline = after_m.group(1).strip()
                position = "after"
            else:
                # Priority 4 — no "(1)" in body. Glue soft-wrapped lines into
                # one logical line, then validate against tightened heuristics
                # (rejects multi-comma enumerations, §-references, footnote
                # markers, sentences ending in . : ; etc.).
                candidate = _glue_soft_wrap(body)
                if _is_headline_valid(candidate):
                    headline = candidate
                    position = "after"

        # Priority 2 (also as fallback) — headline BEFORE § (last line above
        # "§ N\n"). Runs when no AFTER headline was found, regardless of
        # whether the body starts with "(1)" or with prose.
        if headline is None:
            before = text[: m.start()].rstrip()
            before_m = re.search(rf"\n([{CAPITAL}][^\n]*?)$", before)
            if before_m:
                candidate = before_m.group(1).strip()
                if not candidate.endswith("."):
                    headline = candidate
                    position = "before"
            # else: Priority 3 — no headline (position stays "none").

        results.append(
            {
                "marker": f"§ {number}",
                "number": number,
                "headline": headline,
                "position": position,
                "body_start": body_start,
                "body_end": body_end,
            }
        )

    return results


if __name__ == "__main__":
    text = load_pdf_text(PDF_PATH)
    paragraphs = detect_paragraphs(text)

    for p in paragraphs:
        headline = p["headline"] or "—"
        print(f"{p['marker']}  {headline}\n")
