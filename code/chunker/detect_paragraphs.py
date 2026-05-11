"""Detect §-paragraphs and their headlines in Slovak law PDFs.

Walks every § marker in already-loaded text and resolves its headline using
three priority-ordered patterns. PDF loading lives in `chunker.Chunker.load_pdf_text`.

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

# Slovak uppercase letters (incl. diacritics) — used to spot heading starts.
CAPITAL = "A-ZÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ"
# Slovak lowercase letters (incl. diacritics) — used to detect soft-wrap continuations.
LOWERCASE = "a-záäčďéíľĺňóôŕšťúýž"

# Every §-block opens with "\n§ <number><0-2 letters>\n", e.g. "§ 4", "§ 16a", "§ 68cb".
SECTION_RE = re.compile(r"\n§\s*(\d+[a-z]{0,2})\n")

# First Odsek marker "(N) " at line start — used only to find the boundary
# between a §'s own prose and its subsections. Full Odsek parsing lives in
# detect_subsections.py.
ODSEK_PROBE_RE = re.compile(r"^\(\d+\)\s", re.MULTILINE)


def _glue_soft_wrap(body: str) -> tuple[str, int]:
    """Join soft-wrapped lines from the start of `body` into one logical line.

    Returns (glued, end_offset) where:
        glued      — the joined first paragraph (lines combined with ' ')
        end_offset — char offset in `body` immediately after the last consumed
                     line's trailing '\\n'; body[end_offset:] is whatever
                     follows the glued headline. Needed because soft-wrap
                     replaces '\\n' with ' ' ambiguously, so the cut point
                     cannot be recovered from `glued` alone.

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
        return "", 0

    glued = lines[i].strip()
    last_j = i
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
        last_j = j

    end_offset = sum(len(lines[k]) + 1 for k in range(last_j + 1))
    return glued, end_offset


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


def page_for_offset(offset: int, page_offsets: list[tuple[int, int, int]]) -> int:
    """Return the 1-indexed PDF page number containing `offset`. Falls back to
    the last page if `offset` lies at or past EOF."""
    for page_num, start, end in page_offsets:
        if start <= offset < end:
            return page_num
    return page_offsets[-1][0]


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
            "text":       str,   # prose between headline and first "(N)"
                                 # (or full prose if no subsections — this is
                                 # what captures the body of §s like 85l/85m
                                 # that have content but no odseky)
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
        headline_end_in_body = 0  # where the prose body begins (post-headline)

        if not body_starts_with_1:
            # Priority 1 — headline AFTER § (between "§ N\n" and "\n(1)")
            # `re.DOTALL` so `.+?` can span newlines for multi-line headlines.
            after_m = re.match(r"(.+?)\n\(1\)", body, re.DOTALL)
            if after_m and after_m.group(1).strip():
                headline = after_m.group(1).strip()
                position = "after"
                headline_end_in_body = after_m.end(1)
            else:
                # Priority 4 — no "(1)" in body. Glue soft-wrapped lines into
                # one logical line, then validate against tightened heuristics
                # (rejects multi-comma enumerations, §-references, footnote
                # markers, sentences ending in . : ; etc.).
                candidate, candidate_end = _glue_soft_wrap(body)
                if _is_headline_valid(candidate):
                    headline = candidate
                    position = "after"
                    headline_end_in_body = candidate_end

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

        # Prose body: everything after the headline up to the first "(N)"
        # subsection marker (or the entire remainder if no subsections).
        prose = body[headline_end_in_body:]
        first_odsek = ODSEK_PROBE_RE.search(prose)
        prose_text = (prose[: first_odsek.start()] if first_odsek else prose).strip()

        results.append(
            {
                "marker": f"§ {number}",
                "number": number,
                "headline": headline,
                "position": position,
                "body_start": body_start,
                "body_end": body_end,
                "text": prose_text,
            }
        )

    return results


if __name__ == "__main__":
    from chunker.chunker import Chunker  # lazy: avoids module-level circular import

    PDF_PATH = Path(__file__).parent / "assets" / "ZZ_2004_222_20260101.pdf"
    text, _page_offsets = Chunker().load_pdf_text(PDF_PATH)
    paragraphs = detect_paragraphs(text)

    for p in paragraphs:
        headline = p["headline"] or "—"
        print(f"{p['marker']}  {headline}\n")
