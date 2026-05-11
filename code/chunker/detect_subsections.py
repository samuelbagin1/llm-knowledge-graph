"""Detect Odseky (subsections), Písmená (letter items), and Body (numeric items)
inside each § paragraph of a Slovak law PDF.

Augments the output of `detect_paragraphs` with a nested tree:

    §  →  (N) Odsek  →  a) Písmeno  →  N. Bod

In the PyPDFLoader text, every level-marker sits at column 0 of its line
(visual indentation is lost, but '\\n' is preserved):

    Odsek    ^\\((\\d+)\\)\\s     "(1) "  ... "(15) "
    Písmeno  ^([a-z])\\)\\s       "a) "   ... "i) "
    Bod      ^(\\d+)\\.\\s        "1. "   ... "4. "

The line-start anchor disambiguates these from the *inline* tokens that share
the same shape — cross-references like `písm. a)` and footnote markers like
`predpisu.4c)` or `osobitného predpisu.33)` never appear at column 0.
"""

import json
import re
from pathlib import Path
from time import sleep

from chunker.detect_paragraphs import (
    detect_paragraphs,
    page_for_offset,
)


PDF_PATH = Path(__file__).parent / "assets" / "ZZ_2004_222_20260101.pdf"

ODSEK_RE = re.compile(r"^\((\d+)\)\s", re.MULTILINE)
LETTER_RE = re.compile(r"^([a-z])\)\s", re.MULTILINE)
BOD_RE = re.compile(r"^(\d+)\.\s", re.MULTILINE)


def _split_children(
    pattern: re.Pattern,
    body: str,
    body_start: int,
) -> tuple[str, list[tuple[str, str, int, int]]]:
    """Walk every `pattern` match in `body`, return (lead, [child_tuples]).

    Each child tuple is (marker_text, captured_key, abs_body_start, abs_body_end).
    Offsets are absolute — `body_start` is added so they index back into the
    full source text. `lead` is body[0 : first_match.start()] (empty when body
    begins with a child marker; full body when there are no matches).
    """
    matches = list(pattern.finditer(body))
    if not matches:
        return body, []

    lead = body[: matches[0].start()]
    children = [
        (
            m.group(0).strip(),
            m.group(1),
            body_start + m.end(),
            body_start + (matches[i + 1].start() if i + 1 < len(matches) else len(body)),
        )
        for i, m in enumerate(matches)
    ]
    return lead, children


def _build_letters(text: str, raw) -> list[dict]:
    letters = []
    for marker, ltr, bs, be in raw:
        lead, bod_raw = _split_children(BOD_RE, text[bs:be], bs)
        letters.append(
            {
                "marker": marker,
                "letter": ltr,
                "body_start": bs,
                "body_end": be,
                "lead": lead,
                "bode": [
                    {
                        "marker": m,
                        "number": k,
                        "body_start": s,
                        "body_end": e,
                        "lead": text[s:e],
                    }
                    for m, k, s, e in bod_raw
                ],
            }
        )
    return letters


def _build_odseky(text: str, raw) -> list[dict]:
    odseky = []
    for marker, num, bs, be in raw:
        lead, letter_raw = _split_children(LETTER_RE, text[bs:be], bs)
        odseky.append(
            {
                "marker": marker,
                "number": num,
                "body_start": bs,
                "body_end": be,
                "lead": lead,
                "letters": _build_letters(text, letter_raw),
            }
        )
    return odseky


def _attach_pages(paragraphs: list[dict], page_offsets) -> None:
    """Add `page_start` / `page_end` (1-indexed PDF page numbers) to each
    top-level paragraph that carries body offsets. Skips entries without
    body_start/body_end (the trailing appendix dict)."""
    for p in paragraphs:
        if "body_start" not in p or "body_end" not in p:
            continue
        p["page_start"] = page_for_offset(p["body_start"], page_offsets)
        # body_end is exclusive; clamp for empty bodies so we don't query the
        # offset of the previous paragraph's last character.
        last_pos = max(p["body_end"] - 1, p["body_start"])
        p["page_end"] = page_for_offset(last_pos, page_offsets)


def _trim_last_paragraph_tail(paragraphs: list[dict]) -> None:
    """Trim back-matter from the last paragraph's text.

    detect_paragraphs lumps everything from the last §'s marker to EOF into
    its text. In Slovak statute formatting, the only blank line ('\\n\\n') in
    this trailing region appears between the closing block (the §'s prose +
    its two signature lines) and the back-matter (Príloha č. N, footnotes,
    publisher footer). Splitting at the first blank line keeps the closing
    block and discards the back-matter — the appendix content is not preserved
    in the final output.

    Mutates `paragraphs[-1]` in place; no-op if the list is empty.
    """
    if not paragraphs:
        return
    last = paragraphs[-1]
    head, _, _ = last["text"].partition("\n\n")
    last["text"] = head.rstrip()


def write_json(paragraphs: list[dict], path: str | Path) -> None:
    """Serialize the parsed tree to `path` as UTF-8 JSON (no ASCII escapes,
    so Slovak diacritics stay readable)."""
    Path(path).write_text(
        json.dumps(paragraphs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def detect_subsections(text: str, page_offsets=None) -> list[dict]:
    """Return paragraphs (as from `detect_paragraphs`) with nested children.

    Output schema (in addition to the detect_paragraphs fields):
        "odseky": [
            {
                "marker": "(1)", "number": "1",
                "body_start": int, "body_end": int,
                "lead": str,                  # prose before first letter
                "letters": [
                    {
                        "marker": "a)", "letter": "a",
                        "body_start": int, "body_end": int,
                        "lead": str,          # prose before first bod
                        "bode": [
                            {"marker": "1.", "number": "1",
                             "body_start": int, "body_end": int,
                             "lead": str}    # the bod's full text body
                        ]
                    }
                ]
            }
        ]
    """
    paragraphs = detect_paragraphs(text)
    for p in paragraphs:
        body = text[p["body_start"]:p["body_end"]]
        _, odsek_raw = _split_children(ODSEK_RE, body, p["body_start"])
        p["odseky"] = _build_odseky(text, odsek_raw)

    _trim_last_paragraph_tail(paragraphs)

    if page_offsets is not None:
        _attach_pages(paragraphs, page_offsets)
    return paragraphs


if __name__ == "__main__":
    from chunker.chunker import Chunker  # lazy: avoids module-level circular import

    text, page_offsets = Chunker().load_pdf_text(PDF_PATH)
    paragraphs = detect_subsections(text, page_offsets)
    
    write_json(paragraphs, "./chunked_document.json")

    # for p in paragraphs:
    #     headline = p["headline"] or "—"
    #     print(f"{p['marker']}  {headline}  ({len(p['odseky'])} odseky)")
    #     for o in p["odseky"]:
    #         tag = f"{len(o['letters'])} letters" if o["letters"] else "—"
    #         print(f"  {o['marker']}  {tag}")
    #         for l in o["letters"]:
    #             if l["bode"]:
    #                 print(f"    {l['marker']}  ({len(l['bode'])} bode)")
    #                 for b in l["bode"]:
    #                     print(f"      {b['marker']}")
    #             else:
    #                 print(f"    {l['marker']}")
    #     sleep(5)
    #     print()
