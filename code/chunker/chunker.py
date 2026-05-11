"""Linearize the nested § → odsek → letter → bod tree from `detect_subsections`
into a flat list of self-contained text chunks.

Each chunk is emitted at the deepest available level of its path; parent
`lead`s are forward-concatenated so the chunk reads as standalone prose.
A § with no odseky becomes a single chunk built from its `text` field.
"""

from dataclasses import asdict, dataclass
import json
import re
import sys
from pathlib import Path
from langchain_neo4j.graphs.graph_document import Node
from langchain_community.document_loaders import PyPDFLoader

# When run as a script (python chunker.py), sys.path[0] is this file's directory
# and `import chunker` would resolve to *this file*, masking the package. Add
# the parent (`code/`) so `chunker` resolves to the directory namespace package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chunker.detect_subsections import detect_subsections


_LAW_ID = r"\d+/\d{4}"
_PAGE_HEADER_RE = re.compile(
    rf"(?:Strana\s+\d+\s+Zbierka\s+zákonov\s+Slovenskej\s+republiky\s+{_LAW_ID}\s+Z\.\s*z\.)"
    rf"|(?:{_LAW_ID}\s+Z\.\s*z\.\s+Zbierka\s+zákonov\s+Slovenskej\s+republiky\s+Strana\s+\d+)"
)


@dataclass
class Chunk:
    text: str
    path: list[str]
    headline: str | None = None


def _join(*parts: str) -> str:
    """Strip each part, drop empties, join with a single space."""
    return " ".join(s for s in (p.strip() for p in parts) if s)


def write_json(chunks: list[Chunk], path: str | Path) -> None:
    """Serialize a list of `Chunk`s to `path` as UTF-8 JSON (no ASCII escapes,
    so Slovak diacritics stay readable)."""
    Path(path).write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class Chunker:
    """Flattens the nested paragraph tree into a list of `Chunk`s."""

    def load_pdf_text(self, pdf_path: str | Path) -> tuple[str, list[tuple[int, int, int]]]:
        """Load PDF, strip running headers, return (text, page_offsets).

        `page_offsets` is a list of (page_number_1indexed, start_inclusive,
        end_exclusive) tuples covering the concatenated text. Use it with
        `page_for_offset` to map any character offset back to a PDF page.
        """
        pages = PyPDFLoader(str(pdf_path)).load()
        cleaned = [_PAGE_HEADER_RE.sub("", page.page_content) for page in pages]

        # Make spans contiguous by absorbing each page's trailing "\n" separator
        # (inserted by join) into that page's range — otherwise a position landing
        # exactly on a separator belongs to no page and the lookup falls through.
        page_offsets: list[tuple[int, int, int]] = []
        offset = 0
        total = len(cleaned)
        for page_num, content in enumerate(cleaned, start=1):
            advance = len(content) + (1 if page_num < total else 0)
            page_offsets.append((page_num, offset, offset + advance))
            offset += advance

        return "\n".join(cleaned), page_offsets
    


    def split_document(self, pdf_path: str | Path) -> dict:
        """Load a Slovak law PDF and return {"chunks": [...], "last_page": int | None}.

        `last_page` is the `page_start` of the final § paragraph (the page on
        which the closing block of the law begins). `None` if no paragraphs.
        """
        text, page_offsets = self.load_pdf_text(pdf_path)
        paragraphs = detect_subsections(text, page_offsets)
        
        all_nodes = self.get_nodes(paragraphs)
        chunks = self.linearize(paragraphs)
        
        last_page = paragraphs[-1].get("page_start") if paragraphs else None
        return {"chunks": chunks, "nodes": all_nodes, "last_page": last_page}



    def _make_node(
        self,
        node_id: str,
        node_type: str,
        path: list[str],
        text: str,
        extras: dict | None = None,
    ) -> Node:
        """Build a single graph Node. `extras` lets the paragraph-level caller
        attach `headline` / `page_start` / `page_end` (None values are dropped
        so the graph isn't littered with nulls)."""
        properties = {"path": path, "text": text}
        if extras:
            properties.update({k: v for k, v in extras.items() if v is not None})
        return Node(id=node_id, type=node_type, properties=properties)


    def get_nodes(self, paragraphs: list[dict]) -> list[Node]:
        """Walk the 4-level tree depth-first and emit one Node at every level.

        IDs are human-readable and accumulate ancestor labels:
            "Paragraf §16a"
            "Paragraf §16a Odsek 2"
            "Paragraf §16a Odsek 2 Pismeno c"
            "Paragraf §16a Odsek 2 Pismeno c Bod 3"
        """
        nodes: list[Node] = []

        for para in paragraphs:
            if "marker" not in para:
                continue
            p_id = f"Paragraf §{para['number']}"
            p_path = [para["marker"]]
            p_extras = {
                "headline": para.get("headline"),
                "page_start": para.get("page_start"),
                "page_end": para.get("page_end"),
            }
            nodes.append(self._make_node(p_id, "Paragraf", p_path, para.get("text", ""), p_extras))

            for odsek in para.get("odseky", []):
                o_id = f"{p_id} Odsek {odsek['number']}"
                o_path = p_path + [odsek["marker"]]
                nodes.append(self._make_node(o_id, "Odsek", o_path, odsek["lead"]))

                for letter in odsek.get("letters", []):
                    l_id = f"{o_id} Pismeno {letter['letter']}"
                    l_path = o_path + [letter["marker"]]
                    nodes.append(self._make_node(l_id, "Pismeno", l_path, letter["lead"]))

                    for bod in letter.get("bode", []):
                        b_id = f"{l_id} Bod {bod['number']}"
                        b_path = l_path + [bod["marker"]]
                        nodes.append(self._make_node(b_id, "Bod", b_path, bod["lead"]))

        return nodes


    def linearize(self, paragraphs: list[dict]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for para in paragraphs:
            if "marker" not in para:
                continue
            self._walk_paragraph(para, chunks)
            
        write_json(chunks, "./chunks.json")
        return chunks
    
    

    def _walk_paragraph(self, para: dict, out: list[Chunk]) -> None:
        p_marker = para["marker"]
        p_text = para.get("text", "")
        headline = para.get("headline")
        odseky = para.get("odseky", [])

        if not odseky:
            out.append(Chunk(text=_join(p_text), path=[p_marker], headline=headline))
            return

        for odsek in odseky:
            o_marker, o_lead = odsek["marker"], odsek["lead"]
            letters = odsek.get("letters", [])

            if not letters:
                out.append(Chunk(
                    text=_join(p_text, o_lead),
                    path=[p_marker, o_marker],
                    headline=headline,
                ))
                continue

            for letter in letters:
                l_marker, l_lead = letter["marker"], letter["lead"]
                bode = letter.get("bode", [])

                if not bode:
                    out.append(Chunk(
                        text=_join(p_text, o_lead, l_lead),
                        path=[p_marker, o_marker, l_marker],
                        headline=headline,
                    ))
                    continue

                for bod in bode:
                    out.append(Chunk(
                        text=_join(p_text, o_lead, l_lead, bod["lead"]),
                        path=[p_marker, o_marker, l_marker, bod["marker"]],
                        headline=headline,
                    ))


if __name__ == "__main__":
    PDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "ZZ_2004_222_20260101.pdf"
    result = Chunker().split_document(PDF_PATH)
    chunks = result["chunks"]
    print(f"Total chunks: {len(chunks)}")
    print(f"Last page: {result['last_page']}")
    for c in chunks[:5]:
        print(f"{' › '.join(c.path):30s}  {c.text[:80]}…")
