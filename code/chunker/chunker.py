"""Linearize the nested § → odsek → letter → bod tree from `detect_subsections`
into a flat list of self-contained text chunks.

Each chunk is emitted at the deepest available level of its path; parent
`lead`s are forward-concatenated so the chunk reads as standalone prose.
A § with no odseky becomes a single chunk built from its `text` field.
"""

import json
import re
import sys
from pathlib import Path
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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


def _join(*parts: str) -> str:
    """Strip each part, drop empties, join with a single space."""
    return " ".join(s for s in (p.strip() for p in parts) if s)


_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=3000,
    chunk_overlap=300,
)


def _split_if_long(doc: Document) -> list[Document]:
    if len(doc.page_content) <= 3000:
        return [doc]
    return [Document(page_content=p, metadata=doc.metadata) for p in _SPLITTER.split_text(doc.page_content)]


def _make_doc(text: str, path: list[str], headline: str | None) -> Document:
    """Build a langchain `Document` with `path` + `headline` in metadata."""
    metadata: dict = {"path": path}
    if headline is not None:
        metadata["headline"] = headline
    return Document(page_content=text, metadata=metadata)


def write_json(chunks: list[Document], path: str | Path) -> None:
    """Serialize a list of `Document`s to `path` as UTF-8 JSON (no ASCII escapes,
    so Slovak diacritics stay readable)."""
    Path(path).write_text(
        json.dumps(
            [{"text": d.page_content, **d.metadata} for d in chunks],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class Chunker:
    """Flattens the nested paragraph tree into a list of `Document`s."""

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
        """Load a Slovak law PDF and return {"chunks": [...], "tree_graph": GraphDocument, "last_page": int | None}.

        `tree_graph` is a GraphDocument with a Document root node linked to
        Paragraf/Odsek/Pismeno/Bod nodes via IN_DOCUMENT / IN_SECTION edges.
        `last_page` is the `page_start` of the final § paragraph (the page on
        which the closing block of the law begins). `None` if no paragraphs.
        """
        text, page_offsets = self.load_pdf_text(pdf_path)
        paragraphs = detect_subsections(text, page_offsets)

        tree_graph = self.get_tree_graph(paragraphs, pdf_path)
        chunks = self.linearize(paragraphs)

        last_page = paragraphs[-1].get("page_start") if paragraphs else None
        return {"chunks": chunks, "tree_graph": tree_graph, "last_page": last_page}



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


    def get_tree_graph(self, paragraphs: list[dict], pdf_path: str | Path) -> GraphDocument:
        """Walk the 4-level tree depth-first and emit a GraphDocument linking
        Document -> Paragraf -> Odsek -> Pismeno -> Bod.

        IDs are human-readable and accumulate ancestor labels:
            "Paragraf § 16a"
            "Paragraf § 16a Odsek 2"
            "Paragraf § 16a Odsek 2 Pismeno c)"
            "Paragraf § 16a Odsek 2 Pismeno c) Bod 3"

        `path` stores the per-level token that follows each label, so joining
        label + path[i] reproduces the id segment-by-segment.
        """
        document_id = Path(pdf_path).stem
        document_node = Node(id=document_id, type="Document", properties={})
        nodes: list[Node] = [document_node]
        relationships: list[Relationship] = []

        for para in paragraphs:
            if "marker" not in para:
                continue
            p_token = para["marker"]               # "§ 16a"
            p_id = f"Paragraf {p_token}"
            p_path = [p_token]
            p_extras = {
                "headline": para.get("headline"),
                "page_start": para.get("page_start"),
                "page_end": para.get("page_end"),
            }
            p_node = self._make_node(p_id, "Paragraf", p_path, para.get("text", ""), p_extras)
            nodes.append(p_node)
            relationships.append(Relationship(source=p_node, target=document_node, type="IN_DOCUMENT"))

            for odsek in para.get("odseky", []):
                o_token = odsek["number"]          # "1"
                o_id = f"{p_id} Odsek {o_token}"
                o_path = p_path + [o_token]
                o_node = self._make_node(o_id, "Odsek", o_path, odsek["lead"])
                nodes.append(o_node)
                relationships.append(Relationship(source=o_node, target=p_node, type="IN_SECTION"))

                for letter in odsek.get("letters", []):
                    l_token = letter["marker"]     # "a)"
                    l_id = f"{o_id} Pismeno {l_token}"
                    l_path = o_path + [l_token]
                    l_node = self._make_node(l_id, "Pismeno", l_path, letter["lead"])
                    nodes.append(l_node)
                    relationships.append(Relationship(source=l_node, target=o_node, type="IN_SECTION"))

                    for bod in letter.get("bode", []):
                        b_token = bod["number"]    # "3"
                        b_id = f"{l_id} Bod {b_token}"
                        b_path = l_path + [b_token]
                        b_node = self._make_node(b_id, "Bod", b_path, bod["lead"])
                        nodes.append(b_node)
                        relationships.append(Relationship(source=b_node, target=l_node, type="IN_SECTION"))

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"id": document_id}),
        )



    def linearize(self, paragraphs: list[dict]) -> list[Document]:
        chunks: list[Document] = []
        for para in paragraphs:
            if "marker" not in para:
                continue
            self._walk_paragraph(para, chunks)

        write_json(chunks, "./chunks.json")
        return chunks



    def _walk_paragraph(self, para: dict, out: list[Document]) -> None:
        # Path tokens match the per-level id segments built in `get_tree_graph`:
        #   Paragraf → marker ("§ 16a"), Odsek → number ("1"),
        #   Pismeno → marker ("a)"),     Bod   → number ("3").
        # Prose ("o_part", "l_part") keeps using markers since "(1) ..." and
        # "a) ..." are the natural reading form for an LLM.
        p_token = para["marker"]
        p_text = para.get("text", "")
        headline = para.get("headline")
        odseky = para.get("odseky", [])

        if not odseky:
            out.extend(_split_if_long(_make_doc(_join(p_text), [p_token], headline)))
            return

        for odsek in odseky:
            o_token = odsek["number"]
            o_part = f"{odsek['marker']} {odsek['lead']}"
            letters = odsek.get("letters", [])

            if not letters:
                out.extend(_split_if_long(_make_doc(
                    _join(p_text, o_part),
                    [p_token, o_token],
                    headline,
                )))
                continue

            for letter in letters:
                l_token = letter["marker"]
                l_part = f"{letter['marker']} {letter['lead']}"
                bode = letter.get("bode", [])

                if bode:
                    bod_lines = "\n".join(
                        f"{bod['marker']} {bod['lead'].strip()}" for bod in bode
                    )
                    l_part = f"{l_part}\n{bod_lines}"

                out.extend(_split_if_long(_make_doc(
                    _join(p_text, o_part, l_part),
                    [p_token, o_token, l_token],
                    headline,
                )))


if __name__ == "__main__":
    PDF_PATH = Path(__file__).resolve().parent.parent / "assets" / "ZZ_2004_222_20260101.pdf"
    result = Chunker().split_document(PDF_PATH)
    chunks = result["chunks"]
    print(f"Total chunks: {len(chunks)}")
    print(f"Last page: {result['last_page']}")
    for c in chunks[:5]:
        print(f"{' › '.join(c.metadata['path']):30s}  {c.page_content[:80]}…")
