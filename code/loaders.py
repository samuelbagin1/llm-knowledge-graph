"""Loaders that reverse the writers in `to_json.py`.

Each loader reads a JSON file produced by the corresponding `*_to_json`
function and reconstructs the in-memory object(s):

    odd_to_json         -> load_odd          -> Schema
    refinement_to_json  -> load_refinement   -> Schema
    sde_to_json         -> load_sde          -> List[GraphDocument]
    table_to_json       -> load_tables       -> List[GraphDocument]
    formula_to_json     -> load_formula      -> Optional[GraphDocument]

Lossy round-trip notes:
- `sde_to_json` truncates `source.page_content` to 200 chars; loaders return
  a Document with that truncated text (no way to recover the original).
- ODD's `chunk_texts` are not reattached to a Schema; they're discarded here.
"""

import json
from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

from classes import Schema


def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _dict_to_node(d: dict) -> Node:
    node_id = d.get("id")
    if not isinstance(node_id, (str, int)):
        raise ValueError(f"_dict_to_node: node 'id' must be str or int, got {type(node_id).__name__}")
    return Node(
        id=node_id,
        type=d.get("type", ""),
        properties=d.get("properties") or {},
    )


def _dict_to_relationship(d: dict) -> Relationship:
    return Relationship(
        source=_dict_to_node(d["source"]),
        target=_dict_to_node(d["target"]),
        type=d.get("type", ""),
        properties=d.get("properties") or {},
    )


def load_odd(path: str) -> Schema:
    """Load an ODD JSON file into a single deduplicated Schema.

    Prefers the precomputed `nested` block (already deduped via set()). Falls
    back to recomputing the union from `chunked` if `nested` is missing,
    matching the shape used at pdf_graphrag.py:2444 to feed schema_refinement.
    """
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"load_odd: expected list at top level, got {type(data).__name__}")

    nested: Optional[dict] = None
    chunked: Optional[list] = None
    for section in data:
        if not isinstance(section, dict):
            continue
        if "nested" in section and isinstance(section["nested"], dict):
            nested = section["nested"]
        if "chunked" in section and isinstance(section["chunked"], list):
            chunked = section["chunked"]

    if nested is not None:
        return Schema(
            nodes=list(nested.get("nodes") or []),
            relationships=list(nested.get("relationships") or []),
        )

    if chunked is None:
        raise ValueError("load_odd: file has neither 'nested' nor 'chunked' section")

    nodes: set[str] = set()
    rels: set[str] = set()
    for entry in chunked:
        if not isinstance(entry, dict):
            continue
        nodes.update(entry.get("nodes") or [])
        rels.update(entry.get("relationships") or [])
    return Schema(nodes=list(nodes), relationships=list(rels))


def load_refinement(path: str) -> Schema:
    """Load a refinement JSON into a Schema.

    Maps `node_types` -> Schema.nodes and `relationship_types` -> Schema.relationships.
    The `merge_log` is intentionally dropped (Schema has no field for it).
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"load_refinement: expected dict, got {type(data).__name__}")
    return Schema(
        nodes=list(data.get("node_types") or []),
        relationships=list(data.get("relationship_types") or []),
    )


def load_sde(path: str) -> List[GraphDocument]:
    """Load an SDE JSON into a list of GraphDocuments.

    Note: `source.page_content` was truncated to 200 chars by the writer, so
    the reconstructed Document carries only that prefix.
    """
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"load_sde: expected list, got {type(data).__name__}")

    docs: List[GraphDocument] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        source_text = entry.get("source")
        source = Document(page_content=source_text) if source_text is not None else None
        docs.append(GraphDocument(
            nodes=[_dict_to_node(n) for n in (entry.get("nodes") or [])],
            relationships=[_dict_to_relationship(r) for r in (entry.get("relationships") or [])],
            source=source,
        ))
    return docs


def load_tables(path: str) -> List[GraphDocument]:
    """Load a tables JSON (from `table_to_json`) into GraphDocuments.

    Reconstructs Document with the full source HTML and `page_range` metadata.
    """
    data = _read_json(path)
    if not isinstance(data, list):
        raise ValueError(f"load_tables: expected list, got {type(data).__name__}")

    docs: List[GraphDocument] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        html = entry.get("source_html")
        page_range = entry.get("page_range")
        if html is not None:
            metadata = {"page_range": page_range} if page_range is not None else {}
            source = Document(page_content=html, metadata=metadata)
        else:
            source = None
        docs.append(GraphDocument(
            nodes=[_dict_to_node(n) for n in (entry.get("nodes") or [])],
            relationships=[_dict_to_relationship(r) for r in (entry.get("relationships") or [])],
            source=source,
        ))
    return docs


def load_formula(path: str) -> Optional[GraphDocument]:
    """Load a formulas JSON (from `formula_to_json`) into a single GraphDocument.

    Returns None if the file represents an empty payload (no nodes and no
    document_id), mirroring the `data is None` branch of the writer.
    """
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"load_formula: expected dict, got {type(data).__name__}")

    nodes_raw = data.get("nodes") or []
    rels_raw = data.get("relationships") or []
    document_id = data.get("document_id")

    if document_id is None and not nodes_raw and not rels_raw:
        return None

    source = Document(page_content="", metadata={"id": document_id}) if document_id is not None else None
    return GraphDocument(
        nodes=[_dict_to_node(n) for n in nodes_raw],
        relationships=[_dict_to_relationship(r) for r in rels_raw],
        source=source,
    )
