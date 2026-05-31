import json
import os
from datetime import datetime
from pathlib import Path
from classes import Schema
from typing import List, Optional, Any
from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship


DEFAULT_OUTPUT_DIR = "./file_output"


def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d%H%M%S')


def _safe_dump(payload: Any, path: str) -> Optional[str]:
    """
    Try json.dump; on failure write raw repr to <path>.failed.txt and re-raise.

    Returns the saved path on success. Re-raises the original exception on
    failure so callers know persistence broke, while still preserving as
    much of the payload as possible to disk.
    """
    try:
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path
    except (TypeError, ValueError, OSError) as e:
        fallback = path + ".failed.txt"
        try:
            with open(fallback, "w") as f:
                f.write(repr(payload))
            print(f"[to_json] json.dump failed for {path}: {e}; raw payload at {fallback}")
        except OSError as e2:
            print(f"[to_json] both json.dump and fallback write failed: {e}; {e2}")
        raise


def _node_to_dict(node: Node) -> dict:
    return {"id": node.id, "type": node.type, "properties": node.properties}


def _relationship_to_dict(rel: Relationship) -> dict:
    return {
        "source": _node_to_dict(rel.source),
        "target": _node_to_dict(rel.target),
        "type": rel.type,
        "properties": rel.properties,
    }


def odd_to_json(
    documents: List[Schema],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    name: str = "",
    chunks: Optional[list] = None,
    timestamp: Optional[str] = None,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)

    if chunks is None:
        chunks = []

    schemas = []
    all_nodes: List[str] = []
    all_relationships: List[str] = []

    for doc in (documents or []):
        if doc is None:
            continue
        try:
            entry = {"nodes": list(doc.nodes), "relationships": list(doc.relationships)}
            schemas.append(entry)
            all_nodes.extend(doc.nodes)
            all_relationships.extend(doc.relationships)
        except AttributeError as e:
            print(f"[odd_to_json] skipping malformed schema entry: {e}")
            continue

    output: List[dict] = []
    output.append({"chunked": schemas})

    chunk_texts = []
    for chunk in chunks:
        try:
            chunk_text = {"text": chunk.page_content, "page": chunk.metadata.get("page", 0)}
            chunk_texts.append(chunk_text)
        except AttributeError as e:
            print(f"[odd_to_json] skipping malformed chunk: {e}")
            continue

    output.append({"chunk_texts": chunk_texts})

    merged = {
        "nodes": list(set(all_nodes)),
        "relationships": list(set(all_relationships)),
    }
    output.append({"nested": merged})

    ts = timestamp or _timestamp()
    filename = f"{name}_odd_{ts}.json"
    path = os.path.join(output_dir, filename)
    return _safe_dump(output, path)


def refinement_to_json(
    data: Any,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    name: str = "",
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)

    safe_data = data if isinstance(data, dict) else {}
    raw_merge_log = safe_data.get("merge_log")
    merge_log = raw_merge_log if isinstance(raw_merge_log, dict) else {}

    output = {
        "node_types": safe_data.get("node_types", []) or [],
        "relationship_types": safe_data.get("relationship_types", []) or [],
        "merge_log": {
            "node_types": merge_log.get("node_types", {}) or {},
            "relationship_types": merge_log.get("relationship_types", {}) or {},
        },
    }

    filename = f"{name}_ref_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    path = os.path.join(output_dir, filename)
    return _safe_dump(output, path)


def sde_to_json(
    data: List[GraphDocument],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    name: str = "",
    timestamp: Optional[str] = None,
) -> Optional[str]:
    os.makedirs(output_dir, exist_ok=True)

    def node_to_dict(node: Node) -> dict:
        return {
            "id": node.id,
            "type": node.type,
            "properties": node.properties,
        }

    def relationship_to_dict(rel: Relationship) -> dict:
        return {
            "source": node_to_dict(rel.source),
            "target": node_to_dict(rel.target),
            "type": rel.type,
            "properties": rel.properties,
        }

    output: List[dict] = []
    skipped = 0
    for graph_doc in (data or []):
        if graph_doc is None:
            skipped += 1
            continue
        try:
            entry = {
                "source": graph_doc.source.page_content[:200] if graph_doc.source else None,
                "nodes": [node_to_dict(n) for n in (graph_doc.nodes or [])],
                "relationships": [relationship_to_dict(r) for r in (graph_doc.relationships or [])],
            }
            output.append(entry)
        except (AttributeError, TypeError) as e:
            skipped += 1
            print(f"[sde_to_json] skipping malformed GraphDocument: {e}")
            continue

    if skipped:
        print(f"[sde_to_json] skipped {skipped} malformed/None entries")

    ts = timestamp or _timestamp()
    filename = f"{name}_sde_{ts}.json"
    path = os.path.join(output_dir, filename)
    return _safe_dump(output, path)


def table_to_json(
    data: List[GraphDocument],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    name: str = "",
    timestamp: Optional[str] = None,
) -> Optional[str]:
    """Serialize per-table-group GraphDocuments produced by transform_html_to_graph_document."""
    os.makedirs(output_dir, exist_ok=True)

    output: List[dict] = []
    skipped = 0
    for graph_doc in (data or []):
        if graph_doc is None:
            skipped += 1
            continue
        try:
            meta = graph_doc.source.metadata if graph_doc.source else {}
            entry = {
                "page_range": meta.get("page_range"),
                "source_html": graph_doc.source.page_content if graph_doc.source else None,
                "nodes": [_node_to_dict(n) for n in (graph_doc.nodes or [])],
                "relationships": [_relationship_to_dict(r) for r in (graph_doc.relationships or [])],
            }
            output.append(entry)
        except (AttributeError, TypeError) as e:
            skipped += 1
            print(f"[table_to_json] skipping malformed GraphDocument: {e}")
            continue

    if skipped:
        print(f"[table_to_json] skipped {skipped} malformed/None entries")

    ts = timestamp or _timestamp()
    prefix = f"{name}_" if name else ""
    filename = f"{prefix}tables_graph_doc_{ts}.json"
    path = os.path.join(output_dir, filename)
    return _safe_dump(output, path)


def formula_to_json(
    data: Optional[GraphDocument],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    name: str = "",
    timestamp: Optional[str] = None,
) -> Optional[str]:
    """Serialize the single formulas GraphDocument produced by convert_formulas_to_graph."""
    os.makedirs(output_dir, exist_ok=True)

    if data is None:
        output: dict = {"document_id": None, "nodes": [], "relationships": []}
    else:
        try:
            meta = data.source.metadata if data.source else {}
            output = {
                "document_id": meta.get("id"),
                "nodes": [_node_to_dict(n) for n in (data.nodes or [])],
                "relationships": [_relationship_to_dict(r) for r in (data.relationships or [])],
            }
        except (AttributeError, TypeError) as e:
            print(f"[formula_to_json] malformed GraphDocument, writing empty payload: {e}")
            output = {"document_id": None, "nodes": [], "relationships": []}

    ts = timestamp or _timestamp()
    prefix = f"{name}_" if name else ""
    filename = f"{prefix}formula_graph_doc_{ts}.json"
    path = os.path.join(output_dir, filename)
    return _safe_dump(output, path)


def chunker_to_json(
    chunks: List[Document],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    timestamp: Optional[str] = None,
) -> Optional[str]:
    """Serialize Chunker output (list of Documents) as `chunks_<timestamp>.json`.

    Mirrors the shape previously emitted by `chunker.write_json`: each entry is
    `{id, text, ...metadata}` with `id` first for readability. Slovak diacritics
    are preserved (ensure_ascii=False).
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp or _timestamp()
    payload = [
        {
            "id": d.metadata.get("id"),
            "text": d.page_content,
            **{k: v for k, v in d.metadata.items() if k != "id"},
        }
        for d in (chunks or [])
    ]
    path = os.path.join(output_dir, f"chunks_{ts}.json")
    return _safe_dump(payload, path)


def sections_to_json(
    paragraphs: List[dict],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    timestamp: Optional[str] = None,
) -> Optional[str]:
    """Serialize `detect_subsections` output as `sections_<timestamp>.json`."""
    os.makedirs(output_dir, exist_ok=True)
    ts = timestamp or _timestamp()
    path = os.path.join(output_dir, f"sections_{ts}.json")
    return _safe_dump(paragraphs or [], path)
