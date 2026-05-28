"""Render kg_dataset-linearized.json as a Markdown report.

Each entry becomes a block of sections: chunk, path, path_as_text, text,
relations, id. Sections (and entries) are separated by `---`.
"""

import json
from pathlib import Path

INPUT_PATH = Path("./test_dataset/kg_dataset-linearized.json")
OUTPUT_PATH = Path("./test_dataset/batchNew.md")


def build_legal_context(path_segments: list[str]) -> tuple[str, str]:
    """Return (path_str, path_label) following the canonical legal ID format."""
    if not path_segments:
        return "", ""
    path_str = "Paragraf " + path_segments[0]
    path_label = "Paragraf"
    if len(path_segments) > 1:
        path_str += " Odsek " + path_segments[1]
        path_label = "Odsek"
    if len(path_segments) > 2:
        path_str += " Pismeno " + path_segments[2]
        path_label = "Pismeno"
    return path_str, path_label


def format_relation(rel: dict) -> str:
    src = rel["source"]["id"]
    tgt = rel["target"]["id"]
    return f"  {src} -> [{rel['type']}] -> {tgt}"


def format_node(node: dict) -> str:
    return f"  {node['type']}: {node['id']}"


def render_entry(entry: dict) -> str:
    path_segments = entry.get("path", [])
    path_as_text, _ = build_legal_context(path_segments)

    relations = entry.get("relationships", [])
    nodes = entry.get("nodes", [])
    text = entry.get("text", "").replace("\n", " ")

    sections = [
        f"chunk: {entry.get('chunk')}",
        f"path: {path_segments}",
        f"path_as_text: {path_as_text}",
        f"text: {text}",
        "\nrelations:\n" + "\n".join(format_relation(r) for r in relations),
        "\nnodes:\n" + "\n".join(format_node(n) for n in nodes),
    ]
    return "\n".join(sections)


def main() -> None:
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    blocks = [render_entry(entry) for entry in data]
    OUTPUT_PATH.write_text("\n\n---\n\n".join(blocks) + "\n\n---\n\n", encoding="utf-8")
    print(f"Wrote {len(blocks)} entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
