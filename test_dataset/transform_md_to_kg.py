import argparse
import json
import re
import sys
from pathlib import Path


DOCUMENT_ID = "ZZ_222_2004"
RELATIONSHIP_RE = re.compile(
    r"^\s*(?P<source>.+?)\s*->\s*\[(?P<relationship>.+?)\]\s*->\s*(?P<target>.+?)\s*$"
)
RELATIONSHIP_WITH_ADD_RE = re.compile(
    r'^(?P<type>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*"(?P<add>.*)"$'
)
SIMPLE_PISMENO_RE = re.compile(r"^Pismeno\s+([A-Za-z])\)?$", re.IGNORECASE)


def as_int_if_possible(value):
    value = value.strip()
    try:
        return int(value)
    except ValueError:
        return value


def node_name(node_id, node_type):
    simple_pismeno = SIMPLE_PISMENO_RE.match(node_id)
    if node_type == "Pismeno" and simple_pismeno:
        return simple_pismeno.group(1).lower()
    return node_id.lower()


def make_node(node_id, node_type):
    return {
        "id": node_id,
        "type": node_type,
        "properties": {
        },
    }


def parse_relationship_line(line, line_number, strict=False):
    match = RELATIONSHIP_RE.match(line)
    if not match:
        if strict:
            raise ValueError(f"Invalid relationship on line {line_number}: {line}")
        return None

    relationship = match.group("relationship").strip()
    rel_type = relationship
    add = ""

    add_match = RELATIONSHIP_WITH_ADD_RE.match(relationship)
    if add_match:
        rel_type = add_match.group("type").strip()
        add = add_match.group("add").strip()

    return {
        "source": match.group("source").strip(),
        "target": match.group("target").strip(),
        "type": rel_type,
    }


def parse_node_line(line, line_number, strict=False):
    stripped = line.strip()
    if ":" not in stripped:
        if strict:
            raise ValueError(f"Invalid node on line {line_number}: {line}")
        return None

    node_type, node_id = stripped.split(":", 1)
    node_type = node_type.strip()
    node_id = node_id.strip()
    if not node_type or not node_id:
        if strict:
            raise ValueError(f"Invalid node on line {line_number}: {line}")
        return None

    return {"type": node_type, "id": node_id}


def build_graph_document(raw_document, document_id):
    nodes = []
    node_by_id = {}
    seen_nodes = set()

    for raw_node in raw_document["nodes"]:
        node_id = raw_node["id"]
        node_type = raw_node["type"]
        node_key = (node_id, node_type)
        if node_key in seen_nodes:
            continue

        node = make_node(node_id, node_type)
        nodes.append(node)
        seen_nodes.add(node_key)
        node_by_id.setdefault(node_id, node)

    relationships = []
    for raw_relationship in raw_document["relationships"]:
        source_id = raw_relationship["source"]
        target_id = raw_relationship["target"]

        if source_id not in node_by_id:
            node = make_node(source_id, "Unknown")
            nodes.append(node)
            node_by_id[source_id] = node

        if target_id not in node_by_id:
            node = make_node(target_id, "Unknown")
            nodes.append(node)
            node_by_id[target_id] = node

        relationships.append(
            {
                "source": node_by_id[source_id],
                "target": node_by_id[target_id],
                "type": raw_relationship["type"],
                "properties": {
                },
            }
        )

    return {
        "chunk": raw_document["chunk"],
        "page": raw_document["page"],
        "text": raw_document["text"].strip(),
        "nodes": nodes,
        "relationships": relationships,
    }


def parse_markdown(markdown, document_id=DOCUMENT_ID, strict=False):
    raw_documents = []
    current = None
    section = None

    def finish_current():
        if current is not None:
            raw_documents.append(current)

    for line_number, line in enumerate(markdown.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped.startswith("chunk:"):
            finish_current()
            current = {
                "chunk": as_int_if_possible(stripped.split(":", 1)[1]),
                "page": None,
                "text": "",
                "relationships": [],
                "nodes": [],
            }
            section = None
            continue

        if current is None:
            continue

        if stripped.startswith("page:"):
            current["page"] = as_int_if_possible(stripped.split(":", 1)[1])
            section = None
            continue

        if stripped.startswith("text:"):
            current["text"] = stripped.split(":", 1)[1].strip()
            section = "text"
            continue

        if stripped == "relationships:":
            section = "relationships"
            continue

        if stripped == "nodes:":
            section = "nodes"
            continue

        if section == "text":
            current["text"] = " ".join(part for part in [current["text"], stripped] if part)
            continue

        if section == "relationships":
            relationship = parse_relationship_line(line, line_number, strict=strict)
            if relationship is not None:
                current["relationships"].append(relationship)
            continue

        if section == "nodes":
            node = parse_node_line(line, line_number, strict=strict)
            if node is not None:
                current["nodes"].append(node)

    finish_current()
    return [build_graph_document(raw_document, document_id) for raw_document in raw_documents]


def parse_markdown_file(input_path, document_id=DOCUMENT_ID, strict=False):
    markdown = Path(input_path).read_text(encoding="utf-8")
    return parse_markdown(markdown, document_id=document_id, strict=strict)


def load_existing_graph_documents(output_path):
    output_path = Path(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        return []

    with output_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Output file must contain a JSON array: {output_path}")

    return data


def write_json(graph_documents, output_path, indent=2, append=True):
    existing_graph_documents = load_existing_graph_documents(output_path) if append else []
    all_graph_documents = existing_graph_documents + graph_documents

    Path(output_path).write_text(
        json.dumps(all_graph_documents, ensure_ascii=False, indent=indent) + "\n",
        encoding="utf-8",
    )
    return len(existing_graph_documents), len(all_graph_documents)


def build_arg_parser():
    default_input = Path(__file__).with_name("batch5.md")
    default_output = Path(__file__).with_name("kg_dataset.json")

    parser = argparse.ArgumentParser(
        description="Transform a markdown knowledge-graph batch into GraphDocument JSON."
    )
    parser.add_argument(
        "-i",
        "--input",
        default=default_input,
        help=f"Input markdown file. Defaults to {default_input}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help=f"Output JSON file. Defaults to {default_output}.",
    )
    parser.add_argument(
        "--document-id",
        default=DOCUMENT_ID,
        help=f"Relationship document_id value. Defaults to {DOCUMENT_ID}.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Defaults to 2.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print JSON to stdout instead of writing the output file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output JSON array instead of appending to it.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on malformed relationship or node lines.",
    )
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    graph_documents = parse_markdown_file(
        args.input,
        document_id=args.document_id,
        strict=args.strict,
    )

    if args.stdout:
        json.dump(graph_documents, sys.stdout, ensure_ascii=False, indent=args.indent)
        sys.stdout.write("\n")
        return 0

    previous_count, total_count = write_json(
        graph_documents,
        args.output,
        indent=args.indent,
        append=not args.overwrite,
    )
    action = "Appended" if not args.overwrite else "Wrote"
    print(
        f"{action} {len(graph_documents)} GraphDocuments to {args.output} "
        f"({total_count} total, {previous_count} existing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
