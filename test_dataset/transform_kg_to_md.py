import json
import os
import re

directory = "./test_dataset/batch6"
output_path = "./test_dataset/batch6.md"

lines = []

for filename in sorted(os.listdir(directory)):
    if not filename.endswith(".json"):
        continue

    match = re.match(r"(\d{1,3})", filename)
    chunk_num = match.group(1) if match else filename

    with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        nodes = entry.get("nodes", [])
        relationships = entry.get("relationships", [])

        chunk_node = next((n for n in nodes if n.get("type") == "Chunk"), None)
        if not chunk_node:
            continue

        props = chunk_node.get("properties", {})
        page = props.get("page", "")
        text = props.get("text", "").replace("\n", " ")

        def fmt_rel(r):
            add = r.get("properties", {}).get("add", "")
            rel_type = f"{r['type']}:\"{add}\"" if add else r['type']
            return f"  {r['source']['id']} -> [{rel_type}] -> {r['target']['id']}"

        rel_lines = [fmt_rel(r) for r in relationships]
        node_lines = [
            f"  {n['type']}: {n['id']}"
            for n in nodes
            if n.get("type") != "Chunk"
        ]

        lines.append(f"chunk: {chunk_num}")
        lines.append(f"page: {page}")
        lines.append(f"text: {text}")
        lines.append("relationships:")
        lines.extend(rel_lines)
        lines.append("nodes:")
        lines.extend(node_lines)
        lines.append("")

with open(output_path, "a", encoding="utf-8") as f:
    f.write("\n".join(lines))

print(f"Appended {len([l for l in lines if l.startswith('chunk:')])} chunks to {output_path}")
