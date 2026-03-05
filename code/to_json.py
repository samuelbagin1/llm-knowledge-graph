import json
import os
from classes import Schema
from typing import List


def odd_to_json(documents: List[Schema], output_dir: str = "./extracted_data"):
    os.makedirs(output_dir, exist_ok=True)

    schemas = []
    all_nodes = []
    all_relationships = []

    for doc in documents:
        entry = {"nodes": doc.nodes, "relationships": doc.relationships}
        schemas.append(entry)
        all_nodes.extend(doc.nodes)
        all_relationships.extend(doc.relationships)

    merged = {
        "nodes": list(set(all_nodes)),
        "relationships": list(set(all_relationships)),
    }
    schemas.append(merged)

    with open(os.path.join(output_dir, "schemas.json"), "w") as f:
        json.dump(schemas, f, indent=2)