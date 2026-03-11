import json
import os
from datetime import datetime
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
        
    output = []
    output.append({"chunked": schemas})

    merged = {
        "nodes": list(set(all_nodes)),
        "relationships": list(set(all_relationships)),
    }
    output.append({"nested": merged})

    name = f"schemas_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    with open(os.path.join(output_dir, name), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)  # ensure_ascii=False
        
        
        
def refinement_to_json(data, output_dir: str = "./extracted_data"):
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "node_types": data.node_types,
        "relationship_types": data.relationship_types,
        "merge_log": {
            "node_types": data.merge_log.node_types,
            "relationship_types": data.merge_log.relationship_types,
        },
    }

    name = f"refinement_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    with open(os.path.join(output_dir, name), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
