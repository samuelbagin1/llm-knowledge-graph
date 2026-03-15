import json
import os
from datetime import datetime
from classes import Schema
from typing import List
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


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
        "node_types": data['node_types'],
        "relationship_types": data['relationship_types'],
        "merge_log": {
            "node_types": data['merge_log']['node_types'],
            "relationship_types": data['merge_log']['relationship_types'],
        },
    }

    name = f"refinement_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    with open(os.path.join(output_dir, name), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
        

def sde_to_json(data: List[GraphDocument], output_dir: str = "./extracted_data"):
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

    output = []
    for graph_doc in data:
        entry = {
            "source": graph_doc.source.page_content[:200] if graph_doc.source else None,
            "nodes": [node_to_dict(n) for n in graph_doc.nodes],
            "relationships": [relationship_to_dict(r) for r in graph_doc.relationships],
        }
        output.append(entry)

    name = f"sde_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

    with open(os.path.join(output_dir, name), "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
