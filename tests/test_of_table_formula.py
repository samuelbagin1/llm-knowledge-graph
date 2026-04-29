import asyncio
import json
import os
import re
from pathlib import Path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from classes import Schema
from pdf_graphrag import PDFGraphRAG
from to_json import odd_to_json, refinement_to_json, sde_to_json, table_to_json, formula_to_json

from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship

load_dotenv()

pdf_path = './code/assets/ZZ_2004_222_20260101.pdf'
graphrag = PDFGraphRAG(
    neo4j_uri='neo4j://127.0.0.1:7687',
    neo4j_user='neo4j',
    neo4j_password='fseijkfbsj48@',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    database="factkg"
)


# Load PDF documents
documents = graphrag.load_pdf(pdf_path)

document_id = Path(pdf_path).stem



# detect tables in pdf (cached on disk to avoid re-running YOLO each test)
# detections = graphrag.detect_tables(pdf_path)


def load_cached_detections(detections_dir: str) -> list[dict]:
    """Reconstruct the list returned by detect_tables() from cached PNG crops.

    Filenames follow the pattern: page{N}_{class}_{idx}_conf{conf}.png
    where {class} may contain underscores (e.g. 'isolate_formula').

    Each dict must include the keys consumed downstream:
      - 'page' (int, 1-based)
      - 'class' (str)
      - 'confidence' (float)
      - 'image_path' (str, absolute or relative path to the PNG)
    'bbox' is not used by group_table_detections / transform_table_to_html
    in this test, so it can be omitted or set to None.
    """
    pattern = re.compile(r'^page(\d+)_(table)_(\d+)_conf([\d.]+)\.png$')
    detections = []
    for filename in sorted(os.listdir(detections_dir)):
        match = pattern.match(filename)
        if not match:
            continue
        page, cls, _, conf = match.groups()
        detections.append({
            "page": int(page),
            "class": cls,
            "confidence": float(conf),
            "image_path": os.path.join(detections_dir, filename),
        })
    return detections


detections = load_cached_detections('./code/assets/detected_tables_figures')

# group consecutive table detections into multi-page groups
table_groups = graphrag.group_table_detections(detections)

# convert each table group to HTML and then to GraphDocument
table_graph_docs = []
table_pages_to_exclude = set()

for group in table_groups:
    image_paths = [d["image_path"] for d in group]
    result = graphrag.transform_table_to_html(image_paths)
    if result is None:
        pages = sorted(d["page"] for d in group)
        print(f"Skipping table on pages {pages} — extraction returned None")
        continue
    print(f"Saved: {result['html_path']}")

    # access HTML from in-memory response
    response = result['response']
    html = response.get("html", "") if isinstance(response, dict) else response.html if hasattr(response, "html") else str(response)

    # compute page range for this group
    pages = sorted(d["page"] for d in group)
    page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0])

#     # transform HTML table into GraphDocument via LLM
#     table_gd = graphrag.transform_html_to_graph_document(html, page_range, document_id)
#     table_graph_docs.append(table_gd)

#     # collect interior table pages to exclude from text processing
#     # keep first and last pages (may have text above/below the table)
#     if len(pages) > 2:
#         table_pages_to_exclude.update(pages[1:-1])

# # persist intermediate results to JSON
# table_to_json(table_graph_docs, name=document_id)
    
    
# formulas = graphrag.get_formulas()

# formula_nodes: list[Node] = []
# for i, formula in enumerate(formulas):
#     response = graphrag.transform_formula(formula)
#     formula_nodes.append(
#         graphrag._convert_formula_to_node(formula, response, i, document_id)
#     )

# formula_graph_doc = graphrag.convert_formulas_to_graph(formula_nodes, document_id)
# print(f"Processed {len(formulas)} formula(s) into {0 if formula_graph_doc is None else len(formula_nodes)} node(s).")



# formula_to_json(formula_graph_doc, name=document_id)