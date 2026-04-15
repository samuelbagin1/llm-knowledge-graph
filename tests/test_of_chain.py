import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from classes import Schema
from pdf_graphrag import PDFGraphRAG
from to_json import odd_to_json, refinement_to_json, sde_to_json

load_dotenv()

pdf_path = './code/assets/ZZ_2004_222_20260101.pdf'
graphrag = PDFGraphRAG(
    neo4j_uri='neo4j://127.0.0.1:7687',
    neo4j_user='neo4j',
    neo4j_password='fseijkfbsj48@',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    claude_api_key=os.getenv("ANTHROPIC_API_KEY")
)

name = "three-chunks-example"

# Load PDF documents
documents = graphrag.load_pdf(pdf_path)
documents = documents[58:60]

# ------ open domain detection ------

splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
chunked_documents = splitter.split_documents(documents)

chunked_documents = chunked_documents[:3]

extracted_schema_list = asyncio.run(
    graphrag.async_open_domain_detection(
        chunked_documents,
    )
)

print(f"\nAll chunks processed into list of schema.")
odd_to_json(extracted_schema_list, name = name, chunks = chunked_documents)




# ------ schema refinement ------

# with open('./extracted_data/schemas_20260312002414.json') as f:
#     data = json.load(f)

# extracted_schema = Schema(
#     nodes=data[1]['nested']['nodes'],
#     relationships=data[1]['nested']['relationships']
# )

merged_schema = Schema(
    nodes=[n for s in extracted_schema_list for n in s.nodes],
    relationships=[r for s in extracted_schema_list for r in s.relationships]
)
refined_schema_data = graphrag.schema_refinement(odd_schema=merged_schema)

print(f"\nRefined schema.")
refinement_to_json(refined_schema_data, name = name)




# ------ schema driven extraction ------
# splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
# chunked_documents = splitter.split_documents(documents)

# with open('./extracted_data/refinement_20260312235044.json') as f:
#     data = json.load(f)

# refined_schema = Schema(
#     nodes=data['node_types'],
#     relationships=data['relationship_types']
# )

refined_schema = Schema(
    nodes=refined_schema_data['node_types'],  # type: ignore[index]
    relationships=refined_schema_data['relationship_types']  # type: ignore[index]
)

graph_docs = asyncio.run(
    graphrag.async_schema_driven_extraction(
        chunked_documents,
        schema=refined_schema
    )
)

print(f"\nAll chunks processed into graph documents.")
sde_to_json(graph_docs, name = name)