import asyncio
import json
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from classes import Schema
from pdf_graphrag import PDFGraphRAG
from to_json import odd_to_json, refinement_to_json

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

# ------ open domain detection ------

# Load PDF documents
# documents = graphrag.load_pdf(pdf_path)
# documents = documents[:60]
    


# splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
# chunked_documents = splitter.split_documents(documents)

# extracted_schema_list = asyncio.run(
#     graphrag.async_open_domain_detection(
#         chunked_documents,
#     )
# )
# print(f"\nAll chunks processed into graph documents.")


# odd_to_json(extracted_schema_list)




# ------ schema refinement ------

with open('./extracted_data/schemas_20260312002414.json') as f:
    data = json.load(f)

extracted_schema = Schema(
    nodes=data[1]['nested']['nodes'],
    relationships=data[1]['nested']['relationships']
)
refined_schema = graphrag.schema_refinement(odd_schema=extracted_schema)

refinement_to_json(refined_schema)




# ------ schema driven extraction ------
