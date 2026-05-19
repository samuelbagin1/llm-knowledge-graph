import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from dotenv import load_dotenv
from classes import Schema
from pdf_graphrag import PDFGraphRAG
from chunker.chunker import Chunker
from to_json import odd_to_json, refinement_to_json, sde_to_json

load_dotenv()

pdf_path = './code/assets/ZZ_2004_222_20260101.pdf'
graphrag = PDFGraphRAG(
    neo4j_uri='neo4j://127.0.0.1:7687',
    neo4j_user='neo4j',
    neo4j_password='fseijkfbsj48@',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    database="zz-2004-222"
)



# graphrag.process(pdf_path, "zz2004222", True)

docs = graphrag.load_pdf(pdf_path=pdf_path)

doc_id = graphrag.get_document_id(pdf_path)

table_graph_docs, formula_graph_docs, last_page = graphrag.tables_and_formulas(pdf_path=pdf_path, document_id=doc_id, documents=docs, write_json=True)

graphrag.merge_new(table_graph_docs)