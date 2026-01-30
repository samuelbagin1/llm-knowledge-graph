import os
from dotenv import load_dotenv
from pdf_graphrag import PDFGraphRAG

load_dotenv()

graphrag = PDFGraphRAG(
    neo4j_uri='neo4j://127.0.0.1:7687',
    neo4j_user='neo4j',
    neo4j_password='fseijkfbsj48@',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    claude_api_key=os.getenv("ANTHROPIC_API_KEY"),
    vector_store_chunk_name='pdf_romeo_juliet',
    vector_store_nodes_name='pdf_romeo_juliet_nodes',
    vector_store_relationships_name='pdf_romeo_juliet_relationships'
)

# processing
graphrag.process("datasets/vyhlasene_znenie_4480183-2.pdf")
print("\nKnowledge graph successfully created in Neo4j!")