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
    claude_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# processing
graphrag.process("./code/assets/ZZ_2004_222_20260101.pdf")
print("\nKnowledge graph successfully created in Neo4j!")