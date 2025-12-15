import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Set up Google API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up Neo4j credentials
os.environ["NEO4J_URI"] = os.getenv("NEO4J_URI", "bolt://localhost:7687")
os.environ["NEO4J_USERNAME"] = os.getenv("NEO4J_USERNAME", "neo4j")
os.environ["NEO4J_PASSWORD"] = os.getenv("NEO4J_PASSWORD")


# Initialize Google Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# Initialize graph transformer
graph_transformer = LLMGraphTransformer(llm=llm)


# Load PDF document
pdf_path = "code/romeo-juliet/pdf/romeo-and-juliet.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Optionally limit to first few pages for testing (remove if you want full document)
# documents = documents[:5]

print(f"Loaded {len(documents)} pages from PDF")


# Transform documents into graph documents
graph_docs = graph_transformer.convert_to_graph_documents(documents)

print(f"Generated {len(graph_docs)} graph documents")


# Connect to Neo4j and add graph documents
graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)

# Add graph documents to Neo4j
graph.add_graph_documents(graph_docs)

print("Knowledge graph successfully created in Neo4j!")
print(f"Added {sum(len(doc.nodes) for doc in graph_docs)} nodes")
print(f"Added {sum(len(doc.relationships) for doc in graph_docs)} relationships")