import os
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

class PDFGraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, ai: str = None):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )

        self.llm = self.getLlm(ai=ai)
        self.graph_transformer = LLMGraphTransformer(llm=self.llm)
        
    
    def getLlm(ai: str = None):
        if ai == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
    
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))


    def load_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    def process_pdf(self, pdf_path: str, max_pages: int = None):
        
        # Load PDF documents
        documents = self.load_pdf(pdf_path)
        if max_pages:
            documents = documents[:max_pages]
        print(f"Loaded {len(documents)} pages from PDF")


        # Transform documents into graph documents using LLMGraphTransformer
        graph_docs = self.graph_transformer.convert_to_graph_documents(documents)
        print(f"Generated {len(graph_docs)} graph documents")


        # Add graph documents to Neo4j
        # dependency: APOC plugin in neo4j database
        self.graph.add_graph_documents(graph_docs)
        print(f"Added {sum(len(doc.nodes) for doc in graph_docs)} nodes")
        print(f"Added {sum(len(doc.relationships) for doc in graph_docs)} relationships")
        
        
        
    def add_graph_docs_without_apoc(self, graph_docs):
        """Add graph documents without using APOC procedures"""
        
        for doc in graph_docs:
            # Add nodes
            for node in doc.nodes:
                # Create node with MERGE to avoid duplicates
                query = f"""
                MERGE (n:{node.type} {{id: $id}})
                SET n += $properties
                """
                self.graph.query(query, {
                    "id": node.id,
                    "properties": node.properties or {}
                })
            
            # Add relationships
            for rel in doc.relationships:
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel.type}]->(target)
                SET r += $properties
                """
                self.graph.query(query, {
                    "source_id": rel.source.id,
                    "target_id": rel.target.id,
                    "properties": rel.properties or {}
                })
    
    
    # TODO: while processing the pdf to graph database, create a node that will represent the chunk,
    #       and every created node from the chunk`s text, connect nodes to chunk node via HAS relationship
    
    # TODO: implement method for cypher querying of graph database
    # TODO: implemet method for vector query of graph database, then return chunks nodes
    # TODO: implement method for vector search inside vector database



# ------- MAIN ---------

load_dotenv()

graphrag = PDFGraphRAG(
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_user=os.getenv("NEO4J_USER"),
    neo4j_password=os.getenv("NEO4J_PASSWORD")
)

# processing
graphrag.process_pdf("code/romeo-juliet/pdf/romeo-and-juliet.pdf")
print("\nKnowledge graph successfully created in Neo4j!")
