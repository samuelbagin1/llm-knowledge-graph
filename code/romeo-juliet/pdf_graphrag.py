import os
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


class PDFGraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, google_api_key: str = None):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )

        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=google_api_key or os.getenv("GOOGLE_API_KEY"),
            client_type="rest"
        )

        self.graph_transformer = LLMGraphTransformer(llm=self.llm)


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
        self.graph.add_graph_documents(graph_docs)

        print(f"Added {sum(len(doc.nodes) for doc in graph_docs)} nodes")
        print(f"Added {sum(len(doc.relationships) for doc in graph_docs)} relationships")


def main():
    load_dotenv()

    graphrag = PDFGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Process PDF - optionally limit pages for testing with max_pages parameter
    graphrag.process_pdf("code/romeo-juliet/pdf/romeo-and-juliet.pdf")

    print("\nKnowledge graph successfully created in Neo4j!")


if __name__ == "__main__":
    main()
