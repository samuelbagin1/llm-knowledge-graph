"""
Example usage script for Legal Document GraphRAG

This script demonstrates how to use the system with a sample document.
"""

import os
from dotenv import load_dotenv
from legal_graphrag import LegalDocumentGraphRAG


def example_basic_usage():
    """Basic usage example"""
    print("="*70)
    print("EXAMPLE 1: Basic Document Processing")
    print("="*70 + "\n")
    
    load_dotenv()
    
    # Initialize system
    graphrag = LegalDocumentGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
    )
    
    # Process document
    pdf_path = "path/to/your/document.pdf"  # CHANGE THIS
    
    document_metadata = {
        "title": "Sample Contract",
        "date": "2024-01-15",
        "jurisdiction": "Federal",
        "doc_type": "Employment Agreement"
    }
    
    # Process first 3 pages for testing
    report = graphrag.process_legal_pdf(
        pdf_path=pdf_path,
        document_metadata=document_metadata,
        max_pages=3  # Remove for full document
    )
    
    return graphrag


def example_querying(graphrag):
    """Example queries"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Querying the Knowledge Graph")
    print("="*70 + "\n")
    
    queries = [
        "What definitions are in the document?",
        "What obligations are mentioned?",
        "What are the key dates?",
        "Who are the parties involved?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 70)
        response = graphrag.query_graph(query)
        print(response)
        print()


def example_custom_extraction():
    """Example with custom settings"""
    print("="*70)
    print("EXAMPLE 3: Custom Configuration")
    print("="*70 + "\n")
    
    load_dotenv()
    
    # Use Gemini instead of OpenAI
    graphrag = LegalDocumentGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        ai_provider="gemini",  # Use Google Gemini
        confidence_threshold=0.95  # Lower threshold
    )
    
    print("✓ Initialized with Gemini and 95% confidence threshold")
    
    return graphrag


def example_batch_processing():
    """Example of processing multiple documents"""
    print("="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70 + "\n")
    
    load_dotenv()
    
    graphrag = LegalDocumentGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
    )
    
    # List of documents to process
    documents = [
        {
            "path": "contracts/contract_001.pdf",
            "metadata": {"title": "Contract 001", "type": "NDA"}
        },
        {
            "path": "contracts/contract_002.pdf",
            "metadata": {"title": "Contract 002", "type": "Service Agreement"}
        },
    ]
    
    for doc in documents:
        print(f"\nProcessing: {doc['metadata']['title']}")
        print("-" * 70)
        
        try:
            report = graphrag.process_legal_pdf(
                pdf_path=doc["path"],
                document_metadata=doc["metadata"],
                max_pages=5  # Process only first 5 pages for demo
            )
            print(f"✓ Successfully processed {doc['metadata']['title']}")
        except Exception as e:
            print(f"✗ Error processing {doc['metadata']['title']}: {e}")


def example_cypher_queries():
    """Example of direct Cypher queries"""
    print("="*70)
    print("EXAMPLE 5: Direct Cypher Queries")
    print("="*70 + "\n")
    
    load_dotenv()
    
    graphrag = LegalDocumentGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
    )
    
    # Custom Cypher queries
    queries = {
        "All Definitions": """
            MATCH (d:Definition)
            RETURN d.term, d.definition, d.page
            LIMIT 5
        """,
        
        "Document Structure": """
            MATCH (s:Section)
            RETURN s.number, s.title, s.page
            ORDER BY s.number
            LIMIT 10
        """,
        
        "Citations by Type": """
            MATCH (c:Citation)
            RETURN c.type, count(*) as count
            ORDER BY count DESC
        """,
        
        "Cross-References": """
            MATCH (n)-[r:REFERENCES]->(m)
            RETURN labels(n)[0] as from_type, 
                   labels(m)[0] as to_type, 
                   count(*) as count
        """
    }
    
    for name, query in queries.items():
        print(f"\n📊 {name}")
        print("-" * 70)
        try:
            results = graphrag.graph.query(query)
            for result in results[:5]:  # Limit output
                print(result)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║         LEGAL DOCUMENT GRAPHRAG - USAGE EXAMPLES                   ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print("\n")
    
    # Choose which example to run
    print("Available examples:")
    print("1. Basic document processing")
    print("2. Querying the knowledge graph")
    print("3. Custom configuration")
    print("4. Batch processing")
    print("5. Direct Cypher queries")
    print("\nNote: Update the PDF paths in this script before running!")
    print("\nTo run specific examples, uncomment them below:\n")
    
    # Uncomment the examples you want to run:
    
    # graphrag = example_basic_usage()
    # example_querying(graphrag)
    # example_custom_extraction()
    # example_batch_processing()
    # example_cypher_queries()
    
    print("\n✓ Examples ready to run. Uncomment desired examples in the script.")
