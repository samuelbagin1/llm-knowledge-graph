import datetime
import json
import os
from typing import Dict, List, Any, Optional, Generic, TypeVar
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy


def serialize_for_json(obj):
    """Convert Neo4j and other non-serializable objects to JSON-serializable format."""
    # Handle Neo4j Node objects
    if hasattr(obj, 'labels') and hasattr(obj, 'items'):
        return {
            '_type': 'Node',
            'labels': list(obj.labels),
            'properties': dict(obj.items())
        }
    # Handle Neo4j Relationship objects
    if hasattr(obj, 'type') and hasattr(obj, 'start_node'):
        return {
            '_type': 'Relationship',
            'type': obj.type,
            'properties': dict(obj.items()) if hasattr(obj, 'items') else {}
        }
    # Handle Neo4j Path objects
    if hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
        return {
            '_type': 'Path',
            'nodes': [serialize_for_json(n) for n in obj.nodes],
            'relationships': [serialize_for_json(r) for r in obj.relationships]
    }
    # Handle datetime objects
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    # Handle dict-like objects
    if hasattr(obj, 'items') and not isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    # Handle lists recursively
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    # Handle primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Fallback to string representation
    return str(obj)

class PDFGraphRAG:
    
    # CONSTRUCTOR
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, openai_api_key: str, google_api_key: str = None):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )
        
        # Initialize LLM clients
        # ChatOpenAI for question generation
        self.openai_client = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            api_key=openai_api_key
        )
        
        self.openai_graph_transform = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )

        # Google Gemini for everything else
        self.gemini_client = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=google_api_key
        )

        self.graph_transformer = LLMGraphTransformer(llm=self.openai_graph_transform)
        

            
            
            

    # ----------------- METHODS -----------------
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
                
                
                
    def get_graph_schema(self):
        """Get Neo4j graph schema information with sample data
        
        Args:
            None
            
        Returns:
            Tuple
            schema, node_labels, rel_types
        
        """
        try:
            # Get node labels and relationship types
            node_labels = self.graph.query("CALL db.labels()")
            rel_types = self.graph.query("CALL db.relationshipTypes()")

            # Get sample nodes with properties to understand schema
            sample_nodes = self.graph.query("""
                MATCH (n)
                WITH labels(n)[0] as label, n
                RETURN label, properties(n) as props
                LIMIT 10
            """)

            # Get sample relationships
            sample_rels = self.graph.query("""
                MATCH (a)-[r]->(b)
                RETURN labels(a)[0] as from_label,
                        type(r) as rel_type,
                        labels(b)[0] as to_label,
                        properties(a) as from_props,
                        properties(b) as to_props
                LIMIT 10
            """)
            

            # Build comprehensive schema
            schema = "Node Types:\n"
            for label in node_labels:
                schema += f"  - {label['label']}\n"

            schema += "\nRelationship Types:\n"
            for rel in rel_types:
                schema += f"  - {rel['relationshipType']}\n"

            # Add sample data to understand property names
            schema += "\nSample Nodes (showing property structure):\n"
            for node in sample_nodes[:5]:
                schema += f"  - {node['label']}: {node['props']}\n"

            schema += "\nSample Relationships:\n"
            for rel in sample_rels[:5]:
                from_id = rel['from_props'].get('id', rel['from_props'].get('name', 'unknown'))
                to_id = rel['to_props'].get('id', rel['to_props'].get('name', 'unknown'))
                schema += f"  - ({rel['from_label']}: {from_id}) --[{rel['rel_type']}]--> ({rel['to_label']}: {to_id})\n"

            return schema, node_labels, rel_types

        except Exception as e:
            print(f"Error retrieve schema: {e}")
            return "Schema information unavailable", [], []
    
    


    # ---------------- PDF to Graph and Vector Processing
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
        self.graph.add_graph_documents(graph_documents=graph_docs, include_source=True)
        print(f"Added {sum(len(doc.nodes) for doc in graph_docs)} nodes")
        print(f"Added {sum(len(doc.relationships) for doc in graph_docs)} relationships")
        
        
        

    # ---------------- QUERYING METHODS ----------------
    
    def query_graph_database(self, question: str) -> Dict[str, Any]:
        """
        Function 2: Convert question to Cypher query and retrieve data from Neo4j

        Args:
            question: Test question to answer using the graph

        Returns:
            query_data, structured_answer.strip()
        """
        print(f"\n Querying graph database...")

        # First, check if database has any data
        node_count = self.graph.query("MATCH (n) RETURN count(n) as count")[0]['count']
        rel_count = self.graph.query("MATCH ()-[r]->() RETURN count(r) as count")[0]['count']

        if node_count == 0:
            print(f" Database is EMPTY")

        # Get schema information
        schema_info, node_labels, rel_types = self.get_graph_schema()

        # Build node labels and relationship types lists for the prompt
        node_labels_list = [node['label'] for node in node_labels]
        rel_types_list = [rel['relationshipType'] for rel in rel_types]

        # System prompt - defines the agent's role and capabilities
        system_prompt = """You are a Neo4j Cypher expert agent specialized in querying knowledge graphs.

Your task is to answer questions by querying a Neo4j graph database about Romeo and Juliet.

## Your Capabilities
You have access to the `search_database` tool which executes Cypher queries against Neo4j.

## Query Strategy
1. **Analyze the question** to identify what nodes and relationships are relevant
2. **Start with exploration queries** to understand what data exists:
   - For nodes: `MATCH (n:Label) RETURN n.id, labels(n)`
   - For relationships: `MATCH (a)-[r:TYPE]->(b) RETURN a.id, type(r), b.id`
3. **Refine iteratively** - use results from initial queries to build more specific queries
4. **Find the best matches** - keep querying until you find the most relevant data

## Cypher Query Rules
- Use backticks for labels/types with special characters: `MATCH (n:`Special-Label`) ...`
- For text matching use case-insensitive: `WHERE toLower(n.id) CONTAINS toLower('romeo')`
- Use undirected relationships `-[r]-` when direction is unknown
- Always add `LIMIT 25` to prevent large result sets
- Return useful properties: `RETURN n.id, labels(n), type(r), properties(n)`

## Important
- You MUST use the search_database tool to query the database
- Make multiple queries if needed to find the best answer
- When you have found sufficient data, provide your final answer with the best Cypher query"""

        # User prompt - provides the specific question and schema
        user_prompt = f"""Answer this question about Romeo and Juliet by querying the graph database:

**Question:** {question}

## Available Graph Schema

### Node Labels (use these exact labels in queries):
{json.dumps(node_labels_list, indent=2)}

### Relationship Types (use these exact types in queries):
{json.dumps(rel_types_list, indent=2)}

### Sample Data (shows actual property structure):
{schema_info}

## Your Task
1. Analyze the question to determine which node labels and relationship types are relevant
2. Use the `search_database` tool to query the database with Cypher queries
3. Start broad, then refine based on results
4. Continue querying until you find the best matching nodes, properties, and relationships
5. Return your final answer with the most effective Cypher query and the data you found

Begin by identifying the relevant node labels and relationship types, then query the database."""

        # Response schema
        response_schema = {
            "title": "GraphQueryResult",
            "type": "object",
            "description": "Final query results from graph database exploration",
            "properties": {
                "cypher_query": {
                    "type": "string",
                    "description": "The final/best Cypher query that answers the question"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the query strategy and what was found"
                },
                "data": {
                    "type": "string",
                    "description": "The relevant data returned from the database (JSON string)"
                },
                "nodes_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node IDs that are relevant to the answer"
                },
                "relationships_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relationships found (format: 'nodeA -[REL_TYPE]-> nodeB')"
                }
            },
            "required": ["cypher_query", "explanation", "data"]
        }


        @tool
        def search_database(cypher_query: str) -> str:
            """Execute a Cypher query against the Neo4j graph database.

            Args:
                cypher_query: A valid Cypher query string to execute

            Returns:
                JSON string of query results, or error message if query fails
            """
            def serialize_neo4j_object(obj):
                """Convert Neo4j objects to JSON-serializable format."""
                # Handle Neo4j Node objects
                if hasattr(obj, 'labels') and hasattr(obj, 'items'):
                    return {
                        '_type': 'Node',
                        'labels': list(obj.labels),
                        'properties': dict(obj.items())
                    }
                # Handle Neo4j Relationship objects
                if hasattr(obj, 'type') and hasattr(obj, 'start_node'):
                    return {
                        '_type': 'Relationship',
                        'type': obj.type,
                        'properties': dict(obj.items()) if hasattr(obj, 'items') else {}
                    }
                # Handle Neo4j Path objects
                if hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
                    return {
                        '_type': 'Path',
                        'nodes': [serialize_neo4j_object(n) for n in obj.nodes],
                        'relationships': [serialize_neo4j_object(r) for r in obj.relationships]
                    }
                # Handle dict-like objects
                if hasattr(obj, 'items'):
                    return {k: serialize_neo4j_object(v) for k, v in obj.items()}
                # Handle lists
                if isinstance(obj, list):
                    return [serialize_neo4j_object(item) for item in obj]
                # Handle primitives
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                # Fallback to string representation
                return str(obj)

            try:
                result = self.graph.query(cypher_query)
                records = [dict(record) for record in result]

                serialized = []
                for record in records:
                    serialized_record = {}
                    for key, value in record.items():
                        try:
                            serialized_record[key] = serialize_neo4j_object(value)
                        except Exception as e:
                            serialized_record[key] = f"<serialization error: {str(e)}>"
                    serialized.append(serialized_record)

                return json.dumps(serialized, indent=2, default=str)

            except Exception as e:
                return f"Query error: {str(e)}"



        # Create and run the agent
        agent = create_agent(
            model=self.openai_client,
            tools=[search_database],
            response_format=ProviderStrategy(schema=response_schema),
            system_prompt=system_prompt
        )
        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

        # structured_response is already a dict when using ProviderStrategy
        query_data = response["structured_response"]

        try:
            cypher_query = query_data['cypher_query']
            # data field might be a JSON string or already parsed
            data_field = query_data.get('data', '[]')
            records = json.loads(data_field) if isinstance(data_field, str) else (data_field or [])

            print(f"Generated Cypher query: {cypher_query}")
            print(f"Found {len(records) if isinstance(records, list) else 'N/A'} results")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse query response: {e}")
            cypher_query = "MATCH (n) RETURN n LIMIT 10"
            records = []

        # Fallback if no results
        if not records and node_count > 0:
            try:
                fallback_query = "MATCH (n)-[r]-(m) RETURN n.id as node1, type(r) as rel_type, m.id as node2 LIMIT 50"
                result = self.graph.query(fallback_query)
                records = [dict(record) for record in result]
                if records:
                    print(f"Fallback query returned {len(records)} results")
            except Exception as e:
                print(f"Fallback query failed: {e}")
                records = []



        # Format results into natural language answer
        format_prompt = f"""Based on the following graph database query results, provide a clear, concise answer to the original question.

Question: {question}

Query Results:
{json.dumps(records, indent=2) if records else "No results found"}

Provide a natural language answer that:
1. Directly answers the question
2. Includes specific names, relationships, and details from the results
3. Acknowledges if information is missing or incomplete
4. Is clear and concise (2-4 sentences)

Return ONLY the answer text, no preamble or JSON formatting."""


        structured_answer = self.openai_client.invoke(format_prompt).content

        print(f"Answer: {structured_answer}")

        return {"graph_data": query_data, "structured_answer": structured_answer.strip()}

    
    
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
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# processing
graphrag.process_pdf("code/romeo-juliet/pdf/romeo-and-juliet.pdf")
print("\nKnowledge graph successfully created in Neo4j!")
