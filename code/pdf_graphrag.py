import datetime
import json
import os
from typing import Dict, List, Any, Optional, Generic, TypeVar
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from openai import embeddings
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, SpacyTextSplitter
import spacy

from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


# extract nodes and relationships from spacy doc
def spacy_to_graph_document(doc, source_document):
    nodes = {}
    relationships = []
    
    # Extract entities as nodes
    for ent in doc.ents:
        nodes[ent.text] = Node(id=ent.text, type=ent.label_)
    
    # Extract SVO triples from dependency parse
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            subj = [w for w in token.children if w.dep_ in ("nsubj", "nsubjpass")]
            obj = [w for w in token.children if w.dep_ in ("dobj", "pobj", "attr")]
            
            for s in subj:
                for o in obj:
                    if s.text in nodes and o.text in nodes:
                        relationships.append(Relationship(
                            source=nodes[s.text],
                            target=nodes[o.text],
                            type=token.lemma_.upper()
                        ))
    
    return GraphDocument(
        nodes=list(nodes.values()),
        relationships=relationships,
        source=source_document
    )


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
    def __init__(self, vector_store_chunk_name: str, vector_store_nodes_name: str, vector_store_relationships_name: str, 
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 openai_api_key: str = None, google_api_key: str = None, 
                 claude_api_key: str = None, advanced_search: bool = False):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )

        # Initialize embeddings first - needed for vector stores
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large', api_key=openai_api_key)

        # Store vector store configuration for lazy initialization
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._vector_store_chunk_name = vector_store_chunk_name
        self._vector_store_nodes_name = vector_store_nodes_name
        self._vector_store_relationships_name = vector_store_relationships_name
        self._advanced_search = advanced_search

        # Initialize vector stores - will be created when first documents are added
        self._init_vector_stores()
        
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
        
        # use claude-sonnet-4-5
        self.claude_client = ChatAnthropic(
            model="claude-haiku-4-5",
            temperature=0,
            api_key=claude_api_key
        )

        # Google Gemini for everything else
        self.gemini_client = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=google_api_key
        )

        self.graph_transformer = LLMGraphTransformer(llm=self.claude_client)

    def _init_vector_stores(self):
        """Initialize vector stores, creating empty ones if indices don't exist"""
        try:
            # Try to load existing indices
            self.vector_store_nodes = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self._neo4j_uri,
                username=self._neo4j_user,
                password=self._neo4j_password,
                index_name=self._vector_store_nodes_name,
            )
            self.vector_store_relationships = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self._neo4j_uri,
                username=self._neo4j_user,
                password=self._neo4j_password,
                index_name=self._vector_store_relationships_name,
            )
            print("Loaded existing vector store indices")
        except ValueError:
            # Indices don't exist yet - set to None and they'll be created on first use
            print("Vector store indices not found - will be created when documents are added")
            self.vector_store_chunk = None
            self.vector_store_nodes = None
            self.vector_store_relationships = None

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
        
    # TODO
    def get_help():
        help_text = """
        Help Instructions:
        
        - To process a PDF and create the knowledge graph, use:
            graphrag.process_pdf("path/to/your.pdf")
        
        - To query the graph database, use:
            graphrag.query_graph_database("Your question here")
        
        - To query the vector database for nodes or relationships, use:
            graphrag.query_vector_database(database=graphrag.vector_store_nodes, question="Your question here")
            graphrag.query_vector_database(database=graphrag.vector_store_relationships, question="Your question here")
        
        - To view the graph schema, use:
            graphrag.get_graph_schema()"""
    
    


    # ---------------- PDF to Graph and Vector Processing
    def load_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        print("PDF loaded successfully.")
        return loader.load()

    # send to claude api the file and then process
    def process_pdf(self, pdf_path: str, max_pages: int = None, allowed_entities: Optional[List[str]] = None):
        
        # Load PDF documents
        documents = self.load_pdf(pdf_path)
        if max_pages:
            documents = documents[:max_pages]

        # TODO: study and implement better chunking strategy
        # TODO: study and implement better NER and relationship extraction
        # TODO: research spaCy
        
        splitter = SpacyTextSplitter()
        chunked_documents = splitter.split_documents(documents)
        # nlp = spacy.load("en_core_web_sm")

        all_graph_docs = []
        all_nodes = []
        for i, document in enumerate(chunked_documents):
            print(f"Processing chunk {i+1}/{len(chunked_documents)}")
            chunk_id = f"chunk_{i}"
            
            chunk_embedding = self.embeddings.embed_query(document.page_content)
            chunk_node = Node(
                id=chunk_id,
                type="Chunk",
                properties={
                    "text": document.page_content,
                    "embedding": chunk_embedding,
                    "page": document.metadata.get("page", 0)
                }
            )
            
            # spacy NLP and NER 
            # doc = nlp(document.page_content)
            # entities = [ent.text for ent in doc.ents]
            
            # Transform documents into graph documents using LLMGraphTransformer
            graph_docs = self.graph_transformer.convert_to_graph_documents([document])
            
            chunk_relationships = []
            # forEach graph doc, add Chunk node and HAS relationship
            for graph_doc in graph_docs:
                for node in graph_doc.nodes:
                    all_nodes.append(Document(page_content=node.id))
                    chunk_relationships.append(
                        Relationship(
                            source=chunk_node,
                            target=node,
                            type="HAS"
                        )
                    )
                        
                all_graph_docs.append(graph_doc)
            
            chunk_graph_doc = GraphDocument(
                nodes=[chunk_node],
                relationships=chunk_relationships,
                source=document
            )
            all_graph_docs.append(chunk_graph_doc)
        print("\nAll chunks processed into graph documents.")
        # ------------------ END OF LOOP ------------------
        
        # Add graph documents to Neo4j
        # dependency: APOC plugin in neo4j database
        self.graph.add_graph_documents(
            graph_documents=all_graph_docs,
            include_source=False,
            baseEntityLabel=True
        )
        
        self.graph.refresh_schema()
            
        
        rel_types = self.graph.query("CALL db.relationshipTypes()")
        all_relationships = [Document(page_content=rel['relationshipType']) for rel in rel_types]

        # Initialize vector stores if they don't exist yet
        if self.vector_store_nodes is None:
            print("Creating vector store indices...")
            self.vector_store_nodes = Neo4jVector.from_documents(
                all_nodes,
                embedding=self.embeddings,
                url=self._neo4j_uri,
                username=self._neo4j_user,
                password=self._neo4j_password,
                index_name=self._vector_store_nodes_name,
            )
        else:
            self.vector_store_nodes.add_documents(all_nodes)

        if self.vector_store_relationships is None:
            self.vector_store_relationships = Neo4jVector.from_documents(
                all_relationships,
                embedding=self.embeddings,
                url=self._neo4j_uri,
                username=self._neo4j_user,
                password=self._neo4j_password,
                index_name=self._vector_store_relationships_name,
            )
        else:
            self.vector_store_relationships.add_documents(all_relationships)
        
        print("Knowledge graph and vector stores successfully updated in Neo4j!")
            
            
        
        
        
        

    # ---------------- QUERYING METHODS ----------------
    
    def query_graph_database(self, question: str, similar_nodes: str, similar_relationships: str, svo: Dict) -> Dict[str, Any]:
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

Your task is to answer questions by querying a Neo4j graph database.

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
- Keep queries efficient, focused and short

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
{self.graph.schema()}

### Similar Nodes (based on question context):
{similar_nodes}

### Similar Relationships (based on question context):
{similar_relationships}

### Subject-Verb-Object from question:
{json.dumps(svo, indent=2)}

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
            "required": ["cypher_query", "explanation", "nodes_found", "relationships_found"]
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

        return query_data
    
    
    
    def query_vector_database(self, database: Neo4jVector, question: str, svo: List = None, k: int = 5):
        """
        Function: Query vector database to retrieve relevant chunks and nodes

        Args:
            question: Test question to answer using vector search
            database: 'chunks' or 'nodes' to specify which vector store to query
            svo: Subject-Verb-Object dictionary extracted from the question (optional)

        Returns:
            vector results
        """

        vector_results = database.similarity_search(
            query=question,
            k=k
        )
        
        if svo is not None:
            for item in svo:
                vector_results.append(database.similarity_search(query=item))

        return vector_results
    
    
    
    def query_chunks_by_similarity(self, question: str, k: int = 5):
        """
        Function: Embed question and retrieve similar chunks from graph database

        Args:
            question: Test question to answer using vector search
            k: number of top similar chunks to retrieve

        Returns:
            vector results as text, page number and score
        """

        question_embedding = self.embeddings.embed_query(question)
        
        result = self.graph.query("""
                                  MATCH (c:Chunk)
                                  WITH c, gds.similarity.cosine(c.embedding, $embedding) AS score
                                  ORDER BY score DESC
                                  LIMIT $k
                                  RETURN c.text AS text, c.page AS page, score
                                  """, { "embedding": question_embedding, "k": k})

        return result
    
    
    
    def validate_and_answer(self, question, node_result, relationship_result, chunk_result, graph_result, advanced_search: str = None) -> str:
        # Format results into natural language answer
        if advanced_search is not None:
            advanced_search_text = f"Advanced Deeper Search:\n{json.dumps(advanced_search, indent=2)}\n"
        else:
            advanced_search_text = ""
        
        format_prompt = f"""Based on the following graph database query results, provide a clear, concise answer to the original question.

Question: {question}

Node Vector Results: {node_result}

Relationship Vector Results: {relationship_result}

Chunk Vector Results: {chunk_result}

Query Results:
{graph_result if graph_result else "No results found"}

{advanced_search_text}

Provide a natural language answer that:
1. Directly answers the question
2. Includes specific names, relationships, and details from the results
3. Acknowledges if information is missing or incomplete
4. Is clear and concise

Return ONLY the answer text, no preamble or JSON formatting."""


        structured_answer = self.openai_client.invoke(format_prompt).content
        
        return structured_answer.strip()
    
    
    
    def create_variety_questions(self, question: str, number_of_questions: int = 3) -> List[str]:
        """
        A function to create a variety of reformulated questions from the original question
        
        Args:
            question: Original user question
            number_of_questions: Number of reformulated questions to generate
            
        Returns:
            List of reformulated questions
        """
        system_prompt = """You are a question reformulation expert. Your task is to create alternative phrasings of a given question while preserving the exact same meaning and context.

Each reformulated question should:
- Ask for the same information as the original
- Use different wording, sentence structure, or perspective
- Maintain the same level of specificity
- Be clear and well-formed

Do not add new constraints, change the scope, or alter the intent of the original question."""

        user_prompt = f"""Create exactly {number_of_questions} different reformulations of the following question. Each version should ask for the same information but use different wording.

Original question: {question}

Generate {number_of_questions} alternative phrasings."""

        response_schema = {
            "title": "VarietyQuestions",
            "type": "object",
            "description": "A list of reformulated questions",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "A list of reformulated questions"
                }
            },
            "required": ["questions"]
        }
        
        agent = create_agent(model=self.claude_client, tools=[], response_format=ToolStrategy(schema=response_schema), system_prompt=system_prompt)

        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        questions = response["structured_response"]["questions"]
        return questions
    
    
    
    # possibly use or implement spacy or other NLP
    def find_svo(self, question: str) -> Dict[str, str]:
        """
        A function to extract subject, verb, object from a question using LLM
        
        Args:
            question: Original user question
        Returns:
            Dictionary with question, subject, verb, object
        """
        
        system_prompt = """You are a linguistic analysis expert specialized in extracting grammatical components from questions.

Your task is to identify the Subject, Verb, and Object (SVO) from a given question:
- Subject: The entity performing the action or being asked about
- Verb: The main action or state being queried
- Object: The entity receiving the action or being related to the subject

Guidelines:
- For questions, convert the interrogative form to a declarative statement to identify SVO
- Extract the core semantic components, not just surface-level words
- If a component is implicit or missing, infer it from context
- Keep each component concise (a few words maximum)"""

        user_prompt = f"""Extract the subject, verb, and object from the following question:

Question: {question}

Identify the SVO components."""

        response_schema = {
            "title": "SubjectVerbObject",
            "type": "object",
            "description": "A dictionary with subject, verb, and object extracted from the question",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "The subject extracted from the question"
                },
                "verb": {
                    "type": "string",
                    "description": "The verb extracted from the question"
                },
                "object": {
                    "type": "string",
                    "description": "The object extracted from the question"
                }
            },
            "required": ["subject", "verb", "object"]
        }
        
        agent = create_agent(model=self.claude_client, tools=[], response_format=ToolStrategy(schema=response_schema), system_prompt=system_prompt)

        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        svo = response["structured_response"]
        return svo
    
    
    
        """ semanticke vyhladavanie:
    1. poslat otazku na preformulovanie a vytvorenie 3-5 roznych otazok (kontext otazky ten isty)
    2. pre kazdu otazku najst podmet, predmet, vztah
    3. pomocou MCP posielat a skusat query na KG, opakovat dokym nevrati najblizsie nody a edge k podmetu, prisudku a vztahu
    4. poslat vytvotene otazky, UQ, vretene KGs a poslat LLM ci vratene hodnoty zodpovedaju otazke, najst Multi-hop
    5. zobrat vsetky chunky, kde sa nachadzaju tieto nody 
    6. poslat LLM na vyhodnotenie a spracovanie vyslednej odpovede:
       vytvorene otazky, povodna pouzivatelova otazka, grafy (vratene entity a vztahy), text z chunkov, (system prompt na vyhodnotenie)
    """
    # ---------------- INTERACTIVE QUESTIONING ----------------
    def invoke_question(self):
        """
        A function for question input and invoking the question LLM, Graph and Vector Databases
        """
        
        question = input("Enter your question: ")
        
        if (question=='-h'):
            print("Help Instructions: \n - To exit, type 'exit' \n - To view graph schema, type '-s' \n")
        elif (question=='-s'):
            print(self.get_graph_schema())
        elif (question.lower()=='exit'):
            print("Exiting...")
            return
        
        various_questions = self.create_variety_questions(question, number_of_questions=3)
        
        questions = [{'id': 'question0', 'question': question, 'svo': self.find_svo(question)}]
        for i, q in enumerate(various_questions):
            questions.append(
                {'id': f'question{i}', 'question': q, 'svo': self.find_svo(q)}
            )  
        print(f"\nGenerated Reformulated Questions\n\n: {[q['question'] + '\nSVO: ' + str(q['svo']) + '\n' for q in questions]}")
        
        
        for i, q in enumerate(questions):
            sub_obj = q['svo']['subject'] + q['svo']['object']
            q['similar_nodes'] = self.query_vector_database(database=self.vector_store_nodes, question=q['question'], svo=sub_obj)
            q['similar_relationships'] = self.query_vector_database(database=self.vector_store_relationships, question=q['question'], svo=q['svo']['verb'])
        
        
        graph_schema = self.graph.get_schema()
        
        
        advanced_search_result = None
        graph_query_result = chain.invoke( {"query": question} )
        if (self._advanced_search):
            advanced_search_result = self.query_graph_database(question=question)['query_data']
        
        
        
        # validate search results and generate final answer
        final_answer = self.validate_and_answer(
            question=question,
            node_result=nodes_vector_results,
            relationship_result=relationship_vector_results,
            chunk_result=chunk_vector_results,
            graph_result=graph_query_result,
            advanced_search=advanced_search_result
        )
        
        print(final_answer)