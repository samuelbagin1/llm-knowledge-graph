import datetime
import json
import os
from dataclasses import dataclass
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

from classification import classify
from classes import Schema, ClassifiedDocument, Type, SVO, Question
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
import asyncio
from prompts import response_schema_for_sde, system_prompt_for_sde, system_prompt_for_generating_query, response_schema_for_generating_query, response_schema_for_odd, system_prompt_for_odd, system_prompt_for_schema_refinement
from examples import examples_for_extraction
from pydantic import BaseModel, Field


# Default node type when type is missing or empty
DEFAULT_NODE_TYPE = "Entity"


def format_property_key(s: str) -> str:
    """Convert property key to camelCase format.

    Example: "first name" -> "firstName"
    """
    words = s.split()
    if not words:
        return s
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def format_node_type(node_type: Optional[str]) -> str:
    """Normalize node type to capitalized format.

    Example: "person" -> "Person", "" -> DEFAULT_NODE_TYPE
    """
    if not node_type or not node_type.strip():
        return DEFAULT_NODE_TYPE
    return node_type.strip().capitalize()


def format_relationship_type(rel_type: str) -> str:
    """Normalize relationship type to uppercase with underscores.

    Example: "works for" -> "WORKS_FOR"
    """
    if not rel_type:
        return "RELATED_TO"
    return rel_type.strip().replace(" ", "_").upper()


def graph_document_to_json(graph_doc: "GraphDocument") -> dict:
    """Convert a GraphDocument to a JSON-serializable dictionary.

    Args:
        graph_doc: A GraphDocument containing nodes, relationships, and source

    Returns:
        Dictionary with all node and relationship data
    """
    nodes_json = []
    for node in graph_doc.nodes:
        node_dict = {
            "id": node.id,
            "type": node.type,
            "properties": node.properties if node.properties else {}
        }
        nodes_json.append(node_dict)

    relationships_json = []
    for rel in graph_doc.relationships:
        rel_dict = {
            "source_id": rel.source.id,
            "source_type": rel.source.type,
            "relation": rel.type,
            "target_id": rel.target.id,
            "target_type": rel.target.type,
            "properties": rel.properties if rel.properties else {}
        }
        relationships_json.append(rel_dict)

    return {
        "nodes": nodes_json,
        "relationships": relationships_json,
        "source": graph_doc.source.page_content if graph_doc.source else None
    }




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
    def __init__(self,
                 neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 openai_api_key: str = None, google_api_key: str = None,
                 claude_api_key: str = None, advanced_search: bool = False,
                 strict_mode: bool = False):
        
        self.strict_mode = strict_mode  # Enforce schema compliance via post-extraction filtering
        
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
        self._vector_store_chunk_name = "chunk_vector_store"
        self._vector_store_nodes_name = "nodes_vector_store"
        self._vector_store_relationships_name = "relationships_vector_store"
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
            model="gpt-4o-mini",     # -mini 
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

        self.graph_transformer = LLMGraphTransformer(
            llm=self.claude_client,
            allowed_nodes=["Paragraph", "LegalConcept", "Institution", "Subject", "Document"],
            allowed_relationships=["ODKAZUJE_NA", "DEFINUJE", "UPRAVUJE", "RUŠUJE", "DOPLŇUJE"],
            strict_mode=True,
            node_properties=["context"],
            additional_instructions="Extrahuj právne entity zo slovenského právneho textu. Zameraj sa na paragrafy (§), právne pojmy, inštitúcie a krížové odkazy."
        )


    def _init_vector_stores(self):
        """Initialize vector stores, creating empty ones if indices don't exist"""
        try:
            # Try to load existing indices
            
            self.vector_store_relationships = Neo4jVector.from_existing_index(
                self.embeddings,
                url=self._neo4j_uri,
                username=self._neo4j_user,
                password=self._neo4j_password,
                index_name=self._vector_store_relationships_name,
            )
            
            
        except ValueError:
            # Indices don't exist yet - set to None and they'll be created on first use
            print("Vector store indice for relationship not found - will be created when documents are added")
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
                
                
    def get_graph_schema(self) -> Schema:
        # Get node labels and relationship types
        node_labels = self.graph.query("CALL db.labels()")
        rel_types = self.graph.query("CALL db.relationshipTypes()")
        
        schema = Schema(
            nodes=[node['label'] for node in node_labels],
            relationships=[rel['relationshipType'] for rel in rel_types]
        )
        
        return schema
                
                
                
    def get_graph_schema_detailed(self):
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
    
    


    # ---------------- PDF to Graph and Vector Processing ---------------
        
    
    def _convert_to_graph_document(self, data, i, document) -> GraphDocument:
        """
        Convert extracted data into a GraphDocument.

        Includes:
        - Property key formatting (camelCase)
        - Validation for missing node IDs (skips invalid nodes)
        - Node type fallback to DEFAULT_NODE_TYPE
        - Relationship type normalization
        """
        nodes = []
        relationships = []

        chunk_id = f"chunk_{i}"

        chunk_node = Node(
            id=chunk_id,
            type="Chunk",
            properties={
                "text": document.page_content,
                "page": document.metadata.get("page", 0)
            }
        )

        # Process nodes with validation and formatting
        for node_data in data.get("nodes", []):
            # Skip nodes without valid IDs
            node_id = node_data.get("id")
            if not node_id or not str(node_id).strip():
                continue

            # Format node type with fallback
            node_type = format_node_type(node_data.get("label") or node_data.get("type"))

            # Format property keys to camelCase
            raw_properties = node_data.get("properties", {})
            formatted_properties = {
                format_property_key(k): v
                for k, v in raw_properties.items()
            } if raw_properties else {}

            # Normalize node ID (title case for consistency)
            normalized_id = str(node_id).strip()
            if normalized_id and not normalized_id[0].isdigit():
                normalized_id = normalized_id.title()

            node = Node(
                id=normalized_id,
                type=node_type,
                properties=formatted_properties
            )
            nodes.append(node)

        # Process relationships with validation and formatting
        for rel_data in data.get("relationships", []):
            source_id = rel_data.get("source_node_id")
            target_id = rel_data.get("target_node_id")
            rel_type = rel_data.get("relation") or rel_data.get("type")

            # Skip relationships with missing mandatory fields
            if not source_id or not target_id or not rel_type:
                continue

            # Find matching nodes (case-insensitive)
            source_node = next(
                (n for n in nodes if n.id.lower() == str(source_id).strip().lower()),
                None
            )
            target_node = next(
                (n for n in nodes if n.id.lower() == str(target_id).strip().lower()),
                None
            )

            if source_node and target_node:
                # Format relationship properties
                raw_rel_props = rel_data.get("properties", {})
                formatted_rel_props = {
                    format_property_key(k): v
                    for k, v in raw_rel_props.items()
                } if raw_rel_props else {}

                relationship = Relationship(
                    source=source_node,
                    target=target_node,
                    type=format_relationship_type(rel_type),
                    properties=formatted_rel_props
                )
                relationships.append(relationship)

        # Link chunk to all extracted nodes
        for node in nodes:
            relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node,
                    type="HAS"
                )
            )

        nodes.append(chunk_node)

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=document
        )
        
        
        
    def _convert_to_schema(self, data) -> Schema:
        """
        Convert LLM response data (matching response_schema_for_odd or
        response_schema_for_schema_refinement) into a Schema instance.
        """
        return Schema(
            nodes=data.get("node_types", []),
            relationships=data.get("relationship_types", []),
        )



    def _filter_by_strict_mode(
        self,
        graph_doc: GraphDocument,
        allowed_entities: Optional[List[str]] = None,
        allowed_relationships: Optional[List[str]] = None
    ) -> GraphDocument:
        """
        Apply strict mode filtering to ensure extracted entities conform to allowed types.

        This is a post-extraction safety net that filters out any nodes or relationships
        that don't match the specified schema, similar to LangChain's strict_mode.

        Args:
            graph_doc: The GraphDocument to filter
            allowed_entities: List of allowed node types (case-insensitive)
            allowed_relationships: List of allowed relationship types (case-insensitive)

        Returns:
            Filtered GraphDocument with only conforming nodes and relationships
        """
        if not allowed_entities and not allowed_relationships:
            return graph_doc

        filtered_nodes = list(graph_doc.nodes)
        filtered_relationships = list(graph_doc.relationships)

        # Filter nodes by allowed types
        if allowed_entities:
            lower_allowed_entities = [e.lower() for e in allowed_entities]
            # Keep Chunk nodes regardless of allowed_entities
            filtered_nodes = [
                node for node in filtered_nodes
                if node.type.lower() in lower_allowed_entities or node.type == "Chunk"
            ]

            # Also filter relationships to only include those between valid nodes
            valid_node_ids = {node.id for node in filtered_nodes}
            filtered_relationships = [
                rel for rel in filtered_relationships
                if rel.source.id in valid_node_ids and rel.target.id in valid_node_ids
            ]

        # Filter relationships by allowed types
        if allowed_relationships:
            lower_allowed_rels = [r.lower().replace(" ", "_") for r in allowed_relationships]
            # Keep HAS relationships (chunk-to-entity links) regardless
            filtered_relationships = [
                rel for rel in filtered_relationships
                if rel.type.lower() in lower_allowed_rels or rel.type == "HAS"
            ]

        return GraphDocument(
            nodes=filtered_nodes,
            relationships=filtered_relationships,
            source=graph_doc.source
        )





    def load_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        print("PDF loaded successfully.")
        return loader.load()
    
    
    def classification(documents: List[Document]) -> ClassifiedDocument:
        return classify(documents)




    async def open_domain_detection(
        self,
        i: int,
        document: Document,
    ) -> Schema:
        """
        Async function to extract entity labels and relationships from a document
        and transform into a Schema.

        Args:
            document: The document to process

        Returns:
            Schema with extracted node labels and relationships
        """
        print(f"ODD: chunk {i}")
        
        text = document.page_content
        user_prompt = f"""
        Identifikuj každý odlišný typ uzla a typ vzťahu prítomný v nasledujúcom texte. Pred vytvorením odpovede postupuj textom krok za krokom a zdôvodni svoje závery:

        # PRAVIDLÁ
        1. Pozorne si prečítaj text.
        2. Identifikuj všetky odlišné typy uzlov pomocou všeobecných elementárnych označení. Pri každom uveď podpornú frázu (frázy) z textu, ktoré odôvodňujú jeho zaradenie.
        3. Identifikuj všetky odlišné typy vzťahov medzi entitami pomocou všeobecných nadčasových označení. Pri každom uveď podpornú frázu (frázy).
        4. Ak je niektorý typ nejednoznačný alebo závislý od kontextu, stručne poznač túto nejednoznačnosť.
        5. Všetky extrahované názvy typov (uzlov aj vzťahov) píš BEZ DIAKRITIKY — nahraď znaky s diakritikou ich ASCII ekvivalentmi (napr. č→c, š→s, ž→z, á→a, é→e, í→i, ó→o, ú→u, ý→y, ň→n, ť→t, ď→d, ľ→l, ô→o).

        # TEXT
        {text}
        """

        # Create and run the agent
        agent = create_agent(
            model=self.openai_graph_transform,
            response_format=ProviderStrategy(schema=response_schema_for_odd),
            system_prompt=system_prompt_for_odd
        )
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})

        # structured_response is already a dict when using ProviderStrategy
        data = response["structured_response"]
        
        print(data)

        # Convert extracted data to Schema
        schema = self._convert_to_schema(data)


        return schema
    
    
    
    async def async_open_domain_detection(
        self,
        documents: List[Document],
        max_concurrent: int = 10,
    ) -> List[Schema]:
        """
        Asynchronously process documents to extract schemas.

        Args:
            documents: List of documents to process
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of Schema
        """
        print("creating tasks: OPEN-DOMAIN DETECTION")
        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited(i, doc):
            async with semaphore:
                return await self.open_domain_detection(i, doc)

        tasks = [
            asyncio.create_task(limited(i, doc))
            for i, doc in enumerate(documents)
        ]

        res = await asyncio.gather(*tasks)
        return res
    
    
    
    
    
    def schema_refinement(self, odd_schema: Schema, existing_schema: Schema = None) -> Schema:
        """
        Function to refine and consolidate extracted schema information across documents, ensuring consistency and resolving conflicts.

        Args:
            schema: The schema to refine

        Returns:
            Schema with extracted node labels and relationships
        """
        
        
        class SchemaRefinementResponse(BaseModel):
            """Result of schema refinement of entities and relationships from text."""

            class MergeLog(BaseModel):
                """Log of merged types mapping canonical types to their original variants."""
                node_types: Dict[str, List[str]] = Field(description="Mapping of canonical node types to lists of original types that were merged")
                relationship_types: Dict[str, List[str]] = Field(description="Mapping of canonical relationship types to lists of original types that were merged")

            node_types: List[str] = Field(description="List of canonical node type labels")
            relationship_types: List[str] = Field(description="List of canonical relationship types")
            merge_log: MergeLog = Field(description="Log of merged types mapping canonical types to their original variants")
            
            
        
        print("SCHEMA REFINEMENT")
        
        user_prompt = f"""
        Spresni nasledujúcu surovú schému vytvorenú detekciou z otvorenej domény. Tvojím cieľom je vytvoriť čistú, konzistentnú a deduplikovanú schému vhodnú na extrakciu entít a vzťahov riadenú schémou.

        # REŽIM SPRESŇOVANIA
        Dostávaš dva vstupy:
        1. Surovú schému z detekcie z otvorenej domény.
        2. Základnú ontológiu (existujúcu schému), ak je dostupná.

        {"""
        Ak je poskytnutá základná ontológia:
        - Zarovnaj surovú schému na základnú ontológiu tam, kde existuje jasný sémantický prekryv. Mapuj surové typy na existujúce základné typy, keď sú ekvivalentné alebo takmer synonymné.
        - Zachovaj všetky surové typy, ktoré sú skutočne odlišné a nie sú zastúpené v základnej ontológii. Pridaj ich do spresnenej schémy ako nové typy.
        - Nenúť odlišné surové typy do typov základnej ontológie. Zlučuj iba vtedy, keď je sémantické zarovnanie jasné.
        - Základná ontológia má prednosť v pomenovaniach: ak sa surový typ zlučuje so základným typom, preberaj označenie základnej ontológie.
        
        """ if existing_schema else """ 
        
        Ak nie je poskytnutá žiadna základná ontológia:
        - Spresni surovú schému samostatne zlúčením sémanticky podobných typov a normalizáciou označení.
        """}

        # PRAVIDLÁ
        - Ide o ľahký spresňovací prechod. Zlučuj iba typy s jasným sémantickým prekryvom.
        - Ak sú typy odlišné, ponechaj ich — neprekonsolidovávaj.
        - Pri zlučovaní uprednostňuj všeobecnejšie a znovupoužiteľnejšie označenie.
        - Nevymýšľaj typy, ktoré sa nenachádzajú ani v surovej schéme, ani v základnej ontológii.
        - Neodstraňuj skutočne odlišné typy kvôli minimalizácii schémy.
        - Všetky názvy typov (uzlov aj vzťahov) píš BEZ DIAKRITIKY — nahraď znaky s diakritikou ich ASCII ekvivalentmi (napr. č→c, š→s, ž→z, á→a, é→e, í→i, ó→o, ú→u, ý→y, ň→n, ť→t, ď→d, ľ→l, ô→o).

        Pred vytvorením odpovede prejdi nasledujúcim uvažovaním:
        1. Prezri všetky typy uzlov oproti základnej ontológii (ak je poskytnutá). Identifikuj sémantické zhody, prekryvy a skutočne nové typy.
        2. Prezri všetky typy vzťahov oproti základnej ontológii (ak je poskytnutá). Identifikuj synonymné, takmer duplicitné alebo už pokryté typy oproti odlišným novým.
        3. Zlučuj iba tam, kde je sémantická podobnosť jasná. Odlišné typy ponechaj bez zmeny.
        4. Over krížovú konzistenciu: zabezpeč, aby každý typ uzla, ktorý sa podieľa na vzťahu, bol prítomný vo výslednom zozname typov uzlov.
        5. Zdokumentuj každé rozhodnutie o zlúčení v merge_log s kanonickým označením ako kľúčom a pôvodnými označeniami ako hodnotami.

        # ZÁKLADNÁ ONTOLÓGIA
        ## Typy uzlov:
        {existing_schema.nodes if existing_schema else "Neposkytnutá"}

        ## Typy vzťahov:
        {existing_schema.relationships if existing_schema else "Neposkytnuté"}

        # SCHÉMA DETEGOVANÁ Z OTVORENEJ DOMÉNY
        ## Typy uzlov:
        {odd_schema.nodes}

        ## Typy vzťahov:
        {odd_schema.relationships}
        """

        # Create and run the agent (ProviderStrategy uses Gemini's native JSON output)
        agent = create_agent(
            model=self.gemini_client,
            system_prompt=system_prompt_for_schema_refinement,
            response_format=SchemaRefinementResponse
        )
        for attempt in range(3):
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
                break
            except Exception as e:
                print(f"Schema refinement attempt {attempt + 1}/3 failed: {e}")
                if attempt == 2:
                    raise

        # structured_response is already a dict when using ProviderStrategy
        data = response["structured_response"]

        print(data)

        # Convert extracted data to Schema
        schema = self._convert_to_schema(data)


        return schema
    
    
    
    
    
    async def schema_driven_extraction(
        self,
        i: int,
        document: Document,
        schema: Schema
    ) -> GraphDocument:
        """
        Async function to extract named entities and relationships from a document
        and transform into a GraphDocument.

        Args:
            i: Document index (used for chunk ID generation)
            document: The document to process
            schema: The Schema defining allowed node labels and relationship types

        Returns:
            GraphDocument with extracted and optionally filtered nodes/relationships
        """
        print(f"SDE: chunk {i}")
        
        text = document.page_content
        user_prompt = f"""
        Extract all entities and relationships from the following text using ONLY the entity types and relationship types defined in the schema below. Do not use types outside this schema. If an entity or relationship does not match the schema, omit it.

        # SCHEMA
        ## Entity Types:
        {schema.nodes}

        ## Relationship Types:
        {schema.relationships}

        # TEXT:
        {text}
        """

        print(f"NER: chunk {i}")
        # Create and run the agent
        agent = create_agent(
            model=self.openai_graph_transform,
            response_format=ProviderStrategy(schema=response_schema_for_sde),
            system_prompt=system_prompt_for_sde
        )
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})

        # structured_response is already a dict when using ProviderStrategy
        data = response["structured_response"]
        
        print(data)

        # Convert to graph document with validation and formatting
        graph_document = self._convert_to_graph_document(data, i, document)

        # Apply strict mode filtering if enabled
        # if strict_mode and (allowed_entities or allowed_relationships):
        #     graph_document = self._filter_by_strict_mode(
        #         graph_document,
        #         allowed_entities=allowed_entities,
        #         allowed_relationships=allowed_relationships
        #     )

        return graph_document
    
    
    
    async def async_schema_driven_extraction(
        self,
        documents: List[Document],
        schema: Schema
    ) -> List[GraphDocument]:
        """
        Asynchronously process documents to extract graph documents.

        Args:
            documents: List of documents to process
            allowed_entities: List of allowed node types
            allowed_relationships: List of allowed relationship types
            strict_mode: If True, applies post-extraction filtering

        Returns:
            List of GraphDocuments
        """
        print("creating tasks: SCHEMA-DRIVEN EXTRACTION")
        tasks = [
            asyncio.create_task(
                self.schema_driven_extraction(
                    i, doc, schema
                )
            )
            for i, doc in enumerate(documents)
        ]
        res = await asyncio.gather(*tasks)
        return res
    
    
    



    # --------- PROCESS STRATEGY ----------
    # 1. classification
    # 2. open domain schema detection
    # 3. schema refinement
    # 4. schema guided extraction
    # 5. vector store from graph



    def process(self, pdf_path: str, max_pages: int = None):
        
        # Load PDF documents
        documents = self.load_pdf(pdf_path)
        if max_pages:
            documents = documents[:max_pages]
            
            
        document_classification = self.classification()

        
        # odd
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunked_documents = splitter.split_documents(documents)

        extracted_schema_list = asyncio.run(
            self.async_open_domain_detection(
                chunked_documents,
            )
        )
        print(f"\nAll chunks processed into graph documents. (strict_mode={self.strict_mode})")
        
        
        # refinement
        extracted_schema = Schema(
            nodes=list(set(node for schema in extracted_schema_list for node in schema.nodes)),
            relationships=list(set(rel for schema in extracted_schema_list for rel in schema.relationships))
        )
        refined_schema = self.schema_refinement(odd_schema=extracted_schema, existing_schema=self.get_graph_schema())
        
        
        # sde
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        chunked_documents = splitter.split_documents(documents)
        
        graph_docs = asyncio.run(
            self.async_schema_driven_extraction(
                chunked_documents,
                schema=refined_schema
            )
        )
        print(f"\nAll chunks processed into graph documents. (strict_mode={self.strict_mode})")
        
        
        

        graph_docs_json = [graph_document_to_json(doc) for doc in graph_docs]
        with open("./GRAPH_DOCS.json", "w", encoding="utf-8") as f:
            json.dump(graph_docs_json, f, ensure_ascii=False, indent=2)
        
        
        
        # Add graph documents to Neo4j
        # dependency: APOC plugin in neo4j database
        self.graph.add_graph_documents(
            graph_documents=graph_docs,
            include_source=False,
            baseEntityLabel=True
        )
        
        self.graph.refresh_schema()
        # Remove __Entity__ label from Chunk nodes so the node vector store excludes them
        self.graph.query("MATCH (n:Chunk:__Entity__) REMOVE n:__Entity__")
        
        
        # create vector stores from existing graph
        self.vector_store_nodes = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=self._neo4j_uri,
            username=self._neo4j_user,
            password=self._neo4j_password,
            index_name=self._vector_store_nodes_name,
            node_label="__Entity__",
            text_node_properties=["id"],  # properties to concatenate as text to embed
            embedding_node_property="embedding",
        )
        
        self.vector_store_chunks = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=self._neo4j_uri,
            username=self._neo4j_user,
            password=self._neo4j_password,
            index_name=self._vector_store_chunk_name,
            node_label="Chunk",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        
        
        rel_types = self.graph.query("CALL db.relationshipTypes()")
        all_relationships = [Document(page_content=rel['relationshipType']) for rel in rel_types]
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
            
        
        
        
        
# ====================================================================================================
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

        

        # User prompt - provides the specific question and schema
        user_prompt = f"""Answer this question by querying the graph database:

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
            response_format=ProviderStrategy(schema=response_schema_for_generating_query),
            system_prompt=system_prompt_for_generating_query
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
    
    
    
    def query_vector_database(self, database: Neo4jVector, question: str, array: List = None, k: int = 5) -> List[Any]:
        """
        Function: Query vector database to retrieve relevant chunks and nodes

        Args:
            question: Test question to answer using vector search
            database: 'chunks' or 'nodes' to specify which vector store to query
            svo: Subject-Verb-Object object extracted from the question (optional)

        Returns:
            vector results
        """
        
        result = []
        for a in array:
            result.append(
                database.similarity_search(
                    query=a,
                    k=k
                )
            )
            
        return result
        
        
        
    
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
    
    
    
    
    def convert_sentence_to_graph_document(self, data, text: str = "") -> GraphDocument:
        """
        Convert extracted data into a GraphDocument.

        Includes:
        - Property key formatting (camelCase)
        - Validation for missing node IDs (skips invalid nodes)
        - Node type fallback to DEFAULT_NODE_TYPE
        - Relationship type normalization
        """
        nodes = []
        relationships = []
        source_document = Document(page_content=text)


        # Process nodes with validation and formatting
        for node_data in data.get("nodes", []):
            # Skip nodes without valid IDs
            node_id = node_data.get("id")
            if not node_id or not str(node_id).strip():
                continue

            # Format node type with fallback
            node_type = format_node_type(node_data.get("label") or node_data.get("type"))

            # Format property keys to camelCase
            raw_properties = node_data.get("properties", {})
            formatted_properties = {
                format_property_key(k): v
                for k, v in raw_properties.items()
            } if raw_properties else {}

            # Normalize node ID (title case for consistency)
            normalized_id = str(node_id).strip()
            if normalized_id and not normalized_id[0].isdigit():
                normalized_id = normalized_id.title()

            node = Node(
                id=normalized_id,
                type=node_type,
                properties=formatted_properties
            )
            nodes.append(node)

        # Process relationships with validation and formatting
        for rel_data in data.get("relationships", []):
            source_id = rel_data.get("source_node_id")
            target_id = rel_data.get("target_node_id")
            rel_type = rel_data.get("relation") or rel_data.get("type")

            # Skip relationships with missing mandatory fields
            if not source_id or not target_id or not rel_type:
                continue

            # Find matching nodes (case-insensitive)
            source_node = next(
                (n for n in nodes if n.id.lower() == str(source_id).strip().lower()),
                None
            )
            target_node = next(
                (n for n in nodes if n.id.lower() == str(target_id).strip().lower()),
                None
            )

            if source_node and target_node:
                # Format relationship properties
                raw_rel_props = rel_data.get("properties", {})
                formatted_rel_props = {
                    format_property_key(k): v
                    for k, v in raw_rel_props.items()
                } if raw_rel_props else {}

                relationship = Relationship(
                    source=source_node,
                    target=target_node,
                    type=format_relationship_type(rel_type),
                    properties=formatted_rel_props
                )
                relationships.append(relationship)

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=source_document
        )
        


    def named_entity_extraction_from_sentence(
        self,
        text: str,
        schema: Schema,
    ) -> GraphDocument:
        """
        Async function to extract named entities and relationships from a document
        and transform into a GraphDocument.

        Args:
            text: The text to process
            allowed_entities: List of allowed node types for extraction guidance and filtering
            allowed_relationships: List of allowed relationship types
            strict_mode: If True, applies post-extraction filtering to enforce schema

        Returns:
            GraphDocument with extracted and optionally filtered nodes/relationships
        """
        user_prompt = f"""
        Extract all entities and relationships from the following text using ONLY the entity types and relationship types defined in the schema below. Do not use types outside this schema. If an entity or relationship does not match the schema, omit it.

        # SCHEMA
        ## Entity Types:
        {schema.nodes}

        ## Relationship Types:
        {schema.relationships}

        # TEXT:
        {text}
        """

        # Create and run the agent
        agent = create_agent(
            model=self.openai_graph_transform,
            response_format=ProviderStrategy(schema=response_schema_for_sde),
            system_prompt=system_prompt_for_sde
        )
        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

        # structured_response is already a dict when using ProviderStrategy
        data = response["structured_response"]

        # Convert to graph document with validation and formatting
        graph_document = self.convert_sentence_to_graph_document(data, text)

        return graph_document




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
            
            
            
        # 1. create various reformulations of the question
        various_questions = self.create_variety_questions(question, number_of_questions=3)
        
        # 2. for each question, find subject, verb, object and extraxt nodes from question using SDE
        questions = []
        questions.append(
            Question(id='question0', question=question, svo=self.find_svo(question), extracted_nodes=self.named_entity_extraction_from_sentence(text=question, schema=self.get_graph_schema()))
        )
        for i, q in enumerate(various_questions):
            questions.append(
                Question(id=f'question{i}', question=q, svo=self.find_svo(q), extracted_nodes=self.named_entity_extraction_from_sentence(text=q, schema=self.get_graph_schema()))
            )
            
        print(f"\nGenerated Reformulated Questions\n\n: {[q.question + '\nSVO: ' + str(q.svo) + '\n' for q in questions]}")
        
        
        
        # find similar nodes in graph from extracted nodes in queestion
        for i, q in enumerate(questions): 
            q.similar_nodes = self.query_vector_database(database=self.vector_store_nodes, question=q.question, array=q.extracted_nodes, k=5)
            
        
        
        
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