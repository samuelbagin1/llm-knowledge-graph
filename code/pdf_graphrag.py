from pathlib import Path
import re
import datetime
import json
import os
import time
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Generic, TypeVar, cast
from langchain_neo4j import Neo4jGraph, Neo4jVector, GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy
from openai import embeddings
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

from classes import Schema, ClassifiedDocument, Type, SubSentence, ReasoningStep
from chunker.chunker import Chunker
from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
import asyncio
from prompts import response_schema_for_sde, system_prompt_for_sde, response_schema_for_odd, system_prompt_for_odd, system_prompt_for_schema_refinement, response_schema_for_schema_refinement
from pydantic import BaseModel, Field, SecretStr
import numpy as np
from to_json import odd_to_json, refinement_to_json, sde_to_json, table_to_json, formula_to_json
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image
import base64
from langchain_core.messages import HumanMessage, SystemMessage


# Default node type when type is missing or empty
DEFAULT_NODE_TYPE = "Entity"


# Pre-baked domain schema used when the ODD + refinement pipeline is skipped.
# Shape matches response_schema_for_schema_refinement so _convert_to_schema accepts it.
SCHEMA_NODE_TYPES = [
    "Subjekt",
    "Osoba",
    "Organizacia",
    "Adresa",
    "Lokacia",
    "Stat",
    "Banka",
    "Paragraf",
    "Odsek",
    "Pismeno",
    "Bod",
    "Priloha",
    "PravnyPredpis",
    "Dokument",
    "Konanie",
    "Rozhodnutie",
    "Ziadost",
    "Oznamenie",
    "Dan",
    "DanovePriznanie",
    "ZdanovacieObdobie",
    "SadzbaDane",
    "NadmernyOdpocet",
    "Sankcia",
    "Povinnost",
    "Pravo",
    "Podmienka",
    "Lehota",
    "Obdobie",
    "Datum",
    "Dovod",
    "Status",
    "Tovar",
    "Sluzba",
    "Majetok",
    "Nehnutelnost",
    "Vozidlo",
    "Ucet",
    "BankovyUcet",
    "Platba",
    "Suma",
    "Mnozstvo",
    "Obrat",
    "Mena",
    "Kurz",
    "Zmluva",
    "Pohladavka",
    "Zavazok",
    "Zastupenie",
    "Registracia",
    "Zaznam",
]


SCHEMA_RELATIONSHIP_TYPES = [
    "VZTAHUJE_SA_NA",
    "NEVZTAHUJE_SA_NA",
    "UPRAVUJE",
    "DEFINUJE",
    "URCUJE",
    "ODKAZUJE_NA",
    "VYPLYVA_Z",
    "JE_TYPOM",
    "JE_SUCASTOU",
    "OBSAHUJE",
    "PATRI_DO",
    "NACHADZA_SA_V",
    "MA",
    "MA_ADRESU",
    "MA_IDENTIFIKATOR",
    "MA_STATUS",
    "MA_DATUM",
    "MA_OBDOBIE",
    "MA_LEHOTU",
    "MA_SUMU",
    "MA_HODNOTU",
    "MA_PODMIENKU",
    "MA_PRAVO",
    "MA_POVINNOST",
    "MA_NAROK_NA",
    "NEMA_NAROK_NA",
    "SPLNA_PODMIENKY",
    "NESPLNA_PODMIENKY",
    "KONA_V_MENE",
    "JE_ZASTUPENA",
    "ZODPOVEDA_ZA",
    "PODAVA",
    "PREDKLADA",
    "DORUCUJE",
    "OZNAMUJE",
    "PRIJIMA",
    "VYDAVA",
    "ROZHODUJE_O",
    "REGISTRUJE",
    "UCHOVAVA",
    "DODAVA",
    "POSKYTUJE",
    "PLATI",
    "PODLIEHA",
    "OSLOBODZUJE_OD",
    "VZNIKA",
    "ZANIKA",
    "NADOBUDA",
    "PRECHADZA_NA",
    "MENI",
    "NAHRADZA",
    "RUSI",
    "SUVISI_S",
    "JE_OSLOBODENE_OD_DANE",
    "JE_PREDMETOM_DANE",
    "NIE_JE_PREDMETOM_DANE",
    "JE_PODLA",
]


_HTML_HEADER = (
    "<!DOCTYPE html>\n"
    "<html><head><meta charset='utf-8'><style>"
    "table {border-collapse: collapse; width: 100%;} "
    "th, td {border: 1px solid #ddd; padding: 8px; text-align: left;} "
    "th {background-color: #f2f2f2;}"
    "</style></head><body>\n"
)
_HTML_FOOTER = "\n</body></html>"
_PAGE_NUM_RE = re.compile(r"^page(\d+)_")


_STRIP_COMMENTS_LINE = re.compile(r'//.*$', re.MULTILINE)
_STRIP_COMMENTS_BLOCK = re.compile(r'/\*.*?\*/', re.DOTALL)
_STRIP_STRINGS_RE = re.compile(r'"[^"]*"|\'[^\']*\'|`[^`]*`')
_CYPHER_WRITE_RE = re.compile(
    r'\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|FOREACH|DROP|ALTER|LOAD)\b',
    re.IGNORECASE,
)
_CYPHER_CALL_RE = re.compile(r'\bCALL\b', re.IGNORECASE)

def is_read_only_cypher(query: str) -> bool:
    sanitized = _STRIP_COMMENTS_BLOCK.sub("", query)
    sanitized = _STRIP_COMMENTS_LINE.sub("", sanitized)
    sanitized = _STRIP_STRINGS_RE.sub("", sanitized)

    m = _CYPHER_WRITE_RE.search(sanitized)
    if m:
        raise ValueError(f"Forbidden write keyword: '{m.group().upper()}'")

    if _CYPHER_CALL_RE.search(sanitized):
        raise ValueError("CALL procedures are not allowed in read-only mode")

    return True


def format_property_key(s: str) -> str:
    """Convert property key to camelCase, treating symbols as word separators.

    Neo4j naming rules forbid symbols (other than underscore) in property keys;
    keys that include them require backtick escaping in every Cypher query.
    Splitting on non-word characters yields a clean camelCase identifier.

    Example: "first name" -> "firstName", "price ($)" -> "price", "VAT %" -> "vat"
    """
    words = [w for w in re.split(r"[^\w]+", s, flags=re.UNICODE) if w]
    if not words:
        return ""
    first_word = words[0].lower()
    capitalized_words = [word.capitalize() for word in words[1:]]
    return "".join([first_word] + capitalized_words)


def strip_diacritics(s: str) -> str:
    """Decompose Unicode and drop combining marks so Slovak diacritics map to ASCII.

    Example: "Žilina" -> "Zilina", "Štefan" -> "Stefan", "rád" -> "rad".
    """
    decomposed = unicodedata.normalize("NFKD", s)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


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


def sanitize_property_keys(raw_properties: dict) -> dict:
    r"""Format property keys to be Neo4j-safe identifiers.

    Cypher naming rules: names must start with an alphabetic character, may not
    contain symbols other than underscore, and are case-sensitive. Violations
    don't fail ingest but force every downstream query to wrap the key in
    backticks (`n.\`2024\``), which is fragile and easy to forget.

    Rules applied:
    - Empty / whitespace-only keys -> col_<positional-index>
    - Keys starting with a digit -> prefixed with col_ (e.g. "2024" -> "col_2024")
    - Diacritics stripped so keys are pure ASCII
    """
    sanitized: dict = {}
    for i, (k, v) in enumerate(raw_properties.items()):
        formatted = format_property_key(k) if isinstance(k, str) else str(k)
        formatted = strip_diacritics(formatted).strip()
        if not formatted:
            formatted = f"col_{i}"
        elif formatted[0].isdigit():
            formatted = f"col_{formatted}"
        sanitized[formatted] = v
    return sanitized


def drop_empty_values(properties: dict) -> dict:
    """Strip properties whose values would land as empty in Neo4j.

    Empty = None, blank/whitespace string, empty list, empty dict. Booleans and
    numeric zeros are kept (they are meaningful values, not absence-of-value).
    """
    cleaned: dict = {}
    for k, v in properties.items():
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, dict, tuple, set)) and len(v) == 0:
            continue
        cleaned[k] = v
    return cleaned


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
                 openai_api_key: str | None = None, google_api_key: str | None = None,
                 database: str | None = None):
        
        
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            database=database,
            refresh_schema=True
        )

        # Initialize embeddings first - needed for vector stores
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-large', api_key=SecretStr(openai_api_key) if openai_api_key else None)

        # Store vector store configuration for lazy initialization
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password
        self._vector_store_chunk_name = "chunk_vector_store"
        self._vector_store_nodes_name = "nodes_vector_store"
        self._vector_store_relationships_name = "relationships_vector_store"

        # Initialize vector stores - will be created when first documents are added
        self._init_vector_stores()
        
        # Initialize LLM clients
        # ChatOpenAI for question generation
        self.openai_client = ChatOpenAI(
            model="gpt-5.4-mini",
            temperature=0.05,
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            max_retries=3,
            timeout=180
        )

        self.openai_graph_transform = ChatOpenAI(
            model="gpt-4o-mini",     # -mini
            temperature=0,
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            max_retries=3,
            timeout=120
        )
        
        self.openai_thinking = ChatOpenAI(
            model="gpt-5.5",     # -mini
            temperature=0,
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            max_retries=3,
            timeout=300
        )

        # Google Gemini for everything else
        self.gemini_client = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            google_api_key=google_api_key,
            timeout=600
        )
        
        self.gemini_client_thinking = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            google_api_key=google_api_key,
            timeout=600,
            thinking_level='high'
        )
        
        self.gemini_client_flash = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.2,
            google_api_key=google_api_key,
            timeout=600,
            max_output_tokens=100000,
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
        """
        Read existing node labels and relationship types from Neo4j.

        Read-only auxiliary call: on any Neo4j failure, returns an empty
        Schema rather than raising — empty existing-schema is a valid
        first-run state, handled downstream by schema_refinement.
        """
        try:
            node_labels = self.graph.query("CALL db.labels()")
            rel_types = self.graph.query("CALL db.relationshipTypes()")
        except Exception as e:
            print(f"[get_graph_schema] Neo4j read failed: {e}; returning empty Schema")
            return Schema(nodes=[], relationships=[])

        nodes = [
            n['label'] for n in (node_labels or [])
            if isinstance(n, dict) and isinstance(n.get('label'), str)
        ]
        rels = [
            r['relationshipType'] for r in (rel_types or [])
            if isinstance(r, dict) and isinstance(r.get('relationshipType'), str)
        ]

        return Schema(nodes=nodes, relationships=rels)
    
    
    def add_graph_to_database(self, graph_documents: list[GraphDocument] | GraphDocument):
        if isinstance(graph_documents, GraphDocument):
            graph_documents = [graph_documents]

        self.graph.add_graph_documents(
                graph_documents=graph_documents,
                include_source=False,
                baseEntityLabel=False
            )


    def merge_new(self, graph_documents: list[GraphDocument] | GraphDocument) -> None:
        """Write graph documents to Neo4j with properties applied on both CREATE and MATCH.

        LangChain's Neo4jGraph.add_graph_documents calls apoc.merge.node with an
        empty onMatchProps dict, so any node that already exists in the database
        keeps its old properties and silently drops the new ones. This method
        passes row.properties as both the onCreate and onMatch dicts, making
        re-runs idempotent (the graph converges to the latest extraction) and
        recovering nodes that were previously created without properties.

        Same fix is applied to apoc.merge.relationship for relationship props.
        """
        if isinstance(graph_documents, GraphDocument):
            graph_documents = [graph_documents]

        node_query = (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.type], {id: row.id}, "
            "row.properties, row.properties) YIELD node "
            "RETURN distinct 'done' AS result"
        )
        rel_query = (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.source_label], {id: row.source}, {}, {}) "
            "YIELD node AS source "
            "CALL apoc.merge.node([row.target_label], {id: row.target}, {}, {}) "
            "YIELD node AS target "
            "CALL apoc.merge.relationship(source, row.type, {}, "
            "row.properties, target, row.properties) YIELD rel "
            "RETURN distinct 'done' AS result"
        )

        def _clean(s: str) -> str:
            return s.replace("`", "") if isinstance(s, str) else s

        for doc in graph_documents:
            nodes_data = [
                {
                    "id": n.id,
                    "type": _clean(n.type),
                    "properties": n.properties or {},
                }
                for n in doc.nodes
            ]
            if nodes_data:
                self.graph.query(node_query, {"data": nodes_data})

            rels_data = [
                {
                    "source": r.source.id,
                    "source_label": _clean(r.source.type),
                    "target": r.target.id,
                    "target_label": _clean(r.target.type),
                    "type": _clean(r.type),
                    "properties": r.properties or {},
                }
                for r in doc.relationships
            ]
            if rels_data:
                self.graph.query(rel_query, {"data": rels_data})
                
                
                
    def get_sample_graph_schema(self) -> str:
        """Get Neo4j graph schema information with sample data
        
        Args:
            None
            
        Returns:
            string of sample data
        
        """
        try:
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
            schema = ""

            # Add sample data to understand property names
            schema += "\nSample Nodes (showing property structure):\n"
            for node in sample_nodes[:5]:
                schema += f"  - {node['label']}: {node['props']}\n"

            schema += "\nSample Relationships:\n"
            for rel in sample_rels[:5]:
                from_id = rel['from_props'].get('id', rel['from_props'].get('name', 'unknown'))
                to_id = rel['to_props'].get('id', rel['to_props'].get('name', 'unknown'))
                schema += f"  - ({rel['from_label']}: {from_id}) --[{rel['rel_type']}]--> ({rel['to_label']}: {to_id})\n"

            return schema

        except Exception as e:
            print(f"Error retrieve schema: {e}")
            return "Schema information unavailable"
        
    def query_graph(self, query: str):
        rows = self.graph.query(query)
        
        return rows
        
        
        
        
    def get_help(self):
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
        
    
    def _convert_to_graph_document(self, data, i, document, document_id, section_id: str = "", section_label: str = "") -> GraphDocument:
        """
        Convert extracted data into a GraphDocument.

        Includes:
        - Property key formatting (camelCase)
        - Validation for missing node IDs (skips invalid nodes)
        - Node type fallback to DEFAULT_NODE_TYPE
        - Relationship type normalization
        """
        chunk_id = f"chunk_{i}_{document_id}"
        nodes = []
        relationships = []
        
        if section_id and section_label:
            chunk_node = Node(
                id=section_id,
                type=section_label,
                properties={}
            )
        else:

            chunk_node = Node(
                id=chunk_id,
                type="Chunk",
                properties={
                    "name": chunk_id,
                    "document_id": document_id,
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

            # Format property keys to camelCase; rename empty keys to col_<i>
            raw_properties = node_data.get("properties", {})
            formatted_properties = sanitize_property_keys(raw_properties) if raw_properties else {}
            formatted_properties = drop_empty_values(formatted_properties)

            # Normalize node ID (ASCII-fold + title case for consistency)
            normalized_id = strip_diacritics(str(node_id).strip())
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
            evidence = rel_data.get("evidence")

            # Skip relationships with missing mandatory fields
            if not source_id or not target_id or not rel_type:
                continue

            # Find matching nodes (case- and diacritic-insensitive)
            source_key = strip_diacritics(str(source_id).strip()).lower()
            target_key = strip_diacritics(str(target_id).strip()).lower()
            source_node = next(
                (n for n in nodes if n.id.lower() == source_key),
                None
            )
            target_node = next(
                (n for n in nodes if n.id.lower() == target_key),
                None
            )

            if source_node and target_node:
                # Format relationship properties
                raw_rel_props = rel_data.get("properties", {})
                section_property = section_id if section_id else chunk_id
                formatted_rel_props = drop_empty_values({
                    **sanitize_property_keys(raw_rel_props),
                    "section": section_property,
                    "evidence": evidence
                })

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
                    source=node,
                    target=chunk_node,
                    type="IN_CHUNK"
                )
            )

        nodes.append(chunk_node)

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={})
        )
        
        
        
    def _convert_table_to_graph_document(self, data, i, document, document_id) -> GraphDocument:
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

        # Process nodes with validation and formatting
        for node_data in data.get("nodes", []):
            # Skip nodes without valid IDs
            node_id = node_data.get("id")
            if not node_id or not str(node_id).strip():
                continue

            # Format node type with fallback
            node_type = format_node_type(node_data.get("label") or node_data.get("type"))

            # Format property keys to camelCase; rename empty keys to col_<i>
            raw_properties = node_data.get("properties", {})
            if node_type == "Tabulka":
                formatted_properties = {
                    **sanitize_property_keys(raw_properties),
                    "document_id": document_id,
                    "html": document.page_content,
                    "page": document.metadata.get("page_range")
                }
            else:
                formatted_properties = sanitize_property_keys(raw_properties) if raw_properties else {}

            formatted_properties = drop_empty_values(formatted_properties)

            # Normalize node ID (ASCII-fold + title case for consistency)
            normalized_id = strip_diacritics(str(node_id).strip())
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

            # Find matching nodes (case- and diacritic-insensitive)
            source_key = strip_diacritics(str(source_id).strip()).lower()
            target_key = strip_diacritics(str(target_id).strip()).lower()
            source_node = next(
                (n for n in nodes if n.id.lower() == source_key),
                None
            )
            target_node = next(
                (n for n in nodes if n.id.lower() == target_key),
                None
            )

            if source_node and target_node:
                # Format relationship properties
                raw_rel_props = rel_data.get("properties", {})
                formatted_rel_props = drop_empty_values({
                    **sanitize_property_keys(raw_rel_props),
                    "document_id": document_id
                })

                relationship = Relationship(
                    source=source_node,
                    target=target_node,
                    type=format_relationship_type(rel_type),
                    properties=formatted_rel_props
                )
                relationships.append(relationship)


        document_node = Node(id=document_id, type="Document", properties={})
        nodes.append(document_node)

        tabulka_nodes = [n for n in nodes if n.type == "Tabulka"]

        for tabulka_node in tabulka_nodes:
            relationships.append(
                Relationship(
                    source=tabulka_node,
                    target=document_node,
                    type="IN_DOCUMENT",
                    properties={"document_id": document_id},
                )
            )



        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"id": document_id})
        )
        
        
        
    def _add_document_chunk(self, graph_documents: list[GraphDocument], document_id, properties: dict | None = None) -> GraphDocument:
        if properties is None:
            properties = {}

        document_node = Node(id=document_id, type="Document", properties=properties)

        # Only "Chunk" nodes get linked to the Document. These are created in
        # _convert_to_graph_document for trailing chunks (no structural `path`);
        # structured chunks instead become Paragraf/Odsek/Pismeno section nodes
        # already linked to the Document via IN_DOCUMENT in the tree graph. We
        # reuse the exact Node objects from the produced graph documents so the
        # IN_CHUNK edges attach to the real, fully-propertied Chunk nodes.
        chunk_nodes = [
            node
            for graph_document in graph_documents
            for node in graph_document.nodes
            if node.type == "Chunk"
        ]

        relationships = [
            Relationship(source=chunk_node, target=document_node, type="IN_DOCUMENT")
            for chunk_node in chunk_nodes
        ]

        return GraphDocument(
            nodes=[document_node] + chunk_nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"id": document_id}),
        )
        
        
        
    def _convert_to_schema(self, data: Optional[Dict[str, Any]]) -> Schema:
        """
        Convert LLM response data (matching response_schema_for_odd or
        response_schema_for_schema_refinement) into a Schema instance.

        Raises:
            ValueError: if data is not a dict, or is missing both
                node_types and relationship_types keys.
        """
        if data is None:
            raise ValueError("_convert_to_schema: data is None")
        if not isinstance(data, dict):
            raise ValueError(
                f"_convert_to_schema: expected dict, got {type(data).__name__}"
            )

        if "node_types" not in data and "relationship_types" not in data:
            raise ValueError(
                "_convert_to_schema: schema payload missing node_types and relationship_types"
            )

        raw_nodes = data.get("node_types", []) or []
        raw_rels = data.get("relationship_types", []) or []

        if not isinstance(raw_nodes, list):
            raise ValueError(
                f"_convert_to_schema: node_types must be a list, got {type(raw_nodes).__name__}"
            )
        if not isinstance(raw_rels, list):
            raise ValueError(
                f"_convert_to_schema: relationship_types must be a list, got {type(raw_rels).__name__}"
            )

        nodes = [n for n in raw_nodes if isinstance(n, str) and n]
        rels = [r for r in raw_rels if isinstance(r, str) and r]

        if len(nodes) != len(raw_nodes) or len(rels) != len(raw_rels):
            print(
                f"[_convert_to_schema] dropped non-string entries: "
                f"nodes {len(raw_nodes) - len(nodes)}, rels {len(raw_rels) - len(rels)}"
            )

        return Schema(nodes=nodes, relationships=rels)



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
                if rel.type.lower() in lower_allowed_rels or rel.type == "IN_CHUNK"
            ]

        return GraphDocument(
            nodes=filtered_nodes,
            relationships=filtered_relationships,
            source=graph_doc.source
        )








    def detect_tables(self, pdf_path: str, output_dir: str = "./file_output/tables_and_formulas", conf_threshold: float = 0.3):
        print("detecting tables")
        TARGET_CLASSES = {"table", "picture", "figure", "isolate_formula", "formula_caption"}
        os.makedirs(output_dir, exist_ok=True)

        
        model_path = hf_hub_download(
            repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
            filename="doclayout_yolo_docstructbench_imgsz1024.pt"
        )
        model = YOLOv10(model_path)
        print("model loaded\n")


        pages = convert_from_path(pdf_path, dpi=200)
        print(f"\nconverted pdf to images, total pages: {len(pages)}\n")


        print("running detection\n")
        detections_found = []

        for page_num, page_img in enumerate(pages, start=1):
            results = model.predict(np.array(page_img), imgsz=1024, conf=conf_threshold, device="cpu")

            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                class_names = result.names

                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id].lower()
                    conf = float(box.conf[0])

                    if cls_name not in TARGET_CLASSES:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                    cropped = page_img.crop((x1, y1, x2, y2))
                    filename = f"page{page_num}_{cls_name}_{i}_conf{conf:.2f}.png"
                    image_path = os.path.join(output_dir, filename)
                    cropped.save(image_path)

                    detections_found.append({
                        "page": page_num,
                        "class": cls_name,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                        "image_path": image_path,
                    })

                    print(f"  page {page_num:>3d} | class={cls_name:<8s} | conf={conf:.2f} | bbox=({x1},{y1},{x2},{y2}) | saved: {filename}")



        print(f"detection completed")
        print(f"total detections: {len(detections_found)}")
        
        # format output
        if detections_found:
            table_pages = sorted(set(d["page"] for d in detections_found if d["class"] == "table"))
            figure_pages = sorted(set(d["page"] for d in detections_found if d["class"] in ("picture", "figure")))
            formula_pages = sorted(set(d["page"] for d in detections_found if d["class"] in ("isolate_formula", "formula_caption")))
            
            if table_pages:
                print(f"tables found on pages: {table_pages}")
            if figure_pages:
                print(f"figures found on pages: {figure_pages}")
            if formula_pages:
                print(f"formulas found on pages: {formula_pages}")

        return detections_found



    def transform_table_to_html(self, table_image_paths: list[str], headline: Optional[str] = None, output_dir: str = "./file_output"):

        class TableHTMLResponse(BaseModel):
            """HTML table extracted from image(s)."""
            html: str = Field(description="Complete HTML <table> element with all rows and columns")
            page_range: str = Field(description="Page range of the source table, e.g. '159-169'")
            row_count: int = Field(description="Number of data rows in the table (excluding header)")
            column_count: int = Field(description="Number of columns in the table")


        base_system_prompt = """

        # Role
        You are a document analysis expert specializing in high-fidelity table extraction and structural merging. Your goal is to convert document images into clean, valid HTML.

        # Task
        Analyze the provided image(s) and convert them into a **single** HTML table.

        # Rules for Multi-Page Merging
        *   **Continuous Flow:** If multiple images are provided, treat them as a single continuous dataset.
        *   **Heuristic for Splitting:** Only create separate `<table>` tags if the column structure (number of columns or alignment) changes significantly, or if the bold text represents a completely new section title rather than a continuation of the previous data (often all columns are filled with bold text).

        # Formatting & Data Integrity
        *   **Structure:** Output a complete `<table>` element with a single `<thead>` and one `<tbody>`.
        *   **Tags:** Use `<th>` for header cells and `<td>` for data cells.
        *   **Exactness:** Preserve all data exactly as shown. Do not omit, summarize, or modify any values.
        *   **Language:** The text is in the Slovak language—preserve all original characters and diacritics (e.g., č, š, ž, ť, ľ) exactly as they appear.
        *   **Clean Output:** Do not include CSS or markdown code blocks unless requested; provide the raw HTML structure.
        
        # IMPORTANT
        * Keep layout the same, if there is an empty cell in the image table, add in the html an empty cell `<td></td>`.
        * If you spot that, there is a new table headers (in every column is bold text), close the table with `</table>` and start a new table
        """

        continuation_rules = """

        # Continuation Mode
        *   The user message contains the HTML extracted from PREVIOUS pages of this SAME table. The new images are continuation pages.
        *   Output the COMPLETE merged HTML `<table>` containing ALL rows from the previous HTML PLUS new rows extracted from the new images, in order.
        *   Do NOT drop, rephrase, summarize, or re-format any prior row.
        *   If you spot that, there is a new table headers (in all columns is bold text), close the table with `</table>` and start a new table
        """

        CHUNK_SIZE = 2

        structured_model = self.openai_client.with_structured_output(
            schema=TableHTMLResponse.model_json_schema(),
            method="function_calling",
            include_raw=True,
        )

        os.makedirs(output_dir, exist_ok=True)
        
        

        def invoke_chunk(chunk_paths: list[str], previous_html: str | None) -> dict[str, Any]:
            system_prompt = base_system_prompt + (continuation_rules if previous_html else "")

            if previous_html:
                user_text = (
                    f"PREVIOUS HTML (from earlier pages):\n{previous_html}\n\n"
                )
            else:
                user_text = (
                    f"Convert the following {len(chunk_paths)} table image(s). Preserve all data exactly as shown."
                )

            content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
            for path in chunk_paths:
                with open(path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_data}"}
                })

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=cast(list[str | dict[Any, Any]], content_parts)),
            ]

            for attempt in range(3):
                try:
                    result = cast(dict[str, Any], structured_model.invoke(messages))
                    if result is None:
                        raise RuntimeError("model returned null")
                    parsed = result.get("parsed")
                    if not parsed:
                        raise RuntimeError("model returned no parsed output")
                    if not parsed.get("html"):
                        print(f"[invoke_chunk] empty html (attempt {attempt + 1}/3), retrying...")
                        continue
                    return result
                except Exception as e:
                    if attempt < 2:
                        print(f"TPM limit reached (attempt {attempt + 1}/3, error: {e}), sleeping 60s...")
                        time.sleep(60)
                    else:
                        print(f"[invoke_chunk] all 3 attempts failed: {e}")
                        raise

            raise RuntimeError("invoke_chunk: empty html after 3 attempts")



        try:
            if len(table_image_paths) <= CHUNK_SIZE:
                result = invoke_chunk(table_image_paths, previous_html=None)
                if result is None:
                    return None
            else:
                chunks = [table_image_paths[i:i + CHUNK_SIZE] for i in range(0, len(table_image_paths), CHUNK_SIZE)]
                print(f"[transform_table_to_html] {len(table_image_paths)} images > {CHUNK_SIZE}, splitting into {len(chunks)} chunks")
                result = None
                previous_html: str | None = None
                for idx, chunk in enumerate(chunks):
                    chunk_info = f"chunk {idx + 1}/{len(chunks)}"
                    print(f"[transform_table_to_html] {chunk_info} ({len(chunk)} images)")
                    result = invoke_chunk(chunk, previous_html=previous_html)
                    if result is None:
                        return None
                    chunk_parsed = result.get("parsed")
                    if not chunk_parsed:
                        print(f"[transform_table_to_html] {chunk_info} produced no parsed output; aborting")
                        return None
                    previous_html = chunk_parsed.get("html", "")
        except Exception as e:
            print(f"[transform_table_to_html] SKIPPING table for {table_image_paths!r} — extraction failed: {type(e).__name__}: {e}")
            return None

        if result is None:
            return None

        if result.get("parsing_error"):
            print(f"[transform_table_to_html] parsing error: {result['parsing_error']}")
            return None

        parsed = result.get("parsed")
        if not parsed:
            print(f"[transform_table_to_html] no parsed output for {table_image_paths!r}")
            return None

        page_numbers = sorted(
            int(m.group(1))
            for path in table_image_paths
            if (m := _PAGE_NUM_RE.match(os.path.basename(path)))
        )
        if page_numbers:
            page_range_str = f"{page_numbers[0]}-{page_numbers[-1]}" if len(page_numbers) > 1 else str(page_numbers[0])
        else:
            page_range_str = "unknown"

        html_path = os.path.join(output_dir, f"table_pages{page_range_str}.html")

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(_HTML_HEADER)
            if headline:
                f.write(f"<h2>{headline}</h2>\n")
            f.write(parsed.get("html", ""))
            f.write(_HTML_FOOTER)

        print("created and saved table")

        return {"html_path": html_path, "response": parsed}


    def transform_html_to_graph_document(self, html_content: str, page_range: str, document_id: str) -> GraphDocument:
        """
        Transform an HTML table into a GraphDocument using LLM-based extraction.

        The LLM analyzes the table structure and produces a hierarchical tree
        of nodes and relationships representing the table's logical content.

        Args:
            html_content: Raw HTML string of the table (from transform_table_to_html response)
            page_range: Page range string for identification (e.g. '159-169')

        Returns:
            GraphDocument with nodes and relationships extracted from the table
        """

        class TableGraphNode(BaseModel):
            """A node extracted from a table."""
            id: str = Field(description="Unique identifier for the node")
            label: str = Field(description="Type/label of the node (e.g. Section, Chapter, Item)")
            properties: dict = Field(description="Properties of the node, including 'name' and key-values (column_head: column value) from row")

        class TableGraphRelationship(BaseModel):
            """A relationship between two nodes extracted from a table."""
            source_node_id: str = Field(description="ID of the source/parent node")
            target_node_id: str = Field(description="ID of the target/child node")
            relation: str = Field(description="Type of relationship (e.g. HAS_SECTION, HAS_CHAPTER, CONTAINS)")
            source_node_type: str = Field(description="Type/label of the source node")
            target_node_type: str = Field(description="Type/label of the target node")
            properties: dict = Field(default_factory=dict, description="Optional relationship properties")

        class TableGraphResponse(BaseModel):
            """Knowledge graph extracted from an HTML table."""
            nodes: list[TableGraphNode] = Field(description="All nodes extracted from the table")
            relationships: list[TableGraphRelationship] = Field(description="All relationships between extracted nodes")

        system_prompt = """Si expert na transformaciu HTML tabuliek do struktury znalostneho grafu podla striktne definovaneho modelu.

Tvojim cielom je previesť HTML tabulku na jednoduchu stromovu strukturu:

- 1 rodicovsky uzol typu `Tabulka`
- viacero potomkov typu `Riadok` (kazdy riadok tabulky = 1 uzol)

---

# HLAVNY MODEL

## 1. RODICOVSKY UZOL
- Typ: `Tabulka`
- ID: poskytnute id v user prompte
- Vlastnosti:
  - `name`: nazov tabulky (ak nie je dostupny, pouzi genericky nazov napr. "tabulka")

## 2. RIADKOVE UZLY
- Typ: `Riadok`
- Kazdy riadok (<tr> v <tbody>) = jeden uzol
- ID:
  - musi byt odvodeny z NAJVYRAZNEJSEJ hodnoty v riadku (napr. kod, cislo, oznacenie)
  - priklad: `kod_25`, `polozka_0504`
- Vlastnosti:
  - kazdy stlpec = 1 key-value par:
    - key = hlavicka stlpca (z <th>)
    - value = hodnota bunky (<td>) v danom riadku

## 3. VZTAHY
- Kazdy Riadok je spojeny s Tabulkou:
  - `(:Tabulka)-[:MA_RIADOK]->(:Riadok)`

---

# PRAVIDLA

- Pouzivaj PRESNE iba typy:
  - `Tabulka`
  - `Riadok`
- Pouzivaj PRESNE iba vztah:
  - `MA_RIADOK`

- Zachovaj text PRESNE tak, ako je v tabulke (bez prekladu alebo uprav)
- Vsetky vystupy (id, keys, values) pis BEZ DIAKRITIKY

- Kazdy uzol MUSI mat:
  - `id`
  - `name`

- `name`:
  - pre Tabulka: nazov tabulky
  - pre Riadok: pouzi rovnaku hodnotu ako ID alebo hlavny identifikator riadku

---

# MAPOVANIE STLPCOV

- Najprv extrahuj hlavicky (<th>)
- Potom pre kazdy riadok:
  - prirad hodnoty buniek podla poradia hlaviciek
  - vytvor properties:
    - {hlavicka_1: hodnota_1, hlavicka_2: hodnota_2, ...}

---

# SPECIALNE PRIPADY

- Ak bunka chyba (colspan/rowspan):
  - pouzi najblizsiu relevantnu hodnotu z kontextu
- Ak riadok nema jasny identifikator:
  - pouzi kombinaciu viacerych stlpcov

---

# KONTROLA

Pred vystupom over:

1. Existuje PRESNE 1 uzol typu Tabulka?
2. Kazdy riadok ma vlastny uzol typu Riadok?
3. Kazdy Riadok ma vztah `MA_RIADOK` z Tabulka?
4. Su vsetky stlpce mapovane ako properties?
5. Su vsetky texty bez diakritiky?
6. Ma kazdy uzol `id` a `name`?"""


        user_prompt = f"""Transformuj nasledujucu HTML tabulku na znalostny graf podla definovaneho modelu:

- 1 uzol typu `Tabulka`
- viacero uzlov typu `Riadok` (kazdy riadok = 1 uzol)
- kazdy Riadok obsahuje properties mapovane zo stlpcov
- vsetky Riadky su spojene s Tabulkou cez `MA_RIADOK`

---

# POSTUP

1. Extrahuj hlavicky stlpcov (<th>)
2. Pre kazdy riadok (<tr>):
   - vytvor uzol typu `Riadok`
   - nastav ID podla najdolezitejsieho udaju (napr. kod)
   - prirad properties: hlavicka → hodnota
3. Vytvor vztahy:
   - `Tabulka -> MA_RIADOK -> Riadok`

---

# ID RODICA
- tento retazec pouzi ako id rodica (typ: Tabulka): table_{page_range}_{document_id}

# HTML TABULKA:
{html_content}"""
        
        # Create and run the agent
        agent = create_agent(
            model=self.openai_client,
            response_format=ProviderStrategy(schema=TableGraphResponse.model_json_schema()),  # type: ignore[arg-type]
            system_prompt=system_prompt
        )
        


        

        data = None
        for attempt in range(3):
            try:
                # LLM response is expected to be a plain dict with "nodes" and "relationships"
                response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
                data = response["structured_response"]

                if data is None:
                    raise RuntimeError("model returned null")

                # Optional sanity check on content
                nodes = data.get("nodes") or []
                relationships = data.get("relationships") or []
                print(
                    f"Table graph extraction attempt {attempt + 1}/3 for pages {page_range}: "
                    f"{len(nodes)} nodes, {len(relationships)} relationships"
                )

                break

            except Exception as e:
                if attempt < 2:
                    print(f"TPM limit or error (attempt {attempt + 1}/3, error: {e}), sleeping 60s...")
                    time.sleep(60)
                else:
                    print(f"[table_graph] all 3 attempts failed: {e}")
                    raise


        if data is None:
            raise RuntimeError("table_graph: empty nodes/relationships after 3 attempts")


        source_doc = Document(page_content=html_content, metadata={"page_range": page_range, "source": "table"})
        graph_document = self._convert_table_to_graph_document(data, f"table_{page_range}", source_doc, document_id)
        

        return graph_document


    @staticmethod
    def group_table_detections(detections: list[dict]) -> list[list[dict]]:
        """Group consecutive table detections into multi-page table groups."""
        table_detections = sorted(
            [d for d in detections if d["class"] == "table"],
            key=lambda d: d["page"]
        )

        if not table_detections:
            return []

        groups = []
        current_group = [table_detections[0]]

        for det in table_detections[1:]:
            if det["page"] - current_group[-1]["page"] <= 1:
                current_group.append(det)
            else:
                groups.append(current_group)
                current_group = [det]
        groups.append(current_group)

        return groups


    def _extract_page_headline(self, page_text: str, max_lines: int = 15) -> Optional[str]:
        """
        Return the first ALL-CAPS headline line found near the top of the page,
        or None if no such line exists in the first `max_lines` non-empty lines.

        A headline line is defined as a standalone line (surrounded by \\n on
        both sides) where:
          - line is non-empty after .strip()
          - has >= 2 whitespace-separated words
          - has >= 10 characters
          - has at least one alphabetic character
          - every alphabetic character is uppercase (Slovak diacritics OK)

        Page chrome ("Strana 159", "Príloha č. 7", "k zákonu č. ...",
        "DynamicResources\\...") naturally fails these tests because it contains
        lowercase letters, so no explicit blocklist is needed.
        """
        for raw in page_text.split("\n")[:max_lines]:
            line = raw.strip()
            if not line:
                continue
            alpha_chars = [c for c in line if c.isalpha()]
            if not alpha_chars or not all(c.isupper() for c in alpha_chars):
                continue
            if len(line) < 10 or len(line.split()) < 2:
                continue
            return line
        return None


    def _split_group_by_headlines(
        self,
        group: list[dict],
        page_text_map: dict[int, str],
    ) -> list[tuple[list[dict], Optional[str]]]:
        """
        Split a table group into sub-groups at pages whose top region contains
        an ALL-CAPS headline. Returns a list of (sub_group, headline_or_None).

        Walk pages in order:
          - first page's headline (if any) becomes the first sub-group's headline
          - any subsequent page with a headline closes the current sub-group
            and starts a new one anchored on that page
          - pages without a headline append to the current sub-group
        """
        ordered = sorted(group, key=lambda d: d["page"])

        sub_groups: list[tuple[list[dict], Optional[str]]] = []
        current: list[dict] = []
        current_headline: Optional[str] = None

        for det in ordered:
            page_text = page_text_map.get(det["page"], "")
            headline = self._extract_page_headline(page_text)

            if headline and current:
                sub_groups.append((current, current_headline))
                current = []
                current_headline = headline
            elif headline and not current:
                current_headline = headline

            current.append(det)

        if current:
            sub_groups.append((current, current_headline))

        return sub_groups


    # OCR to LaTeX and then to GraphDocument

    def get_formulas(self, assets_dir: str = "./file_output/tables_and_formulas") -> list[dict]:
        """
        Load all detected formula PNGs from the assets directory.

        Returns a list of dicts with image_path, page, index, and detection_confidence
        parsed from the filename pattern: page{N}_isolate_formula_{i}_conf{score}.png
        """
        formulas: list[dict] = []
        for path in sorted(Path(assets_dir).glob("*isolate_formula*.png")):
            m = re.match(r"page(\d+)_isolate_formula_(\d+)_conf([0-9.]+)\.png$", path.name)
            if not m:
                continue
            formulas.append({
                "image_path": str(path),
                "page": int(m.group(1)),
                "index": int(m.group(2)),
                "detection_confidence": float(m.group(3)),
            })
        return formulas


    def transform_formula(self, formula: dict) -> dict:
        """
        Send one formula image to Gemini Flash and transcribe it to LaTeX.

        Returns a dict with keys "latex" (str) and "confidence" (float).
        """
        
        class FormulaLatexResponse(BaseModel):
            """LaTeX transcription of a formula image, with the model's confidence."""
            latex: str = Field(description="LaTeX representation of the formula, no surrounding $ delimiters")
            confidence: float = Field(description="Confidence in the transcription, 0.0 to 1.0")
    
    
        system_prompt = (
            "You are a math/formula transcription expert. "
            "Convert the formula in the image into valid LaTeX. "
            "Output ONLY the LaTeX body — no surrounding $ delimiters, "
            "no \\begin{equation} / \\end{equation} wrapper. "
            "Also report your confidence in the transcription as a float in [0, 1]."
        )

        with open(formula["image_path"], "rb") as f:
            img_data = base64.b64encode(f.read()).decode()

        content_parts: list[dict[str, Any]] = [
            {"type": "text", "text": "Transcribe the formula in this image to LaTeX."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}},
        ]

        structured_model = self.openai_client.with_structured_output(FormulaLatexResponse)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=cast(list[str | dict[Any, Any]], content_parts)),
        ]

        typed_response: FormulaLatexResponse | None = None
        for attempt in range(3):
            try:
                response = structured_model.invoke(messages)
                if response is None:
                    raise RuntimeError("model returned null")
                candidate = cast(FormulaLatexResponse, response)
                if not candidate.latex:
                    print(f"[transform_formula] empty latex (attempt {attempt + 1}/3), retrying...")
                    continue
                typed_response = candidate
                break
            except Exception as e:
                if attempt < 2:
                    print(f"TPM limit reached (attempt {attempt + 1}/3, error: {e}), sleeping 60s...")
                    time.sleep(60)
                else:
                    print(f"[transform_formula] all 3 attempts failed: {e}")
                    raise

        if typed_response is None:
            raise RuntimeError("transform_formula: empty latex after 3 attempts")

        return typed_response.model_dump()


    def _convert_formula_to_node(
        self,
        formula: dict,
        response: dict,
        i: int,
        document_id: str,
    ) -> Node:
        """Build a Vzorec Node from a formula detection + LLM transcription."""
        node_id = f"formula_{i}_{document_id}"
        return Node(
            id=node_id,
            type="Vzorec",
            properties={
                "name": node_id,
                "document_id": document_id,
                "latex": response["latex"],
                "transcription_confidence": response["confidence"],
                "detection_confidence": formula["detection_confidence"],
                "page": formula["page"],
            },
        )


    def convert_formulas_to_graph(
        self,
        formula_nodes: list[Node],
        document_id: str
    ) -> Optional[GraphDocument]:
        """
        Bundle formula Nodes into a single GraphDocument with each formula
        connected to the Document node via :IN_DOCUMENT.

        Returns None when there are no formula nodes.
        """
        if not formula_nodes:
            return None

        document_node = Node(id=document_id, type="Document")

        relationships = [
            Relationship(source=fn, target=document_node, type="IN_DOCUMENT")
            for fn in formula_nodes
        ]

        return GraphDocument(
            nodes=[document_node, *formula_nodes],
            relationships=relationships,
            source=Document(page_content="", metadata={"id": document_id}),
        )



    def load_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        print("PDF loaded successfully.")
        return loader.load()
    





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
        # Uloha
        Analyzuj nasledujuci text a identifikuj vsetky odlisne typy uzlov a typy vztahov.

        # Instrukcie

        ## 1. Typy uzlov
        - Pouzi vseobecne, ale rozlisitelne oznacenia
        - Zahrn aj implicitne pravne a financne koncepty

        ## 2. Typy vztahov
        - Pouzi nadcasove a genericke oznacenia
        - Zahrn aj normativne a procesne vztahy

        ## 3. Pravna struktura
        - Zachyt entity ako zakon, paragraf, odsek
        - Zachyt mechanizmy, vypocty a casove ramce

        ## 4. Validacia
        - Odstran typy, ktore nedavaju zmysel v pravnom kontexte
        - Dopln chybajuce centralne koncepty, ak mechanizmus nie je pokryty

        # Text
        {text}
        """

        # Create and run the agent
        agent = create_agent(
            model=self.openai_client,
            response_format=ProviderStrategy(schema=response_schema_for_odd),  # type: ignore[arg-type]
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
        documents: List[Document] | Document,
        max_concurrent: int = 5,
        write_json: bool = False,
        name: str = "",
    ) -> List[Schema]:
        """
        Asynchronously process documents to extract schemas.

        Per-chunk failures are tolerated: chunks that exhaust their retries
        return None and are filtered out, so a single bad chunk does not
        abort the whole stage.

        Args:
            documents: List of documents to process
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of Schema (only successfully processed chunks).
        """
        if isinstance(documents, Document):
            documents = [documents]
        
        if not documents:
            print("[async_open_domain_detection] empty document list; returning []")
            return []

        print("creating tasks: OPEN-DOMAIN DETECTION")
        semaphore = asyncio.Semaphore(max_concurrent)
        pause_event = asyncio.Event()
        pause_event.set()
        pause_lock = asyncio.Lock()

        # Shared accumulator for incremental persistence — every successful
        # chunk re-dumps the growing list to the same `<name>_odd_<ts>.json`
        # so a crash mid-stage still leaves the latest successes on disk.
        write_lock = asyncio.Lock()
        odd_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S') if write_json else None
        completed: List[Schema] = []

        async def limited(i: int, doc: Document) -> Optional[Schema]:
            for attempt in range(3):
                await pause_event.wait()
                async with semaphore:
                    try:
                        result = await self.open_domain_detection(i, doc)
                        if write_json and result is not None:
                            async with write_lock:
                                completed.append(result)
                                odd_to_json(list(completed), name=name, timestamp=odd_timestamp)
                        return result
                    except Exception as e:
                        if attempt < 2:
                            async with pause_lock:
                                if pause_event.is_set():
                                    pause_event.clear()
                                    print(f"ODD: error on chunk {i} ({e}), pausing all processing for 60s...")
                                    await asyncio.sleep(60)
                                    pause_event.set()
                        else:
                            print(f"ODD chunk {i}: all 3 attempts failed ({e}); skipping")
                            return None
            return None

        tasks = [
            asyncio.create_task(limited(i, doc))
            for i, doc in enumerate(documents)
        ]

        res = await asyncio.gather(*tasks, return_exceptions=True)

        successes: List[Schema] = []
        failures = 0
        for i, r in enumerate(res):
            if isinstance(r, BaseException):
                failures += 1
                print(f"ODD chunk {i}: unexpected exception leaked through gather: {r}")
            elif r is None:
                failures += 1
            else:
                successes.append(r)

        if failures:
            print(f"[async_open_domain_detection] {failures}/{len(res)} chunks failed; {len(successes)} succeeded")

        return successes





    def schema_refinement(
        self,
        odd_schema: Schema,
        existing_schema: Optional[Schema] = None,
    ) -> Dict[str, Any]:
        """
        Refine and consolidate extracted schema information across documents,
        ensuring consistency and resolving conflicts.

        Args:
            odd_schema: Raw schema detected from open-domain detection.
            existing_schema: Existing graph schema (None or empty falls back
                to the predefined Slovak legal ontology).

        Returns:
            Dict with keys:
              - node_types: List[str]
              - relationship_types: List[str]
              - merge_log: {node_types: Dict[str, List[str]], relationship_types: Dict[str, List[str]]}

        Raises:
            ValueError: if odd_schema is missing or has no node types.
            RuntimeError: if the LLM fails to produce a valid response after 3 attempts.
        """

        if odd_schema is None:
            raise ValueError("schema_refinement: odd_schema is required")
        if not isinstance(odd_schema, Schema):
            raise ValueError(
                f"schema_refinement: odd_schema must be Schema, got {type(odd_schema).__name__}"
            )

        if existing_schema is None or (
            existing_schema.nodes == [] and existing_schema.relationships == []
        ):
            existing_schema = Schema(
                    nodes = ["FyzickaOsoba", "PravnickaOsoba", "Sud", "Zakon", "Vyhlaska", "Nariadenie", "Zmluva", "Zodpovednost", "Pravo", "Povinnost", "Paragraf", "Lokacia", "Urad", "Odsek", "Vozidlo", "Cislo", "Datum", "Pismeno", "Urad", "Banka", "Tuzemsko"],
                    relationships = ["DEFINUJE", "UPRAVUJE", "DOPLNUJE", "PODMIENUJE", "RUSI", "JE_PODLA", "JE_OSLOBODENE_OD_DANE", "JE_PREDMETOM_DANE", "NIE_JE_PREDMETOM_DANE", "ODKAZUJE", "VZTAHUJE_SA_NA", "OBSAHUJE"]
                )
        
        
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
        # ÚLOHA
        Spresni surovú schému detekovanú z otvorenej domény a integruj ju so základnou ontológiou (ak je dostupná).

        # KONTEXTOVÉ PRÍDAVKY
        - Do zoznamu uzlov (node_types) POVINNE pridaj nový typ: **Paragraf**.

        # POKYNY PRE Sémantiku
        - **Nezlučuj agresívne**: Ak existuje nuansa (napr. Banka vs. StatnyOrgan), ponechaj ich oddelené, pokiaľ základná ontológia neurčuje inak.
        - **Dôsledná ASCII normalizácia**: Skontroluj, či vo výsledku neostalo žiadne "š, č, ž, ý, á, í, é, ú, ä, ň, ť, ď, ľ, ô".
        - Snaz sa vytvorit schemu z 50 typov uzlov a 50 typov hran.

        # PROCES UVAŽOVANIA (Chain of Thought)
        Pred vygenerovaním JSONu vykonaj internú analýzu:
        1. Identifikuj duplicity spôsobené preklepmi alebo diakritikou (napr. "Dan" a "Daň").
        2. Porovnaj surové dáta so základnou ontológiou.
        3. Skontroluj, či sa nepokúšaš zlúčiť nesúvisiace ontologické kategórie (napr. dokument a osoba).
        4. Over, či sú všetky vzťahy v UPPER_SNAKE_CASE.

        # VSTUPNÉ DÁTA
        ## ZÁKLADNÁ ONTOLÓGIA
        ### Typy uzlov:
        [{", ".join(existing_schema.nodes) if existing_schema else "Neposkytnuta"}]
        ### Typy vzťahov:
        [{", ".join(existing_schema.relationships) if existing_schema else "Neposkytnute"}]

        ## SUROVÁ SCHÉMA (DETEKCIA)
        ### Typy uzlov:
        [{", ".join(odd_schema.nodes)}]
        ### Typy vzťahov:
        [{", ".join(odd_schema.relationships)}]
        """


        # function_calling (not json_schema/strict): MergeLog's Dict[str, List[str]]
        # fields are open-ended maps, which strict structured-outputs cannot
        # represent (it demands additionalProperties:false on every object).
        structured_model = self.openai_thinking.with_structured_output(
            schema=SchemaRefinementResponse.model_json_schema(),
            method="function_calling",
        )
        
        
        # structured_model = self.gemini_client.with_structured_output(
        #     schema=SchemaRefinementResponse.model_json_schema(), method="json_schema"
        # )

        messages = [
            SystemMessage(content=system_prompt_for_schema_refinement),
            HumanMessage(content=user_prompt),
        ]

        data: Optional[Dict[str, Any]] = None
        for attempt in range(3):
            try:
                response = cast(dict[str, Any], structured_model.invoke(messages))
                if response is None:
                    raise RuntimeError("model returned null")
                if not isinstance(response, dict):
                    raise RuntimeError(
                        f"model returned {type(response).__name__}, expected dict"
                    )
                if not response.get("node_types") or not response.get("relationship_types"):
                    print(f"[schema_refinement] empty node_types or relationship_types (attempt {attempt + 1}/3), retrying...")
                    continue

                merge_log = response.get("merge_log")
                if not isinstance(merge_log, dict):
                    print(f"[schema_refinement] merge_log missing or wrong type (attempt {attempt + 1}/3); patching with empty defaults")
                    response["merge_log"] = {"node_types": {}, "relationship_types": {}}
                else:
                    if not isinstance(merge_log.get("node_types"), dict):
                        merge_log["node_types"] = {}
                    if not isinstance(merge_log.get("relationship_types"), dict):
                        merge_log["relationship_types"] = {}

                data = response
                break
            except Exception as e:
                if attempt < 2:
                    print(f"TPM limit reached (attempt {attempt + 1}/3, error: {e}), sleeping 60s...")
                    time.sleep(60)
                else:
                    print(f"[schema_refinement] all 3 attempts failed: {e}")
                    raise

        if data is None:
            raise RuntimeError("schema_refinement: empty node_types or relationship_types after 3 attempts")

        print(data)

        return data



    
    
    
    async def schema_driven_extraction(
        self,
        i: int,
        document: Document,
        schema: Schema,
        document_id: str
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
        metadata = document.metadata or {}

        # `path` is only present on chunks emitted by Chunker (structured law body).
        # Trailing chunks from RecursiveCharacterTextSplitter have just {page, source},
        # so we gate on the key, not on metadata truthiness.
        #         {"You are in paragraph: " + path_segments[0] if path_segments else ""}
        #         {"This text represents section: " + path_str if path_str else ""}
        path_segments: list[str] = metadata.get("path") or []
        path_str = ""
        path_label = ""
        if path_segments:
            path_str = "Paragraf " + path_segments[0]
            path_label = "Paragraf"
            if len(path_segments) > 1:
                path_str += " Odsek " + path_segments[1]
                path_label = "Odsek"
            if len(path_segments) > 2:
                path_str += " Pismeno " + path_segments[2]
                path_label = "Pismeno"
                

        user_prompt = f"""
        Extract entities and relationships from the Slovak legal text strictly according to the provided schema.

        # CONTEXT
        {"You are currently in Paragraf: " + path_segments[0] if path_segments else ""}
        {"Section context (current section): " + path_str if path_str else ""}

        # RULES
        - use ONLY exact schema types
        - no renaming or invented types
        - prefer most specific valid type
        - remove all diacritics
        - use only Slovak language
        - Node IDs must be in Title Case
        - Relationship labels must be in SCREAMING_SNAKE_CASE
        - decompose entities and relations into atomic legal units whenever valid
        - if decomposition reduces legal meaning → keep compound legal concept
        - skip unsupported information


        # LEGAL CONTEXT
        If text contains:
        - `odsek` without explicit paragraf:
        inherit active paragraf context

        # SCHEMA
        Entities: {", ".join(schema.nodes)}

        Relationships: {", ".join(schema.relationships)}

        # TEXT
        {text}

        # CHECK
        - valid schema types only
        - most specific types used
        - full legal hierarchy in IDs
        - odsek inheritance resolved
        - odsek ranges atomically decomposed
        - detailed legal entities included
        """

        # Create and run the agent
        agent = create_agent(
            model=self.openai_client,
            response_format=ProviderStrategy(schema=response_schema_for_sde),  # type: ignore[arg-type]
            system_prompt=system_prompt_for_sde
        )
        response = await agent.ainvoke({"messages": [{"role": "user", "content": user_prompt}]})
        
        # structured_llm = self.openai_client.with_structured_output(
        #     response_schema_for_sde,
        #     method="json_schema",
        #     strict=True,
        # )

        # response = await structured_llm.ainvoke([
        #     SystemMessage(content=system_prompt_for_sde),
        #     HumanMessage(content=user_prompt),
        # ])

        # structured_response is already a dict when using ProviderStrategy
        data = response["structured_response"]
        
        print(data)

        # Convert to graph document with validation and formatting
        graph_document = self._convert_to_graph_document(data=data, i=i, document=document, document_id=document_id, section_id=path_str, section_label=path_label)

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
        documents: List[Document] | Document,
        schema: Schema,
        document_id: str,
        max_concurrent: int = 5,
        write_json: bool = False,
        name: str = "",
    ) -> List[GraphDocument]:
        """
        Asynchronously process documents to extract graph documents.

        Per-chunk failures are tolerated: chunks that exhaust their retries
        are dropped, so a single bad chunk does not abort the whole stage.

        Args:
            documents: List of documents to process
            schema: Allowed node and relationship types (must be non-empty)
            document_id: Identifier for the source document
            max_concurrent: Maximum number of concurrent API calls

        Returns:
            List of GraphDocuments (only successfully processed chunks).
        """

        if isinstance(documents, Document):
            documents = [documents]

        if not documents:
            print("[async_schema_driven_extraction] empty document list; returning []")
            return []
        if schema is None or not schema.nodes:
            print("[async_schema_driven_extraction] schema has no node types; returning []")
            return []

        print("creating tasks: SCHEMA-DRIVEN EXTRACTION")
        semaphore = asyncio.Semaphore(max_concurrent)
        pause_event = asyncio.Event()
        pause_event.set()
        pause_lock = asyncio.Lock()

        # Same incremental-write pattern as ODD: append-then-rewrite under a
        # lock; the timestamp is fixed for the stage so every dump lands in
        # the same `<name>_sde_<ts>.json` file as it grows.
        write_lock = asyncio.Lock()
        sde_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S') if write_json else None
        completed: list[GraphDocument] = []

        def is_failed(result) -> bool:
            return result is None or isinstance(result, BaseException) or not getattr(result, "nodes", None)

        async def limited(i: int, doc: Document) -> Optional[GraphDocument]:
            for attempt in range(3):
                await pause_event.wait()
                async with semaphore:
                    try:
                        result = await self.schema_driven_extraction(i, doc, schema, document_id)
                        if write_json and not is_failed(result):
                            async with write_lock:
                                completed.append(cast(GraphDocument, result))
                                sde_to_json(list(completed), name=name, timestamp=sde_timestamp)
                        return result
                    except Exception as e:
                        if attempt < 2:
                            async with pause_lock:
                                if pause_event.is_set():
                                    pause_event.clear()
                                    print(f"SDE: error on chunk {i} ({e}), pausing all processing for 60s...")
                                    await asyncio.sleep(60)
                                    pause_event.set()
                        else:
                            print(f"SDE chunk {i}: all 3 attempts failed ({e})")
                            return None
            return None

        tasks = [
            asyncio.create_task(limited(i, doc)) for i, doc in enumerate(documents)
        ]
        pass1 = await asyncio.gather(*tasks, return_exceptions=True)

        successful: list[GraphDocument] = []
        failed: list[tuple[int, Document]] = []
        for i, r in enumerate(pass1):
            if isinstance(r, BaseException):
                print(f"SDE chunk {i}: unexpected exception leaked through gather: {r}")
                failed.append((i, documents[i]))
            elif is_failed(r):
                failed.append((i, documents[i]))
            else:
                successful.append(cast(GraphDocument, r))

        if failed:
            print(f"SDE: retrying {len(failed)} failed chunks: {[i for i,_ in failed]}")
            retry_tasks = [asyncio.create_task(limited(i, doc)) for i, doc in failed]
            pass2 = await asyncio.gather(*retry_tasks, return_exceptions=True)

            still_failed: list[int] = []
            for (i, _), r in zip(failed, pass2):
                if isinstance(r, BaseException):
                    print(f"SDE chunk {i} (retry): unexpected exception: {r}")
                    still_failed.append(i)
                elif is_failed(r):
                    still_failed.append(i)
                else:
                    successful.append(cast(GraphDocument, r))

            if still_failed:
                print(f"SDE: {len(still_failed)} chunks still failed after retry: {still_failed}")
            else:
                print("SDE: all retried chunks recovered")

        return successful
    
    
    
    
    def tables_and_formulas(self, pdf_path: str, document_id: str, documents: list[Document], write_json: bool = False):
         # map 1-based human page number -> PyPDFLoader text (matches detection page convention)
        page_text_map = {
            doc.metadata.get("page", -1) + 1: doc.page_content
            for doc in documents
        }

        # detect tables in pdf
        detections = self.detect_tables(pdf_path)

        # group consecutive table detections into multi-page groups
        table_groups = self.group_table_detections(detections)

        # convert each table group to HTML and then to GraphDocument
        table_graph_docs = []
        table_pages_to_exclude = set()

        for group in table_groups:
            # split a detection group at pages whose top contains a new ALL-CAPS headline
            sub_groups = self._split_group_by_headlines(group, page_text_map)

            for sub_group, headline in sub_groups:
                image_paths = [d["image_path"] for d in sub_group]
                pages = sorted(d["page"] for d in sub_group)

                result = self.transform_table_to_html(table_image_paths=image_paths, headline=headline)
                if result is None:
                    print(f"Skipping table on pages {pages} — extraction returned None")
                    continue
                print(f"Saved: {result['html_path']}")

                # access HTML from in-memory response
                response = result['response']
                html = response.get("html", "") if isinstance(response, dict) else response.html if hasattr(response, "html") else str(response)

                page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0])

                # transform HTML table into GraphDocument via LLM
                table_gd = self.transform_html_to_graph_document(html, page_range, document_id)
                table_graph_docs.append(table_gd)

                # collect interior table pages of THIS sub-group to exclude from text processing
                # keep first and last pages (may have text above/below the table)
                if len(pages) > 2:
                    table_pages_to_exclude.update(pages[1:-1])

        
            
            
        # ----- FORMULAS ----
        formulas = self.get_formulas()

        formula_nodes: list[Node] = []
        for i, formula in enumerate(formulas):
            response = self.transform_formula(formula)
            formula_nodes.append(
                self._convert_formula_to_node(formula, response, i, document_id)
            )

        formula_graph_doc = self.convert_formulas_to_graph(formula_nodes, document_id)
        print(f"Processed {len(formulas)} formula(s) into {0 if formula_graph_doc is None else len(formula_nodes)} node(s).")

        if write_json:
            table_to_json(table_graph_docs)
            formula_to_json(formula_graph_doc)

        return table_graph_docs, formula_graph_doc, table_pages_to_exclude
    
    
    
    
    def build_vector_stores(self):
        # A Neo4j vector index is bound to a single node label, so we can't index
        # "several labels" or "all labels except X" directly. Instead we re-stamp a
        # shared label onto each group on every build and index that:
        #   __Chunk__  -> text-bearing structural nodes, embedded from `text`
        #   __Entity__ -> every other node (entities + Document), embedded from `id`
        # The SET/REMOVE pair keeps the two groups mutually exclusive and self-
        # correcting: it doesn't rely on the legacy __Entity__ label (which new
        # baseEntityLabel=False writes never receive), it actively reapplies it and
        # filters the chunk types out of it.
        chunk_labels = ["Chunk", "Paragraf", "Odsek", "Pismeno", "Bod"]
        is_chunk = " OR ".join(f"n:`{label}`" for label in chunk_labels)

        self.graph.query(
            f"MATCH (n) WHERE {is_chunk} SET n:__Chunk__ REMOVE n:__Entity__"
        )
        self.graph.query(
            f"MATCH (n) WHERE NOT ({is_chunk}) SET n:__Entity__ REMOVE n:__Chunk__"
        )

        # LangChain re-derives node_label from any index that already exists under
        # the same name (Neo4jVector.retrieve_existing_index), which would silently
        # pin a store to its old single label and ignore the relabeling above.
        # Dropping the indexes forces recreation on the new shared label; the
        # `embedding` properties stay on the nodes, so already-embedded nodes are
        # skipped and only the newly-covered ones get embedded.
        self.graph.query(f"DROP INDEX `{self._vector_store_nodes_name}` IF EXISTS")
        self.graph.query(f"DROP INDEX `{self._vector_store_chunk_name}` IF EXISTS")

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
        print("\n\nAdded embedded nodes into Vector database.\n\n")

        self.vector_store_chunks = Neo4jVector.from_existing_graph(
            embedding=self.embeddings,
            url=self._neo4j_uri,
            username=self._neo4j_user,
            password=self._neo4j_password,
            index_name=self._vector_store_chunk_name,
            node_label="__Chunk__",
            text_node_properties=["text"],
            embedding_node_property="embedding",
        )
        print("\n\nAdded embedded chunks into Vector database.\n\n")
        
        
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
            print("\n\nAdded embedded relationships into Vector database.\n\n")
            
        else:
            self.vector_store_relationships.add_documents(all_relationships)
            print("\n\nAdded embedded relationships into Vector database.\n\n")
    
    
    
    def get_document_id(self, pdf_path: str):
        return Path(pdf_path).stem


    # --------- PROCESS STRATEGY ----------
    # 1. tables
    # 2. open domain schema detection
    # 3. schema refinement
    # 4. schema guided extraction
    # 5. vector store from graph



    def process(self, pdf_path: str, name_of_chain: str = "chain", write_json: bool = False):
        
        # Load PDF documents
        documents = self.load_pdf(pdf_path)

        document_id = Path(pdf_path).stem

       
        table_graph_docs, formula_graph_doc, table_pages_to_exclude = self.tables_and_formulas(pdf_path, document_id, documents, write_json=write_json)
        
        if table_graph_docs is not None:
            self.graph.add_graph_documents(
                graph_documents=table_graph_docs,
                include_source=False,
                baseEntityLabel=False
            )

        # add formula-extracted graph document
        if formula_graph_doc is not None:
            self.graph.add_graph_documents(
                graph_documents=[formula_graph_doc],
                include_source=False,
                baseEntityLabel=False
            )
            
        
        
        # remove interior table pages from documents before ODD/SDE
        # PyPDFLoader uses 0-based page numbers, detections use 1-based
        if table_pages_to_exclude:
            documents = [
                doc for doc in documents
                if (doc.metadata.get("page", -1) + 1) not in table_pages_to_exclude
            ]
            print(f"Excluded {len(table_pages_to_exclude)} interior table pages from text processing: {sorted(table_pages_to_exclude)}")



        # ---- ODD ----
        # Stage failures abort the document. Per-chunk failures are tolerated
        # inside async_open_domain_detection (chunks dropped, others continue).
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunked_documents = splitter.split_documents(documents)
        if not chunked_documents:
            raise ValueError(f"[process] no chunks produced for ODD from {pdf_path}")

        extracted_schema_list = asyncio.run(
            self.async_open_domain_detection(
                chunked_documents,
                write_json=write_json,
                name=name_of_chain,
            )
        )
        if not extracted_schema_list:
            raise RuntimeError("[process] ODD produced no schemas (all chunks failed)")
        print(f"\n\nAll chunks processed into schema. ({len(extracted_schema_list)} schemas)\n\n")
        # ODD JSON is now written incrementally inside async_open_domain_detection
        # when write_json=True, so no end-of-stage dump is needed here.


        # ---- Refinement ----
        extracted_schema = Schema(
            nodes=list(set(node for schema in extracted_schema_list for node in schema.nodes)),
            relationships=list(set(rel for schema in extracted_schema_list for rel in schema.relationships))
        )
        if not extracted_schema.nodes:
            raise RuntimeError("[process] ODD union produced empty node set; aborting before refinement")

        # get_graph_schema is internally guarded — returns empty Schema on Neo4j read failure
        refined_schema = self.schema_refinement(
            odd_schema=extracted_schema,
            existing_schema=self.get_graph_schema(),
        )
        # schema_refinement raises on its own retry exhaustion — propagates to abort
        print(f"\n\nSchema refined.\n\n")
        if write_json:
            refinement_to_json(refined_schema, name=name_of_chain)  # save before next stage


        # ---- SDE ----
        chunker_result = Chunker().split_document(pdf_path, write_json=write_json)
        structured_chunks: list[Document] = chunker_result["chunks"]
        tree_graph = chunker_result["tree_graph"]
        last_page = chunker_result["last_page"]
        
        # add structural tree (Document -> Paragraf -> Odsek -> Pismeno -> Bod)
        self.graph.add_graph_documents(
            graph_documents=[tree_graph],
            include_source=False,
            baseEntityLabel=False
        )

        # Pages after `last_page` (1-indexed) hold post-paragraph content (annexes,
        # appendices) that the structural chunker doesn't cover. PyPDFLoader returns
        # pages in order, so slicing at `last_page` skips pages 1..last_page.
        # separators=["\n"] keeps splits near chunk_size and only breaks on newlines.
        if last_page is not None:
            documents = documents[last_page:]

        trailing_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n"],
        )
        trailing_chunks = trailing_splitter.split_documents(documents) if documents else []

        chunked_documents = structured_chunks + trailing_chunks
        if not chunked_documents:
            raise ValueError("[process] no chunks produced for SDE")

        refined_schema_obj = self._convert_to_schema(refined_schema)  # raises ValueError on bad shape
        if not refined_schema_obj.nodes:
            raise RuntimeError("[process] refined schema has empty nodes; aborting before SDE")

        graph_docs = asyncio.run(
            self.async_schema_driven_extraction(
                chunked_documents,
                schema=refined_schema_obj,
                document_id=document_id,
                write_json=write_json,
                name=name_of_chain,
            )
        )
        if not graph_docs:
            raise RuntimeError("[process] SDE produced no graph documents (all chunks failed)")
        print(f"\n\nAll chunks processed into graph documents. ({len(graph_docs)})\n\n")
        # SDE JSON is now written incrementally inside async_schema_driven_extraction
        # when write_json=True, so no end-of-stage dump is needed here.
        

        # add document chunk into graph documents
        graph_docs.append(self._add_document_chunk(graph_docs, document_id))



        # Add graph documents to Neo4j
        # dependency: APOC plugin in neo4j database
        self.graph.add_graph_documents(
            graph_documents=graph_docs,
            include_source=False,
            baseEntityLabel=False
        )
        
        self.graph.refresh_schema()
        # Remove __Entity__ label from Chunk nodes so the node vector store excludes them
        # self.graph.query("MATCH (n:Chunk:__Entity__) REMOVE n:__Entity__")
        # self.graph.query("MATCH (n:Document:__Entity__) REMOVE n:__Entity__")
        
        print("\n\nAdded Graph Documents into Graph database.\n\n")
        
        
        self.build_vector_stores()
            

        
        print("Knowledge graph and vector stores successfully updated in Neo4j!")
            