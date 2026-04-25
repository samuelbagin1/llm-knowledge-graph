import re
import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Generic, TypeVar, cast
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
from classes import Schema, ClassifiedDocument, Type, SubSentence, ReasoningStep
from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
import asyncio
from prompts import response_schema_for_sde, system_prompt_for_sde, response_schema_for_odd, system_prompt_for_odd, system_prompt_for_schema_refinement, response_schema_for_schema_refinement, system_prompt_for_segmentation, response_schema_for_segmentation, system_prompt_for_relation_retrieval, response_schema_for_relation_retrieval, system_prompt_for_inference,system_prompt_for_generating_query, response_schema_for_generating_query, system_prompt_ark_select_relation, system_prompt_ark_reasoning
from examples import examples_for_extraction
from pydantic import BaseModel, Field, SecretStr
import numpy as np
from to_json import odd_to_json, refinement_to_json, sde_to_json
from pdf2image import convert_from_path
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from PIL import Image
import base64
from langchain_core.messages import HumanMessage, SystemMessage


# Default node type when type is missing or empty
DEFAULT_NODE_TYPE = "Entity"

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
                 openai_api_key: str | None = None, google_api_key: str | None = None,
                 claude_api_key: str | None = None,
                 database: str | None = None):
        
        
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            database=database,
            refresh_schema=False,
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
            temperature=0,
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            max_retries=3,
            timeout=120
        )

        self.openai_graph_transform = ChatOpenAI(
            model="gpt-4o-mini",     # -mini
            temperature=0,
            api_key=SecretStr(openai_api_key) if openai_api_key else None,
            max_retries=3,
            timeout=120
        )

        # use claude-sonnet-4-5
        self.claude_client = ChatAnthropic(
            model_name="claude-haiku-4-5",
            temperature=0,
            api_key=SecretStr(claude_api_key) if claude_api_key else SecretStr(""),
            timeout=120,
            stop=None
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
            temperature=0.0,
            google_api_key=google_api_key,
            timeout=600,
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
        
        
        
        
    # TODO
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
                "name": chunk_id,
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
                    source=node,
                    target=chunk_node,
                    type="IN_CHUNK"
                )
            )

        nodes.append(chunk_node)

        return GraphDocument(
            nodes=nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"id": chunk_id})
        )
        
        
        
    def _add_document_chunk(self, count: int, path: str, properties: dict | None = None) -> GraphDocument:
        chunk_ids = [f"chunk_{i}" for i in range(count)]
        name = os.path.basename(path)
        
        if properties is None:
            properties = {}

        document_node = Node(id=name, type="Document", properties=properties)
        chunk_nodes = [Node(id=chunk_id, type="Chunk") for chunk_id in chunk_ids]

        relationships = [
            Relationship(source=chunk_node, target=document_node, type="IN_DOCUMENT")
            for chunk_node in chunk_nodes
        ]

        return GraphDocument(
            nodes=[document_node] + chunk_nodes,
            relationships=relationships,
            source=Document(page_content="", metadata={"id": name}),
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
                if rel.type.lower() in lower_allowed_rels or rel.type == "IN_CHUNK"
            ]

        return GraphDocument(
            nodes=filtered_nodes,
            relationships=filtered_relationships,
            source=graph_doc.source
        )








    def detect_tables(self, pdf_path: str, output_dir: str = "./code/assets/detected_tables_figures", conf_threshold: float = 0.3):
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



    def transform_table_to_html(self, table_image_paths: list[str], output_dir: str = "./code/assets/detected_tables_figures"):

        class TableHTMLResponse(BaseModel):
            """HTML table extracted from image(s)."""
            html: str = Field(description="Complete HTML <table> element with all rows and columns")
            page_range: str = Field(description="Page range of the source table, e.g. '159-169'")
            row_count: int = Field(description="Number of data rows in the table (excluding header)")
            column_count: int = Field(description="Number of columns in the table")


        system_prompt = """You are a document analysis expert specializing in table extraction from images.
        Your task is to convert table image(s) into a valid HTML table.

        RULES:
        - Output a complete HTML <table> element with <thead> and <tbody>.
        - If multiple images are provided, they are consecutive pages of the SAME table — merge them into ONE HTML table.
        - When merging multi-page tables: skip repeated headers, concatenate all data rows.
        - Preserve all data exactly as shown in the images — do not omit, summarize, or modify any cell values.
        - Use <th> for header cells, <td> for data cells.
        - If a cell spans multiple columns or rows, use colspan/rowspan attributes.
        - The text in the tables is in Slovak language — preserve the original text exactly."""

        user_prompt = f"Convert the following {len(table_image_paths)} table image(s) into a single HTML <table>. These images are consecutive pages of the same table. Preserve all data exactly as shown."
        

        content_parts: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for path in table_image_paths:
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"}
            })

        structured_model = self.gemini_client.with_structured_output(
            schema=TableHTMLResponse.model_json_schema(), method="json_schema"
        )

        response = structured_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=cast(list[str | dict[Any, Any]], content_parts)),
        ])

        print(response)

        # extract page numbers from filenames for naming the output
        page_numbers = []
        for path in table_image_paths:
            filename = os.path.basename(path)
            page_num = filename.split("_")[0].replace("page", "")
            if page_num.isdigit():
                page_numbers.append(int(page_num))

        if page_numbers:
            page_range_str = f"{min(page_numbers)}-{max(page_numbers)}" if len(page_numbers) > 1 else str(page_numbers[0])
        else:
            page_range_str = "unknown"

        html_filename = f"table_pages{page_range_str}.html"
        html_path = os.path.join(output_dir, html_filename)

        html_content = response.get("html", "") if isinstance(response, dict) else getattr(response, "html", str(response))

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'><style>table {{border-collapse: collapse; width: 100%;}} th, td {{border: 1px solid #ddd; padding: 8px; text-align: left;}} th {{background-color: #f2f2f2;}}</style></head><body>\n")
            f.write(html_content)
            f.write("\n</body></html>")

        print(f"created and saved table")

        return {"html_path": html_path, "response": response}


    def transform_html_to_graph_document(self, html_content: str, page_range: str) -> GraphDocument:
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
            properties: dict = Field(description="Properties of the node, including 'name' and optionally 'description'")

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

        system_prompt = """Si expert na transformaciu HTML tabuliek do stromovych struktur znalostneho grafu.

        Tvoja uloha je analyzovat HTML tabulku a vytvorit hierarchicky strom uzlov a vztahov, ktory presne zachytava logicku strukturu tabulky.

        # PRAVIDLA
        - Analyzuj stlpce, riadky, atributy rowspan/colspan a identifikuj hierarchiu dat.
        - Vytvor KORENOVY uzol reprezentujuci celu tabulku (napr. 'Colny sadzobnik', 'Priloha', 'Zoznam').
        - Vytvor RODICOVSKE uzly pre kazdu hlavnu skupinu (napr. body/sekcie oznacene rowspan v prvom stlpci).
        - Vytvor DETSKE/LISTOVE uzly pre jednotlive polozky v kazdej skupine.
        - Ak bunka obsahuje viacero hodnot oddelenych ciarkou (napr. 'ex 2, ex 4, ex 7'), vytvor SAMOSTATNY uzol pre kazdu hodnotu.
        - Popisny text z ostatnych stlpcov uloz ako 'description' vo vlastnostiach uzla.
        - Kazdy uzol MUSI mat kluc 'name' vo svojich vlastnostiach.
        - Pre typy vztahov pouzivaj VELKE_PISMENA_S_PODTRZITKAMI (napr. MA_SEKCIU, MA_KAPITOLU, OBSAHUJE, MA_POLOZKU).
        - Zachovaj vsetok text PRESNE tak, ako sa nachadza v tabulke — neprekladaj, neskracuj, nemodifikuj hodnoty buniek.
        - ID uzlov musia byt unikatne a popisne (napr. 'sekcia_1', 'kapitola_ex_2', 'polozka_0504').
        - Vsetky extrahovane hodnoty (ID uzlov, nazvy, vlastnosti, hodnoty vztahov) pis BEZ DIAKRITIKY (napr. c→c, s→s, z→z, a→a, e→e, i→i, o→o, u→u, y→y, n→n, t→t, d→d, l→l, o→o).

        # STRATEGIA TRANSFORMACIE
        1. Identifikuj hlavickove riadky (<thead>/<th>) — tie urcuju vyznam jednotlivych stlpcov.
        2. Stlpec s rowspan typicky oznacuje nadradenu kategoriu (rodic) — pouzivaj ho na vytvorenie rodicovskych uzlov.
        3. Ostatne stlpce su detske uzly alebo vlastnosti rodicovskych uzlov.
        4. Ak ma tabulka iba 2 stlpce (kluc-hodnota), modeluj ich ako vlastnosti jedneho uzla alebo ako pary uzlov.
        5. Pre pravne dokumenty: paragrafy, odseky, pismena, ciselne znaky a predpisy modeluj ako samostatne uzly s presnym odkazom.

        # KONTROLA PRED VYSTUPOM
        Pred tym nez odpovies, over si:
        1. Ma KAZDY uzol unikatne 'id' a kluc 'name' vo vlastnostiach?
        2. Su VSETKY hodnoty napisane BEZ DIAKRITIKY?
        3. Su viacnasobne hodnoty v jednej bunke rozdelene na samostatne uzly?
        4. Zachytava stromova struktura skutocnu hierarchiu tabulky (rodic→dieta)?
        5. Je zachovany VSETOK text z tabulky bez strat?"""

        user_prompt = f"""Transformuj nasledujucu HTML tabulku na znalostny graf s uzlami a vztahmi.
        Vytvor hierarchicku stromovu strukturu, ktora odraža organizaciu dat v tabulke.

        # HTML TABULKA:
        {html_content}"""

        structured_model = self.gemini_client.with_structured_output(schema=TableGraphResponse)

        response = structured_model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])

        typed_response = cast(TableGraphResponse, response)
        print(f"Table graph extraction for pages {page_range}: {len(typed_response.nodes)} nodes, {len(typed_response.relationships)} relationships")

        data = typed_response.model_dump()

        source_doc = Document(page_content=html_content, metadata={"page_range": page_range, "source": "table"})
        graph_document = self._convert_to_graph_document(data, f"table_{page_range}", source_doc)

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



    def load_pdf(self, pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        print("PDF loaded successfully.")
        return loader.load()
    
    
    
    def classification(self, documents: List[Document]) -> ClassifiedDocument:
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
        documents: List[Document],
        max_concurrent: int = 5,
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
        pause_event = asyncio.Event()
        pause_event.set()
        pause_lock = asyncio.Lock()

        async def limited(i, doc):
            for attempt in range(3):
                await pause_event.wait()
                async with semaphore:
                    try:
                        return await self.open_domain_detection(i, doc)
                    except Exception as e:
                        if attempt < 2:
                            async with pause_lock:
                                if pause_event.is_set():
                                    pause_event.clear()
                                    print(f"ODD: error on chunk {i} ({e}), pausing all processing for 60s...")
                                    await asyncio.sleep(60)
                                    pause_event.set()
                        else:
                            print(f"ODD chunk {i}: all 3 attempts failed")
                            raise

        tasks = [
            asyncio.create_task(limited(i, doc))
            for i, doc in enumerate(documents)
        ]

        res = await asyncio.gather(*tasks)
        return [r for r in res if r is not None]





    def schema_refinement(self, odd_schema: Schema, existing_schema: Optional[Schema] = None):
        """
        Function to refine and consolidate extracted schema information across documents, ensuring consistency and resolving conflicts.

        Args:
            schema: The schema to refine

        Returns:
            SchemaRefinement with extracted node_types and relationship_types and merge_log
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
        # ÚLOHA
        Spresni surovú schému detegovanú z otvorenej domény a integruj ju so základnou ontológiou (ak je dostupná).

        # KONTEXTOVÉ PRÍDAVKY
        - Do zoznamu uzlov (node_types) POVINNE pridaj nový typ: **Paragraf**.

        # POKYNY PRE Sémantiku
        - **Nezlučuj agresívne**: Ak existuje nuansa (napr. Banka vs. StatnyOrgan), ponechaj ich oddelené, pokiaľ základná ontológia neurčuje inak.
        - **Dôsledná ASCII normalizácia**: Skontroluj, či vo výsledku neostalo žiadne "š, č, ž, ý, á, í, é, ú, ä, ň, ť, ď, ľ, ô".

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

        
        
        structured_model = self.gemini_client.with_structured_output(
            schema=SchemaRefinementResponse.model_json_schema(), method="json_schema"
        )
        
        response = structured_model.invoke([
            ("system", system_prompt_for_schema_refinement),
            ("human", user_prompt),
        ])
        

        # structured_response is already a dict when using ProviderStrategy
        data = response

        print(data)

        return data



    
    
    
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
        Identifikuj vsetky typy uzlov a typy vztahov v texte.

        # PRAVIDLA
        - iba typy (nie instancie)
        - bez diakritiky
        - pouzi vseobecne, ale presne oznacenia
        - odstran nezmyselne typy
        - dopln chybajuce klucove koncepty
        - pouzivaj iba Slovensky jazyk

        # ZAMERANIE
        - zakon, paragraf, odsek
        - pravne vztahy (povinnost, pravo)
        - financne a vypoctove mechanizmy
        - casove entity

        # TEXT
        {text}
        """

        # Create and run the agent
        agent = create_agent(
            model=self.openai_client,
            response_format=ProviderStrategy(schema=response_schema_for_sde),  # type: ignore[arg-type]
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
        schema: Schema,
        max_concurrent = 5
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
        semaphore = asyncio.Semaphore(max_concurrent)
        pause_event = asyncio.Event()
        pause_event.set()
        pause_lock = asyncio.Lock()

        async def limited(i, doc):
            for attempt in range(3):
                await pause_event.wait()
                async with semaphore:
                    try:
                        return await self.schema_driven_extraction(i, doc, schema)
                    except Exception as e:
                        if attempt < 2:
                            async with pause_lock:
                                if pause_event.is_set():
                                    pause_event.clear()
                                    print(f"SDE: error on chunk {i} ({e}), pausing all processing for 60s...")
                                    await asyncio.sleep(60)
                                    pause_event.set()
                        else:
                            print(f"SDE chunk {i}: all 3 attempts failed")
                            raise

        tasks = [
            asyncio.create_task(limited(i, doc))
            for i, doc in enumerate(documents)
        ]

        res = await asyncio.gather(*tasks)
        return [r for r in res if r is not None]




    # --------- PROCESS STRATEGY ----------
    # 1. classification
    # 2. open domain schema detection
    # 3. schema refinement
    # 4. schema guided extraction
    # 5. vector store from graph



    def process(self, pdf_path: str, max_pages: Optional[int] = None):
        name_of_chain = "chain1"
        
        # Load PDF documents
        documents = self.load_pdf(pdf_path)
        if max_pages:
            documents = documents[:max_pages]
            
            
            
            
        # detect tables in pdf
        detections = self.detect_tables(pdf_path)

        # group consecutive table detections into multi-page groups
        table_groups = self.group_table_detections(detections)

        # convert each table group to HTML and then to GraphDocument
        table_graph_docs = []
        table_pages_to_exclude = set()

        for group in table_groups:
            image_paths = [d["image_path"] for d in group]
            result = self.transform_table_to_html(image_paths)
            print(f"Saved: {result['html_path']}")

            # access HTML from in-memory response
            response = result['response']
            html = response.get("html", "") if isinstance(response, dict) else response.html if hasattr(response, "html") else str(response)

            # compute page range for this group
            pages = sorted(d["page"] for d in group)
            page_range = f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0])

            # transform HTML table into GraphDocument via LLM
            table_gd = self.transform_html_to_graph_document(html, page_range)
            table_graph_docs.append(table_gd)

            # collect interior table pages to exclude from text processing
            # keep first and last pages (may have text above/below the table)
            if len(pages) > 2:
                table_pages_to_exclude.update(pages[1:-1])

        # remove interior table pages from documents before ODD/SDE
        # PyPDFLoader uses 0-based page numbers, detections use 1-based
        if table_pages_to_exclude:
            documents = [
                doc for doc in documents
                if (doc.metadata.get("page", -1) + 1) not in table_pages_to_exclude
            ]
            print(f"Excluded {len(table_pages_to_exclude)} interior table pages from text processing: {sorted(table_pages_to_exclude)}")


        # document_classification = self.classification()


        # odd
        splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
        chunked_documents = splitter.split_documents(documents)

        extracted_schema_list = asyncio.run(
            self.async_open_domain_detection(
                chunked_documents,
            )
        )
        print(f"\n\nAll chunks processed into schema.\n\n")
        odd_to_json(extracted_schema_list, name = name_of_chain)
        
        
        # refinement
        extracted_schema = Schema(
            nodes=list(set(node for schema in extracted_schema_list for node in schema.nodes)),
            relationships=list(set(rel for schema in extracted_schema_list for rel in schema.relationships))
        )
        refined_schema = self.schema_refinement(odd_schema=extracted_schema, existing_schema=self.get_graph_schema())
        
        print(f"\n\nSchema refined.\n\n")
        refinement_to_json(refined_schema, name = name_of_chain)
        
        
        # sde
        splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
        chunked_documents = splitter.split_documents(documents)
        refined_schema = self._convert_to_schema(refined_schema)
        
        graph_docs = asyncio.run(
            self.async_schema_driven_extraction(
                chunked_documents,
                schema=refined_schema
            )
        )
        
        print(f"\n\nAll chunks processed into graph documents.\n\n")
        sde_to_json(graph_docs, name = name_of_chain)
        

        # add document chunk into graph documents
        graph_docs.append(self._add_document_chunk(len(chunked_documents), pdf_path))

        # add table-extracted graph documents
        graph_docs.extend(table_graph_docs)


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
        self.graph.query("MATCH (n:Document:__Entity__) REMOVE n:__Entity__")
        
        print("\n\nAdded Graph Documents into Graph database.\n\n")
        
        
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
            node_label="Chunk",
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
            

        
        print("Knowledge graph and vector stores successfully updated in Neo4j!")
            
        
        
        
        
# ====================================================================================================

        
        
    # ---------------- KG-GPT QUERYING METHODS ----------------

    def segment_question(self, question: str) -> List[SubSentence]:
        """Stage 1 (KG-GPT): Break question into sub-sentences each aligned with one KG triple.

        Args:
            question: The user's natural language question

        Returns:
            List of SubSentence objects, each with text and up to 2 entity mentions
        """
        print("\nStage 1: Sentence segmentation...")
        
        
        
        agent = create_agent(
            model=self.openai_client,
            response_format=ProviderStrategy(schema=response_schema_for_segmentation),  # type: ignore[arg-type]
            system_prompt=system_prompt_for_segmentation
        )
        response = agent.invoke({"messages": [{"role": "user", "content": f"Otázka: {question}"}]})

        # structured_response is already a dict when using ProviderStrategy
        data = cast(dict, response["structured_response"])

        sub_sentences = [
            SubSentence(text=s.get("text", "").strip(), entities=s.get("entities", []) or [])
            for s in data.get("sub_sentences", [])
            if s.get("text")
        ]

        if not sub_sentences:
            sub_sentences = [SubSentence(text=question, entities=[])]


        print(f"  → {len(sub_sentences)} sub-sentence(s): {[s.text for s in sub_sentences]}")
        return sub_sentences
    
    

    def search_agent_retrieve(self, sub_sentence: str, entities: list[str]) -> tuple[list[tuple], list[str]]:
        """ARK-V1-inspired Stage 2: Agent with search_database tool iteratively queries
        the graph to find evidence triples for a single sub-sentence.

        Args:
            sub_sentence: Sub-sentence text from Stage 1 (one implied KG triple)
            entities: Entity mentions from segmentation (Title Case)

        Returns:
            (evidence_triples, node_ids) — triples as (head, rel, tail) tuples and
            node IDs for chunk retrieval, both compatible with Stage 3
        """
        import re

        graph = self.graph

        @tool
        def search_database(cypher_query: str) -> str:
            """Execute a Cypher query against the Neo4j graph database.

            Args:
                cypher_query: A valid Cypher query string to execute

            Returns:
                JSON string of query results, or error message if query fails
            """
            try:
                is_read_only_cypher(cypher_query)
                results = graph.query(cypher_query)
                serialized = serialize_for_json(results)
                return json.dumps(serialized, ensure_ascii=False, indent=2)[:4000]
            except Exception as e:
                return f"Query error: {e}"

        schema = self.get_graph_schema()
        sample_schema = self.get_sample_graph_schema()

        user_prompt = f"""
        Pod-veta:
        {sub_sentence}

        Entity (anchors):
        {entities}

        Schéma grafu:
        - Uzly: {schema.nodes}
        - Vzťahy: {schema.relationships}

        Ukážkové dáta:
        {sample_schema}

        Úloha:
        Nájdi všetky relevantné trojice (uzol-vzťah-uzol) súvisiace s pod-vetou.

        Postup:
        1. Začni od entít (anchors)
        2. Iteratívne vyberaj vzťahy zo schémy
        3. Dopytuj graf pomocou Cypher
        4. Rozhoduj, či pokračovať alebo zastaviť

        Obmedzenia:
        - Používaj iba vzťahy zo schémy
        - Nevymýšľaj dáta
        - Max hĺbka = 3
        - Max retries pre výber vzťahu = 2
        """
        

        agent = create_agent(
            model=self.openai_client,
            tools=[search_database],
            response_format=ToolStrategy(schema=response_schema_for_generating_query), # type: ignore[arg-type]
            system_prompt=system_prompt_for_generating_query,
        )

        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        result = response["structured_response"]

        node_ids: list[str] = list(result.get("nodes_found", []))
        triples: list[tuple] = []

        for rel_str in result.get("relationships_found", []):
            match = re.match(r"^(.+?)\s*-\[(.+?)\]->\s*(.+)$", rel_str.strip())
            if match:
                head, rel, tail = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
                triples.append((head, rel, tail))
                if head not in node_ids:
                    node_ids.append(head)
                if tail not in node_ids:
                    node_ids.append(tail)

        if not triples and result.get("cypher_query"):
            try:
                rows = self.graph.query(result["cypher_query"])
                for row in rows:
                    h = row.get("head") or row.get("a.id", "")
                    r = row.get("rel") or row.get("type(r)", "")
                    t = row.get("tail") or row.get("b.id", "")
                    if h and r and t:
                        triples.append((h, r, t))
            except Exception:
                pass

        return triples, node_ids



    # ---------------- ARK-V1 Stage 2 (state-machine retrieval) ----------------
    
    def _resolve_anchor(self, name: str) -> Optional[str]:
        if not name:
            return None
        
        rows = self.graph.query( "MATCH (n) WHERE toLower(n.id) = toLower($name) RETURN n.id AS id LIMIT 1", {"name": name})
        if rows:
            return rows[0]["id"]
        
        rows = self.graph.query( "MATCH (n) WHERE toLower(n.id) CONTAINS toLower($name) RETURN n.id AS id LIMIT 1", {"name": name})
        return rows[0]["id"] if rows else None
    

    def _retrieve_relations(self, anchor: str) -> list[str]:
        q_out = "MATCH (a)-[r]->() WHERE toLower(a.id) = toLower($anchor) RETURN DISTINCT type(r) AS rel"
        q_in = "MATCH (a)<-[r]-() WHERE toLower(a.id) = toLower($anchor) RETURN DISTINCT type(r) AS rel"
        
        is_read_only_cypher(q_out)
        is_read_only_cypher(q_in)
        
        rels: list[str] = []
        seen: set[str] = set()
        
        for q in (q_out, q_in):
            for row in self.graph.query(q, {"anchor": anchor}):
                
                r = row.get("rel")
                if r and r not in seen:
                    seen.add(r)
                    rels.append(r)
                    
        return rels


    def _retrieve_triples(self, anchor: str, rel: str) -> list[tuple]:
        # f-string for rel type is safe: rel is validated to be in R^k returned by graph.
        TRIPLE_LIMIT = 25
        
        q_out = f"MATCH (a)-[r:`{rel}`]->(t) WHERE toLower(a.id) = toLower($anchor) RETURN a.id AS h, type(r) AS r, t.id AS t LIMIT {TRIPLE_LIMIT}"
        q_in = f"MATCH (a)<-[r:`{rel}`]-(t) WHERE toLower(a.id) = toLower($anchor) RETURN t.id AS h, type(r) AS r, a.id AS t LIMIT {TRIPLE_LIMIT}"
        
        is_read_only_cypher(q_out)
        is_read_only_cypher(q_in)
        
        rows = self.graph.query(q_out, {"anchor": anchor})
        if not rows:
            rows = self.graph.query(q_in, {"anchor": anchor})
            
        out: list[tuple] = []
        for row in rows:
            
            h, r, t = row.get("h"), row.get("r"), row.get("t")
            if h and r and t:
                out.append((h, r, t))
                
        return out


    def _select_relation(self, sub_sentence: str, anchor: str, summary: str, relations: list[str]) -> Optional[str]:
        C_MAX = 2
        
        class RelationChoice(BaseModel):
            relation: Optional[str] = Field(
                default=None,
                description="Exactly one relation string from the provided list, or null if none relevant.",
            )
            rationale: str = Field(default="", description="Brief Slovak justification.")
            
            
        model = self.gemini_client_flash.with_structured_output(
            schema=RelationChoice, method="json_schema"
        )
        
        last_err = ""
        for attempt in range(C_MAX + 1):
            user = (
                f"Pod-veta: {sub_sentence}\n"
                f"Anchor: {anchor}\n"
                f"Doterajšie zhrnutie: {summary or '(prázdne)'}\n"
                f"Kandidátne vzťahy (R^k): {relations}\n"
                + (f"\nPredchádzajúci pokus bol neplatný: {last_err}\n" if last_err else "")
                + "Vyber presne jeden vzťah z R^k alebo null."
            )
            
            try:
                resp = cast(RelationChoice, model.invoke([
                    ("system", system_prompt_ark_select_relation),
                    ("human", user),
                ]))
                
            except Exception as e:
                last_err = f"LLM error: {e}"
                continue
            
            
            chosen = resp.relation
            
            if chosen is None:
                return None
            if chosen in relations:
                return chosen
            
            last_err = f"'{chosen}' nie je v zozname R^k."
            
        return None


    def _reason_step(self, sub_sentence: str, anchor: str, rel: str, triples: list[tuple], summary: str) -> ReasoningStep:
        
        model = self.gemini_client_flash.with_structured_output(
            schema=ReasoningStep, method="json_schema"
        )
        
        numbered = "\n".join(f"[{i}] {h} -[{r}]-> {t}" for i, (h, r, t) in enumerate(triples))
        
        user = (
            f"Pod-veta: {sub_sentence}\n"
            f"Anchor: {anchor}\n"
            f"Vybraný vzťah: {rel}\n"
            f"Doterajšie zhrnutie: {summary or '(prázdne)'}\n"
            f"Trojice:\n{numbered}\n"
            "Vyber relevantné indexy, zhrň implikáciu, rozhodni o pokračovaní a navrhni next_anchor (tail vybratých trojíc) alebo null."
        )
        
        try:
            resp = model.invoke([
                ("system", system_prompt_ark_reasoning),
                ("human", user),
            ])
            
        except Exception as e:
            print(f"  [ark-v1] reasoning LLM error: {e}")
            return ReasoningStep()

        if not isinstance(resp, ReasoningStep):
            print(f"  [ark-v1] reasoning unexpected response type: {type(resp).__name__}")
            return ReasoningStep()
        
        return resp
    
    

    def ark_v1_retrieve(self, sub_sentence: str, entities: list[str]) -> tuple[list[tuple], list[str]]:
        """ARK-V1 (Klein & Ohnemus 2025) state-machine retrieval for a single sub-sentence.

        For each anchor entity we perform up to K_MAX reasoning steps. Each step:
        1) enumerate relations actually present in the graph for the anchor,
        2) LLM picks one (validated against the candidate list, Cmax retries),
        3) enumerate triples (anchor)-[rel]-(*) from the graph,
        4) LLM selects relevant indices, summarizes implication, optionally advances
           the anchor to a tail of a selected triple for the next hop.

        The per-step prompt is rebuilt from scratch (system + Q + sub_sentence +
        running summary + current candidates) so context size stays ~constant in k.
        """
        K_MAX = 3

        evidence: list[tuple] = []
        node_ids: list[str] = []
        summary = ""

        for raw_anchor in (entities or [])[:2]:
            anchor = self._resolve_anchor(raw_anchor)
            
            if anchor is None:
                print(f"  [ark-v1] anchor '{raw_anchor}' not found in graph, skipping")
                continue

            for k in range(K_MAX):
                
                relations = self._retrieve_relations(anchor)
                if not relations:
                    break
                
                rel = self._select_relation(sub_sentence, anchor, summary, relations)
                if rel is None:
                    break
                
                triples = self._retrieve_triples(anchor, rel)
                if not triples:
                    break

                step = self._reason_step(sub_sentence, anchor, rel, triples, summary)
                valid_idx = [i for i in step.selected_triple_indices if 0 <= i < len(triples)]
                selected = [triples[i] for i in valid_idx]
                
                evidence.extend(selected)
                for h, _, t in selected:
                    
                    if h not in node_ids:
                        node_ids.append(h)
                        
                    if t not in node_ids:
                        node_ids.append(t)
                        

                print(f"  [ark-v1] Step {k+1}: anchor={anchor} rel={rel} selected={len(selected)}/{len(triples)} → {step.implication}")
                summary += f"\nKrok {k+1} (anchor={anchor}, rel={rel}): {step.implication}"

                if not step.continue_reasoning:
                    break
                
                tails = {t for (_, _, t) in selected}
                if step.next_anchor and step.next_anchor in tails:
                    anchor = step.next_anchor
                else:
                    break

        # dedup triples while preserving order
        seen = set()
        dedup_triples: list[tuple] = []
        
        for tr in evidence:
            if tr not in seen:
                seen.add(tr)
                dedup_triples.append(tr)
                
        return dedup_triples, node_ids



    def get_chunks_from_nodes(self, node_ids: List[str]) -> List[str]:
        """Retrieve text chunks connected to given nodes via IN_CHUNK relationship.

            in: node_ids: List[str]
            out: List[str] - chunk texts
        """
        if not node_ids:
            return []

        try:
            results = self.graph.query("""
                MATCH (n)-[:IN_CHUNK]->(c:Chunk)
                WHERE n.id IN $node_ids
                RETURN DISTINCT c.text AS text, c.page AS page, n.id AS source_node
                ORDER BY c.page
            """, {"node_ids": node_ids})

            chunks = [record['text'] for record in results if record.get('text')]
            print(f"\nNajdenych {len(chunks)} textovych usekov spojenych s uzlami.\n")
            return chunks

        except Exception as e:
            print(f"Chyba pri ziskavani chunkov: {e}")
            return []
        
        

    def answer(self, question: str, evidence_triples: List[tuple], chunks: List[str] = []) -> str:
        """Stage 3 (KG-GPT): Derive a natural language answer from evidence triples and chunk context.

        Args:
            question: The original user question
            evidence_triples: List of (head, relation, tail) tuples from graph retrieval
            chunks: Text chunks connected to retrieved nodes via IN_CHUNK

        Returns:
            Natural language answer string
        """
        print("\n\nStage 3: Inference...\n\n")

        linearized = [[h, r, t] for h, r, t in evidence_triples]
        chunk_text = "\n---\n".join(chunks) if chunks else "Žiadne textové úseky neboli nájdené."

        user_prompt = f"""
        ## Vstupné údaje pre analýzu

        ### 1. Otázka používateľa
        {question}

        ### 2. Dôkazy z grafovej databázy (Primárne)
        Nasledujúce trojice reprezentujú overené fakty v štruktúre [Subjekt, Relácia, Objekt]:
        {json.dumps(linearized, ensure_ascii=False, indent=2)}

        ### 3. Textový kontext z dokumentov (Doplnkové)
        Tento text slúži na pochopenie širšieho kontextu a detailov:
        {chunk_text}

        ---
        **Inštrukcia:** Na základe vyššie uvedených údajov vypracuj stručnú, ale výstižnú odpoveď. Zameraj sa na to, aby odpoveď priamo adresovala otázku a prioritne využívala fakty z grafových trojíc.
        """

        class InferenceAnswer(BaseModel):
            """Natural language answer derived from evidence triples and chunk context."""
            answer: str = Field(description="Natural language answer grounded in the evidence triples and chunk context")

        structured_model = self.gemini_client_flash.with_structured_output(
            schema=InferenceAnswer, method="json_schema"
        )

        response = structured_model.invoke([
            ("system", system_prompt_for_inference),
            ("human", user_prompt),
        ])

        return cast(InferenceAnswer, response).answer
    
    

    # ---------------- INTERACTIVE QUESTIONING ----------------

    def query(self, question: str | None = None, verbose: bool = True) -> str:
        """KG-GPT 3-stage pipeline: Segment → Retrieve → Infer.

        Args:
            question: the user question. If None, read from stdin (interactive mode).
            verbose: print per-stage progress.

        Returns:
            The final answer string (also printed in interactive mode).
        """
        interactive = question is None
        if interactive:
            question = input("Enter your question: ")
            if question == '-h':
                print("Help Instructions: \n - To exit, type 'exit' \n - To view graph schema, type '-s' \n")
                return ""
            elif question == '-s':
                schema = self.get_graph_schema()
                print(schema)
                return str(schema)
            elif question.lower() == 'exit':
                print("Exiting...")
                return ""

        # Stage 1: Sentence Segmentation
        sub_sentences = self.segment_question(question)

        # Stage 2: Graph Retrieval (per sub-sentence)
        all_triples: List[tuple] = []
        all_node_ids: List[str] = []

        for sub in sub_sentences:
            if verbose:
                print(f"\nStage 2 (ARK-V1): Retrieving evidence for: '{sub.text}'")
            triples, node_ids = self.ark_v1_retrieve(sub.text, sub.entities)
            if verbose:
                print(f"  ARK-V1 found {len(triples)} triples, {len(node_ids)} nodes")
            all_triples.extend(triples)
            all_node_ids.extend(node_ids)

        # Retrieve source text chunks via IN_CHUNK
        chunks = self.get_chunks_from_nodes(list(dict.fromkeys(all_node_ids)))

        # Stage 3: Inference
        final_answer = self.answer(question, all_triples, chunks)
        
        if interactive:
            print(f"\n{final_answer}")
        return final_answer