from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document


@dataclass
class Schema:
    """Schema dataclass to hold extracted node types and relationship types."""
    nodes: list[str] = field(default_factory=list)
    relationships: list[str] = field(default_factory=list)


@dataclass
class Type:
    type_of: str = ""
    scoring: float = 0.0
    reason: str = ""


@dataclass
class ClassifiedDocument:
    legislation_type: Type = field(default_factory=Type)
    document_type: Type = field(default_factory=Type)


class SVO(BaseModel):
    """A class with subject, verb, and object extracted from the question"""
    sub: str = Field(description="The subject extracted from the question")
    verb: str = Field(description="The verb extracted from the question")
    obj: str = Field(description="The object extracted from the question")


@dataclass
class GraphResult:
    cypher_query: str
    explanation: str
    nodes_found: List[str]
    relationships_found: List[str]
    
    
@dataclass
class Similar:
    nodes: List[str]
    relationships: List[str]
    chunks: List[str]
    
    
@dataclass
class ValidationResult:
    is_sufficient: bool
    missing_info: List[str]
    key_nodes: List[str]
    reasoning: str


@dataclass
class Question:
    id: str = ""
    question: str = ""
    svo: SVO = field(default_factory=lambda: SVO(sub="", verb="", obj=""))
    extracted_graph: GraphDocument = field(default_factory=lambda: GraphDocument(nodes=[], relationships=[], source=Document(page_content="")))
    graph_result: GraphResult = field(default_factory=lambda: GraphResult(cypher_query="", explanation="", nodes_found=[], relationships_found=[]))
    similar: Similar = field(default_factory=lambda: Similar(nodes=[], relationships=[], chunks=[]))