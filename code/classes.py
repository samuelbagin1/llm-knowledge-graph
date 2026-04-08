from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List, Tuple
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
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


@dataclass
class SubSentence:
    """A sub-sentence aligned with a single KG triple, with up to 2 entity mentions."""
    text: str = ""
    entities: List[str] = field(default_factory=list)