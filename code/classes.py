from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship


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
class Question:
    id: str = ""
    question: str = ""
    svo: SVO = field(default_factory=lambda: SVO(sub="", verb="", obj=""))
    similar_nodes: List[Node] = field(default_factory=list)
    similar_rel: List[Relationship] = field(default_factory=list)
    