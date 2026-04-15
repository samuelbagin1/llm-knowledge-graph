from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple
from langchain_neo4j.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document


class ReasoningStep(BaseModel):
    """One ARK-V1 reasoning step: which triples were relevant, what they imply,
    and whether (and from which tail) to continue the walk."""
    selected_triple_indices: List[int] = Field(default_factory=list)
    implication: str = Field(default="")
    continue_reasoning: bool = Field(default=False)
    next_anchor: Optional[str] = Field(default=None)


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