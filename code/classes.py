from typing import List

class Schema:
    """Schema dataclass to hold extracted node types and relationship types."""
    nodes: List[str]
    relationships: List[str]
    
    
class Type:
    type_of: str
    scoring: float
    reasong: str

    
class ClassifiedDocument:
    legislation_type: Type
    document_type: Type