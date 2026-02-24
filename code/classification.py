from langchain_core.documents import Document
from classes import Schema, ClassifiedDocument, Type
from typing import List


def classify(documents: List[Document]) -> ClassifiedDocument:
    type_of_legislation = _classify_type_of_legislation(documents)
    type_of_document = _classify_type_of_document(documents)
    
    return ClassifiedDocument(
        legislation_type=type_of_legislation,
        document_type=type_of_document
    )


def _classify_type_of_legislation(documents: List[Document]) -> Type:
    pass

def _classify_type_of_document(documents: List[Document]) -> Type:
    pass