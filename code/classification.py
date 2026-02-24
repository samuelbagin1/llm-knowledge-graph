from langchain_core.documents import Document
from classes import Schema, ClassifiedDocument, Type
from typing import List


def classify(documents: List[Document], client) -> ClassifiedDocument:
    type_of_legislation = _classify_type_of_legislation(documents, client)
    type_of_document = _classify_type_of_document(documents, client)
    
    return ClassifiedDocument(
        legislation_type=type_of_legislation,
        document_type=type_of_document
    )


def _classify_type_of_legislation(documents: List[Document], client) -> Type:
    system_prompt = """ """
    user_prompt = """ """
    response_schema = """ """
    
    # Create and run the agent
    agent = create_agent(
        model=client,
        response_format=ToolStrategy(schema=response_schema),
        system_prompt=system_prompt
    )
    response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

    # structured_response is already a dict when using ProviderStrategy
    data = response["structured_response"]
    return data

def _classify_type_of_document(documents: List[Document], client) -> Type:
    system_prompt = """ """
    user_prompt = """ """
    response_schema = """ """
    
    # Create and run the agent
    agent = create_agent(
        model=client,
        response_format=ToolStrategy(schema=response_schema),
        system_prompt=system_prompt
    )
    response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

    # structured_response is already a dict when using ProviderStrategy
    data = response["structured_response"]
    return data