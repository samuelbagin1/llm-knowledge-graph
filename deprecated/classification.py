from langchain_core.documents import Document
from classes import Schema, ClassifiedDocument
from typing import List
from pydantic import BaseModel, Field

import datetime
import json
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy



class Type(BaseModel):
    type_of: str = Field(description="The specific type of legislation or document, e.g., 'Act', 'Decree', 'Regulation', etc.")
    scoring: float = Field(description="A confidence score between 0 and 100 indicating the model's confidence in the classification.")
    reason: str = Field(description="A brief explanation of why the model classified the document as such, based on the content and features of the document.")
    
    
##########################################################################
# main act
def classify(documents: List[Document]) -> ClassifiedDocument:
    """Classify the type of legislation and document based on the content of the provided documents."""
    
    model = ChatGoogleGenerativeAI(
            model="gemini-2.5",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    type_of_legislation = _classify_type_of_legislation(model, documents[:2])
    type_of_document = _classify_type_of_document(documents, model)
    
    return ClassifiedDocument(
        legislation_type=type_of_legislation,
        document_type=type_of_document
    )
##########################################################################


# ------------ SUBFUNCTIONS ------------------- #

# legislation type classification
#TODO: edit the prompts
def _confirmation_agent_leg(model, documents: List[Document], potential_type: Type, information: str) -> Type:
    text = documents[0].page_content + " " + documents[1].page_content
    
    system_prompt = """
    You are an expert legal document classification confirmation agent. You receive a potential document type classification and supporting information about the document. Your task is to evaluate whether the proposed type is correct, adjust it if necessary, and provide a justified confidence score.

    ## Document Types (non-exhaustive):
    Act (Zákon), Decree (Vyhláška), Regulation (Nariadenie), Constitutional Act (Ústavný zákon), Measure (Opatrenie), Notice (Oznámenie), Resolution (Uznesenie), Finding (Nález), Directive (Smernica), Treaty (Zmluva), Amendment (Novela), Order (Príkaz), Statute (Štatút), Ruling (Rozhodnutie), Guideline (Metodický pokyn)

    ## Rules:
    1. Evaluate the proposed type against the provided information about the document (metadata, title, issuing authority, content description, etc.).
    2. If the proposed type is correct, confirm it with a high confidence score and provide justification.
    3. If the proposed type is incorrect or imprecise, correct it to the most accurate type and explain why the original was wrong.
    4. Base your reasoning strictly on the provided information. Do not speculate beyond what is given.
    5. Consider key indicators: issuing authority, legislative process, numbering format, legal force, scope of application, and document structure.
    """
    
    user_prompt = f"""
    Evaluate and confirm or correct the following proposed document type classification based on the provided information.

    # Proposed Type:
    {potential_type}

    # Document Information:
    {information}
    
    # Text:
    {text}

    Analyze the document information — including issuing authority, title, numbering, scope, and any other available metadata — to determine whether the proposed type is accurate.
    """
    
    structured_model = model.with_structured_output(
        schema=Type.model_json_schema(), method="json_schema"
    )
    
    response = structured_model.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


    data = Type(type_of=response["type_of"], scoring=response["scoring"], reason=response["reason"])
    
    if data.scoring > 50:
        return data
    else:
        return _classify_type_of_legislation(model, documents, data.type_of)



def _classify_type_of_legislation(model, documents: List[Document], type_not: str = None) -> Type:
    """Classify the type of legislation based on the content of the provided documents."""
    text = documents[0].page_content + " " + documents[1].page_content
    
    system_prompt = """
    You are an expert Slovak legal document classification algorithm. Your task is to classify a given legal text into exactly one primary legal category from a predefined taxonomy.

    ## Categories:
    - Finančné právo
    - Hospodárske právo
    - Medzinárodné právo
    - Obchodné právo
    - Občianske právo
    - Pracovné právo
    - Právo EÚ
    - Právo sociálneho zabezpečenia
    - Správne právo
    - Trestné právo
    - Ústavné právo
    - Vojenské právo

    ## Rules:
    1. Classify into exactly ONE primary category. Choose the single best fit.
    2. Base your classification strictly on the content, subject matter, and legal terminology present in the text.
    3. Provide a confidence score (0-100) reflecting how clearly the text fits the chosen category. Use lower scores when the text spans multiple domains or is ambiguous.
    4. Provide a brief reason referencing specific content from the text that supports your classification.
    5. If the text touches multiple legal domains, select the dominant one based on the primary subject matter and legislative purpose.
    6. Do not invent categories outside the provided list.
    """
    
    user_prompt = f"""
    Classify the following legal text into one of these categories:
    - Finančné právo
    - Hospodárske právo
    - Medzinárodné právo
    - Obchodné právo
    - Občianske právo
    - Pracovné právo
    - Právo EÚ
    - Právo sociálneho zabezpečenia
    - Správne právo
    - Trestné právo
    - Ústavné právo
    - Vojenské právo

    Analyze the text's subject matter, legal terminology, and legislative purpose to determine the best fitting primary category.

    # Text:
    {text}
    """
    
    structured_model = model.with_structured_output(
        schema=Type.model_json_schema(), method="json_schema"
    )
    
    response = structured_model.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


    data = Type(type_of=response["type_of"], scoring=response["scoring"], reason=response["reason"])
    
    return _confirmation_agent_leg(model, documents, data)





# document type classification
# TODO: edit the prompts
def _confirmation_agent_doc(model, documents: List[Document], potential_type: Type, information: str) -> Type:
    text = documents[0].page_content + " " + documents[1].page_content
    
    system_prompt = """
    You are an expert legal document type confirmation agent. You receive a potential document type classification, reference information describing that document type's characteristics, and the original text. Your task is to evaluate whether the proposed type is correct by cross-referencing the text against the type's known characteristics, adjust if necessary, and provide a justified confidence score.

    ## Document Types:
    Dekrét, Dohoda, Dohovor, Nález, Nariadenie, Nariadenie vlády, Opatrenie, Oznámenie, Podmienky, Pokyny, Poriadok, Postup, Pravidlá, Redakčné oznámenie, Rozhodnutie, Rozkaz, Smernica, Štatút, Úplné znenie, Ústavný zákon, Uznesenie, Vyhláška, Výnos, Zákon, Zákonné opatření Senátu, Zásady, Zmluva, Ostatné.

    ## Confirmation Criteria:
    Evaluate the match between the proposed type and the text by checking:
    1. **Title/Heading match**: Does the text explicitly name the document type consistent with the proposed type?
    2. **Issuing authority**: Does the issuing body in the text match the expected authority for this document type (as described in the provided information)?
    3. **Structure and formulation**: Does the text's structure (articles, paragraphs, clauses, preamble, enacting formula) match the characteristics described in the type information?
    4. **Legal force and scope**: Does the text's scope and binding nature align with the proposed type?
    5. **Language patterns**: Do the legislative or administrative formulations in the text match what is typical for this type?

    ## Rules:
    1. Compare the text against the provided type information systematically.
    2. If the proposed type is correct and well-supported, confirm it with a high score (75-100) and cite matching evidence from the text.
    3. If the proposed type is partially correct but imprecise (e.g., "Nariadenie" when it should be "Nariadenie vlády"), correct it and explain the distinction.
    4. If the proposed type is incorrect, provide the correct type with justification referencing both the text and the type information.
    5. A score below 50 indicates the proposed type is likely wrong. Always provide a corrected type in that case.
    6. Base reasoning strictly on the text and the provided information. Do not speculate.
    """
    
    user_prompt = f"""
    Evaluate and confirm or correct the following proposed document type classification. Cross-reference the original text against the provided type information to determine if the classification is accurate.

    # Proposed Type:
    {potential_type}

    # Type Information (characteristics of the proposed type):
    {information}

    # Original Text:
    {text}

    Systematically check whether the text's title, issuing authority, structure, legal force, and language patterns match the characteristics described in the type information.
    """
    
    structured_model = model.with_structured_output(
        schema=Type.model_json_schema(), method="json_schema"
    )
    
    response = structured_model.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


    data = Type(type_of=response["type_of"], scoring=response["scoring"], reason=response["reason"])
    
    if data.scoring > 50:
        return data
    else:
        return _classify_type_of_document(documents, model)



def _classify_type_of_document(documents: List[Document], model) -> Type:
    """Classify the type of document based on the content of the provided documents."""
    text = documents[0].page_content + " " + documents[1].page_content
    
    system_prompt = """
    You are an expert Slovak legal document type classification algorithm. Your task is to classify a given legal text into exactly one document type from a predefined list.

    ## Document Types:
    - Dekrét
    - Dohoda
    - Dohovor
    - Nález
    - Nariadenie
    - Nariadenie vlády
    - Opatrenie
    - Oznámenie
    - Podmienky
    - Pokyny
    - Poriadok
    - Postup
    - Pravidlá
    - Redakčné oznámenie
    - Rozhodnutie
    - Rozkaz
    - Smernica
    - Štatút
    - Úplné znenie
    - Ústavný zákon
    - Uznesenie
    - Vyhláška
    - Výnos
    - Zákon
    - Zákonné opatření Senátu
    - Zásady
    - Zmluva
    - Ostatné (use only when no other type fits)

    ## Key Indicators to Consider:
    - **Title and heading**: Often explicitly states the document type (e.g., "ZÁKON č. 40/1964", "VYHLÁŠKA č. 322/2008").
    - **Issuing authority**: Národná rada SR → Zákon/Ústavný zákon; Vláda SR → Nariadenie vlády; Ministerstvo → Vyhláška/Výnos/Opatrenie; Ústavný súd → Nález; Prezident → Dekrét.
    - **Numbering format and Zbierka zákonov reference**.
    - **Document structure**: Preamble, articles (§/Čl.), enacting clauses, effective date formulations.
    - **Language and formulations**: Legislative commands ("ustanovuje", "nariaďuje"), declarative language, contractual language ("zmluvné strany sa dohodli").

    ## Rules:
    1. Classify into exactly ONE document type.
    2. Base classification primarily on explicit indicators in the text (title, heading, issuing authority) before analyzing content.
    3. If the title explicitly names the document type, prioritize that signal.
    4. Provide a confidence score (0-100). Use higher scores when explicit indicators are present, lower when relying on content inference.
    5. Provide a brief reason referencing specific text evidence.
    6. Use "Ostatné" only as a last resort when no other type can be reasonably determined.
    """
    
    user_prompt = f"""
    Classify the following legal text into one of these document types:
    Dekrét, Dohoda, Dohovor, Nález, Nariadenie, Nariadenie vlády, Opatrenie, Oznámenie, Podmienky, Pokyny, Poriadok, Postup, Pravidlá, Redakčné oznámenie, Rozhodnutie, Rozkaz, Smernica, Štatút, Úplné znenie, Ústavný zákon, Uznesenie, Vyhláška, Výnos, Zákon, Zákonné opatření Senátu, Zásady, Zmluva, Ostatné.

    Examine the title, heading, issuing authority, structure, and content of the text to determine the document type.

    # Text:
    {text}
    """
    
    structured_model = model.with_structured_output(
        schema=Type.model_json_schema(), method="json_schema"
    )
    
    response = structured_model.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])


    data = Type(type_of=response["type_of"], scoring=response["scoring"], reason=response["reason"])
    
    return _confirmation_agent_doc(model, documents, data)

# ------------ /SUBFUNCTIONS ------------------- #