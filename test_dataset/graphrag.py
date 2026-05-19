"""
Q&A GraphRAG evaluation against the Slovak VAT-law knowledge graph.

Pipeline per question:
    1. Retrieval agent (gpt-5-mini, tool: search_database) writes Cypher,
       finds supporting Paragraf/Odsek/Pismeno nodes via entity anchoring.
    2. aggregate_section_text() flattens the hierarchy into a context string.
    3. Answer agent (gpt-5-mini) reads the aggregated text + entities and
       produces a natural-language answer.
    4. Result is appended incrementally to q&a_graphrag_validation.json.

Neo4j database: zz-2004-222 (credentials in .env).
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy


load_dotenv()

MODEL_NAME = "gpt-5.4-mini"
DATABASE_NAME = "zz-2004-222"

QA_INPUT_PATH = Path(__file__).parent / "q&a.json"
QA_OUTPUT_PATH = Path(__file__).parent / "q&a_graphrag_validation.json"

# Structural nodes are excluded from `found_entities` (they go in found_sections).
STRUCTURAL_LABELS = {"Paragraf", "Odsek", "Pismeno", "Bod", "Priloha"}


# --------------------------------------------------------------------------- #
# Neo4j helpers
# --------------------------------------------------------------------------- #

def _serialize_value(value: Any) -> Any:
    """Make Neo4j Node/Relationship/Path objects JSON-safe."""
    if hasattr(value, "labels") and hasattr(value, "items"):
        return {
            "_type": "Node",
            "labels": list(value.labels),
            "properties": dict(value.items()),
        }
    if hasattr(value, "type") and hasattr(value, "start_node"):
        return {
            "_type": "Relationship",
            "type": value.type,
            "properties": dict(value.items()) if hasattr(value, "items") else {},
        }
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return {
            "_type": "Path",
            "nodes": [_serialize_value(n) for n in value.nodes],
            "relationships": [_serialize_value(r) for r in value.relationships],
        }
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def get_schema_overview(driver) -> str:
    """Return a compact text description of node labels + relationship types."""
    with driver.session(database=DATABASE_NAME) as session:
        labels = [r["label"] for r in session.run("CALL db.labels()").data()]
        rel_types = [
            r["relationshipType"]
            for r in session.run("CALL db.relationshipTypes()").data()
        ]
        sample_paragraf = session.run(
            "MATCH (p:Paragraf) RETURN properties(p) AS props LIMIT 2"
        ).data()
        sample_odsek = session.run(
            "MATCH (o:Odsek) RETURN properties(o) AS props LIMIT 2"
        ).data()

    lines = [
        "Node labels:",
        ", ".join(labels),
        "",
        "Relationship types:",
        ", ".join(rel_types),
        "",
        "Sample Paragraf properties:",
        json.dumps(sample_paragraf, ensure_ascii=False, indent=2),
        "",
        "Sample Odsek properties:",
        json.dumps(sample_odsek, ensure_ascii=False, indent=2),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Retrieval agent: question -> Cypher -> supporting structural nodes
# --------------------------------------------------------------------------- #

RETRIEVAL_SYSTEM_PROMPT = """You are a Neo4j Cypher expert querying a knowledge graph of the Slovak VAT Act (zákon č. 222/2004 Z. z.).

The graph contains:
  * Structural nodes: Paragraf, Odsek, Pismeno, Bod, Priloha — the hierarchy of the law text.
    They are linked by OBSAHUJE (contains) and/or JE_SUCASTOU (is part of).
  * Domain entities: Subjekt, Osoba, Dan, Sluzba, Tovar, Lehota, Obrat, Suma, etc.
    These are linked to structural nodes via relations like UPRAVUJE, DEFINUJE, VZTAHUJE_SA_NA, MA_PODMIENKU, MA_LEHOTU, MA_SUMU, etc.

Your job: given a question, find the Paragraf/Odsek/Pismeno nodes that contain the answer, and the domain entities involved.

Strategy (entity-anchored):
  1. Identify domain concepts in the question (e.g. obrat, registrácia, platiteľ, oslobodenie, call-off stock, poukaz).
  2. Find matching entity nodes by text match on their `id` / `name` / `text` properties.
       Use case-insensitive CONTAINS: WHERE toLower(n.id) CONTAINS toLower('obrat')
  3. Traverse from those entities back to structural nodes (Paragraf/Odsek/Pismeno).
       MATCH (e)-[r]-(s) WHERE s:Paragraf OR s:Odsek OR s:Pismeno RETURN ...
  4. Pull the structural node IDs and any neighboring entities.
  5. Always LIMIT results (25-50). Iterate: explore first, then refine.

Rules:
  * Use the search_database tool — never claim a node exists without running a query.
  * Quote labels with hyphens/digits in backticks if needed.
  * Stop when you have 1–6 distinct Paragraf/Odsek/Pismeno IDs that plausibly cover the question.
  * Return their IDs verbatim from the database — do not invent IDs.
"""


RETRIEVAL_RESPONSE_SCHEMA = {
    "title": "RetrievalResult",
    "type": "object",
    "properties": {
        "section_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of Paragraf/Odsek/Pismeno/Bod nodes that support the answer (verbatim from DB).",
        },
        "entity_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of non-structural domain entities encountered (Dan, Subjekt, Lehota, ...).",
        },
        "reasoning": {
            "type": "string",
            "description": "Short note on the Cypher path taken.",
        },
    },
    "required": ["section_ids", "entity_ids", "reasoning"],
}


def build_search_tool(driver):
    """Return a LangChain tool that runs Cypher against the configured database."""

    @tool
    def search_database(cypher_query: str) -> str:
        """Execute a read-only Cypher query against the VAT-law Neo4j database.

        Args:
            cypher_query: Any valid read-only Cypher. Always include a LIMIT.

        Returns:
            JSON string of result records, or an error message.
        """
        try:
            with driver.session(database=DATABASE_NAME) as session:
                records = [dict(r) for r in session.run(cypher_query)]
            serialized = [
                {k: _serialize_value(v) for k, v in rec.items()} for rec in records
            ]
            return json.dumps(serialized, ensure_ascii=False, default=str)[:8000]
        except Exception as exc:
            return f"Query error: {exc}"

    return search_database


def retrieve_supporting_sections(
    driver, llm: ChatOpenAI, question: str, schema_overview: str
) -> Dict[str, Any]:
    """Run the retrieval agent. Returns dict with section_ids, entity_ids, reasoning."""
    search_tool = build_search_tool(driver)

    user_prompt = f"""Question (Slovak): {question}

Graph schema overview:
{schema_overview}

Find the Paragraf/Odsek/Pismeno nodes that support an answer to this question, and the domain entities involved. Begin by querying for relevant entities."""

    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        response_format=ProviderStrategy(schema=RETRIEVAL_RESPONSE_SCHEMA),  # type: ignore[arg-type]
    )
    response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    return response["structured_response"]


# --------------------------------------------------------------------------- #
# Section text aggregation
# --------------------------------------------------------------------------- #

def fetch_sections_with_context(driver, section_ids: List[str]) -> List[Dict[str, Any]]:
    """For each section ID, fetch its label, text, and its ancestors/descendants
    in the Paragraf -> Odsek -> Pismeno hierarchy.

    Returns a list of dicts: {id, labels, text, paragraf, odsek, pismeno, children}
    """
    if not section_ids:
        return []

    cypher = """
    UNWIND $ids AS sid
    MATCH (n {id: sid})
    OPTIONAL MATCH (par:Paragraf)-[:OBSAHUJE*0..3]->(n)
    OPTIONAL MATCH (n)-[:OBSAHUJE*0..3]->(child)
        WHERE child:Odsek OR child:Pismeno OR child:Bod
    WITH n, collect(DISTINCT par) AS parents, collect(DISTINCT child) AS children
    RETURN n.id AS id,
           labels(n) AS labels,
           coalesce(n.text, n.nazov, n.title, '') AS text,
           [p IN parents | {id: p.id, text: coalesce(p.text, p.nazov, '')}] AS parents,
           [c IN children | {id: c.id, labels: labels(c), text: coalesce(c.text, c.nazov, '')}] AS children
    """
    with driver.session(database=DATABASE_NAME) as session:
        return [dict(r) for r in session.run(cypher, ids=section_ids)]


def aggregate_section_text(sections: List[Dict[str, Any]]) -> str:
    """Build the context string handed to the answer agent.

    Strategy:
      * Sort by section ID so output reads like the law (§ 4 before § 7, etc.).
      * Header shows ancestor Paragraf → matched node label + id for citation.
      * Skip child text that's already contained in the matched node's text
        (Pismeno content is often quoted inside its parent Odsek).
      * Drop empty blocks.
    """
    blocks: List[str] = []
    for sec in sorted(sections, key=lambda s: s["id"]):
        label = sec["labels"][0] if sec["labels"] else "Node"
        parent_id = sec["parents"][0]["id"] if sec.get("parents") else ""
        header = (
            f"### {parent_id} → {sec['id']} ({label})"
            if parent_id and parent_id != sec["id"]
            else f"### {sec['id']} ({label})"
        )

        body = (sec["text"] or "").strip()
        children: List[str] = []
        for c in sec.get("children", []):
            ctext = c.get("text")
            if isinstance(ctext, str):
                ctext = ctext.strip()
                if ctext and ctext not in body:
                    children.append(ctext)

        block = "\n".join(filter(None, [header, body, *children])).strip()
        if block != header:
            blocks.append(block)

    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Answer agent
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM_PROMPT = """Si expert na slovenský zákon o DPH (zákon č. 222/2004 Z. z.).
Odpovedáš výlučne na základe poskytnutého kontextu z paragrafov, odsekov a písmen zákona.
Odpoveď je vecná, stručná (2–5 viet), v slovenčine. Ak kontext odpoveď neobsahuje, povedz to."""


def generate_answer(
    llm: ChatOpenAI, question: str, context_text: str, entity_ids: List[str]
) -> str:
    """Run the answer agent on the aggregated context."""
    if not context_text.strip():
        return "Z poskytnutého kontextu nemožno odpoveď zostaviť."

    user_prompt = f"""Otázka: {question}

Kontext (úryvky zo zákona):
{context_text}

Súvisiace entity: {', '.join(entity_ids) if entity_ids else '—'}

Odpovedaj na otázku."""

    response = llm.invoke(
        [
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
    )
    content = response.content
    return content.strip() if isinstance(content, str) else str(content).strip()


# --------------------------------------------------------------------------- #
# Incremental output writer
# --------------------------------------------------------------------------- #

def append_result(path: Path, results: List[Dict[str, Any]], new_item: Dict[str, Any]) -> None:
    results.append(new_item)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USERNAME")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    missing = [
        name
        for name, val in [
            ("NEO4J_URI", neo4j_uri),
            ("NEO4J_USERNAME", neo4j_user),
            ("NEO4J_PASSWORD", neo4j_password),
            ("OPENAI_API_KEY", openai_api_key),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")
    assert neo4j_uri and neo4j_user and neo4j_password and openai_api_key  # narrow Optional[str]

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0, api_key=openai_api_key)  # type: ignore[arg-type]

    try:
        schema_overview = get_schema_overview(driver)
        print(f"Schema loaded from database {DATABASE_NAME}.\n")

        questions = json.loads(QA_INPUT_PATH.read_text(encoding="utf-8"))
        results: List[Dict[str, Any]] = []
        QA_OUTPUT_PATH.write_text("[]", encoding="utf-8")  # fresh start each run

        for idx, item in enumerate(questions, start=1):
            question = item["otazka"]
            print(f"\n[{idx}/{len(questions)}] {question[:120]}...")
            t0 = time.time()

            try:
                retrieval = retrieve_supporting_sections(
                    driver, llm, question, schema_overview
                )
                section_ids = retrieval.get("section_ids", []) or []
                entity_ids_raw = retrieval.get("entity_ids", []) or []

                sections = fetch_sections_with_context(driver, section_ids)
                context_text = aggregate_section_text(sections)

                # Filter found_entities to non-structural only.
                found_entities = [
                    eid for eid in entity_ids_raw
                    if not any(lbl in eid for lbl in STRUCTURAL_LABELS)
                ]

                # found_sections enriched with labels + text snippets for human review.
                found_sections = [
                    {
                        "id": s["id"],
                        "labels": s["labels"],
                        "text": (s["text"] or "")[:300],
                    }
                    for s in sections
                ]

                answer = generate_answer(llm, question, context_text, found_entities)

                result = {
                    "question": question,
                    "supporting_sections": {
                        "podporujuce_strany": item.get("podporujuce_strany", []),
                        "podporujuce_ustanovenia": item.get("podporujuce_ustanovenia", []),
                    },
                    "found_sections": found_sections,
                    "found_entities": found_entities,
                    "answer": answer,
                    "retrieval_reasoning": retrieval.get("reasoning", ""),
                    "elapsed_sec": round(time.time() - t0, 2),
                    "timestamp": datetime.now().isoformat(),
                }

            except Exception as exc:
                result = {
                    "question": question,
                    "supporting_sections": {
                        "podporujuce_strany": item.get("podporujuce_strany", []),
                        "podporujuce_ustanovenia": item.get("podporujuce_ustanovenia", []),
                    },
                    "found_sections": [],
                    "found_entities": [],
                    "answer": "",
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(),
                }
                print(f"  ! error: {exc}")

            append_result(QA_OUTPUT_PATH, results, result)
            print(f"  ok  sections={len(result['found_sections'])}  "
                  f"entities={len(result['found_entities'])}  "
                  f"{result.get('elapsed_sec', '-')}s")

        print(f"\nDone. Results -> {QA_OUTPUT_PATH}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
