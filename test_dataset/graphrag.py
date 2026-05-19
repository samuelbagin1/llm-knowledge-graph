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

MODEL_NAME = "gpt-5.5"
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

Schema:
  * Structural nodes — Paragraf, Odsek, Pismeno, Bod — form the hierarchy of the law.
    Linked top-down by OBSAHUJE: (Paragraf)-[:OBSAHUJE]->(Odsek)-[:OBSAHUJE]->(Pismeno)-[:OBSAHUJE]->(Bod).
    Properties on every structural node:
      - `id`   e.g. "Paragraf § 4 Odsek 1 Pismeno f)"
      - `path` e.g. ["§ 4", "1", "f)"]
      - `text` the verbatim Slovak law text — THIS is what you read to verify relevance.
  * Domain entities — Subjekt, Osoba, Dan, Sluzba, Tovar, Lehota, Obrat, Suma, Podmienka, Povinnost, Pravo, Registracia, Oznamenie, Ziadost, Nehnutelnost, Vozidlo, etc.
  * Positive verb relations: UPRAVUJE, DEFINUJE, VZTAHUJE_SA_NA, MA_PODMIENKU, MA_LEHOTU, MA_SUMU,
        JE_PREDMETOM_DANE, PODLIEHA, MA_POVINNOST, MA_PRAVO, MA_NAROK_NA, SPLNA_PODMIENKY, OBSAHUJE.
  * NEGATIVE / EXCLUSIONARY relations — equally important for completeness:
        NEVZTAHUJE_SA_NA, NESPLNA_PODMIENKY, NEMA_NAROK_NA, NIE_JE_PREDMETOM_DANE, OSLOBODZUJE_OD,
        ZANIKA, RUSI. These mark clauses that EXCLUDE or EXEMPT — you MUST also retrieve them
        when they are relevant to the question.
  * Cross-references: ODKAZUJE_NA links one section to another it cites.

Your job: given a question, run MANY Cypher queries to gather a thorough set of Paragraf/Odsek/Pismeno
nodes (with their `text`), positive AND negative provisions, plus all involved domain entities.
Be exhaustive, not minimal. Then read the `text` of each candidate and KEEP ONLY those whose text
actually supports, contradicts, qualifies, or excludes some aspect of the question.

Search strategy — perform these steps in order, multiple tool calls each:

  STEP 1. Decompose the question into 3–6 distinct domain concepts (e.g. "obrat", "registrácia",
          "platiteľ", "lehota oznámenia", "oslobodenie", "call-off stock", "reverse charge").

  STEP 2. For EACH concept, find candidate entity nodes (case-insensitive):
            MATCH (n) WHERE toLower(n.id) CONTAINS toLower('obrat')
            RETURN n.id, labels(n)[0] AS label LIMIT 25
          Also try synonyms / morphological variants: registr, oslobod, platit, lehot, sumy.

  STEP 3. From each entity, traverse to structural nodes BOTH directions, ALL relevant rel types
          including negations:
            MATCH (e {id: 'Obrat'})-[r]-(s)
            WHERE (s:Paragraf OR s:Odsek OR s:Pismeno OR s:Bod)
              AND type(r) IN ['UPRAVUJE','DEFINUJE','VZTAHUJE_SA_NA','MA_PODMIENKU','MA_LEHOTU',
                              'MA_SUMU','JE_PREDMETOM_DANE','PODLIEHA','OSLOBODZUJE_OD',
                              'NEVZTAHUJE_SA_NA','NESPLNA_PODMIENKY','NIE_JE_PREDMETOM_DANE',
                              'NEMA_NAROK_NA','ODKAZUJE_NA','SUVISI_S']
            RETURN DISTINCT s.id, labels(s)[0] AS label, type(r) AS rel LIMIT 50

  STEP 4. GO DEEPER — 2-hop entity chains often reveal related provisions:
            MATCH (e1 {id:'Obrat'})-[*1..2]-(s)
            WHERE (s:Paragraf OR s:Odsek OR s:Pismeno) RETURN DISTINCT s.id, labels(s)[0] LIMIT 50
          Also expand to SIBLINGS within the same Paragraf:
            MATCH (par:Paragraf {id:'Paragraf § 4'})-[:OBSAHUJE*1..3]->(sib)
            RETURN sib.id, labels(sib)[0] LIMIT 50
          And follow ODKAZUJE_NA cross-references between sections.

  STEP 5. Fetch text + parent for every candidate:
            UNWIND $ids AS sid
            MATCH (s {id: sid})
            OPTIONAL MATCH (par:Paragraf)-[:OBSAHUJE*1..3]->(s)
            RETURN s.id AS id, labels(s)[0] AS label, s.text AS text, par.id AS parent_id
          (You can pass a list via parameters or repeat single-id MATCHes.)

  STEP 6. VERIFY by reading the `text`. For each candidate ask yourself:
            - Does this text say something the question asks about?
            - Does it state a condition, threshold, exception, or exclusion relevant to the question?
            - Or is it just incidentally connected via a shared entity but off-topic?
          KEEP sections whose text directly bears on the question (supporting OR excluding).
          DROP sections whose text is unrelated even if they share an entity link.

Quality bar:
  * Aim for 5–15 verified sections covering ALL aspects (conditions, thresholds, deadlines,
    exclusions, exceptions, cross-references) — not just the most obvious one.
  * If the question implies a negative scenario ("nie je", "neuplatňuje sa", "oslobodenie"),
    you MUST search for and include the relevant exclusion/exemption clauses.
  * Every returned section must have its real `text` populated. No empty texts unless the
    DB truly has none.
  * Every section id and every text snippet must come from a real search_database call.
    Never invent ids, never paraphrase the text.
"""


RETRIEVAL_RESPONSE_SCHEMA = {
    "title": "RetrievalResult",
    "type": "object",
    "properties": {
        "supporting_sections": {
            "type": "array",
            "description": "Verified Paragraf/Odsek/Pismeno/Bod nodes — each one's text was read and confirmed to bear on the question (supporting, conditioning, or excluding).",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "Node id verbatim from DB."},
                    "label": {"type": "string", "description": "Paragraf, Odsek, Pismeno, or Bod."},
                    "text": {"type": "string", "description": "Verbatim law text from n.text."},
                    "parent_id": {"type": "string", "description": "Containing Paragraf id, or '' if none."},
                    "role": {
                        "type": "string",
                        "enum": ["supports", "conditions", "excludes", "defines", "cross_reference"],
                        "description": "How this section relates to the question.",
                    },
                    "relevance": {
                        "type": "string",
                        "description": "One short sentence on why this section is in the result — what aspect of the question its text covers.",
                    },
                },
                "required": ["id", "label", "text", "role", "relevance"],
            },
        },
        "entity_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "IDs of non-structural domain entities encountered (Dan, Subjekt, Lehota, ...).",
        },
        "reasoning": {
            "type": "string",
            "description": "Concise log: decomposed concepts, Cypher paths explored, how many candidates were dropped after reading text.",
        },
    },
    "required": ["supporting_sections", "entity_ids", "reasoning"],
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

Run an EXHAUSTIVE search:
  1. Decompose the question into all distinct concepts (incl. negations: "nie je", "neuplatňuje", "oslobodenie", "výnimka").
  2. For each concept, find entities, traverse 1–2 hops, AND look at sibling Pismeno within the same Paragraf and ODKAZUJE_NA cross-references.
  3. Pull text for every candidate; READ each text and KEEP only those whose text actually addresses some aspect of the question — supporting, conditioning, or excluding.
  4. If the question involves exemptions/exclusions, include the explicit NEVZTAHUJE_SA_NA / NIE_JE_PREDMETOM_DANE / OSLOBODZUJE_OD / NEMA_NAROK_NA / NESPLNA_PODMIENKY clauses.
  5. Aim for 5–15 verified sections, each with `role` and a one-sentence `relevance` justification.

Begin now."""

    agent = create_agent(
        model=llm,
        tools=[search_tool],
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
        response_format=ProviderStrategy(schema=RETRIEVAL_RESPONSE_SCHEMA),  # type: ignore[arg-type]
    )
    response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
    return response["structured_response"]


# --------------------------------------------------------------------------- #
# Section text aggregation (consumes data the retrieval agent gathered)
# --------------------------------------------------------------------------- #

def aggregate_section_text(sections: List[Dict[str, Any]]) -> str:
    """Format the agent's `supporting_sections` into a context string for the answer agent.

    Each section dict from the retrieval agent contains: id, label, text, parent_id, role, relevance.
    The header surfaces `role` so the answer agent knows whether the clause supports or excludes.
    """
    blocks: List[str] = []
    for sec in sorted(sections, key=lambda s: s.get("id", "")):
        sid = sec.get("id", "")
        label = sec.get("label", "Node")
        parent_id = sec.get("parent_id", "") or ""
        text = (sec.get("text") or "").strip()
        role = sec.get("role", "")
        if not sid or not text:
            continue

        role_tag = f" [{role}]" if role else ""
        header = (
            f"### {parent_id} → {sid} ({label}){role_tag}"
            if parent_id and parent_id != sid
            else f"### {sid} ({label}){role_tag}"
        )
        blocks.append(f"{header}\n{text}")

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
                sections = retrieval.get("supporting_sections", []) or []
                entity_ids_raw = retrieval.get("entity_ids", []) or []

                context_text = aggregate_section_text(sections)

                # Filter found_entities to non-structural only.
                found_entities = [
                    eid for eid in entity_ids_raw
                    if not any(lbl in eid for lbl in STRUCTURAL_LABELS)
                ]

                # found_sections: surface what the agent collected, snippets only.
                found_sections = [
                    {
                        "id": s.get("id", ""),
                        "label": s.get("label", ""),
                        "parent_id": s.get("parent_id", ""),
                        "role": s.get("role", ""),
                        "relevance": s.get("relevance", ""),
                        "text": (s.get("text") or "")[:300],
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
