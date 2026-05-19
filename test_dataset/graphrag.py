"""
Q&A GraphRAG evaluation against the Slovak VAT-law knowledge graph.

Pipeline per question:
    1. Retrieval: a local Codex custom agent (.codex/agents/kg-graphrag-retriever.toml)
       is invoked via `codex exec`. It uses its sandboxed shell to call
       test_dataset/cypher_query.py with successive Cypher queries, reads each
       result, and returns one JSON object with supporting_sections + entity_ids.
    2. aggregate_section_text() flattens the hierarchy into a context string.
    3. Answer agent (ChatOpenAI) reads the aggregated text + entities and
       produces a natural-language answer.
    4. Result is appended incrementally to q&a_graphrag_validation.json.

Neo4j database: zz-2004-222 (credentials in .env).
"""

import os
import json
import subprocess
import tempfile
import time
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI


load_dotenv()

MODEL_NAME = "gpt-5.5"
DATABASE_NAME = "zz-2004-222"

# Codex retrieval agent — see .codex/agents/kg-graphrag-retriever.toml
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODEX_AGENT_NAME = "kg-graphrag-retriever"
CODEX_AGENT_PATH = PROJECT_ROOT / ".codex" / "agents" / f"{CODEX_AGENT_NAME}.toml"
CODEX_BIN = os.getenv(
    "CODEX_BIN",
    "/Users/samuelbagin/.vscode/extensions/openai.chatgpt-26.506.31421-darwin-arm64/bin/macos-aarch64/codex",
)
CODEX_MODEL = "gpt-5.5"
CODEX_REASONING_EFFORT = "high"
CODEX_TIMEOUT_SECONDS = 900

QA_INPUT_PATH = Path(__file__).parent / "q&a.json"
QA_OUTPUT_PATH = Path(__file__).parent / "q&a_graphrag_validation.json"

# Resume control: skip every question whose `id` is below this value.
# Items with id < START_FROM_ID are preserved from the existing output file
# (if present) so prior runs are not lost. Set to 1 for a full fresh run.
START_FROM_ID = 20

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
# Retrieval agent: question -> Codex (shell -> cypher_query.py) -> sections
# --------------------------------------------------------------------------- #


def _load_codex_agent_instructions(path: Path = CODEX_AGENT_PATH) -> str:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return str(data.get("developer_instructions") or "")


def _iter_json_candidates(text: str) -> List[str]:
    """Pull JSON object candidates out of raw codex output (handles ``` fences)."""
    candidates: List[str] = []
    fence = "```"
    start = 0
    while True:
        fence_start = text.find(fence, start)
        if fence_start == -1:
            break
        content_start = text.find("\n", fence_start + len(fence))
        if content_start == -1:
            break
        fence_end = text.find(fence, content_start + 1)
        if fence_end == -1:
            break
        candidates.append(text[content_start + 1 : fence_end].strip())
        start = fence_end + len(fence)

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and first < last:
        candidates.append(text[first : last + 1])
    return candidates


def _parse_codex_json(output: str) -> Dict[str, Any]:
    stripped = output.strip()
    if not stripped:
        raise ValueError("codex returned empty output")
    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    for candidate in _iter_json_candidates(stripped):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue
    raise ValueError(f"could not parse JSON object from codex output: {stripped[:500]}")


def _build_codex_prompt(question: str, schema_overview: str, agent_instructions: str) -> str:
    return f"""You are executing the local Codex custom agent definition from {CODEX_AGENT_PATH}.

Agent name: {CODEX_AGENT_NAME}

# SYSTEM INSTRUCTIONS
{agent_instructions}

# GRAPH SCHEMA OVERVIEW (live, from db.labels()/db.relationshipTypes())
{schema_overview}

# QUESTION (Slovak)
{question}

Run an EXHAUSTIVE search via the shell tool — call
`python test_dataset/cypher_query.py "<cypher>"` repeatedly. Decompose the
question, traverse positive AND negative relations, fetch text, then verify
by reading each candidate's text before keeping it.

Return ONLY one JSON object matching the schema in the system instructions.
No prose, no fences.
"""


def retrieve_supporting_sections(
    question: str, schema_overview: str, agent_instructions: str
) -> Dict[str, Any]:
    """Drive the Codex retrieval agent and parse its final JSON output."""
    prompt = _build_codex_prompt(question, schema_overview, agent_instructions)

    with tempfile.NamedTemporaryFile(
        mode="w+", encoding="utf-8", suffix=".json", delete=True
    ) as output_file:
        command = [
            CODEX_BIN,
            "exec",
            "--cd",
            str(PROJECT_ROOT),
            "--sandbox",
            "workspace-write",
            "--output-last-message",
            output_file.name,
            "--model",
            CODEX_MODEL,
            "--config",
            f'model_reasoning_effort="{CODEX_REASONING_EFFORT}"',
            prompt,
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=CODEX_TIMEOUT_SECONDS,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            raise RuntimeError(
                f"codex exited with {completed.returncode}: {stderr or stdout}"
            )

        output_file.seek(0)
        final_message = output_file.read().strip()
        payload = _parse_codex_json(final_message or completed.stdout)

    payload.setdefault("supporting_sections", [])
    payload.setdefault("entity_ids", [])
    payload.setdefault("reasoning", "")
    return payload


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
        agent_instructions = _load_codex_agent_instructions()
        print(f"Schema loaded from database {DATABASE_NAME}.")
        print(f"Codex agent loaded from {CODEX_AGENT_PATH.name}.\n")

        all_questions = json.loads(QA_INPUT_PATH.read_text(encoding="utf-8"))

        # Resume: preserve previously-completed records (id < START_FROM_ID)
        # from the existing output file, and only run the remaining questions.
        results: List[Dict[str, Any]] = []
        if START_FROM_ID > 1 and QA_OUTPUT_PATH.exists():
            try:
                prior = json.loads(QA_OUTPUT_PATH.read_text(encoding="utf-8"))
                results = [r for r in prior if r.get("id", 0) < START_FROM_ID]
                print(
                    f"Resuming from id={START_FROM_ID}; "
                    f"kept {len(results)} prior record(s) from {QA_OUTPUT_PATH.name}."
                )
            except Exception as exc:
                print(f"Could not read prior output ({exc}); starting fresh.")
                results = []
        QA_OUTPUT_PATH.write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        questions = [q for q in all_questions if q.get("id", 0) >= START_FROM_ID]
        total = len(questions)

        for idx, item in enumerate(questions, start=1):
            qid = item.get("id")
            question = item["otazka"]
            print(f"\n[{idx}/{total}] id={qid}  {question[:120]}...")
            t0 = time.time()

            try:
                retrieval = retrieve_supporting_sections(
                    question, schema_overview, agent_instructions
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
                    "id": qid,
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
                    "id": qid,
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
