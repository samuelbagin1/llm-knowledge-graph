"""CLI helper invoked by the kg-graphrag-retriever Codex agent.

Usage:
    python test_dataset/cypher_query.py "MATCH (n:Paragraf) RETURN n.id LIMIT 5"

The agent calls this from its sandboxed shell once per Cypher query. Whatever this
script prints to stdout becomes the agent's tool result — so the format, size, and
error shape directly shape what the retrieval agent can reason about.

Reads NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD from .env (same as graphrag.py).
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Reuse the Node/Relationship/Path serializer already proven in graphrag.py.
from graphrag import _serialize_value, DATABASE_NAME


MAX_PAYLOAD_CHARS = 8000


def run_cypher(driver, cypher: str) -> str:
    """Execute the Cypher query and return a JSON string for Codex's shell tool.

    Output shape:
        {"rows": [...], "row_count": N, "truncated": bool}
        {"error": "..."}  on failure (exit 0 so Codex parses the JSON and retries)

    Truncation: drop whole rows from the tail until the JSON fits in
    MAX_PAYLOAD_CHARS. `row_count` always reflects the true number of rows
    returned by Neo4j so the agent knows when its `LIMIT` was honored.
    """
    try:
        with driver.session(database=DATABASE_NAME) as session:
            records = [
                {k: _serialize_value(v) for k, v in dict(r).items()}
                for r in session.run(cypher)
            ]
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)

    total = len(records)
    kept = records
    truncated = False
    while True:
        payload = json.dumps(
            {"rows": kept, "row_count": total, "truncated": truncated},
            ensure_ascii=False,
            default=str,
        )
        if len(payload) <= MAX_PAYLOAD_CHARS or not kept:
            return payload
        kept = kept[:-1]
        truncated = True


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "usage: cypher_query.py <cypher>"}))
        return 2

    cypher = sys.argv[1]
    load_dotenv()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    if not (uri and user and password):
        print(json.dumps({"error": "missing NEO4J_* env vars"}))
        return 2

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        print(run_cypher(driver, cypher))
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
