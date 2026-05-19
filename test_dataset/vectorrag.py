"""
Q&A VectorRAG evaluation against the Slovak VAT-law corpus.

Pipeline per question:
    1. Embed the question with OpenAI text-embedding-3-large.
    2. Query a Qdrant collection of pre-embedded law chunks for top-k.
    3. Answer agent (gpt-5.5) reads aggregated chunk text + section labels
       and produces a natural-language answer.
    4. Result is appended incrementally to predictions JSON.

Run twice — once with K_NEIGHBORS=5, once with K_NEIGHBORS=10 — to get
two comparable prediction files.

Embeddings are persisted on disk via Qdrant local mode; collection
is reused across re-runs (skipping embedding cost) if vector count matches.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_openai import ChatOpenAI


load_dotenv()

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

ANSWER_MODEL = "gpt-5.5"
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
EMBED_BATCH = 96

# Switch this to 5 or 10 to run the two experiments.
K_NEIGHBORS = 10

BASE_DIR = Path(__file__).parent
REPO_ROOT = BASE_DIR.parent

CHUNKS_PATH = REPO_ROOT / "file_output" / "chunks_20260518182455.json"
QA_INPUT_PATH = BASE_DIR / "q&a" / "q&a.json"

QDRANT_PATH = BASE_DIR / "qdrant_storage"
COLLECTION_NAME = "vat_law_chunks"

OUTPUT_DIR = BASE_DIR / "eval_results" / f"vectorrag_k{K_NEIGHBORS}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QA_OUTPUT_PATH = OUTPUT_DIR / "q&a_vectorrag_validation.json"


# --------------------------------------------------------------------------- #
# Section label formatting (turns ["§ 4", "1", "b)"] -> "§ 4 ods. 1 písm. b)")
# --------------------------------------------------------------------------- #

def format_section_label(path: List[str]) -> str:
    if not path:
        return ""
    parts: List[str] = []
    for i, segment in enumerate(path):
        seg = segment.strip()
        if i == 0:
            parts.append(seg)
            continue
        if seg.endswith(")") and any(c.isalpha() for c in seg.rstrip(")")):
            parts.append(f"písm. {seg}")
        elif seg.endswith("."):
            parts.append(f"bod {seg}")
        else:
            parts.append(f"ods. {seg}")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #

def embed_texts(openai_client: OpenAI, texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of strings with text-embedding-3-large."""
    out: List[List[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([d.embedding for d in resp.data])
    return out


# --------------------------------------------------------------------------- #
# Qdrant collection bootstrap
# --------------------------------------------------------------------------- #

def ensure_collection(
    qdrant: QdrantClient,
    openai_client: OpenAI,
    chunks: List[Dict[str, Any]],
) -> None:
    """Create + populate the collection only if it isn't already populated."""
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME in existing:
        info = qdrant.get_collection(COLLECTION_NAME)
        if info.points_count == len(chunks):
            print(
                f"Collection '{COLLECTION_NAME}' already populated "
                f"({info.points_count} pts) — skipping embedding."
            )
            return
        print(
            f"Collection '{COLLECTION_NAME}' has {info.points_count} pts, "
            f"expected {len(chunks)} — recreating."
        )
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=EMBED_DIM, distance=qmodels.Distance.COSINE
        ),
    )

    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL}…")
    t0 = time.time()
    embeddings = embed_texts(openai_client, [c["text"] for c in chunks])
    print(f"  done in {time.time() - t0:.1f}s")

    points = [
        qmodels.PointStruct(
            id=chunk["id"],
            vector=vec,
            payload={
                "text": chunk["text"],
                "path": chunk.get("path", []),
                "section_label": format_section_label(chunk.get("path", [])),
                "headline": chunk.get("headline", ""),
            },
        )
        for chunk, vec in zip(chunks, embeddings)
    ]

    # Upsert in batches to keep memory + payload sizes reasonable.
    for i in range(0, len(points), 256):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 256])
    print(f"Upserted {len(points)} points into '{COLLECTION_NAME}'.")


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def retrieve_chunks(
    qdrant: QdrantClient,
    openai_client: OpenAI,
    question: str,
    k: int,
) -> List[Dict[str, Any]]:
    q_emb = embed_texts(openai_client, [question])[0]
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_emb,
        limit=k,
        with_payload=True,
    ).points

    results: List[Dict[str, Any]] = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "chunk_id": h.id,
                "score": float(h.score) if h.score is not None else 0.0,
                "text": payload.get("text", ""),
                "path": payload.get("path", []),
                "section_label": payload.get("section_label", ""),
                "headline": payload.get("headline", ""),
            }
        )
    return results


def aggregate_chunk_text(retrieved: List[Dict[str, Any]]) -> str:
    """Build the answer-agent context from the retrieved chunks.

    Each chunk gets a header with its section label + headline so the
    agent can ground its answer in identifiable provisions.
    """
    blocks: List[str] = []
    for r in retrieved:
        label = r.get("section_label") or " / ".join(r.get("path", []))
        headline = r.get("headline", "")
        text = (r.get("text") or "").strip()
        if not text:
            continue
        header = f"### {label} — {headline}" if headline else f"### {label}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------- #
# Answer agent (identical prompt + model to graphrag.py for fair comparison)
# --------------------------------------------------------------------------- #

ANSWER_SYSTEM_PROMPT = """Si expert na slovenský zákon o DPH (zákon č. 222/2004 Z. z.).
Odpovedáš výlučne na základe poskytnutého kontextu z paragrafov, odsekov a písmen zákona.
Odpoveď je vecná, stručná (2–5 viet), v slovenčine. Ak kontext odpoveď neobsahuje, povedz to."""


def generate_answer(llm: ChatOpenAI, question: str, context_text: str) -> str:
    if not context_text.strip():
        return "Z poskytnutého kontextu nemožno odpoveď zostaviť."

    user_prompt = f"""Otázka: {question}

Kontext (úryvky zo zákona):
{context_text}

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
# Incremental writer
# --------------------------------------------------------------------------- #

def append_result(
    path: Path, results: List[Dict[str, Any]], new_item: Dict[str, Any]
) -> None:
    results.append(new_item)
    path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #

def main() -> None:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Missing env var: OPENAI_API_KEY")

    openai_client = OpenAI(api_key=openai_api_key)
    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0, api_key=openai_api_key)  # type: ignore[arg-type]

    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    qdrant = QdrantClient(path=str(QDRANT_PATH))

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH.name}.")

    ensure_collection(qdrant, openai_client, chunks)

    all_questions = json.loads(QA_INPUT_PATH.read_text(encoding="utf-8"))
    results: List[Dict[str, Any]] = []
    QA_OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = len(all_questions)
    print(f"\nRunning vectorrag eval with k={K_NEIGHBORS} on {total} questions → {QA_OUTPUT_PATH}\n")

    for idx, item in enumerate(all_questions, start=1):
        qid = item.get("id")
        question = item["otazka"]
        print(f"[{idx}/{total}] id={qid}  {question[:120]}...")
        t0 = time.time()

        try:
            retrieved = retrieve_chunks(qdrant, openai_client, question, K_NEIGHBORS)
            context_text = aggregate_chunk_text(retrieved)
            answer = generate_answer(llm, question, context_text)

            result = {
                "id": qid,
                "question": question,
                "k": K_NEIGHBORS,
                "supporting_sections": {
                    "podporujuce_strany": item.get("podporujuce_strany", []),
                    "podporujuce_ustanovenia": item.get("podporujuce_ustanovenia", []),
                },
                "retrieved_chunks": [
                    {
                        "chunk_id": r["chunk_id"],
                        "score": round(r["score"], 4),
                        "section_label": r["section_label"],
                        "headline": r["headline"],
                        "text": r["text"][:400],
                    }
                    for r in retrieved
                ],
                "answer": answer,
                "elapsed_sec": round(time.time() - t0, 2),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as exc:
            result = {
                "id": qid,
                "question": question,
                "k": K_NEIGHBORS,
                "supporting_sections": {
                    "podporujuce_strany": item.get("podporujuce_strany", []),
                    "podporujuce_ustanovenia": item.get("podporujuce_ustanovenia", []),
                },
                "retrieved_chunks": [],
                "answer": "",
                "error": str(exc),
                "timestamp": datetime.now().isoformat(),
            }
            print(f"  ! error: {exc}")

        append_result(QA_OUTPUT_PATH, results, result)
        chunks_count = len(result.get("retrieved_chunks", []))
        print(
            f"  ok  retrieved={chunks_count}  "
            f"{result.get('elapsed_sec', '-')}s"
        )

    print(f"\nDone. Results → {QA_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
