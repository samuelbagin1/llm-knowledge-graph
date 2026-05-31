"""Entry point exposing PDFGraphRAG pipeline stages as standalone callables.

Each stage of `PDFGraphRAG.process()` is wrapped in its own thin function so
it can be triggered independently:

    tables/formulas      -> graphrag.tables_and_formulas(...)
    structural chunker   -> get_structured_chunks(...)
    open-domain detect.  -> run_odd(...)
    schema refinement    -> run_refinement(...)
    schema-driven extr.  -> run_sde(...)  /  run_sde_sample(...)

ODD/Refinement/SDE accept an optional `cache_path`. When the file exists, the
loader from `loaders.py` is used and the LLM call is skipped — making it cheap
to re-run only the part of the pipeline that failed.
"""

import asyncio
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter

from classes import Schema
from pdf_graphrag import PDFGraphRAG
from chunker.chunker import Chunker
from to_json import odd_to_json, refinement_to_json, sde_to_json
from loaders import load_odd, load_refinement, load_sde

load_dotenv()


# ---------- CONFIG ----------
PDF_PATH = './code/assets/ZZ_2004_222_20260101.pdf'
DOCUMENT_ID = Path(PDF_PATH).stem
NAME_OF_CHAIN = 'chain'
WRITE_JSON = True


def get_graphrag() -> PDFGraphRAG:
    """Construct a PDFGraphRAG client wired to the local Neo4j + API keys."""
    return PDFGraphRAG(
        neo4j_uri='neo4j://127.0.0.1:7687',
        neo4j_user='neo4j',
        neo4j_password='fseijkfbsj48@',
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        database="zz-2004-222",
    )


# ---------- CHUNKING HELPERS ----------

def get_structured_chunks(pdf_path: str = PDF_PATH, write_json: bool = WRITE_JSON) -> dict:
    """Run the Slovak-law structural chunker (§ -> odsek -> pismeno -> bod).
    Only works for financial laws and not late edits and amendments.

    Returns {"chunks": List[Document], "tree_graph": GraphDocument, "last_page": int | None}.
    Used by SDE to preserve legal hierarchy in chunk metadata.
    """
    return Chunker().split_document(pdf_path, write_json=write_json)


def get_text_chunks(documents: list[Document], chunk_size: int = 1024, chunk_overlap: int = 128) -> list[Document]:
    """Pure-text chunking via RecursiveCharacterTextSplitter (used by ODD)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def build_sde_chunks(
    pdf_path: str,
    documents: list[Document],
    write_json: bool = WRITE_JSON,
) -> tuple[list[Document], GraphDocument]:
    """Build the exact chunk set that PDFGraphRAG.process() feeds to SDE.

    Mirrors process(): structural Chunker chunks (carry `path` metadata so the
    SDE prompt can inject "Paragraf § / Odsek / Pismeno" context) + trailing
    pages after the last § split by RecursiveCharacterTextSplitter on newlines.

    Returns (chunks, tree_graph). `tree_graph` MUST be ingested into Neo4j
    BEFORE the SDE output — SDE relationships reference Paragraf/Odsek/Pismeno
    nodes by id, and the tree provides their properties (headline, page_start).
    """
    chunker_result = Chunker().split_document(pdf_path, write_json=write_json)
    structured_chunks: list[Document] = chunker_result["chunks"]
    tree_graph: GraphDocument = chunker_result["tree_graph"]
    last_page = chunker_result["last_page"]

    # Pages after the final § are annexes/appendices — Chunker does not cover
    # them; chunk them on newlines so the splitter never cuts mid-line.
    trailing_documents = documents[last_page:] if last_page is not None else documents
    trailing_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=["\n"],
    )
    trailing_chunks = trailing_splitter.split_documents(trailing_documents) if trailing_documents else []

    return structured_chunks + trailing_chunks, tree_graph


def filter_table_pages(documents: list[Document], table_pages_to_exclude: set[int]) -> list[Document]:
    """Drop interior table pages (1-indexed) from PyPDFLoader documents (0-indexed).

    Mirrors the table-page exclusion process() applies between tables_and_formulas
    and ODD/SDE so the LLM does not re-process the structured table content.
    """
    if not table_pages_to_exclude:
        return documents
    return [
        doc for doc in documents
        if (doc.metadata.get("page", -1) + 1) not in table_pages_to_exclude
    ]


# ---------- INDEPENDENT PIPELINE STAGES ----------

def run_odd(
    graphrag: PDFGraphRAG,
    documents: list[Document],
    name: str = NAME_OF_CHAIN,
    write_json: bool = WRITE_JSON,
    cache_path: str | None = None,
) -> Schema:
    """Stage 1: open-domain detection — extract per-chunk Schemas then union.

    If `cache_path` exists, loads the previously-written ODD JSON instead of
    calling the LLM.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[run_odd] reusing {cache_path}")
        return load_odd(cache_path)

    chunked = get_text_chunks(documents, chunk_size=1200, chunk_overlap=200)
    schemas = asyncio.run(
        graphrag.async_open_domain_detection(
            chunked,
            write_json=write_json,
            name=name,
        )
    )
    # Union mirrors PDFGraphRAG.process() at the refinement boundary.
    return Schema(
        nodes=list({n for s in schemas for n in s.nodes}),
        relationships=list({r for s in schemas for r in s.relationships}),
    )


def run_refinement(
    graphrag: PDFGraphRAG,
    odd_schema: Schema,
    name: str = NAME_OF_CHAIN,
    write_json: bool = WRITE_JSON,
    cache_path: str | None = None,
) -> Schema:
    """Stage 2: schema refinement — canonicalize + merge with existing graph schema.

    If `cache_path` exists, loads a previously-written refinement JSON.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[run_refinement] reusing {cache_path}")
        return load_refinement(cache_path)

    data = graphrag.schema_refinement(
        odd_schema=odd_schema,
        existing_schema=graphrag.get_graph_schema(),
    )
    if write_json:
        refinement_to_json(data, name=name)
    return Schema(
        nodes=data['node_types'],
        relationships=data['relationship_types'],
    )


def run_sde(
    graphrag: PDFGraphRAG,
    chunks: list[Document],
    schema: Schema,
    document_id: str = DOCUMENT_ID,
    name: str = NAME_OF_CHAIN,
    write_json: bool = WRITE_JSON,
    cache_path: str | None = None,
) -> list[GraphDocument]:
    """Stage 3: schema-driven extraction on pre-built chunks.

    `chunks` MUST already be chunked — either from `build_sde_chunks()` (the
    process()-equivalent: Chunker + trailing splitter, retains `path` metadata
    for legal-context injection) or from `get_text_chunks()` (pure-text,
    loses paragraf context). If `cache_path` exists, loads SDE JSON instead.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[run_sde] reusing {cache_path}")
        return load_sde(cache_path)

    return asyncio.run(
        graphrag.async_schema_driven_extraction(
            chunks,
            schema=schema,
            document_id=document_id,
            write_json=write_json,
            name=name,
        )
    )


# ---------- SAMPLE RUNNER (for prompt / schema iteration) ----------

def run_sde_sample(
    graphrag: PDFGraphRAG,
    chunks: list[Document],
    schema: Schema,
    sample_size: int = 30,
    index_range: tuple[int, int] | None = None,
    seed: int | None = 42,
    document_id: str = DOCUMENT_ID,
    output_dir: str = './test_dataset/sample',
) -> list[int]:
    """Run SDE on `sample_size` random chunks inside `index_range`.

    `chunks` is the pre-built list (typically from `build_sde_chunks()`).
    `seed` makes the sample reproducible — same seed + range yields the same
    indices, so prompt/schema tweaks can be compared apples-to-apples. Pass
    `seed=None` for a fresh random pick each run. One JSON per chunk is
    written under `output_dir` so each result can be inspected in isolation.

    Returns the list of indices that were processed.
    """
    lo, hi = index_range or (0, len(chunks))
    hi = min(hi, len(chunks))
    if lo >= hi:
        raise ValueError(f"run_sde_sample: empty index_range ({lo}, {hi})")

    k = min(sample_size, hi - lo)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(lo, hi), k))

    print(f"[run_sde_sample] sampling {k} chunks from range [{lo}, {hi}) with seed={seed}")
    for i in indices:
        graph_docs = asyncio.run(
            graphrag.async_schema_driven_extraction(
                chunks[i:i + 1],
                schema=schema,
                document_id=document_id,
            )
        )
        sde_to_json(graph_docs, output_dir, name=str(i))

    print(f"\nSampled {len(indices)} chunks into graph documents at {output_dir}")
    return indices


# ---------- MAIN ----------

def main():
    graphrag = get_graphrag()
    documents = graphrag.load_pdf(pdf_path=PDF_PATH)
    document_id = graphrag.get_document_id(PDF_PATH)

    # # --- Full end-to-end pipeline ---
    # # tables + formulas -> ODD -> refinement -> SDE -> Neo4j ingest + vector stores.
    # # Use this when you want to ingest a fresh PDF in one shot.
    # graphrag.process(PDF_PATH, NAME_OF_CHAIN, write_json=WRITE_JSON)

#########################################################################################################
    # INDIVIDUAL PROCESSING

    # # --- Tables & formulas ---
    # # Returns (table_graph_docs, formula_graph_doc, table_pages_to_exclude).
    # # Ingest both into Neo4j; table_pages_to_exclude filters ODD/SDE later so the
    # # LLM does not re-process pages whose interior is already captured as tables.
    # table_graph_docs, formula_graph_doc, table_pages_to_exclude = graphrag.tables_and_formulas(
    #     pdf_path=PDF_PATH,
    #     document_id=document_id,
    #     documents=documents,
    #     write_json=WRITE_JSON,
    # )
    # if table_graph_docs:
    #     graphrag.add_graph_to_database(table_graph_docs)
    # if formula_graph_doc is not None:
    #     graphrag.add_graph_to_database(formula_graph_doc)
    # documents = filter_table_pages(documents, table_pages_to_exclude)


    # # --- Structural chunker (Slovak § -> odsek -> pismeno -> bod) ---
    # # Returns chunks + tree_graph (Document -> Paragraf -> ...) + last_page.
    # # Use this if you only need the structural tree (e.g. to inspect headlines).
    # # The full SDE flow below already invokes Chunker internally via build_sde_chunks().
    # chunker_result = get_structured_chunks(PDF_PATH, write_json=WRITE_JSON)
    # structured_chunks = chunker_result["chunks"]
    # tree_graph = chunker_result["tree_graph"]


    # # --- ODD: open-domain detection ---
    # # Discovers candidate node/relationship types from the raw text. Pass
    # # cache_path=./file_output/<name>_odd_<ts>.json to skip the LLM call.
    # # cache_path = "./extracted_data/test-10_odd_20260501185523.json"
    # odd_schema = run_odd(graphrag, documents, name=NAME_OF_CHAIN, write_json=WRITE_JSON, cache_path="./extracted_data/test-10_odd_20260501185523.json")


    # # --- Schema refinement ---
    # # Canonicalizes ODD output (dedup, casing, ASCII) and merges with the
    # # existing Neo4j schema. cache_path supported for resume.
    # refined_schema = run_refinement(graphrag, odd_schema, name=NAME_OF_CHAIN, write_json=WRITE_JSON, cache_path="./file_output/chain_ref_20260531174129.json")


    # # --- SDE: full schema-driven extraction (all chunks) ---
    # # build_sde_chunks() returns Chunker output (with `path` metadata used by the
    # # SDE prompt for legal-context injection) + a trailing splitter for annex pages.
    # # tree_graph MUST be persisted BEFORE the SDE output so Paragraf/Odsek/Pismeno
    # # nodes already exist when SDE relationships reference them.
    # sde_chunks, tree_graph = build_sde_chunks(PDF_PATH, documents, write_json=WRITE_JSON)
    # graphrag.add_graph_to_database(tree_graph)
    #
    # graph_docs = run_sde(
    #     graphrag,
    #     sde_chunks,
    #     schema=refined_schema,
    #     document_id=document_id,
    #     name=NAME_OF_CHAIN,
    #     write_json=WRITE_JSON,
    # )
    # # Append a Document -> Chunk skeleton so each Chunk node is linked to the source.
    # graph_docs.append(graphrag._add_document_chunk(len(sde_chunks), PDF_PATH, document_id))
    # graphrag.add_graph_to_database(graph_docs)


    # # --- SDE sample (30 random chunks) ---
    # # Fast iteration on prompts / schemas: reproducible random pick via seed,
    # # one JSON per chunk under output_dir for inspection. Chunks come from the
    # # SAME builder as full SDE so legal-context metadata is preserved.
    # sde_chunks, _tree_graph = build_sde_chunks(PDF_PATH, documents, write_json=WRITE_JSON)
    # refined_schema_from_disk = load_refinement('./extracted_data/test-10_ref_20260501185709.json')
    # run_sde_sample(
    #     graphrag,
    #     sde_chunks,
    #     schema=refined_schema_from_disk,
    #     sample_size=30,
    #     index_range=(0, len(sde_chunks)),
    #     seed=42,
    #     document_id=document_id,
    #     output_dir='./file_output/sample',
    # )


    # # --- Idempotent Neo4j re-ingest ---
    # # merge_new applies properties on both CREATE and MATCH (apoc-based) — use
    # # when re-running SDE on existing nodes to refresh their props.
    # # add_graph_to_database is the plain MERGE path (APOC required).
    # graphrag.add_graph_to_database(graph_docs)
    # graphrag.merge_new(graph_docs)


if __name__ == "__main__":
    main()
