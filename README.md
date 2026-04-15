# PDF GraphRAG

A retrieval-augmented generation (RAG) pipeline that converts PDF documents into a Neo4j knowledge graph + vector stores, then answers natural-language questions by combining symbolic graph traversal with LLM reasoning.

The core implementation lives in [code/pdf_graphrag.py](code/pdf_graphrag.py).

---

## 1. Overview

`PDFGraphRAG` is an end-to-end system that:

1. **Ingests** PDFs (text + tables + figures).
2. **Builds a knowledge graph** in Neo4j using a two-phase schema construction (open-domain detection → schema refinement) followed by schema-driven extraction.
3. **Embeds** nodes, chunks, and relationship types into Neo4j vector indices.
4. **Answers questions** via a 3-stage pipeline inspired by **KG-GPT** (segment → retrieve → infer), where Stage 2 uses an **ARK-V1** state-machine retriever (Klein & Ohnemus 2025) for grounded, schema-constrained graph traversal.

The domain target is Slovak legal/financial documents (laws, paragraphs, tariff tables), so many prompts are written in Slovak and enforce ASCII normalization (no diacritics).

---

## 2. Architecture & Components

### 2.1 External dependencies

- **Neo4j** (via `langchain_neo4j`) — graph store + vector store backend. APOC plugin is used by `add_graph_documents`.
- **LLM clients** (initialized in `__init__`):
  - `openai_client` (ChatOpenAI) — used for agent-based extraction stages (ODD, SDE, segmentation, search agent).
  - `openai_graph_transform` (gpt-4o-mini) — reserved for graph transformation.
  - `claude_client` (claude-haiku-4-5) — available but not used in the main pipeline.
  - `gemini_client` / `gemini_client_thinking` / `gemini_client_flash` — used for schema refinement, table→graph transformation, relation selection, reasoning steps, and final inference.
- **OpenAI embeddings** (`text-embedding-3-large`) — used for all three vector stores.
- **DocLayout-YOLO** (`doclayout_yolo` + Hugging Face Hub) — detects tables/figures/formulas on PDF page images.
- **pdf2image / PyPDFLoader / RecursiveCharacterTextSplitter** — PDF I/O and chunking.

### 2.2 Module-level helpers

- `is_read_only_cypher(query)` — regex-based sanitizer that rejects any Cypher containing write keywords (CREATE/MERGE/DELETE/SET/…) or `CALL` procedures. Used before every ad-hoc query in the retrieval path.
- `format_property_key` / `format_node_type` / `format_relationship_type` — normalize raw LLM output (camelCase properties, Capitalized node types, UPPER_SNAKE_CASE relations).
- `graph_document_to_json` / `serialize_for_json` — JSON-safe serializers for `GraphDocument` and Neo4j driver objects (Node, Relationship, Path, datetime).

---

## 3. Ingestion Pipeline — `process(pdf_path)`

The `process` method ([pdf_graphrag.py:1235](code/pdf_graphrag.py#L1235)) orchestrates everything. Flow:

### Step 1 — Load & detect layout
- `load_pdf` → `PyPDFLoader` returns per-page `Document` objects.
- `detect_tables` ([pdf_graphrag.py:621](code/pdf_graphrag.py#L621)) rasterizes each page at 200 DPI, runs DocLayout-YOLO, and saves crops of tables/figures/formulas to `code/assets/detected_tables_figures/`.

### Step 2 — Group & transform tables
- `group_table_detections` merges **consecutive page** table detections (for multi-page tables).
- `transform_table_to_html` ([pdf_graphrag.py:699](code/pdf_graphrag.py#L699)) feeds the cropped images to Gemini with a strict prompt that:
  - Produces a single merged `<table>` per group,
  - Preserves Slovak text exactly,
  - Uses `colspan`/`rowspan` for merged cells.
- `transform_html_to_graph_document` ([pdf_graphrag.py:772](code/pdf_graphrag.py#L772)) asks Gemini to interpret the HTML as a **hierarchical tree** (root → sections → items), splitting comma-separated cells into separate nodes, stripping diacritics, and producing `TableGraphResponse` (nodes + relationships).
- Interior table pages (i.e. pages fully inside a multi-page table) are **excluded** from subsequent text processing to avoid duplicate extraction. First/last pages are kept because they may contain surrounding prose.

### Step 3 — Open-Domain Detection (ODD)
- Text is chunked with `RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)`.
- `async_open_domain_detection` ([pdf_graphrag.py:967](code/pdf_graphrag.py#L967)) runs `open_domain_detection` concurrently (semaphore limit 5) with retry + 60 s global pause on failures.
- Each call uses a LangChain **agent** with `ProviderStrategy(response_schema_for_odd)` against `openai_client`. The response gives candidate `node_types` and `relationship_types` per chunk.
- Results are aggregated into one union `Schema`.

### Step 4 — Schema refinement
- `schema_refinement` ([pdf_graphrag.py:1018](code/pdf_graphrag.py#L1018)) sends the raw merged schema + the existing graph schema to Gemini, which:
  - Deduplicates (including diacritic variants — e.g. "Dan" vs "Daň"),
  - Adds `Paragraf` as a mandatory node type,
  - Produces a canonical list plus a `merge_log`.
- Output persisted via `refinement_to_json` in [to_json.py](code/to_json.py).

### Step 5 — Schema-Driven Extraction (SDE)
- Text is re-chunked at `chunk_size=1024, chunk_overlap=128` (smaller for extraction precision).
- `async_schema_driven_extraction` runs `schema_driven_extraction` per chunk against `openai_client` with `response_schema_for_sde`.
- `_convert_to_graph_document` ([pdf_graphrag.py:412](code/pdf_graphrag.py#L412)) turns the LLM JSON into a `GraphDocument`:
  - Creates a `Chunk` node (id=`chunk_i`, stores text + page),
  - Validates/normalizes each node id, type, and properties,
  - Matches relationships case-insensitively against extracted nodes,
  - Links every entity to its chunk via `IN_CHUNK` — this edge is what later enables chunk retrieval from graph traversal results.
- `_filter_by_strict_mode` ([pdf_graphrag.py:557](code/pdf_graphrag.py#L557)) is available as a post-filter against the refined schema (currently commented out — kept as an optional safety net).

### Step 6 — Persist to Neo4j
- A synthetic `Document` node plus `IN_DOCUMENT` edges to each `Chunk` are added via `_add_document_chunk`.
- Table graph documents are appended.
- `graph.add_graph_documents(..., baseEntityLabel=True)` writes everything; then the `__Entity__` label is removed from `Chunk` and `Document` nodes so they don't pollute the entity vector index.

### Step 7 — Vector stores
Three `Neo4jVector` indices are created/updated from the live graph:
- `nodes_vector_store` — embeds `__Entity__.id`.
- `chunk_vector_store` — embeds `Chunk.text`.
- `relationships_vector_store` — embeds the set of distinct relationship type names (built from `CALL db.relationshipTypes()`).

---

## 4. Query Pipeline — `query(question)`

`query` ([pdf_graphrag.py:1859](code/pdf_graphrag.py#L1859)) implements a 3-stage KG-GPT pipeline.

### Stage 1 — `segment_question`
An OpenAI agent splits the question into `SubSentence` objects, each supposed to map to **one KG triple**, with up to 2 anchor entities (Title Case). If segmentation fails, the whole question is used as one sub-sentence with no anchors.

### Stage 2 — ARK-V1 state-machine retrieval
For each sub-sentence, `ark_v1_retrieve` ([pdf_graphrag.py:1700](code/pdf_graphrag.py#L1700)) performs up to `K_MAX = 3` hops per anchor (first 2 anchors only):

1. `_resolve_anchor` — exact-match then `CONTAINS` fallback on `n.id` (case-insensitive).
2. `_retrieve_relations` — enumerates actual in/out relationship types for the anchor (this gives the candidate set R^k).
3. `_select_relation` — Gemini Flash picks exactly one relation from R^k (validated; up to `C_MAX = 2` retries with error feedback). Returns `None` to stop.
4. `_retrieve_triples` — executes a parameterized Cypher with the selected relation (limit 25 triples). The relation name is interpolated via f-string, which is safe because it came from the graph itself.
5. `_reason_step` — Gemini Flash picks relevant triple indices, writes a Slovak implication summary, decides whether to continue, and optionally proposes a `next_anchor` (which MUST be a tail of a selected triple — prevents hallucinated hops).

The running `summary` string is rebuilt each step so context stays roughly constant in k. Deduplicated triples and collected node ids are returned.

A legacy alternative, `search_agent_retrieve` ([pdf_graphrag.py:1450](code/pdf_graphrag.py#L1450)), exposes a `search_database` tool to a LangChain agent that authors its own Cypher — not used by default in favor of ARK-V1's stricter state machine.

### Stage 3 — `answer`
`get_chunks_from_nodes` ([pdf_graphrag.py:1779](code/pdf_graphrag.py#L1779)) follows `IN_CHUNK` from every retrieved node to pull source `Chunk.text` values. Then `answer` ([pdf_graphrag.py:1806](code/pdf_graphrag.py#L1806)) asks Gemini Flash (structured `InferenceAnswer`) to synthesize a Slovak natural-language answer using:
- Triples as **primary** evidence (serialized as `[subject, relation, object]` lists),
- Chunks as supplementary context.

---

## 5. Data Types

Defined in [code/classes.py](code/classes.py) (referenced throughout):
- `Schema` — `{nodes, relationships}`.
- `ClassifiedDocument` — output of the (currently unused) `classification` stage.
- `SubSentence` — `{text, entities}` from Stage 1.
- `ReasoningStep` — `{selected_triple_indices, implication, continue_reasoning, next_anchor}` from ARK-V1 Stage 2.

Prompts and response schemas live in [code/prompts.py](code/prompts.py).

---

## 6. Design Notes

- **Two-phase schema building** (ODD → refinement) produces a stable type vocabulary **before** extraction, so SDE across many chunks stays consistent.
- **ARK-V1 over LLM-authored Cypher**: relation choices are always validated against the graph's actual schema, and next-hop anchors must exist in the selected triples. This eliminates the most common failure mode of agentic Cypher (fabricated labels/relations).
- **`IN_CHUNK` edges** are the bridge between graph retrieval and textual grounding — without them, Stage 3 would have triples but no original text.
- **Table handling is separate from text**: vision LLM → HTML → hierarchical graph. Interior pages of multi-page tables are removed from the text stream to prevent double-counting.
- **Read-only guard**: every non-parametrized Cypher on the query path passes through `is_read_only_cypher`, rejecting writes and `CALL`s.
- **Resilience**: async extraction stages use a semaphore + global pause-on-error event, so a rate-limit hit pauses all concurrent tasks for 60 s rather than cascading failures.
