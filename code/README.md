# PDF GraphRAG — znalostný graf z PDF dokumentov

Pipeline na extrakciu **znalostného grafu** zo slovenských právnych a finančných PDF dokumentov. Zo zákona, vyhlášky alebo inej legislatívy postupne vytvorí:

1. **Štruktúrované uzly** pre paragrafy / odseky / písmená / body (§ → ods. → písm. → bod)
2. **Tabuľky a vzorce** rozpoznané cez YOLO + LLM (HTML → graf, LaTeX)
3. **Otvorenú schému** (Open-Domain Detection) z textu chunkov
4. **Spresnenú schému** (deduplikácia, normalizácia, ASCII)
5. **Entity a vzťahy** podľa schémy (Schema-Driven Extraction)
6. **Vektorové úložiská** pre Chunk-y, uzly a typy vzťahov v Neo4j

Výsledok je perzistovaný do **Neo4j** ako graf s vektorovými indexami pre RAG dotazovanie.

Pri používaní kódu, musí byť vždy zapnutá inštancia databázy v Neo4j, kvôli konštruktoru triedy.

---

## Obsah

- [Inštalácia](#inštalácia)
  - [1. Neo4j Desktop 2 + APOC](#1-neo4j-desktop-2--apoc)
  - [2. Python závislosti](#2-python-závislosti)
  - [3. API kľúče](#3-api-kľúče)
- [Rýchly štart](#rýchly-štart)
- [Dokumentácia súborov](#dokumentácia-súborov)
  - [main.py](#mainpy)
  - [pdf_graphrag.py](#pdf_graphragpy)
  - [chunker/chunker.py](#chunkerchunkerpy)
  - [loaders.py](#loaderspy)
  - [to_json.py](#to_jsonpy)
  - [prompts.py](#promptspy)

---

## Inštalácia

### 1. Neo4j Desktop 2 + APOC

Pipeline ukladá graf cez `langchain-neo4j` a používa **APOC procedúry** (`apoc.merge.node`, `apoc.merge.relationship`) v metóde [`PDFGraphRAG.add_graph_to_database`](pdf_graphrag.py). Bez APOC bude fungovať iba fallback [`add_graph_docs_without_apoc`](pdf_graphrag.py).

**Kroky:**

1. Stiahni **Neo4j Desktop 2** zo stránky <https://neo4j.com/download/>
2. Vytvor novú lokálnu inštanciu (DBMS) — odporúčaná verzia **Neo4j 5.x**
3. V detaile inštancie otvor **Plugins** a nainštaluj **APOC** (APOC Core)
4. Spusti inštanciu — predvolený bolt port je `7687`, predvolený user je `neo4j`
5. Heslo a názov databázy nastav v [main.py](main.py) v `get_graphrag()`:

```python
return PDFGraphRAG(
    neo4j_uri='neo4j://127.0.0.1:7687',
    neo4j_user='neo4j',
    neo4j_password='tvoje-heslo',
    database='nazov-databazy',          # napr. "neo4j" alebo vlastný
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
```

> **Poznámka:** ak inštancia beží v Docker / Aura, použi príslušné `neo4j://` URI a uprav `neo4j_user`.

### 2. Python závislosti

Vyžaduje sa **Python 3.11+** (kód používa `str | None` syntax a `list[Document]` generics).

#### Vytvorenie virtuálneho prostredia + inštalácia

```bash
# 1. Vytvor a aktivuj virtuálne prostredie
python -m venv .venv
source .venv/bin/activate                 # macOS/Linux
# .venv\Scripts\activate                  # Windows

# 2. Aktualizuj pip (odporúčané)
pip install --upgrade pip

# 3. Nainštaluj všetky závislosti z requirements.txt
pip install -r requirements.txt
```

#### Čo sa nainštaluje

Súbor [requirements.txt](requirements.txt) je rozdelený do nasledovných kategórií:

| Kategória | Balíky | Účel |
|---|---|---|
| **LangChain ekosystém** | `langchain`, `langchain-core`, `langchain-community`, `langchain-text-splitters`, `langchain-experimental`, `langchain-neo4j`, `langchain-openai`, `langchain-google-genai` | Orchestrácia pipeline-u, splittery, Neo4j integrácia, LLM wrappery |
| **LLM provider SDK** | `openai`, `anthropic`, `google-generativeai` | Priame SDK pre GPT-5/4-mini, Claude (voliteľné), Gemini 2.5 |
| **PDF spracovanie** | `pypdf`, `pdf2image`, `Pillow` | Načítanie PDF + konverzia stránok na obrázky pre YOLO |
| **Detekcia layoutu** | `doclayout-yolo`, `huggingface-hub`, `numpy` | YOLO model (DocStructBench) pre tabuľky/vzorce + sťahovanie váh z HF |
| **Validácia / typovanie** | `pydantic` | Schémy pre štruktúrované LLM výstupy (ODD, SDE) |
| **Konfigurácia** | `python-dotenv` | Načítanie `.env` s API kľúčmi |

#### Systémové závislosti (mimo `pip`)

- **Poppler** — vyžadovaný `pdf2image` pre konverziu PDF stránok na obrázky (detekcia tabuliek/vzorcov)
  - macOS: `brew install poppler`
  - Ubuntu: `sudo apt install poppler-utils`
  - Windows: stiahni z <https://github.com/oschwartz10612/poppler-windows/releases> a pridaj do PATH

> **Tip:** Pri prvom spustení sa stiahne YOLO model (~50 MB) z HuggingFace do lokálneho cache (`~/.cache/huggingface/`). Toto je jednorazové.

### 3. API kľúče

Pipeline volá **OpenAI** (GPT-5/4-mini, embeddings `text-embedding-3-large`) aj **Google Gemini** (Gemini 2.5 Pro/Flash). Vytvor súbor `.env` v koreni projektu:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
```

Kľúče sa nahrávajú cez `python-dotenv` priamo v [main.py](main.py).

| Kľúč | Použitie |
|---|---|
| `OPENAI_API_KEY` | ODD, SDE, refinement, formulas (`gpt-5-mini`, `gpt-4o-mini`, `text-embedding-3-large`) |
| `GOOGLE_API_KEY` | Záložné/experimentálne extrakcie cez Gemini 2.5 Pro/Flash |

---

## Rýchly štart

```python
from main import get_graphrag, build_sde_chunks, run_odd, run_refinement, run_sde

graphrag = get_graphrag()
documents = graphrag.load_pdf("assets/ZZ_2004_222.pdf")
document_id = graphrag.get_document_id("assets/ZZ_2004_222.pdf")

# 1. ODD: discover schema (LLM call ~5-15 min)
odd_schema = run_odd(graphrag, documents, name="zakon222", write_json=True)

# 2. Refinement (LLM call ~30s)
refined = run_refinement(graphrag, odd_schema, name="zakon222", write_json=True)

# 3. SDE: extract entities + relationships per chunk
sde_chunks, tree_graph = build_sde_chunks("assets/ZZ_2004_222.pdf", documents)
graphrag.add_graph_to_database(tree_graph)
graph_docs = run_sde(graphrag, sde_chunks, refined, document_id, name="zakon222", write_json=True)
graphrag.add_graph_to_database(graph_docs)
```

Alebo jednorazovo cez plný pipeline:

```python
graphrag.process("assets/ZZ_2004_222.pdf", name_of_chain="zakon222", write_json=True)
```

---

## Dokumentácia súborov

## main.py

Vstupný bod, ktorý rozdeľuje pipeline na **samostatne volateľné stupne**. Každý stupeň má voliteľný `cache_path` — ak existuje, načíta sa JSON z disku namiesto LLM volania. Toto umožňuje pokračovať v rozbitom behu bez opakovaného plytvania kreditmi.

### Konštanty

| Konštanta | Význam |
|---|---|
| `PDF_PATH` | Cesta k PDF, ktoré sa spracuje |
| `DOCUMENT_ID` | ID dokumentu (predvolene odvodené zo súboru) |
| `NAME_OF_CHAIN` | Prefix pre všetky JSON výstupy (`<name>_odd_<ts>.json`, ...) |
| `WRITE_JSON` | Či sa majú medzivýsledky zapisovať do `./file_output/` |

### Funkcie

#### `get_graphrag() -> PDFGraphRAG`

Vytvorí inštanciu [`PDFGraphRAG`](pdf_graphrag.py) napojenú na lokálny Neo4j a načíta API kľúče z `.env`. Tu uprav credentials.

#### `get_structured_chunks(pdf_path, write_json) -> dict`

Volá [`Chunker.split_document`](chunker/chunker.py) a vráti:
```python
{
  "chunks": list[Document],      # plochý zoznam chunkov s `path` metadátami
  "tree_graph": GraphDocument,   # Document → Paragraf → Odsek → Pismeno → Bod
  "last_page": int | None,       # 1-indexovaná stránka posledného §
}
```
Otestovaný a funkčný **iba pre slovenské finančné zákony** (paragraf § 44 → odsek (3) → písm. a) → bod 2.). Nezvláda novely a doplnenia, kvôli výskytu doplňujúcich bodov. V texte sú označené ako 2., 3., 4. a odhaľovanie len za pomoci stringov a \n nie je dostačujúce, lebo aj zmienky dátum a čisiel z textu sú takto odsadené. Nevyhnutný OCR model na transformáciu a identifikáciu textu na štruktúrovaný text z obrázku.

#### `get_text_chunks(documents, chunk_size=1024, chunk_overlap=128) -> list[Document]`

Čisto-textové chunkovanie cez `RecursiveCharacterTextSplitter`. Používa sa pre **ODD**, kde nepotrebujeme zachovať právnu hierarchiu.

#### `build_sde_chunks(pdf_path, documents, write_json) -> tuple[list[Document], GraphDocument]`

Vytvorí **presne tú istú sadu chunkov**, akú používa `PDFGraphRAG.process()`:

- štruktúrované chunky cez `Chunker` (s `path` metadátami pre prompt)
- trailing chunky pre stránky **za posledným §** (prílohy) rozdelené `RecursiveCharacterTextSplitter` po riadkoch (`separators=["\n"]`)

Vracia `(chunks, tree_graph)`. **Dôležité:** `tree_graph` musí byť ingestovaný do Neo4j **pred** SDE výstupom, lebo SDE vzťahy odkazujú na uzly `Paragraf`/`Odsek`/`Pismeno` podľa ID.

#### `filter_table_pages(documents, table_pages_to_exclude) -> list[Document]`

Vyhodí stránky obsahujúce **vnútro tabuliek** z `PyPDFLoader` dokumentov, aby ODD/SDE nepracovali znova s obsahom, ktorý už máme zachytený ako tabuľky. PyPDFLoader používa 0-indexovanie, detekcie 1-indexovanie — funkcia s tým ráta.

#### `run_odd(graphrag, documents, name, write_json, cache_path=None) -> Schema`

**1. stupeň: Open-Domain Detection** — pre každý chunk LLM vyextrahuje kandidátne typy uzlov a vzťahov. Výsledné per-chunk schémy sa zjednotia (union).

- `cache_path` = cesta k existujúcemu `<name>_odd_<ts>.json` → preskočí LLM
- inak chunkuje cez `get_text_chunks(..., 1200, 200)` a volá `async_open_domain_detection`

#### `run_refinement(graphrag, odd_schema, name, write_json, cache_path=None) -> Schema`

**2. stupeň: Schema Refinement** — LLM kanonicizuje (PascalCase, UPPER_SNAKE_CASE, bez diakritiky), deduplikuje synonymá a merguje so schémou už existujúcou v Neo4j (`get_graph_schema()`).

- `cache_path` = `<name>_ref_<ts>.json`

#### `run_sde(graphrag, chunks, schema, document_id, name, write_json, cache_path=None) -> list[GraphDocument]`

**3. stupeň: Schema-Driven Extraction** — pre každý chunk LLM vyextrahuje konkrétne entity a vzťahy **iba** v rámci dodaných typov zo `schema`.

- `chunks` musí byť už predchunkované! Použi `build_sde_chunks()` (zachová `path` metadáta) alebo `get_text_chunks()` (stratíš právny kontext v prompte)
- `cache_path` = `<name>_sde_<ts>.json`

#### `run_sde_sample(graphrag, chunks, schema, sample_size=30, index_range, seed=42, ...) -> list[int]`

Sample runner pre **iteráciu promptov / schémy** — vyberie `sample_size` náhodných chunkov z rozsahu `index_range` a každý zapíše do samostatného JSONu pod `output_dir`. `seed` zabezpečuje reprodukovateľnosť — rovnaký seed + range = rovnaké indexy, takže porovnávaš jablká s jablkami.

Vracia zoznam spracovaných indexov.

### Príklady použitia loaderov v `main.py`

```python
from loaders import load_odd, load_refinement, load_sde

# Resume z existujúcej cache
odd = load_odd("./file_output/zakon222_odd_20260520123045.json")
refined = run_refinement(graphrag, odd, name="zakon222")

# Alebo preskoč obe LLM fázy a rovno do SDE:
refined = load_refinement("./file_output/zakon222_ref_20260520130000.json")
sde_chunks, tree = build_sde_chunks(PDF_PATH, documents)
graph_docs = run_sde(graphrag, sde_chunks, refined, document_id,
                    cache_path="./file_output/zakon222_sde_20260520140000.json")
```

V `main()` (riadky 262–357) je kompletný komentovaný príklad celej pipeline rozdelenej na stupne.

---

## pdf_graphrag.py

Hlavná trieda **`PDFGraphRAG`**, ktorá obsahuje všetky stupne pipeline. Implementuje:

- Pripojenie na Neo4j + 3 vektorové úložiská (uzly, chunky, typy vzťahov)
- LLM klientov: OpenAI (`gpt-5.4-mini`, `gpt-4o-mini`, `gpt-5` thinking), Google (`gemini-2.5-pro`, `gemini-2.5-flash`)
- Pomocné formátovacie funkcie (camelCase, ASCII fold, sanitizácia)
- YOLO detekciu tabuliek/vzorcov + LLM extrakciu HTML/LaTeX
- Async retry/pause logiku pre rate limit-y

### Konštruktor

```python
PDFGraphRAG(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    openai_api_key: str | None = None,
    google_api_key: str | None = None,
    database: str | None = None,
)
```

Inicializuje:
- `self.graph` — `Neo4jGraph` s `refresh_schema=True`
- `self.embeddings` — `OpenAIEmbeddings('text-embedding-3-large')`
- `self.vector_store_relationships` — lazy load existujúceho indexu (None ak neexistuje)
- LLM klienti: `openai_client`, `openai_graph_transform`, `openai_thinking`, `gemini_client`, `gemini_client_thinking`, `gemini_client_flash`

### Top-level helpers (mimo triedy)

| Funkcia | Účel |
|---|---|
| `is_read_only_cypher(query)` | Sanity-check, či Cypher dotaz neobsahuje write keywords (CREATE/MERGE/DELETE/...). Vyhodí `ValueError`. |
| `format_property_key(s)` | camelCase + odstránenie diakritiky pre názvy properties |
| `strip_diacritics(s)` | NFKD fold + ASCII filter |
| `format_node_type(t)` | PascalCase + diacritics strip, fallback na `Entity` |
| `format_relationship_type(t)` | UPPER_SNAKE_CASE + diacritics strip |
| `sanitize_property_keys(props)` | Aplikuje `format_property_key` na všetky kľúče |
| `drop_empty_values(props)` | Vyhodí None/prázdne kľúče |
| `graph_document_to_json(gd)`, `serialize_for_json(obj)` | Lossy JSON serializácia pre debug |

### Public metódy triedy

#### `load_pdf(pdf_path: str) -> list[Document]`

Wrapper okolo `PyPDFLoader`. Vracia jeden `Document` na stránku PDF (s `metadata["page"]` 0-indexovaným).

#### `get_document_id(pdf_path: str) -> str`

Vracia `Path(pdf_path).stem` — používa sa ako ID pre `Document` uzol v grafe.

#### `get_graph_schema() -> Schema`

Načíta existujúce node labels a relationship types z Neo4j (`CALL db.labels()`, `CALL db.relationshipTypes()`). Pri zlyhaní vracia prázdnu `Schema(nodes=[], relationships=[])` — používa sa ako vstup do refinement.

#### `get_sample_graph_schema() -> str`

Vracia ľudský sumár schémy s 5 sample uzlami a 5 sample vzťahmi (LIMIT 10 v Cypher).

#### `query_graph(query: str)`

Surový Cypher dotaz. Vracia zoznam rowov.

#### `add_graph_to_database(graph_documents)`

Wrapper okolo `Neo4jGraph.add_graph_documents` s `include_source=False, baseEntityLabel=False`. Vyžaduje APOC.

#### `merge_new(graph_documents)`

**Idempotentný ingest.** LangChain default volá `apoc.merge.node` s prázdnym `onMatchProps`, takže pri re-rune si stará verzia ponechá staré properties. Táto metóda predáva `row.properties` ako **`onCreate` aj `onMatch`** → graf konverguje k poslednej extrakcii. Použi pri opätovných behoch nad rovnakými chunkmi.

#### `add_graph_docs_without_apoc(graph_docs)`

Fallback bez APOC — čisté `MERGE (n:{type} {id: $id}) SET n += $properties`. Pomalšie, ale funguje na vanilla Neo4j.

#### `async_open_domain_detection(documents, max_concurrent=5, write_json, name) -> list[Schema]`

**ODD stupeň.** Pre každý chunk paralelne (default 5 súbežných) zavolá `open_domain_detection(i, doc)`, ktorý cez `create_agent` + `ProviderStrategy` vyplní `response_schema_for_odd`.

- **Retry logika:** chyba na chunk-u → `pause_event.clear()` + `sleep(60)` + 3 pokusy. Po 3 neúspechoch sa chunk dropuje (nezhodí celý stupeň).
- **Inkrementálne zapisovanie:** každý úspešný chunk re-dumpuje rastúci zoznam do toho istého `<name>_odd_<ts>.json` (pod `write_lock`), takže pád v polovici nezahodí dokončené chunky.

#### `schema_refinement(odd_schema, existing_schema=None) -> dict`

**Refinement stupeň** (synchronný). Volá `openai_thinking` s `with_structured_output(SchemaRefinementResponse, method="json_schema", strict=True)`. Vracia:

```python
{
  "node_types": list[str],
  "relationship_types": list[str],
  "merge_log": {
    "node_types": dict[str, list[str]],         # canonical -> [varianty]
    "relationship_types": dict[str, list[str]],
  }
}
```

Ak `existing_schema` je `None` alebo prázdna, použije sa **predpečená slovenská právna ontológia** (riadok 1943). Počiatočná schéma, používateľ môže upraviť podľa svojej potreby. Zapísané sú konštantné a generické typy, ktoré sa vyskytujú najčastejšie. 3 pokusy s `sleep(60)`, potom `RuntimeError`.

> **Vstupom** je vždy `Schema` objekt. **Výstupom** je `dict` — pre konverziu späť do `Schema` použi `_convert_to_schema(data)` alebo (v `main.py`) konštruuj `Schema(nodes=data['node_types'], relationships=data['relationship_types'])`.

#### `async_schema_driven_extraction(documents, schema, document_id, max_concurrent=5, write_json, name) -> list[GraphDocument]`

**SDE stupeň.** Identická retry/pause schéma ako ODD + **2-pass retry**: po prvom kole sa zlyhané chunky skúsia ešte raz. Per-chunk volanie ide cez `openai_client.with_structured_output(response_schema_for_sde, method="json_schema", strict=True)`.

Prompt automaticky **injekuje právny kontext** zo `document.metadata["path"]`:
```
Paragraf § 16a Odsek 2 Pismeno c)
```
Toto funguje iba ak chunky vyrobil `Chunker` (cez `build_sde_chunks`). Trailing chunky tieto metadáta nemajú, ale prompt nezhodia.

#### `tables_and_formulas(pdf_path, document_id, documents, write_json) -> tuple`

Vracia `(table_graph_docs, formula_graph_doc, table_pages_to_exclude)`.

**Tabuľky:**
1. `detect_tables()` — YOLO (`DocLayout-YOLO-DocStructBench`) rozpozná tabuľky/figure/formuly na 200 DPI obrázkoch stránok. Cropy uloží do `./file_output/tables_and_formulas/`.
2. `group_table_detections()` — zoskupí po sebe idúce stránky s tabuľkou.
3. `_split_group_by_headlines()` — rozdelí skupinu, ak nová stránka začína veľkým nadpisom (signál novej tabuľky).
4. `transform_table_to_html()` — Gemini/OpenAI prevedie obrázky na **jednu mergovanú HTML tabuľku** (chunked po 2 obrázkoch s previous-HTML continuation).
5. `transform_html_to_graph_document()` — LLM prevedie HTML na `(:Tabulka)-[:MA_RIADOK]->(:Riadok)` graf.

**Vzorce:**
1. `get_formulas()` — zoznam PNG-iek z `tables_and_formulas/`.
2. `transform_formula()` — Gemini Flash -> LaTeX + confidence.
3. `_convert_formula_to_node()` + `convert_formulas_to_graph()` — `(:Vzorec)-[:IN_DOCUMENT]->(:Document)`.

`table_pages_to_exclude` je set 1-indexovaných **vnútorných** stránok tabuliek (prvá a posledná stránka tabuľky sa neradia — môžu obsahovať aj iný text), ktoré následne odfiltruje `filter_table_pages` pred ODD/SDE.

#### `process(pdf_path, name_of_chain="chain", write_json=False)`

**Plný end-to-end pipeline** v jednom volaní. Kroky:

1. `load_pdf()` + `get_document_id()`
2. `tables_and_formulas()` → ingest do Neo4j, odfiltrovanie stránok
3. ODD na text-only chunkoch (`RecursiveCharacterTextSplitter` 1200/200)
4. Union ODD schém → `schema_refinement` (uses `get_graph_schema()`)
5. `Chunker().split_document()` + trailing chunker → SDE chunky
6. Ingest `tree_graph` (Document → Paragraf → Odsek → Pismeno → Bod)
7. `async_schema_driven_extraction()` + append `_add_document_chunk()` → ingest
8. `refresh_schema()`
9. Vytvor / aktualizuj 3 vektorové indexy: `nodes_vector_store` (label `__Entity__`, embed `id`), `chunk_vector_store` (label `Chunk`, embed `text`), `relationships_vector_store` (typy vzťahov ako `Document` objekty).

Pri zlyhaní akéhokoľvek stupňa pipeline padne s `RuntimeError`/`ValueError` — ale per-chunk zlyhania v ODD/SDE sú tolerované.

### Private/internal metódy

| Metóda | Účel |
|---|---|
| `_init_vector_stores()` | Lazy load `relationships_vector_store`; ostatné sa robia až v `process()`. |
| `_convert_to_graph_document(data, i, doc, doc_id, section_id, section_label)` | LLM response → `GraphDocument`. Sanitizuje properties, vyhodí uzly bez ID, normalizuje IDs (ASCII + Title Case), pripojí každý uzol k chunku cez `IN_CHUNK`. Ak je dodaný `section_id` (z `path`), použije sa namiesto generického chunk uzla. |
| `_convert_table_to_graph_document(data, i, doc, doc_id)` | Variant pre tabuľky — pridáva `html`, `page_range`, `document_id` properties pre `Tabulka` uzol. |
| `_add_document_chunk(count, path, doc_id, props=None)` | Vyrobí `Document` + `Chunk_0..N` skeleton s `IN_DOCUMENT` vzťahmi. |
| `_convert_to_schema(data)` | Validátor — `dict` → `Schema`. Vyhodí `ValueError` pri zlej štruktúre. |
| `_filter_by_strict_mode(graph_doc, allowed_entities, allowed_relationships)` | Post-extraction safety net (aktuálne nevolaný — zakomentované v SDE). |
| `_split_group_by_headlines(group, page_text_map)` | Rozdelí detekčnú skupinu na podskupiny podľa nadpisov. |
| `_extract_page_headline(page_text, max_lines=15)` | Hľadá ALL-CAPS nadpis v top N riadkoch stránky. |
| `group_table_detections(detections)` | Static — zoskupí po sebe idúce stránky s tabuľkou. |

---

## chunker/chunker.py

Modul ktorý **rozseká slovenský právny PDF** na semanticky úplné textové chunky a vybuduje stromový graf právnej hierarchie.

Hierarchia: **§ paragraf → odsek → písm. → bod**.

Overený a otestovaný na zákonoch finančného práva, ktoré následujú danú hierarchiu **§ 44 → (2) → a) → 2.**, kde pri novélach a neskorších doplneniach tento systém nemusí fungovať. Výskyt doplnkoch ako je **2., 3., 4.,..**, identifikovanie len na základe reťazcov a nových riadkov \n nie je možné. Prekáža tomu výskyt dátumov a čísiel, ktoré následujú podobný vzor.

### Konštanty

- `_PAGE_HEADER_RE` — regex na bežiace hlavičky typu `"Strana 12 Zbierka zákonov SR 222/2004 Z.z."`. Strip-uje ich pred parsovaním aby nešumili medzi paragrafmi.
- `_SPLITTER` — `RecursiveCharacterTextSplitter(separators=["\n"], chunk_size=3000, chunk_overlap=300)` na rozdelenie príliš dlhých listových uzlov.

### Trieda `Chunker`

#### `load_pdf_text(pdf_path) -> tuple[str, list[tuple[int, int, int]]]`

Načíta PDF cez `PyPDFLoader`, vyhodí header regex, vráti:
- `text` — celé PDF ako jeden string (stránky spojené `"\n"`)
- `page_offsets` — zoznam `(page_num_1indexed, start_inclusive, end_exclusive)` na mapovanie char offsetu späť na stránku PDF

Spans sú **contiguous** (každá stránka pohltí svoj trailing `"\n"`), aby offset pristávajúci priamo na separátore nepadol medzi stránky.

#### `split_document(pdf_path, write_json=False) -> dict`

**Hlavný entry point.** Vracia:

```python
{
  "chunks": list[Document],          # plochý zoznam chunkov
  "tree_graph": GraphDocument,       # Document -> Paragraf -> Odsek -> Pismeno -> Bod
  "last_page": int | None,           # 1-indexovaná stránka posledného §
}
```

Volá:
1. `load_pdf_text(pdf_path)`
2. `detect_subsections(text, page_offsets, ...)` (modul `chunker.detect_subsections`) — regex-based parser, ktorý vyrobí nested zoznam dictov s kľúčmi `marker`, `headline`, `page_start`, `page_end`, `text`, `odseky`, `letters`, `bode`, `lead`.
3. `get_tree_graph(paragraphs, pdf_path)` → `GraphDocument`
4. `linearize(paragraphs)` → `list[Document]`

#### `get_tree_graph(paragraphs, pdf_path) -> GraphDocument`

Prechádza cez 4 levely stromu DFS (Depth-First Search) a emituje uzly s **human-readable ID**:

| Úroveň | Príklad ID | Vzťah |
|---|---|---|
| Paragraf | `Paragraf § 16a` | `IN_DOCUMENT` |
| Odsek | `Paragraf § 16a Odsek 2` | `IN_SECTION` |
| Pismeno | `Paragraf § 16a Odsek 2 Pismeno c)` | `IN_SECTION` |
| Bod | `Paragraf § 16a Odsek 2 Pismeno c) Bod 3` | `IN_SECTION` |

Properties uzla obsahujú `path` (per-úrovňové tokeny) + `text` + extras (`headline`, `page_start`, `page_end`). None properties sú drop-nuté.

**Dôležité:** ID musia byť **konzistentné** s tým, čo generuje SDE prompt v `pdf_graphrag.py` (formát `Paragraf § X Odsek Y Pismeno z)`). Inak SDE vzťahy nenájdu cieľové uzly.

#### `linearize(paragraphs, write_json=False) -> list[Document]`

Prejde strom DFS-om — každý chunk emituje na **najhlbšej dostupnej úrovni** svojej cesty. Parent `lead`-y sa **forward-concatuje** aby chunk znel ako samostatná próza:

```
"§ 16a [Headline]
 [text §]
 (2) [lead odseku 2]
 c) [lead písmena c]
 1. [lead bodu 1]
 2. [lead bodu 2]
 ..."
```

Po vygenerovaní všetkých chunkov priraďuje sekvenčný `id` (0, 1, 2...) do `metadata`. Toto poradie je **stabilné** — slúži ako index pre `run_sde_sample`.

Ak `write_json=True`, volá `chunker_to_json(chunks)` z [to_json.py](to_json.py).

#### `_walk_paragraph(para, out)`

DFS implementácia `linearize`. Path tokeny:
- Paragraf → `marker` (`"§ 16a"`)
- Odsek → `number` (`"1"`)
- Pismeno → `marker` (`"a)"`)
- Bod → `number` (`"3"`)

Pre Body sa neemituje samostatný chunk — ich `marker` + `lead` sa skonkatenujú do textu rodičovského písmena (`\n` separated). To šetrí počet chunkov ale zachová atomické čísla.

#### `_make_node`, `_make_doc`, `_join`, `_split_if_long`

Privátne helpery:
- `_join(*parts)` — strip + join space, vyhodí prázdne
- `_make_doc(text, path, headline)` — `Document` s `path` + `headline` metadátami
- `_make_node(...)` — `Node` s `path` + `text` properties (+ optional extras)
- `_split_if_long(doc)` — ak `len(page_content) > 3000`, rozdelí cez `_SPLITTER` na newlines

### Príklad samostatného spustenia

```bash
python -m chunker.chunker
# alebo
python chunker/chunker.py
```

Spustí pipeline na `assets/ZZ_2004_222_20260101.pdf` a vypíše počet chunkov + prvých 5.

---

## loaders.py

**Reverzia `to_json.py`.** Každý loader prečíta JSON vyprodukovaný príslušným `*_to_json` writerom a zrekonštruuje in-memory objekt(y).

| Writer | Loader | Vracia |
|---|---|---|
| `odd_to_json` | `load_odd(path)` | `Schema` |
| `refinement_to_json` | `load_refinement(path)` | `Schema` |
| `sde_to_json` | `load_sde(path)` | `list[GraphDocument]` |
| `table_to_json` | `load_tables(path)` | `list[GraphDocument]` |
| `formula_to_json` | `load_formula(path)` | `GraphDocument \| None` |

### `load_odd(path) -> Schema`

Preferuje `nested` blok (už deduplikovaný cez `set()`). Fallback: ak `nested` chýba, rekonštruuje union z `chunked`. Zhadzuje `ValueError` ak nie je ani jeden.

### `load_refinement(path) -> Schema`

Mapuje `node_types` → `Schema.nodes` a `relationship_types` → `Schema.relationships`. `merge_log` sa **zámerne dropuje** (`Schema` ho nemá ako pole).

### `load_sde(path) -> list[GraphDocument]`

⚠️ **Lossy roundtrip:** `sde_to_json` orezáva `source.page_content` na **200 znakov**, takže rekonštruovaný `Document` má iba prefix. To stačí pre ingest do Neo4j (text bol už ingestovaný cez `tree_graph` + chunk skeleton), ale nie pre re-použitie ako vstup do ďalšieho LLM volania.

### `load_tables(path) -> list[GraphDocument]`

Plný roundtrip — HTML `source_html` sa zachováva, `page_range` ide do metadát.

### `load_formula(path) -> GraphDocument | None`

Vracia `None` ak JSON reprezentuje prázdny payload (žiadne uzly, žiadne ID) — mirror `data is None` vetvy vo writeri.

### Použitie v `main.py`

```python
from loaders import load_odd, load_refinement, load_sde

# Preskoč ODD ak už beží JSON existuje
odd = load_odd("./file_output/zakon222_odd_20260520123045.json")

# Alebo cez built-in cache support:
odd = run_odd(graphrag, docs, cache_path="./file_output/zakon222_odd_20260520123045.json")
```

---

## to_json.py

JSON writery pre všetky stupne pipeline. Defaultný output: `./file_output/`.

| Funkcia | Output filename | Vstup |
|---|---|---|
| `odd_to_json(documents, output_dir, name, chunks, timestamp)` | `<name>_odd_<ts>.json` | `list[Schema]` + optional `chunks` |
| `refinement_to_json(data, output_dir, name)` | `<name>_ref_<ts>.json` | `dict` z `schema_refinement` |
| `sde_to_json(data, output_dir, name, timestamp)` | `<name>_sde_<ts>.json` | `list[GraphDocument]` |
| `table_to_json(data, output_dir, name, timestamp)` | `<name>_tables_graph_doc_<ts>.json` | `list[GraphDocument]` |
| `formula_to_json(data, output_dir, name, timestamp)` | `<name>_formula_graph_doc_<ts>.json` | `GraphDocument \| None` |
| `chunker_to_json(chunks, output_dir, timestamp)` | `chunks_<ts>.json` | `list[Document]` |
| `sections_to_json(paragraphs, output_dir, timestamp)` | `sections_<ts>.json` | nested dict z `detect_subsections` |

### Spoločné správanie

- **`_safe_dump`** — pri zlyhaní `json.dump` zapíše raw `repr(payload)` do `<path>.failed.txt` a **re-raises** pôvodnú exception. Caller sa dozvie, že perzistencia padla, ale dáta sú aspoň čiastočne na disku.
- **`_timestamp()`** — `YYYYMMDDHHMMSS`. Pre inkrementálne zapisovanie v async kóde sa rovnaký timestamp predáva ako parameter, aby všetky write-y v rámci stupňa landovali do toho istého súboru.
- **`ensure_ascii=False`** — zachováva slovenskú diakritiku v `chunker_to_json` a `sections_to_json` (užitočné pre debug).

### Štruktúra ODD JSON

```json
[
  {"chunked": [{"nodes": [...], "relationships": [...]}, ...]},
  {"chunk_texts": [{"text": "...", "page": 0}, ...]},
  {"nested": {"nodes": ["..."], "relationships": ["..."]}}
]
```

### Štruktúra SDE JSON

```json
[
  {
    "source": "<prvych 200 znakov chunku>",
    "nodes": [{"id": "...", "type": "...", "properties": {}}],
    "relationships": [{"source": {}, "target": {}, "type": "...", "properties": {}}]
  }
]
```

---

## prompts.py

System prompty + JSON schémy pre LLM volania. Všetko v slovenčine (okrem SDE, ktorý je v angličtine — Gemini/GPT lepšie nasleduje anglické inštrukcie nad slovenským textom - neviem prečo ale bolo menej halucinácii). Pre zmenu promptov, treba pozmeniť príslušný system prompt a taktiež user prompt v príslušnej funkcii v `pdf_graphrag.py`.

### `system_prompt_for_odd` + `response_schema_for_odd`

**Open-Domain Detection.** Volaný v `open_domain_detection()`. Inštruuje LLM:

- Extrahovať iba **TYPY**, nie inštancie
- PascalCase uzly, UPPER_SNAKE_CASE vzťahy
- Bez diakritiky
- Slovenčina
- Pokryť: právne akty, subjekty, finančné koncepty, abstrakcie (povinnosť, právo, lehota), mechanizmy (koeficient, sadzba), čas

Response: `{node_types: [str], relationship_types: [str]}`.

### `system_prompt_for_schema_refinement` + `response_schema_for_schema_refinement`

**Schema Refinement.** Volaný v `schema_refinement()`. Inštruuje LLM ako **expertný ontológ**:

- Deduplikovať synonymá → kanonický typ
- PascalCase / UPPER_SNAKE_CASE
- ASCII normalizácia
- **NIKDY** nezlučovať Aktér vs. Dokument, Miesto vs. Útvar, opačné vzťahy
- Generalizovať (`Sluha` → `Osoba`)
- Mapovať na základnú ontológiu ak existuje

Response: `{node_types, relationship_types, merge_log: {node_types, relationship_types}}` kde `merge_log` je canonical → list of variants.

### `system_prompt_for_sde` + `response_schema_for_sde`

**Schema-Driven Extraction.** Najprísnejší prompt — extrahuje konkrétne entity a vzťahy z chunku **iba** v rámci dodanej schémy.

Hlavné pravidlá:
- Iba **exact schema types**
- Title Case node IDs, SCREAMING_SNAKE_CASE relationships
- Bez diakritiky, slovenčina
- **Legal hierarchy v IDs:** `Paragraf § 16 Odsek 1 Pismeno a)`
- **Range decomposition:** `odsek 3 az 5` → `Odsek 3`, `Odsek 4`, `Odsek 5` + `ODKAZUJE_NA`
- **Inheritance:** `odsek` bez explicitného paragrafu → zdedí aktívny kontext
- **Atomic decomposition** — preferuj multi-hop pred broad direct relations
- `evidence` field — povinný **verbatim 5–15 word fragment** z textu na podporu vzťahu

Response: `{nodes: [{id, label}], relationships: [{source_node_id, source_node_type, relation, target_node_id, target_node_type, evidence}]}`. 

> **Pozn.:** `evidence` nie je zapísané v response schema pre SDE kvôli šetrnosti output API tokenov — treba doplniť v properties `response_schema_for_sde`.

V `pdf_graphrag.py` user-prompt (riadky 2103–2143) dynamicky injekuje `Section context (current section): Paragraf § 16a Odsek 2 Pismeno c)` zo `document.metadata["path"]` — tento kontext umožňuje LLM korektne interpretovať lokálne referencie ako "podľa odseku 1".

---

## Štruktúra projektu

```
code/
├── README.md                        ← tento súbor
├── requirements.txt
├── main.py                          ← entrypoint, samostatné stupne
├── pdf_graphrag.py                  ← hlavná trieda PDFGraphRAG
├── classes.py                       ← Schema, ClassifiedDocument, ...
├── prompts.py                       ← system prompty + schémy
├── loaders.py                       ← JSON -> in-memory
├── to_json.py                       ← in-memory -> JSON
├── chunker/
│   ├── chunker.py                   ← Slovak-law structural splitter
│   └── detect_subsections.py        ← regex parser (§/odsek/písm./bod)
├── assets/                          ← vstupné PDF
└── file_output/                     ← výstupy z `write_json=True`
```

---

## Tipy a gotchas

- **APOC chýba?** Použi `add_graph_docs_without_apoc()` namiesto `add_graph_to_database()`. Pomalšie, ale netreba plugin.
- **Tabuľky chýbajú na konci?** YOLO confidence threshold v `detect_tables(conf_threshold=0.3)` možno znížiť.
- **SDE chunky bez `path`?** Používaš čisto `RecursiveCharacterTextSplitter` namiesto `build_sde_chunks()` — stratíš legal context injekciu. Pre Slovak právne PDF vždy použi `Chunker`.
- **Re-run nad rovnakými chunkmi → staré properties?** Použi `merge_new()` namiesto `add_graph_to_database()` — predáva properties aj do `onMatch`.
- **Rate limit-y?** Async loop má built-in `pause_event` + `sleep(60)`. Pre väčšie PDF zníž `max_concurrent` z 5 na 2–3.
- **Cache miss na re-rune?** `cache_path` musí byť **presná** cesta. Pomôcka: timestamp je v názve, takže najnovší súbor je vždy lexikograficky najväčší.

---

## Kontakt

V prípade problémov alebo otázok ma neváhajte kontaktovať:

- **Email (FEI):** [xbagins@stuba.sk](mailto:xbagins@stuba.sk)
- **Email (osobný):** [samuel.bagin1@gmail.com](mailto:samuel.bagin1@gmail.com)
- **Web:** [samuelbagin.xyz](https://samuelbagin.xyz)

> Pre spolahlivosť a istotu systému odporúčam mať aspoň 20€ kreditu pre OpenAI API. Spracovanie 150 stranového dokumentu pomocou modelu GPT 5.4 mini s plne automatickou funkciou môže stáť v rozmedzí 11€ až 16€ (PDF ich môže mať minimálne 1600). Pri GPT 5.5 je táto cena len za 100 chunkov. Najdrahšie volanie je Schema Driven Extraction, kde sú extrahované konkrétne inštancie a z tohto dôvodu je výstup najdrahší (entity s id a typom, a to aj pre vzťajy, kde 2x entita plus vzťah).

> Odhadovaný čas pribehu celej funckie `process` je zväčša 20 až 25 minút.