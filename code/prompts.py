

# classification
system_prompt_for_classification = """
You are an expert at classifying text into predefined categories. """

response_schema_for_classification = {
    "title": "TextClassificationResult",
    "type": "object",
    "description": "Result schema for classifying text into predefined categories",
    "properties": {
        "type_legislation": {
            "type": "object",
            "description": "The category that best fits the provided text",
            "properties": {
                "name": {"type": "string", "description": "Name of the category"},
                "confidence": {"type": "number", "description": "A confidence score between 0 and 100 indicating the certainty of the classification"}
            },
        },
        "type_category": {
            "type": "object",
            "description": "The category that best fits the provided text",
            "properties": {
                "name": {"type": "string", "description": "Name of the category"},
                "confidence": {"type": "number", "description": "A confidence score between 0 and 100 indicating the certainty of the classification"}
            },
        }
    },
    "required": ["type_legislation", "type_category"]
}



# open domain detection
system_prompt_for_odd = """
Extrahuj TYPY ENTIT (uzlov) a TYPY VZTAHOV z pravneho alebo financneho textu.

# PRAVIDLA
- Extrahuj iba TYPY, nie instancie
- Pouzi vseobecne, ale rozlisitelne typy
- Bez diakritiky
- Uzly: PascalCase | Vztahy: UPPER_SNAKE_CASE
- Nehalucinuj, zahrn len to, co dava zmysel v pravnom kontexte
- Vyber spravny vyznam slova (napr. sluzba != sluha)
- Pouzivaj jedine Slovensky jazyk.

# COVERAGE
Zachyt VSETKO relevantne:
- pravne akty: Zakon, Paragraf, Odsek, Rozhodnutie
- subjekty: Osoba, PravnickaOsoba, Organ, SpravcaDane
- financne: Dan, Poplatok, Prijem, Zavazok
- abstrakcie: Povinnost, Pravo, Narok, Sankcia, Lehota
- mechanizmy: Koeficient, Vypocet, Sadzba
- cas: ZdanovacieObdobie, KalendarnyRok

# LOGIKA
- Zachyt aj implicitne koncepty
- Ak je pojem centralny (opakujuci sa), zahrn ho
- Minimalizuj duplicity
- Pouzi konzistentne nazvy

# VZTAHY
- Pouzi vseobecne vztahy (napr. UPRAVUJE, MA_POVINNOST, MA_PRAVO)
- Zachyt aj ciel alebo protistranu akcie
"""


response_schema_for_odd = {
            "title": "DocumentOpenDomainDetectionResult",
            "type": "object",
            "description": "Result schema for open domain detection of entities and relationships from text",
            "properties": {
                "node_types": {
                    "type": "array",
                    "description": "List of canonical node type labels",
                    "items": {
                        "type": "string",
                        "description": "The label/type of the entity (e.g., Person, Organization)"
                    }
                },
                "relationship_types": {
                    "type": "array",
                    "description": "List of canonical relationship types",
                    "items": {
                        "type": "string",
                        "description": "The type of relationship between nodes (e.g., WORKS_FOR, LOCATED_IN)"
                    }
                }
            },
            "required": ["node_types", "relationship_types"]
        }




# schema refinement
system_prompt_for_schema_refinement = """
# ROLE
Si expertný ontológ a dátový inžinier špecializujúci sa na znalostné grafy. Tvojou úlohou je vyčistiť, normalizovať a deduplikovať schému entít a vzťahov.

# CIELE
1. **Deduplikácia a sémantické zjednotenie**: Zlúč synonymá a sémanticky blízke typy do jedného kanonického typu.
2. **Normalizácia formátu**: 
    - Uzly: PascalCase (napr. DanovePriznanie).
    - Vzťahy: UPPER_SNAKE_CASE (napr. MA_POVINNOST).
    - **STRIKTNÉ PRAVIDLO**: Odstráň všetku diakritiku (á->a, č->c, atď.) zo všetkých názvov.
3. **Ontologická integrita (KRITICKÉ)**:
    - NIKDY nezlučuj entity z rôznych kategorií: Aktér (Osoba) vs. Proces/Dokument (Ziadost), Miesto (Kraj) vs. Politický útvar (Stat).
    - NIKDY nezlučuj protichodné vzťahy (VSTUPUJE_DO vs. VYSTUPUJE_Z).
4. **Generalizácia**: Príliš špecifické uzly nahraď všeobecnejšími (napr. "Sluha" -> "Osoba"), ak to neznižuje zrozumiteľnosť právneho/biznisového kontextu.

# PRAVIDLÁ PRE ZÁKLADNÚ ONTOLÓGIU
- Ak je poskytnutá Základná ontológia, má prednosť v pomenovaní.
- Ak surový typ (Open Domain) zodpovedá typu v základnej ontológii, mapuj ho naň.
- Ak je surový typ unikátny a dôležitý, pridaj ho ako nový typ.
"""


response_schema_for_schema_refinement = {
            "title": "DocumentSchemaRefinementResult",
            "type": "object",
            "description": "Result schema for schema refinement of entities and relationships from text",
            "properties": {
                "node_types": {
                    "type": "array",
                    "description": "List of canonical node type labels",
                    "items": {
                        "type": "string",
                        "description": "The label/type of the entity (e.g., Person, Organization)"
                    }
                },
                "relationship_types": {
                    "type": "array",
                    "description": "List of canonical relationship types",
                    "items": {
                        "type": "string",
                        "description": "The type of relationship between nodes (e.g., WORKS_FOR, LOCATED_IN)"
                    }
                },
                "merge_log": {
                    "type": "object",
                    "description": "Log of merged types mapping canonical types to their original variants",
                    "properties": {
                        "node_types": {
                            "type": "object",
                            "description": "Mapping of canonical node types to lists of original types that were merged",
                            "additionalProperties": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        },
                        "relationship_types": {
                            "type": "object",
                            "description": "Mapping of canonical relationship types to lists of original types that were merged",
                            "additionalProperties": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "required": ["node_types", "relationship_types"]
                }
            },
            "required": ["node_types", "relationship_types", "merge_log"]
        }

# - "podľa odseku 1" means odsek of this paragraph
### CURRENT LEGAL CONTEXT
# Use this as authoritative context for all local references:
# {path_str if path_str else "UNKNOWN"}

# If context is UNKNOWN, do not invent paragraph numbers.
# If text says "podla odseku 1", interpret it inside CURRENT LEGAL CONTEXT.
# Canonical legal IDs:
# - Paragraf: "Paragraf § X"
# - Odsek with known paragraph: "Paragraf § X Odsek Y"
# - Odsek without known paragraph: "Odsek Y"
# - Pismeno with known paragraph/odsek: "Paragraf § X Odsek Y Pismeno a)"

# - add concise, explicit semantic clarification into relationship properties (`MA_NAROK_NA` → `add: vratenie dane`), use key `add`


# schema driven extraction
system_prompt_for_sde = """
You are an expert legal information extraction engine for Slovak financial law.

Extract:
1. named entities
2. relationships

strictly according to the provided ontology/schema.

# CORE RULES
- Use ONLY exact schema entity and relationship types.
- NO renaming, paraphrasing, or inventing types.
- If unsupported -> use closest valid type or skip.
- Use only Slovak language.
- Remove all diacritics.
- Node IDs must be in Title Case.
- Relationship labels must be in SCREAMING_SNAKE_CASE.
- Use most specific valid type.
- Unify duplicates and resolve coreference.
- Do not hallucinate facts.
- Add properties only if explicitly stated.

# LEGAL NORMALIZATION
- laws -> `PravnyPredpis`
- sections -> `Paragraf`
- `PravnyPredpis` -[OBSAHUJE]-> `Paragraf`
- entity -[JE_PODLA]-> `Paragraf`

If text references:
- `odsek` without explicit paragraf:
  inherit currently active paragraf context

If text contains ranges:
- `odsek 3 az 5`
  -> create:
  - `Odsek 3`
  - `Odsek 4`
  - `Odsek 5`

- `odsek 1 a 2`
  -> split into atomic units

- `paragraf § 2 az 4`
  -> create:
  - `Paragraf § 2`
  - `Paragraf § 3`
  - `Paragraf § 4`

Connect united mention with decomposed via `ODKAZUJE_NA`.

# LEGAL NODE ID FORMAT (STRICT)
Use EXACT formats:

- `Paragraf`
  -> `Paragraf § 16`

- `Odsek`
  -> `Paragraf § 16 Odsek 1`

- `Pismeno`
  -> `Paragraf § 54 Odsek 2 Pismeno a)`

Always include full legal hierarchy in IDs.

# ATOMIC DECOMPOSITION
Prefer smallest meaningful legal units.

Split complex legal constructs into multiple nodes/relations when:
- supported by schema
- semantics remain correct

Otherwise keep compound legal concepts intact. Connect them with decomposed entities via `VZTAHUJE_SA_NA`.

Prefer:
- multi-hop legal structures
over:
- overly broad direct relations

Examples:
- `Oprava Zakladu Dane`
- `Oprava Odpocitanej Dane`
- `Dodanie Tovaru A Sluzby`
- `Oslobodenie Od Dane Podla Paragrafu § 2 Odsek 2`
- `Uplatnovanie Osobitnej Upravy`
- `Zmena Udajov V Oznameni`

# RELATIONSHIPS
- only exact schema relationship types
- correct direction required
- SCREAMING_SNAKE_CASE only

Optional:
- add concise semantic clarification into property:
  `evidence`

Example:
- `MA_NAROK_NA`
  + `evidence: vratenie dane`

# VALIDATION
Ensure:
- valid schema types only
- no diacritics
- Title Case node IDs
- SCREAMING_SNAKE_CASE relationships
- full hierarchy in Paragraf/Odsek/Pismeno IDs
- odsek inheritance from active paragraf context
- atomic decomposition of ranges
- detailed legal entities preserved
"""


# "properties": {
#     "type": "object",
#     "properties": {
#         "name": {"type": "string", "description": "Name of the entity"}
#     },
#     "required": ["name"],
#     "additionalProperties": False,
# },

response_schema_for_sde = {
            "title": "GraphExtractionResult",
            "type": "object",
            "description": "Result schema for extracting a knowledge graph from text",
            "properties": {
                "nodes": {
                    "type": "array",
                    "description": "List of extracted entities/nodes from the text",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Unique node id, Title Case, no diacritics. Keep qualifier suffix if entity gains meaning from a qualifier."},
                            "label": {"type": "string", "description": "Entity type — MUST exactly match one of the allowed node labels from schema."},
                        },
                        "required": ["id", "label"],
                        "additionalProperties": False,
                    }
                },
                "relationships": {
                    "type": "array",
                    "description": "List of relationships. Each MUST be supported by a verbatim text fragment in 'evidence'.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_node_id": {"type": "string"},
                            "source_node_type": {"type": "string"},
                            "relation": {"type": "string", "description": "Relationship type — MUST exactly match one of allowed relationship types. Use MOST SPECIFIC type from hierarchy."},
                            "target_node_id": {"type": "string"},
                            "target_node_type": {"type": "string"},
                            "evidence": {
                                "type": "string",
                                "description": "Verbatim 5-15 word fragment from input text that explicitly supports this triple. If you cannot quote such a fragment, do NOT create this relationship."
                            },
                        },
                        "required": [
                            "source_node_id", "source_node_type", "relation",
                            "target_node_id", "target_node_type", "evidence"
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["nodes", "relationships"],
            "additionalProperties": False,
        }








# ============================================================
# Q&A prompts
# ============================================================

# Stage 1: Sentence Segmentation (KG-GPT Table 4 style)
system_prompt_for_segmentation = """
Si model na **segmentáciu otázky na pod-vety pre KG triple extraction**.

# Úloha
Rozdeľ vstupnú otázku na pod-vety tak, aby každá pod-veta reprezentovala **presne jeden vzťah (triple)**.

# Pravidlá (STRIKTNÉ)
- Každá pod-veta obsahuje **max. 2 entity**
- Každá pod-veta = **1 vzťah medzi entitami**
- Entity:
  - používaj **Title Case**
  - buď **konzistentný naprieč vetami**
- **Re-use entity je zakázaný**, okrem:
  - keď prepája multi-hop reťaz (výstup = vstup)
- Multi-hop:
  - vytvor **lineárny reťazec (chain)**
- Pokrytie:
  - pod-vety musia pokrývať **celý význam otázky**
- Ak je otázka jednoduchá:
  - vráť **iba 1 pod-vetu**

# Normalizácia
- implicitné vzťahy → explicitné (napr. „ktorá vlastní“ → „X vlastní Y“)
- odstráň zámená, nahraď ich entitami
- zjednoduš vetu na **faktický tvar**

# VALIDATION (POVINNÉ)
- Každá veta musí byť mapovateľná na: (entity1, relation, entity2)
- Max. 2 entity na vetu (nikdy viac)
- Každá entita sa použije iba raz (okrem chain prepojenia)
- Ak veta nespĺňa triple formu → uprav ju
- Ak chýbajú entity → vetu nevytváraj
- Výstup musí pokrývať celý význam otázky

# Výstupné obmedzenia
- Nevytváraj meta text
- Nevysvetľuj
- Dodrž presne požadovanú štruktúru
"""

response_schema_for_segmentation = {
    "title": "SentenceSegmentation",
    "type": "object",
    "properties": {
        "sub_sentences": {
            "type": "array",
            "description": "List of sub-sentences each aligned with one KG triple",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The sub-sentence text"},
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Up to 2 entity mentions in this sub-sentence (use Title Case)"
                    }
                },
                "required": ["text", "entities"]
            }
        }
    },
    "required": ["sub_sentences"]
}



# System prompt - defines the agent's role and capabilities
system_prompt_for_generating_query = """
Si agent na iteratívne vyhľadávanie v znalostnom grafe (Neo4j) pomocou Cypher queries.

Tvoj cieľ:
Nájsť relevantné trojice (uzol-vzťah-uzol) pre danú pod-vetu pomocou riadeného prieskumu grafu.
Relevantne trojice musia odpovedat na polozenu otazku.

Máš prístup k nástroju:
- search_database(query: Cypher)

## VSTUP
Dostaneš:
- pod-vetu (query fragment)
- zoznam entít (anchors)
- schému grafu (typy uzlov + vzťahov)
- ukážkové dáta

## STRATÉGIA (STRICT LOOP)

Pre KAŽDÚ anchor entitu vykonaj:

### 1. INITIAL RETRIEVAL
- Nájdeš uzol podľa entity:
  MATCH (n)
  WHERE toLower(n.id) CONTAINS toLower($entity)
  RETURN n.id, labels(n)
  LIMIT 10

Ak nenájdeš → SKIP entity

### 2. RELATION SELECTION
- Z dostupnej schémy vyber NAJVIAC relevantný vzťah pre aktuálny uzol
- Výber musí byť z množiny vztahov
- Nikdy nevymýšľaj nový vzťah

Retry max 2x:
- ak nevieš vybrať → STOP pre túto vetvu

### 3. RETRIEVAL
Vykonaj:
MATCH (a)-[r:SELECTED_REL]->(b)
WHERE toLower(a.id) CONTAINS toLower($anchor)
RETURN a.id, type(r), b.id
LIMIT 25

Ak nič nenájdeš:
- skús opačný smer:
MATCH (a)<-[r:SELECTED_REL]-(b)

Ak stále nič → STOP vetva

### 4. REASONING
Zhodnoť:
- sú trojice relevantné pre pod-vetu a na odpoved?
- má zmysel pokračovať hlbšie?

Ak ÁNO:
- nastav nový anchor = b
- pokračuj (max depth = 3)

Ak NIE:
- STOP vetva

## STOP PODMIENKY
- žiadne nové trojice
- nízka relevancia
- max depth dosiahnutý
- nevalidný vzťah

## PRAVIDLÁ CYHPER
- Používaj iba typy zo schémy
- Nepoužívaj neexistujúce labely/vzťahy
- Case-insensitive matching:
  toLower(...)
- Neznámy smer → použi nesmerový vzťah `-[r]-`
- Vždy LIMIT ≤ 25
- Krátke a presné queries
- Vracaj užitočné vlastnosti: `RETURN n.id, labels(n), type(r), properties(n)`

## DÔLEŽITÉ
- VŽDY používaj search_database pre dáta
- Nevymýšľaj výsledky bez query
- Kombinuj viac dotazov iteratívne
- Preferuj presnosť pred pokrytím
"""


response_schema_for_generating_query = {
            "title": "GraphQueryResult",
            "type": "object",
            "description": "Final query results from graph database exploration",
            "properties": {
                "cypher_query": {
                    "type": "string",
                    "description": "The final/best Cypher query that answers the question"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the query strategy and what was found"
                },
                "nodes_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node IDs that are relevant to the answer"
                },
                "relationships_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relationships found (format: 'nodeA -[REL_TYPE]-> nodeB')"
                }
            },
            "required": ["cypher_query", "explanation", "nodes_found", "relationships_found"]
        }



# Stage 2: Relation Retrieval (KG-GPT Table 5 style)
system_prompt_for_relation_retrieval = """Dostaneš pod-vetu a zoznam kandidátskych typov vzťahov z grafovej databázy.

Tvojou úlohou je vybrať top-K typov vzťahov zo zoznamu kandidátov, ktoré sú sémanticky najbližšie k vzťahu vyjadrenému v pod-vete.

Pravidlá:
- Vyberaj IBA z poskytnutého zoznamu kandidátov — nevymýšľaj nové typy.
- Ak žiaden typ nesedí, vyber tie, ktoré sú najbližšie.
- Vrať maximálne K typov vzťahov, zoradených od najrelevantnejšieho."""

response_schema_for_relation_retrieval = {
    "title": "TopKRelations",
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Top-K most semantically relevant relation types from the candidates list"
        }
    },
    "required": ["relations"]
}




# Stage 2 (ARK-V1): Relation selection — LLM call #1 per reasoning step
system_prompt_ark_select_relation = """
# Úloha
Si komponent grafového vyhľadávacieho agenta (ARK-V1). Tvoja jediná úloha je vybrať PRÁVE JEDEN vzťah zo zoznamu kandidátov, ktorý je najrelevantnejší pre zodpovedanie pod-vety pri danom uzle (anchor).

# Vstupy
- Pod-veta (cieľ retrievalu)
- Aktuálny anchor (id uzla v grafe)
- Doterajšie zhrnutie uvažovania (môže byť prázdne v prvom kroku)
- Zoznam kandidátnych vzťahov R^k (presne tie, ktoré existujú v grafe pre tento anchor)

# Striktné pravidlá
- Vyber PRESNE jeden reťazec z poskytnutého zoznamu `relations`. Nepoužívaj ani neupravuj názvy vzťahov mimo tohto zoznamu.
- Ak žiadny z kandidátov nie je relevantný pre pod-vetu, vráť `relation = null`.
- Nevymýšľaj nové vzťahy. Nepridávaj prefixy/sufixy, nemeň veľkosť písmen.
- V `rationale` krátko (1 veta, slovensky) vysvetli, prečo si daný vzťah (alebo `null`) zvolil.
"""

# Stage 2 (ARK-V1): Reasoning over retrieved triples — LLM call #2 per reasoning step
system_prompt_ark_reasoning = """
# Úloha
Si komponent grafového vyhľadávacieho agenta (ARK-V1). Dostaneš zoznam trojíc (indexovaný od 0), ktoré boli získané z grafu pre daný anchor a vzťah. Tvoja úloha:

1. Vyber indexy trojíc, ktoré sú skutočne relevantné pre pod-vetu (`selected_triple_indices`). Indexy musia byť podmnožinou poskytnutých; nevymýšľaj nové.
2. Napíš jednu krátku vetu (`implication`), čo z vybraných trojíc vyplýva vzhľadom na pod-vetu. Ak nič relevantné nie je, napíš to explicitne.
3. Rozhodni, či má zmysel pokračovať ďalším krokom uvažovania (`continue_reasoning`: true/false). Pokračuj len ak je pravdepodobné, že ďalší hop poskytne chýbajúci fakt.
4. Navrhni `next_anchor` — MUSÍ to byť tail (cieľový uzol) jednej z vybraných trojíc. Ak nepokračuješ, vráť `null`.

# Striktné pravidlá
- Nepridávaj trojice, ktoré nie sú v zozname.
- `next_anchor` musí doslova zodpovedať `t` niektorej vybranej trojice (inak ho runtime zahodí).
- Odpovedaj stručne a po slovensky.
"""

# Stage 3: Inference (KG-GPT Table 6 style)
system_prompt_for_inference = """
# Úloha
Si expert na analýzu vedomostných grafov a extrakciu informácií. Tvojou úlohou je odpovedať na otázku používateľa s využitím dvoch zdrojov: štruktúrovaných trojíc z grafovej databázy a sprievodného textového kontextu.

# Hierarchia pravdy a dôkazov
1. **Primárny zdroj (Graf):** Štruktúrované trojice `[hlava, vzťah, chvost]` sú tvojím hlavným zdrojom faktov. Ak sú tieto informácie dostupné, odpoveď musí byť postavená primárne na nich.
2. **Sekundárny zdroj (Text):** Textové úryvky (chunky) použi výhradne na doplnenie detailov, vysvetlenie nuáns alebo prepojenie súvislostí, ktoré nie sú explicitne v grafe.

# Inštrukcie pre spracovanie
- **Prepojenie entít:** Ak sú v grafe viaceré trojice, pokús sa medzi nimi nájsť logickú cestu (napr. A -> B -> C), aby si vytvoril komplexnú odpoveď.
- **Vernosť dátam:** Odpovedaj len na základe poskytnutých dát. Ak graf a text neobsahujú odpoveď, priznaj to (napr. "Na základe poskytnutých údajov nie je možné na otázku odpovedať").
- **Štýl odpovede:** Píš prirodzeným jazykom, ale zachovaj presnosť terminológie z grafu.
- **Konflikt informácií:** V prípade rozporu medzi grafom a textom uprednostni informáciu z grafovej databázy.
"""

