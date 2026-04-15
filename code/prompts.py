

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

# FORMÁT VÝSTUPU
Výstup musí byť JSON s kľúčmi:
- "node_types": Zoznam spresnených typov uzlov.
- "relationship_types": Zoznam spresnených typov vzťahov.
- "merge_log": Objekt, kde kľúč je cieľový názov a hodnota je zoznam pôvodných názvov, ktoré doň boli zlúčené.
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


# schema driven extraction
system_prompt_for_sde = """
Extract a knowledge graph strictly using the provided schema.

### RULES
- Use ONLY exact schema types (entities + relationships).
- NO renaming, NO variations.
- If type not available → use closest OR skip.
- Prefer MOST SPECIFIC type.
- Use only Slovak language.

### COVERAGE
Extract ALL:
taxes, fees, laws (`PravnyPredpis`), documents, activities (`Cinnost`),
time (`CasoveObdobie`), rights, obligations, amounts.

### LEGAL
- laws → `PravnyPredpis`
- sections → `Paragraf`
- `PravnyPredpis` -[OBSAHUJE]-> `Paragraf`
- entity -[PODLA]-> `Paragraf` (if referenced)

### NODES
- ID = full readable name (not numeric only)
- properties only if explicit
- unify duplicates (coreference)

### RELATIONSHIPS
- only allowed types
- correct direction
- exact match required

### FORMAT
- NO DIACRITICS

### VALIDATION
Ensure:
- all types valid
- most specific used
- includes detailed + legal entities
"""


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
                            "id": {"type": "string", "description": "Unique identifier for the node"},
                            "label": {"type": "string", "description": "Type/label of the entity (e.g., Person, Organization)"},
                            "properties": {
                                "type": "object",
                                "description": "Properties of the entity as mentioned in the text",
                                "properties": {
                                    "name": {"type": "string", "description": "Name of the entity"}
                                },
                                "required": ["name"]
                            },
                        },
                        "required": ["id", "label", "properties"]
                    }
                },
                "relationships": {
                    "type": "array",
                    "description": "List of relationships between nodes",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_node_id": {"type": "string", "description": "ID of the source node"},
                            "source_node_type": {"type": "string", "description": "Type/label of the source node"},
                            "relation": {"type": "string", "description": "Type of relationship between source and target (e.g., KNOWS, WORKS_AT)"},
                            "target_node_id": {"type": "string", "description": "ID of the target node"},
                            "target_node_type": {"type": "string", "description": "Type/label of the target node"},
                            "properties": {
                                "type": "object",
                                "description": "Properties of the relationship as mentioned in the text",
                                "properties": { }
                            },
                        },
                        "required": ["relation", "source_node_id", "target_node_id", "source_node_type", "target_node_type"]
                    }
                }
            },
            "required": ["nodes", "relationships"]
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

