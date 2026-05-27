

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