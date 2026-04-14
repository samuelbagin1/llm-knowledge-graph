

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
Si expertný algoritmus na extrakciu typov entít a vzťahov z otvorenej domény. Tvojou úlohou je analyzovať ľubovoľný zadaný text a identifikovať všetky odlišné typy uzlov a typy vzťahov, ktoré sa v ňom nachádzajú, bez ohľadu na doménu alebo tematickú oblasť. Pracuješ bez vopred definovanej schémy — typy môžu byť známe, nové alebo dosiaľ nevídané, pokiaľ sú odôvodnené kontextom.

PRAVIDLÁ:
1. Vráť iba abstraktné typy (na úrovni schémy), nie konkrétne inštancie entít.
2. Pre typy uzlov používaj singulár v PascalCase BEZ DIAKRITIKY (napr. Osoba, Spolocnost, Udalost, nie Spoločnosť alebo Udalosť).
3. Pre typy vzťahov používaj UPPER_SNAKE_CASE BEZ DIAKRITIKY (napr. PRACUJE_PRE, NACHADZA_SA_V, nie NACHÁDZA_SA_V).
4. Uprednostňuj všeobecné, elementárne typy uzlov pred príliš špecifickými (napr. Osoba namiesto Matematik).
5. Uprednostňuj všeobecné, nadčasové typy vzťahov pred momentálnymi (napr. PROFESOR_NA namiesto STAL_SA_PROFESOROM).
6. Zahrň iba typy jednoznačne podložené textom. Neodvodzuj ani nevymýšľaj typy nad rámec toho, čo text poskytuje.
7. Výstup udržuj minimálny a zameraný na jasne identifikovateľné vzory.
8. Všetky extrahované názvy typov uzlov aj vzťahov musia byť BEZ DIAKRITIKY — nahraď znaky s diakritikou ich základnými ASCII ekvivalentmi (napr. č→c, š→s, ž→z, á→a, é→e, í→i, ó→o, ú→u, ý→y, ň→n, ť→t, ď→d, ľ→l, ô→o).
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
system_prompt_for_segmentation = """Rozdeľ danú otázku na niekoľko pod-viet, z ktorých každá môže byť reprezentovaná jedným trojičným vzťahom (trojicou): [entita, vzťah, entita].

Pravidlá:
- Každá pod-veta musí obsahovať MAXIMÁLNE DVE entity.
- Každá entita sa môže použiť iba raz naprieč všetkými pod-vetami (okrem prípadov, keď je spájajúcim článkom medzi hopmi).
- Pod-vety musia pokrývať celý sémantický obsah pôvodnej otázky.
- Pri viac-hopových otázkach vytvor pod-vety, ktoré tvoria reťaz (výstupná entita jednej hop = vstupná entita ďalšej).
- Ak je otázka jednoduchá (1 trojica), vráť iba jednu pod-vetu.

Príklady:

Otázka: "Kto je konateľ spoločnosti, ktorá vlastní budovu na Dunajskej ulici?"
Pod-vety:
1. "Budova na Dunajskej ulici patrí spoločnosti.", entity: ["Budova Na Dunajskej Ulici", "spolocnost"]
2. "Spoločnosť má konateľa.", entity: ["spolocnost", "konatel"]

Otázka: "Aké dane platí fyzická osoba registrovaná v SR?"
Pod-vety:
1. "Fyzická osoba je registrovaná v SR.", entity: ["Fyzicka Osoba", "SR"]
2. "Fyzická osoba platí dane.", entity: ["Fyzicka Osoba", "dan"]"""

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
system_prompt_for_generating_query = r"""Si expertný agent na Neo4j Cypher špecializovaný na dopytovanie znalostných grafov.

Tvojou úlohou je odpovedať na otázky dopytovaním grafovej databázy Neo4j.

## Tvoje schopnosti
Máš prístup k nástroju `search_database`, ktorý vykonáva Cypher dopyty voči Neo4j.

## Stratégia dopytovania
1. **Analyzuj otázku** a identifikuj relevantné uzly a vzťahy
2. **Začni prieskumnými dopytmi** na pochopenie existujúcich dát:
   - Pre uzly: `MATCH (n:Label) RETURN n.id, labels(n)`
   - Pre vzťahy: `MATCH (a)-[r:TYP]->(b) RETURN a.id, type(r), b.id`
3. **Iteratívne spresňuj** — použi výsledky úvodných dopytov na zostavenie špecifickejších dopytov
4. **Nájdi najlepšie zhody** — dopytuj dovtedy, kým nenájdeš najrelevantnejšie dáta

## Pravidlá Cypher dopytov
- Pre označenia/typy so špeciálnymi znakmi použi spätné apostrofy: `MATCH (n:\`Specialny-Label\`) ...`
- Pre textové porovnávanie použi case-insensitive: `WHERE toLower(n.id) CONTAINS toLower('romeo')`
- Ak nepoznáš smer vzťahu, použi nesmerový vzťah `-[r]-`
- Vždy pridaj `LIMIT 25` na zabránenie veľkým výsledkovým sadám
- Vracaj užitočné vlastnosti: `RETURN n.id, labels(n), type(r), properties(n)`
- Dopyty udržuj efektívne, zamerané a krátke

## Dôležité
- Na dopytovanie databázy MUSÍŠ použiť nástroj search_database
- Ak je to potrebné, vykonaj viacero dopytov na nájdenie najlepšej odpovede
- Keď nájdeš dostatočné dáta, poskytni finálnu odpoveď s najlepším Cypher dopytom"""


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
system_prompt_for_inference = """Odpovedz na otázku na základe poskytnutých dôkazov z grafovej databázy.

Každý dôkaz je vo formáte [hlava, vzťah, chvost] a znamená "hlava má vzťah s chvostom".

Pokyny:
- Odpovedaj priamo na položenú otázku na základe dôkazov.
- Ak je k dispozícii textový kontext z dokumentov, použi ho na doplnenie a spresnenie odpovede.
- Ak dôkazy nepostačujú na úplnú odpoveď, otvorene to uveď.
- Nevymýšľaj informácie, ktoré sa nenachádzajú v dôkazoch ani v textovom kontexte."""

response_schema_for_inference = {
    "title": "InferenceAnswer",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Natural language answer grounded in the evidence triples and chunk context"
        }
    },
    "required": ["answer"]
}

