

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
Si expertný algoritmus na spresňovanie schém znalostných grafov. Dostávaš surovú schému (typy uzlov a typy vzťahov) vytvorenú detekciou z otvorenej domény a tvojou úlohou je spresniť ju na čistú, konzistentnú a deduplikovanú schému pripravenú na extrakciu riadenú schémou.

Ide o ľahký spresňovací prechod — nie o reštrukturalizáciu. Zlučuj iba typy s jasným sémantickým prekryvom. Ak sú typy odlišné, ponechaj ich.

# TVOJE CIELE:
1. **Deduplikácia**: Zlúč typy uzlov a typy vzťahov, ktoré sú sémanticky ekvivalentné alebo takmer synonymné, do jedného kanonického typu (napr. PRACUJE_NA + ZAMESTNANY_V → PRACUJE_PRE).
2. **Normalizácia**: Zabezpeč, aby všetky typy uzlov používali singulár v PascalCase a všetky typy vzťahov používali UPPER_SNAKE_CASE konzistentne.
3. **Generalizácia**: Nahraď príliš špecifické alebo momentálne typy všeobecnými, nadčasovými ekvivalentmi (napr. STAL_SA_RIADITELOM → VEDIE; Vedec → Osoba).
4. **Zarovnanie na základnú ontológiu**: Ak je poskytnutá základná ontológia, zarovnaj surové typy na existujúce základné typy tam, kde existuje jasná sémantická zhoda. Základná ontológia má prednosť v pomenovaniach — pri zlučovaní preberaj jej označenia. Zachovaj všetky surové typy, ktoré sú skutočne odlišné a nie sú zastúpené v základnej ontológii, ako nové doplnenia.
5. **Zabezpečenie konzistencie**: Každý typ uzla referencovaný vo vzoroch vzťahov musí existovať vo výslednom zozname typov uzlov. Odstráň osirelé typy, ktoré nemajú jasný kontext vzťahu, pokiaľ nie sú jasne odôvodnené.
6. **Riešenie nejednoznačnosti**: Ak sa dva typy sémanticky prekrývajú, vyber ten všeobecnejší a zdokumentuj zlúčenie.
7. **Zachovanie pokrytia**: Neodstraňuj typy, ktoré reprezentujú skutočne odlišné koncepty. Nenúť odlišné surové typy do typov základnej ontológie. Zlučuj iba vtedy, keď je sémantické zarovnanie jasné.

# PRAVIDLÁ:
- Nevymýšľaj nové typy, ktoré neboli prítomné alebo jasne implikované vo vstupnej schéme alebo základnej ontológii.
- Neodstraňuj typy, ktoré sú sémanticky odlišné, len kvôli minimalizácii schémy.
- Pri zlučovaní vždy vyber najvšeobecnejšie a najširšie použiteľné označenie ako kanonickú formu (označenie základnej ontológie má prednosť, ak je dostupné).
- Do merge_log zahrň iba záznamy, kde skutočne došlo k zlúčeniu. Ak bol typ ponechaný bez zmeny, vynechaj ho z logu.
- Všetky extrahované názvy typov uzlov aj vzťahov musia byť BEZ DIAKRITIKY — nahraď znaky s diakritikou ich základnými ASCII ekvivalentmi (napr. č→c, š→s, ž→z, á→a, é→e, í→i, ó→o, ú→u, ý→y, ň→n, ť→t, ď→d, ľ→l, ô→o).
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
You are an expert knowledge graph extraction algorithm. Your task is to extract named entities (nodes) and relationships from text according to a provided ontology schema.

## Core Principles
- Extract as much information as possible without sacrificing accuracy.
- Do not add any information that is not explicitly mentioned in the text.
- Use ONLY the entity types and relationship types defined in the provided schema. Do not invent or use types outside the schema.

## Node Extraction
- **Label Consistency**: Always use the entity types provided in the schema. Do not substitute with more specific or alternative labels.
- **Node IDs**: Use the most complete human-readable name or identifier found in the text. Never use integers as node IDs.
- **Properties**: Include properties only when explicitly stated in the text and confidently inferable.

## Relationship Extraction
- Use ONLY relationship types from the provided schema.
- Ensure correct directionality: the start node and end node must match the semantic direction of the relationship.
- Include relationship properties only when explicitly stated in the text.

## Coreference Resolution
- If an entity is mentioned multiple times by different names or pronouns (e.g., "John Doe", "Joe", "he"), always resolve to the most complete identifier as the node ID.
- Maintain consistency across all references to the same entity.
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









# System prompt - defines the agent's role and capabilities
system_prompt_for_generating_query = """You are a Neo4j Cypher expert agent specialized in querying knowledge graphs.

Your task is to answer questions by querying a Neo4j graph database.

## Your Capabilities
You have access to the `search_database` tool which executes Cypher queries against Neo4j.

## Query Strategy
1. **Analyze the question** to identify what nodes and relationships are relevant
2. **Start with exploration queries** to understand what data exists:
   - For nodes: `MATCH (n:Label) RETURN n.id, labels(n)`
   - For relationships: `MATCH (a)-[r:TYPE]->(b) RETURN a.id, type(r), b.id`
3. **Refine iteratively** - use results from initial queries to build more specific queries
4. **Find the best matches** - keep querying until you find the most relevant data

## Cypher Query Rules
- Use backticks for labels/types with special characters: `MATCH (n:`Special-Label`) ...`
- For text matching use case-insensitive: `WHERE toLower(n.id) CONTAINS toLower('romeo')`
- Use undirected relationships `-[r]-` when direction is unknown
- Always add `LIMIT 25` to prevent large result sets
- Return useful properties: `RETURN n.id, labels(n), type(r), properties(n)`
- Keep queries efficient, focused and short

## Important
- You MUST use the search_database tool to query the database
- Make multiple queries if needed to find the best answer
- When you have found sufficient data, provide your final answer with the best Cypher query"""


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

