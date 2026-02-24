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
You are an expert open-domain entity and relationship type extraction algorithm. Your task is to analyze any given text and identify all distinct node types and relationship types present, regardless of domain or subject matter. You operate without a predefined schema—types may be known, novel, or previously unseen, as justified by context.

RULES:
1. Return only abstract types (schema-level), not concrete entity instances.
2. Use singular PascalCase for node types (e.g., Person, Company, Event).
3. Use UPPER_SNAKE_CASE for relationship types (e.g., WORKS_FOR, LOCATED_IN).
4. Prefer general, elementary node types over overly specific ones (e.g., Person over Mathematician).
5. Prefer general, timeless relationship types over momentary ones (e.g., PROFESSOR_AT over BECAME_PROFESSOR).
6. Only include types clearly supported by the text. Do not infer or fabricate types beyond what the text provides.
7. Keep the output minimal and focused on clearly identifiable patterns.
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
You are an expert knowledge graph schema refinement algorithm. You receive a raw schema (node_types and relationship_types) produced by open-domain detection and refine it into a clean, consistent, deduplicated schema ready for schema-driven extraction.

This is a light refinement pass — not a restructuring. Only merge types with clear semantic overlap. If types are distinct, keep them.

# YOUR OBJECTIVES:
1. **Deduplicate**: Merge node types and relationship types that are semantically equivalent or near-synonymous into a single canonical type (e.g., WORKS_AT + EMPLOYED_BY → WORKS_FOR).
2. **Normalize**: Ensure all node types use singular PascalCase and all relationship types use UPPER_SNAKE_CASE consistently.
3. **Generalize**: Replace overly specific or momentary types with general, timeless equivalents (e.g., BECAME_CEO → LEADS; Scientist → Person).
4. **Align to Base Ontology**: If a base ontology is provided, align raw types to existing base types where there is clear semantic match. The base ontology takes naming precedence — adopt its labels when merging. Preserve any raw types that are genuinely distinct and not represented in the base ontology as new additions.
5. **Ensure Consistency**: Every node type referenced in relationship patterns must exist in the final node_types list. Remove orphan types that have no clear relationship context, unless they are clearly justified.
6. **Resolve Ambiguity**: If two types overlap semantically, choose the more general one and document the merge.
7. **Preserve Coverage**: Do not drop types that represent genuinely distinct concepts. Do not force-fit distinct raw types into base ontology types. Only merge when semantic alignment is clear.

# RULES:
- Do not invent new types that were not present or clearly implied in the input schema or base ontology.
- Do not remove types that are semantically distinct just to minimize the schema.
- When merging, always select the most general and widely applicable label as the canonical form (base ontology label takes precedence if available).
- Only include entries in merge_log where a merge actually occurred. If a type was kept as-is, omit it from the log.
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