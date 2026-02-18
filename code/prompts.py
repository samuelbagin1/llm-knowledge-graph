system_prompt_for_sde = """
# Knowledge Graph Instructions for GPT-4

## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
Try to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.
Extract the entities (nodes) and specify their type from the following text. Also extract the relationships between these nodes.

## 2. Labeling Nodes
- **Nodes** represent entities.
- **Node IDs**: assign a unique ID (string) to each node, and reuse it to define relationships.
- **Node Name**: Use the most complete and specific name available in the text for each entity.
- **Node Types**: Use only the provided node types for labeling entities.
- **Relationships** represent connections between entities, do respect the source and target node types for relationship and the relationship direction.
- **Consistency**: Ensure you use available types for node labels.
Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms like 'mathematician' or 'scientist'.

Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!

## 3. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity throughout the knowledge graph.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

## 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""


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


# TODO: edit the prompt
# open domain detection
system_prompt_for_odd = """
# Knowledge Graph Instructions for GPT-4

## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
Try to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.
Extract the entities (nodes) and specify their type from the following text. Also extract the relationships between these nodes.

## 2. Labeling Nodes
- **Nodes** represent entities.
- **Node IDs**: assign a unique ID (string) to each node, and reuse it to define relationships.
- **Node Name**: Use the most complete and specific name available in the text for each entity.
- **Node Types**: Use only the provided node types for labeling entities.
- **Relationships** represent connections between entities, do respect the source and target node types for relationship and the relationship direction.
- **Consistency**: Ensure you use available types for node labels.
Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms like 'mathematician' or 'scientist'.

Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!

## 3. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity throughout the knowledge graph.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.

## 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination."""


response_schema_for_odd = {
            "title": "DocumentOpenDomainDetectionResult",
            "type": "object",
            "description": "Result schema for open domain detection of entities and relationships from text",
            "properties": {
                "list_nodes": {
                    "type": "array",
                    "description": "List of extracted entity labels from the text",
                    "items": {
                        "type": "string",
                        "description": "The label/type of the entity (e.g., Person, Organization)"
                    }
                },
                "list_relationships": {
                    "type": "array",
                    "description": "List of relationships between nodes",
                    "items": {
                        "type": "string",
                        "description": "The type of relationship between nodes (e.g., KNOWS, WORKS_AT)"
                    }
                }
            },
            "required": ["list_nodes", "list_relationships"]
        }