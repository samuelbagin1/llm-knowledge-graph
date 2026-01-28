# Prompts

## neo4j_graphrag - prompts.py

### RagTemplate

**System Instructions:**
```
Answer the user question using the provided context.
```

**Template:**
```
Context:
{context}

Examples:
{examples}

Question:
{query_text}

Answer:
```

---

### Text2CypherTemplate

**Template:**
```
Task: Generate a Cypher statement for querying a Neo4j graph database from a user input.

Schema:
{schema}

Examples (optional):
{examples}

Input:
{query_text}

Do not use any properties or relationships not included in the schema.
Do not include triple backticks ``` or any additional text except the generated Cypher statement in your response.

Cypher query:
```

---

### ERExtractionTemplate

**Template:**
```
You are a top-tier algorithm designed for extracting
information in structured formats to build a knowledge graph.

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

Return result as JSON using the following format:
{"nodes": [ {"id": "0", "label": "Person", "properties": {"name": "John"} }],
"relationships": [{"type": "KNOWS", "start_node_id": "0", "end_node_id": "1", "properties": {"since": "2024-08-01"} }] }

Use only the following node and relationship types (if provided):
{schema}

Assign a unique ID (string) to each node, and reuse it to define relationships.
Do respect the source and target node types for relationship and
the relationship direction.

Make sure you adhere to the following rules to produce valid JSON objects:
- Do not return any additional information other than the JSON in it.
- Omit any backticks around the JSON - simply output the JSON on its own.
- The JSON object must not wrapped into a list - it is its own JSON object.
- Property names must be enclosed in double quotes

Examples:
{examples}

Input text:

{text}
```

---

### SchemaExtractionTemplate

**Template:**
```
You are a top-tier algorithm designed for extracting a labeled property graph schema in
structured formats.

Generate a generalized graph schema based on the input text. Identify key node types,
their relationship types, and property types.

IMPORTANT RULES:
1. Return only abstract schema information, not concrete instances.
2. Use singular PascalCase labels for node types (e.g., Person, Company, Product).
3. Use UPPER_SNAKE_CASE labels for relationship types (e.g., WORKS_FOR, MANAGES).
4. Include property definitions only when the type can be confidently inferred, otherwise omit them.
5. When defining patterns, ensure that every node label and relationship label mentioned exists in your lists of node types and relationship types.
6. Do not create node types that aren't clearly mentioned in the text.
7. Keep your schema minimal and focused on clearly identifiable patterns in the text.

Accepted property types are: BOOLEAN, DATE, DURATION, FLOAT, INTEGER, LIST,
LOCAL_DATETIME, LOCAL_TIME, POINT, STRING, ZONED_DATETIME, ZONED_TIME.

Return a valid JSON object that follows this precise structure:
{
  "node_types": [
    {
      "label": "Person",
      "properties": [
        {
          "name": "name",
          "type": "STRING"
        }
      ]
    },
    ...
  ],
  "relationship_types": [
    {
      "label": "WORKS_FOR"
    },
    ...
  ],
  "patterns": [
    ["Person", "WORKS_FOR", "Company"],
    ...
  ]
}

Examples:
{examples}

Input text:
{text}
```

---

## langchain_experimental - LLMGraphTransformer (llm.py)

### System Prompt (structured output mode)

```
# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
Try to capture as much information from the text as possible without sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text.
- **Nodes** represent entities and concepts.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it
accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use available types for node labels.
Ensure you use basic or elementary types for node labels.
- For example, when you identify an entity representing a person, always label it as **'person'**. Avoid using more specific terms like 'mathematician' or 'scientist'.
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
- **Relationships** represent connections between entities or concepts.
Ensure consistency and generality in relationship types when constructing knowledge graphs. Instead of using specific and momentary types such as 'BECAME_PROFESSOR', use more general and timeless relationship types like 'PROFESSOR'. Make sure to use general and timeless relationship types!
## 3. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
```

**Human Message (structured output mode):**
```
{additional_instructions} Tip: Make sure to answer in the correct format and do not include any explanations. Use the given format to extract information from the following input: {input}
```

---

### Few-Shot Examples (used in unstructured mode)

```json
[
  {
    "text": "Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent",
    "head": "Adam",
    "head_type": "Person",
    "relation": "WORKS_FOR",
    "tail": "Microsoft",
    "tail_type": "Company"
  },
  {
    "text": "Adam is a software engineer in Microsoft since 2009, and last year he got an award as the Best Talent",
    "head": "Adam",
    "head_type": "Person",
    "relation": "HAS_AWARD",
    "tail": "Best Talent",
    "tail_type": "Award"
  },
  {
    "text": "Microsoft is a tech company that provide several products such as Microsoft Word",
    "head": "Microsoft Word",
    "head_type": "Product",
    "relation": "PRODUCED_BY",
    "tail": "Microsoft",
    "tail_type": "Company"
  },
  {
    "text": "Microsoft Word is a lightweight app that accessible offline",
    "head": "Microsoft Word",
    "head_type": "Product",
    "relation": "HAS_CHARACTERISTIC",
    "tail": "lightweight app",
    "tail_type": "Characteristic"
  },
  {
    "text": "Microsoft Word is a lightweight app that accessible offline",
    "head": "Microsoft Word",
    "head_type": "Product",
    "relation": "HAS_CHARACTERISTIC",
    "tail": "accessible offline",
    "tail_type": "Characteristic"
  }
]
```

---

### Unstructured Prompt (non-tool-calling mode)

**System Message:**
```
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Your task is to identify the entities and relations requested with the user prompt from a given text. You must generate the output in a JSON format containing a list with JSON objects. Each object should have the keys: "head", "head_type", "relation", "tail", and "tail_type". The "head" key must contain the text of the extracted entity with one of the types from the provided list in the user prompt.
The "head_type" key must contain the type of the extracted head entity, which must be one of the types from {node_labels}.
The "relation" key must contain the type of relation between the "head" and the "tail", which must be one of the relations from {rel_types}.
The "tail" key must represent the text of an extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail entity from {node_labels}.
Attempt to extract as many entities and relations as you can. Maintain Entity Consistency: When extracting entities, it's vital to ensure consistency. If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity. The knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
IMPORTANT NOTES:
- Don't add any explanation and text.
```

**Human Message:**
```
Based on the following example, extract entities and relations from the provided text.

Use the following entity types, don't use other entity that is not defined below:
# ENTITY TYPES:
{node_labels}

Use the following relation types, don't use other relation that is not defined below:
# RELATION TYPES:
{rel_types}

Below are a number of examples of text and their extracted entities and relationships.
{examples}

For the following text, extract entities and relations as in the provided example.
{format_instructions}
Text: {input}
```
