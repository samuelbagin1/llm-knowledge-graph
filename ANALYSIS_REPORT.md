# Deep Analysis: PDFGraphRAG vs LangChain LLMGraphTransformer

## Executive Summary

This report compares your custom `PDFGraphRAG` implementation in [pdf_graphrag.py](code/pdf_graphrag.py) with LangChain's `LLMGraphTransformer` from `langchain_experimental.graph_transformers.llm`. Both aim to extract knowledge graphs from text, but take different architectural approaches.

---

## 1. Architecture Comparison

### LangChain LLMGraphTransformer

| Aspect | Implementation |
|--------|----------------|
| **Design Pattern** | Single-purpose transformer class |
| **LLM Integration** | Uses structured output via `with_structured_output()` |
| **Schema Enforcement** | Pydantic models with dynamic field generation |
| **Async Support** | Native `asyncio.gather()` for parallel processing |
| **Fallback Handling** | JSON repair library for non-function-calling LLMs |

### Your PDFGraphRAG

| Aspect | Implementation |
|--------|----------------|
| **Design Pattern** | Full RAG pipeline (PDF → Graph → Query) |
| **LLM Integration** | Agent-based with `ProviderStrategy` schema |
| **Schema Enforcement** | JSON schema dictionary |
| **Async Support** | `asyncio.gather()` with task creation |
| **Additional Features** | Chunk embeddings, vector stores, query pipeline |

---

## 2. Entity Extraction Approach

### LangChain ([llm.py:715-830](llm.py#L715-L830))

```python
# Creates dynamic Pydantic models at runtime
schema = create_simple_model(
    allowed_nodes,           # List[str]
    allowed_relationships,   # List[str] or List[Tuple[str,str,str]]
    node_properties,         # bool or List[str]
    llm_type,               # For enum optimization
    relationship_properties,
    relationship_type,       # "string" or "tuple"
)
structured_llm = llm.with_structured_output(schema, include_raw=True)
```

**Key Features:**
- Dynamic Pydantic model creation via `create_model()` ([llm.py:401](llm.py#L401))
- OpenAI-specific enum optimization for constrained outputs ([llm.py:167-173](llm.py#L167-L173))
- Tuple-based relationship constraints `(SourceType, REL_TYPE, TargetType)` ([llm.py:697-706](llm.py#L697-L706))
- Raw response preservation for error recovery ([llm.py:828](llm.py#L828))

### Your Implementation ([pdf_graphrag.py:369-403](code/pdf_graphrag.py#L369-L403))

```python
# Uses agent with JSON schema
agent = create_agent(
    model=self.openai_graph_transform,
    response_format=ProviderStrategy(schema=response_schema_for_extraction),
    system_prompt=system_prompt_for_extracting
)
response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
data = response["structured_response"]
```

**Key Differences:**
1. Agent-based rather than direct chain
2. JSON schema instead of Pydantic models
3. No enum constraints on allowed values
4. Manual prompt construction with `{nodes}` and `{relationships}` variables (bug: undefined at runtime)

---

## 3. Critical Bug in Your Code

At [pdf_graphrag.py:377-384](code/pdf_graphrag.py#L377-L384):

```python
user_prompt = f"""
Extract named entities and relationships from the text

Allowed nodes:
{nodes}           # ← BUG: 'nodes' is undefined here!

Allowd relationships:
{relationships}   # ← BUG: 'relationships' is undefined here!

Text:
{text}
"""
```

The variables `nodes` and `relationships` are not defined in the `named_entity_extraction` method scope. This will raise `NameError` at runtime. Compare with LangChain's approach at [llm.py:227-266](llm.py#L227-L266) which properly injects these values.

---

## 4. Graph Document Construction

### LangChain ([llm.py:503-522](llm.py#L503-L522))

```python
def map_to_base_node(node: Any) -> Node:
    properties = {}
    if hasattr(node, "properties") and node.properties:
        for p in node.properties:
            properties[format_property_key(p.key)] = p.value
    return Node(id=node.id, type=node.type, properties=properties)
```

- Property key formatting to camelCase
- Validation that nodes have IDs ([llm.py:530-531](llm.py#L530-L531))
- Node type fallback to `DEFAULT_NODE_TYPE` ([llm.py:539](llm.py#L539))

### Your Implementation ([pdf_graphrag.py:312-367](code/pdf_graphrag.py#L312-L367))

```python
def convert_to_graph_document(self, data, i, document) -> GraphDocument:
    chunk_node = Node(
        id=chunk_id,
        type="Chunk",
        properties={
            "text": document.page_content,
            "embedding": chunk_embedding,  # ← Stores embedding in node
            "page": document.metadata.get("page", 0)
        }
    )
    # Links extracted entities to chunk
    for node in nodes:
        relationships.append(
            Relationship(source=chunk_node, target=node, type="HAS")
        )
```

**Your advantages:**
1. **Chunk-Entity Linking**: Creates `HAS` relationships between chunks and extracted entities
2. **Embedding Storage**: Embeds chunk content directly in node properties
3. **Metadata Preservation**: Stores page numbers for traceability

**Missing from yours:**
1. No property key formatting
2. No validation for missing node IDs
3. No type normalization/capitalization

---

## 5. Strict Mode Filtering

### LangChain ([llm.py:880-917](llm.py#L880-L917))

```python
if self.strict_mode and (self.allowed_nodes or self.allowed_relationships):
    if self.allowed_nodes:
        lower_allowed_nodes = [el.lower() for el in self.allowed_nodes]
        nodes = [node for node in nodes if node.type.lower() in lower_allowed_nodes]
        relationships = [
            rel for rel in relationships
            if rel.source.type.lower() in lower_allowed_nodes
            and rel.target.type.lower() in lower_allowed_nodes
        ]
```

LangChain performs **post-extraction filtering** to enforce schema compliance even when LLM outputs non-conforming entities.

### Your Implementation

**No equivalent strict mode filtering.** Your code relies entirely on the LLM to respect the schema, which is less reliable.

---

## 6. Async Processing Comparison

### LangChain ([llm.py:1022-1033](llm.py#L1022-L1033))

```python
async def aconvert_to_graph_documents(
    self, documents: Sequence[Document], config: Optional[RunnableConfig] = None
) -> List[GraphDocument]:
    tasks = [
        asyncio.create_task(self.aprocess_response(document, config))
        for document in documents
    ]
    results = await asyncio.gather(*tasks)
    return results
```

### Your Implementation ([pdf_graphrag.py:405-411](code/pdf_graphrag.py#L405-L411))

```python
async def async_process(self, documents: List[Document], ...) -> List[GraphDocument]:
    tasks = [asyncio.create_task(self.named_entity_extraction(i, doc, allowed_entities))
             for i, doc in enumerate(documents)]
    res = await asyncio.gather(*tasks)
    return res
```

Both use the same pattern. However, your `process()` method at [pdf_graphrag.py:431](code/pdf_graphrag.py#L431) calls `self.async_process()` synchronously without `await` or `asyncio.run()`:

```python
graph_docs = self.async_process(chunked_documents)  # ← Returns coroutine, not results!
```

This is a **critical bug** - `graph_docs` will be a coroutine object, not a list of GraphDocuments.

---

## 7. Prompt Engineering Comparison

### LangChain System Prompt ([llm.py:72-107](llm.py#L72-L107))

```
# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information...

## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types...
- **Node IDs**: Never utilize integers as node IDs...

## 3. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities...

## 4. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
```

### Your System Prompt ([prompts.py:1-27](code/prompts.py#L1-L27))

Nearly identical to LangChain's (appears derived from it), with minor wording changes. Good foundation.

---

## 8. Schema Comparison

### LangChain Dynamic Schema ([llm.py:356-446](llm.py#L356-L446))

Creates Pydantic models with:
- `id`: Required string with description
- `type`: String with optional enum constraint
- `properties`: Optional list of key-value Property objects

### Your JSON Schema ([prompts.py:30-77](code/prompts.py#L30-L77))

```json
{
  "nodes": [{
    "id": "string",
    "label": "string",  // ← Named 'label' not 'type'
    "properties": {
      "name": "string"  // ← Fixed 'name' property
    }
  }],
  "relationships": [{
    "source_node_id": "string",
    "relation": "string",  // ← Named 'relation' not 'type'
    ...
  }]
}
```

**Inconsistencies:**
1. You use `label` for node type, LangChain uses `type`
2. You use `relation` for relationship type, LangChain uses `type`
3. Your `convert_to_graph_document` expects `node_data["label"]` ([pdf_graphrag.py:335](code/pdf_graphrag.py#L335)) - matches schema
4. Your `convert_to_graph_document` expects `rel_data["relation"]` ([pdf_graphrag.py:347](code/pdf_graphrag.py#L347)) - matches schema

This is internally consistent but differs from LangChain conventions.

---

## 9. Feature Matrix

| Feature | LangChain | Your Implementation |
|---------|-----------|---------------------|
| Entity extraction | ✅ | ✅ |
| Relationship extraction | ✅ | ✅ |
| Node properties | ✅ | ✅ |
| Relationship properties | ✅ | ✅ |
| Strict mode filtering | ✅ | ❌ |
| Tuple relationship constraints | ✅ | ❌ |
| Coreference resolution (prompt) | ✅ | ✅ |
| Async processing | ✅ | ⚠️ (buggy) |
| JSON repair fallback | ✅ | ❌ |
| Chunk-entity linking | ❌ | ✅ |
| Embedding in nodes | ❌ | ✅ |
| Vector store integration | ❌ | ✅ |
| Query pipeline | ❌ | ✅ |
| PDF loading | ❌ | ✅ |

---

## 10. Recommendations

### Critical Fixes

1. **Fix undefined variables bug** at [pdf_graphrag.py:381-384](code/pdf_graphrag.py#L381-L384):
```python
async def named_entity_extraction(self, i, document: Document,
                                   allowed_entities: Optional[List[str]] = None) -> GraphDocument:
    allowed_nodes = allowed_entities or ["Person", "Organization", "Location", ...]
    allowed_rels = ["KNOWS", "WORKS_AT", ...]

    user_prompt = f"""
    Allowed nodes:
    {allowed_nodes}

    Allowed relationships:
    {allowed_rels}
    ...
    """
```

2. **Fix async call** at [pdf_graphrag.py:431](code/pdf_graphrag.py#L431):
```python
# Change from:
graph_docs = self.async_process(chunked_documents)

# To:
graph_docs = asyncio.run(self.async_process(chunked_documents))
```

### Enhancements

3. **Add strict mode filtering** after extraction to enforce schema compliance

4. **Add validation** for node IDs before creating GraphDocuments

5. **Consider using LangChain's LLMGraphTransformer** for extraction and wrapping it with your chunk-linking logic:

```python
def process(self, pdf_path: str, max_pages: int = None):
    documents = self.load_pdf(pdf_path)
    chunked_documents = splitter.split_documents(documents)

    # Use LangChain for extraction
    graph_docs = self.graph_transformer.convert_to_graph_documents(chunked_documents)

    # Add your chunk-entity linking
    enhanced_docs = self.add_chunk_links(graph_docs, chunked_documents)

    self.graph.add_graph_documents(enhanced_docs, ...)
```

---

## 11. Conclusion

Your implementation extends beyond LangChain's by providing a complete RAG pipeline with chunk tracking and vector search. However, it has critical runtime bugs that need fixing. The extraction logic is similar but lacks LangChain's robustness features (strict mode, JSON repair, enum constraints).

**Recommendation:** Fix the critical bugs, then consider a hybrid approach using LangChain's battle-tested extraction with your enhanced storage and retrieval pipeline.
