# Legal GraphRAG - Quick Reference Guide

## 🎯 Common Legal Patterns for spaCy

### Adding Custom Legal Patterns

Edit `LegalNLPProcessor._setup_legal_patterns()` in `legal_graphrag.py`:

```python
patterns = [
    # Contract-specific terms
    {"label": "PARTY_ROLE", "pattern": [
        {"LOWER": {"IN": ["buyer", "seller", "lessor", "lessee", "employer", "employee"]}}
    ]},
    
    # Termination clauses
    {"label": "TERMINATION", "pattern": [
        {"LEMMA": "terminate"},
        {"LOWER": "upon"}
    ]},
    
    # Payment terms
    {"label": "PAYMENT", "pattern": [
        {"LOWER": "payment"},
        {"LOWER": {"IN": ["of", "shall", "must"]}},
        {"ENT_TYPE": "MONEY"}
    ]},
    
    # Confidentiality
    {"label": "CONFIDENTIAL", "pattern": [
        {"LOWER": {"IN": ["confidential", "proprietary"]}},
        {"LOWER": "information"}
    ]},
    
    # Indemnification
    {"label": "INDEMNIFY", "pattern": [
        {"LEMMA": {"IN": ["indemnify", "hold", "harmless"]}}
    ]},
    
    # Governing law
    {"label": "GOVERNING_LAW", "pattern": [
        {"LOWER": "governed"},
        {"LOWER": "by"},
        {"LOWER": "the"},
        {"LOWER": "laws"},
        {"LOWER": "of"}
    ]},
    
    # Severability
    {"label": "SEVERABILITY", "pattern": [
        {"LOWER": {"IN": ["severability", "severable"]}}
    ]},
]
```

## 📋 Document Type Templates

### Contract Analysis

```python
CONTRACT_PATTERNS = {
    "parties": r'(?:between|by and between)\s+([^,\n]+)\s+(?:and|&)\s+([^,\n]+)',
    "effective_date": r'(?:effective|dated|as of)\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',
    "term": r'(?:term of|for a period of)\s+(\d+)\s+(days?|months?|years?)',
    "termination": r'(?:may terminate|shall terminate|terminat(?:e|ion) upon)',
    "payment": r'\$[\d,]+(?:\.\d{2})?',
    "notice": r'(?:notice|notification)\s+(?:to|shall be)',
}
```

### Compliance Documents

```python
COMPLIANCE_PATTERNS = {
    "requirements": r'(?:must|shall|required to)\s+([^.]+)',
    "prohibitions": r'(?:shall not|must not|prohibited from)\s+([^.]+)',
    "deadlines": r'(?:within|no later than|by)\s+(\d+)\s+(days?|months?|years?)',
    "reporting": r'(?:report|reporting|shall report)\s+([^.]+)',
}
```

### Policy Documents

```python
POLICY_PATTERNS = {
    "scope": r'(?:applies to|applicable to|scope of)\s+([^.]+)',
    "exceptions": r'(?:except|excluding|with the exception of)\s+([^.]+)',
    "responsibilities": r'(?:responsible for|shall be responsible)\s+([^.]+)',
}
```

## 🔧 Customization Guide

### 1. Add New Entity Types

```python
# In legal_graphrag.py, update LEGAL_NODE_TYPES:
LEGAL_NODE_TYPES = {
    # ... existing types ...
    "YourNewType": ["property1", "property2", "property3"],
}
```

### 2. Add New Relationship Types

```python
# Update LEGAL_RELATIONSHIP_TYPES:
LEGAL_RELATIONSHIP_TYPES = [
    # ... existing types ...
    "YOUR_NEW_RELATIONSHIP",
]
```

### 3. Custom Extraction Logic

```python
class CustomLegalNLP(LegalNLPProcessor):
    def extract_custom_entities(self, text: str, page_num: int):
        """Your custom extraction logic"""
        
        # Example: Extract signature blocks
        signature_pattern = r'(?:Signed|Executed)\s+by[^\n]+\n+([^\n]+)\n+([^\n]+)'
        signatures = []
        
        for match in re.finditer(signature_pattern, text):
            signatures.append({
                "signer": match.group(1).strip(),
                "title": match.group(2).strip(),
                "page": page_num
            })
        
        return signatures
```

### 4. Custom Validation Rules

```python
class CustomValidation(ValidationLayer):
    def validate_contract_essentials(self, entities):
        """Ensure contract has essential elements"""
        
        essentials = {
            "parties": len(entities.get("parties", [])) >= 2,
            "consideration": any("payment" in str(e).lower() 
                                 for e in entities.get("amounts", [])),
            "signature": True,  # Add signature detection
        }
        
        score = sum(essentials.values()) / len(essentials)
        return score
```

## 🎨 Query Templates

### Cypher Query Examples

```python
# Find all obligations for a specific party
"""
MATCH (p:Party {name: $party_name})<-[:OBLIGATES]-(o:Obligation)
RETURN o.description, o.deadline, o.page
"""

# Find document structure
"""
MATCH (d:Document)-[:CONTAINS]->(s:Section)-[:CONTAINS]->(c:Clause)
RETURN d.title, s.number, s.title, c.text
ORDER BY s.number
"""

# Find all definitions used in a section
"""
MATCH (s:Section {number: $section_num})-[:REFERENCES]->(d:Definition)
RETURN d.term, d.definition
"""

# Find cross-referenced sections
"""
MATCH (s1:Section)-[:CROSS_REFERENCES]->(s2:Section)
RETURN s1.number, s1.title, s2.number, s2.title
"""

# Trace obligation dependencies
"""
MATCH path = (o:Obligation)-[:DEPENDS_ON*]->(condition)
RETURN path
"""
```

## 📊 Quality Metrics to Monitor

```python
QUALITY_METRICS = {
    "extraction_completeness": {
        "definitions_found": "Count of extracted definitions",
        "citations_found": "Count of legal citations",
        "parties_identified": "Number of parties found",
        "structure_preserved": "Section hierarchy maintained"
    },
    
    "validation_scores": {
        "citation_format": "Valid citation format percentage",
        "definition_quality": "Definition completeness score",
        "relationship_accuracy": "Relationship validation score",
        "overall_confidence": "Final confidence score"
    },
    
    "performance": {
        "pages_per_minute": "Processing speed",
        "llm_api_calls": "Number of API calls made",
        "neo4j_write_time": "Graph write duration"
    }
}
```

## 🚨 Common Issues and Solutions

### Issue: Low Confidence Scores

**Solutions:**
1. Add domain-specific patterns
2. Train custom spaCy model
3. Provide more examples to LLM
4. Review validation logic

```python
# Example: Lower threshold temporarily
graphrag = LegalDocumentGraphRAG(
    ...,
    confidence_threshold=0.95  # Instead of 0.99
)
```

### Issue: Missing Citations

**Solutions:**
1. Add more citation patterns
2. Check OCR quality
3. Verify pattern regex

```python
# Add to patterns:
{"label": "STATUTE", "pattern": [
    {"TEXT": {"REGEX": r"\d+"}},
    {"TEXT": {"REGEX": r"[A-Z]\.?[A-Z]\.?[A-Z]\.?"}},  # ABC, U.S.C., etc.
    {"TEXT": {"REGEX": r"§?\d+"}}
]}
```

### Issue: Incorrect Party Detection

**Solutions:**
1. Train NER on legal documents
2. Add party role patterns
3. Use contextual clues

```python
# Look for party indicators:
party_indicators = [
    "hereinafter referred to as",
    "party of the first part",
    "party of the second part",
    "collectively referred to as"
]
```

## 💡 Best Practices

### 1. Document Preprocessing

```python
def preprocess_legal_document(pdf_path):
    """Clean and prepare document"""
    
    # Remove headers/footers
    # Normalize whitespace
    # Handle multi-column layouts
    # Fix OCR errors
    
    return cleaned_document
```

### 2. Incremental Processing

```python
# For large documents
for page_batch in chunk_pages(total_pages, batch_size=10):
    graphrag.process_legal_pdf(
        pdf_path=path,
        page_range=page_batch
    )
```

### 3. Version Control

```python
document_metadata = {
    "version": "2.1",
    "previous_version": "2.0",
    "amendment_date": "2024-01-15",
    "amended_by": "Amendment No. 3"
}
```

### 4. Audit Trail

```python
# Store extraction metadata
extraction_metadata = {
    "extracted_by": "LegalGraphRAG v1.0",
    "extraction_date": datetime.now().isoformat(),
    "confidence_score": report["validation"]["confidence_score"],
    "review_status": "pending" if score < 0.99 else "approved"
}
```

## 🔗 Integration Examples

### Flask API Endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
graphrag = LegalDocumentGraphRAG(...)

@app.route('/api/query', methods=['POST'])
def query_endpoint():
    user_query = request.json.get('query')
    response = graphrag.query_graph(user_query)
    return jsonify({"response": response})
```

### Batch Processing Script

```python
import glob

for pdf_file in glob.glob("documents/*.pdf"):
    try:
        report = graphrag.process_legal_pdf(pdf_file)
        print(f"✓ {pdf_file}: {report['validation']['confidence_score']:.2%}")
    except Exception as e:
        print(f"✗ {pdf_file}: {e}")
```

## 📈 Performance Optimization

```python
# 1. Batch Neo4j writes
BATCH_SIZE = 100

# 2. Use connection pooling
from neo4j import GraphDatabase
driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_pool_size=50)

# 3. Parallel processing
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_doc, doc) for doc in documents]
```

## 🎓 Training Custom Models

```python
# Train spaCy NER on legal documents
import spacy
from spacy.training import Example

nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

# Add labels
ner.add_label("LEGAL_OBLIGATION")
ner.add_label("LEGAL_RIGHT")

# Train with your annotated data
# ... training code ...
```

---

**Need More Help?**
- Check README.md for detailed documentation
- Review example_usage.py for working examples
- See legal_graphrag.py source code for implementation details
