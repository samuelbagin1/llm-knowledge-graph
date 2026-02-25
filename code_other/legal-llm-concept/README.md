# Legal Document GraphRAG System

High-precision Knowledge Graph extraction system for legal documents with 99% accuracy target.

## 🎯 Key Features

- **Hybrid Extraction**: Combines rule-based (spaCy) and LLM-based extraction
- **Multi-Layer Validation**: 3-layer validation with confidence scoring
- **Legal-Specific NLP**: Custom patterns for citations, statutes, obligations
- **Provenance Tracking**: Every entity linked to source page/section
- **Structure Preservation**: Maintains document hierarchy and cross-references
- **Query Interface**: Natural language querying with legal context

## 📋 Prerequisites

1. **Neo4j Database** (v5.0+)
   - Install Neo4j Desktop or use Neo4j AuraDB
   - Install APOC plugin (recommended for performance)
   - Start your Neo4j instance

2. **Python** (v3.9+)

3. **API Keys**
   - OpenAI API key (for GPT-4) OR
   - Google API key (for Gemini)

## 🚀 Installation

### Step 1: Clone or Download Files

Ensure you have these files:
- `legal_graphrag.py`
- `requirements.txt`
- `setup.sh`
- `.env.example`

### Step 2: Install Dependencies

**On Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows or Manual Installation:**
```bash
# Install Python packages
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm
```

### Step 3: Configure Environment

Create a `.env` file from the example:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

OPENAI_API_KEY=sk-your-key-here
# OR
GOOGLE_API_KEY=your-google-key
```

### Step 4: Verify Neo4j Setup

1. Start Neo4j
2. Open Neo4j Browser (http://localhost:7474)
3. Verify connection with: `RETURN "Connected!" as status`
4. Check APOC: `RETURN apoc.version()`

## 📖 Usage

### Basic Usage

```python
from legal_graphrag import LegalDocumentGraphRAG
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize system
graphrag = LegalDocumentGraphRAG(
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_user=os.getenv("NEO4J_USER"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
)

# Process a legal document
document_metadata = {
    "title": "Employment Agreement",
    "date": "2024-01-01",
    "jurisdiction": "California",
    "doc_type": "Contract"
}

report = graphrag.process_legal_pdf(
    pdf_path="path/to/legal_document.pdf",
    document_metadata=document_metadata,
    max_pages=None  # Process all pages
)

# Query the knowledge graph
response = graphrag.query_graph("What are the termination conditions?")
print(response)
```

### Command Line Usage

Edit the `main()` function in `legal_graphrag.py`:

```python
def main():
    load_dotenv()
    
    graphrag = LegalDocumentGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
    )
    
    # YOUR DOCUMENT PATH HERE
    pdf_path = "contracts/employment_agreement.pdf"
    
    document_metadata = {
        "title": "Employment Agreement",
        "date": "2024-01-01",
        "jurisdiction": "California"
    }
    
    report = graphrag.process_legal_pdf(
        pdf_path=pdf_path,
        document_metadata=document_metadata
    )
```

Then run:
```bash
python legal_graphrag.py
```

## 🏗️ Architecture

### Three-Layer Extraction Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Legal PDF                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Rule-Based Extraction (spaCy)                     │
│  • Legal citations (USC, CFR, case law)                     │
│  • Defined terms and definitions                            │
│  • Obligations (shall/must)                                 │
│  • Section headers and hierarchy                            │
│  • Parties, dates, amounts                                  │
│  ✓ Speed: Fast | Precision: 95%+                            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Semantic Extraction (LLM)                         │
│  • Complex relationships                                    │
│  • Conditional logic                                        │
│  • Implicit references                                      │
│  • Context-dependent meanings                               │
│  ✓ Understanding: Deep | Precision: Variable                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: Validation & Reconciliation                       │
│  • Cross-validate both layers                               │
│  • Confidence scoring                                       │
│  • Deduplication                                            │
│  • Quality checks                                           │
│  ✓ Target: 99% precision                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Neo4j Knowledge Graph                          │
│              with Provenance Tracking                       │
└─────────────────────────────────────────────────────────────┘
```

### Extracted Entity Types

**Document Structure:**
- Document, Section, Clause

**Legal Entities:**
- Definition, Party, Statute, Caselaw, Obligation, Right, Condition

**Metadata:**
- Date, Amount, Citation

**Relationships:**
- CITES, DEFINES, REFERENCES, OBLIGATES, GRANTS, DEPENDS_ON, etc.

## 🔍 Query Examples

```python
# Definition lookup
response = graphrag.query_graph("What is the definition of 'Effective Date'?")

# Obligation search
response = graphrag.query_graph("What are the seller's obligations?")

# Citation retrieval
response = graphrag.query_graph("What statutes are cited?")

# Party identification
response = graphrag.query_graph("Who are the parties to this agreement?")
```

## 📊 Quality Metrics

The system provides a comprehensive quality report:

```
📊 QUALITY REPORT
----------------------------------------------------------------------
Pages processed: 25

Structured entities extracted:
  - definitions: 47
  - citations: 23
  - obligations: 89
  - parties: 4
  - dates: 15
  - amounts: 12
  - sections: 18

Semantic extraction:
  - Nodes: 245
  - Relationships: 412

Validation:
  - Confidence: 98.50%
  - Issues: 2
```

## 🎛️ Configuration Options

### Confidence Threshold

```python
graphrag = LegalDocumentGraphRAG(
    ...,
    confidence_threshold=0.99  # Default: 0.99 (99%)
)
```

Lower threshold accepts more extractions but may reduce precision.

### LLM Provider

```python
# Use OpenAI (default)
graphrag = LegalDocumentGraphRAG(..., ai_provider=None)

# Use Google Gemini
graphrag = LegalDocumentGraphRAG(..., ai_provider="gemini")
```

### Page Limit (Testing)

```python
# Process only first 5 pages (for testing)
report = graphrag.process_legal_pdf(
    pdf_path="document.pdf",
    max_pages=5
)
```

## 🔧 Advanced Features

### Custom spaCy Patterns

Add domain-specific patterns in `LegalNLPProcessor._setup_legal_patterns()`:

```python
patterns = [
    {"label": "YOUR_LABEL", "pattern": [
        {"TEXT": {"REGEX": r"your_pattern"}}
    ]},
]
```

### Custom Validation Rules

Extend `ValidationLayer` class:

```python
def _custom_validation(self, entities):
    # Your validation logic
    return score
```

### Custom Query Types

Add to `QueryType` enum and implement in `_generate_legal_cypher()`.

## 🐛 Troubleshooting

### APOC Plugin Not Found

If you see "APOC not available", the system automatically falls back to a slower method. To enable APOC:

1. Neo4j Desktop: Add APOC plugin in Settings
2. Neo4j Server: Add to `neo4j.conf`:
   ```
   dbms.security.procedures.unrestricted=apoc.*
   ```

### spaCy Model Download Fails

```bash
# Try alternative download method
python -m spacy download en_core_web_sm
# Or download manually from https://github.com/explosion/spacy-models/releases
```

### Low Confidence Scores

- Check if document is truly legal (not scanned/OCR errors)
- Review validation_issues in quality report
- Consider domain-specific training of spaCy model
- Adjust confidence threshold if acceptable

### Memory Issues

For large documents (100+ pages):

```python
# Process in chunks
for chunk in range(0, total_pages, 10):
    report = graphrag.process_legal_pdf(
        pdf_path="large_doc.pdf",
        max_pages=chunk+10
    )
```

## 🚦 Production Considerations

### For 99% Precision in Production

1. **Human-in-the-Loop Review**
   - Flag extractions below confidence threshold
   - Implement review queue
   - Collect feedback for continuous improvement

2. **Domain-Specific Training**
   - Train spaCy on your specific legal domain
   - Fine-tune LLM with domain examples
   - Build pattern library from reviewed documents

3. **Version Control**
   - Track document versions in Neo4j
   - Store amendment relationships
   - Maintain audit trail

4. **Performance Optimization**
   - Enable Neo4j APOC plugin
   - Use connection pooling
   - Implement caching for repeated queries
   - Batch process multiple documents

5. **Monitoring**
   - Log confidence scores
   - Track extraction patterns
   - Monitor query performance
   - Alert on low-quality extractions

## 📚 Next Steps

1. **Web Interface**: Build Flask/FastAPI backend with Neo4j queries
2. **Chat Interface**: Integrate with LLM for conversational queries
3. **Document Comparison**: Compare entities across multiple documents
4. **Visualization**: Create graph visualizations with D3.js or Cytoscape
5. **Export**: Generate reports from graph data

## 📄 License

This is example code for educational purposes. Adapt as needed for your use case.

## ⚠️ Legal Disclaimer

This tool is for information extraction and analysis only. Always consult qualified legal professionals for legal advice. The creators assume no liability for decisions made based on this tool's output.

## 🤝 Contributing

To improve precision:
1. Add legal-specific patterns to spaCy configuration
2. Enhance validation rules
3. Expand entity and relationship types
4. Share anonymized quality reports

## 📧 Support

For issues or questions, please refer to:
- Neo4j documentation: https://neo4j.com/docs/
- spaCy documentation: https://spacy.io/
- LangChain documentation: https://python.langchain.com/
