# PDF GraphRAG with Google Gemini

A comprehensive Graph-based Retrieval Augmented Generation (GraphRAG) system for Named Entity Recognition (NER) on PDF books using Python, LangChain, Google Gemini, and Neo4j.

## 🎯 Features

- **PDF Input**: Process any PDF book or document
- **Google Gemini AI**: Uses Gemini 1.5 Flash for intelligent entity extraction
- **Rich Entity Types**: Characters, Locations, Organizations, Events, Concepts
- **Automatic Relationship Detection**: Discovers connections between entities
- **Knowledge Graph**: Builds a Neo4j graph database
- **Interactive Queries**: Pre-built query interface
- **Entity Search**: Find entities by name or type
- **Path Finding**: Discover connections between entities
- **Visualization Ready**: Export to multiple formats

## 📊 System Architecture

```
PDF Document
    ↓
PDF Text Extraction (PyPDF)
    ↓
Text Chunking (LangChain)
    ↓
Entity Recognition (Google Gemini)
    ├── Characters
    ├── Locations
    ├── Organizations
    ├── Events
    └── Concepts
    ↓
Relationship Extraction
    ↓
Neo4j Knowledge Graph
    ↓
Query & Analysis Interface
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+**
2. **Neo4j Database** 
   - [Neo4j Desktop](https://neo4j.com/download/) (recommended)
   - Or [Neo4j AuraDB](https://neo4j.com/cloud/aura/) (cloud)
3. **Google API Key**
   - Get from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt
```

Your `.env` file should contain:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GOOGLE_API_KEY=your_google_api_key
```

### Usage

#### 1. Process Your PDF Book

```bash
python pdf_graphrag.py
```

The script will:
1. Ask for your PDF file path
2. Ask how many chunks to process:
   - **Quick demo** (20 chunks, ~5-10 minutes)
   - **Balanced** (50 chunks, ~15-25 minutes)
   - **Complete** (all chunks, ~30-60 minutes)
3. Extract entities and build the knowledge graph
4. Display statistics

#### 2. Query the Graph

```bash
python query_pdf_graph.py
```

Available queries:
1. **Show all characters** - List all people/characters found
2. **Show all locations** - List all places mentioned
3. **Show all organizations** - List all groups/organizations
4. **Show all events** - List all events described
5. **Show all concepts** - List themes and concepts
6. **Entity relationships** - See all connections for a specific entity
7. **Search entities** - Find entities by name
8. **Find paths** - Discover connections between two entities
9. **Most connected** - See hub entities in the graph
10. **Statistics** - View graph metrics
11. **Custom query** - Run your own Cypher queries

#### 3. Explore in Neo4j Browser

Open Neo4j Browser and try these queries:

```cypher
// View all characters and their connections
MATCH (c:Character)-[r]-(other)
RETURN c, r, other
LIMIT 50

// Find all organizations
MATCH (o:Organization)
RETURN o

// Find paths between two characters
MATCH path = shortestPath(
  (e1 {name: 'Character1'})-[*]-(e2 {name: 'Character2'})
)
RETURN path
```



## 📊 Example Queries

### Python API

```python
from pdf_graphrag_gemini import PDFGraphRAG
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize
graphrag = PDFGraphRAG(
    neo4j_uri=os.getenv("NEO4J_URI"),
    neo4j_user=os.getenv("NEO4J_USER"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Get character relationships
relationships = graphrag.get_entity_relationships("Harry Potter")

# Search for entities
results = graphrag.search_entities("London", entity_type="Location")

# Find path between entities
path = graphrag.find_path_between_entities("Harry", "Voldemort")

# Get most connected entities
connected = graphrag.get_most_connected_entities(10)
```

### Cypher Queries

```cypher
// All characters in London
MATCH (c:Character)-[:LOCATED_IN]->(l:Location {name: 'London'})
RETURN c.name

// Organizations and their members
MATCH (c:Character)-[:WORKS_FOR]->(o:Organization)
RETURN o.name as organization, collect(c.name) as members

// Most connected entities
MATCH (n)-[r]-()
RETURN n.name, labels(n), count(r) as connections
ORDER BY connections DESC
LIMIT 10

// Find communities (characters who work together)
MATCH (c1:Character)-[:WORKS_FOR]->(o:Organization)<-[:WORKS_FOR]-(c2:Character)
WHERE c1 <> c2
RETURN c1.name, o.name, c2.name
```

## 🎨 Graph Schema

```
(Character)-[:KNOWS]->(Character)
(Character)-[:WORKS_FOR]->(Organization)
(Character)-[:LOCATED_IN]->(Location)
(Character)-[:PARTICIPATES_IN]->(Event)
(Character)-[:AFFILIATED_WITH]->(Organization)
(Event)-[:OCCURS_IN]->(Location)
(Organization)-[:LOCATED_IN]->(Location)
(Concept)-[:RELATED_TO]->(Concept)
```



### Custom Entity Types

Modify the extraction prompt in `extract_entities_from_chunk()` to add new entity types.


### Neo4j Connection

```bash
# Test connection
python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password')); driver.verify_connectivity(); print('✓ Connected!')"
```

## 📈 Performance

- **Small PDF** (50 pages): ~5-10 minutes
- **Medium PDF** (200 pages): ~15-30 minutes  
- **Large PDF** (500+ pages): ~45-90 minutes

Processing time depends on:
- PDF length and complexity
- Number of entities per page
- API rate limits
- Internet speed

## 🌟 Advanced Features

### Batch Processing

Process multiple PDFs:
```python
pdfs = ["book1.pdf", "book2.pdf", "book3.pdf"]
for pdf in pdfs:
    graphrag.process_pdf(pdf)
```

### Export Graph Data

```python
# Export for visualization
stats = graphrag.get_graph_summary()
connected = graphrag.get_most_connected_entities(100)
```

### Custom Relationship Extraction

Modify the prompt in `extract_entities_from_chunk()` to focus on specific relationship types relevant to your domain.


## Acknowledgments

- **Google Gemini** - AI model for NER
- **Neo4j** - Graph database
- **LangChain** - AI framework
- **PyPDF** - PDF processing

## 🔗 Resources

- [Google AI Studio](https://makersuite.google.com/)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [LangChain Docs](https://python.langchain.com/)
- [GraphRAG Research](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)
