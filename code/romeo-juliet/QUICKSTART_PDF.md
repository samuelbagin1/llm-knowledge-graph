# 🚀 Quick Start Guide - PDF GraphRAG with Google Gemini

Get up and running in 5 minutes!

## Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

Add your credentials:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
GOOGLE_API_KEY=your_google_api_key
```

### Get Google API Key
1. Go to: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key to your `.env` file

### Get Neo4j Running
**Option A: Neo4j Desktop (Recommended)**
1. Download: https://neo4j.com/download/
2. Create new database
3. Set password
4. Start database

**Option B: Neo4j AuraDB (Cloud)**
1. Go to: https://neo4j.com/cloud/aura/
2. Create free instance
3. Copy connection details

## Step 3: Process Your PDF (1 minute to start)

```bash
python pdf_graphrag.py
```

Follow the prompts:
1. Enter path to your PDF file
2. Choose processing option (start with option 2 for quick demo)
3. Wait for processing to complete

## Step 4: Query Your Graph (Now!)

```bash
python query_pdf_graph.py
```

Try these queries:
- Option 1: See all characters
- Option 6: Explore entity relationships
- Option 8: Find connections between entities

## 🎯 Example Session

```bash
$ python pdf_graphrag.py

Enter the path to your PDF file:
PDF path: /path/to/harry_potter.pdf

✓ Found PDF: /path/to/harry_potter.pdf

PROCESSING OPTIONS
How many chunks would you like to process?
  1. Process all chunks (complete analysis)
  2. Process first 20 chunks (quick demo, ~5-10 minutes)
  3. Process first 50 chunks (balanced, ~15-25 minutes)

Enter your choice (1/2/3) [default: 2]: 2

Processing first 20 chunks (demo mode)...
Processing chunk 1/20... [5 entities, 3 relationships]
Processing chunk 2/20... [8 entities, 6 relationships]
...

PROCESSING COMPLETE!

📊 Graph Statistics:
  Characters:    45
  Locations:     12
  Organizations: 8
  Events:        5
  Concepts:      7
  Relationships: 127

⭐ Most Connected Entities:
  Harry Potter (Character): 23 connections
  Hogwarts (Location): 18 connections
  Ron Weasley (Character): 15 connections
```

## 📝 What's Happening?

1. **PDF Loading**: Extracts text from your PDF
2. **Text Chunking**: Splits into manageable pieces
3. **AI Analysis**: Google Gemini extracts entities and relationships
4. **Graph Building**: Creates Neo4j knowledge graph
5. **Ready to Query**: Your knowledge graph is ready!

## 🎨 Visualize in Neo4j Browser

1. Open Neo4j Browser (usually http://localhost:7474)
2. Run this query:

```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 50
```

3. Click the graph icon to see visual representation!

## 💡 Pro Tips

### Faster Processing
```python
# Edit pdf_graphrag_gemini.py, line ~80
model="gemini-1.5-flash",  # Already set - fastest option
```

### Better Quality
```python
# Use Pro model for better extraction
model="gemini-1.5-pro",  # Higher quality, slower
```

### Process More Content
Choose option 3 (50 chunks) or option 1 (all chunks) when processing.

## 🐛 Troubleshooting

### "GOOGLE_API_KEY not set"
→ Check your `.env` file exists and has the API key

### "NEO4J_PASSWORD not set"  
→ Add Neo4j password to `.env` file

### "File not found"
→ Use absolute path: `/full/path/to/file.pdf`

### "Error loading PDF"
→ Ensure PDF is readable (not password-protected)

### Neo4j won't connect
→ Make sure Neo4j is running (check Neo4j Desktop)
