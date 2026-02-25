#!/bin/bash

# Legal Document GraphRAG Setup Script

echo "=== Legal Document GraphRAG Setup ==="
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --break-system-packages -r requirements.txt

# Download spaCy models
echo ""
echo "Downloading spaCy models..."
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_sm

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Create a .env file with your credentials:"
echo "   NEO4J_URI=bolt://localhost:7687"
echo "   NEO4J_USER=neo4j"
echo "   NEO4J_PASSWORD=your_password"
echo "   OPENAI_API_KEY=your_openai_key"
echo "   GOOGLE_API_KEY=your_google_key (optional)"
echo ""
echo "2. Ensure Neo4j is running with APOC plugin installed"
echo "3. Run: python legal_graphrag.py"
