#!/usr/bin/env python3
"""
Quick test script to verify query generation improvements
Tests a single question without running the full test suite
"""

import os
import json
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize connections
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

openai_client = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Check database status
with driver.session() as session:
    node_count = session.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
    rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']

    print(f"\nDatabase Status:")
    print(f"  Nodes: {node_count}")
    print(f"  Relationships: {rel_count}")

    if node_count == 0:
        print("\nWARNING: Database is EMPTY!")
        driver.close()
        exit(1)
        
        
        

    # Get schema
    node_labels = session.run("CALL db.labels()").data()
    print("\nNode Types:")
    for label in node_labels:
        count = session.run(f"MATCH (n:{label['label']}) RETURN count(n) as count").data()[0]['count']
        print(f"  - {label['label']}: {count} nodes")

    rel_types = session.run("CALL db.relationshipTypes()").data()
    print("\nRelationship Types:")
    for rel in rel_types:
        count = session.run(f"MATCH ()-[r:`{rel['relationshipType']}`]->() RETURN count(r) as count").data()[0]['count']
        print(f"  - {rel['relationshipType']}: {count} relationships")
        
        
        
        

    # Sample nodes
    print("\nSample Nodes:")
    sample_nodes = session.run("""
        MATCH (n)
        RETURN labels(n)[0] as label, properties(n) as props
        LIMIT 5
    """).data()
    for node in sample_nodes:
        print(f"  - {node['label']}: {node['props']}")
        
        
        
        

    # Sample relationships
    print("\nSample Relationships:")
    sample_rels = session.run("""
        MATCH (a)-[r]->(b)
        RETURN labels(a)[0] as from_label,
               type(r) as rel_type,
               labels(b)[0] as to_label,
               properties(a) as from_props,
               properties(b) as to_props
        LIMIT 5
    """).data()
    for rel in sample_rels:
        from_id = rel['from_props'].get('id', rel['from_props'].get('name', 'unknown'))
        to_id = rel['to_props'].get('id', rel['to_props'].get('name', 'unknown'))
        print(f"  ({rel['from_label']}: {from_id}) --[{rel['rel_type']}]--> ({rel['to_label']}: {to_id})")
        
        

# Test query
test_question = "Who are the friends of Mercutio in the play?"

print(f"\nQuestion: {test_question}")

# Generate query
prompt = f"""You are a Neo4j Cypher expert. Based on the schema information provided, generate a Cypher query to answer this question.

Node Labels: {node_labels}
Relationship Types: {rel_types}

Question: {test_question}

Generate a simple, broad query that will find relevant information. Use the property 'id' for matching character names.

Return ONLY a valid Cypher query, no explanation or JSON formatting."""

response = openai_client.invoke(prompt)
cypher_query = response.content.strip()

print("\nCypher Query Generated: ", cypher_query)

# Clean up response
if cypher_query.startswith("```"):
    lines = cypher_query.split("\n")
    cypher_query = "\n".join([l for l in lines if not l.startswith("```")])
    
    


# Execute query
print("\n\nQuery Results:\n")

with driver.session() as session:
    try:
        result = session.run(cypher_query)
        records = [dict(record) for record in result]

        print(f"\nReturned {len(records)} results\n")

        if len(records) > 0:
            for i, record in enumerate(records[:5], 1):
                print(f"{i}. {record}")
        else:
            print("NO RESULTS FOUND")
            print("\nTrying simple fallback query:")
            fallback = "MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 10"
            print(f"  {fallback}")

            result = session.run(fallback)
            fallback_records = [dict(record) for record in result]
            print(f"\nFallback returned {len(fallback_records)} results")

            if len(fallback_records) > 0:
                for i, record in enumerate(fallback_records[:3], 1):
                    print(f"{i}. {record}\n")

    except Exception as e:
        print(f"Query failed: {e}")

driver.close()