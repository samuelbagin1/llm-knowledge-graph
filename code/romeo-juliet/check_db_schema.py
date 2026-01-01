#!/usr/bin/env python3
"""Quick script to check Neo4j database schema and sample data"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)

print("="*70)
print("NEO4J DATABASE INSPECTION")
print("="*70)

with driver.session() as session:
    # Check node labels
    labels = session.run("CALL db.labels()").data()
    print("\n=== Node Labels ===")
    for label in labels:
        count = session.run(f"MATCH (n:`{label['label']}`) RETURN count(n) as count").data()[0]['count']
        print(f"  - {label['label']}: {count} nodes")

    print("\n=== Relationship Types ===")
    rels = session.run("CALL db.relationshipTypes()").data()
    for rel in rels:
        count = session.run(f"MATCH ()-[r:`{rel['relationshipType']}`]->() RETURN count(r) as count").data()[0]['count']
        print(f"  - {rel['relationshipType']}: {count} relationships")

    print("\n=== Sample Nodes (first 15) ===")
    nodes = session.run("MATCH (n) RETURN labels(n) as labels, properties(n) as props LIMIT 15").data()
    for i, node in enumerate(nodes, 1):
        print(f"\n{i}. Labels: {node['labels']}")
        print(f"   Properties: {node['props']}")

    print("\n=== Sample Relationships (first 15) ===")
    rels = session.run("""
        MATCH (a)-[r]->(b)
        RETURN labels(a)[0] as from_label,
               a.id as from_id,
               type(r) as rel_type,
               labels(b)[0] as to_label,
               b.id as to_id
        LIMIT 15
    """).data()
    for i, rel in enumerate(rels, 1):
        print(f"{i}. ({rel['from_label']}: {rel['from_id']}) --[{rel['rel_type']}]--> ({rel['to_label']}: {rel['to_id']})")

driver.close()

print("\n" + "="*70)
