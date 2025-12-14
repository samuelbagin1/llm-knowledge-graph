"""
Interactive Query Interface for PDF GraphRAG
"""

from pdf_graphrag_gemini import PDFGraphRAG
import os
from dotenv import load_dotenv

def print_separator():
    print("\n" + "="*70 + "\n")

def display_menu():
    """Display the interactive menu"""
    print_separator()
    print("PDF GRAPHRAG - INTERACTIVE QUERIES")
    print_separator()
    print("Available Queries:")
    print("  1. Show all characters")
    print("  2. Show all locations")
    print("  3. Show all organizations")
    print("  4. Show all events")
    print("  5. Show all concepts/themes")
    print("  6. Show relationships for a specific entity")
    print("  7. Search entities by name")
    print("  8. Find path between two entities")
    print("  9. Show most connected entities")
    print(" 10. Show graph statistics")
    print(" 11. Custom Cypher query")
    print("  0. Exit")
    print_separator()

def query_entities_by_type(graphrag, entity_type: str):
    """Query all entities of a specific type"""
    query = f"""
    MATCH (n:{entity_type})
    RETURN n.name as name, 
           n.description as description
    ORDER BY n.name
    LIMIT 50
    """
    results = graphrag.query_graph(query)
    
    print(f"\n{entity_type.upper()}S:")
    print("-" * 70)
    for r in results:
        desc = r.get('description', 'No description')
        print(f"• {r['name']}")
        if desc and desc != 'No description':
            print(f"  {desc[:150]}...")
    print(f"\nTotal: {len(results)} (showing first 50)")

def query_entity_relationships(graphrag):
    """Query relationships for a specific entity"""
    entity_name = input("\nEnter entity name: ").strip()
    
    relationships = graphrag.get_entity_relationships(entity_name)
    
    if not relationships:
        print(f"\n❌ No entity found named '{entity_name}'")
        
        # Suggest similar entities
        search_results = graphrag.search_entities(entity_name)
        if search_results:
            print("\nDid you mean one of these?")
            for r in search_results[:5]:
                print(f"  • {r['name']} ({r['type'][0]})")
        return
    
    print(f"\nRELATIONSHIPS FOR {entity_name.upper()}:")
    print("-" * 70)
    
    # Group by relationship type
    rel_groups = {}
    for rel in relationships:
        rel_type = rel['relationship_type']
        if rel_type not in rel_groups:
            rel_groups[rel_type] = []
        rel_groups[rel_type].append(rel)
    
    for rel_type, rels in sorted(rel_groups.items()):
        print(f"\n{rel_type}:")
        for rel in rels:
            related_type = rel['related_entity_type'][0] if rel['related_entity_type'] else 'Unknown'
            context = f" - {rel['context']}" if rel.get('context') else ""
            print(f"  → {rel['related_to']} ({related_type}){context}")
    
    print(f"\nTotal relationships: {len(relationships)}")

def search_entities(graphrag):
    """Search for entities by name"""
    search_term = input("\nEnter search term: ").strip()
    
    print("\nFilter by entity type? (leave blank for all)")
    print("  Character, Location, Organization, Event, Concept")
    entity_type = input("Entity type: ").strip() or None
    
    results = graphrag.search_entities(search_term, entity_type)
    
    if not results:
        print(f"\n❌ No entities found matching '{search_term}'")
        return
    
    print(f"\nSEARCH RESULTS FOR '{search_term}':")
    print("-" * 70)
    for r in results:
        entity_type = r['type'][0] if r['type'] else 'Unknown'
        desc = r.get('description', '')
        print(f"\n• {r['name']} ({entity_type})")
        if desc:
            print(f"  {desc[:150]}...")
    
    print(f"\nTotal results: {len(results)}")

def find_path_between_entities(graphrag):
    """Find shortest path between two entities"""
    entity1 = input("\nEnter first entity name: ").strip()
    entity2 = input("Enter second entity name: ").strip()
    
    print(f"\nSearching for path from '{entity1}' to '{entity2}'...")
    
    results = graphrag.find_path_between_entities(entity1, entity2)
    
    if not results:
        print(f"\n❌ No path found between '{entity1}' and '{entity2}'")
        print("\nMake sure both entity names are spelled correctly.")
        return
    
    for result in results:
        path = result['path']
        types = result['types']
        rels = result['relationships']
        
        print(f"\nSHORTEST PATH (length: {result['path_length']}):")
        print("-" * 70)
        
        for i in range(len(path)):
            print(f"\n{i+1}. {path[i]} ({types[i]})")
            if i < len(rels):
                print(f"   ↓ [{rels[i]}]")

def show_most_connected(graphrag):
    """Show most connected entities"""
    try:
        limit = int(input("\nHow many entities to show? [default: 10]: ").strip() or "10")
    except:
        limit = 10
    
    results = graphrag.get_most_connected_entities(limit)
    
    print(f"\nTOP {limit} MOST CONNECTED ENTITIES:")
    print("-" * 70)
    for i, entity in enumerate(results, 1):
        entity_type = entity['type'][0] if entity['type'] else 'Unknown'
        print(f"{i:2d}. {entity['entity']:30s} ({entity_type:15s}) - {entity['connections']} connections")

def show_statistics(graphrag):
    """Show graph statistics"""
    stats = graphrag.get_graph_summary()
    
    print("\nGRAPH STATISTICS:")
    print("-" * 70)
    print(f"Characters:    {stats.get('characters', 0)}")
    print(f"Locations:     {stats.get('locations', 0)}")
    print(f"Organizations: {stats.get('organizations', 0)}")
    print(f"Events:        {stats.get('events', 0)}")
    print(f"Concepts:      {stats.get('concepts', 0)}")
    print(f"Relationships: {stats.get('relationships', 0)}")
    
    print("\nTop Relationship Types:")
    for rel in stats.get('top_relationships', [])[:10]:
        print(f"  {rel['rel_type']:25s}: {rel['count']}")

def custom_query(graphrag):
    """Execute a custom Cypher query"""
    print("\nEnter your Cypher query (or 'back' to return):")
    print("Example: MATCH (c:Character)-[r]->(o) RETURN c.name, type(r), o.name LIMIT 10")
    query = input("\n> ").strip()
    
    if query.lower() == 'back':
        return
    
    try:
        results = graphrag.query_graph(query)
        print("\nQUERY RESULTS:")
        print("-" * 70)
        
        if not results:
            print("No results returned")
        else:
            # Print first 50 results
            for i, r in enumerate(results[:50], 1):
                print(f"{i}. {r}")
            
            if len(results) > 50:
                print(f"\n... and {len(results) - 50} more results")
        
        print(f"\nTotal results: {len(results)}")
    except Exception as e:
        print(f"\n❌ Error executing query: {e}")

def main():
    """Main interactive loop"""
    load_dotenv()
    
    # Initialize GraphRAG
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    if not NEO4J_PASSWORD or not GOOGLE_API_KEY:
        print("❌ Error: Please set NEO4J_PASSWORD and GOOGLE_API_KEY in your .env file")
        return
    
    print("Connecting to Neo4j...")
    graphrag = PDFGraphRAG(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        google_api_key=GOOGLE_API_KEY
    )
    
    print("✓ Connected successfully!")
    
    # Main loop
    while True:
        display_menu()
        choice = input("Enter your choice (0-11): ").strip()
        
        try:
            if choice == '1':
                query_entities_by_type(graphrag, "Character")
            elif choice == '2':
                query_entities_by_type(graphrag, "Location")
            elif choice == '3':
                query_entities_by_type(graphrag, "Organization")
            elif choice == '4':
                query_entities_by_type(graphrag, "Event")
            elif choice == '5':
                query_entities_by_type(graphrag, "Concept")
            elif choice == '6':
                query_entity_relationships(graphrag)
            elif choice == '7':
                search_entities(graphrag)
            elif choice == '8':
                find_path_between_entities(graphrag)
            elif choice == '9':
                show_most_connected(graphrag)
            elif choice == '10':
                show_statistics(graphrag)
            elif choice == '11':
                custom_query(graphrag)
            elif choice == '0':
                print("\nGoodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please try again.")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
