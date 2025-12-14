import os
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


class PDFGraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, google_api_key: str = None):
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )

        if google_api_key:
            os.environ["GOOGLE_API_KEY"] = google_api_key

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2-flash",
            temperature=0,
            convert_system_message_to_human=True
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_pdf(self, pdf_path: str) -> str:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        return "\n\n".join([page.page_content for page in pages])

    def setup_graph_schema(self):
        schema_queries = [
            "CREATE CONSTRAINT character_name IF NOT EXISTS FOR (c:Character) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
            "CREATE CONSTRAINT organization_name IF NOT EXISTS FOR (o:Organization) REQUIRE o.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (co:Concept) REQUIRE co.name IS UNIQUE",
            "CREATE CONSTRAINT event_name IF NOT EXISTS FOR (e:Event) REQUIRE e.name IS UNIQUE",
            "CREATE INDEX character_name_idx IF NOT EXISTS FOR (c:Character) ON (c.name)",
            "CREATE INDEX location_name_idx IF NOT EXISTS FOR (l:Location) ON (l.name)",
            "CREATE INDEX organization_name_idx IF NOT EXISTS FOR (o:Organization) ON (o.name)"
        ]

        for query in schema_queries:
            try:
                self.graph.query(query)
            except:
                pass

    def extract_entities_from_chunk(self, text_chunk: str) -> Dict[str, Any]:
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("human", """Extract named entities and relationships from the text.

Return JSON with this structure:
{{
    "characters": [{{"name": "character name", "description": "brief description", "affiliation": "organization if mentioned"}}],
    "locations": [{{"name": "location name", "description": "brief description"}}],
    "organizations": [{{"name": "organization name", "description": "brief description"}}],
    "events": [{{"name": "event name", "description": "brief description"}}],
    "concepts": [{{"name": "concept/theme name", "description": "brief description"}}],
    "relationships": [{{
        "source": "entity name",
        "target": "entity name",
        "type": "relationship type (KNOWS, WORKS_FOR, LOCATED_IN, PARTICIPATES_IN, RELATED_TO, LOVES, HATES, LEADS, FOLLOWS)",
        "description": "context"
    }}]
}}

Guidelines:
- Extract only explicitly mentioned entities
- Use specific names
- Use clear relationship types
- Keep descriptions concise

Text: {text}

Respond with valid JSON only.""")
        ])

        chain = extraction_prompt | self.llm | JsonOutputParser()

        try:
            return chain.invoke({"text": text_chunk})
        except:
            return {
                "characters": [], "locations": [], "organizations": [],
                "events": [], "concepts": [], "relationships": []
            }

    def add_entities_to_graph(self, entities: Dict[str, Any]):
        for char in entities.get("characters", []):
            self.graph.query("""
                MERGE (c:Character {name: $name})
                ON CREATE SET c.description = $description
                ON MATCH SET c.description = CASE
                    WHEN c.description IS NULL THEN $description
                    WHEN length($description) > length(c.description) THEN $description
                    ELSE c.description
                END
            """, {"name": char["name"], "description": char.get("description", "")})

            if char.get("affiliation"):
                self.graph.query("""
                    MERGE (o:Organization {name: $org_name})
                    WITH o
                    MATCH (c:Character {name: $char_name})
                    MERGE (c)-[:AFFILIATED_WITH]->(o)
                """, {"org_name": char["affiliation"], "char_name": char["name"]})

        for loc in entities.get("locations", []):
            self.graph.query("""
                MERGE (l:Location {name: $name})
                ON CREATE SET l.description = $description
                ON MATCH SET l.description = CASE
                    WHEN l.description IS NULL THEN $description
                    WHEN length($description) > length(l.description) THEN $description
                    ELSE l.description
                END
            """, {"name": loc["name"], "description": loc.get("description", "")})

        for org in entities.get("organizations", []):
            self.graph.query("""
                MERGE (o:Organization {name: $name})
                ON CREATE SET o.description = $description
                ON MATCH SET o.description = CASE
                    WHEN o.description IS NULL THEN $description
                    WHEN length($description) > length(o.description) THEN $description
                    ELSE o.description
                END
            """, {"name": org["name"], "description": org.get("description", "")})

        for event in entities.get("events", []):
            self.graph.query("""
                MERGE (e:Event {name: $name})
                ON CREATE SET e.description = $description
                ON MATCH SET e.description = CASE
                    WHEN e.description IS NULL THEN $description
                    WHEN length($description) > length(e.description) THEN $description
                    ELSE e.description
                END
            """, {"name": event["name"], "description": event.get("description", "")})

        for concept in entities.get("concepts", []):
            self.graph.query("""
                MERGE (co:Concept {name: $name})
                ON CREATE SET co.description = $description
                ON MATCH SET co.description = CASE
                    WHEN co.description IS NULL THEN $description
                    WHEN length($description) > length(co.description) THEN $description
                    ELSE co.description
                END
            """, {"name": concept["name"], "description": concept.get("description", "")})

        for rel in entities.get("relationships", []):
            rel_type = rel["type"].upper().replace(" ", "_").replace("-", "_")
            try:
                self.graph.query(f"""
                    MATCH (source) WHERE source.name = $source_name
                    MATCH (target) WHERE target.name = $target_name
                    MERGE (source)-[r:{rel_type}]->(target)
                    ON CREATE SET r.description = $description
                    ON MATCH SET r.description = CASE
                        WHEN r.description IS NULL THEN $description
                        ELSE r.description
                    END
                """, {
                    "source_name": rel["source"],
                    "target_name": rel["target"],
                    "description": rel.get("description", "")
                })
            except:
                pass

    def process_pdf(self, pdf_path: str, max_chunks: int = None):
        text = self.load_pdf(pdf_path)
        chunks = self.text_splitter.split_text(text)

        if max_chunks:
            chunks = chunks[:max_chunks]

        for chunk in chunks:
            entities = self.extract_entities_from_chunk(chunk)
            self.add_entities_to_graph(entities)


def main():
    load_dotenv()

    graphrag = PDFGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    graphrag.setup_graph_schema()
    graphrag.process_pdf("code/romeo-juliet/pdf/romeo-and-juliet.pdf")


if __name__ == "__main__":
    main()
