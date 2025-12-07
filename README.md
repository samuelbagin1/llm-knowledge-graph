# AI Knowledge Graph for Legal Texts

Automatická extrakcia a prepojenie informácií z právnych textov do interaktívneho znalostného grafu.

## 📋 Obsah

- [Prehľad](#prehľad)
- [Architektúra](#architektúra)
- [Technologický Stack](#technologický-stack)
- [Inštalácia](#inštalácia)
- [Implementácia](#implementácia)
- [Komponenty](#komponenty)
- [Študijné Materiály](#študijné-materiály)
- [MVP Prototyp](#mvp-prototyp)
- [Roadmap](#roadmap)

## 🎯 Prehľad

Cieľom projektu je vytvoriť AI systém na automatickú extrakciu a prepojenie informácií z právnych textov do znalostného grafu. Výsledkom je interaktívny graf znázorňujúcy väzby medzi právnymi pojmami a ustanoveniami v zákone.

### Hlavné funkcie

- 🔍 Automatická extrakcia právnych entít (paragrafy, pojmy, inštitúcie)
- 🔗 Identifikácia vzťahov medzi právnymi ustanoveniami
- 📊 Vizualizácia znalostného grafu
- 🔎 Interaktívne prehľadávanie a explorácia
- 💬 Sémantické vyhľadávanie a Q&A

## 🏗️ Architektúra

### Tri-fázový proces

```
┌─────────────────────────────────────────────────────────────┐
│ FÁZA 1: Spracovanie a Extrakcia                             │
├─────────────────────────────────────────────────────────────┤
│ Právny text → Preprocessing → NER & Relation Extraction     │
│                              → Štruktúrované dáta            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ FÁZA 2: Konštrukcia Grafu                                   │
├─────────────────────────────────────────────────────────────┤
│ Štruktúrované dáta → Graph Building                         │
│                    → Knowledge Graph (Neo4j/NetworkX)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ FÁZA 3: Vizualizácia a Interakcia                           │
├─────────────────────────────────────────────────────────────┤
│ Knowledge Graph → Query & Visualization                     │
│                 → Interaktívne rozhranie                     │
└─────────────────────────────────────────────────────────────┘
```

## 🛠️ Technologický Stack

### Core Frameworky

| Komponent | Technológia | Účel |
|-----------|-------------|------|
| **AI Orchestrácia** | LangGraph + LangChain | Workflow management, multi-step processing |
| **LLM** | OpenAI API / Anthropic Claude | Entity & relation extraction |
| **NLP** | Spacy (`sk_core_news_lg`) | Preprocessing, tokenizácia, NER |
| **Graph DB** | Neo4j | Úložisko znalostného grafu |
| **Vizualizácia** | Pyvis, Plotly, D3.js | Interaktívne grafy |
| **Web Framework** | FastAPI / Dash | API a dashboard |

### Doplnkové nástroje

- **Hugging Face Transformers** - Fine-tuning modelov
- **NetworkX** - Graph manipulation v Pythone
- **ArangoDB / Amazon Neptune** - Alternatívne graph DB
- **Docker** - Kontajnerizácia

## 📦 Inštalácia

### Požiadavky

```bash
Python 3.9+
Neo4j 5.0+
```

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/legal-knowledge-graph.git
cd legal-knowledge-graph

# Vytvorenie virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Inštalácia závislostí
pip install -r requirements.txt

# Download Spacy Slovak model
python -m spacy download sk_core_news_lg

# Setup Neo4j (Docker)
docker run -d \
  --name neo4j-legal \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### requirements.txt

```txt
langchain>=0.1.0
langgraph>=0.0.20
openai>=1.0.0
anthropic>=0.8.0
spacy>=3.7.0
neo4j>=5.14.0
networkx>=3.2
pyvis>=0.3.2
plotly>=5.18.0
dash>=2.14.0
fastapi>=0.104.0
uvicorn>=0.24.0
python-dotenv>=1.0.0
pydantic>=2.5.0
```

## 💻 Implementácia

### Krok 1: Data Pipeline

```python
# legal_processor.py

from typing import List, Dict
import spacy
from dataclasses import dataclass

@dataclass
class Entity:
    text: str
    type: str
    context: str
    span: tuple

@dataclass
class Relation:
    source: str
    target: str
    type: str
    confidence: float

class LegalDocumentProcessor:
    def __init__(self):
        self.nlp = spacy.load("sk_core_news_lg")
    
    def preprocess(self, text: str) -> Dict:
        """
        Čistenie a segmentácia právneho textu
        - Rozpoznanie §, odsekov, bodov
        - Normalizácia textu
        """
        # Implementácia preprocessing logiky
        pass
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        NER: právne pojmy, inštitúcie, § čísla
        Využitie LLM-based extraction
        """
        # Implementácia entity extraction
        pass
    
    def extract_relations(self, entities: List[Entity], 
                         context: str) -> List[Relation]:
        """
        Extrakcia vzťahov medzi entitami
        Typy: odkazuje_na, definuje, upravuje
        """
        # Implementácia relation extraction
        pass
```

### Krok 2: LangGraph Workflow

```python
# workflow.py

from langgraph.graph import StateGraph
from typing import TypedDict, List
import networkx as nx

class GraphState(TypedDict):
    document: str
    entities: List[Entity]
    relations: List[Relation]
    graph: nx.Graph
    metadata: Dict

def parse_legal_document(state: GraphState) -> GraphState:
    """Parse a segmentuj právny dokument"""
    # Implementácia
    return state

def entity_extraction(state: GraphState) -> GraphState:
    """Extrahuj právne entity"""
    # Implementácia
    return state

def relation_extraction(state: GraphState) -> GraphState:
    """Extrahuj vzťahy medzi entitami"""
    # Implementácia
    return state

def construct_knowledge_graph(state: GraphState) -> GraphState:
    """Vybuduj znalostný graf"""
    # Implementácia
    return state

# Definícia workflow
workflow = StateGraph(GraphState)

# Pridanie nodov
workflow.add_node("parse", parse_legal_document)
workflow.add_node("extract_entities", entity_extraction)
workflow.add_node("extract_relations", relation_extraction)
workflow.add_node("build_graph", construct_knowledge_graph)

# Pridanie edges
workflow.add_edge("parse", "extract_entities")
workflow.add_edge("extract_entities", "extract_relations")
workflow.add_edge("extract_relations", "build_graph")

# Compile
app = workflow.compile()
```

### Krok 3: Prompt Engineering

```python
# prompts.py

ENTITY_EXTRACTION_PROMPT = """
Analyzuj tento slovenský právny text a identifikuj:

1. **Právne pojmy** (napr. "zmluva", "náhrada škody", "zodpovednosť")
2. **Odkazy na paragrafy** (§ X, § Y ods. Z)
3. **Právne inštitúcie** (súd, orgán, komisia)
4. **Subjekty práva** (fyzická osoba, právnická osoba)

Text: {legal_text}

Vráť výsledok v JSON formáte:
{{
    "entities": [
        {{
            "text": "názov entity",
            "type": "LEGAL_CONCEPT|PARAGRAPH|INSTITUTION|SUBJECT",
            "context": "kontext výskytu",
            "span": [start, end]
        }}
    ]
}}
"""

RELATION_EXTRACTION_PROMPT = """
Pre nasledujúce entity z právneho textu identifikuj vzťahy medzi nimi.

**Entities:** {entities}

**Kontext:** {context}

**Typy vzťahov:**
- ODKAZUJE_NA: § X odkazuje na § Y
- DEFINUJE: § X definuje pojem Y
- UPRAVUJE: § X upravuje oblasť Y
- RUŠUJE: § X ruší ustanovenie Y
- DOPLŇUJE: § X dopĺňa § Y
- PODMIEŇUJE: § X podmieňuje § Y

Vráť výsledok v JSON formáte:
{{
    "relations": [
        {{
            "from": "source entity",
            "to": "target entity",
            "type": "RELATION_TYPE",
            "confidence": 0.0-1.0,
            "evidence": "textový dôkaz"
        }}
    ]
}}
"""
```

### Krok 4: Neo4j Integration

```python
# graph_store.py

from neo4j import GraphDatabase
from typing import List

class LegalKnowledgeGraph:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def create_entity(self, entity: Entity):
        """Vytvor node v grafe"""
        with self.driver.session() as session:
            session.run(
                """
                CREATE (e:Entity {
                    text: $text,
                    type: $type,
                    context: $context
                })
                """,
                text=entity.text,
                type=entity.type,
                context=entity.context
            )
    
    def create_relation(self, relation: Relation):
        """Vytvor edge v grafe"""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Entity {text: $source})
                MATCH (b:Entity {text: $target})
                CREATE (a)-[r:RELATES_TO {
                    type: $type,
                    confidence: $confidence
                }]->(b)
                """,
                source=relation.source,
                target=relation.target,
                type=relation.type,
                confidence=relation.confidence
            )
    
    def query_related_entities(self, entity_text: str, depth: int = 2):
        """Nájdi prepojené entity"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = (e:Entity {text: $text})-[*1..$depth]-(related)
                RETURN path
                """,
                text=entity_text,
                depth=depth
            )
            return [record["path"] for record in result]
```

## 🧩 Komponenty

### Entity Recognition

| Typ Entity | Príklady | Popis |
|------------|----------|-------|
| **LEGAL_CONCEPT** | zmluva, zodpovednosť, právo | Právne pojmy |
| **PARAGRAPH** | § 123, § 45 ods. 2 písm. a) | Odkazy na paragrafy |
| **SUBJECT** | fyzická osoba, právnická osoba | Subjekty práva |
| **INSTITUTION** | súd, ministerstvo, komisia | Právne inštitúcie |
| **DOCUMENT** | zákon, vyhláška, nariadenie | Typy dokumentov |

### Relation Types

```python
RELATION_TYPES = {
    "ODKAZUJE_NA": "§ X odkazuje na § Y",
    "DEFINUJE": "§ X definuje pojem Y",
    "UPRAVUJE": "§ X upravuje oblasť Y",
    "RUŠUJE": "§ X ruší § Y",
    "DOPLŇUJE": "§ X dopĺňa § Y",
    "PODMIEŇUJE": "§ X podmieňuje § Y",
    "VYLUČUJE": "§ X vylučuje aplikáciu § Y",
    "SPRESŇUJE": "§ X spresňuje § Y"
}
```

## 📚 Študijné Materiály

### LangGraph & LangChain

- 📖 [LangGraph Official Documentation](https://langchain-ai.github.io/langgraph/)
- 🎥 Tutorial: "Building Knowledge Graphs with LangChain"
- 💻 [GitHub: langgraph-examples](https://github.com/langchain-ai/langgraph/tree/main/examples)
- 📝 [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)

### Knowledge Graphs & NLP

- 📘 **Kniha:** "Knowledge Graphs" - Hogan et al. (2021)
- 📄 **Paper:** "Construction of the Literature Graph in Semantic Scholar" (2018)
- 🎓 **Course:** Stanford CS224W - Machine Learning with Graphs
- 📊 **Tutorial:** "Knowledge Graph Construction from Text" - ACL 2020

### Legal NLP

- 🏛️ **Workshop:** Natural Legal Language Processing (NLLP)
- 📦 **Library:** [LexNLP](https://github.com/LexPredict/lexpredict-lexnlp) - Legal NLP toolkit
- ⚖️ **Project:** [BlackStone](https://github.com/ICLRandD/Blackstone) - Spacy model pre UK legal texts
- 📑 **Dataset:** [Legal BERT](https://huggingface.co/nlpaueb/legal-bert-base-uncased)

### Neo4j & Graph Databases

- 🎓 [Neo4j Graph Academy](https://graphacademy.neo4j.com/) (free courses)
- 📘 **Kniha:** "Graph Databases" - Robinson, Webber, Eifrem
- 🔗 [Neo4j + LangChain Integration Guide](https://python.langchain.com/docs/integrations/graphs/neo4j_cypher)
- 📊 [Cypher Query Language Reference](https://neo4j.com/docs/cypher-manual/current/)

### Slovenské Právne Zdroje

- 📜 [Slov-Lex](https://www.slov-lex.sk/) - Slovenská legislatíva
- ⚖️ Judikáty slovenských súdov
- 📰 [Zbierka zákonov SR](https://www.zakonypreludi.sk/)
- 🏛️ [Najvyšší súd SR](https://www.nsud.sk/)

### Akademické Papers

- "Legal Information Extraction with NLP" (2021)
- "Automated Knowledge Graph Construction from Legal Documents" (2022)
- "Cross-lingual Legal NLP: Challenges and Opportunities" (2023)

## 🚀 MVP Prototyp

### Minimálny funkčný systém

**Rozsah:**
- ✅ Input: Jeden zákon (napr. časť Občianskeho zákonníka)
- ✅ Extrakcia všetkých § a ich textov
- ✅ Identifikácia krížových odkazov
- ✅ Extrakcia 3-5 typov entít
- ✅ Graf: Nodes (paragrafy, pojmy), Edges (odkazy, definície)
- ✅ Interaktívna vizualizácia

### Quick Start - MVP Implementácia

```python
# simple_legal_kg.py

import spacy
import networkx as nx
from pyvis.network import Network
import re

class SimpleLegalKG:
    def __init__(self):
        self.nlp = spacy.load("sk_core_news_lg")
        self.graph = nx.DiGraph()
    
    def extract_paragraphs(self, text: str):
        """Extrahuj § čísla z textu"""
        pattern = r'§\s*(\d+)'
        return re.findall(pattern, text)
    
    def split_by_paragraph(self, text: str):
        """Rozdel text podľa paragrafov"""
        pattern = r'§\s*(\d+)[^\§]*'
        sections = re.finditer(pattern, text)
        
        results = []
        for match in sections:
            section_id = f"§{match.group(1)}"
            section_text = match.group(0)
            results.append((section_id, section_text))
        
        return results
    
    def build_graph(self, legal_text: str):
        """Vybuduj graf z právneho textu"""
        sections = self.split_by_paragraph(legal_text)
        
        for section_id, section_text in sections:
            # Pridaj node pre paragraf
            self.graph.add_node(
                section_id, 
                text=section_text[:200],  # Preview
                type="paragraph"
            )
            
            # Nájdi odkazy na iné paragrafy
            references = self.extract_paragraphs(section_text)
            for ref in references:
                ref_id = f"§{ref}"
                if ref_id != section_id:
                    self.graph.add_edge(
                        section_id, 
                        ref_id, 
                        type="references"
                    )
    
    def visualize(self, output_file: str = "legal_kg.html"):
        """Vytvor interaktívnu vizualizáciu"""
        net = Network(
            height="750px", 
            width="100%", 
            directed=True,
            notebook=False
        )
        
        # Styling
        net.barnes_hut(gravity=-80000, central_gravity=0.3)
        
        # Load from NetworkX
        net.from_nx(self.graph)
        
        # Customize nodes
        for node in net.nodes:
            node["color"] = "#97C2FC"
            node["size"] = 25
            if node.get("type") == "paragraph":
                node["shape"] = "box"
        
        # Save
        net.show(output_file)
        print(f"Graf uložený do: {output_file}")
    
    def query_connections(self, paragraph_id: str, depth: int = 2):
        """Nájdi prepojené paragrafy"""
        if paragraph_id not in self.graph:
            return []
        
        # BFS na nájdenie spojených nodov
        connected = nx.single_source_shortest_path_length(
            self.graph, 
            paragraph_id, 
            cutoff=depth
        )
        return list(connected.keys())

# Použitie
if __name__ == "__main__":
    # Načítaj právny text
    with open("zakon.txt", "r", encoding="utf-8") as f:
        legal_text = f.read()
    
    # Vytvor knowledge graph
    kg = SimpleLegalKG()
    kg.build_graph(legal_text)
    kg.visualize()
    
    # Query
    connections = kg.query_connections("§123", depth=2)
    print(f"Paragrafy prepojené s §123: {connections}")
```

### Spustenie MVP

```bash
# Priprav sample legal text
echo "§1 Táto zmluva sa riadi § 2 a § 5..." > zakon.txt

# Spusti MVP
python simple_legal_kg.py

# Otvor legal_kg.html v prehliadači
```

## 🗺️ Roadmap

### Fáza 1: MVP (4-6 týždňov)
- [ ] Basic paragraph extraction
- [ ] Simple cross-reference detection
- [ ] NetworkX graph construction
- [ ] Pyvis visualization
- [ ] CLI interface

### Fáza 2: AI Integration (6-8 týždňov)
- [ ] LangGraph workflow setup
- [ ] LLM-based entity extraction
- [ ] Relation extraction s confidence scores
- [ ] Neo4j migration
- [ ] Prompt optimization pre slovenčinu

### Fáza 3: Advanced Features (8-12 týždňov)
- [ ] Semantic search
- [ ] Conflict detection (protirečivé §)
- [ ] Historical versioning
- [ ] Multi-document linking
- [ ] Q&A chatbot nad grafom
- [ ] Automatic summarization

### Fáza 4: Production (12+ týždňov)
- [ ] Web dashboard (Dash/Streamlit)
- [ ] REST API (FastAPI)
- [ ] User authentication
- [ ] Batch processing
- [ ] Performance optimization
- [ ] Deployment (Docker + Cloud)

## 📊 Metriky Úspešnosti

- **Entity Extraction Accuracy:** > 90%
- **Relation Extraction F1:** > 85%
- **Graph Completeness:** > 95% cross-references captured
- **Query Response Time:** < 200ms
- **Visualization Load Time:** < 3s pre 500 nodes

## 🤝 Prispievanie

Contributions sú vítané! Prosím pozri [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 Licencia

MIT License - pozri [LICENSE](LICENSE) pre detaily.

## 👥 Autori

- Tvoje meno - Initial work

## 🙏 Poďakovanie

- LangChain & LangGraph community
- Neo4j Graph Academy
- Spacy SK model contributors
- Legal NLP research community

---

**Poznámka:** Tento projekt je vo vývoji. Odporúčame začať s MVP prototypom a iteratívne pridávať funkcie podľa roadmapy.

**Kontakt:** your.email@example.com

**Dokumentácia:** [Wiki](https://github.com/your-username/legal-knowledge-graph/wiki)
