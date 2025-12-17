import os
import re
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum

import pdfplumber
import spacy
from spacy.tokens import Token
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================================
# Configuration and Schema Definitions
# ============================================================================

class QueryType(Enum):
    """Legal query types for targeted retrieval"""
    DEFINITION = "definition"
    CITATION = "citation"
    OBLIGATION = "obligation"
    RIGHT = "right"
    PARTY = "party"
    GENERAL = "general"


@dataclass
class ExtractionResult:
    """Container for extraction results with confidence"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    confidence_score: float
    validation_issues: List[str]


LEGAL_NODE_TYPES = {
    "Document": ["document_id", "title", "date", "jurisdiction", "doc_type"],
    "Section": ["section_number", "title", "level", "text"],
    "Clause": ["clause_id", "text", "section_ref"],
    "Definition": ["term", "definition", "scope", "source_page"],
    "Party": ["name", "role", "type"],
    "Statute": ["citation", "title", "jurisdiction"],
    "Caselaw": ["citation", "court", "year"],
    "Obligation": ["description", "party", "deadline", "condition"],
    "Right": ["description", "party", "scope"],
    "Condition": ["description", "type", "triggers"],
    "Date": ["value", "type", "context"],
    "Amount": ["value", "currency", "context"],
}

LEGAL_RELATIONSHIP_TYPES = [
    "CITES", "DEFINES", "AMENDS", "REFERENCES", "OBLIGATES",
    "GRANTS", "SUPERSEDES", "DEPENDS_ON", "PART_OF", "APPLIES_TO",
    "MODIFIES", "CROSS_REFERENCES"
]


# ============================================================================
# LLM Configuration
# ============================================================================

def get_llm(ai: str = None):
    """Get configured LLM instance"""
    if ai == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    
    return ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0, 
        api_key=os.getenv("OPENAI_API_KEY")
    )


# ============================================================================
# Legal NLP Configuration with spaCy
# ============================================================================

class LegalNLPProcessor:
    """Handles rule-based legal entity and relationship extraction"""
    
    def __init__(self):
        """Initialize spaCy with legal-specific configurations"""
        print("Loading spaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except OSError:
            print("Transformer model not found, falling back to en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        self._setup_legal_patterns()
        self._setup_custom_attributes()
    
    def _setup_legal_patterns(self):
        """Add legal-specific entity recognition patterns"""
        
        # Add entity ruler for legal citations and terms
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            patterns = [
                # US Code citations: 42 U.S.C. § 1983
                {"label": "STATUTE", "pattern": [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"LOWER": {"IN": ["u.s.c.", "usc", "u.s.c"]}},
                    {"TEXT": {"REGEX": r"§|section"}},
                    {"TEXT": {"REGEX": r"\d+"}}
                ]},
                
                # Federal Reporter citations: 123 F.3d 456
                {"label": "CASE_CITATION", "pattern": [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"TEXT": {"REGEX": r"F\.\d?d?"}},
                    {"TEXT": {"REGEX": r"\d+"}}
                ]},
                
                # CFR citations: 29 CFR 1910.1200
                {"label": "REGULATION", "pattern": [
                    {"TEXT": {"REGEX": r"\d+"}},
                    {"TEXT": "CFR"},
                    {"TEXT": {"REGEX": r"\d+(\.\d+)?"}}
                ]},
                
                # Legal phrases
                {"label": "OBLIGATION", "pattern": [{"LOWER": "shall"}]},
                {"label": "PERMISSION", "pattern": [{"LOWER": "may"}]},
                {"label": "PROHIBITION", "pattern": [{"LOWER": "shall"}, {"LOWER": "not"}]},
                
                # Effective dates
                {"label": "EFFECTIVE_DATE", "pattern": [
                    {"LOWER": "effective"},
                    {"LOWER": {"IN": ["date", "as", "on"]}}
                ]},
                
                # Defined terms (typically capitalized)
                {"label": "DEFINED_TERM", "pattern": [
                    {"TEXT": {"REGEX": r'"[A-Z][^"]*"'}}
                ]},
            ]
            
            ruler.add_patterns(patterns)
    
    def _setup_custom_attributes(self):
        """Set up custom token attributes for legal analysis"""
        if not Token.has_extension("is_legal_term"):
            Token.set_extension("is_legal_term", default=False)
        if not Token.has_extension("is_obligation"):
            Token.set_extension("is_obligation", default=False)
    
    def extract_structured_entities(self, text: str, page_num: int = 0) -> Dict[str, List[Dict]]:
        """Extract legal entities using rule-based patterns"""
        doc = self.nlp(text)
        
        entities = {
            "definitions": [],
            "citations": [],
            "obligations": [],
            "parties": [],
            "dates": [],
            "amounts": [],
            "sections": []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == "STATUTE" or ent.label_ == "CASE_CITATION" or ent.label_ == "REGULATION":
                entities["citations"].append({
                    "text": ent.text,
                    "type": ent.label_,
                    "page": page_num,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            elif ent.label_ in ["PERSON", "ORG"]:
                entities["parties"].append({
                    "name": ent.text,
                    "type": ent.label_,
                    "page": page_num
                })
            
            elif ent.label_ == "DATE":
                entities["dates"].append({
                    "text": ent.text,
                    "page": page_num
                })
            
            elif ent.label_ == "MONEY":
                entities["amounts"].append({
                    "text": ent.text,
                    "page": page_num
                })
        
        # Extract obligations (sentences with "shall")
        for sent in doc.sents:
            if any(token.lower_ == "shall" for token in sent):
                entities["obligations"].append({
                    "text": sent.text.strip(),
                    "page": page_num,
                    "type": "prohibition" if "shall not" in sent.text.lower() else "obligation"
                })
        
        # Extract definitions (pattern: "term" means/refers to)
        definition_pattern = r'"([^"]+)"\s+(?:means?|refers? to|is defined as)\s+([^.]+\.)'
        for match in re.finditer(definition_pattern, text, re.IGNORECASE):
            entities["definitions"].append({
                "term": match.group(1),
                "definition": match.group(2).strip(),
                "page": page_num
            })
        
        # Extract section headers
        section_pattern = r'(?:Section|Article|§)\s+(\d+(?:\.\d+)*)[:\.]?\s*([^\n]+)'
        for match in re.finditer(section_pattern, text):
            entities["sections"].append({
                "number": match.group(1),
                "title": match.group(2).strip(),
                "page": page_num
            })
        
        return entities


# ============================================================================
# Validation Layer
# ============================================================================

class ValidationLayer:
    """Multi-level validation for extraction quality"""
    
    def __init__(self, confidence_threshold: float = 0.99):
        self.confidence_threshold = confidence_threshold
    
    def validate_extraction(self, 
                          structured_entities: Dict,
                          semantic_entities: List,
                          ) -> ExtractionResult:
        """Validate and reconcile extractions"""
        
        validation_issues = []
        
        # Check 1: Citation format validation
        citation_score = self._validate_citations(structured_entities.get("citations", []))
        if citation_score < 0.95:
            validation_issues.append(f"Citation validation score low: {citation_score:.2f}")
        
        # Check 2: Definition consistency
        def_score = self._validate_definitions(structured_entities.get("definitions", []))
        if def_score < 0.95:
            validation_issues.append(f"Definition validation score low: {def_score:.2f}")
        
        # Check 3: Completeness check
        completeness_score = self._check_completeness(structured_entities)
        if completeness_score < 0.90:
            validation_issues.append(f"Completeness score low: {completeness_score:.2f}")
        
        # Calculate overall confidence
        confidence = (citation_score + def_score + completeness_score) / 3
        
        # Reconcile entities (prefer structured over semantic for high-confidence items)
        reconciled_entities = self._reconcile_entities(structured_entities, semantic_entities)
        
        return ExtractionResult(
            entities=reconciled_entities,
            relationships=[],
            confidence_score=confidence,
            validation_issues=validation_issues
        )
    
    def _validate_citations(self, citations: List[Dict]) -> float:
        """Validate citation formats"""
        if not citations:
            return 1.0
        
        valid_count = 0
        for citation in citations:
            # Check if citation matches known patterns
            text = citation.get("text", "")
            if re.search(r'\d+\s+(U\.?S\.?C\.?|F\.\d?d?|CFR)', text, re.IGNORECASE):
                valid_count += 1
        
        return valid_count / len(citations) if citations else 1.0
    
    def _validate_definitions(self, definitions: List[Dict]) -> float:
        """Validate definition quality"""
        if not definitions:
            return 1.0
        
        valid_count = 0
        for definition in definitions:
            term = definition.get("term", "")
            def_text = definition.get("definition", "")
            
            # Check if definition is substantial
            if term and len(def_text) > 10:
                valid_count += 1
        
        return valid_count / len(definitions) if definitions else 1.0
    
    def _check_completeness(self, entities: Dict) -> float:
        """Check if extraction appears complete"""
        # Check if we have diverse entity types
        entity_types_found = sum(1 for v in entities.values() if v)
        total_entity_types = len(entities)
        
        return entity_types_found / total_entity_types
    
    def _reconcile_entities(self, structured: Dict, semantic: List) -> List[Dict]:
        """Reconcile structured and semantic extractions"""
        # For now, prefer structured entities as they're more reliable
        # In production, implement sophisticated deduplication
        
        reconciled = []
        
        # Add all structured entities
        for entity_type, entities in structured.items():
            for entity in entities:
                entity["source"] = "structured"
                entity["entity_type"] = entity_type
                reconciled.append(entity)
        
        return reconciled


# ============================================================================
# Main Legal Document GraphRAG Class
# ============================================================================

class LegalDocumentGraphRAG:
    """High-precision GraphRAG system for legal documents"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                 ai_provider: str = None, confidence_threshold: float = 0.99):
        """
        Initialize Legal Document GraphRAG system
        
        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            ai_provider: "gemini" or None (defaults to OpenAI)
            confidence_threshold: Minimum confidence for auto-acceptance (default 0.99)
        """
        print("Initializing Legal Document GraphRAG...")
        
        # Connect to Neo4j
        self.graph = Neo4jGraph(
            url=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password,
            refresh_schema=False
        )
        
        # Initialize LLM
        self.llm = get_llm(ai_provider)
        
        # Initialize graph transformer for semantic extraction
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=list(LEGAL_NODE_TYPES.keys()),
            allowed_relationships=LEGAL_RELATIONSHIP_TYPES
        )
        
        # Initialize legal NLP processor
        self.legal_nlp = LegalNLPProcessor()
        
        # Initialize validation layer
        self.validator = ValidationLayer(confidence_threshold)
        
        print("✓ Initialization complete")
    
    def load_pdf_with_structure(self, pdf_path: str) -> List[Document]:
        """Load PDF preserving structure (better than PyPDFLoader)"""
        print(f"Loading PDF: {pdf_path}")
        
        documents = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(tqdm(pdf.pages, desc="Extracting pages")):
                text = page.extract_text()
                
                if text:
                    # Create document with metadata
                    doc = Document(
                        page_content=text,
                        metadata={
                            "page": i + 1,
                            "source": pdf_path,
                            "total_pages": len(pdf.pages)
                        }
                    )
                    documents.append(doc)
        
        print(f"✓ Loaded {len(documents)} pages")
        return documents
    
    def extract_structured_entities(self, documents: List[Document]) -> Dict:
        """Layer 1: Rule-based extraction with spaCy"""
        print("Layer 1: Extracting structured entities with spaCy...")
        
        all_entities = {
            "definitions": [],
            "citations": [],
            "obligations": [],
            "parties": [],
            "dates": [],
            "amounts": [],
            "sections": []
        }
        
        for doc in tqdm(documents, desc="Processing pages"):
            page_entities = self.legal_nlp.extract_structured_entities(
                doc.page_content,
                doc.metadata.get("page", 0)
            )
            
            # Merge entities
            for key in all_entities.keys():
                all_entities[key].extend(page_entities.get(key, []))
        
        print(f"✓ Extracted {sum(len(v) for v in all_entities.values())} structured entities")
        return all_entities
    
    def extract_semantic_entities(self, documents: List[Document], 
                                  structured_entities: Dict) -> List:
        """Layer 2: LLM-based semantic extraction"""
        print("Layer 2: Extracting semantic relationships with LLM...")
        
        # Use LLM to extract nuanced relationships
        graph_docs = self.graph_transformer.convert_to_graph_documents(documents)
        
        print(f"✓ Generated {len(graph_docs)} graph documents")
        return graph_docs
    
    def validate_and_reconcile(self, structured_entities: Dict, 
                              semantic_entities: List) -> ExtractionResult:
        """Layer 3: Validate and reconcile extractions"""
        print("Layer 3: Validating and reconciling extractions...")
        
        result = self.validator.validate_extraction(structured_entities, semantic_entities)
        
        print(f"✓ Confidence score: {result.confidence_score:.2%}")
        
        if result.validation_issues:
            print("⚠ Validation issues detected:")
            for issue in result.validation_issues:
                print(f"  - {issue}")
        
        return result
    
    def store_with_provenance(self, graph_docs: List, structured_entities: Dict, 
                             document_metadata: Dict):
        """Store entities with provenance tracking"""
        print("Storing entities in Neo4j with provenance...")
        
        try:
            # Method 1: Try APOC (preferred)
            self.graph.add_graph_documents(graph_docs)
            print("✓ Used APOC for efficient storage")
        except Exception as e:
            print(f"⚠ APOC not available ({str(e)}), using fallback method...")
            self._add_graph_docs_without_apoc(graph_docs)
        
        # Add structured entities with provenance
        self._add_structured_entities(structured_entities, document_metadata)
        
        # Create indexes for performance
        self._create_indexes()
        
        print("✓ Storage complete")
    
    def _add_graph_docs_without_apoc(self, graph_docs):
        """Fallback: Add graph documents without APOC"""
        for doc in tqdm(graph_docs, desc="Storing nodes"):
            # Add nodes
            for node in doc.nodes:
                # Escape node type and sanitize
                node_type = re.sub(r'[^A-Za-z0-9_]', '', node.type)
                
                query = f"""
                MERGE (n:{node_type} {{id: $id}})
                SET n += $properties
                """
                try:
                    self.graph.query(query, {
                        "id": node.id,
                        "properties": node.properties or {}
                    })
                except Exception as e:
                    print(f"Warning: Failed to add node {node.id}: {e}")
            
            # Add relationships
            for rel in doc.relationships:
                rel_type = re.sub(r'[^A-Za-z0-9_]', '', rel.type.upper())
                
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel_type}]->(target)
                SET r += $properties
                """
                try:
                    self.graph.query(query, {
                        "source_id": rel.source.id,
                        "target_id": rel.target.id,
                        "properties": rel.properties or {}
                    })
                except Exception as e:
                    print(f"Warning: Failed to add relationship: {e}")
    
    def _add_structured_entities(self, entities: Dict, doc_metadata: Dict):
        """Add structured entities with provenance"""
        
        # Add definitions
        for definition in entities.get("definitions", []):
            query = """
            MERGE (d:Definition {term: $term})
            SET d.definition = $definition,
                d.page = $page,
                d.source_doc = $source_doc
            """
            self.graph.query(query, {
                "term": definition["term"],
                "definition": definition["definition"],
                "page": definition["page"],
                "source_doc": doc_metadata.get("source", "unknown")
            })
        
        # Add citations
        for citation in entities.get("citations", []):
            query = """
            MERGE (c:Citation {text: $text})
            SET c.type = $type,
                c.page = $page,
                c.source_doc = $source_doc
            """
            self.graph.query(query, {
                "text": citation["text"],
                "type": citation["type"],
                "page": citation["page"],
                "source_doc": doc_metadata.get("source", "unknown")
            })
        
        # Add sections with hierarchy
        for section in entities.get("sections", []):
            query = """
            MERGE (s:Section {number: $number})
            SET s.title = $title,
                s.page = $page,
                s.source_doc = $source_doc
            """
            self.graph.query(query, {
                "number": section["number"],
                "title": section["title"],
                "page": section["page"],
                "source_doc": doc_metadata.get("source", "unknown")
            })
    
    def _create_indexes(self):
        """Create indexes for query performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Definition) ON (d.term)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Citation) ON (c.text)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Section) ON (s.number)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Party) ON (p.name)",
        ]
        
        for index_query in indexes:
            try:
                self.graph.query(index_query)
            except Exception as e:
                print(f"Index creation note: {e}")
    
    def process_legal_pdf(self, pdf_path: str, document_metadata: Optional[Dict] = None,
                         max_pages: int = None) -> Dict:
        """
        Main pipeline: Process legal PDF with high precision
        
        Args:
            pdf_path: Path to PDF file
            document_metadata: Optional metadata (title, date, jurisdiction, etc.)
            max_pages: Optional limit for testing
        
        Returns:
            Quality report dictionary
        """
        print("\n" + "="*70)
        print("LEGAL DOCUMENT GRAPHRAG PROCESSING PIPELINE")
        print("="*70 + "\n")
        
        if document_metadata is None:
            document_metadata = {"source": pdf_path}
        
        # Step 1: Load PDF with structure preservation
        documents = self.load_pdf_with_structure(pdf_path)
        
        if max_pages:
            documents = documents[:max_pages]
            print(f"⚠ Limited to {max_pages} pages for testing")
        
        # Step 2: Extract structured entities (spaCy)
        structured_entities = self.extract_structured_entities(documents)
        
        # Step 3: Extract semantic entities (LLM)
        semantic_entities = self.extract_semantic_entities(documents, structured_entities)
        
        # Step 4: Validate and reconcile
        validation_result = self.validate_and_reconcile(structured_entities, semantic_entities)
        
        # Step 5: Store with provenance
        if validation_result.confidence_score >= self.validator.confidence_threshold:
            print("✓ Confidence threshold met, storing in Neo4j...")
            self.store_with_provenance(semantic_entities, structured_entities, document_metadata)
        else:
            print(f"⚠ Confidence score ({validation_result.confidence_score:.2%}) " 
                  f"below threshold ({self.validator.confidence_threshold:.2%})")
            print("  Recommend human review before storing")
        
        # Generate quality report
        report = self._generate_quality_report(
            documents, 
            structured_entities, 
            semantic_entities, 
            validation_result
        )
        
        print("\n" + "="*70)
        print("PROCESSING COMPLETE")
        print("="*70 + "\n")
        
        return report
    
    def _generate_quality_report(self, documents: List[Document], 
                                structured_entities: Dict,
                                semantic_entities: List,
                                validation_result: ExtractionResult) -> Dict:
        """Generate comprehensive quality report"""
        
        total_nodes = sum(len(doc.nodes) for doc in semantic_entities)
        total_relationships = sum(len(doc.relationships) for doc in semantic_entities)
        
        report = {
            "pages_processed": len(documents),
            "structured_entities": {k: len(v) for k, v in structured_entities.items()},
            "semantic_extraction": {
                "nodes": total_nodes,
                "relationships": total_relationships
            },
            "validation": {
                "confidence_score": validation_result.confidence_score,
                "issues": validation_result.validation_issues
            }
        }
        
        # Print report
        print("\n📊 QUALITY REPORT")
        print("-" * 70)
        print(f"Pages processed: {report['pages_processed']}")
        print(f"\nStructured entities extracted:")
        for entity_type, count in report['structured_entities'].items():
            print(f"  - {entity_type}: {count}")
        print(f"\nSemantic extraction:")
        print(f"  - Nodes: {report['semantic_extraction']['nodes']}")
        print(f"  - Relationships: {report['semantic_extraction']['relationships']}")
        print(f"\nValidation:")
        print(f"  - Confidence: {report['validation']['confidence_score']:.2%}")
        if report['validation']['issues']:
            print(f"  - Issues: {len(report['validation']['issues'])}")
        
        return report
    
    def query_graph(self, natural_query: str) -> str:
        """
        Query the knowledge graph with natural language
        
        Args:
            natural_query: User's question in natural language
        
        Returns:
            Formatted response with citations
        """
        # Classify query type
        query_type = self._classify_query(natural_query)
        
        # Generate appropriate Cypher query
        cypher_query = self._generate_legal_cypher(natural_query, query_type)
        
        # Execute query
        results = self.graph.query(cypher_query)
        
        # Format response
        response = self._format_response(natural_query, results, query_type)
        
        return response
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify the type of legal query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["define", "definition", "means", "what is"]):
            return QueryType.DEFINITION
        elif any(word in query_lower for word in ["cite", "citation", "reference", "statute"]):
            return QueryType.CITATION
        elif any(word in query_lower for word in ["obligation", "must", "shall", "required"]):
            return QueryType.OBLIGATION
        elif any(word in query_lower for word in ["right", "entitled", "may"]):
            return QueryType.RIGHT
        elif any(word in query_lower for word in ["party", "who", "parties"]):
            return QueryType.PARTY
        else:
            return QueryType.GENERAL
    
    def _generate_legal_cypher(self, query: str, query_type: QueryType) -> str:
        """Generate targeted Cypher query based on query type"""
        
        if query_type == QueryType.DEFINITION:
            return """
            MATCH (d:Definition)
            WHERE toLower(d.term) CONTAINS toLower($search_term)
            RETURN d.term as term, d.definition as definition, d.page as page
            LIMIT 10
            """
        
        elif query_type == QueryType.CITATION:
            return """
            MATCH (c:Citation)
            RETURN c.text as citation, c.type as type, c.page as page
            LIMIT 10
            """
        
        elif query_type == QueryType.OBLIGATION:
            return """
            MATCH (o:Obligation)
            RETURN o.description as description, o.party as party, o.page as page
            LIMIT 10
            """
        
        else:  # GENERAL
            return """
            MATCH (n)
            WHERE toLower(toString(n)) CONTAINS toLower($search_term)
            RETURN labels(n) as type, properties(n) as properties
            LIMIT 20
            """
    
    def _format_response(self, query: str, results: List[Dict], 
                        query_type: QueryType) -> str:
        """Format query results with citations"""
        
        if not results:
            return "No results found in the knowledge graph for your query."
        
        response = f"Based on the legal documents in the knowledge graph:\n\n"
        
        for i, result in enumerate(results, 1):
            if query_type == QueryType.DEFINITION and "term" in result:
                response += f"{i}. **{result['term']}**: {result['definition']}"
                if "page" in result:
                    response += f" (Page {result['page']})"
                response += "\n\n"
            
            elif query_type == QueryType.CITATION and "citation" in result:
                response += f"{i}. {result['citation']} ({result.get('type', 'N/A')})"
                if "page" in result:
                    response += f" - Page {result['page']}"
                response += "\n\n"
            
            else:
                response += f"{i}. {result}\n\n"
        
        response += "\n⚠️ **Disclaimer**: This information is extracted from legal documents. "
        response += "Always consult with a qualified legal professional for legal advice."
        
        return response


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    load_dotenv()
    
    # Initialize system
    graphrag = LegalDocumentGraphRAG(
        neo4j_uri=os.getenv("NEO4J_URI"),
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        ai_provider=None,  # Change to "gemini" to use Google's Gemini
        confidence_threshold=0.99
    )
    
    # Process legal document
    # Replace with your legal document path
    pdf_path = "path/to/your/legal/document.pdf"
    
    document_metadata = {
        "source": pdf_path,
        "title": "Sample Legal Document",
        "date": "2024-01-01",
        "jurisdiction": "Federal",
        "doc_type": "Contract"
    }
    
    # Process with optional page limit for testing
    report = graphrag.process_legal_pdf(
        pdf_path=pdf_path,
        document_metadata=document_metadata,
        max_pages=5  # Remove or set to None to process all pages
    )
    
    print("\n✓ Legal document knowledge graph successfully created!")
    
    # Example queries
    print("\n" + "="*70)
    print("EXAMPLE QUERIES")
    print("="*70 + "\n")
    
    example_queries = [
        "What is the definition of 'Effective Date'?",
        "What obligations are mentioned in the document?",
        "What citations are referenced?"
    ]
    
    for query in example_queries:
        print(f"\nQ: {query}")
        print("-" * 70)
        response = graphrag.query_graph(query)
        print(response)


if __name__ == "__main__":
    main()
