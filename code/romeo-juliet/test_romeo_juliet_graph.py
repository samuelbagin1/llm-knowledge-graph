"""
Romeo & Juliet Graph Database Test Script

This script validates the accuracy and completeness of a Neo4j graph database
containing nodes and relationships from Shakespeare's Romeo and Juliet by
comparing graph query results against web-sourced information.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Generic, TypeVar
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ToolStrategy

# Load environment variables
load_dotenv()


def serialize_for_json(obj):
    """Convert Neo4j and other non-serializable objects to JSON-serializable format."""
    # Handle Neo4j Node objects
    if hasattr(obj, 'labels') and hasattr(obj, 'items'):
        return {
            '_type': 'Node',
            'labels': list(obj.labels),
            'properties': dict(obj.items())
        }
    # Handle Neo4j Relationship objects
    if hasattr(obj, 'type') and hasattr(obj, 'start_node'):
        return {
            '_type': 'Relationship',
            'type': obj.type,
            'properties': dict(obj.items()) if hasattr(obj, 'items') else {}
        }
    # Handle Neo4j Path objects
    if hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
        return {
            '_type': 'Path',
            'nodes': [serialize_for_json(n) for n in obj.nodes],
            'relationships': [serialize_for_json(r) for r in obj.relationships]
        }
    # Handle datetime objects
    if isinstance(obj, datetime):
        return obj.isoformat()
    # Handle dict-like objects
    if hasattr(obj, 'items') and not isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    # Handle dicts recursively
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    # Handle lists recursively
    if isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    # Handle primitives
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    # Fallback to string representation
    return str(obj)


class Neo4jJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Neo4j objects."""
    def default(self, obj):
        try:
            return serialize_for_json(obj)
        except Exception:
            return str(obj)


class RomeoJulietGraphTester:
    """Test harness for validating Romeo & Juliet knowledge graph accuracy"""

    # CONSTRUCTOR
    def __init__(self):
        """Initialize Neo4j connection and LLM clients"""
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")


        # Validate environment variables
        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Missing Neo4j credentials in environment variables")
        if not self.openai_api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")
        if not self.google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment variables")


        # Initialize connections
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )

        # Initialize LLM clients
        # ChatOpenAI for question generation
        self.openai_client = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0,
            api_key=self.openai_api_key
        )
        
        self.openai_creative_client = ChatOpenAI(
            model="gpt-5-mini",
            temperature=0.7,
            api_key=self.openai_api_key
        )

        # Google Gemini for everything else
        self.gemini_client = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=self.google_api_key
        )


        # Test tracking
        self.previous_questions = []
        self.test_results = []
        self.scores = []




# ----------------- METHODS -------------------

    def __del__(self):
        """Close Neo4j connection on cleanup"""
        if hasattr(self, 'driver'):
            self.driver.close()


    def execute_with_retry(self, func, max_retries=3, backoff_base=2):
        """Execute function with exponential backoff retry logic"""
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = backoff_base ** attempt
                print(f" Retry attempt {attempt + 1}/{max_retries} after error: {str(e)[:100]}")
                time.sleep(wait_time)


    def get_graph_schema(self):
        """Get Neo4j graph schema information with sample data
        
        Args:
            None
            
        Returns:
            Tuple
            schema, node_labels, rel_types
        
        """
        try:
            with self.driver.session() as session:
                # Get node labels and relationship types
                node_labels = session.run("CALL db.labels()").data()
                rel_types = session.run("CALL db.relationshipTypes()").data()

                # Get sample nodes with properties to understand schema
                sample_nodes = session.run("""
                    MATCH (n)
                    WITH labels(n)[0] as label, n
                    RETURN label, properties(n) as props
                    LIMIT 10
                """).data()

                # Get sample relationships
                sample_rels = session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN labels(a)[0] as from_label,
                           type(r) as rel_type,
                           labels(b)[0] as to_label,
                           properties(a) as from_props,
                           properties(b) as to_props
                    LIMIT 10
                """).data()



                # Build comprehensive schema
                schema = "Node Types:\n"
                for label in node_labels:
                    schema += f"  - {label['label']}\n"

                schema += "\nRelationship Types:\n"
                for rel in rel_types:
                    schema += f"  - {rel['relationshipType']}\n"

                # Add sample data to understand property names
                schema += "\nSample Nodes (showing property structure):\n"
                for node in sample_nodes[:5]:
                    schema += f"  - {node['label']}: {node['props']}\n"

                schema += "\nSample Relationships:\n"
                for rel in sample_rels[:5]:
                    from_id = rel['from_props'].get('id', rel['from_props'].get('name', 'unknown'))
                    to_id = rel['to_props'].get('id', rel['to_props'].get('name', 'unknown'))
                    schema += f"  - ({rel['from_label']}: {from_id}) --[{rel['rel_type']}]--> ({rel['to_label']}: {to_id})\n"

                return schema, node_labels, rel_types

        except Exception as e:
            print(f"Error retrieve schema: {e}")
            return "Schema information unavailable", [], []
        
        
    def generate_final_report(self):
        """Generate comprehensive JSON and text reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Calculate statistics
        valid_scores = [s for s in self.scores if s > 0]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0

        # Aggregate recommendations and issues
        all_recommendations = []
        all_missing_data = []
        all_discrepancies = []
        all_correct_elements = []

        for result in self.test_results:
            if 'comparison' in result:
                comp = result['comparison']
                all_recommendations.extend(comp.get('recommendations', []))
                all_missing_data.extend(comp.get('missing_data', []))
                all_discrepancies.extend(comp.get('discrepancies', []))
                all_correct_elements.extend(comp.get('correct_elements', []))

        # Pre-serialize test results to handle Neo4j objects
        try:
            serialized_test_results = serialize_for_json(self.test_results)
        except Exception as e:
            print(f"Warning: Error serializing test results: {e}")
            serialized_test_results = []
            for result in self.test_results:
                try:
                    serialized_test_results.append(serialize_for_json(result))
                except Exception as inner_e:
                    serialized_test_results.append({
                        "test_number": result.get("test_number", "unknown"),
                        "error": f"Serialization error: {str(inner_e)}",
                        "timestamp": datetime.now().isoformat()
                    })

        # Create JSON report
        json_report = {
            "test_run_date": datetime.now().isoformat(),
            "total_tests": len(self.test_results),
            "successful_tests": len([r for r in self.test_results if 'error' not in r]),
            "failed_tests": len([r for r in self.test_results if 'error' in r]),
            "average_score": round(avg_score, 2),
            "min_score": min(valid_scores) if valid_scores else 0,
            "max_score": max(valid_scores) if valid_scores else 0,
            "scores": self.scores,
            "individual_tests": serialized_test_results,
            "summary": {
                "overall_grade": self._get_grade(avg_score),
                "strengths": list(set(all_correct_elements))[:10],
                "weaknesses": list(set(all_missing_data))[:10],
                "discrepancies_found": list(set(all_discrepancies))[:10],
                "recommendations": list(set(all_recommendations))[:15]
            }
        }

        # Save JSON report
        json_filename = f"test_results_{timestamp}.json"
        json_path = f"code/romeo-juliet/tests/{json_filename}"

        try:
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=2, cls=Neo4jJSONEncoder)
            print(f"JSON report saved: {json_filename}")
        except Exception as e:
            print(f"Error saving JSON report: {e}")
            # Try with default=str as last resort
            with open(json_path, 'w') as f:
                json.dump(json_report, f, indent=2, default=str)
            print(f"JSON report saved with fallback encoding: {json_filename}")

        return json_path
    

    def _get_grade(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        if score >= 90:
            return "A (Excellent)"
        elif score >= 80:
            return "B (Good)"
        elif score >= 70:
            return "C (Acceptable)"
        elif score >= 60:
            return "D (Poor)"
        else:
            return "F (Failing)"
        
# ----------------- METHODS -------------------



# ---------- Core Test Functions ----------

    # 1- generate a question via llm function
    def generate_test_question(self, iteration: int) -> Dict[str, Any]:
        """
        Function 1: Generate varied test questions about Romeo and Juliet

        Args:
            iteration: Current test iteration number

        Returns:
            Dictionary containing question, question_type, expected_nodes, expected_relationships
        """
        print(f"\nGenerating test question {iteration}...")

        previous_q_text = "\n".join([f"- {q}" for q in self.previous_questions])
        
        graph_schema, node_labels, rel_types = self.get_graph_schema()

        prompt = f"""You are a Shakespeare expert creating test questions to validate a knowledge graph about Romeo and Juliet.

Generate ONE unique test question that can be answered using graph database queries. The question should test different aspects of the story across these categories:

Categories to rotate through:
1. Character relationships (family, romantic, friendship, rivalries)
2. Character attributes (roles, traits, family affiliations)
3. Plot events and their connections to characters
4. Locations and settings in the story
5. Multi-hop relationships (e.g., "Who is Romeo's love partner?")

Graph Schema (use schema to determine expected_nodes and expected_relationships):
{graph_schema}

Previously asked questions (DO NOT REPEAT):
{previous_q_text if previous_q_text else "None yet"}

Generate a question for iteration {iteration}/5. Try to vary the question type.
"""

        schema = {
            "title": "TestQuestion",
            "type": "object",
            "description": "A test question about Romeo and Juliet",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The test question as a clear, specific question"
                },
                "question_type": {
                    "type": "string",
                    "enum": ["relationship", "character_attribute", "event", "location", "multi_hop"]
                },
                "expected_nodes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [node['label'] for node in node_labels]
                    },
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": "List of node types expected in answer"
                },
                "expected_relationships": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [rel['relationshipType'] for rel in rel_types]
                    },
                    "minItems": 1,
                    "uniqueItems": True,
                    "description": "List of relationship types expected"
                }
            },
            "required": ["question", "question_type", "expected_nodes", "expected_relationships"]
        }


        agent = create_agent(model="gpt-5-mini", tools=[], response_format=ProviderStrategy(schema=schema))
        response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

        # structured_response is already a dict when using ProviderStrategy
        question_data = response["structured_response"]

        self.previous_questions.append(question_data['question'])

        print(f" Question: {question_data['question']}")
        print(f" Type: {question_data['question_type']}")

        return question_data



    # 2- query database function
    def query_graph_database(self, question: str) -> Dict[str, Any]:
        """
        Function 2: Convert question to Cypher query and retrieve data from Neo4j

        Args:
            question: Test question to answer using the graph

        Returns:
            query_data, structured_answer.strip()
        """
        print(f"\n Querying graph database...")

        # First, check if database has any data
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']

            if node_count == 0:
                print(f" Database is EMPTY")

        # Get schema information
        schema_info, node_labels, rel_types = self.get_graph_schema()

        # Build node labels and relationship types lists for the prompt
        node_labels_list = [node['label'] for node in node_labels]
        rel_types_list = [rel['relationshipType'] for rel in rel_types]

        # System prompt - defines the agent's role and capabilities
        system_prompt = """You are a Neo4j Cypher expert agent specialized in querying knowledge graphs.

Your task is to answer questions by querying a Neo4j graph database about Romeo and Juliet.

## Your Capabilities
You have access to the `search_database` tool which executes Cypher queries against Neo4j.

## Query Strategy
1. **Analyze the question** to identify what nodes and relationships are relevant
2. **Start with exploration queries** to understand what data exists:
   - For nodes: `MATCH (n:Label) RETURN n.id, labels(n)`
   - For relationships: `MATCH (a)-[r:TYPE]->(b) RETURN a.id, type(r), b.id`
3. **Refine iteratively** - use results from initial queries to build more specific queries
4. **Find the best matches** - keep querying until you find the most relevant data

## Cypher Query Rules
- Use backticks for labels/types with special characters: `MATCH (n:`Special-Label`) ...`
- For text matching use case-insensitive: `WHERE toLower(n.id) CONTAINS toLower('romeo')`
- Use undirected relationships `-[r]-` when direction is unknown
- Always add `LIMIT 25` to prevent large result sets
- Return useful properties: `RETURN n.id, labels(n), type(r), properties(n)`

## Important
- You MUST use the search_database tool to query the database
- Make multiple queries if needed to find the best answer
- When you have found sufficient data, provide your final answer with the best Cypher query"""

        # User prompt - provides the specific question and schema
        user_prompt = f"""Answer this question about Romeo and Juliet by querying the graph database:

**Question:** {question}

## Available Graph Schema

### Node Labels (use these exact labels in queries):
{json.dumps(node_labels_list, indent=2)}

### Relationship Types (use these exact types in queries):
{json.dumps(rel_types_list, indent=2)}

### Sample Data (shows actual property structure):
{schema_info}

## Your Task
1. Analyze the question to determine which node labels and relationship types are relevant
2. Use the `search_database` tool to query the database with Cypher queries
3. Start broad, then refine based on results
4. Continue querying until you find the best matching nodes, properties, and relationships
5. Return your final answer with the most effective Cypher query and the data you found

Begin by identifying the relevant node labels and relationship types, then query the database."""

        # Response schema
        response_schema = {
            "title": "GraphQueryResult",
            "type": "object",
            "description": "Final query results from graph database exploration",
            "properties": {
                "cypher_query": {
                    "type": "string",
                    "description": "The final/best Cypher query that answers the question"
                },
                "explanation": {
                    "type": "string",
                    "description": "Explanation of the query strategy and what was found"
                },
                "data": {
                    "type": "string",
                    "description": "The relevant data returned from the database (JSON string)"
                },
                "nodes_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node IDs that are relevant to the answer"
                },
                "relationships_found": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of relationships found (format: 'nodeA -[REL_TYPE]-> nodeB')"
                }
            },
            "required": ["cypher_query", "explanation", "data"]
        }

        # Define the database query tool - capture driver for closure
        driver = self.driver

        @tool
        def search_database(cypher_query: str) -> str:
            """Execute a Cypher query against the Neo4j graph database.

            Args:
                cypher_query: A valid Cypher query string to execute

            Returns:
                JSON string of query results, or error message if query fails
            """
            def serialize_neo4j_object(obj):
                """Convert Neo4j objects to JSON-serializable format."""
                # Handle Neo4j Node objects
                if hasattr(obj, 'labels') and hasattr(obj, 'items'):
                    return {
                        '_type': 'Node',
                        'labels': list(obj.labels),
                        'properties': dict(obj.items())
                    }
                # Handle Neo4j Relationship objects
                if hasattr(obj, 'type') and hasattr(obj, 'start_node'):
                    return {
                        '_type': 'Relationship',
                        'type': obj.type,
                        'properties': dict(obj.items()) if hasattr(obj, 'items') else {}
                    }
                # Handle Neo4j Path objects
                if hasattr(obj, 'nodes') and hasattr(obj, 'relationships'):
                    return {
                        '_type': 'Path',
                        'nodes': [serialize_neo4j_object(n) for n in obj.nodes],
                        'relationships': [serialize_neo4j_object(r) for r in obj.relationships]
                    }
                # Handle dict-like objects
                if hasattr(obj, 'items'):
                    return {k: serialize_neo4j_object(v) for k, v in obj.items()}
                # Handle lists
                if isinstance(obj, list):
                    return [serialize_neo4j_object(item) for item in obj]
                # Handle primitives
                if isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                # Fallback to string representation
                return str(obj)

            try:
                with driver.session() as session:
                    result = session.run(cypher_query)
                    records = [dict(record) for record in result]

                serialized = []
                for record in records:
                    serialized_record = {}
                    for key, value in record.items():
                        try:
                            serialized_record[key] = serialize_neo4j_object(value)
                        except Exception as e:
                            serialized_record[key] = f"<serialization error: {str(e)}>"
                    serialized.append(serialized_record)

                return json.dumps(serialized, indent=2, default=str)

            except Exception as e:
                return f"Query error: {str(e)}"



        # Create and run the agent
        agent = create_agent(
            model=self.openai_client,
            tools=[search_database],
            response_format=ProviderStrategy(schema=response_schema),
            system_prompt=system_prompt
        )
        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})

        # structured_response is already a dict when using ProviderStrategy
        query_data = response["structured_response"]

        try:
            cypher_query = query_data['cypher_query']
            # data field might be a JSON string or already parsed
            data_field = query_data.get('data', '[]')
            records = json.loads(data_field) if isinstance(data_field, str) else (data_field or [])

            print(f"Generated Cypher query: {cypher_query}")
            print(f"Found {len(records) if isinstance(records, list) else 'N/A'} results")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse query response: {e}")
            cypher_query = "MATCH (n) RETURN n LIMIT 10"
            records = []

        # Fallback if no results
        if not records and node_count > 0:
            try:
                with self.driver.session() as session:
                    fallback_query = "MATCH (n)-[r]-(m) RETURN n.id as node1, type(r) as rel_type, m.id as node2 LIMIT 50"
                    result = session.run(fallback_query)
                    records = [dict(record) for record in result]
                    if records:
                        print(f"Fallback query returned {len(records)} results")
            except Exception as e:
                print(f"Fallback query failed: {e}")
                records = []



        # Format results into natural language answer
        format_prompt = f"""Based on the following graph database query results, provide a clear, concise answer to the original question.

Question: {question}

Query Results:
{json.dumps(records, indent=2) if records else "No results found"}

Provide a natural language answer that:
1. Directly answers the question
2. Includes specific names, relationships, and details from the results
3. Acknowledges if information is missing or incomplete
4. Is clear and concise (2-4 sentences)

Return ONLY the answer text, no preamble or JSON formatting."""


        structured_answer = self.openai_client.invoke(format_prompt).content

        print(f"Answer: {structured_answer}")

        return {"graph_data": query_data, "structured_answer": structured_answer.strip()}
    
        
        
        
        
    # 3 - search function
    def search_web_for_answer(self, question: str) -> Dict[str, Any]:
        """
        Function 3: Find authoritative answer using web search

        Args:
            question: Question to research on the web

        Returns:
            Dictionary containing sources, web answer, and confidence level
        """
        print("Web searching")

        user_prompt = f"""You are a Shakespeare scholar researching Romeo and Juliet. Answer this question using your knowledge and provide authoritative information.

Question: {question}

Provide a comprehensive, accurate answer based on Shakespeare's Romeo and Juliet. Include:
1. Direct answer to the question
2. Specific details from the play (character names, relationships, events)
3. Act/Scene references if relevant
4. Any important context or nuances

"""

        response_schema = {
            "title": "WebSearch",
            "type": "object",
            "description": "Web search for answers on provided question",
            "properties": {
                "web_answer": {
                    "type": "string",
                    "description": "Detailed answer with specific information from the play"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "your confidence in this answer"
                },
                "sources": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of authoritative sources or references (e.g., Act/Scene numbers, web links, books)"
                }
            },
            "required": ["web_answer", "confidence", "sources"]
        }


        agent = create_agent(model=self.gemini_client, tools=[], response_format=ToolStrategy(schema=response_schema))

        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        web_data = response["structured_response"]

        print(f"Web answer confidence: {web_data['confidence']}")
        print(f"Answer: {web_data['web_answer']}...")

        return web_data
            
            
            

    def compare_and_score(
        self,
        question: str,
        graph_answer: Dict[str, Any],
        web_answer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Function 4: Compare graph answer vs web answer and assign accuracy score

        Args:
            question: Original question
            graph_answer: Answer from graph database
            web_answer: Answer from web search

        Returns:
            Dictionary containing score, assessment, discrepancies, and recommendations
        """
        print(f"\n  📊 Comparing answers and scoring...")

        user_prompt = f"""You are evaluating the accuracy and completeness of a knowledge graph about Romeo and Juliet.

Compare these two answers to the same question:

QUESTION: {question}

GRAPH DATABASE ANSWER:
{graph_answer['structured_answer']}

AUTHORITATIVE ANSWER (from Shakespeare's text):
{web_answer['web_answer']}

RAW GRAPH DATA:
{json.dumps(graph_answer.get('graph_data', {}) if isinstance(graph_answer.get('graph_data'), dict) else graph_answer.get('graph_data', [])[:5], indent=2) if graph_answer.get('graph_data') else "No results returned"}

Evaluate the graph database answer and assign a score using this rubric:
- 100: Perfect match, all information correct and complete
- 80-99: Mostly accurate, minor details missing or imprecise
- 60-79: Partially accurate, some key information missing or incorrect
- 40-59: Significantly inaccurate or incomplete
- 0-39: Mostly incorrect or no useful data returned

"""

        response_schema = {
            "title": "CompareAndScore",
            "type": "object",
            "description": "Compare and score data of web search and graph data",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "numerical score 0-100"
                },
                "accuracy_assessment": {
                    "type": "string",
                    "description": "Detailed explanation of why you gave this score"
                },
                "discrepancies": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List specific differences or errors in the graph answer"
                },
                "missing_data": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List important information missing from the graph"
                },
                "correct_elements": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List what the graph got right"
                },
                "recommendations": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Specific suggestions to improve the graph (e.g., 'Add relationship LOVES between Romeo and Juliet')"
                }
            },
            "required": ["score", "accuracy_assessment", "discrepancies", "missing_data", "correct_elements", "recommendations"]
        }


        agent = create_agent(model=self.gemini_client, tools=[], response_format=ToolStrategy(schema=response_schema))

        response = agent.invoke({"messages": [{"role": "user", "content": user_prompt}]})
        comparison = response["structured_response"]

        print(f"  ✓ Score: {comparison['score']}/100")
        print(f"    Assessment: {comparison['accuracy_assessment'][:150]}...")

        return comparison




    # ------------ TEST LOOP ------------
    def run_test_suite(self, num_tests: int = 5):
        """
        Execute the complete test suite

        Args:
            num_tests: Number of test iterations to run (default 5)
        """
        print("\n\nROMEO & JULIET KNOWLEDGE GRAPH TEST SUITE\n\n")
        print(f"\nRunning {num_tests} test iterations\n")

        for i in range(1, num_tests + 1):
            print(f"------ TEST {i}/{num_tests} ------")

            try:
                # Step 1: Generate question
                question_data = self.generate_test_question(i)

                # Step 2: Query graph database
                graph_answer = self.query_graph_database(question_data["question"])

                # Step 3: Search web for ground truth
                web_answer = self.search_web_for_answer(question_data['question'])

                # Step 4: Compare and score
                comparison = self.compare_and_score(
                    question_data['question'],
                    graph_answer,
                    web_answer
                )


                # Store results (serialize to avoid Neo4j object issues)
                test_result = {
                    'test_number': i,
                    'question': serialize_for_json(question_data),
                    'graph_answer': serialize_for_json(graph_answer),
                    'web_answer': serialize_for_json(web_answer),
                    'comparison': serialize_for_json(comparison),
                    'timestamp': datetime.now().isoformat()
                }


                self.test_results.append(test_result)
                self.scores.append(comparison['score'])

                # Display summary
                print('------------')
                print(f"  TEST {i} SUMMARY")
                print(f"  Question: {question_data['question']}")
                print(f"  Score: {comparison['score']}/100")
                print(f"  Status: {'PASS' if comparison['score'] >= 70 else 'FAIL'}")

            except Exception as e:
                print(f"\nTEST {i} FAILED WITH ERROR: {str(e)}")

                # Store failed test
                self.test_results.append({
                    'test_number': i,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                self.scores.append(0)


        # Generate final report
        self.generate_final_report()
        
        





def main():
    """Main entry point for the test script"""
    try:
        tester = RomeoJulietGraphTester()
        tester.run_test_suite(num_tests=5)

    except Exception as e:
        print(f"Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


RAGAS