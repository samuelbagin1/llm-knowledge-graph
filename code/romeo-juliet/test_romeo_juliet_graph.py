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
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.tools import tool

# Load environment variables
load_dotenv()


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


    def get_graph_schema(self) -> str:
        """Get Neo4j graph schema information with sample data"""
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

                return schema

        except Exception as e:
            print(f"Error retrieve schema: {e}")
            return "Schema information unavailable"
        
        
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
            "individual_tests": self.test_results,
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

        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2)

        print(f"JSON report saved: {json_filename}")

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

    # generate a question via llm function
    def generate_test_question(self, iteration: int) -> Dict[str, Any]:
        """
        Function 1: Generate varied test questions about Romeo and Juliet

        Args:
            iteration: Current test iteration number

        Returns:
            Dictionary containing question, type, and expected node/relationship types
        """
        print(f"\nGenerating test question {iteration}...")

        previous_q_text = "\n".join([f"- {q}" for q in self.previous_questions])

        prompt = f"""You are a Shakespeare expert creating test questions to validate a knowledge graph about Romeo and Juliet.

Generate ONE unique test question that can be answered using graph database queries. The question should test different aspects of the story across these categories:

Categories to rotate through:
1. Character relationships (family, romantic, friendship, rivalries)
2. Character attributes (roles, traits, family affiliations)
3. Plot events and their connections to characters
4. Locations and settings in the story
5. Multi-hop relationships (e.g., "Who is Romeo's love partner?")

Previously asked questions (DO NOT REPEAT):
{previous_q_text if previous_q_text else "None yet"}

Generate a question for iteration {iteration}/5. Try to vary the question type.
"""

        nodes = self.get_graph_schema()

        schema = {
            "name": "QuestionSpec",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "title": "QuestionSpec",
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 1,
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
                            "enum": ["Character", "Family", "Location", "Event", "Object", "Organization"]
                        },
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": "List of node types expected in answer"
                    },
                    "expected_relationships": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["LOVES", "MEMBER_OF", "KILLS", "FRIENDS_WITH", "ENEMY_OF", "LIVES_IN", "ATTENDS", "OWNS"]
                        },
                        "minItems": 1,
                        "uniqueItems": True,
                        "description": "List of relationship types expected"
                    }
                },
                "required": ["question", "question_type", "expected_nodes", "expected_relationships"],
                "additionalProperties": False
            },
            "strict": True
        }


        def call_api():
            response = self.openai_creative_client.invoke(prompt)
            return response.content

        response = self.execute_with_retry(call_api)

        # Parse JSON response
        try:
            # Clean response to extract JSON if needed
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            question_data = json.loads(response)
            self.previous_questions.append(question_data['question'])

            print(f" Question: {question_data['question']}")
            print(f" Type: {question_data['question_type']}")

            return question_data

        except json.JSONDecodeError as e:
            print(f" Failed to parse JSON response: {e}")
            print(f" Raw response: {response[:200]}")
            raise



    @tool
    def search_database(self, cypher_query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query against Neo4j and return results"""

        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [dict(record) for record in result]

            return records

        except Exception as e:
            return "An Exception occurred: " + str(e)

    # query database function
    def query_graph_database(self, question: str) -> Dict[str, Any]:
        """
        Function 2: Convert question to Cypher query and retrieve data from Neo4j

        Args:
            question: Test question to answer using the graph

        Returns:
            Dictionary containing cypher query, raw results, and structured answer
        """
        print(f"\n Querying graph database...")

        # First, check if database has any data
        with self.driver.session() as session:
            node_count = session.run("MATCH (n) RETURN count(n) as count").data()[0]['count']
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']

            if node_count == 0:
                print(f" Database is EMPTY")

        # Get schema information
        schema_info = self.get_graph_schema()

        prompt = f"""You are a Neo4j Cypher expert. Generate a Cypher query to answer this question about Romeo and Juliet.

Question: {question}

Available graph schema (CRITICAL - study the sample data to see actual property names):
{schema_info}

CRITICAL RULES FOR QUERY GENERATION:
1. **EXAMINE THE SAMPLE NODES ABOVE** to see what properties actually exist
2. **USE THE ACTUAL PROPERTY NAMES** shown in the sample data - do NOT assume property names
3. For text matching, try MULTIPLE approaches to maximize matches:
   - Use CONTAINS for partial matching: WHERE n.id CONTAINS 'Romeo'
   - Try case-insensitive: WHERE toLower(n.id) CONTAINS toLower('romeo')
   - Try multiple properties: WHERE n.id CONTAINS 'Romeo' OR n.name CONTAINS 'Romeo'
4. Keep queries SIMPLE and BROAD to ensure results:
   - Start with simple patterns: MATCH (n)-[r]-(m)
   - Add filters incrementally
   - Use undirected relationships: -[r]- instead of -[r]-> when unsure of direction
5. Return full nodes and relationships: RETURN n, r, m
6. Always add: LIMIT 25

QUERY STRATEGY - Use this decision tree:
- For character questions: Look for Character nodes and their relationships
- For relationship questions: Match patterns between two entities
- For location questions: Look for Location nodes and connections
- For event questions: Look for events and participating characters
- When unsure: Use broad patterns and filter results

EXAMPLE QUERIES (adapt based on actual schema):


Return your response as a JSON object:
{{
  "cypher_query": "Your Cypher query here - must be valid Cypher",
  "explanation": "Brief explanation including any assumptions about property names based on the schema"
}}

Return ONLY the JSON object, no other text."""

        def call_api():
            response = self.openai_creative_client.invoke(prompt)
            return response.content

        response = self.execute_with_retry(call_api)

        # Parse query
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            query_data = json.loads(response)
            cypher_query = query_data['cypher_query']

            print(f"Generated Cypher query: {cypher_query}")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse query response: {e}")
            # Fallback: try to extract cypher directly
            cypher_query = response
            
            

        # Execute query against Neo4j
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [dict(record) for record in result]

            print(f"Returned {len(records)} results")


            # If no results, try a fallback broader query
            if len(records) == 0 and node_count > 0:

                # Extract key terms from question for fallback
                fallback_query = """
                MATCH (n)-[r]-(m)
                RETURN n, r, m
                LIMIT 50
                """

                try:
                    result = session.run(fallback_query)
                    fallback_records = [dict(record) for record in result]

                    if len(fallback_records) > 0:
                        print(f"Fallback: {len(fallback_records)} results")
                        records = fallback_records
                except:
                    pass

        except Exception as e:
            print(f"Query execution failed: {str(e)}")
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

        def format_api_call():
            response = self.openai_client.invoke(format_prompt)
            return response.content

        structured_answer = self.execute_with_retry(format_api_call)

        print(f"Answer: {structured_answer}")

        return {
            "cypher_query": cypher_query,
            "results": records,
            "structured_answer": structured_answer.strip()
        }
        
        
        
        
    # search function
    def search_web_for_answer(self, question: str) -> Dict[str, Any]:
        """
        Function 3: Find authoritative answer using web search

        Args:
            question: Question to research on the web

        Returns:
            Dictionary containing sources, web answer, and confidence level
        """
        print("Web searching")

        prompt = f"""You are a Shakespeare scholar researching Romeo and Juliet. Answer this question using your knowledge and provide authoritative information.

Question: {question}

Provide a comprehensive, accurate answer based on Shakespeare's Romeo and Juliet. Include:
1. Direct answer to the question
2. Specific details from the play (character names, relationships, events)
3. Act/Scene references if relevant
4. Any important context or nuances

Return your response as a JSON object:
{{
  "web_answer": "Detailed answer with specific information from the play",
  "confidence": "high/medium/low - your confidence in this answer",
  "sources": ["List of authoritative sources or references (e.g., Act/Scene numbers)"],
  "key_details": ["List of important specific details that should be in a knowledge graph"]
}}

Return ONLY the JSON object, no other text."""

        def call_api():
            response = self.gemini_client.invoke(prompt)
            return response.content

        response = self.execute_with_retry(call_api)

        # Parse response
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            web_data = json.loads(response)

            print(f"Web answer confidence: {web_data['confidence']}")
            print(f"Answer: {web_data['web_answer']}...")

            return web_data

        except json.JSONDecodeError as e:
            print(f"Failed to parse web search response: {e}")
            return {
                "web_answer": response,
                "confidence": "low",
                "sources": [],
                "key_details": []
            }

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

        prompt = f"""You are evaluating the accuracy and completeness of a knowledge graph about Romeo and Juliet.

Compare these two answers to the same question:

QUESTION: {question}

GRAPH DATABASE ANSWER:
{graph_answer['structured_answer']}

AUTHORITATIVE ANSWER (from Shakespeare's text):
{web_answer['web_answer']}

KEY DETAILS THAT SHOULD BE PRESENT:
{json.dumps(web_answer.get('key_details', []), indent=2)}

RAW GRAPH DATA:
{json.dumps(graph_answer['results'][:5], indent=2) if graph_answer['results'] else "No results returned"}

Evaluate the graph database answer and assign a score using this rubric:
- 100: Perfect match, all information correct and complete
- 80-99: Mostly accurate, minor details missing or imprecise
- 60-79: Partially accurate, some key information missing or incorrect
- 40-59: Significantly inaccurate or incomplete
- 0-39: Mostly incorrect or no useful data returned

Return your response as a JSON object:
{{
  "score": <numerical score 0-100>,
  "accuracy_assessment": "Detailed explanation of why you gave this score",
  "discrepancies": ["List specific differences or errors in the graph answer"],
  "missing_data": ["List important information missing from the graph"],
  "correct_elements": ["List what the graph got right"],
  "recommendations": ["Specific suggestions to improve the graph (e.g., 'Add relationship LOVES between Romeo and Juliet')"]
}}

Be thorough and specific. Return ONLY the JSON object."""

        def call_api():
            response = self.gemini_client.invoke(prompt)
            return response.content

        response = self._execute_with_retry(call_api)

        # Parse response
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            comparison = json.loads(response)

            print(f"  ✓ Score: {comparison['score']}/100")
            print(f"    Assessment: {comparison['accuracy_assessment'][:150]}...")

            return comparison

        except json.JSONDecodeError as e:
            print(f"  ✗ Failed to parse comparison response: {e}")
            print(f"  Raw response: {response[:300]}")
            return {
                "score": 0,
                "accuracy_assessment": "Failed to parse scoring response",
                "discrepancies": [],
                "missing_data": [],
                "correct_elements": [],
                "recommendations": []
            }




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
                graph_answer = self.query_graph_database(question_data['question'])

                # Step 3: Search web for ground truth
                web_answer = self.search_web_for_answer(question_data['question'])

                # Step 4: Compare and score
                comparison = self.compare_and_score(
                    question_data['question'],
                    graph_answer,
                    web_answer
                )


                # Store results
                test_result = {
                    'test_number': i,
                    'question': question_data,
                    'graph_answer': graph_answer,
                    'web_answer': web_answer,
                    'comparison': comparison,
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
