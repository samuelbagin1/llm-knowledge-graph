link:
https://arxiv.org/pdf/2003.02320
description:
"Knowledge Graphs" - A comprehensive survey paper by Aidan Hogan et al. examining knowledge graphs as structured representations of information. Covers foundational concepts including graph-based data models, query languages, schema design, entity identity management, and contextual information representation. Explores both deductive and inductive methods for representing and extracting knowledge, along with techniques for creating, enhancing, and validating knowledge graphs. Published in ACM Computing Surveys, Vol. 54, No. 4 (2021).


link:
https://arxiv.org/pdf/1805.02262
description:
"Construction of the Literature Graph in Semantic Scholar" - Describes a large-scale system that organizes scientific literature into a structured network containing over 280 million nodes representing papers, authors, entities, and their relationships (authorship and citation connections). The work reduces literature graph construction into familiar NLP tasks including entity extraction and linking, and powers semantic search and discovery features on Semantic Scholar. Published at NAACL 2018 industry track.


link:
https://blog.langchain.com/exploring-prompt-optimization/
description:
"Exploring Prompt Optimization" - LangChain blog post examining systematic approaches to improving LLM prompts through automated optimization. Benchmarks five optimization techniques (few-shot prompting, meta-prompting, meta-prompting with reflection, prompt gradients, and evolutionary optimization) across five datasets using GPT-4o, O1, and Claude-3.5-Sonnet. Key finding: Claude-3.5-Sonnet emerged as the most reliable optimizer model. Demonstrates that prompt optimization is most effective on tasks where the underlying model lacks domain knowledge.


link:
https://lexpredict-lexnlp.readthedocs.io/en/latest/modules/extract/extract.html#pattern-based-extraction-methods
description:
LexNLP Extract Module Documentation - Describes the lexnlp.extract module for extracting structured data from unstructured textual sources. Supports multiple languages (English, German, Spanish) with pattern-based extraction capabilities for over 20 data types including financial data (amounts, money, currency, percentages), temporal information (dates, durations), legal references (acts, citations, regulations, courts), organizational entities, and geographic locations. Particularly useful for contract analysis and legal document processing.


link:
https://medium.com/@claudiubranzan/from-llms-to-knowledge-graphs-building-production-ready-graph-systems-in-2025-2b4aff1ec99a
description:
"From LLMs to Knowledge Graphs: Building Production-Ready Graph Systems in 2025" by Claudiu Branzan - Technical deep-dive into tools, approaches, and architectural patterns for extracting structured knowledge from unstructured text. Discusses AutoSchemaKG for autonomous knowledge graph construction, and the ATLAS system which constructed 900+ million nodes and 5.9 billion edges from 50 million documents with 95% semantic alignment. Key takeaway: fine-tuning economics favor scale with break-even at ~1,500 documents.


link:
https://medium.com/@yu-joshua/a-unified-framework-for-ai-native-knowledge-graphs-8bd823385fe6
description:
"A Unified Framework for AI-Native Knowledge Graphs" by Fanghua (Joshua) Yu - Proposes Generative Knowledge Modeling (GenKM), a comprehensive methodology introducing a modular four-stage architecture (Document, Entity-Relation, Cluster, Ontology) that unifies 40+ existing Graph RAG systems under a common formalism. Includes a generative operator algebra classifying 20+ operators into deterministic, generative, and embedding categories with explicit provenance tracking, and the GenKG Lifecycle for end-to-end knowledge graph governance.


link:
https://pub.towardsai.net/why-llms-fail-at-knowledge-graph-extraction-and-what-works-instead-dcb029f35f5b
description:
"Why LLMs Fail at Knowledge Graph Extraction (And What Works Instead)" - Analyzes key reasons LLMs struggle with knowledge graph extraction: information noise in real-world documents, lack of domain-specific knowledge leading to misinterpretation of correlative vs. causal relationships, hallucinations when generating relations between entities, and context/reasoning limitations where traditional RAG methods fall short for complex multi-layered questions requiring cross-referencing. Discusses solutions including careful prompt engineering, domain-specific fine-tuning, and validation mechanisms.


link:
https://pub.towardsai.net/building-a-self-updating-knowledge-graph-from-meeting-notes-with-llm-extraction-and-neo4j-b02d3d62a251
description:
"Building a Self-Updating Knowledge Graph From Meeting Notes with LLM Extraction and Neo4j" - Comprehensive walkthrough for turning meeting notes into a live-updating Neo4j knowledge graph using CocoIndex and LLMs. Features automated change tracking (unchanged files never hit LLM API), specialized collectors for different entity types (Meeting Nodes, Attendance Relationships, Task Decision Relationships, Task Assignment Relationships), and intelligent deduplication by primary-key fields.


link:
https://medium.com/@visrow/semantic-graphrag-implementation-guide-build-real-world-ai-knowledge-systems-with-neo4j-qdrant-9d272d2f99c4
description:
"Semantic GraphRAG Implementation Guide" by Vishal Mysore - Comprehensive tutorial demonstrating GraphRAG implementation combining Protégé, Neo4j, LLMs, and Qdrant. The GraphRAG ingestion pipeline combines Graph Database and Vector Database to improve RAG workflows, benefiting from both semantic search and relationship-based retrieval. Covers seamless import/export to Neo4j, natural language query translation to Cypher and SPARQL, and local vector storage with Qdrant for improved recall and precision.


link:
https://medium.com/@shereshevsky/building-knowledge-graphs-from-homers-iliad-open-domain-vs-schema-guided-extraction-ef0bf3874a33
description:
"Building Knowledge Graphs from Homer's Iliad: Open-Domain vs. Schema-Guided Extraction" by Alexander Shereshevsky - Practical comparison of two LLM-powered approaches for transforming ancient epic poetry into queryable knowledge. Open-domain extraction yielded 141 entity types and 1,356 relationship types; schema-guided extraction used 6 entity types and 19 relationship types but told a clearer story. Key insight: structure enables discovery rather than opposing it. Recommends hybrid approach: use open-domain to discover what matters, then constrain with schema for consistent extraction.


link:
https://arxiv.org/html/2509.04696v1
description:
"ODKE+: Ontology-Guided Open-Domain Knowledge Extraction with LLMs" - Presents a production-ready system for automatically extracting and integrating millions of facts from web sources into knowledge graphs. Uses a modular pipeline with five components: Extraction Initiator, Evidence Retriever, hybrid Knowledge Extractors, lightweight Grounder, and Corroborator. Processed over 9 million Wikipedia pages, successfully ingesting 19 million high-confidence facts while maintaining 98.8% precision across 195 different predicates.


link:
https://arxiv.org/pdf/2410.21306
description:
"Natural Language Processing for the Legal Domain: A Survey of Tasks, Datasets, Models, and Challenges" - Comprehensive survey reviewing 154 studies on NLP applications in legal practice and research. Addresses challenges of extensive document lengths, complex language, and limited open legal datasets. Explores NLP tasks including document summarization, named entity recognition, question answering, argument mining, text classification, and judgment prediction. Identifies sixteen open research challenges including bias detection/mitigation and need for interpretable, explainable systems. Published in ACM Computing Surveys (Volume 58, Issue 6, 2025).


link:
https://medium.com/@khrizellelascano/analysis-of-legal-texts-using-nlp-774913be9a03
description:
"Analysis of Legal Texts Using NLP" by Khrizelle Lascano - Project using NLP to identify relations to floodgate policies and outcomes within tort law. Successfully identified new floodgate key terms not previously considered, analyzed distribution of allowed vs. dismissed cases by grounds of claims, and found that floodgate policies are constrained to specific legal questions rather than being inter-referential. Discusses challenges of applying NLP to legal documents including cross-references and ambiguous/literal definitions.


link:
https://www.mdpi.com/2079-9292/13/3/648
description:
"A Survey on Challenges and Advances in Natural Language Processing with a Focus on Legal Informatics and Low-Resource Languages" - Published in Electronics journal (February 2024). Extensive literature review of NLP research focused on legislative documents, presenting current state-of-the-art NLP tasks related to Law Consolidation and highlighting challenges that arise in low-resource languages.


link:
https://arxiv.org/html/2508.06368v1
description:
"Automated Creation of the Legal Knowledge Graph Addressing Legislation on Violence Against Women" - Research presenting a system for automatically constructing specialized knowledge graphs focused on legal cases involving violence against women. Develops and compares two methodologies: a systematic bottom-up approach tailored to legal documents and an LLM-based solution. Extracts information from European Court of Justice sentences, integrating structured data extraction, ontology development, and semantic enrichment for predictive justice and legal decision-making support.


link:
https://ai.gov.uk/blogs/understanding-legislative-networks-building-a-knowledge-graph-of-uk-legislation/
description:
"Understanding Legislative Networks: Building a Knowledge Graph of UK Legislation" - UK Government's Incubator for AI (i.AI) developed Lex Graph for data-driven analysis of UK legislation. Contains over 820,000 nodes (legislation and provisions) and 2.2 million edges (connections between them). Maps the intricate web of relationships between different pieces of legislation, making it possible to uncover patterns and connections that might otherwise remain hidden. Code available on GitHub, data on Hugging Face.


link:
https://medium.com/neo4j/from-legal-documents-to-knowledge-graphs-ccd9cb062320
description:
"From Legal Documents to Knowledge Graphs" by Tomaz Bratanic (Neo4j) - Comprehensive example of legal document processing pipeline using LlamaParse for PDF parsing, LLM for contract type classification, LlamaExtract for attribute extraction tailored to each contract category, and Neo4j for knowledge graph storage. Addresses how legal documents are inherently interconnected with complex webs of references between cases, statutes, regulations, and precedents that traditional vector search often fails to capture effectively.


link:
https://znalosti.gov.sk
description:
Slovak Government Central Knowledge Portal - A portal for machine-processable knowledge for information systems of public administration, developed and operated in testing mode by the Data Office of MIRRI (Ministry of Investments, Regional Development and Informatization of the Slovak Republic). Available under the EUPL license with source code publicly available on GitHub under the Dátová kancelária (Data Office) organization.


link:
https://arxiv.org/pdf/2512.20136
description:
"M³KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation" - Addresses limitations in multimodal RAG for audio-visual content. Introduces two main components: Multi-hop MMKG Construction (a lightweight multi-agent pipeline building knowledge graphs with context-enriched triplets of multimodal entities) and GRASP (Grounded Retrieval And Selective Pruning) for accurate entity alignment, relevance assessment, and redundant knowledge removal. Significantly enhances multimodal reasoning and grounding over existing approaches.


link:
https://arxiv.org/pdf/2404.19234
description:
"Multi-hop Question Answering over Knowledge Graphs using Large Language Models" by Abir Chakraborty - Explores how LLMs can answer complex questions about knowledge graphs requiring reasoning across multiple steps. Addresses the challenge that different KG sizes require different approaches for extracting and feeding relevant information to LLMs with fixed context windows. Demonstrates that LLMs can effectively adopt existing KG querying methods including information-retrieval based techniques and semantic parsing approaches (SPARQL generation). Tests approaches across six different knowledge graphs.


link:
https://medium.com/neo4j/knowledge-graphs-llms-multi-hop-question-answering-322113f53f51
description:
"Knowledge Graphs & LLMs: Multi-Hop Question Answering" by Tomaz Bratanic (Neo4j) - Part of Neo4j's NaLLM project exploring practical uses of LLMs with Neo4j. Demonstrates that multi-hop question-answering issues can be solved by preprocessing data before ingestion and connecting it to a knowledge graph. Knowledge graphs store data as a network of nodes and relationships, allowing RAG apps to navigate efficiently from one piece of information to another. Benefits include improved query efficiency, multi-hop reasoning capabilities, and support for both structured and unstructured information.
