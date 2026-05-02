import asyncio
import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'code'))
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from classes import Schema
from pdf_graphrag import PDFGraphRAG
from to_json import odd_to_json, refinement_to_json, sde_to_json

load_dotenv()

pdf_path = './code/assets/ZZ_2004_222_20260101.pdf'
graphrag = PDFGraphRAG(
    neo4j_uri='neo4j://127.0.0.1:7687',
    neo4j_user='neo4j',
    neo4j_password='fseijkfbsj48@',
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    database="colota"
)

name = "test-chunk-313"

# Load PDF documents
documents = graphrag.load_pdf(pdf_path)
# documents = documents[10:25]

# ------ open domain detection ------

# splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
# chunked_documents = splitter.split_documents(documents)

# # chunked_documents = chunked_documents[3:4]

# extracted_schema_list = asyncio.run(
#     graphrag.async_open_domain_detection(
#         chunked_documents,
#     )
# )

# print(f"\nAll chunks processed into list of schema.")
# odd_to_json(extracted_schema_list, name = name, chunks = chunked_documents)




# # ------ schema refinement ------

# # with open('./extracted_data/schemas_20260312002414.json') as f:
# #     data = json.load(f)

# # extracted_schema = Schema(
# #     nodes=data[1]['nested']['nodes'],
# #     relationships=data[1]['nested']['relationships']
# # )

# merged_schema = Schema(
#     nodes=[n for s in extracted_schema_list for n in s.nodes],
#     relationships=[r for s in extracted_schema_list for r in s.relationships]
# )
# refined_schema_data = graphrag.schema_refinement(odd_schema=merged_schema)

# print(f"\nRefined schema.")
# refinement_to_json(refined_schema_data, name = name)




random_number_list = [165, 492, 77, 311, 529, 44, 586, 203, 9, 367,
258, 114, 599, 421, 32, 188, 540, 73, 276, 5,

394, 221, 337, 60, 483, 146, 517, 28, 602, 351,
89, 270, 413, 7, 335, 568, 192, 124, 439, 82,

299, 555, 48, 233, 14, 382, 175, 603, 267, 96,
427, 320, 531, 63, 211, 501, 36, 143, 459, 19,

372, 246, 585, 109, 298, 534, 56, 187, 448, 92,
261, 516, 38, 226, 11, 365, 153, 578, 101, 327,

69, 414, 23, 309, 552, 132, 473, 58, 205,
341, 97, 520, 27, 286, 601, 119, 389, 45, 254]

# ------ schema driven extraction ------
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunked_documents = splitter.split_documents(documents)

# chunked_documents = chunked_documents[313:314]

with open('./extracted_data/test-10_ref_20260501185709.json') as f:
    data = json.load(f)

refined_schema = Schema(
    nodes=data['node_types'],
    relationships=data['relationship_types']
)

# refined_schema = Schema(
#     nodes=refined_schema_data['node_types'],  # type: ignore[index]
#     relationships=refined_schema_data['relationship_types']  # type: ignore[index]
# )

for i in random_number_list:
    graph_docs = asyncio.run(
        graphrag.async_schema_driven_extraction(
            chunked_documents[i:i+1],
            schema=refined_schema,
            document_id="ZZ_222_2004"
        )
    )
    
    sde_to_json(graph_docs, './test_dataset/', name = str(i))


# graph_docs = asyncio.run(
#     graphrag.async_schema_driven_extraction(
#         chunked_documents,
#         schema=refined_schema,
#         document_id="ZZ_222_2004"
#     )
# )

print(f"\nAll chunks processed into graph documents.")
# sde_to_json(graph_docs, name = name)


# graphrag.graph.add_graph_documents(
#     graph_documents=graph_docs,
#     include_source=False,
#     baseEntityLabel=True
# )