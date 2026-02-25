import json
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv


load_dotenv()

def load_pdf(pdf_path: str):
        loader = PyPDFLoader(pdf_path)
        return loader.load()

path = "code/romeo-juliet/pdf/romeo-and-juliet.pdf"
documents = load_pdf(path)

openai_api_key=os.getenv("OPENAI_API_KEY")
openai_client = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=openai_api_key
        )
graph_transformer = LLMGraphTransformer(llm=openai_client)
graph_docs = graph_transformer.convert_to_graph_documents(documents)

with open('graph_docs_debug.json', 'w') as f:
    json.dump([doc.model_dump() for doc in graph_docs], f, indent=2, default=str)