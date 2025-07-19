from langchain_chroma import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_core.messages import HumanMessage
import memory
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()
llm = OllamaLLM(model="mannix/jan-nano",base_url=os.getenv("LOCAL_ENDPOINT"))
collection_name = "rag"
mem = memory.Memory()

file_path = "./docs/test.pdf"

vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"), base_url=os.getenv("SERVER_ENDPOINT")),
    persist_directory="./chroma_db",
)

retriever = vector_store.as_retriever()
@tool
def query_vector_db(query):
    """Searches the vector database for relevant memory based on the input query string."""
    q_result = retriever.invoke(input=query)
    return q_result

def store_vector_db(data):
    """stores the data into the vector database"""
    retriever.add_documents(documents=data)

agent_tools = [query_vector_db]
agent = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

loader = PyPDFLoader(file_path)
pages = []
for page in loader.lazy_load():
    pages.append(page)
store_vector_db(pages)
question = [HumanMessage("What's my name")]
result = agent.invoke(question)
