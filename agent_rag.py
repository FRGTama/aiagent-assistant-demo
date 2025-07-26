from langchain_chroma import Chroma
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_perplexity import ChatPerplexity
from ocr_bot import extract_text_from_pdf
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
llm = OllamaLLM(model="mannix/jan-nano",base_url=os.getenv("LOCAL_ENDPOINT"))
stupid_bot = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
collection_name = "rag"

file_path = "./docs/test.pdf"
perplexity_model = ChatPerplexity(
    model="r1-1776",   # or another supported model
    temperature=0.5
)
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

@tool
def store_vector_db(data):
    """stores the data into the vector database"""
    retriever.add_documents(documents=data)

@tool
def ocr(file_path):
    """Extract text data from a file, input argument is file_path which contain the info on the location of file you need to perform ocr on"""
    ## output_dir: "./output/ocr/."
    content = extract_text_from_pdf(file_path)
    return content

@tool
def write_to_file(content:str):
    """take in the argument content and save to disk"""
    with open("output.txt", "w") as f:
        f.write(content)

@tool
def translate(content):
    """takes in content argument as text and translate content to vietnamese and return text data"""
    prompt = [HumanMessage(f"translate the given text to vietnamese, this is the content: {content}")]
    stupid_bot.invoke(prompt)
agent_tools = [query_vector_db,ocr,translate,write_to_file]
agent = initialize_agent(
    tools=agent_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

question = [HumanMessage("perform ocr to extract chinese texts on the given pdf file consisting of scanned images of a chinese book which has vertical formatting, the file path is ./docs/testpdf.pdf, after that saves the raw text to disk, using the write_to_disk tool, the output directory is hardcoded, just pass the content needed to be written into file as the argument")]
result = agent.invoke(question)
