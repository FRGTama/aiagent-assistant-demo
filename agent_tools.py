import pymupdf
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import models
from ocr_bot import extract_text_from_pdf
load_dotenv()

collection_name = "rag"
vector_store = Chroma(
    collection_name=collection_name,
    embedding_function=OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"), base_url=os.getenv("SERVER_ENDPOINT")),
    persist_directory="./chroma_db",
)
retriever = vector_store.as_retriever()
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


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
    with open("./output/output.txt", "w") as f:
        f.write(content)

@tool
def translate(content):
    """takes in content argument as text and translate content to vietnamese and return text data"""
    prompt = [HumanMessage(f"translate the given text to vietnamese, this is the content: {content}")]
    models.global_models.get_trans_llm().invoke(prompt)

@tool
def search_the_web(query):
    """Uses a search tool to look up relevant information on the internet, this tool has one argument that takes in the search query"""
    respond = tavily_client.search(query)
    return respond

@tool
def extract_text_from_pdf(path):
    """Use a function to extract all the text element from a pdf file and returns a string of all the text contents"""
    doc = pymupdf.open(path)
    content = ""
    for page in doc:
        content += page.get_text()
    return content

tool_list = [query_vector_db,ocr,translate,write_to_file,search_the_web, extract_text_from_pdf]