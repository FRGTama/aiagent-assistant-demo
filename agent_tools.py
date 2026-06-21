import base64
import pymupdf
import os

from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings
from tavily import TavilyClient
from dotenv import load_dotenv
import models
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
def ocr(image_file_path):
    """Extract text data from an image,
    ocr(image_file_path);
    input:
        image_file_path: path to the image file
    output:
        str: a string containing the raw text contents of the image
     """
    with open(image_file_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    message_local = HumanMessage(
        content=[
            {"type": "text", "text": "Extract all text from the given image, return the raw text."},
            {"type": "image_url", "image_url": f"data:image/png;base64,{encoded_image}"},
        ]
    )
    result_local = models.global_models.get_ocr_llm().invoke([message_local])
    return result_local.content

@tool
def extract_text_from_pdf_images(path):
    """This function is used to extract text from a pdf file consisting of images and saves them to ./output/ocr/page-<page number>,
        extract_text_from_pdf_images(path);
        input:
            path: is the path to the pdf file consisting of images
        output:
            str: a list of strings containing the raw text contents of each page,
            each element in the list is the raw text from each page
    """

    doc = pymupdf.open(path)  # open document
    content = []
    for page in doc:  # iterate through the pages
        pix = page.get_pixmap()  # render page to an image
        pix.save("./output/ocr/page-%i.png" % page.number)  # store image as a PNG
        content.append([ocr("./output/ocr/page-%i.png" % page.number)])
    return content

@tool
def write_to_file(content:str):
    """saves content to disk onto a predetermined location
        write_to_file(content);
        input:
            content: string of text that need to be written to file
        output:
            has no output, the contents of the input is saved to a predetermined location
        """
    with open("./output/output.txt", "w") as f:
        f.write(content)

@tool
def translate_to_vn(content):
    """Translate text data to Vietnamese
        translate( content);
        intput:
            content: the text that needs to be translated
        output:
            str: a string of the translated text
    """
    prompt = [HumanMessage(f"translate the given text to Vietnamese, this is the content: {content}")]
    return models.global_models.get_trans_llm().invoke(prompt).content

@tool
def search_the_web(query):
    """Uses a search tool to look up relevant information on the internet, this tool has one argument that takes in the search query"""
    respond = tavily_client.search(query)
    return respond

@tool
def extract_text_from_pdf(path):
    """Use a function to extract all the text element from a text only pdf file and returns a string of all the text contents"""
    doc = pymupdf.open(path)
    content = ""
    for page in doc:
        content += page.get_text()
    return content

tool_list = [query_vector_db,ocr,translate_to_vn,write_to_file,search_the_web, extract_text_from_pdf, extract_text_from_pdf_images]