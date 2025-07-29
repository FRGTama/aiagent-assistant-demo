import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_perplexity import ChatPerplexity
from langchain_ollama import OllamaLLM

load_dotenv()

class Models:
    def __init__(self: str, ollama_model = "mannix/jan-nano", endpoint = os.getenv("SERVER_ENDPOINT"), translation_model = "gemini-2.5-pro"):
        self.__ollama_llm = OllamaLLM(model=ollama_model,base_url=endpoint)
        self.__ocr_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        self.__trans_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
        self.__agent_llm = ChatPerplexity(model="sonar",timeout=300)

    ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    def get_ollama_llm(self):
        return self.__ollama_llm

    def get_ocr_llm(self):
        return self.__ocr_llm

    def get_trans_llm(self):
        return self.__trans_llm

global_models = Models(endpoint=os.getenv("SERVER_ENDPOINT"))