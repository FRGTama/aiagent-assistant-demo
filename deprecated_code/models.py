import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_perplexity import ChatPerplexity
from langchain_ollama import OllamaLLM

load_dotenv()

class Models:
    def __init__(self: str, ollama_model = "mannix/jan-nano", endpoint = os.getenv("LOCAL_ENDPOINT"), translation_model = "gemini-2.5-pro", ocr_model = "gemini-2.5-pro", agent_model = "sonar"):
        self.__ollama_llm = OllamaLLM(model=ollama_model,base_url=endpoint)
        self.__ocr_llm = ChatGoogleGenerativeAI(model=ocr_model)
        self.__trans_llm = ChatGoogleGenerativeAI(model=translation_model)
        self.__agent_llm = ChatPerplexity(model=agent_model,timeout=300)

    def get_ollama_llm(self):
        return self.__ollama_llm

    def get_ocr_llm(self):
        return self.__ocr_llm

    def get_trans_llm(self):
        return self.__trans_llm

    def get_agent_llm(self):
        return self.__agent_llm
global_models = Models(endpoint=os.getenv("LOCAL_ENDPOINT"),ocr_model="gemini-2.0-flash-lite",translation_model="gemini-2.0-flash-lite")