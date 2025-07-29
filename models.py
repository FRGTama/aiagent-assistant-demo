import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

load_dotenv()

class Models:
    def __init__(self: str, ollama_model = "mannix/jan-nano", endpoint = os.getenv("LOCAL_ENDPOINT"), ):
        self.ollama_llm = OllamaLLM(model=ollama_model,base_url=endpoint)
        self.ocr_llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

    def get_ollama_llm(self):
        return self.ollama_llm

    def get_ocr_llm(self):
        return self.ocr_llm


global_models = Models(endpoint=os.getenv("SERVER_ENDPOINT"))