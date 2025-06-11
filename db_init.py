import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os
load_dotenv()

ollama_server = f'{os.getenv("HOST")}:{os.getenv("OLLAMA_PORT")}'
embedding_model = f'{os.getenv("EMBEDDING_MODEL")}'
ollama_ef = embedding_functions.OllamaEmbeddingFunction(
    url=ollama_server,
    model_name=embedding_model
)
def vectorize(data):
    return ollama_ef(data)
# chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMADB_PATH"))
chroma_client = chromadb.HttpClient(host=os.getenv("HOST"),port=int(os.getenv("CHROMA_PORT")))
