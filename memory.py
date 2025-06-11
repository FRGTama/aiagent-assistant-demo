import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()

class Memory:
    def __init__(self: str, embed_url = os.getenv("EMBEDDING_ENDPOINT"), embed_model = os.getenv("EMBEDDING_MODEL")):
        self.client = chromadb.HttpClient(host=os.getenv("HOST"),port=int(os.getenv("CHROMA_PORT")))
        self.ef = embedding_functions.OllamaEmbeddingFunction(
            url= embed_url,
            model_name = embed_model
        )

    def __str__(self):
        print("hello")

    def vectorize(self,data):
        return self.ef(data)

    def get_instance(self):
        return self.client

    def get_col(self,col_name: str):
        return self.get_instance().get_collection(
            name=col_name
        )

    def add_mem(self, col_name: str, mem: list[str]):
        col = self.get_col(col_name)
        def make_id_list():
            return [f'id{x}' for x in [x for x in range(col.count(),col.count() + len(mem))]]
        ids=make_id_list()
        print(ids)
        col.add(
            documents=mem,
            ids=ids,
            embeddings=self.vectorize(mem)
        )

    def get_mem(self, col_name: str):
        col = self.get_col(col_name)
        def get_id_list_all():
            return [f'id{x}' for x in range(0,col.count())]
        ids=get_id_list_all()
        return col.get(
            ids=ids
        )

    def update_mem(self,col_name: str, new_mem: list[str]):
        col = self.get_col(col_name)




mem0 = Memory()
print(mem0.get_mem("dev_test"))
