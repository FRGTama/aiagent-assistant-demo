import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()

class Memory:
    def __init__(self: str, embed_url = os.getenv("SERVER_ENDPOINT"), embed_model = os.getenv("EMBEDDING_MODEL")):
        self.client = chromadb.HttpClient(host=os.getenv("HOST"),port=int(os.getenv("CHROMA_PORT")))
        self.ef = embedding_functions.OllamaEmbeddingFunction(
            url= embed_url,
            model_name = embed_model
        )
        self.what_worked = set()
        self.what_to_avoid = set()

    def __str__(self):
        print("hello")

    def update_ww(self, data: str):
        self.what_worked.update(data)
    def update_wta(self,data: str):
        self.what_to_avoid.update(data)

    def vectorize(self,data):
        return self.ef(data)

    def get_instance(self):
        return self.client

    def clear_col(self,col):
        self.client.delete_collection(col)
        pass

    def get_col(self,col_name: str):
        return self.get_instance().get_or_create_collection(
            name=col_name,
            embedding_function=self.ef
        )

    def add_mem(self, col_name: str, mem: list[str], reflection):
        col = self.get_col(col_name)
        ids=f'id{str(col.count())}'
        col.add(
            documents=mem,
            ids=ids,
            embeddings=self.vectorize(mem),
            metadatas=[
                {
                    "context_tags": reflection['context_tags'],
                    "conversation_summary": reflection['conversation_summary'],
                    "what_worked": reflection['what_worked'],
                    "what_to_avoid": reflection['what_to_avoid']
                }
            ]
        )
    def query_mem(self,col_name: str, query: str,where=None):
        col = self.get_col(col_name)
        result = col.query(
            query_texts=query,
            n_results=1
        )
        if where:
            result = col.query(
                query_texts=query,
                n_results=1,
                where=where
            )
        return result

    # def get_mem(self, col_name: str):
    #     col = self.get_col(col_name)
    #     def get_id_list_all():
    #         return [f'id{x}' for x in range(0,col.count())]
    #     ids=get_id_list_all()
    #     return col.get(
    #         ids=ids
    #     )

    def update_mem(self,col_name: str, new_mem: list[str]):
        col = self.get_col(col_name)
        old_mem = self.query_mem(col_name,new_mem[0])
        col.update(new_mem)