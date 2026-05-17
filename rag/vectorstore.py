from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

import rag_tg_project.config as cf



class ArcticEmbeddings(OllamaEmbeddings):
    def embed_query(self, text: str,) -> list[float]:
        return super().embed_query("query: " + text)



class VectorStoreManager:
    def __init__(self):
        self.vectorstore = None

        self.embeddings = ArcticEmbeddings(
            model=cf.EMBEDDING_MODEL,
            base_url=cf.OLLAMA_BASE_URL,
        )


    def build_vectorstore(self, chunks):
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=cf.persist_directory
        )

        return self.vectorstore


    def get(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=cf.persist_directory
            )

        return self.vectorstore
    


vectorstore_manager = VectorStoreManager()