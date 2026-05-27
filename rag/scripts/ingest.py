from rag_tg_project.rag.ingestion import build_ingestion_pipeline
from rag_tg_project.rag.vectorstore import vectorstore_manager
from langchain_core.documents import Document

vectorstore = vectorstore_manager.get()
ordered_vs = vectorstore.get()


def main():
    chunks = build_ingestion_pipeline()
    vectorstore_manager.build_vectorstore(chunks)

    vectorstore = vectorstore_manager.get()
    data = vectorstore.get()

    for i, chunk_id in enumerate(data["ids"]):
        doc = Document(
        page_content=data["documents"][i],
        metadata=data["metadatas"][i]
        )
        doc.metadata['chunk_id'] = chunk_id
        vectorstore.update_document(chunk_id, doc)



if __name__ == "__main__":
    main()