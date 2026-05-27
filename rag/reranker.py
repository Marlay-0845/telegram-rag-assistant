from sentence_transformers import CrossEncoder

import rag_tg_project.config as cf



def reranking_retrieval_docs(query, docs, top_k):
    model = CrossEncoder(cf.RERANKING_MODEL)

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    paired = list(zip(scores, docs))
    paired.sort(key=lambda x: x[0], reverse=True)

    top_docs_scores = paired[:top_k]
    top_docs = [doc[1] for doc in top_docs_scores]

    return top_docs