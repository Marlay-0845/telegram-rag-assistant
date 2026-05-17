from rag_tg_project.rag.retrieval_config import SECTION, HYBRID
from rag_tg_project.rag.validators import get_unique_and_relevant_docs
from rag_tg_project.rag.vectorstore import vectorstore_manager



vectorstore = vectorstore_manager.get()

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 22,
        "fetch_k": 32,
    },
)


def section_retriever(question_and_section):
    retrieved_docs_from_section = vectorstore.as_retriever(
            search_kwargs={
                    "k": 22,
                    "filter": {"section": question_and_section.section}
                }
            ).invoke(question_and_section.question)
    
    return retrieved_docs_from_section


def retrieve(r_question_and_section, retrieval_mode):
    if retrieval_mode == SECTION:
        retrieved_docs = section_retriever(question_and_section=r_question_and_section)
    elif retrieval_mode == HYBRID:
        retrieved_docs_from_section = section_retriever(question_and_section=r_question_and_section)
        retrieved_docs_from_global = default_retriever(r_question_and_section.question)

        retrieved_docs = retrieved_docs_from_section + retrieved_docs_from_global

        retrieved_docs = get_unique_and_relevant_docs(retrieved_docs)
    else:
        retrieved_docs = default_retriever(r_question_and_section.question)

    return retrieved_docs


def default_retriever(docs):
    return mmr_retriever.invoke(docs)