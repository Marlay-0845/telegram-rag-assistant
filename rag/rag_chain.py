import os
import re
import logging
from pathlib import Path
from dotenv import load_dotenv
from urllib.parse import urlsplit
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

from rag_tg_project.rag.rewrite_question import rewrite_question_if_needed
from rag_tg_project.rag.retrieval_config import section_keywords, SECTION, HYBRID
import rag_tg_project.config as cf


load_dotenv()
logger = logging.getLogger("rag_tg_project/logs")



def format_docs(docs):
    formatted = []
    total_len = 0

    for doc in docs:
        source = doc.metadata.get("source", "unknown_source")
        page = doc.metadata.get("page", None)
        section = doc.metadata.get("section", None)

        header = f"Source: {source}"
        if page is not None:
            header += f" | Page: {page}"

        if section is not None:
            header += f" | Section: {str(section)}"

        text = doc.page_content.strip()
        block = f"{header}\n{text}"

        if total_len + len(block) > cf.MAX_CHARS_IN_FORMATTED_PART:
            break

        formatted.append(block)
        total_len += len(block)

    return "\n\n---\n\n".join(formatted)


def doc_to_text_cleaner(docs):
    for doc in docs:
        clean_text = re.sub(r'\[\{.*?\}\]', '', doc.page_content)
        doc.page_content = clean_text

    return docs


def add_section_to_text_part(text_parts):
    for text_part in text_parts:
        url = urlsplit(text_part.metadata['source'])
        url_path = url.path
        if url_path == '/':
            text_part.metadata['section'] = ['main']
        else:
            result = url_path.strip('/').split('/')
            text_part.metadata['section'] = result


def ensure_context(input_dict: dict) -> dict:
    context = input_dict.get("context", "").strip()
    if not context:
        input_dict["context"] = (
            "The context is empty: the search engine didn't find any relevant results. "
            "If the answer is important, it's best to explicitly let the user know. "
        )
    return input_dict


def check_retriever_quality(retrieved_docs):
    if len(retrieved_docs) <= 3:
        return False
    
    return True

def get_unique_and_relevant_docs(retrieved_docs):
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        key = doc.page_content + doc.metadata.get("source", "")
        if key in seen:
            continue
        
        seen.add(key)
        unique_docs.append(doc)
    
    return unique_docs[:8]


def check_keywords_in_question(question_and_section):
    if question_and_section.section is None:
        return None
    
    question_lower = question_and_section.question.lower()
    keywords_matches = 0

    for i in section_keywords[question_and_section.section]:
        if i in question_lower:
            keywords_matches += 1
    logger.info(f"Matched keywords: {keywords_matches}")

    if keywords_matches >= 2:
        return SECTION
    elif keywords_matches == 1:
        return HYBRID
    else:
        return None
    

def section_retriever(question_and_section):
    retrieved_docs_from_section = vectorstore.as_retriever(
            search_kwargs={
                    "k": 8,
                    "filter": {"section": question_and_section.section}
                }
            ).invoke(question_and_section.question)
    
    return retrieved_docs_from_section


def retrieve(r_question_and_section, retrieval_mode):
    if retrieval_mode == SECTION:
        retrieved_docs = section_retriever(question_and_section=r_question_and_section)
    elif retrieval_mode == HYBRID:
        retrieved_docs_from_section = section_retriever(question_and_section=r_question_and_section)
        retrieved_docs_from_global = mmr_retriever.invoke(r_question_and_section.question)

        retrieved_docs = retrieved_docs_from_section + retrieved_docs_from_global

        retrieved_docs = get_unique_and_relevant_docs(retrieved_docs)
    else:
        retrieved_docs = mmr_retriever.invoke(r_question_and_section.question)

    return retrieved_docs


def get_user_question_and_return_answer(user_question):
    logger.info(f"User question: {user_question}")
    right_question_and_section = rewrite_question_if_needed(question=user_question)
    logger.info(f"Rewrite question: {right_question_and_section.question}")
    logger.info(f"Section for question: {right_question_and_section.section}")

    retrieval_mode = check_keywords_in_question(question_and_section=right_question_and_section)
    logger.info(f"Retrieval strategy: {retrieval_mode}")

    retrieved_docs = retrieve(r_question_and_section=right_question_and_section, retrieval_mode=retrieval_mode)
    logger.info(f"Docs count: {len(retrieved_docs)}")

    verified = check_retriever_quality(retrieved_docs)
        
    if not verified:
        retrieved_docs = mmr_retriever.invoke(right_question_and_section.question)
        logger.info(f"Data reorganization, number of documents: {len(retrieved_docs)}")

    ctx = format_docs(retrieved_docs)
    logger.info(f"Context len: {len(ctx)}")

    answer = answer_question(question=right_question_and_section.question, context=ctx)

    return answer


@traceable(name="AW_answer_question")
def answer_question(question: str, context: str) -> str:
    inputs = {
        "question": question,
        "context": context,
    }
    return rag_chain.invoke(inputs)


os.environ["USER_AGENT"] = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"

sitemap_loader = SitemapLoader(
    web_path=cf.SITEMAP_URL,
    filter_urls=[cf.ROOT_URL],
)
sitemap_docs = sitemap_loader.load()

clean_sitemap_docs = doc_to_text_cleaner(docs=sitemap_docs)

add_section_to_text_part(clean_sitemap_docs)

token_splitter = TokenTextSplitter(
    chunk_size=256,
    chunk_overlap=32,
)
text_parts = token_splitter.split_documents(clean_sitemap_docs)


embeddings = OllamaEmbeddings(
    model=cf.EMBEDDING_MODEL,
    base_url=cf.OLLAMA_BASE_URL,
)


if Path(cf.persist_directory).exists():
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=cf.persist_directory
    )
else:
    vectorstore = Chroma.from_documents(
        documents=text_parts,
        embedding=embeddings,
        persist_directory=cf.persist_directory
    )


mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,      
        "fetch_k": 32,
    },
)


promt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant that RESPONDS STRICTLY IN ENGLISH. "
        "Even if the context or question is partly in other languages. "
        "Use the following context snippets to answer the question. "
        "If there's no answer in the context or the information is insufficient, just be honest and ask them to contact customer support. "
        "Always cite the source using the format from the header (Source: a link where you can read more about it ). "
        "Keep your answers brief and to the point—usually 5 to 7 sentences. "
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "Context: {context}\n\nQuestion: {question}"
    ),
])


llm = ChatOpenAI(
    api_key='ollama',
    base_url=cf.OLLAMA_BASE_URL_V1,
    model=cf.LLM_MODEL,
    temperature=0.2,
    max_completion_tokens=512,
    top_p=0.9
)


rag_chain = (
    # {
    #     "context": mmr_retriever | format_docs, 
    #     "question": RunnablePassthrough(),
    #     "history": lambda _: [],
    # }
    {
        "context": lambda d: d.get("context", ""),
        "question": lambda d: d.get("question", ""),
        "history": lambda _: [],
    }
    | RunnableLambda(ensure_context)
    | promt
    | llm
    | StrOutputParser()
).with_config(run_name="rag_chain")