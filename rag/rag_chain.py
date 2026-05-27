import asyncio
import logging
from dotenv import load_dotenv

from rag_tg_project.rag.rewrite_question import rewrite_question_if_needed
from rag_tg_project.rag.retrieval import retrieve, default_retriever
from rag_tg_project.rag.formatting import format_docs
from rag_tg_project.rag.chain import answer_question
from rag_tg_project.rag.validators import check_keywords_in_question, check_retriever_quality
from rag_tg_project.rag.reranker import reranking_retrieval_docs


load_dotenv()
logger = logging.getLogger("rag_tg_project/logs")



async def get_user_question_and_return_answer(user_question):
    logger.info(f"User question: {user_question}")
    right_question_and_section = await rewrite_question_if_needed(question=user_question)
    logger.info(f"Rewrite question: {right_question_and_section.question}")
    logger.info(f"Section for question: {right_question_and_section.section}")

    retrieval_mode = check_keywords_in_question(question_and_section=right_question_and_section)
    logger.info(f"Retrieval strategy: {retrieval_mode}")

    retrieved_docs = await retrieve(r_question_and_section=right_question_and_section, retrieval_mode=retrieval_mode)
    logger.info(f"Docs count: {len(retrieved_docs)}")

    verified = check_retriever_quality(retrieved_docs)
        
    if not verified:
        retrieved_docs = await default_retriever(right_question_and_section.question)
        logger.info(f"Data reorganization, number of documents: {len(retrieved_docs)}")

    scores = await asyncio.to_thread(reranking_retrieval_docs, right_question_and_section.question, retrieved_docs, 8)
    logger.info(f"Score docs count: {len(scores)}")

    ctx = format_docs(scores)
    logger.info(f"Context len: {len(ctx)}")

    answer = await answer_question(question=right_question_and_section.question, context=ctx)

    return answer