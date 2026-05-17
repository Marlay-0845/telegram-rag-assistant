import logging

from rag_tg_project.rag.retrieval_config import section_keywords, SECTION, HYBRID



logger = logging.getLogger("rag_tg_project/logs")


def get_unique_and_relevant_docs(retrieved_docs):
    seen = set()
    unique_docs = []
    for doc in retrieved_docs:
        key = doc.page_content + doc.metadata.get("source", "")
        if key in seen:
            continue
        
        seen.add(key)
        unique_docs.append(doc)
    
    return unique_docs[:22]


def check_retriever_quality(retrieved_docs):
    if len(retrieved_docs) <= 3:
        return False
    
    return True


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