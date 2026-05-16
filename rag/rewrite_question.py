from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

import rag_tg_project.config as cf
from rag_tg_project.schemas.rewrite_model import RewriteQuestion



llm = ChatOpenAI(
    api_key='ollama',
    base_url=cf.OLLAMA_BASE_URL_V1,
    model=cf.LLM_MODEL,
    temperature=0.2,
    max_completion_tokens=512,
    top_p=0.9
)


question_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a query preprocessor for a RAG system about Antarctic Wallet.\n"
        "\n"
        "Your task:\n"
        "1. Rewrite the user's question to make it clear, self-contained, and optimized for semantic search.\n"
        "2. Classify the question into the most relevant section.\n"
        "\n"
        "Return ONLY valid JSON. No explanations, no extra text.\n"
        "\n"
        "JSON format:\n"
        "{{\n"
        '  "question": "rewritten question",\n'
        '  "section": "one_of_the_allowed_sections_or_null"\n'
        "}}\n"
        "\n"
        "Allowed sections:\n"
        "ref-program, faq, contact, company, install, amba-program, offer-concierge\n"
        "\n"
        "Rules:\n"
        "- Always return both fields: question and section\n"
        "- If the question is already good, keep it but still rewrite minimally if needed\n"
        "- If no section fits, return null for section\n"
        "- Do NOT invent new sections\n"
        "- Always include 'Antarctic Wallet' when it improves clarity\n"
        "- Output must be valid JSON only"
    ),
    (
        "human", "{question}"
    )

])


question_rewrite_chain = (
    question_rewrite_prompt
    | llm
    | PydanticOutputParser(pydantic_object=RewriteQuestion)
)


def rewrite_question_if_needed(question: str):
    result_model = question_rewrite_chain.invoke({"question": question})
    return result_model