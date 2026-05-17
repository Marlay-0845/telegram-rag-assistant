from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langsmith import traceable

import rag_tg_project.config as cf
from rag_tg_project.rag.prompts import promt
from rag_tg_project.rag.formatting import ensure_context



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


@traceable(name="AW_answer_question")
def answer_question(question: str, context: str) -> str:
    inputs = {
        "question": question,
        "context": context,
    }
    return rag_chain.invoke(inputs)