import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_tg_project.rag.vectorstore import vectorstore_manager
import rag_tg_project.config as cf
from rag_tg_project.rag.retrieval_config import section_keywords, SECTION, HYBRID


vectorstore = vectorstore_manager.get()

ordered_vs = vectorstore.get()
def check_keywords_in_question(question, section):
    if section is None:
        return None
    if section == "main":
        return HYBRID
    
    question_lower = question.lower()
    keywords_matches = 0

    for i in section_keywords[section]:
        if i in question_lower:
            keywords_matches += 1

    if keywords_matches >= 2:
        return SECTION
    elif keywords_matches == 1:
        return HYBRID
    else:
        return None
    

def section_retriever(question, section):
    retrieved_docs_from_section = vectorstore.as_retriever(
            search_kwargs={
                    "k": 84,
                    "filter": {"section": section}
                }
            ).invoke(question)
    
    return retrieved_docs_from_section


def retrieve(question, section, retrieval_mode):
    if retrieval_mode == SECTION:
        retrieved_docs = section_retriever(question, section)
    elif retrieval_mode == HYBRID:
        retrieved_docs_from_section = section_retriever(question, section)
        retrieved_docs_from_global = default_retriever(question)

        retrieved_docs = retrieved_docs_from_section + retrieved_docs_from_global
    else:
        retrieved_docs = default_retriever(question)

    return retrieved_docs


def default_retriever(question):
    return mmr_retriever.invoke(question)
    

llm = ChatOpenAI(
    api_key='ollama',
    base_url=cf.OLLAMA_BASE_URL_V1,
    model=cf.LLM_MODEL,
    temperature=0.2,
    max_completion_tokens=512,
    top_p=0.9
)

mmr_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 84,
        "fetch_k": 84,
    },
)


question_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You create questions based on the provided text./n"
        "/n"
        "Your task:/n"
        "Generate a question that this section of text answers./n"
        "Don't make anything up, stick strictly to the text."
    ),
    (
        "human", "{chunk}"
    )

])


question_rewrite_chain = (
    question_rewrite_prompt
    | llm
    | StrOutputParser()
)


with open(cf.file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


questions = list(data.values())

ids = list(data.keys())
result = 0
count = 0
for i in ordered_vs["metadatas"]:
    retrieval_mode = check_keywords_in_question(question=questions[count], section=i['section'])

    retrieved_docs = retrieve(question=questions[count], section=i['section'], retrieval_mode=retrieval_mode)

    retrieved_ids = [doc.metadata['chunk_id'] for doc in retrieved_docs]

    if ids[count] in retrieved_ids:
        result += 1

    count += 1

print(f"Number of hits: {result}, total: {len(ordered_vs["ids"])}")

# def answers_to_questions(ordered_vs):
#     my_dict = {}
#     for i, chunk_id in enumerate(ordered_vs["ids"]):
#         result_model = question_rewrite_chain.invoke({"chunk": ordered_vs["documents"][i]})
#         my_dict[chunk_id] = result_model
#         print(f"Still to be processed: {int(len(ordered_vs["ids"]) - i)}")

#     return my_dict


# result = answers_to_questions(ordered_vs)
# with open('my_dict.json', 'w', encoding='utf-8') as file:
#     json.dump(result, file, ensure_ascii=False, indent=4)