

import rag_tg_project.config as cf


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


def ensure_context(input_dict: dict) -> dict:
    context = input_dict.get("context", "").strip()
    if not context:
        input_dict["context"] = (
            "The context is empty: the search engine didn't find any relevant results. "
            "If the answer is important, it's best to explicitly let the user know. "
        )
    return input_dict