import re
from urllib.parse import urlsplit
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import TokenTextSplitter

import rag_tg_project.config as cf



def add_section_to_text_part(docs):
    new_docs = []
    
    for doc in docs:
        url = urlsplit(doc.metadata['source'])
        url_path = url.path
        if url_path == '/':
            doc.metadata['section'] = 'main'
        else:
            result = url_path.strip('/').split('/')
            doc.metadata['section'] = result[0]
        
        new_docs.append(doc)

    return new_docs


def load_sitemap():
    sitemap_loader = SitemapLoader(
        web_path=cf.SITEMAP_URL,
        filter_urls=[cf.ROOT_URL],
    )
    return sitemap_loader.load()


def doc_to_text_cleaner(docs):
    for doc in docs:
        clean_text = re.sub(r'\[\{.*?\}\]', '', doc.page_content)
        doc.page_content = clean_text

    return docs


def split_docs(docs):
    token_splitter = TokenTextSplitter(
        chunk_size=256,
        chunk_overlap=32,
    )
    return token_splitter.split_documents(docs)


def build_ingestion_pipeline():
    docs = load_sitemap()
    docs = doc_to_text_cleaner(docs)
    docs = add_section_to_text_part(docs)
    chunks = split_docs(docs)

    return chunks