# Telegram RAG Assistant

Telegram bot with Retrieval-Augmented Generation (RAG) for answering user questions.

## Features

- Question rewriting
- Section-based retrieval
- Hybrid retrieval
- Vector search
- Context generation
- Multi-user support
- Evaluation system

## Stack

- Python
- Aiogram
- LangChain
- ChromaDB
- Ollama

## Installation

```bash
git clone ...
```

```bash
pip install -r requirements.txt
```

Create `.env`

In the config file, replace the directories with your own 

## Run

```bash
python rag/scripts/ingest.py
```

```bash
python app/bot.py
```

## Project structure

```text
app/
rag/
storage/
config.py
```

## Future improvements

- caching
- database
- monitoring
