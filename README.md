---
title: Multi-Modal RAG Study Agent
sdk: streamlit
python_version: "3.11"
app_file: app.py
pinned: false
---

# Multi-Modal RAG Study Agent

Build study materials (summaries, quizzes, flashcards, timestamps, study plans) from YouTube lectures, PDFs, notes, and transcripts.

## Features

- Multi-source ingestion (YouTube, PDF, notes, transcripts)
- Retrieval with FAISS + embeddings
- Task-specific agents: notes, quiz, flashcards, timestamps, citations, study plans
- Streamlit UI for Hugging Face Spaces

## Environment Variables

- `HF_TOKEN`: Hugging Face token for Inference API
- `EMBEDDING_MODEL`: Embedding model id
- `LLM_MODEL`: LLM id for generation

Local embeddings (optional)

- To run embeddings locally (no HF API), set `USE_LOCAL_EMBEDDINGS=1` in your `.env` and install `sentence-transformers` and `torch`.
- Example `.env` entries:
  - `USE_LOCAL_EMBEDDINGS=1`
  - `LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

Run a quick local embedding test:

```bash
pip install -r requirements.txt
python -m src.tools.test_embeddings
```

## Run locally

1. Create and activate a venv.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
