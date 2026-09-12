# AI RAG Document Chat

A beginner-friendly Retrieval-Augmented Generation (RAG) application.

## Features
- Preloaded PDFs from `documents/`
- Multiple user-uploaded PDFs
- PDF text extraction
- Text chunking with overlap
- Gemini embeddings
- ChromaDB vector search
- Gemini grounded answers
- Source document and page display

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Set `GEMINI_API_KEY` using an environment variable or Streamlit Secrets.

Never put your API key in GitHub.
