# Student RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions using **only** the content of provided study documents (PDFs) — no hallucinated information. Built for a course assignment following the full RAG lab sequence: documents → preprocessing → chunking → embeddings → vector store → retrieval → prompting → deployed UI.

## How it works

```
PDFs (data/)
   ↓  01_documents.py       – load PDFs
   ↓  02_preprocessing.py   – clean extracted text
   ↓  03_chunking.py        – split into chunks, filter out junk (ToC/index pages)
   ↓  04_vector_representation.py – MiniLM embeddings
   ↓  05_create_chroma_store.py   – build & persist ChromaDB
   ↓  06_retrieve_context.py      – MMR retrieval + junk filtering
   ↓  07_prompting.py             – build prompt, call OpenRouter (gpt-4o-mini)
   ↓  streamlit_app.py            – chat UI with cited sources
```

- **Vector store:** ChromaDB (persistent, local)
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **LLM:** OpenRouter (`openai/gpt-4o-mini`), OpenAI-compatible API
- **UI:** Streamlit chat interface with a "View Sources" expander showing document + page number for every answer

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Add your PDFs to a `data/` folder in the project root.

3. Create a `.env` file (copy `.env.example`) and add your OpenRouter key:
   ```
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_MODEL=openai/gpt-4o-mini
   ```
   Get a key at [openrouter.ai/keys](https://openrouter.ai/keys).

4. Build the vector store (run once, or whenever `data/` changes):
   ```
   python 05_create_chroma_store.py
   ```

5. Test the pipeline in the terminal:
   ```
   python 07_prompting.py
   ```

6. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

## Deployment (Streamlit Cloud)

In your deployed app's **Secrets** panel, add:
```toml
OPENROUTER_API_KEY = "your_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```
`streamlit_app.py` reads these automatically at runtime — no key is ever stored in the repo.

## Project structure

```
├── data/                          # source PDFs (not committed if large/private)
├── 01_documents.py
├── 02_preprocessing.py
├── 03_chunking.py
├── 04_vector_representation.py
├── 05_create_chroma_store.py
├── 06_retrieve_context.py
├── 07_prompting.py
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Notes

- Chunks resembling Table of Contents / index pages are automatically filtered out at both store-build and retrieval time (`is_low_quality_chunk` in `03_chunking.py`).
- Retrieval uses Maximal Marginal Relevance (MMR) to reduce duplicate/near-duplicate chunks.
- If the answer isn't in the provided documents, the assistant responds "I don't know" instead of guessing.
