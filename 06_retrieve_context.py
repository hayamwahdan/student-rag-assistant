"""
Step 6: Retrieve relevant chunks from the Chroma store for a user question.
Uses Maximal Marginal Relevance (MMR) and filters out low-quality
(e.g. Table of Contents / index) chunks that slipped through at store-build time.
"""

import importlib
from pathlib import Path
from langchain_chroma import Chroma

vector_module = importlib.import_module("04_vector_representation")
chunking_module = importlib.import_module("03_chunking")

CHROMA_PATH = Path(__file__).parent / "chroma_db"

embedding_model = vector_module.get_embedding_model()

if not CHROMA_PATH.exists():
    print(f"WARNING: Chroma DB not found at {CHROMA_PATH}. Run 05_create_chroma_store.py first.")

vector_store = Chroma(
    persist_directory=str(CHROMA_PATH),
    embedding_function=embedding_model,
)


def retrieve_context(query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5):
    """Return up to k relevant, cleaned document chunks for a query."""
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": fetch_k,           # over-fetch so we still have k left after filtering junk
            "fetch_k": fetch_k * 2,
            "lambda_mult": lambda_mult,
        },
    )
    candidates = retriever.invoke(query)

    filtered = [
        doc for doc in candidates
        if not chunking_module.is_low_quality_chunk(doc.page_content)
    ]

    return (filtered or candidates)[:k]


if __name__ == "__main__":
    test_query = "What is overfitting?"
    results = retrieve_context(test_query)
    print(f"Retrieved {len(results)} chunks for: {test_query}\n")
    for i, doc in enumerate(results, start=1):
        print(f"[{i}] {doc.metadata.get('source', 'unknown')} (page {doc.metadata.get('page', 'N/A')})")
        print(doc.page_content[:200])
        print("-" * 50)
