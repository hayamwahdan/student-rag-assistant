"""
Step 5: Build and persist the Chroma vector store from the source PDFs.
Run this once (or whenever /data changes) to regenerate chroma_db/.
"""

import importlib
import shutil
from pathlib import Path

from langchain_chroma import Chroma

documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vector_module = importlib.import_module("04_vector_representation")

CHROMA_PATH = Path("chroma_db")


def build_vector_store():
    if CHROMA_PATH.exists():
        print(f"Removing existing store at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    print("Loading documents...")
    documents = documents_module.load_documents()

    print("Cleaning text...")
    documents = preprocessing_module.preprocess_documents(documents)

    print("Chunking (junk chunks filtered automatically)...")
    chunks = chunking_module.chunk_documents(documents)
    print(f"Kept {len(chunks)} chunks.")

    print("Loading embedding model...")
    embedding_model = vector_module.get_embedding_model()

    vector_store = Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embedding_model,
    )

    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        print(f"Adding chunks {start} -> {end}")
        vector_store.add_documents(chunks[start:end])

    print("\nVector store created successfully at", CHROMA_PATH.resolve())
    return vector_store


if __name__ == "__main__":
    build_vector_store()
