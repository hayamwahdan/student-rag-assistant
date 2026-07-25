"""
Step 1: Load raw PDF documents from the /data folder.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFDirectoryLoader

DATA_PATH = Path("data")


def load_documents():
    """Load every PDF inside DATA_PATH and return a list of LangChain Documents (one per page)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data folder not found at: {DATA_PATH.resolve()}")

    pdf_files = list(DATA_PATH.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_PATH.resolve()}")

    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Found PDFs: {[p.name for p in DATA_PATH.glob('*.pdf')]}")
    print(f"Loaded {len(docs)} pages total.\n")
    print("First Document")
    print("-" * 50)
    print("Metadata:", docs[0].metadata)
    print("\nContent Preview:")
    print(docs[0].page_content[:500])
