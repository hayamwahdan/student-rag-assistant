"""
Step 3: Split documents into overlapping chunks, and filter out
low-quality chunks (Table of Contents pages, index pages, near-empty text).
"""

import importlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

preprocessing_module = importlib.import_module("02_preprocessing")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOC_MARKERS = ["table of contents", "index", "...................", "……"]


def is_low_quality_chunk(text: str) -> bool:
    """Heuristic filter for junk chunks such as Table-of-Contents or index pages."""
    stripped = text.strip()

    if len(stripped) < 40:
        return True

    lowered = stripped.lower()
    if any(marker in lowered for marker in TOC_MARKERS):
        return True

    # Pages that are mostly numbers (e.g. index/page-number listings)
    digit_ratio = sum(c.isdigit() for c in stripped) / max(len(stripped), 1)
    if digit_ratio > 0.3:
        return True

    return False


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)
    filtered = [c for c in chunks if not is_low_quality_chunk(c.page_content)]
    return filtered


if __name__ == "__main__":
    docs = preprocessing_module.documents_module.load_documents()
    docs = preprocessing_module.preprocess_documents(docs)
    chunks = chunk_documents(docs)

    print(f"Original documents: {len(docs)}")
    print(f"Generated chunks (after junk filter): {len(chunks)}")
    print("\nFirst Chunk")
    print("-" * 50)
    print("Metadata:", chunks[0].metadata)
    print("\nContent Preview:")
    print(chunks[0].page_content[:500])
