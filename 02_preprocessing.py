"""
Step 2: Clean text extracted from PDFs
(normalize unicode, strip surrogate chars, collapse whitespace).
"""

import re
import unicodedata
import importlib

documents_module = importlib.import_module("01_documents")


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\ud800-\udfff]", "", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cs")
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_documents(documents):
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    return documents


if __name__ == "__main__":
    docs = documents_module.load_documents()
    docs = preprocess_documents(docs)
    print(f"Cleaned {len(docs)} documents.")
    print("\nSample cleaned content:")
    print(docs[0].page_content[:500])
