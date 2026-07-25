"""
Step 4: Vector representation - the embedding model used across
the whole project (store creation AND retrieval must use this same model).
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


if __name__ == "__main__":
    import importlib

    preprocessing_module = importlib.import_module("02_preprocessing")
    chunking_module = importlib.import_module("03_chunking")

    docs = preprocessing_module.documents_module.load_documents()
    docs = preprocessing_module.preprocess_documents(docs)
    chunks = chunking_module.chunk_documents(docs)

    model = get_embedding_model()
    embedding = model.embed_query(chunks[0].page_content)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding dimension: {len(embedding)}")
    print("First 10 values:", embedding[:10])
