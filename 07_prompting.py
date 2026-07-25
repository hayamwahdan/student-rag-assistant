"""
Step 7: Build the grounded prompt, call the LLM via OpenRouter,
and produce an answer with cited sources.
"""

import os
import importlib
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

retrieve_module = importlib.import_module("06_retrieve_context")

# ---------------------------------------------------------
# OpenRouter configuration.
# Reads from environment (.env) locally. When deployed on
# Streamlit Cloud, streamlit_app.py overrides these from
# st.secrets (TOML), per the assignment instructions.
# ---------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

PROMPT_TEMPLATE = """You are a helpful study assistant.

Answer the question using ONLY the information in the context below.
If the answer is not contained in the context, say "I don't know" - do not guess.

Each context block is tagged with its source. Ground your answer in that context.

Context:
{context}

Question:
{question}

Answer:"""


def build_prompt(question: str, documents) -> str:
    context_blocks = []
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        page_display = page + 1 if isinstance(page, int) else page  # loaders are 0-indexed
        tag = f"[Source: {source}, Page {page_display}]"
        context_blocks.append(f"{tag}\n{doc.page_content}")

    context = "\n\n".join(context_blocks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def format_sources(documents) -> list:
    sources = []
    for doc in documents:
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "N/A")
        page_display = page + 1 if isinstance(page, int) else page
        entry = f"{source} (Page {page_display})"
        if entry not in sources:
            sources.append(entry)
    return sources


def call_llm(prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file locally, "
            "or to Streamlit Secrets when deployed."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(question: str):
    """Full pipeline: retrieve -> build prompt -> call LLM -> attach sources."""
    documents = retrieve_module.retrieve_context(question)

    if not documents:
        return "I don't know - I couldn't find relevant information in the documents.", []

    prompt = build_prompt(question, documents)
    answer = call_llm(prompt)
    sources = format_sources(documents)

    return answer, sources


if __name__ == "__main__":
    print("=" * 60)
    print("Student RAG Assistant - Terminal Mode")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue

        try:
            answer, sources = answer_question(query)
            print("\nANSWER:")
            print(answer)
            if sources:
                print("\nSOURCES:")
                for s in sources:
                    print(f" - {s}")
        except Exception as e:
            print(f"Error: {e}")
