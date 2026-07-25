import importlib
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------------
# First-run setup: chroma_db/ is NOT committed to the repo (it's in
# .gitignore), so on a fresh deploy it won't exist yet. Build it once,
# from the PDFs in data/, before importing anything that connects to it.
# ------------------------------------------------------------------
CHROMA_PATH = Path("chroma_db")

if not CHROMA_PATH.exists():
    with st.spinner("First-time setup: building the document index (this can take a minute or two)..."):
        store_builder = importlib.import_module("05_create_chroma_store")
        store_builder.build_vector_store()

prompting_module = importlib.import_module("07_prompting")

# ------------------------------------------------------------------
# Pull secrets from Streamlit Cloud (TOML) if not already set via .env
# ------------------------------------------------------------------
try:
    if not prompting_module.OPENROUTER_API_KEY:
        prompting_module.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompting_module.OPENROUTER_MODEL = st.secrets.get(
        "OPENROUTER_MODEL", prompting_module.OPENROUTER_MODEL
    )
except Exception:
    pass

st.set_page_config(page_title="Student RAG Assistant", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
    .source-box {
        background-color: #e1f5fe;
        border-left: 5px solid #0288d1;
        padding: 10px;
        margin-top: 10px;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.title("🎓 RAG Assistant")
    st.markdown(
        """
    ### About
    Answers questions from your study documents using:
    - **Vector Store:** ChromaDB
    - **LLM:** OpenRouter (gpt-4o-mini)
    - **Method:** Retrieval-Augmented Generation (RAG)
    """
    )
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Study Material Assistant")
st.info("Ask a question about the uploaded documents. Answers include cited sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Sources"):
                for src in message["sources"]:
                    st.write(f"📍 {src}")

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = prompting_module.answer_question(prompt)
                st.markdown(answer)
                if sources:
                    with st.expander("View Sources"):
                        for src in sources:
                            st.write(f"📍 {src}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

st.divider()
st.caption("Built for Students | AI-Powered Document Search")
