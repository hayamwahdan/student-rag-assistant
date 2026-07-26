import importlib
from pathlib import Path
import streamlit as st

# ------------------------------------------------------------------
# chroma_db/ is committed to the repo (prebuilt), so no rebuild is
# needed on startup. If it's ever missing (e.g. a fresh clone before
# running the pipeline), fall back to building it once from data/.
# ------------------------------------------------------------------
CHROMA_PATH = Path("chroma_db")

if not CHROMA_PATH.exists():
    with st.spinner("Document index not found — building it now (this can take a minute or two)..."):
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

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

SUGGESTED_QUESTIONS = [
    "What is overfitting?",
    "Explain the bias-variance tradeoff",
    "What is the difference between supervised and unsupervised learning?",
    "What is a neural network?",
]


def process_question(question: str):
    """Shared handler so suggested-question buttons and the chat box
    both go through the exact same retrieval/answer logic."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = prompting_module.answer_question(question)
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


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("View Sources"):
                for src in message["sources"]:
                    st.write(f"📍 {src}")

# Only show suggested questions before the conversation has started
if not st.session_state.messages:
    st.markdown("**💡 Try asking:**")
    cols = st.columns(2)
    for i, question in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(question, use_container_width=True):
            st.session_state.pending_question = question

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.pending_question = prompt

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    process_question(question)

st.divider()
st.caption("Built for Students | AI-Powered Document Search")