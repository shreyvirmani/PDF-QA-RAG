import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# -----------------------------
# Optional HuggingFace Token
# -----------------------------

from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")

#------------------------------
# Google API Key 
#------------------------------

from src.llm import get_llm

llm_model = get_llm()

# -----------------------------
# Project Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

VECTOR_STORE_PATH = BASE_DIR / "vector_store"
STATS_FILE = "metadata.json"

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🤖 PDF AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
<style>

.main{
    padding-top:2rem;
}

.block-container{
    padding-top:2rem;
}

[data-testid="stSidebar"]{
    background:#0E1117;
}

div[data-testid="metric-container"]{
    border:1px solid #2d2d2d;
    border-radius:12px;
    padding:12px;
}

.stButton>button{
    width:100%;
    border-radius:10px;
    height:3em;
    font-weight:bold;
}

.stChatMessage{
    border-radius:12px;
}

</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Title
# -----------------------------
st.title("🤖 PDF AI Assistant")

st.caption(
    "Chat with your PDF using Gemini + LangChain + FAISS"
)
# -----------------------------
# Backend Imports
# -----------------------------
from src.loader import load_pdf
from src.splitter import split_documents
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store,
)
from src.rag import answer_question


# -----------------------------
# Cache Embedding Model
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embedding():
    return get_embedding_model()


# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

if "pages" not in st.session_state:
    st.session_state.pages = 0

if "chunks" not in st.session_state:
    st.session_state.chunks = 0

if "processed" not in st.session_state:
    st.session_state.processed = False


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("📂 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    st.divider()

    st.subheader("📊 Document Information")

    col1, col2 = st.columns(2)

    col1.metric(
        "Pages",
        st.session_state.pages
    )

    col2.metric(
        "Chunks",
        st.session_state.chunks
    )

    st.metric(
        "Model",
        "Gemini 2.5 Flash"
    )

    st.metric(
        "Embeddings",
        "MiniLM-L6-v2"
    )

    st.divider()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.markdown(
        """
### 🚀 Tech Stack

- Gemini
- LangChain
- FAISS
- HuggingFace
- Streamlit
"""
    )
# -----------------------------
# PDF Processing
# -----------------------------
if uploaded_pdf is not None and (
    not st.session_state.processed
    or uploaded_pdf.name != st.session_state.pdf_name
):

    progress = st.progress(0)

    try:

        progress.progress(10, text="📄 Reading PDF...")

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        progress.progress(30, text="📚 Loading document...")

        documents = load_pdf(pdf_path)

        progress.progress(50, text="✂ Splitting into chunks...")

        chunks = split_documents(documents)

        progress.progress(70, text="🧠 Creating embeddings...")

        embedding_model = get_embedding()

        vector_store = create_vector_store(
            chunks,
            embedding_model
        )

        progress.progress(90, text="💾 Saving vector database...")

        save_vector_store(vector_store)

        st.session_state.vector_store = load_vector_store(
            embedding_model
        )

        st.session_state.pages = len(documents)
        st.session_state.chunks = len(chunks)
        st.session_state.pdf_name = uploaded_pdf.name
        st.session_state.processed = True

        progress.progress(100, text="✅ Done!")

        os.remove(pdf_path)

        st.toast("PDF processed successfully! 🎉")

    except Exception as e:

        st.error(f"Error processing PDF\n\n{e}")

        st.stop()

elif uploaded_pdf is None:

    st.session_state.vector_store = None
    st.session_state.processed = False
    st.session_state.chat_history = []
    st.session_state.pages = 0
    st.session_state.chunks = 0
    st.session_state.pdf_name = None


# -----------------------------
# Main Area
# -----------------------------
if st.session_state.vector_store is None:

    st.info("👈 Upload a PDF from the sidebar to begin.")

    st.stop()


# -----------------------------
# Document Card
# -----------------------------
st.success(f"📄 **Current Document:** {st.session_state.pdf_name}")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Pages",
    st.session_state.pages
)

col2.metric(
    "Chunks",
    st.session_state.chunks
)

col3.metric(
    "Status",
    "Ready ✅"
)

st.divider()
# -----------------------------
# Display Chat History
# -----------------------------
for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant" and "sources" in message:

            with st.expander("📚 View Sources"):

                for i, doc in enumerate(message["sources"], start=1):

                    page = doc.metadata.get("page", "Unknown")

                    if isinstance(page, int):
                        page += 1

                    st.markdown(f"### 📄 Source {i}")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.info(f"**Page:** {page}")

                    with col2:
                        st.info(
                            f"**File:** {doc.metadata.get('source','Uploaded PDF')}"
                        )

                    st.markdown("**Preview**")

                    st.write(doc.page_content[:500] + "...")

                    st.divider()


# -----------------------------
# Chat Input
# -----------------------------
question = st.chat_input(
    "Ask anything about your PDF..."
)

if question:

    # Display User Message
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Generate Answer
    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            try:

                answer, docs = answer_question(
                    st.session_state.vector_store,
                    question,
                )

            except Exception as e:

                st.error(e)

                st.stop()

        st.markdown(answer)

        with st.expander("📚 View Sources"):

            for i, doc in enumerate(docs, start=1):

                page = doc.metadata.get("page", "Unknown")

                if isinstance(page, int):
                    page += 1

                st.markdown(f"### 📄 Source {i}")

                col1, col2 = st.columns(2)

                with col1:
                    st.info(f"Page: {page}")

                with col2:
                    st.info(
                        f"File: {doc.metadata.get('source','Uploaded PDF')}"
                    )

                st.write(doc.page_content[:500] + "...")

                st.divider()

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": docs,
        }
    )
