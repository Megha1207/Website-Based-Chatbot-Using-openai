import streamlit as st
import subprocess
import sys
import os
import shutil
from pathlib import Path

from embedding_pipeline.src.qa_engine import QAEngine
from embedding_pipeline.src.site_id import website_id

# --------------------------------------------------
# Streamlit setup
# --------------------------------------------------

st.set_page_config(
    page_title="AI Website Chatbot",
    layout="centered"
)

st.title("AI Website Chatbot")
st.caption("Answers strictly from the provided website content")

# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

def run_script(script_path, args=None):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    command = [sys.executable, script_path]
    if args:
        command.extend(args)

    result = subprocess.run(
        command,
        env=env,
        text=True,
        capture_output=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout


def delete_vector_store(site_key: str):
    vector_path = Path("embedding_pipeline/vector_store") / site_key
    if vector_path.exists():
        shutil.rmtree(vector_path)

# --------------------------------------------------
# Step 1: Website URL input
# --------------------------------------------------

st.subheader("Index a Website")

website_url = st.text_input(
    "Website URL",
    placeholder="https://example.com"
)

col1, col2 = st.columns(2)

with col1:
    index_clicked = st.button("Index Website")

with col2:
    reindex_clicked = st.button("Re-index Website")

# --------------------------------------------------
# Step 2: Indexing logic (SAFE + GUARDED)
# --------------------------------------------------

if index_clicked or reindex_clicked:
    if not website_url:
        st.error("Please enter a valid website URL.")
    else:
        site_key = website_id(website_url)
        vector_path = Path("embedding_pipeline/vector_store") / site_key

        try:
            if reindex_clicked:
                delete_vector_store(site_key)
                st.info("Previous index removed. Re-indexing website...")

            if vector_path.exists() and not reindex_clicked:
                st.success("Website already indexed. Using cached embeddings.")
            else:
                with st.spinner("Crawling and indexing website..."):
                    run_script("crawler/test_crawler.py", args=[website_url])
                    run_script("embedding_pipeline/test_pipeline.py")

                st.success("Website indexed successfully.")

            # Update session state
            st.session_state.indexed = True
            st.session_state.website_url = website_url
            st.session_state.chat_history = []

        except RuntimeError:
            st.warning(
                "Website could not be indexed. "
                "It may be empty, blocked, or not suitable for crawling."
            )

# --------------------------------------------------
# Step 3: Question answering
# --------------------------------------------------

if st.session_state.indexed:
    st.divider()
    st.subheader("Ask Questions")

    @st.cache_resource(show_spinner=False)
    def load_qa(url):
        return QAEngine(url)

    qa = QAEngine(st.session_state.website_url, llm_provider="openai")


    question = st.text_input(
        "Your question",
        placeholder="Ask something about the website..."
    )

    if question:
        with st.spinner("Searching website content..."):
            answer = qa.answer(
                question,
                chat_history=st.session_state.chat_history
            )

        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

else:
    st.info("Please index a website first to enable question answering.")
