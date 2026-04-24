"""
embeddings.py — HuggingFace embedding model loading
=====================================================

Why embeddings matter:
  Text is just characters — a computer can't measure "similarity" between
  two strings in a meaningful way.  An embedding model converts text into
  a dense numerical vector (384 dimensions for MiniLM) where semantically
  similar texts end up close together in vector space.

  Example:
    "climate change effects" and "global warming impact"
    → vectors that are very close (small L2 distance)

Model choice — all-MiniLM-L6-v2:
  • 384-dimensional output vectors
  • ~22 million parameters (small, fast)
  • Trained on 1B+ sentence pairs
  • Good balance of speed vs. quality for local RAG systems
  • Runs on CPU without issues
"""


"""
embeddings.py — HuggingFace embedding model loading
"""

import os
import streamlit as st

# Try modern import first (LangChain new version)
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# MODIFY THIS: Better, faster embedding model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"


# ADD THIS: Cache embedding model in Streamlit to avoid reloading
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME, hf_token: str | None = None):
    """
    Load embedding model with optional HF token support.
    Works for both local (.env) and Streamlit Cloud (secrets).
    """

    try:
        # Set token ONLY if provided
        if hf_token:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

        # MODIFY THIS: Add batch processing for speed
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'batch_size': 32}
        )
        return embeddings

    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None


def embed_text(text: str, embeddings):
    """Convert text → embedding vector"""
    return embeddings.embed_query(text)