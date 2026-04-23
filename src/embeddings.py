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

from langchain_community.embeddings import HuggingFaceEmbeddings

# ── Model identifier on HuggingFace Hub ──
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME):
    """
    Load and return a HuggingFace embedding model instance.

    Step 1: Download (or load from cache) the model weights.
    Step 2: Return a LangChain-compatible embeddings object.

    Why we wrap this in a function:
      The model download can fail (network issues, disk space, etc.).
      By isolating it here, the caller can handle the error gracefully
      without crashing the entire app.

    Args:
        model_name: HuggingFace model identifier. Defaults to MiniLM.

    Returns:
        HuggingFaceEmbeddings instance, or None on failure.

    Raises:
        Returns None instead of raising — the UI layer handles the error.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        return embeddings
    except Exception:
        return None


def embed_text(text: str, embeddings) -> list[float]:
    """
    Convert a single text string into an embedding vector.

    This is useful for embedding a user query before searching FAISS.

    Args:
        text:       The string to embed.
        embeddings: A loaded HuggingFaceEmbeddings instance.

    Returns:
        A list of floats (384 dimensions for MiniLM).
    """
    return embeddings.embed_query(text)
