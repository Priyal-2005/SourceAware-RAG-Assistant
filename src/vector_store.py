"""
vector_store.py — FAISS index creation and persistence
=======================================================

Why FAISS?
  FAISS (Facebook AI Similarity Search) is an optimized library for
  nearest-neighbor search over dense vectors.  Given a query vector, it
  can find the most similar vectors among millions in milliseconds.

  For our RAG system, each chunk becomes a vector.  When a user asks a
  question, we embed the question and use FAISS to find the chunks whose
  embeddings are closest — these are the most semantically relevant chunks.

Why persist to disk?
  Embedding hundreds of chunks takes time (especially on CPU).  By saving
  the FAISS index + metadata to disk, we avoid reprocessing on every app
  restart.  The user only needs to re-index when they upload new documents.

File layout on disk:
  ./vector_store/
    ├── index.faiss          ← the raw FAISS index (binary)
    ├── index.pkl            ← LangChain docstore mapping (pickle)
    └── chunk_metadata.json  ← our custom metadata for display
"""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS

# ── Default paths ──
VECTOR_STORE_DIR = Path("./vector_store")
METADATA_FILE = VECTOR_STORE_DIR / "chunk_metadata.json"


def build_faiss_index(chunks: list[dict], embeddings):
    """
    Build a FAISS vector store from text chunks.

    Step 1: Separate chunk texts from their metadata.
    Step 2: Embed all texts and store in a FAISS index.
    Step 3: Attach metadata (doc_name, page) to each vector for retrieval.

    Why we store metadata alongside vectors:
      FAISS itself only stores vectors and integer IDs.  LangChain's wrapper
      adds a docstore that maps each vector ID to its original text + metadata.
      This is what allows us to return "file.pdf, page 7" with search results.

    Args:
        chunks:     List of {"text", "doc_name", "page"} dicts.
        embeddings: A loaded embedding model instance.

    Returns:
        A FAISS vector store object, or None on failure.
    """
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"doc_name": chunk["doc_name"], "page": chunk["page"]} for chunk in chunks]

    try:
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
        )
        return vector_store
    except Exception:
        return None


def save_index(
    vector_store,
    chunks: list[dict],
    store_dir: Path = VECTOR_STORE_DIR,
    metadata_file: Path = METADATA_FILE,
) -> bool:
    """
    Persist the FAISS index and chunk metadata to disk.

    Step 1: Create the output directory if it doesn't exist.
    Step 2: Save FAISS index files (handled by LangChain internally).
    Step 3: Save chunk metadata as a human-readable JSON file.

    Why save metadata separately?
      The FAISS pickle files are binary and not easy to inspect.  A JSON
      file lets us quickly display chunk stats on reload and is useful
      for debugging.

    Args:
        vector_store:  The FAISS vector store to save.
        chunks:        The chunk metadata list.
        store_dir:     Directory to save into.
        metadata_file: Path for the JSON metadata file.

    Returns:
        True on success, False on failure.
    """
    try:
        store_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (LangChain serializes index.faiss + index.pkl)
        vector_store.save_local(str(store_dir))

        # Save chunk metadata as JSON for easy inspection and reload
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return True
    except Exception:
        return False


def load_index(
    embeddings,
    store_dir: Path = VECTOR_STORE_DIR,
    metadata_file: Path = METADATA_FILE,
):
    """
    Load a previously saved FAISS index + metadata from disk.

    Step 1: Check if the directory exists.
    Step 2: Load the FAISS index using the same embedding model.
    Step 3: Load the chunk metadata JSON.

    Why we need the same embedding model:
      FAISS stores raw vectors — it doesn't know which model produced them.
      We must pass the same model so that future queries are embedded in
      the same vector space.  Using a different model would make distances
      meaningless.

    Args:
        embeddings:    The embedding model (must match the one used to build).
        store_dir:     Directory containing saved index files.
        metadata_file: Path to the JSON metadata file.

    Returns:
        (vector_store, chunks) if found, or (None, None) otherwise.
    """
    if not store_dir.exists():
        return None, None

    try:
        vector_store = FAISS.load_local(
            str(store_dir),
            embeddings,
            allow_dangerous_deserialization=True,  # required by LangChain for pickle
        )

        chunks = []
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        return vector_store, chunks
    except Exception:
        return None, None
