"""
RAG Research Assistant — Phase 1 + Phase 2
==========================================
Phase 1: Upload PDF documents and extract text with source attribution.
Phase 2: Chunk text, generate embeddings, build FAISS index, persist locally.

This module handles:
  • Multi-file PDF upload via Streamlit sidebar
  • Page-by-page text extraction using pypdf
  • Structured storage in session_state for reuse in later phases
  • Clean display of extracted content with document + page metadata
  • [Phase 2] Text chunking with LangChain RecursiveCharacterTextSplitter
  • [Phase 2] Embedding generation with HuggingFace all-MiniLM-L6-v2
  • [Phase 2] FAISS vector store creation and local persistence
"""

import json
import os
from pathlib import Path

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Constants

# Directory where FAISS index + metadata are persisted between sessions
VECTOR_STORE_DIR = Path("./vector_store")
METADATA_FILE = VECTOR_STORE_DIR / "chunk_metadata.json"

# Default chunking parameters (user can override via sidebar)
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80

# Embedding model — lightweight & fast, good for semantic search
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ──────────────────────────────────────────────
# Phase 1 — PDF text extraction
# ──────────────────────────────────────────────

def extract_text_from_pdf(file) -> list[dict]:
    """
    Read a Streamlit UploadedFile (PDF) and return a list of page records.

    Each record is a dict:
        {"doc_name": str, "page": int, "text": str}

    Pages with no extractable text are skipped (a warning is shown later).
    """
    doc_name = file.name
    pages: list[dict] = []

    try:
        reader = PdfReader(file)
    except Exception as exc:
        # Handles corrupted / unreadable PDFs
        st.error(f"❌ Could not read **{doc_name}** — the file may be corrupted.\n\n`{exc}`")
        return pages

    if len(reader.pages) == 0:
        st.warning(f"⚠️ **{doc_name}** has no pages.")
        return pages

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        # Skip pages with no extractable text
        if not text.strip():
            st.warning(f"⚠️ **{doc_name}** — Page {page_number} has no extractable text (skipped).")
            continue

        pages.append({
            "doc_name": doc_name,
            "page": page_number,
            "text": text,
        })

    return pages


# ──────────────────────────────────────────────
# Phase 2 — Text chunking
# ──────────────────────────────────────────────

def chunk_text(extracted_pages: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """
    Split page-level text into smaller chunks for more precise retrieval.

    Why chunk?
      Large pages may contain multiple topics. Smaller chunks let the
      retriever return only the relevant portion instead of an entire page.

    Why overlap?
      Overlap ensures that sentences at chunk boundaries aren't cut in half,
      preserving context across adjacent chunks.

    Each output chunk retains the source metadata (doc_name, page) so we
    can always trace a chunk back to its origin.

    Returns a list of dicts:
        {"text": str, "doc_name": str, "page": int}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # try natural boundaries first
    )

    chunks: list[dict] = []

    for record in extracted_pages:
        # Split the page text into sub-chunks
        text_chunks = splitter.split_text(record["text"])

        for chunk_text_piece in text_chunks:
            chunks.append({
                "text": chunk_text_piece,
                "doc_name": record["doc_name"],
                "page": record["page"],
            })

    return chunks


# ──────────────────────────────────────────────
# Phase 2 — Embeddings
# ──────────────────────────────────────────────

def create_embeddings():
    """
    Create and return a HuggingFace embedding model instance.

    Model: all-MiniLM-L6-v2
      • 384-dimensional vectors
      • Fast inference, small footprint
      • Good balance of speed vs. quality for semantic search
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        return embeddings
    except Exception as exc:
        st.error(f"❌ Failed to load embedding model.\n\n`{exc}`")
        return None


# ──────────────────────────────────────────────
# Phase 2 — FAISS index
# ──────────────────────────────────────────────

def build_faiss_index(chunks: list[dict], embeddings):
    """
    Build a FAISS vector store from text chunks and their embeddings.

    FAISS stores the embedding vectors and allows fast approximate
    nearest-neighbor search.  We attach each chunk's metadata so that
    search results can be traced back to their source document + page.

    Returns a FAISS vector store object, or None on failure.
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
    except Exception as exc:
        st.error(f"❌ Failed to build FAISS index.\n\n`{exc}`")
        return None


# ──────────────────────────────────────────────
# Phase 2 — Persistence (save / load)
# ──────────────────────────────────────────────

def save_index(vector_store, chunks: list[dict]):
    """
    Persist the FAISS index and chunk metadata to disk.

    Two things are saved:
      1. FAISS index files   → ./vector_store/  (managed by LangChain)
      2. Chunk metadata JSON → ./vector_store/chunk_metadata.json

    The metadata file lets us display chunk info on reload without
    having to reprocess the original PDFs.
    """
    try:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

        # Save FAISS index (LangChain handles serialization)
        vector_store.save_local(str(VECTOR_STORE_DIR))

        # Save chunk metadata separately as JSON
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        return True
    except Exception as exc:
        st.error(f"❌ Failed to save index to disk.\n\n`{exc}`")
        return False


def load_index(embeddings):
    """
    Load a previously saved FAISS index + metadata from disk.

    Returns (vector_store, chunks) if found, or (None, None) otherwise.
    On app restart this avoids reprocessing all PDFs from scratch.
    """
    if not VECTOR_STORE_DIR.exists():
        return None, None

    try:
        # Load FAISS index
        vector_store = FAISS.load_local(
            str(VECTOR_STORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,  # required by LangChain for pickle
        )

        # Load chunk metadata
        chunks = []
        if METADATA_FILE.exists():
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                chunks = json.load(f)

        return vector_store, chunks
    except Exception as exc:
        st.warning(f"⚠️ Could not load saved index — it may be corrupted.\n\n`{exc}`")
        return None, None


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main():
    # --- Page config ---
    st.set_page_config(
        page_title="RAG Research Assistant",
        page_icon="📄",
        layout="wide",
    )

    # --- Title ---
    st.title("📄 RAG Research Assistant (Phase 1 + 2)")
    st.markdown("**Upload documents, extract text, and build a searchable vector index.**")

    # --- Sidebar ---
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.divider()

        # Phase 2 — Chunking settings (user-adjustable)
        st.header("⚙️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200,
            max_value=1500,
            value=DEFAULT_CHUNK_SIZE,
            step=50,
            help="Size of each text chunk. Smaller = more precise retrieval, larger = more context.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=300,
            value=DEFAULT_CHUNK_OVERLAP,
            step=10,
            help="Overlap between consecutive chunks to avoid losing context at boundaries.",
        )

        st.divider()

        # Placeholder for Phase 3
        st.header("🔍 Retrieval Settings")
        top_k = st.slider(
            "Top-K results (for future use)",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of most similar chunks to retrieve. Used in Phase 3.",
        )

    # ── Try to load a previously saved index on startup ──
    if "vector_store" not in st.session_state:
        embeddings = create_embeddings()
        if embeddings:
            saved_store, saved_chunks = load_index(embeddings)
            if saved_store is not None:
                st.session_state["vector_store"] = saved_store
                st.session_state["chunks"] = saved_chunks
                st.session_state["embeddings"] = embeddings
                st.info(
                    f"📂 Loaded previously saved index — "
                    f"**{len(saved_chunks)}** chunks ready."
                )

    # --- Guard: nothing uploaded yet ---
    if not uploaded_files:
        st.info("👈 Upload PDF files from the sidebar to get started.")
        return

    # ── Phase 1: Extract text from every uploaded PDF ──
    # We rebuild on every run because Streamlit re-executes the script on
    # each interaction.  The extracted data is stored in session_state so
    # future phases can access it without re-extracting.

    extracted_pages: list[dict] = []

    for file in uploaded_files:
        pages = extract_text_from_pdf(file)
        extracted_pages.extend(pages)

    # Persist in session_state for later phases
    st.session_state["extracted_pages"] = extracted_pages

    # --- Guard: no text extracted from any document ---
    if not extracted_pages:
        st.error("No text could be extracted from any uploaded document.")
        return

    st.success(f"✅ Successfully processed **{len(uploaded_files)}** document(s).")

    # ── Document summaries ──
    st.subheader("📊 Document Summaries")

    # Build per-document stats
    doc_stats: dict[str, int] = {}
    for record in extracted_pages:
        doc_stats.setdefault(record["doc_name"], 0)
        doc_stats[record["doc_name"]] += 1

    # Display stats in columns (up to 3 per row)
    cols = st.columns(min(len(doc_stats), 3))
    for idx, (doc_name, page_count) in enumerate(doc_stats.items()):
        with cols[idx % len(cols)]:
            st.metric(label=doc_name, value=f"{page_count} pages")

    # Total extracted pages across all documents (bonus)
    st.caption(f"**Total extracted pages:** {len(extracted_pages)}")

    st.divider()

    # ══════════════════════════════════════════════
    # Phase 2 — Process Documents button
    # ══════════════════════════════════════════════

    st.subheader("🔧 Phase 2 — Build Vector Index")

    if st.button("⚡ Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing documents — chunking, embedding, indexing..."):

            # Step 1: Chunk the extracted pages
            chunks = chunk_text(extracted_pages, chunk_size, chunk_overlap)

            if not chunks:
                st.error("No chunks were created — the extracted text may be empty.")
            else:
                # Step 2: Load embedding model
                embeddings = create_embeddings()

                if embeddings is None:
                    st.error("Embedding model could not be loaded. Processing aborted.")
                else:
                    # Step 3: Build FAISS index
                    vector_store = build_faiss_index(chunks, embeddings)

                    if vector_store is None:
                        st.error("FAISS index could not be built. Processing aborted.")
                    else:
                        # Step 4: Persist to disk
                        saved = save_index(vector_store, chunks)

                        # Step 5: Store in session_state
                        st.session_state["vector_store"] = vector_store
                        st.session_state["chunks"] = chunks
                        st.session_state["embeddings"] = embeddings

                        st.success(
                            f"✅ Documents indexed successfully!  \n"
                            f"**{len(chunks)}** chunks created and stored."
                        )
                        if saved:
                            st.info("💾 Index saved to disk — it will persist across app restarts.")

    # Show current index stats if available
    if "chunks" in st.session_state and st.session_state["chunks"]:
        stored_chunks = st.session_state["chunks"]

        # Per-document chunk breakdown
        chunk_stats: dict[str, int] = {}
        for c in stored_chunks:
            chunk_stats.setdefault(c["doc_name"], 0)
            chunk_stats[c["doc_name"]] += 1

        st.caption(
            f"**Current index:** {len(stored_chunks)} chunks across "
            f"{len(chunk_stats)} document(s)"
        )

    st.divider()

    # ── Phase 1: Extracted content display ──
    st.subheader("📝 Extracted Content")

    for record in extracted_pages:
        doc_name = record["doc_name"]
        page = record["page"]
        text = record["text"]

        # Show a preview (first 500 chars) with an expander for full text
        preview = text[:500]
        is_truncated = len(text) > 500

        with st.expander(f"📑 **{doc_name}** — Page {page}", expanded=False):
            st.markdown(f"**Document:** `{doc_name}`  \n**Page:** {page}")
            if is_truncated:
                st.text_area(
                    "Preview (first 500 characters)",
                    value=preview + "…",
                    height=180,
                    disabled=True,
                    key=f"preview_{doc_name}_{page}",
                )
                st.markdown("**Full text:**")
            st.text_area(
                "Full page text",
                value=text,
                height=300,
                disabled=True,
                key=f"full_{doc_name}_{page}",
            )


if __name__ == "__main__":
    main()
