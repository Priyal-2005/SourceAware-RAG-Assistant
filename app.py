"""
RAG Research Assistant — Phase 1 + 2 + 3
==========================================
Phase 1: Upload PDF documents and extract text with source attribution.
Phase 2: Chunk text, generate embeddings, build FAISS index, persist locally.
Phase 3: Query documents via semantic search and display source-attributed results.

This module handles:
  • Multi-file PDF upload via Streamlit sidebar
  • Page-by-page text extraction using pypdf
  • Structured storage in session_state for reuse in later phases
  • Clean display of extracted content with document + page metadata
  • [Phase 2] Text chunking with LangChain RecursiveCharacterTextSplitter
  • [Phase 2] Embedding generation with HuggingFace all-MiniLM-L6-v2
  • [Phase 2] FAISS vector store creation and local persistence
  • [Phase 3] Semantic query → embedding → FAISS similarity search
  • [Phase 3] Source-attributed result display with similarity scores
"""

import json
import os
import re
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
# Phase 3 — Query & Retrieval
# ──────────────────────────────────────────────

def search_documents(query: str, vector_store, top_k: int) -> list[dict]:
    """
    Search the FAISS index for chunks most similar to the user's query.

    How it works:
      1. The query string is converted into a 384-dim embedding vector
         using the same all-MiniLM-L6-v2 model that was used to embed
         the document chunks.
      2. FAISS computes the L2 (Euclidean) distance between the query
         vector and every stored chunk vector.
      3. The top_k closest vectors are returned, along with their
         metadata (doc_name, page) and distance scores.

    Score interpretation:
      FAISS returns L2 distance — lower = more similar.  We convert
      this to a 0–1 similarity score:  similarity = 1 / (1 + distance)
      so that higher values = better match (more intuitive for users).

    Returns a list of dicts:
        {"text": str, "doc_name": str, "page": int, "score": float}
    """
    try:
        # similarity_search_with_score returns (Document, distance) tuples
        raw_results = vector_store.similarity_search_with_score(query, k=top_k)
    except Exception as exc:
        st.error(f"❌ Search failed.\n\n`{exc}`")
        return []

    results = []
    for doc, distance in raw_results:
        # Convert L2 distance → similarity score (0–1, higher = better)
        similarity = 1.0 / (1.0 + distance)

        results.append({
            "text": doc.page_content,
            "doc_name": doc.metadata.get("doc_name", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "score": round(similarity, 4),
        })

    return results


def highlight_query_terms(text: str, query: str) -> str:
    """
    Wrap query terms in the text with **bold** markers for visibility.

    Uses case-insensitive word matching.  This is a simple string-level
    highlight — not semantic — but helps users quickly spot why a chunk
    was retrieved.
    """
    # Split query into individual words, ignore short/common words
    terms = [t for t in query.split() if len(t) > 2]

    highlighted = text
    for term in terms:
        # Case-insensitive replacement, preserve original casing
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"**{m.group()}**", highlighted)

    return highlighted


def display_results(results: list[dict], query: str):
    """
    Render search results in the Streamlit UI.

    Each result is shown inside an expander with:
      • Source attribution (document name + page number)
      • Similarity score
      • Text preview (first 500 chars) with query terms highlighted
      • Full text available via expansion
    """
    st.markdown(f"### 🎯 Top {len(results)} Matching Results")

    for rank, result in enumerate(results, start=1):
        doc_name = result["doc_name"]
        page = result["page"]
        score = result["score"]
        text = result["text"]

        # Score label for quick readability
        if score >= 0.6:
            score_label = "🟢 High"
        elif score >= 0.4:
            score_label = "🟡 Medium"
        else:
            score_label = "🔴 Low"

        with st.expander(
            f"Result #{rank}  —  📄 {doc_name}  |  📍 Page {page}  |  "
            f"📊 {score:.2f} ({score_label})",
            expanded=(rank <= 2),  # auto-expand top 2 results
        ):
            # Source attribution header
            st.markdown(
                f"**📄 Document:** `{doc_name}`  \n"
                f"**📍 Page:** {page}  \n"
                f"**📊 Similarity Score:** {score:.4f}  ({score_label})"
            )

            st.divider()

            # Preview with highlighted query terms
            preview = text[:500]
            is_truncated = len(text) > 500
            highlighted_preview = highlight_query_terms(preview, query)

            st.markdown("**Preview:**")
            st.markdown(highlighted_preview + ("…" if is_truncated else ""))

            if is_truncated:
                st.markdown("**Full chunk text:**")
                st.text_area(
                    "Full text",
                    value=text,
                    height=250,
                    disabled=True,
                    key=f"search_full_{doc_name}_{page}_{rank}",
                )


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
    st.title("📄 RAG Research Assistant (Phase 1 + 2 + 3)")
    st.markdown("**Upload documents, build a vector index, and search with source attribution.**")

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

        # Phase 3 — Retrieval settings
        st.header("🔍 Retrieval Settings")
        top_k = st.slider(
            "Top-K results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of most similar chunks to retrieve when searching.",
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

    # ══════════════════════════════════════════════
    # Phase 3 — Query & Search
    # ══════════════════════════════════════════════

    st.subheader("🔍 Phase 3 — Search Your Documents")

    # Check whether an index is available for searching
    index_ready = (
        "vector_store" in st.session_state
        and st.session_state["vector_store"] is not None
    )

    if not index_ready:
        st.warning(
            "⚠️ No vector index found. Upload PDFs and click "
            "**⚡ Process Documents** above before searching."
        )
    else:
        # Query input
        query = st.text_input(
            "Ask a question about your documents",
            placeholder="e.g. What are the key findings about climate change?",
            key="search_query",
        )

        # Search and Clear buttons side by side
        col_search, col_clear = st.columns([1, 1])
        with col_search:
            search_clicked = st.button(
                "🔎 Search", type="primary", use_container_width=True
            )
        with col_clear:
            clear_clicked = st.button(
                "🗑️ Clear Results", use_container_width=True
            )

        # Handle clear
        if clear_clicked:
            st.session_state.pop("search_results", None)
            st.session_state.pop("last_query", None)

        # Handle search
        if search_clicked:
            if not query or not query.strip():
                st.error("Please enter a question before searching.")
            else:
                with st.spinner("Searching documents..."):
                    results = search_documents(
                        query,
                        st.session_state["vector_store"],
                        top_k,
                    )
                    st.session_state["search_results"] = results
                    st.session_state["last_query"] = query

        # Display results (persisted across reruns via session_state)
        if "search_results" in st.session_state:
            results = st.session_state["search_results"]
            last_query = st.session_state.get("last_query", "")

            if not results:
                st.info("No relevant results found. Try rephrasing your question.")
            else:
                display_results(results, last_query)

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
