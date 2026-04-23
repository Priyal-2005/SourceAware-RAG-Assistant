"""
streamlit_app.py — UI layer for the SourceAware RAG Assistant
==============================================================
This file contains ONLY Streamlit UI code.
All core logic lives in the src/ package.
"""

import streamlit as st

# ── Import core logic from src modules ──
from src.pdf_processing import extract_text_from_pdfs
from src.chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import load_embedding_model
from src.vector_store import build_faiss_index, save_index, load_index
from src.retrieval import retrieve_chunks, highlight_query_terms


# ──────────────────────────────────────────────
# UI helper — display search results
# ──────────────────────────────────────────────

def display_results(results: list[dict], query: str):
    """Render search results with source attribution and score badges."""
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
            expanded=(rank <= 2),
        ):
            st.markdown(
                f"**📄 Document:** `{doc_name}`  \n"
                f"**📍 Page:** {page}  \n"
                f"**📊 Similarity Score:** {score:.4f}  ({score_label})"
            )
            st.divider()

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
    st.title("📄 RAG Research Assistant")
    st.markdown("**Upload documents, build a vector index, and search with source attribution.**")

    # ═══════════════════════════════════════════
    # Sidebar
    # ═══════════════════════════════════════════
    with st.sidebar:
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.divider()

        st.header("⚙️ Chunking Settings")
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200, max_value=1500,
            value=DEFAULT_CHUNK_SIZE, step=50,
            help="Size of each text chunk. Smaller = more precise, larger = more context.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0, max_value=300,
            value=DEFAULT_CHUNK_OVERLAP, step=10,
            help="Overlap between consecutive chunks to preserve context at boundaries.",
        )

        st.divider()

        st.header("🔍 Retrieval Settings")
        top_k = st.slider(
            "Top-K results",
            min_value=1, max_value=10, value=5,
            help="Number of most similar chunks to retrieve when searching.",
        )

    # ═══════════════════════════════════════════
    # Auto-load saved index on startup
    # ═══════════════════════════════════════════
    if "vector_store" not in st.session_state:
        embeddings = load_embedding_model()
        if embeddings:
            saved_store, saved_chunks = load_index(embeddings)
            if saved_store is not None:
                st.session_state["vector_store"] = saved_store
                st.session_state["chunks"] = saved_chunks
                st.session_state["embeddings"] = embeddings
                st.info(f"📂 Loaded previously saved index — **{len(saved_chunks)}** chunks ready.")

    # --- Guard: nothing uploaded yet ---
    if not uploaded_files:
        st.info("👈 Upload PDF files from the sidebar to get started.")
        return

    # ═══════════════════════════════════════════
    # Phase 1 — Extract text
    # ═══════════════════════════════════════════
    extracted_pages, warnings = extract_text_from_pdfs(uploaded_files)

    # Show any extraction warnings
    for w in warnings:
        if w.startswith("❌"):
            st.error(w)
        else:
            st.warning(w)

    st.session_state["extracted_pages"] = extracted_pages

    if not extracted_pages:
        st.error("No text could be extracted from any uploaded document.")
        return

    st.success(f"✅ Successfully processed **{len(uploaded_files)}** document(s).")

    # ── Document summaries ──
    st.subheader("📊 Document Summaries")

    doc_stats: dict[str, int] = {}
    for record in extracted_pages:
        doc_stats.setdefault(record["doc_name"], 0)
        doc_stats[record["doc_name"]] += 1

    cols = st.columns(min(len(doc_stats), 3))
    for idx, (doc_name, page_count) in enumerate(doc_stats.items()):
        with cols[idx % len(cols)]:
            st.metric(label=doc_name, value=f"{page_count} pages")

    st.caption(f"**Total extracted pages:** {len(extracted_pages)}")
    st.divider()

    # ═══════════════════════════════════════════
    # Phase 2 — Build Vector Index
    # ═══════════════════════════════════════════
    st.subheader("🔧 Build Vector Index")

    if st.button("⚡ Process Documents", type="primary", use_container_width=True):
        with st.spinner("Processing documents — chunking, embedding, indexing..."):
            chunks = chunk_documents(extracted_pages, chunk_size, chunk_overlap)

            if not chunks:
                st.error("No chunks were created — the extracted text may be empty.")
            else:
                embeddings = load_embedding_model()
                if embeddings is None:
                    st.error("Embedding model could not be loaded. Processing aborted.")
                else:
                    vector_store = build_faiss_index(chunks, embeddings)
                    if vector_store is None:
                        st.error("FAISS index could not be built. Processing aborted.")
                    else:
                        saved = save_index(vector_store, chunks)
                        st.session_state["vector_store"] = vector_store
                        st.session_state["chunks"] = chunks
                        st.session_state["embeddings"] = embeddings
                        st.success(f"✅ Documents indexed!  **{len(chunks)}** chunks created.")
                        if saved:
                            st.info("💾 Index saved to disk — persists across restarts.")

    # Index stats
    if "chunks" in st.session_state and st.session_state["chunks"]:
        stored = st.session_state["chunks"]
        doc_count = len({c["doc_name"] for c in stored})
        st.caption(f"**Current index:** {len(stored)} chunks across {doc_count} document(s)")

    st.divider()

    # ═══════════════════════════════════════════
    # Phase 3 — Search
    # ═══════════════════════════════════════════
    st.subheader("🔍 Search Your Documents")

    index_ready = (
        "vector_store" in st.session_state
        and st.session_state["vector_store"] is not None
    )

    if not index_ready:
        st.warning("⚠️ No vector index found. Click **⚡ Process Documents** above first.")
    else:
        query = st.text_input(
            "Ask a question about your documents",
            placeholder="e.g. What are the key findings about climate change?",
            key="search_query",
        )

        col_search, col_clear = st.columns([1, 1])
        with col_search:
            search_clicked = st.button("🔎 Search", type="primary", use_container_width=True)
        with col_clear:
            clear_clicked = st.button("🗑️ Clear Results", use_container_width=True)

        if clear_clicked:
            st.session_state.pop("search_results", None)
            st.session_state.pop("last_query", None)

        if search_clicked:
            if not query or not query.strip():
                st.error("Please enter a question before searching.")
            else:
                with st.spinner("Searching documents..."):
                    results = retrieve_chunks(query, st.session_state["vector_store"], top_k)
                    st.session_state["search_results"] = results
                    st.session_state["last_query"] = query

        if "search_results" in st.session_state:
            results = st.session_state["search_results"]
            last_query = st.session_state.get("last_query", "")
            if not results:
                st.info("No relevant results found. Try rephrasing your question.")
            else:
                display_results(results, last_query)

    st.divider()

    # ═══════════════════════════════════════════
    # Extracted Content (expandable)
    # ═══════════════════════════════════════════
    st.subheader("📝 Extracted Content")

    for record in extracted_pages:
        doc_name = record["doc_name"]
        page = record["page"]
        text = record["text"]

        preview = text[:500]
        is_truncated = len(text) > 500

        with st.expander(f"📑 **{doc_name}** — Page {page}", expanded=False):
            st.markdown(f"**Document:** `{doc_name}`  \n**Page:** {page}")
            if is_truncated:
                st.text_area(
                    "Preview (first 500 characters)",
                    value=preview + "…",
                    height=180, disabled=True,
                    key=f"preview_{doc_name}_{page}",
                )
                st.markdown("**Full text:**")
            st.text_area(
                "Full page text",
                value=text,
                height=300, disabled=True,
                key=f"full_{doc_name}_{page}",
            )


if __name__ == "__main__":
    main()
