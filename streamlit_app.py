"""
streamlit_app.py — SourceAware RAG Assistant (Phase 5)
=======================================================
Chat-style UI with conversation memory, source attribution,
and document management in the sidebar.

All core logic lives in the src/ package.
"""

import streamlit as st

# ── Import core logic from src modules ──
from src.pdf_processing import extract_text_from_pdfs
from src.chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import load_embedding_model
from src.vector_store import build_faiss_index, save_index, load_index
from src.retrieval import retrieve_chunks, highlight_query_terms
from src.llm import generate_answer

# ── Constants ──
MAX_CHAT_HISTORY = 10  # Store last 5 exchanges (5 user + 5 assistant = 10 messages)


# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────

def init_session_state():
    """
    Initialize all session_state keys on first run.

    Why we do this upfront:
      Streamlit re-executes the entire script on every interaction.
      By initializing defaults here, we avoid KeyError crashes and
      keep the rest of the code clean with direct access.
    """
    defaults = {
        "chat_history": [],      # List of {"role", "content"} dicts
        "extracted_pages": [],   # Phase 1 page records
        "chunks": [],            # Phase 2 chunk records
        "vector_store": None,    # FAISS index
        "embeddings": None,      # Loaded embedding model
        "index_loaded": False,   # Whether we've tried loading from disk
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ──────────────────────────────────────────────
# Chat history management
# ──────────────────────────────────────────────

def add_to_chat(role: str, content: str):
    """
    Append a message to chat history, enforcing the max size.

    Why we limit history:
      Each message takes up tokens in the LLM prompt.  Keeping only
      the last 5 exchanges (10 messages) ensures we don't eat into
      the space needed for document context, while still supporting
      follow-up questions like "tell me more" or "what about page 5?".
    """
    st.session_state["chat_history"].append({"role": role, "content": content})

    # Trim to max size (keep most recent messages)
    if len(st.session_state["chat_history"]) > MAX_CHAT_HISTORY:
        st.session_state["chat_history"] = st.session_state["chat_history"][-MAX_CHAT_HISTORY:]


def clear_chat():
    """Reset chat history for a fresh conversation."""
    st.session_state["chat_history"] = []


# ──────────────────────────────────────────────
# Sidebar — document management & settings
# ──────────────────────────────────────────────

def render_sidebar():
    """
    Render the sidebar with upload, processing, and settings.
    Returns (uploaded_files, chunk_size, chunk_overlap, top_k, groq_api_key).
    """
    with st.sidebar:
        st.header("📁 Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        st.divider()

        # ── Process button ──
        st.header("🔧 Index")

        if st.button("⚡ Process Documents", type="primary", use_container_width=True):
            if not uploaded_files:
                st.error("Upload PDFs first.")
            else:
                process_documents(uploaded_files)

        # Show index stats
        if st.session_state["chunks"]:
            stored = st.session_state["chunks"]
            doc_count = len({c["doc_name"] for c in stored})
            st.caption(f"✅ {len(stored)} chunks · {doc_count} doc(s)")
        else:
            st.caption("No index yet")

        st.divider()

        # ── Settings ──
        st.header("⚙️ Settings")
        chunk_size = st.slider(
            "Chunk size", 200, 1500, DEFAULT_CHUNK_SIZE, 50,
            help="Characters per chunk.",
        )
        chunk_overlap = st.slider(
            "Chunk overlap", 0, 300, DEFAULT_CHUNK_OVERLAP, 10,
            help="Overlap between chunks.",
        )
        top_k = st.slider(
            "Top-K results", 1, 10, 5,
            help="Number of chunks to retrieve per query.",
        )

        st.divider()

        # ── LLM ──
        st.header("🤖 LLM")
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Get a free key at console.groq.com",
        )

        st.divider()

        # ── Chat controls ──
        if st.button("🗑️ New Chat", use_container_width=True):
            clear_chat()
            st.rerun()

    return uploaded_files, chunk_size, chunk_overlap, top_k, groq_api_key


# ──────────────────────────────────────────────
# Document processing pipeline
# ──────────────────────────────────────────────

def process_documents(uploaded_files):
    """Run the full Phase 1→2 pipeline: extract → chunk → embed → index."""
    with st.spinner("Extracting text from PDFs..."):
        pages, warnings = extract_text_from_pdfs(uploaded_files)
        for w in warnings:
            if w.startswith("❌"):
                st.sidebar.error(w)
            else:
                st.sidebar.warning(w)

        if not pages:
            st.sidebar.error("No text extracted from any document.")
            return

        st.session_state["extracted_pages"] = pages

    with st.spinner("Chunking and building vector index..."):
        chunks = chunk_documents(pages)
        if not chunks:
            st.sidebar.error("No chunks created.")
            return

        embeddings = load_embedding_model()
        if not embeddings:
            st.sidebar.error("Failed to load embedding model.")
            return

        vector_store = build_faiss_index(chunks, embeddings)
        if not vector_store:
            st.sidebar.error("Failed to build FAISS index.")
            return

        saved = save_index(vector_store, chunks)

        st.session_state["vector_store"] = vector_store
        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings

    st.sidebar.success(f"✅ Indexed {len(chunks)} chunks!")
    if saved:
        st.sidebar.info("💾 Saved to disk.")


# ──────────────────────────────────────────────
# Chat message rendering
# ──────────────────────────────────────────────

def render_chat_history():
    """
    Display all messages in the chat history using Streamlit chat UI.

    Why we render the full history on every run:
      Streamlit re-executes the script on each interaction.  The chat
      messages are stored in session_state and re-rendered each time
      to maintain the conversation view.
    """
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def format_sources(sources: list[dict]) -> str:
    """Format source citations as a markdown string."""
    if not sources:
        return ""

    lines = ["\n---", "**📚 Sources:**"]
    for s in sources:
        lines.append(f"- `{s['doc_name']}` — Page {s['page']}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="SourceAware RAG Assistant",
        page_icon="🔍",
        layout="wide",
    )

    init_session_state()

    # ── Try loading saved index once ──
    if not st.session_state["index_loaded"]:
        embeddings = load_embedding_model()
        if embeddings:
            saved_store, saved_chunks = load_index(embeddings)
            if saved_store is not None:
                st.session_state["vector_store"] = saved_store
                st.session_state["chunks"] = saved_chunks
                st.session_state["embeddings"] = embeddings
        st.session_state["index_loaded"] = True

    # ── Sidebar ──
    uploaded_files, chunk_size, chunk_overlap, top_k, groq_api_key = render_sidebar()

    # ── Main area — Title ──
    st.title("🔍 SourceAware RAG Assistant")
    st.caption("Upload documents, ask questions, get answers with source attribution.")

    # ── Status indicators ──
    index_ready = st.session_state["vector_store"] is not None
    has_api_key = bool(groq_api_key)

    if not index_ready:
        st.info("👈 Upload PDFs and click **⚡ Process Documents** in the sidebar to get started.")

    if index_ready and not has_api_key:
        st.warning("💡 Enter your Groq API key in the sidebar to enable AI answers.")

    # ── Render chat history ──
    render_chat_history()

    # ── Chat input ──
    user_query = st.chat_input(
        "Ask something about your documents...",
        disabled=(not index_ready),
    )

    if user_query and index_ready:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_query)

        # Add to history
        add_to_chat("user", user_query)

        # ── Retrieve relevant chunks ──
        with st.spinner("Searching..."):
            results = retrieve_chunks(
                user_query, st.session_state["vector_store"], top_k
            )

        if not results:
            response = "I couldn't find any relevant information in your documents. Try rephrasing your question."
            with st.chat_message("assistant"):
                st.markdown(response)
            add_to_chat("assistant", response)

        elif not has_api_key:
            # No API key — show retrieved chunks directly
            response_parts = ["**Retrieved chunks** (add Groq API key for AI answers):\n"]
            for i, r in enumerate(results, 1):
                response_parts.append(
                    f"**{i}. {r['doc_name']}** — Page {r['page']} "
                    f"(score: {r['score']:.2f})\n> {r['text'][:300]}...\n"
                )
            response = "\n".join(response_parts)
            with st.chat_message("assistant"):
                st.markdown(response)
            add_to_chat("assistant", response)

        else:
            # ── Generate LLM answer with conversation memory ──
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer_data = generate_answer(
                        query=user_query,
                        retrieved_chunks=results,
                        chat_history=st.session_state["chat_history"][:-1],  # exclude current query
                        api_key=groq_api_key,
                    )

                if answer_data["error"]:
                    response = f"⚠️ {answer_data['error']}"
                    st.error(response)
                else:
                    # Display answer + sources
                    sources_str = format_sources(answer_data["sources"])
                    response = answer_data["answer"] + sources_str
                    st.markdown(response)

            add_to_chat("assistant", response)


if __name__ == "__main__":
    main()
