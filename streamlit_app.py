"""
streamlit_app.py — SourceAware RAG Assistant (Production-Ready)
================================================================
Chat-style UI with conversation memory, source attribution,
and document management in the sidebar.

All core logic lives in the src/ package.
"""

import os
import io
import glob
import json
from pathlib import Path
import time
import hashlib
import streamlit as st

# ── Import core logic from src modules ──
from src.pdf_processing import extract_text_from_pdfs
from src.chunking import chunk_documents, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import load_embedding_model
from src.vector_store import build_faiss_index, save_index, load_index
from src.retrieval import retrieve_chunks
from src.llm import generate_answer

# ── Constants ──
MAX_CHAT_HISTORY = 10  # Store last 5 exchanges (5 user + 5 assistant = 10 messages)


# ──────────────────────────────────────────────
# Session state initialization
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# Chat history management (Persistent Multi-Session)
# ──────────────────────────────────────────────

CHATS_DIR = Path("./chats")

def load_chats():
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    chats = {}
    for chat_file in CHATS_DIR.glob("*.json"):
        try:
            with open(chat_file, "r", encoding="utf-8") as f:
                chat_data = json.load(f)
                chats[chat_file.stem] = chat_data
        except Exception:
            pass
            
    if not chats:
        default_id = f"chat_{int(time.time())}"
        chats[default_id] = {"title": "New Chat", "created_at": time.time(), "messages": []}
        
    return dict(sorted(chats.items(), key=lambda x: x[1].get("created_at", 0), reverse=True))

def save_chat(chat_id, chat_data):
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHATS_DIR / f"{chat_id}.json", "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=2)

def init_session_state():
    """Initialize all session_state keys on first run."""
    defaults = {
        "extracted_pages": [],
        "chunks": [],
        "vector_store": None,
        "embeddings": None,
        "index_loaded": False,
        "processing": False,
        "docs_hash": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "chats" not in st.session_state:
        st.session_state.chats = load_chats()
        
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = next(iter(st.session_state.chats.keys()))

def get_current_chat_history():
    """Retrieve the messages list for the active chat session."""
    return st.session_state.chats[st.session_state.current_chat_id]["messages"]

def add_to_chat(role: str, content: str):
    """Append a message to the active chat history."""
    chat_id = st.session_state.current_chat_id
    chat = st.session_state.chats[chat_id]
    
    chat["messages"].append({"role": role, "content": content})
    
    # Auto-generate title on first user message
    if len(chat["messages"]) == 1 and role == "user":
        chat["title"] = content[:30] + "..." if len(content) > 30 else content
        
    if len(chat["messages"]) > MAX_CHAT_HISTORY:
        chat["messages"] = chat["messages"][-MAX_CHAT_HISTORY:]
        
    save_chat(chat_id, chat)

def clear_chat():
    """Reset active chat history."""
    chat_id = st.session_state.current_chat_id
    st.session_state.chats[chat_id]["messages"] = []
    st.session_state.chats[chat_id]["title"] = "New Chat"
    save_chat(chat_id, st.session_state.chats[chat_id])
    
    # Clear debug panel data
    st.session_state.pop("last_query", None)
    st.session_state.pop("last_results", None)
    st.session_state.pop("last_confidence", None)


# ──────────────────────────────────────────────
# Sidebar — document management & settings
# ──────────────────────────────────────────────
def get_secret(key):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)

def get_api_key():
    """Retrieve API key securely from secrets or environment."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def render_sidebar():
    """
    Render the sidebar with upload, processing, and settings.
    Returns (uploaded_files, chunk_size, chunk_overlap, top_k, compare_mode).
    """
    with st.sidebar:
        # ── Feature 5: Chat Sessions UI ──
        st.header("💬 Chat History")
        if st.button("➕ New Chat", use_container_width=True, key="new_chat_btn"):
            new_id = f"chat_{int(time.time())}"
            # Prepend new chat
            st.session_state.chats = {new_id: {"title": "New Chat", "created_at": time.time(), "messages": []}, **st.session_state.chats}
            st.session_state.current_chat_id = new_id
            save_chat(new_id, st.session_state.chats[new_id])
            st.rerun()
            
        st.markdown("<div style='max-height: 250px; overflow-y: auto; padding-right: 5px;'>", unsafe_allow_html=True)
        for chat_id, chat_data in st.session_state.chats.items():
            btn_type = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
            title = chat_data.get('title', 'New Chat')
            if len(title) > 25:
                title = title[:22] + "..."
            if st.button(f"🗨️ {title}", key=f"btn_{chat_id}", use_container_width=True, type=btn_type):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.divider()

        # ── Settings ──
        st.header("⚙️ Settings")
        chunk_size = st.slider(
            "Chunk size", 200, 1500, 400, 50,
            help="Characters per chunk.",
            key="chunk_size_slider",
        )
        chunk_overlap = st.slider(
            "Chunk overlap", 0, 300, 50, 10,
            help="Overlap between chunks.",
            key="chunk_overlap_slider",
        )
        top_k = st.slider(
            "Top-K results", 1, 10, 5,
            help="Number of chunks to retrieve per query.",
            key="top_k_slider",
        )
        st.divider()

        # ── Section 1: Documents ──
        st.header("📁 Documents")
        
        use_sample = st.checkbox("Use sample documents", key="use_sample_docs_checkbox")
        
        uploaded_files = []
        if use_sample:
            sample_paths = glob.glob("data/*.pdf")
            if not sample_paths:
                st.warning("⚠️ No sample PDFs found in data/ folder.")
            else:
                st.success(f"✅ Loaded {len(sample_paths)} sample document(s)")
                with st.expander("📚 Sample Documents"):
                    for p in sample_paths:
                        st.caption(f"• {os.path.basename(p)}")
                
                # Load files into BytesIO objects
                for p in sample_paths:
                    try:
                        with open(p, "rb") as f:
                            b = io.BytesIO(f.read())
                            b.name = os.path.basename(p)
                            uploaded_files.append(b)
                    except Exception as e:
                        st.error(f"Failed to load {os.path.basename(p)}: {e}")
                        
                # ── Auto-index sample documents ──
                if uploaded_files:
                    current_hash = get_docs_hash(uploaded_files, chunk_size, chunk_overlap)
                    if st.session_state.get("docs_hash") != current_hash:
                        st.session_state["processing"] = True
                        process_documents(uploaded_files)
                        st.session_state["processing"] = False
        else:
            uploaded_files = st.file_uploader(
                "Upload PDF files",
                type=["pdf"],
                accept_multiple_files=True,
                key="pdf_uploader",
            )

        st.divider()

        # ── Section 2: Index ──
        st.header("⚡ Index")
        
        if st.button(
            "⚡ Process Documents", 
            type="primary", 
            use_container_width=True,
            key="process_docs_button",
            disabled=st.session_state.get("processing", False)
        ):
            if not uploaded_files:
                st.error("Upload PDFs or check 'Use sample documents' first.")
            else:
                st.session_state["processing"] = True
                process_documents(uploaded_files)
                st.session_state["processing"] = False

        # Show index stats
        if st.session_state.get("chunks"):
            stored = st.session_state["chunks"]
            doc_count = len({c["doc_name"] for c in stored})
            st.caption(f"✅ {len(stored)} chunks · {doc_count} doc(s)")
            st.success("Documents indexed and ready")
        else:
            st.info("No documents indexed yet")

        st.divider()

        # ── Section 4: View Mode & Chat ──
        st.header("🗂️ View Mode")

        compare_mode = st.checkbox(
            "Compare Across Documents", 
            help="Group retrieved chunks by document",
            key="compare_mode_checkbox"
        )
        
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_button"):
            clear_chat()
            st.rerun()

        st.divider()

        # ── Feature 3: Query Debug Panel ──
        if "last_query" in st.session_state and "last_results" in st.session_state:
            with st.expander("🔍 Debug Panel"):
                st.write("**Query:**", st.session_state["last_query"])
                if "last_confidence" in st.session_state:
                    conf = st.session_state["last_confidence"]
                    st.write("**Confidence:**", conf.upper())
                
                st.write("**Retrieved Context:**")
                for i, r in enumerate(st.session_state["last_results"]):
                    st.json({
                        "rank": i+1,
                        "doc": r["doc_name"],
                        "page": r["page"],
                        "score": r["score"],
                        "reason": r.get("reason", "N/A")
                    })

    return uploaded_files, chunk_size, chunk_overlap, top_k, compare_mode


# ──────────────────────────────────────────────
# Document processing pipeline
# ──────────────────────────────────────────────

# ADD THIS: Hash function to detect document changes
def get_docs_hash(uploaded_files, chunk_size, chunk_overlap):
    """Generate a hash based on file names, sizes, and chunk settings."""
    h = hashlib.md5()
    h.update(str(chunk_size).encode('utf-8'))
    h.update(str(chunk_overlap).encode('utf-8'))
    for f in uploaded_files:
        h.update(f.name.encode('utf-8'))
        h.update(str(getattr(f, 'size', len(f.getvalue() if hasattr(f, 'getvalue') else '0'))).encode('utf-8'))
    return h.hexdigest()

def process_documents(uploaded_files):
    """Run the full Phase 1→2 pipeline: extract → chunk → embed → index."""
    try:
        # ADD THIS: Check if we really need to reprocess
        # The chunk settings were not passed to this function directly before, 
        # but they are available in session_state via the keys.
        chunk_size = st.session_state.get("chunk_size_slider", 400)
        chunk_overlap = st.session_state.get("chunk_overlap_slider", 50)
        
        current_hash = get_docs_hash(uploaded_files, chunk_size, chunk_overlap)
        
        if st.session_state.get("docs_hash") == current_hash and st.session_state["vector_store"] is not None:
            st.sidebar.info("⚡ Already indexed. Clear chat or change settings to reprocess.")
            return

        with st.spinner("📄 Extracting text from PDFs..."):
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
            chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            if not chunks:
                st.sidebar.error("No chunks created.")
                return

            hf_token = get_secret("HF_TOKEN")
            embeddings = load_embedding_model(hf_token=hf_token)
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
            st.session_state["docs_hash"] = current_hash  # ADD THIS: Save hash

        st.sidebar.success(f"✅ Indexed {len(chunks)} chunks!")
        if saved:
            st.sidebar.info("💾 Saved to disk.")
    
    except Exception as e:
        st.sidebar.error(f"❌ Processing failed: {str(e)}")
        st.session_state["processing"] = False


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
    for msg in get_current_chat_history():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def format_sources(sources: list[dict]) -> str:
    """Format source citations as a markdown string."""
    if not sources:
        return ""

    lines = ["\n\n**Sources:**"]
    for s in sources:
        lines.append(f"- **{s['doc_name']}** (Page {s['page']})")
    return "\n".join(lines)


def stream_text(text: str, delay: float = 0.02):
    """Yield text chunk by chunk for a ChatGPT-like typewriter effect."""
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


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
        try:
            hf_token = get_secret("HF_TOKEN")
            embeddings = load_embedding_model(hf_token=hf_token)
            if embeddings:
                saved_store, saved_chunks = load_index(embeddings)
                if saved_store is not None:
                    st.session_state["vector_store"] = saved_store
                    st.session_state["chunks"] = saved_chunks
                    st.session_state["embeddings"] = embeddings
        except Exception:
            pass  # Silent fail - user will upload docs manually
        finally:
            st.session_state["index_loaded"] = True

    # ── Sidebar ──
    uploaded_files, chunk_size, chunk_overlap, top_k, compare_mode = render_sidebar()
    groq_api_key = get_api_key()

    # ── Premium Custom CSS for UI Improvements ──
    st.markdown(
        """
        <style>
        /* 1. REMOVE STREAMLIT DEFAULT UI */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0d0d12;
            color: #ececf1;
        }
        
        .main-header {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            line-height: 1.2;
            text-align: center;
        }
        
        .sub-header {
            font-size: 1.2rem;
            font-weight: 400;
            color: #A0A0A0;
            margin-bottom: 2.5rem;
            letter-spacing: 0.5px;
            text-align: center;
        }
        
        /* Assistant Chat Bubble */
        .stChatMessage[data-testid="stChatMessage"][aria-label="assistant"] {
            background: rgba(30, 30, 46, 0.6);
            border: 1px solid rgba(78, 205, 196, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }
        
        /* User Chat Bubble */
        .stChatMessage[data-testid="stChatMessage"][aria-label="user"] {
            background: rgba(45, 45, 60, 0.8);
            border: 1px solid rgba(255, 107, 107, 0.2);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 1.5rem;
        }
        
        /* Sidebar Polish (Glassmorphism) */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #11111b 0%, #0d0d12 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        /* Cards / Expanders */
        .streamlit-expanderHeader {
            background-color: rgba(30, 30, 46, 0.5) !important;
            border-radius: 8px;
        }
        
        /* Button Hover Effects */
        .stButton>button {
            border-radius: 10px;
            transition: all 0.3s ease;
            font-weight: 600;
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: white;
        }
        
        .stButton>button:active {
            transform: translateY(0px);
        }

        /* Primary Button */
        .stButton>button[kind="primary"] {
            background: linear-gradient(135deg, #4ECDC4 0%, #2b8a82 100%);
            border: none;
            color: white;
        }
        
        .stButton>button[kind="primary"]:hover {
            box-shadow: 0 6px 20px rgba(78, 205, 196, 0.4);
            background: linear-gradient(135deg, #5de0d7 0%, #34a89f 100%);
        }
        
        /* Info boxes */
        .stAlert {
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ── Main area — Title ──
    st.markdown('<div class="main-header">SourceAware RAG Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered document search with source attribution</div>', unsafe_allow_html=True)
    st.divider()

    # ── Status indicators ──
    index_ready = st.session_state["vector_store"] is not None
    has_api_key = bool(groq_api_key)

    # Show helpful status messages
    if not index_ready:
        st.info("### 🚀 Get Started:\n1. Upload PDFs **OR** check 'Use sample documents' in the sidebar.\n2. Click **⚡ Process Documents**.\n3. Ask questions below!")
    elif not has_api_key:
        st.warning("⚠️ **LLM service not configured.** The app will show retrieved document chunks only.\n\n💡 To enable AI-generated answers, set `GROQ_API_KEY` in Streamlit secrets or environment variables.")

    # ── Render chat history ──
    render_chat_history()

    # ── Feature 3 & 4: Empty State & Example Queries ──
    # FIX #3: Proper handling of preset queries
    if index_ready and not get_current_chat_history():
        with st.container():
            st.info("💡 **Try asking:**")
            col1, col2, col3 = st.columns(3)
            
            # FIX #1: Add unique keys to buttons
            with col1:
                if st.button("📄 Summarize Document", use_container_width=True, key="preset_summarize"):
                    st.session_state["preset_query"] = "Summarize the key points of the documents."
                    st.rerun()
            with col2:
                if st.button("🔑 Key Findings", use_container_width=True, key="preset_findings"):
                    st.session_state["preset_query"] = "What are the key findings or main conclusions?"
                    st.rerun()
            with col3:
                if st.button("👶 Explain Simply", use_container_width=True, key="preset_beginner"):
                    st.session_state["preset_query"] = "Explain the core concepts in these documents simply, as if I am a beginner."
                    st.rerun()

    # ── Chat input ──
    preset = st.session_state.pop("preset_query", None)
    user_query = st.chat_input(
        "Ask something about your documents...",
        disabled=(not index_ready),
        key="chat_input_main",
    )
    
    # FIX #3: Use preset if available
    if preset:
        user_query = preset

    if user_query and index_ready:
        # FIX #7: Wrap query processing in error boundary
        try:
            # Display user message immediately
            with st.chat_message("user"):
                st.markdown(user_query)

            # Add to history
            add_to_chat("user", user_query)

            # ── Retrieve relevant chunks ──
            with st.spinner("🔍 Searching documents..."):
                results = retrieve_chunks(
                    user_query, st.session_state["vector_store"], top_k, compare_mode=compare_mode
                )
                # Save for debug panel (flatten for debug view if compare mode)
                st.session_state["last_query"] = user_query
                st.session_state["last_results"] = [c for chunks in results.values() for c in chunks] if compare_mode else results

            if not results:
                response = "I couldn't find any relevant information in your documents. Try rephrasing your question or uploading different documents."
                st.session_state["last_confidence"] = "no context"
                with st.chat_message("assistant"):
                    st.markdown(response)
                add_to_chat("assistant", response)

            elif not has_api_key:
                st.session_state["last_confidence"] = "no API key"
                # No API key — show retrieved chunks directly
                response_parts = ["**📄 Retrieved Context** (Configure GROQ_API_KEY for AI-generated answers):\n"]
                
                # ADD THIS: Handle dictionary results in compare mode
                if compare_mode:
                    for doc, chunks in results.items():
                        response_parts.append(f"\n**Document: `{doc}`**")
                        for i, r in enumerate(chunks, 1):
                            response_parts.append(
                                f"**{i}.** Page {r['page']} "
                                f"(Similarity: {r['score']:.2f})\n> {r['text'][:300]}...\n"
                            )
                else:
                    for i, r in enumerate(results, 1):
                        response_parts.append(
                            f"**{i}. {r['doc_name']}** — Page {r['page']} "
                            f"(Similarity: {r['score']:.2f})\n> {r['text'][:300]}...\n"
                        )
                response = "\n".join(response_parts)
                with st.chat_message("assistant"):
                    st.markdown(response)
                add_to_chat("assistant", response)

            else:
                # ── Generate LLM answer with conversation memory ──
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Generating answer..."):
                        answer_data = generate_answer(
                            query=user_query,
                            retrieved_chunks=results,
                            chat_history=get_current_chat_history()[:-1],  # exclude current query
                            api_key=groq_api_key,
                            compare_mode=compare_mode, 
                        )
                        st.session_state["last_confidence"] = answer_data.get("confidence", "unknown")

                    # FIX #4: Properly handle LLM errors
                    if answer_data.get("error"):
                        response = "⚠️ **LLM temporarily unavailable.** Showing retrieved document chunks instead:\n\n"
                        st.warning("LLM service is temporarily unavailable. Displaying retrieved context.")
                        
                        # Fallback display: Show retrieved chunks directly
                        fallback_parts = [response]
                        if compare_mode:
                            for doc, chunks in results.items():
                                fallback_parts.append(f"\n📄 **Document: `{doc}`**")
                                for i, r in enumerate(chunks, 1):
                                    fallback_parts.append(
                                        f"**{i}.** (Page {r['page']}) - Score: {r['score']:.2f}\n"
                                        f"> {r['text'][:250]}...\n"
                                    )
                        else:
                            for i, r in enumerate(results, 1):
                                fallback_parts.append(
                                    f"**{i}. {r['doc_name']}** (Page {r['page']}) - Score: {r['score']:.2f}\n"
                                    f"> {r['text'][:250]}...\n"
                                )
                        response = "\n".join(fallback_parts)
                        st.markdown(response)
                        
                        # FIX #4: Add fallback response to chat history
                        add_to_chat("assistant", response)
                    else:
                        # Display answer + sources in a clean format
                        sources_str = format_sources(answer_data["sources"])
                        response = answer_data["answer"] + sources_str
                        
                        # ── Feature 5: Streaming Response ──
                        st.write_stream(stream_text(response))

                        # ── Feature 4: Multi-Document Comparison Mode Display ──
                        with st.expander("📄 View Retrieved Context", expanded=compare_mode):
                            if compare_mode:
                                # MODIFY THIS: results is already grouped in compare mode
                                for doc, chunks in results.items():
                                    st.markdown(f"**Document: `{doc}`**")
                                    for c in chunks:
                                        st.markdown(f"- **Page {c['page']}** (Score: {c['score']:.2f}): {c['text'][:200]}...")
                                    st.divider()
                            else:
                                # Standard linear display
                                for i, r in enumerate(results, 1):
                                    st.markdown(f"**{i}. `{r['doc_name']}` (Page {r['page']})** - Score: {r['score']:.2f}")
                                    st.markdown(f"> {r['text'][:200]}...")

                        add_to_chat("assistant", response)
        
        except Exception as e:
            # FIX #7: Graceful error handling
            error_msg = f"⚠️ An error occurred while processing your query: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            add_to_chat("assistant", error_msg)


if __name__ == "__main__":
    main()