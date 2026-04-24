# SourceAware RAG Assistant

A Retrieval-Augmented Generation (RAG) system that extracts information from PDF documents, performs semantic search, and generates context-aware answers using a Large Language Model (LLM). Crucially, the system provides exact source attribution (document name and page number) for every answer generated to prevent hallucinations and allow users to verify the information.


## Features

- **Premium UI / UX:** Dark modern theme with glassmorphism, gradient typography, and custom chat bubbles.
- **Multi-Session Chat History:** Persistent ChatGPT-style sidebar to manage, switch between, and store multiple chat threads.
- **High-Performance Architecture:** Employs optimized vector search parameters (`k`), fast string matching for text overlaps, and lazy loading to guarantee sub-second latency.
- **Multi-Document Ingestion & Auto-Indexing:** Seamlessly process uploaded PDFs, or auto-index sample documents instantly.
- **Source Attribution:** Maintains document metadata (filename and page number) throughout the entire pipeline, ensuring all LLM responses include exact citations.
- **Semantic Search:** Uses HuggingFace embeddings (`BAAI/bge-small-en`) and FAISS for fast and accurate similarity search.
- **Strict Context Boundaries:** The LLM is explicitly prompted to answer queries using *only* the retrieved context and to gracefully decline if the information is missing.
- **Local Indexing:** Saves the vectorized FAISS index to disk to prevent re-processing identical documents across application restarts.

## Architecture

The system operates in two main phases:

**Phase 1: Ingestion Pipeline**
1. **Extraction:** Raw text is extracted from uploaded PDFs page-by-page.
2. **Chunking:** The text is split into smaller, overlapping chunks (default: 600 characters with 80-character overlap) to preserve context boundaries while isolating specific topics.
3. **Embedding:** Each chunk is converted into a 384-dimensional vector representation.
4. **Indexing:** Vectors and their associated metadata are stored in a local FAISS index.

**Phase 2: Retrieval and Generation Pipeline**
1. **Query Processing:** The user's query is converted into a vector using the same embedding model.
2. **Similarity Search:** FAISS retrieves the top-K chunks that are most semantically similar to the query vector.
3. **Prompt Construction:** The retrieved chunks are formatted into a strict prompt alongside the user's query and recent chat history.
4. **Generation:** The Groq API (LLaMA 3.1) processes the prompt and returns a synthesized answer with source citations.

## Tech Stack

- **Language:** Python 3.10+
- **Frontend:** Streamlit
- **PDF Parsing:** pypdf
- **Chunking & Vector Store:** LangChain, FAISS (CPU)
- **Embeddings:** HuggingFaceEmbeddings (`BAAI/bge-small-en`) via LangChain
- **LLM Inference:** Groq API (`llama-3.1-70b-versatile`)
- **Environment Management:** python-dotenv

## Project Structure

```text
SourceAware-RAG-Assistant/
├── streamlit_app.py          # Main application entry point and UI layout
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not tracked in git)
├── .env.example              # Example environment template
├── data/                     # Directory for sample or local PDF files
│   ├── climate_report.pdf    # Sample: Climate change impacts
│   ├── llama2_paper.pdf      # Sample: LLaMA 2 research paper
│   ├── rag_paper.pdf         # Sample: Retrieval-Augmented Generation paper
│   └── transformer_paper.pdf # Sample: "Attention Is All You Need"
├── vector_store/             # Persisted FAISS index and metadata (auto-generated)
└── src/                      # Core logic modules
    ├── __init__.py
    ├── pdf_processing.py     # PDF upload and text extraction
    ├── chunking.py           # Recursive text splitting logic
    ├── embeddings.py         # HuggingFace model initialization
    ├── vector_store.py       # FAISS index construction and persistence
    └── llm.py                # Groq API integration and prompt formatting
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Priyal-2005/SourceAware-RAG-Assistant.git
   cd SourceAware-RAG-Assistant
   ```

2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   streamlit run streamlit_app.py
   ```

## Environment Variables

The application requires a Groq API key for LLM generation. 

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Add your API key to the `.env` file:
   ```text
   GROQ_API_KEY=your_api_key_here
   ```
*(Alternatively, you can input the API key directly through the application's sidebar interface.)*

## Example Usage

The repository includes four sample research papers in the `data/` directory. You can upload them via the sidebar, click **⚡ Process Documents**, and try asking:

- **RAG Paper:** "What is the difference between RAG-Sequence and RAG-Token models?"
- **Transformer Paper:** "How does multi-head attention work according to the authors?"
- **LLaMA 2 Paper:** "What were the primary safety alignment techniques used for LLaMA 2-Chat?"
- **Climate Report:** "Summarize the major economic impacts mentioned in the climate report."
- **Follow-up Memory Test:** "Can you elaborate on your previous answer and explain the limitations?"

## Limitations

- **Scanned Documents:** The system relies on `pypdf` for text extraction. It cannot read text from scanned PDFs or images without OCR preprocessing.
- **Chunking Artifacts:** Strict character-based chunking can sometimes split important context across two separate chunks, slightly degrading retrieval quality.
- **Table Parsing:** Complex tables and multi-column layouts may lose formatting during extraction, confusing the LLM.
- **Memory Overhead:** Storing all conversational context limits the number of document chunks that can be passed to the LLM due to context window size restrictions.

## Future Improvements

- **Hybrid Search:** Implement keyword-based (BM25) search alongside semantic vector search to improve retrieval for exact names and acronyms.
- **Reranking:** Add a cross-encoder reranker to re-order the retrieved chunks before passing them to the LLM.
- **OCR Integration:** Integrate Tesseract or AWS Textract to support scanned documents and images.
- **Advanced UI:** Add features to view the original PDF document side-by-side with the chat interface, auto-scrolled to the cited page.

## Why This Project Matters

As Large Language Models become more common in enterprise environments, the risk of "hallucination" (models inventing facts) remains a critical bottleneck. Retrieval-Augmented Generation solves this by restricting the model's knowledge to a verified corpus of documents. This project demonstrates a production-style RAG architecture where source attribution is a first-class citizen, ensuring that users do not have to blindly trust the AI—they can immediately verify the source of the claim themselves.

## Limitations
- Retrieval quality depends on chunking strategy and embedding model
- LLM responses are constrained to retrieved context and may miss implicit information
- Performance may degrade on scanned PDFs (no OCR)

## Future Work
- Hybrid search (keyword + semantic)
- Multi-modal support (tables/images)
- Re-ranking models for improved retrieval accuracy

# Deployed Link (Streamlit)
https://sourceaware-rag-assistant.streamlit.app/