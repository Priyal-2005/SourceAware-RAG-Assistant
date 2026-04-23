"""
llm.py — LLM-based answer generation using Groq API
=====================================================

Why we use an LLM in a RAG pipeline:
  Retrieval alone returns raw text chunks.  A user doesn't want to read
  5 chunks and piece together an answer — they want a single, coherent
  response.  The LLM synthesizes the retrieved chunks into a natural-
  language answer while citing sources.

Why Groq?
  Groq provides extremely fast inference for open-source models like
  LLaMA 3.1.  For a RAG system, speed matters — users are waiting for
  an answer after every query.

Why a strict prompt?
  Without guardrails, LLMs will "hallucinate" — confidently generate
  information that isn't in the documents.  Our prompt explicitly tells
  the model to ONLY use the provided context and to say "I don't know"
  when the answer isn't there.  This is critical for a research assistant
  where accuracy and source-traceability matter more than creativity.
"""

import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY not found. Check your .env file")

def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise ValueError("Missing GROQ_API_KEY")
    return key

# ── Model configuration ──
DEFAULT_MODEL = "llama-3.1-70b-versatile"

# ── Prompt template ──
# Why this template works:
#   1. "ONLY using the provided context" — prevents hallucination
#   2. "If the answer is not present" — gives the model an explicit fallback
#   3. Context is injected with source labels — so the model can cite them
#   4. Clean separation of context / question / answer sections
PROMPT_TEMPLATE = """You are a helpful AI research assistant.

Answer the question ONLY using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Always mention which document and page your answer comes from.

Context:
{context}

Question:
{query}

Answer:"""


def _build_context(retrieved_chunks: list[dict]) -> str:
    """
    Combine retrieved chunks into a single context string with source labels.

    Step 1: For each chunk, prepend a source header [Document: X | Page: Y].
    Step 2: Join all chunks with blank lines for readability.

    Why we include source labels in the context:
      The LLM can see which chunk came from which document/page.  This lets
      it cite sources in its answer (e.g., "According to report.pdf, page 3...").

    Args:
        retrieved_chunks: List of {"text", "doc_name", "page", "score"} dicts.

    Returns:
        A formatted context string ready for the prompt.
    """
    context_parts = []

    for chunk in retrieved_chunks:
        header = f"[Document: {chunk['doc_name']} | Page: {chunk['page']}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n".join(context_parts)


def _get_groq_client(api_key: str | None = None) -> Groq | None:
    """
    Initialize the Groq client with an API key.

    Key resolution order:
      1. Explicit api_key argument (e.g., from Streamlit secrets)
      2. GROQ_API_KEY environment variable

    Returns None if no key is available.
    """
    key = api_key or os.environ.get("GROQ_API_KEY")

    if not key:
        return None

    return Groq(api_key=key)


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Generate an LLM answer from retrieved chunks using Groq.

    Step 1: Build context string from retrieved chunks (with source labels).
    Step 2: Fill the prompt template with context + query.
    Step 3: Call Groq API to generate the answer.
    Step 4: Return the answer + source metadata.

    Why we return sources separately:
      The UI can display a clean "Sources used" section below the answer,
      making it easy for users to verify the information.

    Args:
        query:            The user's question.
        retrieved_chunks: List of {"text", "doc_name", "page", "score"} dicts
                          from the retrieval step.
        api_key:          Optional Groq API key (falls back to env var).
        model:            Groq model identifier.

    Returns:
        A dict with:
          "answer"  — the generated answer string
          "sources" — list of {"doc_name", "page"} dicts used
          "error"   — error message string, or None if successful
    """
    # ── Guard: no chunks to work with ──
    if not retrieved_chunks:
        return {
            "answer": None,
            "sources": [],
            "error": "No context available — please process documents and search first.",
        }

    # ── Step 1: Build context ──
    context = _build_context(retrieved_chunks)

    # ── Step 2: Fill prompt ──
    prompt = PROMPT_TEMPLATE.format(context=context, query=query)

    # ── Step 3: Initialize Groq client ──
    client = _get_groq_client(api_key)
    if client is None:
        return {
            "answer": None,
            "sources": [],
            "error": (
                "Groq API key not found. Please set the GROQ_API_KEY environment "
                "variable or add it in the sidebar."
            ),
        }

    # ── Step 4: Call the API ──
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant that answers questions based only on provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # Low temperature for factual, consistent answers
            max_tokens=1024,
        )

        answer = response.choices[0].message.content

    except Exception as exc:
        return {
            "answer": None,
            "sources": [],
            "error": f"Groq API call failed: {exc}",
        }

    # ── Step 5: Collect unique sources ──
    seen = set()
    sources = []
    for chunk in retrieved_chunks:
        key = (chunk["doc_name"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"doc_name": chunk["doc_name"], "page": chunk["page"]})

    return {
        "answer": answer,
        "sources": sources,
        "error": None,
    }
