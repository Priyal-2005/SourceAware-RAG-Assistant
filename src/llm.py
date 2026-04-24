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
# Using llama3-8b-8192 as it is reliable and fast for RAG tasks.
MODEL_NAME="llama-3.3-70b-versatile"
DEFAULT_MODEL = MODEL_NAME

# ── Debug configuration ──
# Set to True to see raw API errors. False for clean UI errors.
DEBUG_MODE = False

# ── Prompt templates ──
# Why this template works:
#   1. "ONLY using the provided context" — prevents hallucination
#   2. "If the answer is not present" — gives the model an explicit fallback
#   3. Context is injected with source labels — so the model can cite them
#   4. Chat history gives continuity for follow-up questions
#   5. Clean separation of history / context / question / answer sections

# Template WITH conversation history (for follow-up questions)
PROMPT_WITH_HISTORY = """You are a helpful AI research assistant.

Answer the question ONLY using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Always mention which document and page your answer comes from.

Previous conversation:
{chat_history}

Context:
{context}

Question:
{query}

Answer:"""

# Template WITHOUT history (first question in a conversation)
PROMPT_NO_HISTORY = """You are a helpful AI research assistant.

Answer the question ONLY using the provided context.
If the answer is not present in the context, say:
"I don't know based on the provided documents."

Always mention which document and page your answer comes from.

Context:
{context}

Question:
{query}

Answer:"""

# ADD THIS: Template for true multi-document comparison
PROMPT_COMPARE = """You are a research assistant.

Context grouped by document:
{context}

Task:
* Summarize key points from EACH document separately
* Then compare them
* Highlight similarities and differences

Output format:
Document-wise Summary:
* [doc1]: ...
* [doc2]: ...

Comparison:
* Similarities: ...
* Differences: ...

Question:
{query}

Answer:"""


def _build_context(retrieved_chunks, compare_mode: bool = False) -> str:
    """
    Combine retrieved chunks into a single context string with source labels.
    """
    # ADD THIS: Handle compare mode grouping
    if compare_mode:
        context_parts = []
        for doc_name, chunks in retrieved_chunks.items():
            context_parts.append(f"Document: {doc_name}")
            for chunk in chunks:
                context_parts.append(f"[Page: {chunk['page']}]\n{chunk['text']}")
            context_parts.append("") # blank line between docs
        return "\n".join(context_parts)
    
    # Existing linear context building
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


def _format_chat_history(chat_history: list[dict], max_turns: int = 3) -> str:
    """
    Format recent chat history into a readable string for the prompt.

    Why we limit to the last few turns:
      LLM context windows are finite.  Including the entire conversation
      would eat into the space available for document context.  The last
      2–3 exchanges are usually enough for follow-up questions like
      "tell me more about that" or "what about page 5?".

    Args:
        chat_history: List of {"role": "user"|"assistant", "content": str}.
        max_turns:    Max Q&A pairs to include (default 3).

    Returns:
        Formatted string, or empty string if no history.
    """
    if not chat_history:
        return ""

    # Take the last N*2 messages (each turn = 1 user + 1 assistant)
    recent = chat_history[-(max_turns * 2):]

    lines = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")

    return "\n".join(lines)


def generate_answer(
    query: str,
    retrieved_chunks, # list or dict depending on compare_mode
    chat_history: list[dict] | None = None,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    compare_mode: bool = False,
) -> dict:
    """
    Generate an LLM answer from retrieved chunks using Groq.

    Step 1: Build context string from retrieved chunks (with source labels).
    Step 2: Format recent chat history (if any).
    Step 3: Fill the prompt template with history + context + query.
    Step 4: Call Groq API to generate the answer.
    Step 5: Return the answer + source metadata.

    Why we return sources separately:
      The UI can display a clean "Sources used" section below the answer,
      making it easy for users to verify the information.

    Args:
        query:            The user's question.
        retrieved_chunks: List of {"text", "doc_name", "page", "score"} dicts
                          from the retrieval step.
        chat_history:     Optional list of {"role", "content"} dicts for memory.
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
            "confidence": "no context"
        }

    # ── MODIFY THIS: Flatten chunks for confidence and source tracking ──
    all_chunks = []
    if compare_mode:
        for chunks in retrieved_chunks.values():
            all_chunks.extend(chunks)
    else:
        all_chunks = retrieved_chunks

    # ── Feature 2: Confidence Detection ──
    top_score = max((chunk.get("score", 0) for chunk in all_chunks), default=0)
    if top_score < 0.4:
        confidence = "low"
    elif top_score < 0.7:
        confidence = "medium"
    else:
        confidence = "high"

    # ── Step 1: Build context ──
    context = _build_context(retrieved_chunks, compare_mode)

    # ── Step 2 & 3: Build prompt ──
    history_str = _format_chat_history(chat_history or [])
    
    # ADD THIS: Select compare prompt or normal prompt
    if compare_mode:
        prompt = PROMPT_COMPARE.format(context=context, query=query)
        if history_str:
            prompt = f"Previous conversation:\n{history_str}\n\n" + prompt
    elif history_str:
        prompt = PROMPT_WITH_HISTORY.format(
            chat_history=history_str, context=context, query=query
        )
    else:
        prompt = PROMPT_NO_HISTORY.format(context=context, query=query)

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

        # Feature 2 (cont): Prepend warning if low confidence
        if confidence == "low":
            warning = "⚠️ *This answer may be unreliable due to low context relevance.* \n\n"
            answer = warning + answer

    except Exception as exc:
        # Feature 3: Debug Mode handling for fallback
        if DEBUG_MODE:
            error_msg = f"Groq API call failed (Debug): {exc}"
        else:
            error_msg = "LLM temporarily unavailable. Please try again."
            
        return {
            "answer": None,
            "sources": [],
            "error": error_msg,
            "confidence": confidence
        }

    # ── Step 5: Collect unique sources ──
    seen = set()
    sources = []
    for chunk in all_chunks:
        key = (chunk["doc_name"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"doc_name": chunk["doc_name"], "page": chunk["page"]})

    return {
        "answer": answer,
        "sources": sources,
        "error": None,
        "confidence": confidence
    }
