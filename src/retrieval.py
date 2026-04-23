"""
retrieval.py — Query embedding and FAISS similarity search
============================================================

How retrieval works:
  1. User types a question.
  2. We embed that question into a 384-dim vector (same model as indexing).
  3. FAISS finds the closest chunk vectors — the most relevant chunks.
  4. We return chunk text + source metadata + similarity score.

Score conversion:
  FAISS returns L2 distance (lower = more similar).
  We convert to:  similarity = 1 / (1 + distance)  →  0–1 scale, higher = better.
"""



def retrieve_chunks(query: str, vector_store, top_k: int = 5) -> list[dict]:
    """
    Search FAISS for the most relevant chunks to the user's query.

    Returns list of {"text", "doc_name", "page", "score"} dicts.
    Returns empty list on failure.
    """
    try:
        raw_results = vector_store.similarity_search_with_score(query, k=top_k)
    except Exception:
        return []

    # Simple keyword extraction for explanation
    # Ignore short words for cleaner overlap matching
    query_words = {w.lower() for w in query.split() if len(w) > 3}

    results = []
    for doc, distance in raw_results:
        similarity = 1.0 / (1.0 + distance)
        
        # ── Feature 1: Explainable Retrieval ──
        # Find which query keywords appear in this chunk
        chunk_text = doc.page_content
        chunk_words = set(chunk_text.lower().split())
        overlap = query_words.intersection(chunk_words)
        
        # Build explanation string
        reason = f"Semantic match (score: {similarity:.2f})"
        if overlap:
            reason += f" + Keyword match: {', '.join(list(overlap)[:3])}"
            
        results.append({
            "text": chunk_text,
            "doc_name": doc.metadata.get("doc_name", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "score": round(similarity, 4),
            "reason": reason
        })

    return results
