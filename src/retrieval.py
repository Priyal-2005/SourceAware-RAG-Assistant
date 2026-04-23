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

    results = []
    for doc, distance in raw_results:
        similarity = 1.0 / (1.0 + distance)
        results.append({
            "text": doc.page_content,
            "doc_name": doc.metadata.get("doc_name", "Unknown"),
            "page": doc.metadata.get("page", 0),
            "score": round(similarity, 4),
        })

    return results
