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



def retrieve_chunks(query: str, vector_store, top_k: int = 5, compare_mode: bool = False):
    """
    Search FAISS for the most relevant chunks to the user's query.

    If compare_mode=False: Returns list of {"text", "doc_name", "page", "score"} dicts.
    If compare_mode=True: Returns dict mapping doc_name to list of chunk dicts.
    """
    try:
        # MODIFY THIS: Fetch plenty if compare mode to ensure we get chunks from multiple docs
        k = 100 if compare_mode else top_k
        raw_results = vector_store.similarity_search_with_score(query, k=k)
    except Exception:
        return {} if compare_mode else []

    # Simple keyword extraction for explanation
    # Ignore short words for cleaner overlap matching
    query_words = {w.lower() for w in query.split() if len(w) > 3}

    # ADD THIS: Grouping logic for true multi-document comparison
    if compare_mode:
        grouped_results = {}
        for doc, distance in raw_results:
            doc_name = doc.metadata.get("doc_name", "Unknown")
            
            # Keep only top_k chunks per document
            if len(grouped_results.get(doc_name, [])) >= top_k:
                continue
                
            similarity = 1.0 / (1.0 + distance)
            chunk_text = doc.page_content
            chunk_words = set(chunk_text.lower().split())
            overlap = query_words.intersection(chunk_words)
            reason = f"Semantic match (score: {similarity:.2f})"
            if overlap:
                reason += f" + Keyword match: {', '.join(list(overlap)[:3])}"
                
            chunk_data = {
                "text": chunk_text,
                "doc_name": doc_name,
                "page": doc.metadata.get("page", 0),
                "score": round(similarity, 4),
                "reason": reason
            }
            
            if doc_name not in grouped_results:
                grouped_results[doc_name] = []
            grouped_results[doc_name].append(chunk_data)
            
        return grouped_results

    else:
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
