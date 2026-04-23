"""
src — Core logic modules for the SourceAware RAG Assistant.

This package separates all processing logic from the Streamlit UI:
  • pdf_processing : PDF upload → structured page-level text
  • chunking       : Page text → overlapping sub-chunks
  • embeddings     : HuggingFace embedding model loading
  • vector_store   : FAISS index build / save / load
  • retrieval      : Query embedding + similarity search
"""
