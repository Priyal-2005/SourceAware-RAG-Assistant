"""
chunking.py — Split page-level text into smaller, overlapping chunks
=====================================================================

Why we chunk:
  A single PDF page can contain 2,000+ characters covering multiple topics.
  If we embed the whole page as one vector, a search query about Topic A
  will also pull in unrelated Topic B text from the same page.  Smaller
  chunks (500–700 chars) give the retriever surgical precision.

Why we use overlap:
  Imagine a sentence that starts at the end of Chunk 1 and finishes at the
  start of Chunk 2.  Without overlap, that sentence would be split in half
  and neither chunk would fully capture its meaning.  An overlap of 50–100
  characters ensures border sentences appear in both adjacent chunks.

Metadata preservation:
  Every chunk inherits the doc_name and page number from its parent page.
  This is what makes the system "source-aware" — we can always trace a
  retrieved chunk back to "file.pdf, page 7".
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


# ── Default parameters ──
DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 80


def chunk_documents(
    extracted_pages: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split page-level text into smaller chunks with metadata.

    Step 1: Create a text splitter with the chosen size and overlap.
    Step 2: For each page record, split its text into sub-chunks.
    Step 3: Attach the original doc_name + page to every sub-chunk.

    Why RecursiveCharacterTextSplitter?
      It tries to split on natural boundaries first (double newlines,
      then single newlines, then sentences, then words) before falling
      back to raw character counts.  This produces more coherent chunks
      than a naive fixed-window split.

    Args:
        extracted_pages: List of {"doc_name", "page", "text"} dicts.
        chunk_size:      Max characters per chunk.
        chunk_overlap:   Characters shared between consecutive chunks.

    Returns:
        List of {"text", "doc_name", "page"} dicts — one per chunk.
    """
    # Step 1: Configure the splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # prefer natural boundaries
    )

    chunks: list[dict] = []

    # Step 2 & 3: Split each page and carry forward metadata
    for record in extracted_pages:
        text_pieces = splitter.split_text(record["text"])

        for piece in text_pieces:
            chunks.append({
                "text": piece,
                "doc_name": record["doc_name"],
                "page": record["page"],
            })

    return chunks
