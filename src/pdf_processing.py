"""
pdf_processing.py — PDF text extraction with source metadata
=============================================================

Why this module exists:
  The first step in any RAG pipeline is getting raw text out of documents.
  This module reads PDF files page-by-page and returns structured records
  that preserve the source (file name + page number).  Keeping this metadata
  is critical — it lets us show users exactly WHERE an answer came from.

Data format produced:
  [{"doc_name": "report.pdf", "page": 1, "text": "..."}, ...]
"""

from pypdf import PdfReader


def extract_text_from_pdf(file) -> tuple[list[dict], list[str]]:
    """
    Extract text from a single PDF file, page by page.

    Step 1: Read the PDF using pypdf.
    Step 2: Iterate over every page and pull out the text.
    Step 3: Skip pages with no extractable text (e.g. scanned images).
    Step 4: Return structured records + any warning messages.

    Why we return warnings separately (instead of calling st.warning):
      This module has NO dependency on Streamlit.  That means it can be
      reused in scripts, notebooks, or tests — the UI layer decides how
      to display the warnings.

    Args:
        file: A file-like object with a `.name` attribute (e.g. Streamlit
              UploadedFile, or a regular open() file handle).

    Returns:
        (pages, warnings) where:
          pages    — list of {"doc_name", "page", "text"} dicts
          warnings — list of human-readable warning strings
    """
    doc_name = file.name
    pages: list[dict] = []
    warnings: list[str] = []

    # ── Step 1: Attempt to read the PDF ──
    try:
        reader = PdfReader(file)
    except Exception as exc:
        # Corrupted or password-protected PDFs end up here
        warnings.append(f"❌ Could not read **{doc_name}** — the file may be corrupted.\n\n`{exc}`")
        return pages, warnings

    # ── Step 2: Check for empty PDF ──
    if len(reader.pages) == 0:
        warnings.append(f"⚠️ **{doc_name}** has no pages.")
        return pages, warnings

    # ── Step 3: Extract text page-by-page ──
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""

        # Skip pages with no extractable text (scanned images, etc.)
        if not text.strip():
            warnings.append(
                f"⚠️ **{doc_name}** — Page {page_number} has no extractable text (skipped)."
            )
            continue

        pages.append({
            "doc_name": doc_name,
            "page": page_number,
            "text": text,
        })

    return pages, warnings


def extract_text_from_pdfs(uploaded_files) -> tuple[list[dict], list[str]]:
    """
    Extract text from multiple PDF files.

    This is a convenience wrapper that calls extract_text_from_pdf()
    for each file and combines all results into a single flat list.

    Args:
        uploaded_files: Iterable of file-like objects.

    Returns:
        (all_pages, all_warnings) — combined across all files.
    """
    all_pages: list[dict] = []
    all_warnings: list[str] = []

    for file in uploaded_files:
        pages, warnings = extract_text_from_pdf(file)
        all_pages.extend(pages)
        all_warnings.extend(warnings)

    return all_pages, all_warnings
