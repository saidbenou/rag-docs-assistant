"""Page-by-page PDF reading that keeps the source file and page number."""
import logging
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> list[Document]:
    """Return one Document per non-empty page, with its metadata."""
    path = Path(path)
    docs: list[Document] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_number,
                        "total_pages": pdf.page_count,
                    },
                )
            )
    if not docs:
        logger.warning("No text extracted from %s (scanned PDF? OCR required)", path.name)
    return docs


def load_directory(directory: str | Path) -> list[Document]:
    """Load every PDF in a directory (recursively)."""
    directory = Path(directory)
    docs: list[Document] = []
    for pdf_path in sorted(directory.rglob("*.pdf")):
        logger.info("Reading %s", pdf_path.name)
        docs.extend(load_pdf(pdf_path))
    return docs
