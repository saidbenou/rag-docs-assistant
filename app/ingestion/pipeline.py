"""Ingestion: PDF -> pages -> chunks -> ChromaDB."""
import logging
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.ingestion.chunker import split_documents
from app.ingestion.loader import load_directory, load_pdf
from app.retrieval.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)
BATCH_SIZE = 64


def _index(pages: list[Document]) -> dict:
    settings = get_settings()
    chunks = split_documents(pages, settings.chunk_size, settings.chunk_overlap)
    if chunks:
        store = get_vectorstore()
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            store.add_documents(batch, ids=[c.metadata["chunk_id"] for c in batch])
    files = {p.metadata["source"] for p in pages}
    logger.info("%d files, %d pages, %d chunks indexed", len(files), len(pages), len(chunks))
    return {"files": len(files), "pages": len(pages), "chunks": len(chunks)}


def ingest_directory(directory: str | Path | None = None) -> dict:
    return _index(load_directory(directory or get_settings().data_dir))


def ingest_file(path: str | Path) -> dict:
    return _index(load_pdf(path))
