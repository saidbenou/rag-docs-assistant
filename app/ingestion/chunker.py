"""Chunking with deterministic IDs (idempotent re-ingestion)."""
import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def make_chunk_id(source: str, page: int, content: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{content}".encode("utf-8")).hexdigest()[:16]
    return f"{source}:p{page}:{digest}"


def split_documents(
    docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)  # page metadata is inherited
    for chunk in chunks:
        chunk.metadata["chunk_id"] = make_chunk_id(
            chunk.metadata["source"], chunk.metadata["page"], chunk.page_content
        )
    return chunks
