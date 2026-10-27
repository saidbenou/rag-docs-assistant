"""Access to ChromaDB and the embedding model (loaded only once)."""
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    settings = get_settings()
    if settings.chroma_host:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    else:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )
