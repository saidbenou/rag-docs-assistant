"""Retrieval of relevant passages.

Baseline version: vector search only.
Next steps: hybrid BM25 + vectors, then a reranker.
"""
from langchain_core.documents import Document

from app.config import get_settings
from app.retrieval.vectorstore import get_vectorstore


def retrieve(question: str, k: int | None = None) -> list[Document]:
    k = k or get_settings().top_k
    results = get_vectorstore().similarity_search_with_relevance_scores(question, k=k)
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(float(score), 4)
        docs.append(doc)
    return docs
