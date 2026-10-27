"""Context and citation formatting (pure functions, testable without an LLM)."""
import re

from langchain_core.documents import Document


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {d.metadata['source']}, page {d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def extract_citations(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


def build_sources(docs: list[Document], cited: set[int]) -> list[dict]:
    return [
        {
            "id": i,
            "source": d.metadata["source"],
            "page": d.metadata["page"],
            "score": d.metadata.get("score"),
            "excerpt": d.page_content[:300],
            "cited": i in cited,
        }
        for i, d in enumerate(docs, start=1)
    ]
