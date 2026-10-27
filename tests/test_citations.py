import re

from langchain_core.documents import Document

from app.generation.citations import build_sources, extract_citations, format_context
from app.ingestion.chunker import make_chunk_id

DOCS = [
    Document(page_content="Max height 12 m.", metadata={"source": "plan.pdf", "page": 4, "score": 0.8}),
    Document(page_content="Setback of 5 m.", metadata={"source": "plan.pdf", "page": 7, "score": 0.6}),
]


def test_format_context_numbers_sources():
    ctx = format_context(DOCS)
    assert "[1] (source: plan.pdf, page 4)" in ctx
    assert "[2] (source: plan.pdf, page 7)" in ctx


def test_cited_flags():
    sources = build_sources(DOCS, extract_citations("The height is 12 m [1]."))
    assert [s["cited"] for s in sources] == [True, False]


def test_chunk_id_format():
    assert re.fullmatch(r"plan\.pdf:p4:[0-9a-f]{16}", make_chunk_id("plan.pdf", 4, "text"))
