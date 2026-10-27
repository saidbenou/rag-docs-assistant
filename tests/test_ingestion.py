import fitz
from langchain_core.documents import Document

from app.ingestion.chunker import split_documents
from app.ingestion.loader import load_pdf


def _make_pdf(path, pages):
    pdf = fitz.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf.save(path)


def test_load_pdf_keeps_page_numbers_and_skips_empty(tmp_path):
    path = tmp_path / "regulation.pdf"
    _make_pdf(path, ["Article 1: zone UA", "", "Article 3: height"])
    docs = load_pdf(path)
    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert all(d.metadata["source"] == "regulation.pdf" for d in docs)


def test_chunks_inherit_metadata_and_have_stable_ids():
    doc = Document(page_content="word " * 600, metadata={"source": "a.pdf", "page": 2})
    first = split_documents([doc], chunk_size=500, chunk_overlap=50)
    second = split_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(first) > 1
    assert all(c.metadata["page"] == 2 for c in first)
    assert [c.metadata["chunk_id"] for c in first] == [c.metadata["chunk_id"] for c in second]
