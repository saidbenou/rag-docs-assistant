"""Answer generation with numbered citations [1], [2]..."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.generation.citations import build_sources, extract_citations, format_context
from app.generation.llm import get_llm
from app.observability import langchain_callbacks, observation
from app.retrieval.retriever import retrieve

SYSTEM_PROMPT = """You are an assistant that answers questions from document excerpts.
Rules:
- Answer only from the excerpts provided.
- Cite your sources with their number in square brackets, e.g. [1] or [2][3], right after the statement they support.
- If the excerpts do not contain the answer, say so clearly, without making anything up.
- Answer in the same language as the question, concisely and precisely."""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "Excerpts:\n\n{context}\n\nQuestion: {question}")]
)

NO_DOCS_ANSWER = "No relevant passage was found. Have the documents been ingested?"


def answer(question: str, k: int | None = None) -> dict:
    """Return the answer, its sources, the full passages (for evaluation)
    and the Langfuse trace ID (None when monitoring is disabled)."""
    with observation("rag-answer", as_type="chain", input={"question": question}) as trace:
        with observation("retrieval", as_type="retriever", input={"question": question, "k": k}) as span:
            docs = retrieve(question, k)
            if span:
                span.update(output=[{**d.metadata, "excerpt": d.page_content[:200]} for d in docs])
        if not docs:
            text, sources = NO_DOCS_ANSWER, []
        else:
            chain = PROMPT | get_llm() | StrOutputParser()
            text = chain.invoke(
                {"context": format_context(docs), "question": question},
                config={"callbacks": langchain_callbacks(), "run_name": "generation"},
            )
            sources = build_sources(docs, extract_citations(text))
        if trace:
            trace.update(output={"answer": text, "cited": [s["id"] for s in sources if s["cited"]]})
    return {
        "answer": text,
        "sources": sources,
        "contexts": [d.page_content for d in docs],
        "trace_id": trace.trace_id if trace else None,
    }
