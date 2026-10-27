"""Langfuse monitoring: traces of RAG requests and scores (evaluation, user feedback).

Everything is optional: without LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY, the functions
below do nothing and the assistant works as usual.
"""
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from typing import Any, Iterator

from app.config import get_settings


@lru_cache
def get_langfuse():
    """Langfuse client, or None when monitoring is not configured."""
    s = get_settings()
    if not (s.langfuse_public_key and s.langfuse_secret_key):
        return None
    from langfuse import Langfuse

    return Langfuse(public_key=s.langfuse_public_key, secret_key=s.langfuse_secret_key, host=s.langfuse_host)


def langchain_callbacks() -> list:
    """LangChain callback that records the LLM call (prompt, tokens, latency) in the current trace."""
    lf = get_langfuse()
    if lf is None:
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler(public_key=get_settings().langfuse_public_key)]


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """Open a Langfuse observation (span, retriever, chain...); yields None when disabled."""
    lf = get_langfuse()
    ctx = lf.start_as_current_observation(name=name, as_type=as_type, **kwargs) if lf else nullcontext()
    with ctx as obs:
        yield obs


def score(trace_id: str | None, name: str, value: float | str, comment: str | None = None) -> bool:
    """Attach a score to a trace. Returns False when Langfuse is disabled or the trace is unknown."""
    lf = get_langfuse()
    if lf is None or not trace_id:
        return False
    lf.create_score(trace_id=trace_id, name=name, value=value, comment=comment)
    return True


def trace_url(trace_id: str | None) -> str | None:
    """Link to the trace in the Langfuse UI (None when unavailable)."""
    lf = get_langfuse()
    if lf is None or not trace_id:
        return None
    try:
        return lf.get_trace_url(trace_id=trace_id)  # asks the server for the project ID
    except Exception:
        return None


def flush() -> None:
    lf = get_langfuse()
    if lf is not None:
        lf.flush()
