"""Ragas metrics ("judge" LLM) to score the RAG answers.

Available metrics (all between 0 and 1):
- faithfulness       : is the answer grounded only in the retrieved passages?
- answer_accuracy    : does the answer agree with the expected answer?
- context_recall     : do the retrieved passages contain the expected information?
- context_precision  : are the relevant passages ranked first?
- answer_relevancy   : does the answer address the question? (also needs embeddings)
- factual_correctness: claim overlap between answer and reference (expensive)
"""
import logging
import math
from functools import lru_cache
from typing import Callable

from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_METRICS = ["faithfulness", "answer_accuracy", "context_recall"]
ALL_METRICS = [*DEFAULT_METRICS, "context_precision", "answer_relevancy", "factual_correctness"]
CACHE_DIR = "eval/.ragas_cache"


def judge_model() -> str:
    s = get_settings()
    provider = s.eval_llm_provider or s.llm_provider
    default = s.ollama_model if provider == "ollama" else s.openai_model
    return s.eval_model or default


@lru_cache
def get_judge():
    """Judge LLM. Ollama is called through its OpenAI-compatible API (/v1)."""
    from openai import AsyncOpenAI  # Ragas metrics call the judge asynchronously
    from ragas.cache import DiskCacheBackend
    from ragas.llms import llm_factory

    s = get_settings()
    provider = s.eval_llm_provider or s.llm_provider
    if provider == "ollama":
        client = AsyncOpenAI(base_url=f"{s.ollama_base_url.rstrip('/')}/v1", api_key="ollama")
    elif provider == "openai":
        client = AsyncOpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url or None)
    else:
        raise ValueError(f"Unknown EVAL_LLM_PROVIDER: {provider}")
    # The disk cache avoids paying again for calls already made when the evaluation is re-run.
    return llm_factory(judge_model(), client=client, cache=DiskCacheBackend(cache_dir=CACHE_DIR), max_tokens=2048)


def build_scorers(names: list[str]) -> dict[str, Callable[[dict], float]]:
    """Map each metric to a row -> score function.

    A row contains: question, answer, contexts (full passages), expected.
    """
    from ragas.metrics import collections as rm

    llm = get_judge()
    scorers: dict[str, Callable[[dict], float]] = {}
    for name in names:
        if name == "faithfulness":
            m = rm.Faithfulness(llm=llm)
            scorers[name] = lambda r, m=m: m.score(
                user_input=r["question"], response=r["answer"], retrieved_contexts=r["contexts"])
        elif name == "answer_accuracy":
            m = rm.AnswerAccuracy(llm=llm)
            scorers[name] = lambda r, m=m: m.score(
                user_input=r["question"], response=r["answer"], reference=r["expected"])
        elif name == "context_recall":
            m = rm.ContextRecall(llm=llm)
            scorers[name] = lambda r, m=m: m.score(
                user_input=r["question"], retrieved_contexts=r["contexts"], reference=r["expected"])
        elif name == "context_precision":
            m = rm.ContextPrecisionWithReference(llm=llm)
            scorers[name] = lambda r, m=m: m.score(
                user_input=r["question"], reference=r["expected"], retrieved_contexts=r["contexts"])
        elif name == "answer_relevancy":
            from ragas.embeddings import HuggingFaceEmbeddings

            m = rm.AnswerRelevancy(llm=llm, embeddings=HuggingFaceEmbeddings(model=get_settings().embedding_model))
            scorers[name] = lambda r, m=m: m.score(user_input=r["question"], response=r["answer"])
        elif name == "factual_correctness":
            m = rm.FactualCorrectness(llm=llm)
            scorers[name] = lambda r, m=m: m.score(response=r["answer"], reference=r["expected"])
        else:
            raise ValueError(f"Unknown metric: {name} (available: {', '.join(ALL_METRICS)})")
    return scorers


def score_row(row: dict, scorers: dict[str, Callable[[dict], float]]) -> dict[str, float | None]:
    """Compute each metric; a failure (invalid JSON from the judge...) gives None."""
    scores: dict[str, float | None] = {}
    for name, scorer in scorers.items():
        try:
            value = float(scorer(row).value)
            scores[name] = None if math.isnan(value) else round(value, 3)
        except Exception as exc:
            logger.warning("%s: %s failed (%s)", row.get("id"), name, exc)
            scores[name] = None
    return scores


def mean(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    return sum(valid) / len(valid) if valid else None
