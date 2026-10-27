from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import observability
from app.api.main import app
from app.evaluation.ragas_metrics import mean, score_row
from app.generation import chain
from scripts.evaluate import REFUSAL, answer_ok

DOCS = [Document(page_content="Max height 18.50 m.", metadata={"source": "plan.pdf", "page": 2, "score": 0.7})]


def test_answer_ok_compares_numbers():
    # The corpus is in French: answers use decimal commas or points.
    assert answer_ok("18,50 mètres, acrotère compris.", "La hauteur maximale est de 18.50 m [1].")
    assert not answer_ok("RTO de 8 heures et RPO de 4 heures.", "Le RTO est de 8 heures [1].")
    assert answer_ok("Oui, mais de 1,00 mètre au maximum.", "Oui, de 1,00 m au plus [1].")


def test_refusal_detection_in_french_and_english():
    assert REFUSAL.search("Les extraits ne mentionnent pas le nombre de jours de télétravail.")
    assert REFUSAL.search("The excerpts do not mention how many remote-work days are allowed.")
    assert not REFUSAL.search("Remote work is allowed 2 days a week [1].")


def test_observability_is_noop_without_keys():
    assert observability.get_langfuse() is None
    assert observability.langchain_callbacks() == []
    assert observability.score("trace", "x", 1) is False
    with observability.observation("span") as obs:
        assert obs is None


def test_answer_returns_contexts_without_monitoring(monkeypatch):
    monkeypatch.setattr(chain, "retrieve", lambda q, k=None: DOCS)
    monkeypatch.setattr(chain, "get_llm", lambda: FakeListChatModel(responses=["18.50 m [1]."]))
    result = chain.answer("Max height?")
    assert result["contexts"] == ["Max height 18.50 m."]
    assert result["trace_id"] is None
    assert result["sources"][0]["cited"] is True


def test_feedback_requires_langfuse():
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok", "monitoring": False}
    assert client.post("/feedback", json={"trace_id": "abc", "value": 1}).status_code == 501


class _Result:
    def __init__(self, value):
        self.value = value


def test_score_row_tolerates_judge_failures():
    def broken(_row):
        raise ValueError("invalid JSON")

    scores = score_row({"id": "q1"}, {"faithfulness": lambda r: _Result(0.8), "context_recall": broken})
    assert scores == {"faithfulness": 0.8, "context_recall": None}
    assert mean([0.8, None, 0.6]) == 0.7
