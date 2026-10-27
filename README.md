# RAG Docs Assistant

Question-answering assistant over a corpus of PDFs (urban planning regulations, reports, technical documentation). Every answer cites its sources: document, page and excerpt.

**Stack:** Python · LangChain · ChromaDB · bge-m3 · Ollama (or any OpenAI-compatible API) · FastAPI · Docker · Ragas · Langfuse

## Architecture

```
PDF ──► PyMuPDF (text + page number) ──► chunks ──► bge-m3 embeddings ──► ChromaDB
                                                                            │
Question ──► embedding ──► top-k passages ◄─────────────────────────────────┘
                              │
                              ▼
          LLM (numbered excerpts in the prompt) ──► answer + citations [1][2]
```

## Results

Evaluation set: 24 questions ([eval/questions.json](eval/questions.json)) over 3 fictional demo documents
written in French, including 3 out-of-corpus questions to check that the assistant does not make things up.

| Metric | Score |
|---|---:|
| Hit@1 — right page ranked first (21 questions) | 90% |
| Hit@5 — right page in the top 5 | 100% |
| MRR | 0.95 |
| Ragas `faithfulness` (sample of 3 questions, qwen2.5:7b judge) | 1.00 |
| Ragas `answer_accuracy` (same) | 0.92 |
| Ragas `context_recall` (same) | 1.00 |

*Setup: bge-m3, 1,000-character chunks (150 overlap), top-k 5, qwen2.5:7b via Ollama.*

## Project structure

```
app/
├── ingestion/        PDF reading (PyMuPDF), chunking, idempotent indexing
├── retrieval/        ChromaDB + bge-m3 embeddings, top-k search
├── generation/       prompt, LLM (Ollama / OpenAI), [n] citations
├── evaluation/       Ragas metrics (judge LLM)
├── observability.py  Langfuse traces and scores
├── api/              FastAPI API
└── config.py         configuration (.env)
scripts/              CLI: ingest, ask, evaluate
eval/                 reference question-answer set
tests/                unit tests (no LLM, no network)
```

## Quick start

### Local

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env

ollama pull qwen2.5:7b                 # free local LLM
# 3 demo PDFs ship in data/pdfs/; add your own there

python -m scripts.ingest
python -m scripts.ask "Quelle est la hauteur maximale des constructions dans le secteur A4 ?"

uvicorn app.api.main:app --reload      # interactive docs: http://localhost:8000/docs
```

The assistant answers in the language of the question, so it works with documents and questions in English, French or any language supported by the LLM and by bge-m3.

### Docker

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull qwen2.5:7b
curl -X POST localhost:8000/ingest
```

## API

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | Health check (also reports whether monitoring is on) |
| POST | `/ingest` | Index every PDF in `DATA_DIR` |
| POST | `/ingest/upload` | Upload and index a PDF |
| POST | `/ask` | Ask a question; returns answer + sources + `trace_id` |
| POST | `/feedback` | Rate an answer (1 helpful / 0 wrong) in Langfuse |

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Quels reculs une villa doit-elle respecter ?"}'
```

## Command line

```bash
python -m scripts.ask "Your question"     # single question
python -m scripts.ask                     # interactive mode (models loaded only once)
```

## Evaluation

```bash
python -m scripts.evaluate --retrieval-only   # retrieval only, fast, no LLM
python -m scripts.evaluate                    # + LLM answers
python -m scripts.evaluate --ragas            # + Ragas metrics (judge LLM)
python -m scripts.evaluate --ragas --metrics faithfulness,context_recall --limit 5
```

| Level | Metrics |
|---|---|
| Retrieval | Hit@1, Hit@k, MRR (whether the expected page is found, and at which rank) |
| Answer | accuracy (expected numbers present), refusal on out-of-corpus questions |
| Ragas | `faithfulness` (answer grounded in the passages), `answer_accuracy` (agreement with the reference), `context_recall` (passages sufficient) — optionally `context_precision`, `answer_relevancy`, `factual_correctness` |

By default the judge LLM is the assistant's own; `EVAL_LLM_PROVIDER` / `EVAL_MODEL` let you pick a
stronger one (a local 7B judge is slow and sometimes unreliable). Judge calls are cached in
`eval/.ragas_cache/`, so re-running an evaluation does not repeat them.
Each run writes `eval/report_<date>.md` and `eval/results_<date>.json`.

## Monitoring (Langfuse)

Set `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` in `.env`
([Langfuse Cloud](https://cloud.langfuse.com) free tier, or a self-hosted instance).
Without keys, monitoring is simply turned off.

Each question produces a `rag-answer` trace:

```
rag-answer (chain)        question → answer, cited sources
├── retrieval (retriever) passages, pages and scores
└── generation (LLM)      full prompt, model, tokens, latency
```

Scores are attached to the traces:

- `user-feedback`: the user's rating via `POST /feedback` (`{"trace_id": "...", "value": 1}`);
- `eval-correct`, `eval-retrieval-rank`, `ragas-*`: sent by `scripts.evaluate`.

`/ask` returns the `trace_id`; `/health` reports whether monitoring is on.

## Tests

```bash
pytest -q
```

## Roadmap

- [x] Ingestion with page metadata and idempotent IDs
- [x] Vector search + generation with citations
- [x] FastAPI API + Docker Compose
- [ ] Hybrid search (BM25 + vectors)
- [ ] Reranking (bge-reranker)
- [x] Evaluation set + Ragas metrics
- [x] Langfuse monitoring (traces, evaluation scores, user feedback)
- [ ] Streamlit UI
- [x] GitHub Actions CI (unit tests)

## License

MIT — see [LICENSE](LICENSE).
