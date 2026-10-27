"""REST API: ingestion and question answering."""
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import observability
from app.config import get_settings
from app.generation.chain import answer
from app.ingestion.pipeline import ingest_directory, ingest_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    observability.flush()  # send the last traces before shutting down


app = FastAPI(title="RAG Docs Assistant", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["Quelle est la hauteur maximale des constructions dans le secteur A4 ?"])
    top_k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    id: int
    source: str
    page: int
    score: float | None
    excerpt: str
    cited: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    trace_id: str | None = Field(default=None, description="Langfuse trace, to send back to /feedback")


class FeedbackRequest(BaseModel):
    trace_id: str
    value: int = Field(ge=0, le=1, description="1 = helpful answer, 0 = wrong answer")
    comment: str | None = None


class IngestResponse(BaseModel):
    files: int
    pages: int
    chunks: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "monitoring": observability.get_langfuse() is not None}


@app.post("/ingest", response_model=IngestResponse)
def ingest_all() -> dict:
    """Index every PDF in the DATA_DIR folder."""
    return ingest_directory()


@app.post("/ingest/upload", response_model=IngestResponse)
def ingest_upload(file: UploadFile) -> dict:
    """Upload a PDF, then index it."""
    filename = Path(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    target = Path(get_settings().data_dir) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return ingest_file(target)


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> dict:
    try:
        return answer(req.question, req.top_k)
    except Exception as exc:  # LLM unreachable, etc.
        logging.exception("Error during /ask")
        raise HTTPException(status_code=503, detail=f"Generation error: {exc}") from exc


@app.post("/feedback", status_code=204)
def feedback(req: FeedbackRequest) -> None:
    """Record the user's opinion of an answer as a Langfuse score."""
    if not observability.score(req.trace_id, "user-feedback", req.value, req.comment):
        raise HTTPException(status_code=501, detail="Langfuse monitoring is not configured.")
