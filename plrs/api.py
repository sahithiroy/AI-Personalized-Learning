"""REST API + web frontend (FastAPI).

Start with:  python -m plrs serve
  http://127.0.0.1:8000/       web frontend (plrs/static/index.html)
  http://127.0.0.1:8000/docs   interactive API docs

Endpoints mirror the modules of Fig. 1: curriculum ingestion, knowledge-gap
analysis, remedial recommendation, MCQ generation and adaptive evaluation.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .data import SAMPLE_DIR, Record, by_learner, load_records
from .evaluation import AdaptiveEvaluation, EvaluationRule
from .gap import analyze_learner
from .kt import KnowledgeTracer
from .qgen import QuestionGenerator
from .rag import CurriculumKB
from .remedial import RemedialRecommender, to_markdown

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
_state: dict = {"sessions": {}}


@asynccontextmanager
async def lifespan(_app):
    """Load the embedding model, vector DB and EKT once at start-up so the first request is fast."""
    from .embeddings import get_embedder

    print("Loading models (the first start can take about a minute) ...", flush=True)
    get_embedder()
    try:
        _kb()
    except HTTPException as exc:
        log.warning("%s", exc.detail)
    _tracer()
    print(f"Ready: open {os.getenv('PLRS_URL', 'http://127.0.0.1:8000')}/ in your browser", flush=True)
    yield


app = FastAPI(title="AI-Driven Personalized Learning & Remedial Recommendation", version="1.0.0",
              lifespan=lifespan)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(STATIC_DIR / "index.html")


def _kb() -> CurriculumKB:
    if "kb" not in _state:
        try:
            _state["kb"] = CurriculumKB.load()
        except FileNotFoundError as exc:
            raise HTTPException(409, str(exc))
    return _state["kb"]


def _tracer() -> KnowledgeTracer | None:
    if "tracer" not in _state:
        try:
            _state["tracer"] = KnowledgeTracer.load()
        except FileNotFoundError:
            _state["tracer"] = None
    return _state["tracer"]


class RecordIn(BaseModel):
    question_id: str
    question_text: str = ""
    concept: str
    difficulty: str = "medium"
    bloom_level: str = ""
    max_marks: float
    marks: float
    exam: str = "OBE"


class LearnerIn(BaseModel):
    learner_id: str
    records: list[RecordIn]


def _records(body: LearnerIn) -> list[Record]:
    return [Record(body.learner_id, r.exam, i, r.question_id, r.question_text, r.concept, "",
                   r.difficulty, r.bloom_level, r.max_marks, r.marks) for i, r in enumerate(body.records)]


def _record_out(r: Record) -> dict:
    return {k: v for k, v in asdict(r).items() if k in RecordIn.model_fields}


def _grouped(records: list[Record]) -> dict[str, list[dict]]:
    return {lid: [_record_out(r) for r in rs] for lid, rs in by_learner(records).items()}


@app.get("/sample-learners")
def sample_learners(limit: int = 50):
    """Learner ids from the bundled current batch (data/sample/current_batch.csv)."""
    path = SAMPLE_DIR / "current_batch.csv"
    if not path.exists():
        raise HTTPException(409, "No sample data - run `python -m plrs sample-data` first.")
    if "sample" not in _state:
        _state["sample"] = _grouped(load_records(path))
    return sorted(_state["sample"])[:limit]


@app.get("/sample-learners/{learner_id}")
def sample_learner(learner_id: str):
    sample_learners()
    if learner_id not in _state["sample"]:
        raise HTTPException(404, f"unknown learner {learner_id!r}")
    return {"learner_id": learner_id, "records": _state["sample"][learner_id]}


@app.post("/results/upload")
async def upload_results(file: UploadFile = File(...)):
    """Upload an OBE results CSV (same columns as data/sample/current_batch.csv); returns records per learner."""
    tmp = Path(tempfile.mkdtemp()) / "results.csv"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        grouped = _grouped(load_records(tmp))
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, f"Could not read CSV: missing or invalid column {exc}")
    return {"learners": grouped}


@app.get("/health")
def health():
    from .llm import get_llm, get_verifiers

    from .config import get_config
    from .gap import obe_levels

    threshold, target = obe_levels()
    rule = get_config()["evaluation"]["default_rule"]
    return {"status": "ok", "primary_llm": get_llm().name, "verifiers": [v.name for v in get_verifiers()],
            "kt_model": _tracer().model_type if _tracer() else None,
            "obe": {"threshold": threshold, "target": target},
            "default_rule": f"{rule['questions_per_level']} and {rule['pass_percentage']:g}"}


@app.post("/curriculum")
async def upload_curriculum(file: UploadFile = File(...), extract_concepts: bool = True):
    """Upload a curriculum PDF/TXT, build the vector DB and extract concepts."""
    tmp = Path(tempfile.mkdtemp()) / (file.filename or "curriculum.pdf")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    kb = CurriculumKB.ingest([tmp])
    if extract_concepts:
        kb.extract_concepts()
    kb.save()
    _state["kb"] = kb
    return {"chunks": len(kb.store), "concepts": [c.__dict__ for c in kb.concepts]}


@app.get("/concepts")
def concepts():
    return [c.__dict__ for c in _kb().concepts]


@app.post("/analyze")
def analyze(body: LearnerIn):
    gaps = analyze_learner(_records(body), _tracer())
    return {"learner_id": body.learner_id, "concepts": [g.as_dict() for g in gaps]}


@app.post("/recommend")
def recommend(body: LearnerIn, round_no: int = 1):
    rec = RemedialRecommender(_kb())
    out = []
    for g in analyze_learner(_records(body), _tracer()):
        if g.is_gap:
            r = rec.recommend(g, round_no)
            out.append({"recommendation": r, "markdown": to_markdown(r)})
    return {"learner_id": body.learner_id, "recommendations": out}


class MaterialsIn(BaseModel):
    learner_id: str
    concepts: list[dict]          # /analyze "concepts"
    recommendations: list[dict]   # the "recommendation" objects from /recommend
    only: str | None = None       # one concept only (the per-concept "View PDF" button)


@app.post("/materials/pdf")
def study_materials_pdf(body: MaterialsIn):
    """Download a personal study-material PDF: recommendations + syllabus pages for every gap.

    Reuses the recommendations already shown to the learner, so no extra LLM calls are made.
    """
    from .materials import build_study_pdf

    try:
        kb = _kb()
    except HTTPException:
        kb = None
    pdf = build_study_pdf(body.learner_id, body.concepts, body.recommendations, kb, only=body.only)
    safe = "".join(ch for ch in body.learner_id if ch.isalnum() or ch in "-_") or "learner"
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="study_material_{safe}.pdf"'})


class MCQRequest(BaseModel):
    concept: str
    level: str = "easy"
    n: int = 5


@app.post("/mcq")
def generate_mcq(req: MCQRequest):
    kb = _kb()
    try:
        concept = kb.concept(req.concept)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    gen = QuestionGenerator(kb)
    qs = gen.generate(concept, req.level, req.n)
    return {"questions": [q.__dict__ for q in qs], "stats": gen.stats}


class EvalStart(BaseModel):
    learner_id: str
    concepts: list[str]
    rule: str | None = None  # e.g. "5 and 80"


@app.post("/evaluation")
def start_evaluation(req: EvalStart):
    rule = EvaluationRule.parse(req.rule) if req.rule else EvaluationRule.default()
    session = AdaptiveEvaluation(req.learner_id, req.concepts, QuestionGenerator(_kb()), rule)
    sid = uuid.uuid4().hex[:12]
    _state["sessions"][sid] = session
    return {"session_id": sid, "questions": {c: [q.as_public() for q in session.next_questions(c)]
                                             for c in req.concepts}}


class Answers(BaseModel):
    concept: str
    answers: dict[str, int]  # question id -> chosen option index


@app.post("/evaluation/{sid}/answer")
def answer(sid: str, body: Answers):
    session: AdaptiveEvaluation | None = _state["sessions"].get(sid)
    if session is None:
        raise HTTPException(404, "unknown session")
    result = session.submit(body.concept, body.answers)
    result["next_questions"] = [q.as_public() for q in session.next_questions(body.concept)]
    return result


@app.get("/evaluation/{sid}")
def evaluation_report(sid: str):
    session = _state["sessions"].get(sid)
    if session is None:
        raise HTTPException(404, "unknown session")
    report = session.report()
    report["records"] = [_record_out(r) for r in session.records]  # feed back into /analyze
    return report
