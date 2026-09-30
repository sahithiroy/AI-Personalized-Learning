"""Adaptive MCQ generation with RAG (paper Sec. III-D, Appendix A).

For a concept and difficulty level:
1. retrieve curriculum context from the vector DB;
2. ask the primary model for ~3x the required number of MCQs (Bloom-aligned);
3. refine each question with the same model (well-formed, grammatical);
4. drop semantic duplicates (cosine > 0.85) against the global question memory;
5. cross-verify with the secondary models (Gemini, DeepSeek) for BTL / difficulty fit;
6. store accepted questions in the global memory and the question bank.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import get_config, store_dir
from .embeddings import cosine_matrix, get_embedder
from .llm import BLOOM_BY_LEVEL, get_llm, get_verifiers
from .rag import Concept, CurriculumKB

log = logging.getLogger(__name__)
LEVELS = ["easy", "medium", "hard"]


@dataclass
class MCQ:
    question: str
    options: list[str]
    answer: int
    concept: str
    difficulty: str
    bloom_level: str
    explanation: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    verification: dict = field(default_factory=dict)

    @property
    def text_for_similarity(self) -> str:
        return f"{self.question} {self.options[self.answer]}"

    def as_public(self) -> dict:
        """Version safe to send to a learner (no answer)."""
        return {"id": self.id, "question": self.question, "options": self.options,
                "concept": self.concept, "difficulty": self.difficulty, "bloom_level": self.bloom_level}


def _normalise(raw: dict, concept: str, level: str) -> MCQ | None:
    try:
        options = [str(o).strip() for o in raw["options"]][:4]
        ans = raw.get("answer", raw.get("correct", 0))
        if isinstance(ans, str):
            a = ans.strip()
            if len(a) == 1 and a.upper() in "ABCD":
                ans = "ABCD".index(a.upper())
            elif a.isdigit():
                ans = int(a)
            else:
                ans = options.index(a)
        ans = int(ans)
        if len(options) < 2 or not 0 <= ans < len(options):
            return None
        return MCQ(question=str(raw["question"]).strip(), options=options, answer=ans, concept=concept,
                   difficulty=level, bloom_level=raw.get("bloom_level") or BLOOM_BY_LEVEL[level][0],
                   explanation=raw.get("explanation", ""))
    except (KeyError, ValueError, TypeError):
        return None


# --------------------------------------------------------------------------- global memory
class QuestionMemory:
    """All accepted questions as semantic vectors, persisted across runs (Appendix A)."""

    def __init__(self, directory: Path | None = None):
        self.dir = directory or store_dir()
        self.questions: list[MCQ] = []
        self.vectors = np.zeros((0, get_embedder().encode(["probe"]).shape[1]), dtype=np.float32)
        path = self.dir / "question_bank.json"
        if path.exists():
            self.questions = [MCQ(**q) for q in json.loads(path.read_text(encoding="utf-8"))]
            if self.questions:
                self.vectors = get_embedder().encode([q.text_for_similarity for q in self.questions])

    def is_duplicate(self, vec: np.ndarray, threshold: float) -> bool:
        if len(self.vectors) == 0:
            return False
        return float(cosine_matrix(vec.reshape(1, -1), self.vectors).max()) > threshold

    def add(self, q: MCQ, vec: np.ndarray) -> None:
        self.questions.append(q)
        self.vectors = np.vstack([self.vectors, vec.reshape(1, -1)])

    def save(self) -> Path:
        path = self.dir / "question_bank.json"
        path.write_text(json.dumps([asdict(q) for q in self.questions], indent=1), encoding="utf-8")
        return path

    def select(self, concept: str, level: str, exclude: set[str] | None = None) -> list[MCQ]:
        exclude = exclude or set()
        return [q for q in self.questions if q.concept == concept and q.difficulty == level and q.id not in exclude]


# --------------------------------------------------------------------------- generator
class QuestionGenerator:
    def __init__(self, kb: CurriculumKB, memory: QuestionMemory | None = None):
        cfg = get_config()["mcq"]
        self.kb = kb
        self.memory = memory or QuestionMemory()
        self.multiplier = cfg["generation_multiplier"]
        self.dedup_threshold = cfg["dedup_threshold"]
        self.min_score = cfg["verification_min_score"]
        self.primary = get_llm()
        self.verifiers = get_verifiers()
        self.stats = {"generated": 0, "invalid": 0, "duplicates": 0, "rejected": 0, "accepted": 0}

    def _prompt(self, concept: Concept, level: str, n: int, context: list[str]) -> str:
        blooms = " / ".join(BLOOM_BY_LEVEL[level])
        ctx = "\n---\n".join(context)
        return (
            f"Using ONLY the course material below, write {n} distinct multiple-choice questions on the "
            f"knowledge concept \"{concept.name}\".\nDifficulty: {level}. Bloom's Taxonomy level(s): {blooms}.\n"
            f"Learning objectives: {'; '.join(concept.learning_objectives)}\n"
            "Each question must have exactly 4 plausible options (distractors should reflect common "
            "misconceptions) and one correct answer. Vary the question structure.\n"
            "Return JSON: {\"questions\": [{\"question\": str, \"options\": [str, str, str, str], "
            "\"answer\": <index 0-3>, \"bloom_level\": str, \"explanation\": str}]}\n\n"
            f"COURSE MATERIAL:\n{ctx}"
        )

    def _refine(self, q: MCQ) -> MCQ:
        if not get_config()["mcq"].get("refine", True):  # saves one LLM call per question (free tiers)
            return q
        prompt = ("Check this multiple-choice question. Fix grammar, ambiguity and formatting without "
                  "changing its meaning or the correct answer. Return the same JSON object.\n"
                  + json.dumps(asdict(q)))
        try:
            raw = self.primary.generate_json(prompt, task="refine_mcq", payload={"question": asdict(q)})
            fixed = _normalise(raw, q.concept, q.difficulty)
            if fixed:
                fixed.id = q.id
                return fixed
        except Exception as exc:
            log.debug("refinement failed: %s", exc)
        return q

    def _verify(self, q: MCQ, concept: Concept) -> tuple[bool, dict]:
        prompt = (
            f"You are reviewing an assessment item for the concept \"{concept.name}\" at difficulty "
            f"\"{q.difficulty}\" (Bloom's level: {', '.join(BLOOM_BY_LEVEL[q.difficulty])}).\n"
            "Rate from 1 (unusable) to 5 (excellent) how well it matches the concept, difficulty and "
            "Bloom's level, and whether the marked answer is correct and unambiguous.\n"
            "Return JSON: {\"score\": int, \"bloom_level_ok\": bool, \"difficulty_ok\": bool, \"feedback\": str}\n\n"
            + json.dumps({"question": q.question, "options": q.options, "answer": q.answer,
                          "bloom_level": q.bloom_level})
        )
        reviews = {}
        for v in self.verifiers:
            try:
                reviews[v.name] = v.generate_json(prompt, task="verify_mcq", payload={"question": asdict(q)})
            except Exception as exc:
                log.warning("verifier %s failed: %s", v.name, exc)
        scores = [float(r.get("score", 0)) for r in reviews.values()]
        avg = sum(scores) / len(scores) if scores else 0.0
        return (avg >= self.min_score if scores else True), {"avg_score": avg, "reviews": reviews}

    def generate(self, concept: Concept, level: str, n_required: int, seed: int = 0) -> list[MCQ]:
        embedder = get_embedder()
        context = self.kb.context_for(concept)
        n = max(n_required * self.multiplier, n_required)
        raw = self.primary.generate_json(
            self._prompt(concept, level, n, context), task="generate_mcq",
            payload={"concept": concept.name, "level": level, "n": n, "context": context,
                     "other_context": self.kb.other_context(concept), "seed": seed},
        )
        items = raw.get("questions", []) if isinstance(raw, dict) else raw
        accepted = []
        for item in items:
            self.stats["generated"] += 1
            q = _normalise(item, concept.name, level)
            if q is None:
                self.stats["invalid"] += 1
                continue
            q = self._refine(q)
            vec = embedder.encode([q.text_for_similarity])[0]
            if self.memory.is_duplicate(vec, self.dedup_threshold):
                self.stats["duplicates"] += 1
                continue
            ok, report = self._verify(q, concept)
            q.verification = report
            if not ok:
                self.stats["rejected"] += 1
                continue
            self.memory.add(q, vec)
            accepted.append(q)
            self.stats["accepted"] += 1
        self.memory.save()
        return accepted

    def ensure_bank(self, concept: Concept, per_level: int, levels: list[str] = LEVELS,
                    exclude: set[str] | None = None, max_attempts: int = 3) -> dict[str, list[MCQ]]:
        """Make sure at least ``per_level`` unused questions exist for each level."""
        bank = {}
        for level in levels:
            attempt = 0
            while len(self.memory.select(concept.name, level, exclude)) < per_level and attempt < max_attempts:
                self.generate(concept, level, per_level, seed=len(self.memory.questions) + attempt)
                attempt += 1
            bank[level] = self.memory.select(concept.name, level, exclude)
        return bank
