"""Concept-based adaptive evaluation (paper Sec. III-E, IV-A.4).

Per concept the learner goes Easy -> Medium -> Hard. A level is presented only
after the previous one is passed, using the instructor rule
"<questions per level> and <pass percentage>" (e.g. "5 and 80" for core courses,
"1 and 100" for optional ones). Category = highest level passed:

    none -> Not Competent, easy -> Beginner, medium -> Intermediate, hard -> Expert

Answers are converted to OBE ``Record`` rows so they can be fed straight back
into knowledge tracing (the continuous-learning loop, Sec. III-F).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

from .config import get_config
from .data import Record
from .qgen import LEVELS, MCQ, QuestionGenerator

CATEGORY = {0: "Not Competent", 1: "Beginner", 2: "Intermediate", 3: "Expert"}


@dataclass
class EvaluationRule:
    questions_per_level: int = 5
    pass_percentage: float = 80.0

    @classmethod
    def default(cls) -> "EvaluationRule":
        r = get_config()["evaluation"]["default_rule"]
        return cls(int(r["questions_per_level"]), float(r["pass_percentage"]))

    @classmethod
    def parse(cls, text: str) -> "EvaluationRule":
        """Parse the paper's '5 and 80' notation."""
        n, p = [x.strip() for x in text.lower().split("and")]
        return cls(int(n), float(p))

    @property
    def required_correct(self) -> int:
        return math.ceil(self.questions_per_level * self.pass_percentage / 100 - 1e-9)


@dataclass
class ConceptProgress:
    concept: str
    level_index: int = 0                 # index into LEVELS of the level being attempted
    passed_levels: int = 0
    finished: bool = False
    history: list[dict] = field(default_factory=list)

    @property
    def category(self) -> str:
        return CATEGORY[self.passed_levels]


class AdaptiveEvaluation:
    """One evaluation session for one learner over several concepts."""

    def __init__(self, learner_id: str, concepts: list[str], generator: QuestionGenerator,
                 rule: EvaluationRule | None = None, seen: set[str] | None = None, seed: int | None = None):
        self.learner_id = learner_id
        self.generator = generator
        self.rule = rule or EvaluationRule.default()
        self.progress = {c: ConceptProgress(c) for c in concepts}
        self.seen = set(seen or ())  # question ids already shown to this learner (fresh questions each round)
        self.records: list[Record] = []
        self.rng = random.Random(seed)
        self._pending: dict[str, MCQ] = {}

    # ---- question flow
    def next_questions(self, concept: str) -> list[MCQ]:
        """Questions for the current level of ``concept`` (empty when the concept is finished)."""
        prog = self.progress[concept]
        if prog.finished:
            return []
        level = LEVELS[prog.level_index]
        kb_concept = self.generator.kb.concept(concept)
        pool = self.generator.ensure_bank(kb_concept, self.rule.questions_per_level, [level], self.seen)[level]
        chosen = self.rng.sample(pool, min(self.rule.questions_per_level, len(pool)))
        for q in chosen:
            self._pending[q.id] = q
            self.seen.add(q.id)
        return chosen

    def submit(self, concept: str, answers: dict[str, int]) -> dict:
        """Grade answers {question_id: option_index} for the current level and advance."""
        prog = self.progress[concept]
        level = LEVELS[prog.level_index]
        results = []
        for qid, choice in answers.items():
            q = self._pending.pop(qid)
            correct = int(choice) == q.answer
            results.append({"id": qid, "correct": correct, "answer": q.answer, "chosen": int(choice),
                            "explanation": q.explanation})
            self.records.append(Record(self.learner_id, "SelfAssessment", len(self.records), q.id, q.question,
                                       concept, "", q.difficulty, q.bloom_level, 1.0, 1.0 if correct else 0.0))
        n_correct = sum(r["correct"] for r in results)
        pct = 100.0 * n_correct / max(len(results), 1)
        passed = len(results) > 0 and pct >= self.rule.pass_percentage
        prog.history.append({"level": level, "correct": n_correct, "total": len(results),
                             "percent": pct, "passed": passed})
        if passed:
            prog.passed_levels += 1
            prog.level_index += 1
            prog.finished = prog.level_index >= len(LEVELS)
        else:
            prog.finished = True
        return {"concept": concept, "level": level, "percent": pct, "passed": passed,
                "category": prog.category, "finished": prog.finished, "results": results}

    # ---- helpers
    def run(self, answer_fn: Callable[[MCQ], int]) -> dict[str, str]:
        """Run the whole session with ``answer_fn(question) -> chosen option``."""
        for concept in self.progress:
            while not self.progress[concept].finished:
                qs = self.next_questions(concept)
                if not qs:
                    self.progress[concept].finished = True
                    break
                self.submit(concept, {q.id: answer_fn(q) for q in qs})
        return self.summary()

    def summary(self) -> dict[str, str]:
        return {c: p.category for c, p in self.progress.items()}

    def report(self) -> dict:
        return {"learner_id": self.learner_id,
                "rule": {"questions_per_level": self.rule.questions_per_level,
                         "pass_percentage": self.rule.pass_percentage},
                "concepts": {c: {"category": p.category, "history": p.history} for c, p in self.progress.items()}}
