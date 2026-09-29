"""End-to-end orchestration - Algorithm 1 of the paper.

Step 1  Data preparation        curriculum PDF -> text -> RAG vector DB      (CurriculumKB.ingest)
Step 2  Knowledge-gap analysis  OBE results -> EKT knowledge state -> gaps   (analyze)
Step 3  Remedial recommendation generate + cross-verify per weak concept     (recommend)
Step 4  MCQ generation          RAG questions, de-duplicated, cross-verified (QuestionGenerator)
Step 5  Personalized evaluation adaptive exam on the weak concepts; results
                                go back to Step 2 until mastery              (run_cycle)
"""
from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import store_dir
from .data import Record
from .evaluation import AdaptiveEvaluation, EvaluationRule
from .gap import ConceptGap, analyze_learner
from .kt import KnowledgeTracer
from .qgen import MCQ, QuestionGenerator
from .rag import CurriculumKB
from .remedial import RemedialRecommender, to_markdown

log = logging.getLogger(__name__)

AnswerFn = Callable[[MCQ], int]
StudyFn = Callable[[list[dict]], None]


@dataclass
class RoundResult:
    round: int
    knowledge_state: dict[str, float]
    gaps: list[str]
    recommendations: list[dict] = field(default_factory=list)
    evaluation: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"round": self.round, "knowledge_state": self.knowledge_state, "gaps": self.gaps,
                "recommendations": self.recommendations, "evaluation": self.evaluation}


class PersonalizedLearningSystem:
    def __init__(self, kb: CurriculumKB | None = None, tracer: KnowledgeTracer | None = None,
                 model_type: str = "ekt"):
        self.kb = kb or CurriculumKB.load()
        try:
            self.tracer = tracer or KnowledgeTracer.load(model_type)
        except FileNotFoundError:
            log.warning("No trained knowledge-tracing model found - using raw OBE scores for gap analysis.")
            self.tracer = None
        self.recommender = RemedialRecommender(self.kb)
        self.generator = QuestionGenerator(self.kb)

    @property
    def concept_names(self) -> list[str]:
        if self.tracer is not None:
            return self.tracer.concepts
        return [c.name for c in self.kb.concepts]

    # Step 2
    def analyze(self, records: list[Record]) -> list[ConceptGap]:
        return analyze_learner(records, self.tracer, self.concept_names)

    # Step 3
    def recommend(self, gaps: list[ConceptGap], round_no: int = 1) -> list[dict]:
        return [self.recommender.recommend(g, round_no) for g in gaps if g.is_gap]

    # Steps 4 + 5
    def evaluation(self, learner_id: str, concepts: list[str], rule: EvaluationRule | None = None,
                   seen: set[str] | None = None, seed: int | None = None) -> AdaptiveEvaluation:
        known = {c.name for c in self.kb.concepts}
        missing = [c for c in concepts if c not in known]
        if missing:
            raise KeyError(f"Concepts {missing} are not in concepts.json - make the curriculum concept "
                           "names match the concept names used in the OBE exam data.")
        return AdaptiveEvaluation(learner_id, concepts, self.generator, rule, seen, seed)

    def run_cycle(self, learner_id: str, records: list[Record], answer_fn: AnswerFn,
                  study_fn: StudyFn | None = None, rule: EvaluationRule | None = None,
                  max_rounds: int = 3, save: bool = True, seed: int = 0) -> list[RoundResult]:
        """Continuous learning loop: analyse -> recommend -> (study) -> evaluate -> re-trace."""
        history = list(records)
        seen: set[str] = set()
        rounds: list[RoundResult] = []
        for rnd in range(1, max_rounds + 1):
            gaps = self.analyze(history)
            state = {g.concept: round(g.mastery, 4) for g in gaps}
            weak = [g for g in gaps if g.is_gap]
            result = RoundResult(rnd, state, [g.concept for g in weak])
            rounds.append(result)
            if not weak:
                log.info("Round %d: no knowledge gaps left - target reached.", rnd)
                break
            result.recommendations = self.recommend(weak, rnd)
            if study_fn:
                study_fn(result.recommendations)
            session = self.evaluation(learner_id, [g.concept for g in weak], rule, seen, seed + rnd)
            session.run(answer_fn)
            seen |= session.seen
            result.evaluation = session.report()
            for r in session.records:  # feed evaluation interactions back into knowledge tracing
                r.order = len(history)
                r.exam = f"SelfAssessment-R{rnd}"
                history.append(r)
        else:
            gaps = self.analyze(history)
            rounds.append(RoundResult(max_rounds + 1, {g.concept: round(g.mastery, 4) for g in gaps},
                                      [g.concept for g in gaps if g.is_gap]))
        if save:
            self.save_rounds(learner_id, rounds)
        return rounds

    @staticmethod
    def save_rounds(learner_id: str, rounds: list[RoundResult]) -> Path:
        out = store_dir() / "learners" / learner_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "cycle.json").write_text(json.dumps([r.as_dict() for r in rounds], indent=2), encoding="utf-8")
        md = [f"# Learning cycle for {learner_id}"]
        for r in rounds:
            md.append(f"\n# Round {r.round}\n\nKnowledge state: " +
                      ", ".join(f"{c}: {v:.0%}" for c, v in r.knowledge_state.items()))
            md.append("\nGaps: " + (", ".join(r.gaps) or "none"))
            md += [to_markdown(rec) for rec in r.recommendations]
            for c, e in r.evaluation.get("concepts", {}).items():
                md.append(f"\n**Evaluation - {c}: {e['category']}** " +
                          " | ".join(f"{h['level']} {h['correct']}/{h['total']}" for h in e["history"]))
        (out / "cycle.md").write_text("\n\n".join(md), encoding="utf-8")
        return out


# --------------------------------------------------------------------------- simulated learner
def simulated_answerer(theta: dict[str, float], rng: random.Random) -> AnswerFn:
    """Answer MCQs like an IRT learner with ability ``theta`` (used by demo / experiments)."""
    from .data import _DIFF

    def answer(q: MCQ) -> int:
        t = theta.get(q.concept, 0.0)
        p = 1 / (1 + math.exp(-1.7 * (t - _DIFF.get(q.difficulty, 0.0))))
        p = 0.25 + 0.75 * p  # guessing floor for 4-option MCQs
        if rng.random() < p:
            return q.answer
        return rng.choice([i for i in range(len(q.options)) if i != q.answer])

    return answer


def simulated_study(theta: dict[str, float], rng: random.Random, gain: float = 0.8) -> StudyFn:
    """Stand-in for a learner studying the recommendations: raises ability on those concepts."""

    def study(recs: list[dict]) -> None:
        for rec in recs:
            theta[rec["concept"]] = theta.get(rec["concept"], 0.0) + max(rng.gauss(gain, 0.3), 0.0)

    return study
