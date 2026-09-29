"""Knowledge-gap analysis against OBE threshold / target (paper Sec. III-A.1, III-B).

* score <  threshold          -> Beginner
* threshold <= score < target -> Intermediate
* score >= target             -> Expert
A concept is a *knowledge gap* when the learner's mastery is below the target.
Mastery comes from the knowledge-tracing model when one is supplied, otherwise
from the raw OBE attainment percentage.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .config import get_config
from .data import Record, by_learner


def obe_levels() -> tuple[float, float]:
    cfg = get_config()["obe"]
    return float(cfg["threshold"]), float(cfg["target"])


def classify(score_pct: float, threshold: float | None = None, target: float | None = None) -> str:
    t, g = obe_levels()
    t = t if threshold is None else threshold
    g = g if target is None else target
    if score_pct < t:
        return "Beginner"
    if score_pct < g:
        return "Intermediate"
    return "Expert"


def concept_scores(records: list[Record]) -> dict[str, float]:
    """OBE attainment % per concept = marks obtained / max marks."""
    got, total = defaultdict(float), defaultdict(float)
    for r in records:
        got[r.concept] += r.marks
        total[r.concept] += r.max_marks
    return {c: 100.0 * got[c] / total[c] for c in total if total[c] > 0}


@dataclass
class ConceptGap:
    concept: str
    obe_score: float | None      # % from exam marks (None if not attempted)
    mastery: float               # 0-1, from EKT (or obe_score / 100)
    level: str
    is_gap: bool
    attempts: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"concept": self.concept, "obe_score": None if self.obe_score is None else round(self.obe_score, 2),
                "mastery": round(self.mastery, 4), "level": self.level, "is_gap": self.is_gap,
                "attempts": self.attempts}


def analyze_learner(records: list[Record], tracer=None, concepts: list[str] | None = None) -> list[ConceptGap]:
    _, target = obe_levels()
    scores = concept_scores(records)
    mastery = tracer.knowledge_state(records) if tracer is not None else {}
    names = concepts or list(dict.fromkeys(list(mastery) + list(scores)))
    out = []
    for c in names:
        m = mastery.get(c, scores.get(c, 50.0) / 100.0)
        mine = [r for r in records if r.concept == c]
        recent = [r for r in mine if r.exam == mine[-1].exam][-10:] if mine else []  # latest exam/session
        attempts = [{"question": r.question_text, "difficulty": r.difficulty, "bloom_level": r.bloom_level,
                     "correct": bool(r.attained())} for r in recent]
        out.append(ConceptGap(c, scores.get(c), m, classify(100 * m), 100 * m < target, attempts))
    return out


def knowledge_gap_matrix(records: list[Record]) -> dict[str, dict[str, float]]:
    """% of learners who attained (>= target) / did not attain each concept (Sec. IV-B.2)."""
    _, target = obe_levels()
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for rs in by_learner(records).values():
        for c, s in concept_scores(rs).items():
            counts[c][0 if s >= target else 1] += 1
    return {c: {"attained_pct": round(100 * a / (a + n), 2), "not_attained_pct": round(100 * n / (a + n), 2),
                "learners": a + n} for c, (a, n) in counts.items()}
