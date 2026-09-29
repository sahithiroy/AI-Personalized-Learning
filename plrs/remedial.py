"""Remedial recommendation (paper Sec. III-C).

For every weak concept the primary model writes a structured recommendation with
five sections - Learning Objectives, Recommended Topics to Revise, Remedial
Explanation, Practice Activities, Concept Gap Rationale - grounded in the
curriculum (RAG) and in the questions the learner attempted. Secondary models
cross-verify it; low-scoring output is regenerated with the reviewers' feedback.
"""
from __future__ import annotations

import json
import logging

from .config import get_config
from .gap import ConceptGap
from .llm import get_llm, get_verifiers
from .rag import CurriculumKB

log = logging.getLogger(__name__)

SECTIONS = ["learning_objectives", "topics_to_revise", "remedial_explanation",
            "practice_activities", "concept_gap_rationale"]


class RemedialRecommender:
    def __init__(self, kb: CurriculumKB | None = None):
        cfg = get_config()["remedial"]
        self.kb = kb
        self.max_rounds = cfg["max_refinement_rounds"]
        self.min_score = cfg["verification_min_score"]
        self.primary = get_llm()
        self.verifiers = get_verifiers()

    def _context(self, concept: str) -> list[str]:
        if self.kb is None:
            return []
        try:
            return self.kb.context_for(self.kb.concept(concept))
        except KeyError:
            return self.kb.retrieve(concept)

    def _prompt(self, gap: ConceptGap, context: list[str], round_no: int, feedback: str) -> str:
        attempts = "\n".join(
            f"- [{'correct' if a['correct'] else 'WRONG'}] ({a['difficulty']}, {a['bloom_level']}) {a['question']}"
            for a in gap.attempts) or "- (no recent attempts recorded)"
        extra = ""
        if round_no > 1:
            extra = ("\nThe learner has NOT yet reached the target after the previous recommendation. "
                     "Give a more detailed explanation, simpler examples and additional practice.")
        if feedback:
            extra += f"\nReviewer feedback on the previous draft: {feedback}"
        return (
            f"Generate a structured remedial recommendation for the concept \"{gap.concept}\" for a learner "
            f"who demonstrates {gap.mastery:.0%} concept mastery. The response should include five sections: "
            "Learning Objectives, Recommended Topics to Revise, Remedial Explanation, Two Practice Activities "
            "and Concept Gap Rationale, which will describe the specific weaknesses detected by the "
            f"knowledge-tracing model and the reasoning behind the generated remedial content.{extra}\n\n"
            f"Questions attempted in the most recent evaluation:\n{attempts}\n\n"
            "Course material:\n" + "\n---\n".join(context) + "\n\n"
            "Return JSON: {\"concept\": str, \"learning_objectives\": [str], \"topics_to_revise\": "
            "[{\"topic\": str, \"points\": [str]}], \"remedial_explanation\": str, \"practice_activities\": "
            "[{\"title\": str, \"description\": str, \"input\": str, \"expected_output\": str}], "
            "\"concept_gap_rationale\": str}"
        )

    def _verify(self, rec: dict, gap: ConceptGap) -> dict:
        prompt = (
            f"Review this remedial recommendation for the concept \"{gap.concept}\" (learner mastery "
            f"{gap.mastery:.0%}). Check it is accurate, clearly written, pitched at the right Bloom's level "
            "and difficulty, and contains all five sections. Return JSON: {\"score\": 1-5, "
            "\"missing_sections\": [str], \"feedback\": str}\n\n" + json.dumps(rec)
        )
        reviews = {}
        for v in self.verifiers:
            try:
                reviews[v.name] = v.generate_json(prompt, task="verify_remedial", payload={"recommendation": rec})
            except Exception as exc:
                log.warning("verifier %s failed: %s", v.name, exc)
        scores = [float(r.get("score", 0)) for r in reviews.values()]
        return {"avg_score": sum(scores) / len(scores) if scores else None, "reviews": reviews}

    def recommend(self, gap: ConceptGap, round_no: int = 1) -> dict:
        context = self._context(gap.concept)
        other = [t for t in self.kb.store.texts if t not in context] if self.kb else []
        feedback, rec, report = "", {}, {}
        for attempt in range(1, self.max_rounds + 2):
            payload = {"concept": gap.concept, "mastery": gap.mastery, "attempts": gap.attempts,
                       "context": context, "other_context": other, "round": round_no}
            rec = self.primary.generate_json(self._prompt(gap, context, round_no, feedback),
                                             task="remedial", payload=payload)
            report = self._verify(rec, gap)
            if report["avg_score"] is None or report["avg_score"] >= self.min_score:
                break
            feedback = " ".join(str(r.get("feedback", "")) for r in report["reviews"].values())
        rec["concept"] = gap.concept
        rec["mastery"] = round(gap.mastery, 4)
        rec["round"] = round_no
        rec["verification"] = report
        return rec


def to_markdown(rec: dict) -> str:
    """Render a recommendation the way Figure 2 of the paper lays it out."""
    lines = [f"## Remedial Recommendation for Concept: {rec['concept']}",
             f"_Current mastery: {rec.get('mastery', 0):.0%} (round {rec.get('round', 1)})_", "",
             "### 1. Learning Objectives"]
    lines += [f"- {o}" for o in rec.get("learning_objectives", [])]
    lines += ["", "### 2. Recommended Topics to Revise"]
    for i, t in enumerate(rec.get("topics_to_revise", []), 1):
        if isinstance(t, dict):
            lines.append(f"{i}. **{t.get('topic', '')}**")
            lines += [f"   - {p}" for p in t.get("points", [])]
        else:
            lines.append(f"{i}. {t}")
    lines += ["", "### 3. Remedial Explanation", str(rec.get("remedial_explanation", "")), "",
              "### 4. Practice Activities"]
    for a in rec.get("practice_activities", []):
        if isinstance(a, dict):
            lines.append(f"- **{a.get('title', '')}**: {a.get('description', '')}")
            if a.get("input") or a.get("expected_output"):
                lines.append(f"  - Input: `{a.get('input', '')}` -> Expected output: `{a.get('expected_output', '')}`")
        else:
            lines.append(f"- {a}")
    lines += ["", "### 5. Concept Gap Rationale", str(rec.get("concept_gap_rationale", ""))]
    ver = rec.get("verification", {})
    if ver.get("avg_score") is not None:
        lines += ["", f"_Cross-verified by {', '.join(ver.get('reviews', {}))}: avg score {ver['avg_score']:.1f}/5_"]
    return "\n".join(lines)
