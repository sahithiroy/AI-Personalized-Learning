import math
import random

import numpy as np
import pytest

from plrs.data import Record, dataset_summary, load_records
from plrs.evaluation import AdaptiveEvaluation, EvaluationRule
from plrs.gap import analyze_learner, classify, concept_scores, knowledge_gap_matrix
from plrs.kt import KnowledgeTracer, auc_score
from plrs.metrics import (diversity_score, flesch_kincaid_grade, paired_t_test, question_metrics,
                          relevance_from_similarity)
from plrs.qgen import QuestionGenerator
from plrs.remedial import SECTIONS, RemedialRecommender, to_markdown


def _rec(concept, marks, max_marks=10, qid="Q1", exam="E"):
    return Record("L1", exam, 0, qid, f"question {qid}", concept, "CO1", "easy", "Understand", max_marks, marks)


# ---------------------------------------------------------------- OBE gap analysis
def test_classify_thresholds():
    assert classify(49.9) == "Beginner"
    assert classify(50) == "Intermediate"
    assert classify(69.9) == "Intermediate"
    assert classify(70) == "Expert"


def test_concept_scores_and_gaps():
    recs = [_rec("A", 9), _rec("A", 7), _rec("B", 2), _rec("B", 4)]
    assert concept_scores(recs) == {"A": 80.0, "B": 30.0}
    gaps = {g.concept: g for g in analyze_learner(recs)}
    assert not gaps["A"].is_gap and gaps["A"].level == "Expert"
    assert gaps["B"].is_gap and gaps["B"].level == "Beginner"
    assert len(gaps["B"].attempts) == 2


def test_gap_matrix():
    m = knowledge_gap_matrix([_rec("A", 9), _rec("B", 1)])
    assert m["A"]["attained_pct"] == 100 and m["B"]["not_attained_pct"] == 100


def test_sample_data_shape(sample_data):
    s = dataset_summary(load_records(sample_data["current"]))
    assert s["Concepts"] == 5 and s["Unique Questions"] == 16 and s["Learners"] == 20


# ---------------------------------------------------------------- knowledge tracing
def test_auc_score():
    assert auc_score(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auc_score(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_tracer_knowledge_state(tracer, sample_data):
    recs = load_records(sample_data["current"])
    learner = [r for r in recs if r.learner_id == recs[0].learner_id]
    state = tracer.knowledge_state(learner)
    assert set(state) == set(tracer.concepts)
    assert all(0 <= v <= 1 for v in state.values())
    steps, states = tracer.trace(learner)
    assert states.shape == (len(steps), len(tracer.concepts))


def test_tracer_save_load(tracer, sample_data):
    loaded = KnowledgeTracer.load("ekt")
    recs = load_records(sample_data["current"])[:9]
    a, b = tracer.knowledge_state(recs), loaded.knowledge_state(recs)
    assert all(math.isclose(a[c], b[c], rel_tol=1e-5) for c in a)


# ---------------------------------------------------------------- MCQ generation
def test_generate_dedup_and_verify(kb):
    gen = QuestionGenerator(kb)
    concept = kb.concept("C2")
    first = gen.generate(concept, "easy", 3, seed=0)
    assert first, gen.stats
    assert all(len(q.options) == 4 and q.verification["avg_score"] >= 3 for q in first)
    again = gen.generate(concept, "easy", 3, seed=0)  # identical batch -> all duplicates
    assert again == []
    assert gen.stats["duplicates"] >= len(first)


# ---------------------------------------------------------------- remedial
def test_remedial_has_five_sections(kb):
    gap = analyze_learner([_rec("Algorithmic Problem Solving and Array Manipulation", 2)])[0]
    rec = RemedialRecommender(kb).recommend(gap)
    for section in SECTIONS:
        assert rec[section], section
    md = to_markdown(rec)
    assert "### 5. Concept Gap Rationale" in md and "20%" in md


# ---------------------------------------------------------------- adaptive evaluation
def test_rule_parse():
    r = EvaluationRule.parse("5 and 80")
    assert (r.questions_per_level, r.pass_percentage, r.required_correct) == (5, 80.0, 4)
    assert EvaluationRule.parse("1 and 100").required_correct == 1


@pytest.mark.parametrize("always_right,expected", [(True, "Expert"), (False, "Not Competent")])
def test_adaptive_evaluation(kb, always_right, expected):
    gen = QuestionGenerator(kb)
    concept = "Fundamentals and System Architecture"
    session = AdaptiveEvaluation("L9", [concept], gen, EvaluationRule(2, 100), seed=1)
    answer = (lambda q: q.answer) if always_right else (lambda q: (q.answer + 1) % 4)
    assert session.run(answer)[concept] == expected
    assert len(session.records) == (6 if always_right else 2)
    assert all(r.concept == concept for r in session.records)


# ---------------------------------------------------------------- metrics
def test_paired_t_test_matches_reference():
    pre = [45, 50, 38, 60, 41, 52, 47, 39]
    post = [60, 66, 50, 70, 55, 61, 64, 49]
    res = paired_t_test(pre, post)
    assert res["t_statistic"] < 0 and res["p_value"] < 0.001
    assert res["mean_diff"] == pytest.approx(np.mean(post) - np.mean(pre))
    # textbook check: t=2.0 with df=10 -> two-sided p = 0.0734
    from plrs.metrics import _betainc
    assert _betainc(5, 0.5, 10 / 14) == pytest.approx(0.07339, abs=1e-4)


def test_text_metrics():
    assert flesch_kincaid_grade("The cat sat on the mat.") < flesch_kincaid_grade(
        "Comprehensive institutional documentation necessitates considerable organisational deliberation.")
    same = ["what is a pointer in c"] * 4
    varied = ["what is a pointer", "explain heap allocation", "describe binary search", "why use strcmp"]
    assert diversity_score(same) > diversity_score(varied)
    assert [relevance_from_similarity(s) for s in (0.95, 0.85, 0.75, 0.65, 0.3)] == [5, 4, 3, 2, 1]
    m = question_metrics(varied, varied)
    assert m["semantic_similarity"] == pytest.approx(1.0, abs=1e-4) and m["relevance_1_5"] == 5


# ---------------------------------------------------------------- full loop
def test_run_cycle(kb, tracer, sample_data, tmp_path):
    from plrs.pipeline import PersonalizedLearningSystem, simulated_answerer, simulated_study

    system = PersonalizedLearningSystem(kb, tracer)
    recs = load_records(sample_data["current"])
    learner = recs[0].learner_id
    mine = [r for r in recs if r.learner_id == learner]
    rng = random.Random(0)
    theta = {c: 0.0 for c in system.concept_names}
    rounds = system.run_cycle(learner, mine, simulated_answerer(theta, rng), simulated_study(theta, rng),
                              rule=EvaluationRule(2, 50), max_rounds=2)
    assert rounds and rounds[0].knowledge_state
    if rounds[0].gaps:
        assert rounds[0].recommendations and rounds[0].evaluation["concepts"]
