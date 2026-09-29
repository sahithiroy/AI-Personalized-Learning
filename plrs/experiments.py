"""Reproduce the paper's evaluation (Sec. V) on whatever data/models are configured.

* ``dataset_tables``   - Tables 1 & 2 (dataset statistics)
* ``kt_comparison``    - Table 5 / Fig. 4 (EKT vs DKT: loss, AUC, MAE per epoch)
* ``mcq_comparison``   - Tables 3 & 4 (generated vs teacher questions, per generator model)
* ``remedial_effect``  - Fig. 7 & Table 6 (competency levels + paired t-test, pre vs post)

NOTE: with the bundled synthetic data and the offline mock models these numbers
demonstrate that the pipeline works; they are *not* the paper's results.
"""
from __future__ import annotations

import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path

from .config import models_dir
from .data import QUESTION_BANK, dataset_summary, load_records, make_learners
from .evaluation import AdaptiveEvaluation, EvaluationRule
from .gap import knowledge_gap_matrix
from .kt import train_kt
from .llm import get_llm
from .metrics import paired_t_test, question_metrics
from .pipeline import simulated_answerer
from .qgen import LEVELS, QuestionGenerator, QuestionMemory
from .rag import CurriculumKB

log = logging.getLogger(__name__)


def dataset_tables(historical_csv: str, current_csv: str) -> dict:
    cur = load_records(current_csv)
    return {"table1_kt_training": dataset_summary(load_records(historical_csv)),
            "table2_recommendation": dataset_summary(cur),
            "knowledge_gap_matrix": knowledge_gap_matrix(cur)}


def kt_comparison(historical_csv: str, models: tuple[str, ...] = ("ekt", "dkt"), epochs: int | None = None) -> dict:
    records = load_records(historical_csv)
    out, rows = {}, []
    for m in models:
        _, hist = train_kt(records, m, epochs=epochs, save=True)
        out[m] = hist[-1]
        rows += [{"model": m, **h} for h in hist]
    path = models_dir() / "kt_history.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    _plot_kt(rows, models_dir() / "kt_curves.png")
    return {"final_epoch": out, "history_csv": str(path)}


def _plot_kt(rows: list[dict], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["auc", "loss", "mae"]):
        for m in sorted({r["model"] for r in rows}):
            rs = [r for r in rows if r["model"] == m]
            ax.plot([r["epoch"] for r in rs], [r[f"train_{metric}"] for r in rs], "--", label=f"{m.upper()} Train")
            ax.plot([r["epoch"] for r in rs], [r[f"test_{metric}"] for r in rs], label=f"{m.upper()} Test")
        ax.set_title(f"{metric.upper()} vs. Epoch")
        ax.set_xlabel("Epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def mcq_comparison(kb: CurriculumKB, generators: tuple[str, ...] = ("openai", "gemini", "deepseek"),
                   per_level: int = 2) -> dict:
    """Generate questions with each model and compare with the teacher-authored question bank."""
    results = {}
    for name in generators:
        mem_dir = models_dir() / f"mcq_{name}"  # isolated, fresh memory per generator model
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "question_bank.json").unlink(missing_ok=True)
        memory = QuestionMemory(directory=mem_dir)
        gen = QuestionGenerator(kb, memory)
        gen.primary = get_llm(name)
        per_concept = {}
        for concept in kb.concepts:
            qs = []
            for level in LEVELS:
                qs += [q.question for q in gen.generate(concept, level, per_level)]
            refs = [t for _, t in QUESTION_BANK.get(concept.name, [])]
            if qs and refs:
                per_concept[concept.name] = question_metrics(qs, refs)
        all_q = [q.question for q in memory.questions]
        all_ref = [t for v in QUESTION_BANK.values() for _, t in v]
        results[gen.primary.name] = {"overall": question_metrics(all_q, all_ref) if all_q else {},
                                     "per_concept": per_concept, "pipeline_stats": gen.stats}
    return results


def remedial_effect(kb: CurriculumKB, n_learners: int = 30, seed: int = 11,
                    rule: EvaluationRule | None = None) -> dict:
    """Simulated pre/post study: adaptive test -> remediation -> fresh adaptive test.

    The learning gain after studying a recommendation is *simulated* (ability +~0.8 logits on
    concepts that received a recommendation); swap in real post-test data for a real study.
    """
    from .gap import obe_levels
    from .pipeline import PersonalizedLearningSystem

    rng = random.Random(seed)
    rule = rule or EvaluationRule(questions_per_level=3, pass_percentage=67)
    system = PersonalizedLearningSystem(kb)
    mem_dir = models_dir() / "mcq_remedial_experiment"  # keep the real question bank untouched
    mem_dir.mkdir(parents=True, exist_ok=True)
    system.generator.memory = QuestionMemory(directory=mem_dir)
    concepts = [c.name for c in kb.concepts]
    learners = make_learners(n_learners, rng, "CSE23")
    pre_scores = {c: [] for c in concepts}
    post_scores = {c: [] for c in concepts}
    pre_cat, post_cat = {c: Counter() for c in concepts}, {c: Counter() for c in concepts}
    _, target = obe_levels()

    def fixed_test(theta, seen):
        """2 questions per level per concept, not adaptive - gives a comparable % score."""
        answer = simulated_answerer(theta, rng)
        scores = {}
        for c in concepts:
            bank = system.generator.ensure_bank(kb.concept(c), 2, LEVELS, seen)
            qs = [q for lvl in LEVELS for q in rng.sample(bank[lvl], min(2, len(bank[lvl])))]
            seen |= {q.id for q in qs}
            scores[c] = 100 * sum(answer(q) == q.answer for q in qs) / max(len(qs), 1)
        return scores

    for lr in learners:
        seen: set[str] = set()
        pre = fixed_test(lr.theta, seen)
        ev = AdaptiveEvaluation(lr.learner_id, concepts, system.generator, rule, seen, rng.random())
        for c, cat in ev.run(simulated_answerer(lr.theta, rng)).items():
            pre_cat[c][cat] += 1
        seen |= ev.seen
        for c in concepts:  # remediation for concepts below target, simulated study effect
            if pre[c] < target:
                lr.theta[c] += max(rng.gauss(0.8, 0.3), 0.0)
        post = fixed_test(lr.theta, seen)
        ev2 = AdaptiveEvaluation(lr.learner_id, concepts, system.generator, rule, seen, rng.random())
        for c, cat in ev2.run(simulated_answerer(lr.theta, rng)).items():
            post_cat[c][cat] += 1
        for c in concepts:
            pre_scores[c].append(pre[c])
            post_scores[c].append(post[c])

    def pct(counter):
        total = sum(counter.values()) or 1
        return {k: round(100 * counter[k] / total, 2) for k in ["Not Competent", "Beginner", "Intermediate", "Expert"]}

    ttests = {}
    for c in concepts:
        try:
            ttests[c] = paired_t_test(pre_scores[c], post_scores[c])
        except ValueError:
            pass
    return {"note": "simulated learners and simulated study effect",
            "competency_before": {c: pct(pre_cat[c]) for c in concepts},
            "competency_after": {c: pct(post_cat[c]) for c in concepts},
            "paired_t_test": ttests}


def run_all(historical_csv: str, current_csv: str, epochs: int | None = None, out: Path | None = None) -> dict:
    kb = CurriculumKB.load()
    report = {"datasets": dataset_tables(historical_csv, current_csv),
              "knowledge_tracing": kt_comparison(historical_csv, epochs=epochs),
              "mcq_generation": mcq_comparison(kb),
              "remedial_effect": remedial_effect(kb)}
    out = out or models_dir() / "experiments.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["saved_to"] = str(out)
    return report
