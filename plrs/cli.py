"""Command-line interface:  python -m plrs <command> [options]

Run ``python -m plrs --help`` or ``python -m plrs <command> --help`` for details.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from .data import SAMPLE_DIR

DEFAULT_CURRICULUM = str(SAMPLE_DIR / "curriculum_c_programming.txt")
DEFAULT_HIST = str(SAMPLE_DIR / "historical.csv")
DEFAULT_CURRENT = str(SAMPLE_DIR / "current_batch.csv")


def _print(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _learner_records(path: str, learner: str | None):
    from .data import by_learner, load_records

    groups = by_learner(load_records(path))
    if not groups:
        sys.exit(f"No records in {path}")
    learner = learner or sorted(groups)[0]
    if learner not in groups:
        sys.exit(f"Learner {learner!r} not found in {path}. Example ids: {sorted(groups)[:5]}")
    return learner, groups[learner]


# --------------------------------------------------------------------------- commands
def cmd_sample_data(a):
    from .data import generate_sample_data

    paths = generate_sample_data(a.out, n_historical=a.historical, n_current=a.current, seed=a.seed)
    _print({k: str(v) for k, v in paths.items()})


def cmd_ingest(a):
    from .llm import get_llm
    from .rag import CurriculumKB

    kb = CurriculumKB.ingest(a.paths)
    if not a.no_extract:
        kb.extract_concepts()
    path = kb.save()
    print(f"Indexed {len(kb.store)} chunks into {path}  (LLM: {get_llm().name})")
    print(f"Extracted {len(kb.concepts)} concepts -> {path / 'concepts.json'}  (review/edit this file)")
    for c in kb.concepts:
        print(f"  {c.id:4} {c.course_outcome:5} {c.name}")


def cmd_concepts(a):
    from .rag import CurriculumKB

    kb = CurriculumKB.load()
    for c in kb.concepts:
        print(f"{c.id:4} {c.course_outcome:5} {c.name}")
        for o in c.learning_objectives:
            print(f"        - {o}")


def cmd_query(a):
    from .rag import CurriculumKB

    for i, chunk in enumerate(CurriculumKB.load().retrieve(a.text, a.k), 1):
        print(f"--- [{i}]\n{chunk}\n")


def cmd_train_kt(a):
    from .data import load_records
    from .kt import train_kt

    records = load_records(a.data)
    models = ["ekt", "dkt"] if a.model == "all" else [a.model]
    summary = {}
    for m in models:
        tracer, hist = train_kt(records, m, epochs=a.epochs)
        last = hist[-1]
        summary[m] = {k: round(v, 4) if isinstance(v, float) else v for k, v in last.items()}
        print(f"[{m}] saved to {tracer.path()}")
    _print(summary)


def cmd_analyze(a):
    from .gap import analyze_learner
    from .kt import KnowledgeTracer

    learner, records = _learner_records(a.data, a.learner)
    tracer = None if a.no_kt else KnowledgeTracer.load(a.model)
    gaps = analyze_learner(records, tracer)
    if a.json:
        _print({"learner_id": learner, "concepts": [g.as_dict() for g in gaps]})
        return
    print(f"Learner {learner} - {len(records)} exam answers")
    print(f"{'Concept':55} {'OBE %':>7} {'Mastery':>8}  Level         Gap")
    for g in gaps:
        obe = "-" if g.obe_score is None else f"{g.obe_score:.1f}"
        print(f"{g.concept:55} {obe:>7} {g.mastery:8.2f}  {g.level:13} {'YES' if g.is_gap else ''}")


def cmd_gap_matrix(a):
    from .data import load_records
    from .gap import knowledge_gap_matrix

    _print(knowledge_gap_matrix(load_records(a.data)))


def cmd_recommend(a):
    from .gap import analyze_learner
    from .kt import KnowledgeTracer
    from .rag import CurriculumKB
    from .remedial import RemedialRecommender, to_markdown

    learner, records = _learner_records(a.data, a.learner)
    gaps = [g for g in analyze_learner(records, KnowledgeTracer.load(a.model)) if g.is_gap]
    if not gaps:
        print(f"Learner {learner} has no knowledge gaps (all concepts at or above target).")
        return
    rec = RemedialRecommender(CurriculumKB.load())
    md = [to_markdown(rec.recommend(g)) for g in gaps]
    text = f"# Remedial plan for {learner}\n\n" + "\n\n".join(md)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"Saved to {a.out}")
    else:
        print(text)


def cmd_generate_mcq(a):
    from .qgen import LEVELS, QuestionGenerator
    from .rag import CurriculumKB

    kb = CurriculumKB.load()
    gen = QuestionGenerator(kb)
    concepts = kb.concepts if a.concept == "all" else [kb.concept(a.concept)]
    levels = LEVELS if a.level == "all" else [a.level]
    for c in concepts:
        for lvl in levels:
            qs = gen.generate(c, lvl, a.n)
            print(f"\n=== {c.name} / {lvl}: {len(qs)} accepted")
            for q in qs[: a.show]:
                print(f"\nQ: {q.question}  [{q.bloom_level}]")
                for i, o in enumerate(q.options):
                    print(f"   {'*' if i == q.answer else ' '} {'ABCD'[i]}. {o}")
    print(f"\nPipeline stats: {gen.stats}")
    print(f"Question bank: {gen.memory.save()} ({len(gen.memory.questions)} questions)")


def cmd_evaluate(a):
    """Interactive adaptive evaluation in the terminal."""
    from .evaluation import EvaluationRule
    from .pipeline import PersonalizedLearningSystem

    system = PersonalizedLearningSystem()
    rule = EvaluationRule.parse(a.rule) if a.rule else EvaluationRule.default()
    concepts = a.concepts or [c.name for c in system.kb.concepts]
    session = system.evaluation(a.learner, concepts, rule)
    print(f"Rule: {rule.questions_per_level} questions per level, pass at {rule.pass_percentage:.0f}%\n")
    for c in concepts:
        while not session.progress[c].finished:
            qs = session.next_questions(c)
            if not qs:
                break
            print(f"\n##### {c} - {qs[0].difficulty.upper()}")
            answers = {}
            for i, q in enumerate(qs, 1):
                print(f"\n{i}. {q.question}")
                for j, o in enumerate(q.options):
                    print(f"   {'ABCD'[j]}. {o}")
                while True:
                    choice = input("Your answer (A-D): ").strip().upper()
                    if choice in ("A", "B", "C", "D")[: len(q.options)]:
                        break
                answers[q.id] = "ABCD".index(choice)
            res = session.submit(c, answers)
            print(f"-> {res['percent']:.0f}% {'PASSED' if res['passed'] else 'not passed'}; "
                  f"current category: {res['category']}")
    print("\nResult:")
    _print(session.summary())
    if a.save:
        from .data import save_records

        path = save_records(session.records, a.save)
        print(f"Answers saved as OBE records to {path} (append them to the learner's history to re-trace).")


def cmd_cycle(a):
    """Full Algorithm-1 loop for one learner (answers simulated unless --interactive)."""
    from .data import make_learners
    from .evaluation import EvaluationRule
    from .pipeline import PersonalizedLearningSystem, simulated_answerer, simulated_study
    from .remedial import to_markdown

    learner, records = _learner_records(a.data, a.learner)
    system = PersonalizedLearningSystem(model_type=a.model)
    rule = EvaluationRule.parse(a.rule) if a.rule else EvaluationRule(3, 67)
    rng = random.Random(a.seed)
    # Estimate a simulated learner from the OBE scores so answers are consistent with the exam data
    theta = make_learners(1, rng)[0].theta
    for g in system.analyze(records):
        if g.obe_score is not None:
            theta[g.concept] = (g.obe_score - 55) / 25
    rounds = system.run_cycle(learner, records, simulated_answerer(theta, rng),
                              simulated_study(theta, rng), rule=rule, max_rounds=a.rounds, seed=a.seed)
    for r in rounds:
        print(f"\n=========== Round {r.round}")
        print("Knowledge state: " + ", ".join(f"{c.split(',')[0]}={v:.2f}" for c, v in r.knowledge_state.items()))
        print("Gaps: " + (", ".join(r.gaps) or "none - target reached"))
        if a.verbose:
            for rec in r.recommendations:
                print("\n" + to_markdown(rec))
        for c, e in r.evaluation.get("concepts", {}).items():
            print(f"  evaluation {c}: {e['category']}  ("
                  + ", ".join(f"{h['level']} {h['correct']}/{h['total']}" for h in e["history"]) + ")")
    print(f"\nFull report: {system.save_rounds(learner, rounds)}")


def cmd_experiments(a):
    from . import experiments as ex
    from .rag import CurriculumKB

    which = set(a.which.split(","))
    report = {}
    if "data" in which:
        report["datasets"] = ex.dataset_tables(a.historical, a.current)
    if "kt" in which:
        report["knowledge_tracing"] = ex.kt_comparison(a.historical, epochs=a.epochs)
    if "mcq" in which:
        report["mcq_generation"] = ex.mcq_comparison(CurriculumKB.load())
    if "remedial" in which:
        report["remedial_effect"] = ex.remedial_effect(CurriculumKB.load(), n_learners=a.learners)
    from .config import models_dir

    out = models_dir() / "experiments.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print(report)
    print(f"\nSaved to {out}")


def cmd_demo(a):
    """Everything end-to-end on the bundled sample course."""
    print("== 1/5 Generating synthetic OBE data")
    cmd_sample_data(argparse.Namespace(out=str(SAMPLE_DIR), historical=a.historical, current=262, seed=7))
    print("\n== 2/5 Ingesting curriculum + extracting concepts (RAG)")
    cmd_ingest(argparse.Namespace(paths=[DEFAULT_CURRICULUM], no_extract=False))
    print("\n== 3/5 Training EKT knowledge-tracing model")
    cmd_train_kt(argparse.Namespace(data=DEFAULT_HIST, model="ekt", epochs=a.epochs))
    print("\n== 4/5 Knowledge-gap analysis for one learner")
    cmd_analyze(argparse.Namespace(data=DEFAULT_CURRENT, learner=a.learner, model="ekt", no_kt=False, json=False))
    print("\n== 5/5 Personalized learning cycle (remedial -> MCQs -> adaptive evaluation -> re-trace)")
    cmd_cycle(argparse.Namespace(data=DEFAULT_CURRENT, learner=a.learner, model="ekt", rule=None,
                                 rounds=3, seed=1, verbose=True))


def cmd_serve(a):
    import os

    import uvicorn

    os.environ["PLRS_URL"] = f"http://{a.host}:{a.port}"
    uvicorn.run("plrs.api:app", host=a.host, port=a.port, reload=False)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m plrs", description="AI-driven personalized learning & remedial recommendation")
    p.add_argument("-v", "--log-level", default="WARNING", help="DEBUG, INFO, WARNING ...")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sample-data", help="generate synthetic OBE exam data")
    s.add_argument("--out", default=str(SAMPLE_DIR))
    s.add_argument("--historical", type=int, default=600, help="learners in historical (KT training) data")
    s.add_argument("--current", type=int, default=262, help="learners in the current batch")
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_sample_data)

    s = sub.add_parser("ingest", help="index curriculum PDF/TXT into the vector DB and extract concepts")
    s.add_argument("paths", nargs="+")
    s.add_argument("--no-extract", action="store_true", help="skip LLM concept extraction (keep concepts.json)")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("concepts", help="list extracted concepts")
    s.set_defaults(func=cmd_concepts)

    s = sub.add_parser("query", help="semantic search over the curriculum")
    s.add_argument("text")
    s.add_argument("-k", type=int, default=3)
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("train-kt", help="train the knowledge-tracing model")
    s.add_argument("--data", default=DEFAULT_HIST)
    s.add_argument("--model", choices=["ekt", "dkt", "all"], default="ekt")
    s.add_argument("--epochs", type=int)
    s.set_defaults(func=cmd_train_kt)

    for name, func, hlp in [("analyze", cmd_analyze, "knowledge-gap analysis for a learner"),
                            ("recommend", cmd_recommend, "remedial recommendations for a learner")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--data", default=DEFAULT_CURRENT)
        s.add_argument("--learner", help="learner id (default: first in file)")
        s.add_argument("--model", default="ekt")
        if name == "analyze":
            s.add_argument("--no-kt", action="store_true", help="use raw OBE scores instead of EKT")
            s.add_argument("--json", action="store_true")
        else:
            s.add_argument("--out", help="write markdown to this file")
        s.set_defaults(func=func)

    s = sub.add_parser("gap-matrix", help="%% of learners attaining each concept")
    s.add_argument("--data", default=DEFAULT_CURRENT)
    s.set_defaults(func=cmd_gap_matrix)

    s = sub.add_parser("generate-mcq", help="generate, de-duplicate and cross-verify MCQs")
    s.add_argument("--concept", default="all", help="concept name or id, or 'all'")
    s.add_argument("--level", choices=["easy", "medium", "hard", "all"], default="all")
    s.add_argument("--n", type=int, default=5, help="required questions (generator makes ~3x)")
    s.add_argument("--show", type=int, default=2, help="questions to print per concept/level")
    s.set_defaults(func=cmd_generate_mcq)

    s = sub.add_parser("evaluate", help="take an adaptive evaluation interactively")
    s.add_argument("--learner", default="me")
    s.add_argument("--concepts", nargs="*")
    s.add_argument("--rule", help="instructor rule, e.g. \"5 and 80\" or \"1 and 100\"")
    s.add_argument("--save", help="save answers as OBE CSV records")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("cycle", help="full personalized-learning loop for one learner (simulated answers)")
    s.add_argument("--data", default=DEFAULT_CURRENT)
    s.add_argument("--learner")
    s.add_argument("--model", default="ekt")
    s.add_argument("--rule", help="e.g. \"3 and 67\"")
    s.add_argument("--rounds", type=int, default=3)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--verbose", action="store_true", help="print the recommendations")
    s.set_defaults(func=cmd_cycle)

    s = sub.add_parser("experiments", help="reproduce the paper's evaluation tables")
    s.add_argument("--which", default="data,kt,mcq,remedial")
    s.add_argument("--historical", default=DEFAULT_HIST)
    s.add_argument("--current", default=DEFAULT_CURRENT)
    s.add_argument("--epochs", type=int)
    s.add_argument("--learners", type=int, default=30)
    s.set_defaults(func=cmd_experiments)

    s = sub.add_parser("demo", help="run everything end-to-end on the sample course")
    s.add_argument("--learner", default="CSE230001")
    s.add_argument("--epochs", type=int, default=12)
    s.add_argument("--historical", type=int, default=600)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="start the web frontend + REST API (needs fastapi)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
