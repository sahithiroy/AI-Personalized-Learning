"""Additional experiments for the research paper (all numbers are computed here, nothing is hand-entered).

1. KT ablation over 3 seeds: DKT, EKT (full), EKT w/o attention, EKT w/o mastery loss, EKT + exercise dropout
2. Mastery calibration (mastery-head AUC) and responsiveness (change after 5 correct / 5 wrong answers)
3. Question-ordering artifact: same simulated learners, exams ordered by concept+difficulty vs shuffled
4. Retrieval purity of three chunking strategies
5. Closed-loop simulation over 40 learners, with and without remediation
"""
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
sys.path.insert(0, str(PROJECT))

from plrs.config import get_config, models_dir  # noqa: E402
from plrs.data import (C_CONCEPTS, _MARKS, Record, _marks, by_learner, load_records, make_learners,  # noqa: E402
                       question_bank)
from plrs.kt import auc_score, train_kt  # noqa: E402

CFG = get_config()["ekt"]
BASE = copy.deepcopy(CFG)
SEEDS = [42, 7, 123]
BANK = question_bank()
BY_C = {c: [q for q in BANK if q["concept"] == c] for c in C_CONCEPTS}
results = {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def set_cfg(**kw):
    CFG.clear()
    CFG.update(BASE)
    CFG.update(kw)


def probe(tracer, records, n_learners=120):
    """Mastery-head AUC on held-out learners, and response of mastery to 5 extra correct / wrong answers."""
    groups = by_learner(records)
    test = [groups[i] for i in tracer.test_ids]
    enc, model = tracer.encoder, tracer.model
    model.eval()
    ys, ps = [], []
    for rs in test:
        c, ex, r, _ = enc.batch([enc.steps(rs)], CFG["max_seq_len"])
        with torch.no_grad():
            _, m = model(c, ex, r)
        prob = torch.sigmoid(m[0, :-1]).gather(-1, c[0, 1:].unsqueeze(-1)).squeeze(-1)
        ys.append(r[0, 1:].numpy())
        ps.append(prob.numpy())
    m_auc = auc_score(np.concatenate(ys), np.concatenate(ps))
    dpos, dneg = [], []
    for rs in test[:n_learners]:
        base = tracer.knowledge_state(rs)
        for k in C_CONCEPTS:
            qs = BY_C[k]
            for correct, store in ((True, dpos), (False, dneg)):
                extra = []
                for j in range(5):
                    q = qs[j % len(qs)]
                    mm = _MARKS[q["difficulty"]]
                    extra.append(Record(rs[0].learner_id, "probe", 10_000 + j, q["question_id"], q["question_text"],
                                        k, "", q["difficulty"], "", mm, mm if correct else 0.0))
                store.append(tracer.knowledge_state(rs + extra)[k] - base[k])
    dpos, dneg = np.array(dpos), np.array(dneg)
    return {"mastery_auc": float(m_auc), "delta_after_correct": float(dpos.mean()),
            "share_up_after_correct": float((dpos > 0).mean()), "delta_after_wrong": float(dneg.mean()),
            "share_down_after_wrong": float((dneg < 0).mean())}


def run_variant(name, records, model_type="ekt", **cfg):
    rows = []
    for seed in SEEDS:
        set_cfg(**cfg)
        tracer, hist = train_kt(records, model_type, seed=seed, save=False)
        last = hist[-1]
        row = {k: last[k] for k in ("test_auc", "test_accuracy", "test_f1", "test_mae", "test_loss", "train_auc")}
        row.update(probe(tracer, records))
        rows.append(row)
        log(name, "seed", seed, {k: round(v, 3) for k, v in row.items()})
    keys = rows[0].keys()
    return {"per_seed": rows, "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
            "std": {k: float(np.std([r[k] for r in rows], ddof=1)) for k in keys}}


def ordered_history(seed=7, n=600):
    """Same generator as the sample data, but each exam presents questions grouped by concept, easy to hard."""
    rng = random.Random(seed)
    diff = {"easy": -1.0, "medium": 0.0, "hard": 1.0}
    recs = []
    for lr in make_learners(n, rng, "H"):
        order = 0
        for rep in range(2):
            chosen = rng.sample(BANK, 20)
            chosen.sort(key=lambda q: (C_CONCEPTS.index(q["concept"]), diff[q["difficulty"]]))
            for q in chosen:
                correct = lr.answer(q["concept"], q["difficulty"], rng)
                mm = _MARKS[q["difficulty"]]
                recs.append(Record(lr.learner_id, f"Exam{rep + 1}", order, q["question_id"], q["question_text"],
                                   q["concept"], q["course_outcome"], q["difficulty"], q["bloom_level"], mm,
                                   _marks(correct, mm, rng)))
                order += 1
    return recs


# ------------------------------------------------------------------ 1-3: knowledge tracing
hist = load_records(PROJECT / "data/sample/historical.csv")
results["kt_ablation"] = {
    "DKT": run_variant("DKT", hist, "dkt"),
    "EKT (full)": run_variant("EKT full", hist, "ekt", use_attention=True, aux_loss_weight=0.5),
    "EKT w/o attention": run_variant("EKT -att", hist, "ekt", use_attention=False, aux_loss_weight=0.5),
    "EKT w/o mastery loss": run_variant("EKT -aux", hist, "ekt", use_attention=True, aux_loss_weight=0.0),
    "EKT + exercise dropout 0.5": run_variant("EKT +xdrop", hist, "ekt", use_attention=True, aux_loss_weight=0.5,
                                              exercise_dropout=0.5),
}
ordered = ordered_history()
results["ordering"] = {
    "EKT, ordered exams": run_variant("EKT ordered", ordered, "ekt", use_attention=True, aux_loss_weight=0.5),
    "DKT, ordered exams": run_variant("DKT ordered", ordered, "dkt"),
    "EKT, shuffled exams": results["kt_ablation"]["EKT (full)"],
    "DKT, shuffled exams": results["kt_ablation"]["DKT"],
}
set_cfg()

# ------------------------------------------------------------------ 4: retrieval purity
from plrs.embeddings import get_embedder  # noqa: E402
from plrs.rag import chunk_text, load_document, split_sections  # noqa: E402

text = load_document(PROJECT / "data/sample/curriculum_c_programming.txt")
concepts = json.load(open(PROJECT / "data/store/concepts.json", encoding="utf-8"))
sent_re = re.compile(r"(?<=[.!?])\s+|\n+")
unit_sents = {}
for title, body in split_sections(text):
    for s in sent_re.split(body):
        s = s.strip()
        if len(s) > 20:
            unit_sents.setdefault(title, set()).add(s)
emb = get_embedder()


def retrieve(chunks, meta, qvec, k=4, section=None):
    vecs = emb.encode(chunks)
    order = np.argsort(-(vecs @ qvec))
    if section is not None:
        own = [i for i in order if meta[i] == section]
        order = own if own else list(order)
    return [chunks[i] for i in order[:k if section is None else 2 * k]]


def purity(ctx, unit):
    sents = [s.strip() for c in ctx for s in sent_re.split(c) if len(s.strip()) > 20]
    own = unit_sents.get(unit, set())
    hit = sum(1 for s in sents if s in own)
    covered = len({s for s in sents if s in own})
    return hit / max(len(sents), 1), covered / max(len(own), 1)


rp = {}
for size in (500, 1000):
    fixed = chunk_text(text, chunk_size=size, overlap=size // 10)
    unit_chunks, unit_meta = [], []
    for title, body in split_sections(text):
        for c in chunk_text(body, chunk_size=size, overlap=size // 10):
            unit_chunks.append(c)
            unit_meta.append(title)
    for label, chunks, meta, filt in [("Fixed windows + similarity", fixed, [None] * len(fixed), False),
                                      ("Unit-aware chunks + similarity", unit_chunks, unit_meta, False),
                                      ("Unit-aware chunks + unit filter (ours)", unit_chunks, unit_meta, True)]:
        ps, rs = [], []
        for c in concepts:
            q = emb.encode([c["name"] + ". " + " ".join(c["learning_objectives"])])[0]
            ctx = retrieve(chunks, meta, q, 4, c["name"] if filt else None)
            p, r = purity(ctx, c["name"])
            ps.append(p)
            rs.append(r)
        rp[f"{label} | {size}"] = {"chunk_size": size, "strategy": label, "chunks": len(chunks),
                                    "purity": float(np.mean(ps)), "coverage": float(np.mean(rs))}
        log("retrieval", label, size, round(np.mean(ps), 3), round(np.mean(rs), 3))
results["retrieval"] = rp

# ------------------------------------------------------------------ 5: closed-loop simulation
from plrs.evaluation import EvaluationRule  # noqa: E402
from plrs.kt import KnowledgeTracer  # noqa: E402
from plrs.pipeline import PersonalizedLearningSystem, simulated_answerer, simulated_study  # noqa: E402
from plrs.qgen import QuestionMemory  # noqa: E402
from plrs.rag import CurriculumKB  # noqa: E402

kb = CurriculumKB.load()
tracer = KnowledgeTracer.load("ekt")
current = by_learner(load_records(PROJECT / "data/sample/current_batch.csv"))
ids = sorted(current)[:40]
loop = {}
for cond, remediate in (("with remediation", True), ("without remediation", False)):
    mem_dir = models_dir() / f"paper_loop_{'rem' if remediate else 'norem'}"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "question_bank.json").unlink(missing_ok=True)
    system = PersonalizedLearningSystem(kb, tracer)
    system.generator.memory = QuestionMemory(directory=mem_dir)
    reached, gaps_per_round, mastery_per_round = [], [[], [], [], []], [[], [], [], []]
    for n, lid in enumerate(ids):
        rng = random.Random(1000 + n)
        theta = make_learners(1, rng)[0].theta
        for g in system.analyze(current[lid]):
            if g.obe_score is not None:
                theta[g.concept] = (g.obe_score - 55) / 25
        rounds = system.run_cycle(lid, current[lid], simulated_answerer(theta, rng),
                                  simulated_study(theta, rng) if remediate else None,
                                  rule=EvaluationRule(3, 67), max_rounds=3, save=False, seed=n)
        first_clear = next((r.round for r in rounds if not r.gaps), None)
        reached.append(first_clear)
        for r in rounds[:4]:
            gaps_per_round[r.round - 1].append(len(r.gaps))
            mastery_per_round[r.round - 1].append(float(np.mean(list(r.knowledge_state.values()))))
        # learners who finished early keep their last state for the later rounds
        for k in range(len(rounds), 4):
            gaps_per_round[k].append(0)
            mastery_per_round[k].append(mastery_per_round[len(rounds) - 1][-1])
    loop[cond] = {
        "learners": len(ids),
        "reached_by_round": {str(k): int(sum(1 for x in reached if x is not None and x <= k)) for k in (1, 2, 3, 4)},
        "mean_gaps_per_round": [float(np.mean(g)) for g in gaps_per_round],
        "mean_mastery_per_round": [float(np.mean(m)) for m in mastery_per_round],
    }
    log("loop", cond, loop[cond])
results["closed_loop"] = loop
results["seeds"] = SEEDS
OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
log("saved", OUT)
