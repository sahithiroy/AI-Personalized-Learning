"""Evaluation metrics (paper Sec. IV-B, V-A.2, V-C).

* semantic similarity - S-BERT cosine between generated and teacher questions
* readability         - Flesch-Kincaid grade level (and Flesch reading ease)
* diversity           - Self-BLEU / 100 (lower = less redundant)
* relevance (1-5)     - S-BERT similarity mapped to a 5-point scale
* paired t-test       - pre vs post remediation scores (scipy used when available)
"""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .embeddings import cosine_matrix, get_embedder, tokenize


# --------------------------------------------------------------------------- similarity / relevance
def semantic_similarity(generated: list[str], references: list[str]) -> float:
    """Mean over generated questions of the best cosine match among the references."""
    if not generated or not references:
        return 0.0
    emb = get_embedder()
    sims = cosine_matrix(emb.encode(generated), emb.encode(references))
    return float(sims.max(axis=1).mean())


def relevance_from_similarity(sim: float) -> int:
    """Heuristic mapping: > 0.9 -> 5 ... < 0.6 -> 1."""
    if sim > 0.9:
        return 5
    if sim > 0.8:
        return 4
    if sim > 0.7:
        return 3
    if sim >= 0.6:
        return 2
    return 1


def relevance_score(generated: list[str], references: list[str]) -> float:
    if not generated or not references:
        return 0.0
    emb = get_embedder()
    sims = cosine_matrix(emb.encode(generated), emb.encode(references)).max(axis=1)
    return float(np.mean([relevance_from_similarity(float(s)) for s in sims]))


# --------------------------------------------------------------------------- readability
def _syllables(word: str) -> int:
    word = word.lower()
    groups = re.findall(r"[aeiouy]+", word)
    n = len(groups)
    if word.endswith("e") and n > 1 and not word.endswith(("le", "ee")):
        n -= 1
    return max(n, 1)


def _counts(text: str) -> tuple[int, int, int]:
    sentences = max(len(re.findall(r"[.!?]+", text)), 1)
    words = re.findall(r"[A-Za-z]+", text)
    return sentences, max(len(words), 1), sum(_syllables(w) for w in words)


def flesch_kincaid_grade(text: str) -> float:
    s, w, syl = _counts(text)
    return 0.39 * (w / s) + 11.8 * (syl / w) - 15.59


def flesch_reading_ease(text: str) -> float:
    s, w, syl = _counts(text)
    return 206.835 - 1.015 * (w / s) - 84.6 * (syl / w)


def readability(questions: list[str]) -> float:
    return float(np.mean([flesch_kincaid_grade(q) for q in questions])) if questions else 0.0


# --------------------------------------------------------------------------- diversity (Self-BLEU)
def _bleu(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """Corpus-free sentence BLEU (0-100) with add-one smoothing, like sacrebleu's 'exp' default."""
    if not candidate:
        return 0.0
    log_p = 0.0
    for n in range(1, max_n + 1):
        cand = Counter(tuple(candidate[i:i + n]) for i in range(len(candidate) - n + 1))
        max_ref: Counter = Counter()
        for ref in references:
            for g, c in Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)).items():
                max_ref[g] = max(max_ref[g], c)
        match = sum(min(c, max_ref[g]) for g, c in cand.items())
        total = max(sum(cand.values()), 0)
        log_p += math.log((match + 1) / (total + 1)) if n > 1 else math.log(max(match, 1e-9) / max(total, 1))
    ref_len = min((len(r) for r in references), key=lambda L: (abs(L - len(candidate)), L))
    bp = 1.0 if len(candidate) > ref_len else math.exp(1 - ref_len / max(len(candidate), 1))
    return 100 * bp * math.exp(log_p / max_n)


def self_bleu(questions: list[str]) -> float:
    """Average BLEU of each question against all the others (0-100)."""
    if len(questions) < 2:
        return 0.0
    try:
        import sacrebleu

        scores = [sacrebleu.sentence_bleu(q, questions[:i] + questions[i + 1:]).score
                  for i, q in enumerate(questions)]
    except ImportError:
        toks = [tokenize(q) for q in questions]
        scores = [_bleu(t, toks[:i] + toks[i + 1:]) for i, t in enumerate(toks)]
    return float(np.mean(scores))


def diversity_score(questions: list[str]) -> float:
    """Self-BLEU normalised to 0-1 (paper: lower = less redundancy)."""
    return self_bleu(questions) / 100.0


def question_metrics(generated: list[str], references: list[str]) -> dict:
    return {
        "semantic_similarity": round(semantic_similarity(generated, references), 4),
        "readability_fk_grade": round(readability(generated), 2),
        "diversity_self_bleu": round(diversity_score(generated), 4),
        "relevance_1_5": round(relevance_score(generated, references), 2),
        "n_questions": len(generated),
    }


# --------------------------------------------------------------------------- paired t-test
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + aa * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + aa / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def paired_t_test(pre: list[float], post: list[float]) -> dict:
    """t = mean(pre - post) / (sd / sqrt(n)), two-sided p-value (paper Table 6 convention)."""
    pre_a, post_a = np.asarray(pre, float), np.asarray(post, float)
    diff = pre_a - post_a
    n = len(diff)
    if n < 2:
        raise ValueError("paired t-test needs at least two pairs")
    try:
        from scipy import stats

        t, p = stats.ttest_rel(pre_a, post_a)
        t, p = float(t), float(p)
    except ImportError:
        sd = diff.std(ddof=1)
        t = float("-inf") if sd == 0 else float(diff.mean() / (sd / math.sqrt(n)))
        df = n - 1
        p = 0.0 if math.isinf(t) else _betainc(df / 2, 0.5, df / (df + t * t))
    return {"t_statistic": t, "p_value": p, "n": n, "mean_pre": float(pre_a.mean()),
            "mean_post": float(post_a.mean()), "mean_diff": float(post_a.mean() - pre_a.mean())}
