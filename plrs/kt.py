"""Knowledge Tracing (paper Sec. III-B.1 and V-B).

* ``EKT``  - Exercise-aware Knowledge Tracing (Liu et al., TKDE 2021): each step's
  input is the *text embedding of the exercise* (+ concept embedding), split by
  correctness, fed to an LSTM. The next answer is predicted from the knowledge
  state and the next exercise's content; with ``use_attention`` the state is an
  attention-weighted sum over past states, weighted by exercise similarity (EKTA).
  A dense sigmoid head ``y_t = sigma(W h_t + b)`` gives per-concept mastery.
* ``DKT``  - Deep Knowledge Tracing baseline (Piech et al., 2015).

Hyper-parameters follow the paper: embedding 100, LSTM 100, dropout 0.3, Adam
lr 1e-3, batch 32, 12 epochs, 75/25 learner split, BCE loss, AUC/MAE metrics.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import get_config, models_dir
from .data import Record, by_learner
from .embeddings import get_embedder

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- models
class DKT(nn.Module):
    def __init__(self, n_concepts: int, text_dim: int = 0, hidden_dim: int = 100, dropout: float = 0.3, **_):
        super().__init__()
        self.k = n_concepts
        self.lstm = nn.LSTM(2 * n_concepts, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, n_concepts)

    def forward(self, concepts, ex_emb, correct):
        x = nn.functional.one_hot(concepts + self.k * correct.long(), 2 * self.k).float()
        h, _ = self.lstm(x)
        mastery = self.out(self.drop(h))                       # (B, T, K) logits
        nxt = concepts[:, 1:].unsqueeze(-1)
        logits = mastery[:, :-1].gather(-1, nxt).squeeze(-1)   # predict step t+1 from h_t
        return logits, mastery


class EKT(nn.Module):
    def __init__(self, n_concepts: int, text_dim: int, embedding_dim: int = 100, hidden_dim: int = 100,
                 dropout: float = 0.3, use_attention: bool = True, exercise_dropout: float = 0.0, **_):
        super().__init__()
        self.use_attention = use_attention
        # randomly hide exercise text during training so the concept embedding also carries signal
        # (keeps the knowledge state sensible for newly generated questions never seen in training)
        self.exercise_dropout = exercise_dropout
        self.ex_proj = nn.Linear(text_dim, embedding_dim)
        self.concept_emb = nn.Embedding(n_concepts, embedding_dim)
        self.lstm = nn.LSTM(2 * embedding_dim, hidden_dim, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mastery = nn.Linear(hidden_dim, n_concepts)
        self.pred = nn.Sequential(nn.Linear(hidden_dim + embedding_dim, hidden_dim), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def encode(self, concepts, ex_emb):
        if self.training and self.exercise_dropout > 0:
            keep = torch.rand(ex_emb.shape[:2] + (1,), device=ex_emb.device) >= self.exercise_dropout
            ex_emb = ex_emb * keep
        return torch.tanh(self.ex_proj(ex_emb) + self.concept_emb(concepts))

    def forward(self, concepts, ex_emb, correct):
        e = self.encode(concepts, ex_emb)                      # (B, T, E)
        c = correct.float().unsqueeze(-1)
        x = torch.cat([e * c, e * (1 - c)], dim=-1)            # response-aware exercise input
        h, _ = self.lstm(x)
        h = self.drop(h)
        mastery = self.mastery(h)                              # (B, T, K)
        e_next, h_prev = e[:, 1:], h[:, :-1]
        if self.use_attention:
            # alpha_{t,j} ~ cos(e_{t+1}, e_j) for j <= t
            en = nn.functional.normalize(e, dim=-1)
            sim = en[:, 1:] @ en[:, :-1].transpose(1, 2)       # (B, T-1, T-1)
            t = sim.size(1)
            causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=sim.device))
            att = torch.softmax(sim.masked_fill(~causal, -1e9) * 5.0, dim=-1)
            h_prev = att @ h[:, :-1]
        nxt = concepts[:, 1:].unsqueeze(-1)
        logits = mastery[:, :-1].gather(-1, nxt).squeeze(-1) + self.pred(torch.cat([h_prev, e_next], -1)).squeeze(-1)
        return logits, mastery


MODELS = {"ekt": EKT, "dkt": DKT}


# --------------------------------------------------------------------------- data prep
@dataclass
class Step:
    concept: int
    text: str
    correct: int


class Encoder:
    """Maps concepts to indices and exercise texts to (cached) embeddings."""

    def __init__(self, concepts: list[str]):
        self.concepts = list(concepts)
        self.index = {c: i for i, c in enumerate(self.concepts)}
        self._cache: dict[str, np.ndarray] = {}
        self.embedder = get_embedder()
        self.text_dim = self.embedder.encode(["probe"]).shape[1]

    def embed(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.embedder.encode(missing)):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.text_dim), np.float32)

    def steps(self, records: list[Record], threshold: float | None = None) -> list[Step]:
        return [Step(self.index[r.concept], r.question_text or r.question_id, r.attained(threshold))
                for r in records if r.concept in self.index]

    def batch(self, seqs: list[list[Step]], max_len: int):
        seqs = [s[-max_len:] for s in seqs]
        t = max(len(s) for s in seqs)
        b = len(seqs)
        concepts = torch.zeros(b, t, dtype=torch.long)
        correct = torch.zeros(b, t, dtype=torch.long)
        mask = torch.zeros(b, t, dtype=torch.bool)
        ex = torch.zeros(b, t, self.text_dim)
        for i, s in enumerate(seqs):
            n = len(s)
            concepts[i, :n] = torch.tensor([x.concept for x in s])
            correct[i, :n] = torch.tensor([x.correct for x in s])
            mask[i, :n] = True
            ex[i, :n] = torch.from_numpy(self.embed([x.text for x in s]))
        return concepts, ex, correct, mask


# --------------------------------------------------------------------------- metrics
def auc_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U statistic (ties get average rank)."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    pos, neg = y_true.sum(), len(y_true) - y_true.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_prob)
    ranks = np.empty(len(y_prob))
    sorted_p = y_prob[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y_true == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def _evaluate(model, enc: Encoder, seqs, bs: int, max_len: int, loss_fn) -> dict:
    model.eval()
    ys, ps, losses = [], [], []
    with torch.no_grad():
        for i in range(0, len(seqs), bs):
            c, ex, r, m = enc.batch(seqs[i:i + bs], max_len)
            logits, _ = model(c, ex, r)
            tm = m[:, 1:]
            y = r[:, 1:][tm].float()
            lg = logits[tm]
            losses.append(loss_fn(lg, y).item() * len(y))
            ys.append(y.numpy())
            ps.append(torch.sigmoid(lg).numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"loss": sum(losses) / len(y), "auc": auc_score(y, p), "mae": float(np.abs(y - p).mean()),
            **classification_metrics(y, p)}


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Accuracy, precision, recall and F1 for 'answers correctly' predicted at ``threshold``."""
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    y = np.asarray(y_true).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": float((pred == y).mean()), "precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- training
def train_kt(records: list[Record], model_type: str = "ekt", epochs: int | None = None,
             seed: int = 42, save: bool = True) -> tuple["KnowledgeTracer", list[dict]]:
    cfg = get_config()["ekt"]
    epochs = epochs or cfg["epochs"]
    torch.manual_seed(seed)
    rng = random.Random(seed)

    concepts = sorted({r.concept for r in records})
    enc = Encoder(concepts)
    seqs = [enc.steps(rs) for rs in by_learner(records).values()]
    seqs = [s for s in seqs if len(s) >= 2]
    rng.shuffle(seqs)
    split = int(len(seqs) * cfg["train_split"])
    train, test = seqs[:split], seqs[split:]

    hparams = {"embedding_dim": cfg["embedding_dim"], "hidden_dim": cfg["hidden_dim"],
               "dropout": cfg["dropout"], "use_attention": cfg["use_attention"],
               "exercise_dropout": cfg.get("exercise_dropout", 0.0)}
    model = MODELS[model_type](len(concepts), enc.text_dim, **hparams)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.BCEWithLogitsLoss()
    bs, max_len = cfg["batch_size"], cfg["max_seq_len"]

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        rng.shuffle(train)
        for i in range(0, len(train), bs):
            c, ex, r, m = enc.batch(train[i:i + bs], max_len)
            logits, mastery = model(c, ex, r)
            tm = m[:, 1:]
            y = r[:, 1:][tm].float()
            # auxiliary loss trains the concept-mastery head y_t = sigma(W h_t + b) directly,
            # so the reported knowledge state is calibrated (not only the next-answer logit)
            aux = mastery[:, :-1].gather(-1, c[:, 1:].unsqueeze(-1)).squeeze(-1)
            loss = loss_fn(logits[tm], y) + 0.5 * loss_fn(aux[tm], y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        tr = _evaluate(model, enc, train, bs, max_len, loss_fn)
        te = _evaluate(model, enc, test, bs, max_len, loss_fn) if test else {}
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()}, **{f"test_{k}": v for k, v in te.items()}}
        history.append(row)
        log.info("[%s] epoch %d  train auc %.4f  test auc %.4f", model_type, epoch,
                 tr["auc"], te.get("auc", float("nan")))

    tracer = KnowledgeTracer(model, enc, model_type, hparams)
    if save:
        tracer.save()
    return tracer, history


# --------------------------------------------------------------------------- inference
class KnowledgeTracer:
    def __init__(self, model: nn.Module, encoder: Encoder, model_type: str, hparams: dict):
        self.model, self.encoder, self.model_type, self.hparams = model, encoder, model_type, hparams
        self.model.eval()

    @property
    def concepts(self) -> list[str]:
        return self.encoder.concepts

    def path(self) -> Path:
        return models_dir() / f"{self.model_type}.pt"

    def save(self, path: Path | None = None) -> Path:
        path = path or self.path()
        torch.save({"model_type": self.model_type, "state_dict": self.model.state_dict(),
                    "concepts": self.concepts, "hparams": self.hparams,
                    "text_dim": self.encoder.text_dim}, path)
        return path

    @classmethod
    def load(cls, model_type: str = "ekt", path: Path | None = None) -> "KnowledgeTracer":
        path = path or models_dir() / f"{model_type}.pt"
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `python -m plrs train-kt` first.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        enc = Encoder(ckpt["concepts"])
        if enc.text_dim != ckpt["text_dim"]:
            raise RuntimeError("Embedding model changed since training - retrain with `python -m plrs train-kt`.")
        model = MODELS[ckpt["model_type"]](len(ckpt["concepts"]), ckpt["text_dim"], **ckpt["hparams"])
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, enc, ckpt["model_type"], ckpt["hparams"])

    def trace(self, records: list[Record]) -> tuple[list[Step], np.ndarray]:
        """Knowledge state after every interaction: (steps, array of shape (T, K))."""
        steps = self.encoder.steps(records)
        if not steps:
            return [], np.full((1, len(self.concepts)), 0.5)
        max_len = get_config()["ekt"]["max_seq_len"]
        c, ex, r, _ = self.encoder.batch([steps], max_len)
        with torch.no_grad():
            _, mastery = self.model(c, ex, r)
        return steps[-max_len:], torch.sigmoid(mastery[0]).numpy()

    def knowledge_state(self, records: list[Record]) -> dict[str, float]:
        """Final per-concept mastery y_t = sigma(W h_t + b) for one learner."""
        _, states = self.trace(records)
        return {c: float(v) for c, v in zip(self.concepts, states[-1])}
