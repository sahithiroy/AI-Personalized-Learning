"""Sentence embeddings used by RAG, de-duplication, EKT exercise encoding and metrics.

Uses sentence-transformers (all-MiniLM-L6-v2, as in the paper) when installed.
Otherwise falls back to a deterministic hashing embedder (word uni/bi-grams +
character tri-grams) so everything keeps working offline.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from functools import lru_cache

import numpy as np

from .config import get_config

log = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """Offline fallback: feature hashing of n-grams, L2-normalised."""

    name = "hashing"

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _index(self, feature: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.md5(feature.encode("utf-8")).digest()[:8], "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        feats = list(tokens)
        feats += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        for tok in tokens:
            padded = f"#{tok}#"
            feats += [f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2)]
        for f in feats:
            idx, sign = self._index(f)
            vec[idx] += sign * (0.5 if f.startswith("c:") else 1.0)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._embed_one(t) for t in texts])


class SentenceTransformerEmbedder:
    name = "sentence-transformers"

    def __init__(self, model_name: str):
        for var in ("HF_HUB_DISABLE_SYMLINKS_WARNING", "HF_HUB_DISABLE_PROGRESS_BARS"):
            os.environ.setdefault(var, "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        from sentence_transformers import SentenceTransformer

        try:  # use the local cache first - avoids slow Hugging Face Hub round-trips
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)  # first run: download
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = dim_fn()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


@lru_cache(maxsize=1)
def get_embedder():
    """config rag.embedding_model (or env PLRS_EMBEDDER); 'hashing' forces the offline embedder."""
    model_name = os.getenv("PLRS_EMBEDDER") or get_config()["rag"]["embedding_model"]
    if model_name == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # not installed, or model download not possible
        log.warning("sentence-transformers unavailable (%s) - using offline hashing embedder", exc)
        return HashingEmbedder()


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between every row of a and every row of b."""
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T


def cosine(a: str, b: str) -> float:
    emb = get_embedder().encode([a, b])
    return float(cosine_matrix(emb[:1], emb[1:])[0, 0])
