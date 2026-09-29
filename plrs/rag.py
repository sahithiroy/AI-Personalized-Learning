"""Curriculum ingestion and retrieval (paper Sec. III-A, III-D and Appendix A).

PDF/text -> overlapping character chunks (1000 / 100) -> embeddings -> vector DB
(FAISS when installed, otherwise an exact numpy index). Concepts are extracted
from the curriculum with the primary LLM and saved to ``concepts.json`` so an
instructor can review and correct them.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import get_config, store_dir
from .embeddings import get_embedder
from .llm import get_llm


# --------------------------------------------------------------------------- loading
def load_document(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


HEADING_RE = re.compile(r"(?im)^\s*(?:unit|module|chapter)\s*[\dIVX]+\s*[:.\-]\s*(.+?)\s*$")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a curriculum into (heading, body) sections on 'Unit/Module/Chapter N:' lines."""
    marks = list(HEADING_RE.finditer(text))
    if not marks:
        return [("", text)]
    sections = [("Overview", text[: marks[0].start()])] if text[: marks[0].start()].strip() else []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        sections.append((m.group(1).rstrip("."), text[m.start():end]))
    return sections


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Character chunks (~1000) with overlap (~100), cut at sentence/line boundaries when possible."""
    cfg = get_config()["rag"]
    size = chunk_size or cfg["chunk_size"]
    overlap = cfg["chunk_overlap"] if overlap is None else overlap
    text = text.strip()
    boundary = re.compile(r"(?<=[.!?])\s+|\n+")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cuts = [m.end() for m in boundary.finditer(text, start + size // 2, end)]
            if cuts:
                end = cuts[-1]
            else:
                space = text.rfind(" ", start + size // 2, end)
                end = space if space != -1 else end
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        # next chunk starts ~overlap chars back, at the start of the sentence containing that point
        back = max(end - overlap, start + 1)
        cuts = [m.end() for m in boundary.finditer(text, start + 1, back + 1) if m.end() <= back]
        if cuts:
            start = cuts[-1]
        else:
            space = text.find(" ", back, end)
            start = space + 1 if space != -1 else end
    return [c for c in chunks if c]


# --------------------------------------------------------------------------- vector store
class VectorStore:
    """Inner-product index over L2-normalised vectors (= cosine similarity)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.texts: list[str] = []
        self.meta: list[dict] = []
        self._faiss = None
        try:
            import faiss

            self._faiss = faiss.IndexFlatIP(dim)
        except ImportError:
            pass

    def __len__(self) -> int:
        return len(self.texts)

    def add(self, vectors: np.ndarray, texts: list[str], meta: list[dict] | None = None) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if len(vectors) == 0:
            return
        self.vectors = np.vstack([self.vectors, vectors])
        self.texts += texts
        self.meta += meta or [{} for _ in texts]
        if self._faiss is not None:
            self._faiss.add(vectors)

    def search(self, query: np.ndarray, k: int = 4) -> list[tuple[float, int]]:
        if len(self) == 0:
            return []
        k = min(k, len(self))
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, k)
            return [(float(s), int(i)) for s, i in zip(scores[0], idx[0])]
        scores = (self.vectors @ q.T).ravel()
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), int(i)) for i in top]

    def save(self, directory: Path, name: str) -> None:
        np.save(directory / f"{name}.npy", self.vectors)
        (directory / f"{name}.json").write_text(
            json.dumps({"texts": self.texts, "meta": self.meta}, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path, name: str) -> "VectorStore":
        vectors = np.load(directory / f"{name}.npy")
        data = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        store = cls(vectors.shape[1])
        store.add(vectors, data["texts"], data["meta"])
        return store


# --------------------------------------------------------------------------- knowledge base
@dataclass
class Concept:
    id: str
    name: str
    course_outcome: str = ""
    learning_objectives: list[str] = field(default_factory=list)


class CurriculumKB:
    """Vector DB over the curriculum plus the instructor-approved concept list."""

    STORE_NAME = "curriculum"

    def __init__(self, store: VectorStore, concepts: list[Concept] | None = None, raw_text: str = ""):
        self.store = store
        self.concepts = concepts or []
        self.raw_text = raw_text

    # ---- build / persist
    @classmethod
    def ingest(cls, paths: list[str | Path]) -> "CurriculumKB":
        embedder = get_embedder()
        texts, chunks, meta = [], [], []
        for p in paths:
            text = load_document(p)
            texts.append(text)
            for section, body in split_sections(text):
                for i, c in enumerate(chunk_text(body)):
                    chunks.append(c)
                    meta.append({"source": Path(p).name, "section": section, "chunk": i})
        store = VectorStore(embedder.encode(["dim probe"]).shape[1])
        store.add(embedder.encode(chunks), chunks, meta)
        return cls(store, raw_text="\n\n".join(texts))

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or store_dir()
        self.store.save(directory, self.STORE_NAME)
        (directory / "curriculum.txt").write_text(self.raw_text, encoding="utf-8")
        self.save_concepts(directory)
        return directory

    def save_concepts(self, directory: Path | None = None) -> Path:
        path = (directory or store_dir()) / "concepts.json"
        path.write_text(json.dumps([asdict(c) for c in self.concepts], indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, directory: Path | None = None) -> "CurriculumKB":
        directory = directory or store_dir()
        if not (directory / f"{cls.STORE_NAME}.npy").exists():
            raise FileNotFoundError(
                f"No curriculum index in {directory}. Run `python -m plrs ingest <curriculum.pdf>` first."
            )
        store = VectorStore.load(directory, cls.STORE_NAME)
        concepts = []
        if (directory / "concepts.json").exists():
            concepts = [Concept(**c) for c in json.loads((directory / "concepts.json").read_text(encoding="utf-8"))]
        raw = (directory / "curriculum.txt").read_text(encoding="utf-8") if (directory / "curriculum.txt").exists() else ""
        return cls(store, concepts, raw)

    # ---- concept extraction (Generative AI, instructor reviews concepts.json afterwards)
    def extract_concepts(self) -> list[Concept]:
        text = self.raw_text[:12000]
        prompt = (
            "From the curriculum below, extract the key knowledge concepts that learners must attain. "
            "Return JSON: {\"concepts\": [{\"id\": \"C1\", \"name\": str, \"course_outcome\": \"CO1\", "
            "\"learning_objectives\": [str, ...]}]}. Use one concept per unit/topic.\n\n"
            f"CURRICULUM:\n{text}"
        )
        data = get_llm().generate_json(prompt, task="extract_concepts", payload={"text": self.raw_text})
        items = data["concepts"] if isinstance(data, dict) else data
        self.concepts = [Concept(**{k: c.get(k) for k in ("id", "name", "course_outcome", "learning_objectives")
                                    if c.get(k) is not None}) for c in items]
        return self.concepts

    def concept(self, key: str) -> Concept:
        for c in self.concepts:
            if key in (c.id, c.name):
                return c
        raise KeyError(f"Unknown concept {key!r}. Known: {[c.name for c in self.concepts]}")

    # ---- retrieval
    def retrieve(self, query: str, k: int | None = None) -> list[str]:
        k = k or get_config()["rag"]["top_k"]
        q = get_embedder().encode([query])[0]
        return [self.store.texts[i] for _, i in self.store.search(q, k)]

    def context_for(self, concept: Concept, k: int | None = None) -> list[str]:
        """The concept's own curriculum section (ranked by similarity), else the nearest chunks."""
        k = k or get_config()["rag"]["top_k"]
        query = concept.name + ". " + " ".join(concept.learning_objectives)
        q = get_embedder().encode([query])[0]
        hits = self.store.search(q, len(self.store))
        own = [i for _, i in hits if self.store.meta[i].get("section", "").lower() == concept.name.lower()]
        ids = own[:2 * k] if own else [i for _, i in hits][:k]  # stay inside the concept's section
        return [self.store.texts[i] for i in ids]

    def other_context(self, concept: Concept, k: int = 2) -> list[str]:
        """Chunks about *other* concepts, used by the mock model to build distractors."""
        own = set(self.context_for(concept))
        out = []
        for c in self.concepts:
            if c.name != concept.name:
                out += [t for t in self.context_for(c, k) if t not in own]
        return list(dict.fromkeys(out))
