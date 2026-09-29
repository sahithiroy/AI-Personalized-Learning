"""Generative-AI providers: OpenAI (primary), Gemini and DeepSeek (cross-verification).

Every call goes through ``LLM.generate(prompt, task=..., payload=...)``:

* real providers send ``prompt`` to their API and return the raw text;
* the offline ``MockProvider`` ignores the prompt and builds a deterministic
  answer from ``payload`` - same JSON shape as the real models are asked for.

So every module works end-to-end without API keys, and switching to real
models is only a matter of filling in ``.env``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
from typing import Any

from .config import get_config

log = logging.getLogger(__name__)

BLOOM_BY_LEVEL = {
    "easy": ["Remember", "Understand"],
    "medium": ["Apply", "Analyze"],
    "hard": ["Evaluate", "Create"],
}


def parse_json(text: str) -> Any:
    """Parse JSON out of an LLM answer (tolerates ```json fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Could not parse JSON from model output: {text[:200]!r}")


# --------------------------------------------------------------------------- providers
class LLM:
    name = "base"
    is_mock = False

    def generate(self, prompt: str, *, task: str = "", payload: dict | None = None,
                 system: str = "You are an expert educator and assessment designer.") -> str:
        raise NotImplementedError

    def generate_json(self, prompt: str, *, task: str = "", payload: dict | None = None) -> Any:
        return parse_json(self.generate(prompt, task=task, payload=payload))


class OpenAIProvider(LLM):
    name = "openai"

    def __init__(self, model: str, temperature: float, api_key: str, base_url: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, *, task="", payload=None, system="You are an expert educator and assessment designer."):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek exposes an OpenAI-compatible endpoint."""

    name = "deepseek"


class GeminiProvider(LLM):
    name = "gemini"

    def __init__(self, model: str, temperature: float, api_key: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def generate(self, prompt, *, task="", payload=None, system="You are an expert educator and assessment designer."):
        resp = self.model.generate_content(
            f"{system}\n\n{prompt}", generation_config={"temperature": self.temperature}
        )
        return resp.text or ""


class FallbackProvider(LLM):
    """Wraps a real provider; on an API error answers with the mock instead.

    Account-level errors (no credit, bad key, unknown model, no permission) will not fix themselves,
    so after one of those the provider is switched off for the rest of the process instead of
    paying a slow failed network call on every request.
    """

    PERMANENT = ("401", "402", "403", "404", "insufficient", "quota", "credit", "not found",
                 "invalid api key", "permission")

    def __init__(self, inner: LLM, mock: "MockProvider"):
        self.inner, self.mock = inner, mock
        self.disabled = False

    @property
    def name(self) -> str:
        return f"{self.inner.name} (unavailable, offline mock)" if self.disabled else self.inner.name

    def _failed(self, exc: Exception) -> None:
        text = str(exc).lower()
        if any(k in text for k in self.PERMANENT):
            self.disabled = True
            log.warning("%s is unavailable (%s) - using the offline mock for the rest of this run",
                        self.inner.name, str(exc)[:160])
        else:
            log.warning("%s call failed (%s); using offline mock for this call", self.inner.name, str(exc)[:160])

    def generate(self, prompt, *, task="", payload=None, **kw):
        if not self.disabled:
            try:
                return self.inner.generate(prompt, task=task, payload=payload, **kw)
            except Exception as exc:  # network, quota, auth ...
                self._failed(exc)
        return self.mock.generate(prompt, task=task, payload=payload)

    def generate_json(self, prompt, *, task="", payload=None):
        if not self.disabled:
            try:
                return parse_json(self.inner.generate(prompt, task=task, payload=payload))
            except Exception as exc:
                self._failed(exc)
        return self.mock.generate_json(prompt, task=task, payload=payload)


_CACHE: dict[str, LLM] = {}


def get_llm(name: str | None = None) -> LLM:
    """Return provider ``name`` (default: config llm.primary), falling back to the mock."""
    cfg = get_config()["llm"]
    name = (name or cfg["primary"]).lower()
    if name in _CACHE:
        return _CACHE[name]
    mock = MockProvider(persona=name)
    provider: LLM = mock
    try:
        if name == "openai" and os.getenv("OPENAI_API_KEY"):
            provider = OpenAIProvider(cfg["openai_model"], cfg["temperature"], os.environ["OPENAI_API_KEY"])
        elif name == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
            provider = DeepSeekProvider(
                cfg["deepseek_model"], cfg["temperature"], os.environ["DEEPSEEK_API_KEY"],
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            )
        elif name == "gemini" and os.getenv("GEMINI_API_KEY"):
            provider = GeminiProvider(cfg["gemini_model"], cfg["temperature"], os.environ["GEMINI_API_KEY"])
        elif name != "mock":
            log.info("No API key for %s - using offline mock provider", name)
    except ImportError as exc:
        log.warning("Package for %s not installed (%s) - using offline mock", name, exc)
        provider = mock
    if not provider.is_mock and cfg.get("fallback_to_mock", True):
        provider = FallbackProvider(provider, mock)
    _CACHE[name] = provider
    return provider


def get_verifiers() -> list[LLM]:
    return [get_llm(n) for n in get_config()["llm"]["verifiers"]]


# --------------------------------------------------------------------------- offline mock
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_STOP = set(
    "a an the of to in on for and or is are be by with as that this it its from at into can "
    "which what when how using used use each their they them than then also such these those "
    "was were has have had not but if any all may more most other some only will should".split()
)


def _sentences(texts: list[str]) -> list[str]:
    out = []
    for t in texts:
        for s in _SENT_RE.split(re.sub(r"\s+", " ", t)):
            s = s.strip(" -*•")
            if 40 <= len(s) <= 260 and s[0].isupper() and not re.match(r"(?i)(unit|module|chapter)\s*[\dIVX]+|course outcome|CO\d", s):
                out.append(s if s.endswith((".", "?", "!")) else s + ".")
    return list(dict.fromkeys(out))


def _keywords(text: str, other: str = "") -> list[str]:
    """Content words of ``text`` ranked by frequency, penalised if common in ``other``."""
    words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_+#.]*[A-Za-z0-9_)]|[A-Za-z]", text)
             if w.lower() not in _STOP and len(w) > 3]
    other_l = other.lower()
    freq: dict[str, float] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    score = {w: f / (1 + other_l.count(w.lower())) for w, f in freq.items()}
    return sorted(score, key=lambda w: (-score[w], words.index(w)))


def _corrupt(sentence: str, rng: random.Random) -> str:
    """Make a plausible but false variant of a true statement (distractor)."""
    swaps = [(" is ", " is not "), (" are ", " are not "), (" can ", " cannot "),
             (" must ", " need not "), (" always ", " never "), (" before ", " after "),
             (" increases", " decreases"), (" first ", " last "), (" zero", " one"),
             (" stack", " heap"), (" heap", " stack"), (" local", " global")]
    rng.shuffle(swaps)
    for a, b in swaps:
        if a in sentence:
            return sentence.replace(a, b, 1)
    return "It is never necessary to consider this: " + sentence[0].lower() + sentence[1:]


class MockProvider(LLM):
    """Deterministic, offline stand-in for a generative model."""

    is_mock = True

    def __init__(self, persona: str = "mock"):
        self.name = f"mock:{persona}" if persona != "mock" else "mock"
        self.persona = persona

    def generate(self, prompt, *, task="", payload=None, **kw):
        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            return json.dumps({"text": "Offline mock provider: no handler for this task."})
        return json.dumps(handler(payload or {}))

    # ---- concept extraction (Sec. III-A)
    def _task_extract_concepts(self, p):
        text = p.get("text", "")
        concepts = []
        for m in re.finditer(r"(?im)^\s*(?:unit|module|chapter)\s*[\dIVX]+\s*[:.\-]\s*(.+)$", text):
            concepts.append(m.group(1).strip().rstrip("."))
        if not concepts:  # fall back to the most frequent key phrases
            freq: dict[str, int] = {}
            for w in _keywords(text):
                freq[w] = freq.get(w, 0) + text.count(w)
            concepts = [w.title() for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:5]]
        outcomes = re.findall(r"(?im)^\s*(CO\d+)\s*[:.\-]\s*(.+)$", text)
        result = []
        for i, name in enumerate(dict.fromkeys(concepts)):
            section = _section_for(text, name)
            objectives = [s for s in _sentences([section])][:3]
            result.append({
                "id": f"C{i + 1}",
                "name": name,
                "course_outcome": outcomes[i][0] if i < len(outcomes) else f"CO{i + 1}",
                "learning_objectives": objectives or [f"Understand the fundamentals of {name}."],
            })
        return {"concepts": result}

    # ---- MCQ generation (Sec. III-D)
    def _task_generate_mcq(self, p):
        concept, level, n = p["concept"], p["level"], int(p.get("n", 5))
        rng = random.Random(f"{self.persona}|{concept}|{level}|{p.get('seed', 0)}")
        own = _sentences(p.get("context", []))
        other = _sentences(p.get("other_context", [])) or own
        rng.shuffle(own)
        blooms = BLOOM_BY_LEVEL.get(level, ["Understand"])
        keywords = _keywords(" ".join(own), " ".join(other))[:25]
        questions = []
        for i in range(n):
            if not own:
                break
            s = own[i % len(own)]
            variant = (i // max(len(own), 1) + i) % 3
            bloom = blooms[i % len(blooms)]
            if level == "easy" and variant == 0 and keywords:
                in_s = _keywords(s)
                key = max([k for k in in_s if k in keywords] or in_s, key=len)
                stem = f"Fill in the blank ({concept}): \"{s.replace(key, '_____', 1)}\""
                pool = [k for k in keywords if k != key]
                rng.shuffle(pool)
                correct, distractors = key, (pool + ["pointer", "array", "loop"])[:3]
            elif level == "hard" or variant == 2:
                stem = f"Which of the following statements about {concept} is INCORRECT?"
                correct = _corrupt(s, rng)
                truths = [x for x in own if x != s]
                rng.shuffle(truths)
                distractors = (truths + other)[:3]
            else:
                stem = (f"Which statement correctly describes {concept}?" if level == "easy"
                        else f"A learner must apply ideas from {concept} to solve a problem. "
                             "Which statement should guide the solution?")
                correct = s
                # distractors must be false: a corrupted copy of this and of other statements
                others = [x for x in own + other if x != s]
                rng.shuffle(others)
                distractors = list(dict.fromkeys([_corrupt(s, rng)] + [_corrupt(x, rng) for x in others[:2]]))
            options = [correct] + [d for d in distractors if d != correct][:3]
            while len(options) < 4:
                options.append(f"None of the above ({len(options)})")
            rng.shuffle(options)
            questions.append({
                "question": stem,
                "options": options,
                "answer": options.index(correct),
                "bloom_level": bloom,
                "difficulty": level,
                "explanation": f"Based on the course material: {s}",
            })
        return {"questions": questions}

    def _task_refine_mcq(self, p):
        q = dict(p["question"])
        q["question"] = re.sub(r"\s+", " ", q["question"]).strip()
        if not q["question"].endswith(("?", "\"", ".")):
            q["question"] += "?"
        return q

    def _task_verify_mcq(self, p):
        q = p["question"]
        opts = q.get("options", [])
        score = 5
        if len(opts) != 4 or len(set(opts)) != 4:
            score -= 2
        if not (0 <= int(q.get("answer", -1)) < len(opts)):
            score -= 3
        if len(q.get("question", "")) < 15:
            score -= 1
        if q.get("bloom_level") not in BLOOM_BY_LEVEL.get(q.get("difficulty", ""), []):
            score -= 1
        return {"score": max(score, 1), "bloom_level_ok": score >= 4,
                "difficulty_ok": score >= 3, "feedback": f"{self.name} heuristic review"}

    # ---- remedial recommendation (Sec. III-C)
    def _task_remedial(self, p):
        concept, mastery = p["concept"], float(p.get("mastery", 0.0))
        round_no = int(p.get("round", 1))
        sents = _sentences(p.get("context", []))
        keywords = _keywords(" ".join(sents), " ".join(p.get("other_context", [])))[:8] or [concept]
        wrong = [q for q in p.get("attempts", []) if not q.get("correct")]
        topics, used = [], set()
        for kw in keywords:
            points = [x for x in sents if kw.lower() in x.lower() and x not in used][:2]
            if points:
                used.update(points)
                topics.append({"topic": kw, "points": points})
        topics = topics or [{"topic": concept, "points": [f"Re-read the course notes on {concept}."]}]
        explanation = " ".join(sents[:3 + round_no]) or f"Revise the core ideas of {concept}."
        if round_no > 1:
            explanation = ("Let's go through it more slowly with a simpler example. " + explanation)
        return {
            "concept": concept,
            "learning_objectives": [f"Explain the key ideas of {concept}.",
                                    f"Apply {keywords[0]} correctly in small programs.",
                                    f"Identify common mistakes involving {keywords[min(1, len(keywords) - 1)]}."],
            "topics_to_revise": topics[:4],
            "remedial_explanation": explanation,
            "practice_activities": [
                {"title": f"Practice {i + 1}: {t['topic']}",
                 "description": f"Write and test a short program that illustrates this rule: {t['points'][0]}",
                 "input": "at least two test cases of your choice",
                 "expected_output": "output that matches your hand-traced result"}
                for i, t in enumerate(topics[:2])
            ],
            "concept_gap_rationale": (
                f"The knowledge-tracing model estimates your mastery of '{concept}' at {mastery:.0%}, "
                f"below the OBE target. In your latest assessment you answered "
                f"{len(p.get('attempts', [])) - len(wrong)} of {len(p.get('attempts', []))} questions on this "
                "concept correctly"
                + (f"; for example you missed \"{wrong[0]['question'][:90]}\"" if wrong else "")
                + ". The plan focuses on the topics that concept's questions depend on."
            ),
        }

    def _task_verify_remedial(self, p):
        rec = p["recommendation"]
        required = ["learning_objectives", "topics_to_revise", "remedial_explanation",
                    "practice_activities", "concept_gap_rationale"]
        missing = [k for k in required if not rec.get(k)]
        return {"score": max(5 - 2 * len(missing), 1), "missing_sections": missing,
                "feedback": f"{self.name} structural review"}


def _section_for(text: str, heading: str) -> str:
    idx = text.find(heading)
    if idx == -1:
        return ""
    nxt = re.search(r"(?im)^\s*(?:unit|module|chapter)\s*[\dIVX]+\s*[:.\-]", text[idx + len(heading):])
    return text[idx: idx + len(heading) + (nxt.start() if nxt else 3000)]
