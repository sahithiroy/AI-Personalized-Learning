"""Generative-AI providers: Gemini, Groq, OpenRouter, Ollama (free options), OpenAI and DeepSeek.

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
import time
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

        # retries are handled by FallbackProvider, which knows about free-tier rate limits
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=60)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt, *, task="", payload=None, system="You are an expert educator and assessment designer."):
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


class FallbackProvider(LLM):
    """Wraps a real provider; on an API error answers with the mock instead.

    * Requests are paced to ``llm.requests_per_minute`` so free tiers are not exceeded.
    * Rate limits (HTTP 429 / "exceeded your current quota" / "rate limit") are temporary: the call
      waits (using the delay the provider suggests when it gives one) and retries. Only after several
      consecutive calls still fail this way is the provider switched off for the rest of the run.
    * Account-level errors (no credit, bad key, retired model, no permission) will not fix themselves,
      so after one of those the provider is switched off at once instead of paying a slow failed
      network call on every request.
    """

    PERMANENT = ("401", "402", "403", "404", "insufficient", "credit", "not found", "invalid api key",
                 "permission", "no longer available", "connection error", "connection refused")
    # temporary: rate limits and busy/overloaded servers - wait and retry instead of giving up
    RATE_LIMIT = ("429", "rate limit", "rate_limit", "exceeded your current quota", "resource_exhausted",
                  "too many requests", "502", "503", "504", "high demand", "overloaded", "temporarily unavailable")

    def __init__(self, inner: LLM, mock: "MockProvider", rpm: float | None = None, retries: int = 2,
                 max_wait: float = 60.0, give_up_after: int = 3):
        self.inner, self.mock = inner, mock
        self.disabled = False
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.retries, self.max_wait, self.give_up_after = retries, max_wait, give_up_after
        self._last = 0.0
        self._rate_failures = 0
        self.sleep = time.sleep  # replaceable in tests

    @property
    def name(self) -> str:
        return f"{self.inner.name} (unavailable, offline mock)" if self.disabled else self.inner.name

    def _pace(self) -> None:
        wait = self._last + self.min_interval - time.monotonic()
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    @staticmethod
    def _retry_delay(text: str, attempt: int) -> float:
        m = re.search(r"retry[^0-9]{0,25}([0-9]+(?:\.[0-9]+)?)\s*s", text, re.I)
        return float(m.group(1)) + 1.0 if m else 10.0 * (attempt + 1)

    def _call(self, prompt, task, payload, **kw):
        """Real provider answer, or None when the mock must answer this call."""
        for attempt in range(self.retries + 1):
            self._pace()
            try:
                text = self.inner.generate(prompt, task=task, payload=payload, **kw)
                self._rate_failures = 0
                return text
            except Exception as exc:  # network, quota, auth ...
                msg = str(exc)
                low = msg.lower()
                if any(k in low for k in self.RATE_LIMIT) and not any(
                        k in low for k in ("insufficient", "credit")):
                    if attempt < self.retries:
                        delay = min(self._retry_delay(msg, attempt), self.max_wait)
                        log.info("%s rate-limited; waiting %.0f s before retrying", self.inner.name, delay)
                        self.sleep(delay)
                        continue
                    self._rate_failures += 1
                    if self._rate_failures >= self.give_up_after:
                        self.disabled = True
                        log.warning("%s keeps hitting its rate limit/quota (%s) - using the offline mock for the "
                                    "rest of this run", self.inner.name, msg[:160])
                    else:
                        log.warning("%s rate-limited; using the offline mock for this call", self.inner.name)
                    return None
                if any(k in low for k in self.PERMANENT):
                    self.disabled = True
                    log.warning("%s is unavailable (%s) - using the offline mock for the rest of this run",
                                self.inner.name, msg[:160])
                else:
                    log.warning("%s call failed (%s); using offline mock for this call", self.inner.name, msg[:160])
                return None
        return None

    def generate(self, prompt, *, task="", payload=None, **kw):
        if not self.disabled:
            text = self._call(prompt, task, payload, **kw)
            if text is not None:
                return text
        return self.mock.generate(prompt, task=task, payload=payload)

    def generate_json(self, prompt, *, task="", payload=None):
        if not self.disabled:
            text = self._call(prompt, task, payload)
            if text is not None:
                try:
                    return parse_json(text)
                except ValueError as exc:
                    log.warning("%s returned unparseable output (%s); using offline mock for this call",
                                self.inner.name, str(exc)[:120])
        return self.mock.generate_json(prompt, task=task, payload=payload)


_CACHE: dict[str, LLM] = {}


# Providers that speak the OpenAI chat API: name -> (API-key env var or None, default base URL, config model key).
# Gemini (Google AI Studio), Groq and OpenRouter offer free, rate-limited keys; Ollama runs locally with no key.
OPENAI_COMPATIBLE = {
    "gemini": ("GEMINI_API_KEY", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini_model"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek_model"),
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1", "groq_model"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "openrouter_model"),
    "ollama": (None, "http://localhost:11434/v1", "ollama_model"),
}


class CompatibleProvider(OpenAIProvider):
    """Any OpenAI-compatible endpoint (Gemini, DeepSeek, Groq, OpenRouter, Ollama)."""


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
        elif name in OPENAI_COMPATIBLE and (OPENAI_COMPATIBLE[name][0] is None
                                            or os.getenv(OPENAI_COMPATIBLE[name][0])):
            key_env, url, model_key = OPENAI_COMPATIBLE[name]
            provider = CompatibleProvider(
                cfg[model_key], cfg["temperature"], os.getenv(key_env) if key_env else "ollama",
                base_url=os.getenv(f"{name.upper()}_BASE_URL", url),
            )
            provider.name = name
        elif name != "mock":
            log.info("No API key for %s - using offline mock provider", name)
    except ImportError as exc:
        log.warning("Package for %s not installed (%s) - using offline mock", name, exc)
        provider = mock
    if not provider.is_mock and cfg.get("fallback_to_mock", True):
        rpm = (cfg.get("requests_per_minute") or {}).get(name)
        provider = FallbackProvider(provider, mock, rpm=rpm)
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
