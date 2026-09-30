from plrs.data import C_CONCEPTS
from plrs.llm import MockProvider, parse_json
from plrs.rag import CurriculumKB, chunk_text, split_sections


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} explains a fact." for i in range(200))
    chunks = chunk_text(text, chunk_size=300, overlap=60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert all(c[0].isupper() for c in chunks)  # every chunk starts at a sentence start
    # consecutive chunks overlap: the next chunk starts with a sentence from the end of the previous one
    assert chunks[1].split(".")[0] in chunks[0]


def test_split_sections_finds_units():
    text = "Intro text.\nUnit 1: Alpha\nA is first.\nUnit 2: Beta\nB is second."
    names = [h for h, _ in split_sections(text)]
    assert names == ["Overview", "Alpha", "Beta"]


def test_ingest_and_concept_extraction(kb):
    assert len(kb.store) > 0
    assert [c.name for c in kb.concepts] == C_CONCEPTS
    assert kb.concepts[0].course_outcome == "CO1"


def test_context_prefers_own_section(kb):
    concept = kb.concept("String Manipulation and Text Processing")
    ctx = kb.context_for(concept)
    assert "null character" in ctx[0]


def test_kb_roundtrip(kb):
    loaded = CurriculumKB.load()
    assert len(loaded.store) == len(kb.store)
    assert [c.name for c in loaded.concepts] == [c.name for c in kb.concepts]


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! Here it is: {"a": [1, 2]} hope it helps') == {"a": [1, 2]}
    assert parse_json("[1, 2]") == [1, 2]


def test_mock_mcq_shape(kb):
    concept = kb.concept("C3")
    out = MockProvider("openai").generate_json(
        "", task="generate_mcq",
        payload={"concept": concept.name, "level": "easy", "n": 6, "context": kb.context_for(concept),
                 "other_context": kb.other_context(concept)})
    qs = out["questions"]
    assert len(qs) == 6
    for q in qs:
        assert len(q["options"]) == 4 and 0 <= q["answer"] < 4


def test_fallback_switches_off_after_account_error():
    from plrs.llm import FallbackProvider, LLM

    class Broke(LLM):
        name = "broke"
        calls = 0

        def generate(self, prompt, **kw):
            Broke.calls += 1
            raise RuntimeError("Error code: 402 - Insufficient Balance")

    fb = FallbackProvider(Broke(), MockProvider("broke"))
    for _ in range(3):
        out = fb.generate_json("", task="verify_mcq",
                               payload={"question": {"question": "What is x?", "options": ["a", "b", "c", "d"],
                                                     "answer": 0, "difficulty": "easy", "bloom_level": "Remember"}})
        assert "score" in out
    assert Broke.calls == 1 and fb.disabled


def test_free_openai_compatible_providers(monkeypatch):
    from plrs import llm

    for name, env in [("groq", "GROQ_API_KEY"), ("openrouter", "OPENROUTER_API_KEY")]:
        llm._CACHE.pop(name, None)
        monkeypatch.setenv(env, "")
        assert llm.get_llm(name).is_mock  # no key -> offline mock
        llm._CACHE.pop(name, None)
        monkeypatch.setenv(env, "test-key")
        p = llm.get_llm(name)  # no network call is made when building the client
        assert p.name == name and isinstance(p.inner, llm.CompatibleProvider)
        assert str(p.inner.client.base_url).startswith(llm.OPENAI_COMPATIBLE[name][1])
        llm._CACHE.pop(name, None)
    llm._CACHE.pop("ollama", None)
    assert llm.get_llm("ollama").name == "ollama"  # local, needs no key
    llm._CACHE.pop("ollama", None)


def test_rate_limit_waits_and_retries():
    from plrs.llm import FallbackProvider, LLM

    class Busy(LLM):
        name = "busy"
        calls = 0

        def generate(self, prompt, **kw):
            Busy.calls += 1
            if Busy.calls <= 2:
                raise RuntimeError("Error code: 429 - You exceeded your current quota. Please retry in 3.5s.")
            return '{"score": 4}'

    waits = []
    fb = FallbackProvider(Busy(), MockProvider("busy"), rpm=None)
    fb.sleep = waits.append
    assert fb.generate_json("", task="verify_mcq", payload={}) == {"score": 4}  # real answer after 2 retries
    assert Busy.calls == 3 and not fb.disabled
    assert waits == [4.5, 4.5]  # used the provider's suggested delay (+1 s)


def test_rate_limit_gives_up_after_repeated_failures():
    from plrs.llm import FallbackProvider, LLM

    class Limited(LLM):
        name = "limited"

        def generate(self, prompt, **kw):
            raise RuntimeError("429 Too Many Requests")

    fb = FallbackProvider(Limited(), MockProvider("limited"), retries=1, give_up_after=2)
    fb.sleep = lambda s: None
    q = {"question": {"question": "What is x?", "options": ["a", "b", "c", "d"], "answer": 0,
                      "difficulty": "easy", "bloom_level": "Remember"}}
    fb.generate_json("", task="verify_mcq", payload=q)
    assert not fb.disabled  # one bad call is not enough
    fb.generate_json("", task="verify_mcq", payload=q)
    assert fb.disabled and "offline mock" in fb.name


def test_request_pacing():
    from plrs.llm import FallbackProvider, LLM

    class Ok(LLM):
        name = "ok"

        def generate(self, prompt, **kw):
            return "{}"

    waits = []
    fb = FallbackProvider(Ok(), MockProvider("ok"), rpm=60)  # at most one request per second
    fb.sleep = waits.append
    fb.generate("")
    fb.generate("")
    assert len(waits) == 1 and 0 < waits[0] <= 1.0
