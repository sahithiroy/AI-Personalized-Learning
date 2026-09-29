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
