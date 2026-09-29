import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(kb, tracer):
    from plrs import api

    api._state.clear()
    api._state.update({"sessions": {}, "kb": kb, "tracer": tracer})
    return TestClient(api.app)


LEARNER = {"learner_id": "L1", "records": [
    {"question_id": "Q31", "question_text": "How are elements of a one-dimensional array indexed in C?",
     "concept": "Algorithmic Problem Solving and Array Manipulation", "difficulty": "easy",
     "max_marks": 2, "marks": 0},
    {"question_id": "Q41", "question_text": "How is the end of a string marked in C?",
     "concept": "String Manipulation and Text Processing", "difficulty": "easy", "max_marks": 2, "marks": 2},
]}


def test_health_and_concepts(client):
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/concepts").json()) == 5


def test_analyze_and_recommend(client):
    res = client.post("/analyze", json=LEARNER).json()
    assert {c["concept"] for c in res["concepts"]} >= {"String Manipulation and Text Processing"}
    recs = client.post("/recommend", json=LEARNER).json()["recommendations"]
    assert recs and "Concept Gap Rationale" in recs[0]["markdown"]


def test_evaluation_flow(client):
    concept = "Operators, Pointers and Control Structures"
    start = client.post("/evaluation", json={"learner_id": "L1", "concepts": [concept], "rule": "1 and 100"}).json()
    q = start["questions"][concept][0]
    assert "answer" not in q  # answers are not leaked to the learner
    res = client.post(f"/evaluation/{start['session_id']}/answer",
                      json={"concept": concept, "answers": {q["id"]: 0}}).json()
    assert res["level"] == "easy" and "category" in res
    assert client.get(f"/evaluation/{start['session_id']}").json()["learner_id"] == "L1"


def test_frontend_and_sample_endpoints(client):
    page = client.get("/")
    assert page.status_code == 200 and "Personalized Learning Assistant" in page.text
    health = client.get("/health").json()
    assert health["obe"] == {"threshold": 50.0, "target": 70.0}


def test_results_upload(client, sample_data):
    with open(sample_data["current"], "rb") as f:
        res = client.post("/results/upload", files={"file": ("results.csv", f, "text/csv")}).json()
    assert len(res["learners"]) == 20
    first = next(iter(res["learners"].values()))[0]
    assert {"concept", "marks", "max_marks", "question_text"} <= set(first)
    bad = client.post("/results/upload", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert bad.status_code == 400


def test_evaluation_records_feed_back(client):
    concept = "String Manipulation and Text Processing"
    start = client.post("/evaluation", json={"learner_id": "L1", "concepts": [concept], "rule": "1 and 100"}).json()
    q = start["questions"][concept][0]
    client.post(f"/evaluation/{start['session_id']}/answer", json={"concept": concept, "answers": {q["id"]: 0}})
    records = client.get(f"/evaluation/{start['session_id']}").json()["records"]
    assert len(records) == 1 and records[0]["concept"] == concept
    body = {"learner_id": "L1", "records": LEARNER["records"] + records}
    assert client.post("/analyze", json=body).status_code == 200
