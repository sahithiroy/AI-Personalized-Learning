"""OBE exam data: record schema, CSV I/O and a synthetic data generator.

CSV columns (one row = one learner's answer to one exam question)::

    learner_id, exam, order, question_id, question_text, concept, course_outcome,
    difficulty, bloom_level, max_marks, marks

A question counts as *attained* (1) when ``marks / max_marks * 100 >= OBE threshold``
(paper Appendix B: "concept-level attainment (1-pass/0-fail)").

The synthetic generator stands in for the AMPLE LMS exports used in the paper
(course "Procedural Programming using C", 5 concepts, 16 exam questions). It
simulates learners with an IRT-style model so the whole pipeline can be run and
tested without institutional data.
"""
from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .config import PROJECT_ROOT, get_config

SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

C_CONCEPTS = [
    "Fundamentals and System Architecture",
    "Operators, Pointers and Control Structures",
    "Algorithmic Problem Solving and Array Manipulation",
    "String Manipulation and Text Processing",
    "Data Structures, Memory Management, and File Handling",
]


@dataclass
class Record:
    learner_id: str
    exam: str
    order: int
    question_id: str
    question_text: str
    concept: str
    course_outcome: str
    difficulty: str
    bloom_level: str
    max_marks: float
    marks: float

    @property
    def percent(self) -> float:
        return 100.0 * self.marks / self.max_marks if self.max_marks else 0.0

    def attained(self, threshold: float | None = None) -> int:
        t = get_config()["obe"]["threshold"] if threshold is None else threshold
        return int(self.percent >= t)


def load_records(path: str | Path) -> list[Record]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append(Record(
            learner_id=r["learner_id"], exam=r.get("exam", ""), order=int(r.get("order") or 0),
            question_id=r["question_id"], question_text=r.get("question_text", ""),
            concept=r["concept"], course_outcome=r.get("course_outcome", ""),
            difficulty=r.get("difficulty", "medium"), bloom_level=r.get("bloom_level", ""),
            max_marks=float(r["max_marks"]), marks=float(r["marks"]),
        ))
    return out


def save_records(records: list[Record], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[fl.name for fl in fields(Record)])
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    return path


def by_learner(records: list[Record]) -> dict[str, list[Record]]:
    """Group records per learner, ordered chronologically."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        groups[r.learner_id].append(r)
    for rs in groups.values():
        rs.sort(key=lambda r: r.order)
    return dict(groups)


def dataset_summary(records: list[Record]) -> dict:
    """Same statistics as Tables 1 and 2 of the paper."""
    groups = by_learner(records)
    n = max(len(groups), 1)
    return {
        "Concepts": len({r.concept for r in records}),
        "Learners": len(groups),
        "Unique Questions": len({r.question_id for r in records}),
        "# Records": len(records),
        "Avg. questions attempted per learner": round(len(records) / n, 2),
        "Avg. concepts attempted per learner": round(sum(len({r.concept for r in rs}) for rs in groups.values()) / n, 2),
        "Avg. attainment percentage for learner": round(
            100 * sum(sum(r.attained() for r in rs) / len(rs) for rs in groups.values()) / n, 2),
    }


# --------------------------------------------------------------------------- synthetic data
QUESTION_BANK: dict[str, list[tuple[str, str]]] = {
    C_CONCEPTS[0]: [
        ("easy", "List the stages a C program passes through from source code to executable."),
        ("easy", "What is the role of the main() function in a C program?"),
        ("medium", "Explain how the compiler, assembler and linker cooperate to build a program."),
        ("medium", "Describe the von Neumann architecture and the role of the CPU and memory."),
        ("hard", "Analyse why a program that compiles without errors can still fail at link time."),
        ("hard", "Evaluate the trade-offs of storing variables in registers versus main memory."),
    ],
    C_CONCEPTS[1]: [
        ("easy", "What is the output of printf(\"%d\", 7 / 2) in C?"),
        ("easy", "Which operator is used to obtain the address of a variable?"),
        ("medium", "Trace the value of x after: int x = 5; x += x++ * 2; and explain the pitfall."),
        ("medium", "Write a switch statement that maps a digit to its English word."),
        ("hard", "Explain how pointer arithmetic on an int* differs from a char* on the same address."),
        ("hard", "Design a loop that uses pointers only (no indexing) to reverse an integer array."),
    ],
    C_CONCEPTS[2]: [
        ("easy", "How are elements of a one-dimensional array indexed in C?"),
        ("easy", "Write a loop that prints every element of an integer array."),
        ("medium", "Write a program to find the largest number in an array."),
        ("medium", "Insert a number at a given position in an array, shifting later elements."),
        ("hard", "Find the top three elements of an array without fully sorting it."),
        ("hard", "Compare bubble sort and selection sort in terms of swaps and comparisons."),
    ],
    C_CONCEPTS[3]: [
        ("easy", "How is the end of a string marked in C?"),
        ("easy", "Which header file declares strlen, strcpy and strcmp?"),
        ("medium", "Write a function that counts the vowels in a string."),
        ("medium", "Explain why comparing two strings with == is incorrect in C."),
        ("hard", "Implement a function that checks whether a string is a palindrome ignoring case."),
        ("hard", "Write a word-frequency counter for a line of text."),
    ],
    C_CONCEPTS[4]: [
        ("easy", "What does malloc return when the allocation fails?"),
        ("easy", "Which function opens a file for reading in C?"),
        ("medium", "Define a struct for a student record and print an array of such records."),
        ("medium", "Explain the difference between stack and heap allocation."),
        ("hard", "Implement insertion at the head of a singly linked list using dynamic memory."),
        ("hard", "Diagnose the memory leak in a loop that calls malloc without free."),
    ],
}

_BLOOM = {"easy": "Understand", "medium": "Apply", "hard": "Analyze"}
_DIFF = {"easy": -1.0, "medium": 0.0, "hard": 1.0}
_MARKS = {"easy": 2.0, "medium": 5.0, "hard": 10.0}


def question_bank() -> list[dict]:
    bank = []
    for ci, concept in enumerate(C_CONCEPTS):
        for qi, (diff, text) in enumerate(QUESTION_BANK[concept]):
            bank.append({"question_id": f"Q{ci + 1}{qi + 1}", "question_text": text, "concept": concept,
                         "course_outcome": f"CO{ci + 1}", "difficulty": diff, "bloom_level": _BLOOM[diff]})
    return bank


@dataclass
class SimLearner:
    """Latent ability per concept; P(correct) = sigmoid(1.7 * (theta - b))."""

    learner_id: str
    theta: dict[str, float]

    def p_correct(self, concept: str, difficulty: str) -> float:
        return 1 / (1 + math.exp(-1.7 * (self.theta[concept] - _DIFF[difficulty])))

    def answer(self, concept: str, difficulty: str, rng: random.Random) -> bool:
        correct = rng.random() < self.p_correct(concept, difficulty)
        self.theta[concept] += 0.04  # practice effect
        return correct


def make_learners(n: int, rng: random.Random, prefix: str = "L") -> list[SimLearner]:
    base = {c: rng.uniform(-0.6, 0.6) for c in C_CONCEPTS}  # cohort-level concept difficulty
    learners = []
    for i in range(n):
        g = rng.gauss(0.3, 0.8)  # general ability
        learners.append(SimLearner(f"{prefix}{i + 1:04d}",
                                   {c: g + base[c] + rng.gauss(0, 0.6) for c in C_CONCEPTS}))
    return learners


def _marks(correct: bool, max_marks: float, rng: random.Random) -> float:
    frac = rng.uniform(0.6, 1.0) if correct else rng.uniform(0.0, 0.45)
    return round(max_marks * frac * 2) / 2


def simulate_exam(learners: list[SimLearner], questions: list[dict], per_learner: int,
                  rng: random.Random, exam: str = "Internal", start_order: int = 0) -> list[Record]:
    records = []
    for lr in learners:
        chosen = rng.sample(questions, min(per_learner, len(questions)))  # random order within the exam
        for k, q in enumerate(chosen):
            correct = lr.answer(q["concept"], q["difficulty"], rng)
            mm = _MARKS[q["difficulty"]]
            records.append(Record(lr.learner_id, exam, start_order + k, q["question_id"], q["question_text"],
                                  q["concept"], q["course_outcome"], q["difficulty"], q["bloom_level"],
                                  mm, _marks(correct, mm, rng)))
    return records


def generate_sample_data(out_dir: str | Path = SAMPLE_DIR, n_historical: int = 600,
                         n_current: int = 262, seed: int = 7) -> dict[str, Path]:
    """Write historical.csv (KT training) and current_batch.csv (live batch, Table 2 shape)."""
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    bank = question_bank()
    # Historical learners: long sequences over the whole bank (several exams)
    hist = []
    for lr in make_learners(n_historical, rng, "H"):
        order = 0
        for rep in range(2):
            recs = simulate_exam([lr], bank, per_learner=20, rng=rng, exam=f"Exam{rep + 1}", start_order=order)
            order += len(recs)
            hist += recs
    # Current batch: 16 exam questions over 5 concepts (3 each + 1), ~9 attempted per learner
    exam_questions = [q for q in bank if q["question_id"][-1] in "123"]
    exam_questions += [q for q in bank if q["question_id"] == "Q34"]
    current = make_learners(n_current, rng, "CSE23")
    cur = simulate_exam(current, exam_questions, per_learner=9, rng=rng, exam="Internal")
    return {
        "historical": save_records(hist, out_dir / "historical.csv"),
        "current": save_records(cur, out_dir / "current_batch.csv"),
    }
