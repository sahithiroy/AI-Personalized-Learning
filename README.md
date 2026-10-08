# AI-Driven Personalized Learning and Remedial Recommendation

An implementation of **"AI-Driven Personalized Learning and Remedial Recommendation Through
Knowledge Concept-Centric Evaluation"** (N. Pradeesh, M. G. Thushara, K. Arun Krishna, V. Pranav,
S. Krishnamoorthy, *IEEE Access*, vol. 13, 2025, DOI 10.1109/ACCESS.2025.3638427).

The system takes a course curriculum and learners' Outcome-Based Education (OBE) exam results. It then:

1. finds each learner's **knowledge gaps**, using Exercise-aware Knowledge Tracing (EKT) and the OBE threshold and target;
2. writes a **remedial recommendation** for each weak concept with a generative model (OpenAI), cross-checked by Gemini and DeepSeek;
3. **generates new MCQs** from the curriculum with RAG, removes duplicates and cross-checks their Bloom's level and difficulty;
4. runs an **adaptive evaluation** (Easy → Medium → Hard, using the instructor's rule, e.g. "5 and 80");
5. feeds the answers back into knowledge tracing and **repeats until the learner reaches the target** (Algorithm 1 in the paper).

It runs **fully offline without any API keys**. Any model without a key is replaced by a deterministic mock
that returns the same JSON structure, and a synthetic data generator stands in for the institutional
LMS data used in the paper.

```
 curriculum.pdf ──► chunks ──► embeddings ──► vector DB ─────────────┐
                                                                     ▼
 OBE exam CSV ──► EKT knowledge state ──► gap analysis ──► remedial recommendation ──► learner
                        ▲                     │              (cross-verified)
                        │                     ▼
                        │               RAG MCQ generation ──► dedup ──► cross-verify
                        │                     │
                        └──── answers ◄── adaptive evaluation (Easy → Medium → Hard)
```

---

## 1. Setup

Requires Python 3.10+ (tested with 3.13 on Windows).

```bash
cd AI-Personalized-Learning
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy`, `torch`, `pyyaml` and `pypdf` are strictly required. Everything else is optional, and the code falls back automatically when a package is missing:

| Package | Used for | If missing |
|---|---|---|
| `sentence-transformers` | all-MiniLM-L6-v2 embeddings (the paper's model) | offline hashing embedder |
| `faiss-cpu` | vector index | exact numpy index |
| `openai`, `google-generativeai` | real LLMs | offline mock provider |
| `scipy`, `sacrebleu` | t-test, Self-BLEU | built-in implementations |
| `fastapi`, `uvicorn` | REST API | the `serve` command is unavailable |
| `matplotlib` | KT training-curve plot | no plot is saved |

### API keys (optional)

```bash
cp .env.example .env      # then fill in the keys you have
```

**Free setup (the default in `config.yaml`):** one generator and two different checking models, all on free tiers.
Free tiers are rate-limited and their terms change, so check each provider's pricing page.

```
GEMINI_API_KEY=...        # generator: free key from https://aistudio.google.com/apikey
GROQ_API_KEY=...          # verifier 1: free key from https://console.groq.com/keys
OPENROUTER_API_KEY=...    # verifier 2: free key from https://openrouter.ai/keys (uses a ":free" model)
```

**Fully offline and free:** install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and set
`primary: ollama` and `verifiers: [ollama]` in `config.yaml`. No key is needed.

**Paid setup used in the paper:** `primary: openai`, `verifiers: [gemini, deepseek]`, with `OPENAI_API_KEY`,
`GEMINI_API_KEY` and `DEEPSEEK_API_KEY`.

Model names change often. If a provider replies that a model is retired or "no longer available", put the model
name it suggests into `config.yaml` (`gemini_model`, `groq_model`, `openrouter_model`, `ollama_model`).

Any provider without a key uses the mock. The same happens if a real API call fails, as long as
`llm.fallback_to_mock: true` is set in `config.yaml`. If a provider reports an account problem (no credit, invalid key, unknown
model), it is switched off for the rest of the run so the system doesn't wait on a failing call every time; the web page then
shows it as "unavailable, offline mock". To force mocks everywhere, set `PLRS_PRIMARY_LLM=mock`.

> **Note:** the first run downloads the `all-MiniLM-L6-v2` model (~90 MB). On some Windows machines, importing
> `transformers` takes about a minute. For quick experiments, set `PLRS_EMBEDDER=hashing` to skip it.

---

## 2. Quick start: everything in one command

```bash
python -m plrs demo
```

This command:
1. generates synthetic OBE data in `data/sample/` (`historical.csv` for training EKT, `current_batch.csv` for a
   live batch shaped like Table 2: 262 learners, 16 questions, 5 concepts);
2. indexes `data/sample/curriculum_c_programming.txt` into the vector DB and extracts its 5 concepts;
3. trains the EKT model (12 epochs, saved to `data/models/ekt.pt`);
4. prints the knowledge-gap analysis for learner `CSE230001`;
5. runs the full personalized-learning loop for that learner. It prints each round's knowledge state, the
   remedial recommendations and the adaptive-evaluation results, and writes the report to
   `data/store/learners/CSE230001/cycle.md`.

---

## 3. Web frontend

Start the server (after running `python -m plrs demo` once, so the curriculum index, EKT model and sample data exist):

```bash
python -m plrs serve
```

Wait for `Ready: open http://127.0.0.1:8000/`. Loading the models takes about a minute on the first start. Then open
**http://127.0.0.1:8000/** in your browser. The page walks through the same loop as the paper:

1. **Learner exam results.** Pick a sample learner, upload an OBE results CSV (same columns as
   `data/sample/current_batch.csv`; ready-made examples are in `sample_uploads/`), or type the questions and marks
   yourself. Every row can be edited.
2. **Analyse & get recommendations.** Shows the EKT mastery for each concept as a bar with the OBE threshold and
   target marked, the level (Beginner / Intermediate / Expert) and whether it is a knowledge gap.
3. **Remedial recommendations.** One expandable plan per weak concept: learning objectives, topics to revise,
   explanation, practice activities, and why it was recommended. At the top, **View study material** opens a
   personal PDF inside the page and **Download PDF** saves it: the learner's weak concepts (weakest first), the
   questions they got wrong, the plan, the matching syllabus pages and a self-check list. Each plan also has a button
   for a PDF of that concept only. The PDF reuses the recommendations on screen, so it makes no extra AI calls.
4. **Self-assessment.** An adaptive test on the weak concepts (Easy → Medium → Hard, using a rule like `5 and 80`).
   After you submit a level, the right and wrong answers are shown with explanations.
5. **Update my knowledge state → next round.** Your answers are added to your history, EKT re-traces your
   knowledge, and you get new recommendations. A table shows your mastery across rounds.

Use `python -m plrs serve --port 9000` for another port, and `--host 0.0.0.0` to open it to other machines on your network.

---

## 4. Running each step yourself

All commands are `python -m plrs <command>`. Add `--help` to any command to see its options, or put `-v INFO` before the command for progress logs.

| Step (paper section) | Command |
|---|---|
| Generate synthetic OBE data | `python -m plrs sample-data [--historical 600 --current 262]` |
| **Step 1** Ingest the curriculum (III-A, App. A) | `python -m plrs ingest path/to/curriculum.pdf` |
| Review the extracted concepts | `python -m plrs concepts` (edit `data/store/concepts.json`) |
| Semantic search over the curriculum | `python -m plrs query "pointer arithmetic"` |
| Train knowledge tracing (III-B.1, V-B) | `python -m plrs train-kt --data data/sample/historical.csv --model ekt` (`dkt` / `all`) |
| **Step 2** Knowledge-gap analysis (III-B) | `python -m plrs analyze --learner CSE230001` |
| Class-wide knowledge-gap matrix (IV-B.2) | `python -m plrs gap-matrix` |
| **Step 3** Remedial recommendation (III-C) | `python -m plrs recommend --learner CSE230001 --out plan.md` |
| **Step 4** MCQ generation (III-D) | `python -m plrs generate-mcq --concept "String Manipulation and Text Processing" --level easy --n 5` |
| **Step 5** Adaptive evaluation, interactive (III-E) | `python -m plrs evaluate --learner me --rule "5 and 80" --save my_answers.csv` |
| Full loop, Algorithm 1 (III-F) | `python -m plrs cycle --learner CSE230001 --rounds 3 --verbose` |
| Reproduce the evaluation tables (V) | `python -m plrs experiments` |
| Web frontend + REST API | `python -m plrs serve` then open http://127.0.0.1:8000/ (API docs at `/docs`) |

`cycle` answers the questions with a **simulated learner** whose ability is estimated from their OBE scores,
and studying a recommendation raises that ability. Use `evaluate` to answer the questions yourself.

---

## 5. Using your own data

### Curriculum
Any PDF or text file works. Sections are detected from lines such as `Unit 1: <concept name>`
(also `Module` and `Chapter`), and course outcomes from lines such as `CO1: ...`. After running `ingest`,
open `data/store/concepts.json`, then correct and approve the concept list. This is the instructor review step in the paper.
Re-running `ingest --no-extract` keeps your edited concept list.

### OBE exam results (CSV)
Each row is one learner's answer to one question:

| column | example | notes |
|---|---|---|
| `learner_id` | `CSE230001` | |
| `exam` | `Internal` | the "recent attempts" in a recommendation come from the learner's latest exam |
| `order` | `3` | chronological order within the learner's history |
| `question_id` | `Q24` | |
| `question_text` | `Write a switch statement ...` | EKT embeds this text |
| `concept` | `Operators, Pointers and Control Structures` | **must match a concept name in `concepts.json`** |
| `course_outcome` | `CO2` | |
| `difficulty` | `easy` / `medium` / `hard` | |
| `bloom_level` | `Apply` | |
| `max_marks`, `marks` | `5`, `3.5` | a question counts as *attained* when `marks / max_marks >= OBE threshold` |

Train on historical data with `train-kt --data <file>`, then run `analyze`, `recommend` or `cycle --data <file>` on the current batch.

---

## 6. Configuration (`config.yaml`)

The defaults follow the values reported in the paper:

| Key | Default | Paper |
|---|---|---|
| `obe.threshold` / `obe.target` | 50 / 70 | Sec. III-A.1 (threshold 50–60 %, target about 20 % higher) |
| `rag.chunk_size` / `chunk_overlap` / `top_k` | 1000 / 100 / 4 | Appendix A |
| `rag.embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Sec. V-A (`hashing` = offline) |
| `mcq.generation_multiplier` | 3 | Appendix A (generate about 3× the required questions) |
| `mcq.dedup_threshold` | 0.85 | Sec. III-D.1 (cosine cut-off) |
| `mcq.verification_min_score` | 3 | minimum average verifier score (1–5) to accept a question |
| `evaluation.default_rule` | 5 questions, 80 % | Sec. IV-A.4 ("5 and 80") |
| `ekt.*` | emb 100, LSTM 100, dropout 0.3, Adam 1e-3, batch 32, 12 epochs, 75/25 split | Sec. V-B |
| `ekt.use_attention` | true | EKTA variant (attention by exercise similarity) |
| `llm.primary` / `llm.verifiers` | openai / [gemini, deepseek] | Sec. V-A.1 |

Environment overrides: `PLRS_CONFIG` (another YAML file), `PLRS_PRIMARY_LLM`, `PLRS_EMBEDDER`.

---

## 7. How the code maps to the paper

| Paper | Module | What it does |
|---|---|---|
| Fig. 1 Input/Output module, App. A | [plrs/rag.py](plrs/rag.py) | PDF → section-aware overlapping chunks → embeddings → FAISS/numpy vector DB; LLM concept extraction; retrieval |
| Sec. III-B.1, V-B (EKT) | [plrs/kt.py](plrs/kt.py) | `EKT` (exercise-text embedding + concept embedding → response-aware LSTM → mastery head `σ(W hₜ+b)`, attention over similar past exercises) and a `DKT` baseline; AUC/MAE/loss; knowledge-state inference |
| Sec. III-A.1, III-B (OBE gaps) | [plrs/gap.py](plrs/gap.py) | Beginner / Intermediate / Expert against the threshold and target; gap flags; gap matrix |
| Sec. III-C (remedial) | [plrs/remedial.py](plrs/remedial.py) | the paper's five-section prompt, cross-verification, regeneration from reviewer feedback, Figure-2-style markdown |
| Sec. III-D (MCQ, RAG) | [plrs/qgen.py](plrs/qgen.py) | RAG prompt at the required Bloom's level, 3× generation, refinement, cosine de-duplication against a persistent global memory, multi-model cross-verification |
| Sec. III-E (adaptive eval) | [plrs/evaluation.py](plrs/evaluation.py) | Easy → Medium → Hard progression, "N and P%" rule, Not Competent / Beginner / Intermediate / Expert |
| Sec. III-F, Algorithm 1 | [plrs/pipeline.py](plrs/pipeline.py) | closed loop: analyse → recommend → evaluate → re-trace |
| Sec. IV-B, V | [plrs/metrics.py](plrs/metrics.py), [plrs/experiments.py](plrs/experiments.py) | S-BERT similarity, Flesch-Kincaid, Self-BLEU diversity, relevance 1–5, paired t-test; tables for Tables 1–6 and Fig. 7 |
| LLM providers | [plrs/llm.py](plrs/llm.py) | OpenAI, Gemini and DeepSeek (OpenAI-compatible), plus the offline mock |
| Data | [plrs/data.py](plrs/data.py) | CSV schema, dataset statistics (Tables 1/2), IRT-based synthetic learners |
| Interfaces | [plrs/cli.py](plrs/cli.py), [plrs/api.py](plrs/api.py), [plrs/static/index.html](plrs/static/index.html) | command line, FastAPI and the web frontend |

---

## 8. Experiments

```bash
python -m plrs experiments                      # all parts
python -m plrs experiments --which kt --epochs 12
```

| Part | Paper | Output |
|---|---|---|
| `data` | Tables 1, 2 and the gap matrix | dataset statistics |
| `kt` | Table 5, Fig. 4 | final train/test loss, AUC, MAE, accuracy, precision, recall and F1 for EKT vs DKT; per-epoch history in `data/models/kt_history.csv`, plot in `kt_curves.png` |
| `mcq` | Tables 3, 4 | questions from each generator model compared with the teacher-written questions: semantic similarity, readability, diversity, relevance |
| `remedial` | Fig. 7, Table 6 | competency distribution before and after remediation, plus a paired t-test for each concept (30 learners) |

The results are saved to `data/models/experiments.json`.

**Example run** (bundled synthetic data, mock LLMs, MiniLM embeddings):

| | Test loss | Test AUC | Test MAE |
|---|---|---|---|
| EKT | 0.504 | **0.827** | 0.331 |
| DKT | 0.637 | 0.665 | 0.447 |

In the simulated remediation study (30 learners), the mean concept score rose on all five concepts
(e.g. String Manipulation went from 66.7 to 92.5, p < 0.001). With the offline mock, MCQ relevance is low (about 1/5),
because the mock copies curriculum sentences instead of writing teacher-style questions. Real LLM keys are needed for meaningful Table 3/4 numbers.

> **Important:** with the bundled synthetic data and mock models, these numbers only show that the pipeline
> works. They are **not** the paper's results. In the `remedial` part, both the learners and their learning
> gain are simulated. Real numbers need real OBE exports, real API keys and real pre/post tests.

---

## 9. REST API

```bash
python -m plrs serve            # http://127.0.0.1:8000/docs for the interactive docs
```

| Method | Path | Body |
|---|---|---|
| GET | `/` | the web frontend (`plrs/static/index.html`) |
| GET | `/health` | shows the active LLMs, KT model, OBE threshold and target |
| GET | `/sample-learners`, `/sample-learners/{id}` | learners and records from `data/sample/current_batch.csv` |
| POST | `/results/upload` | multipart CSV upload; returns records grouped by learner |
| POST | `/curriculum` | multipart file upload (PDF/TXT); builds the vector DB and extracts concepts |
| GET | `/concepts` | |
| POST | `/analyze` | `{"learner_id": "...", "records": [{question_id, question_text, concept, difficulty, max_marks, marks}]}` |
| POST | `/recommend` | same body as `/analyze`; returns JSON and markdown recommendations |
| POST | `/materials/pdf` | `{learner_id, concepts, recommendations}` (outputs of `/analyze` and `/recommend`); returns a study-material PDF: weakest concept first, missed questions, the plan and the matching syllabus pages |
| POST | `/mcq` | `{"concept": "...", "level": "easy", "n": 5}` |
| POST | `/evaluation` | `{"learner_id": "...", "concepts": [...], "rule": "5 and 80"}`; returns a session id and the first questions (without answers) |
| POST | `/evaluation/{id}/answer` | `{"concept": "...", "answers": {"<question id>": 2}}`; grades the level and returns the next level's questions |
| GET | `/evaluation/{id}` | session report, including the answers as records to append for the next `/analyze` |

---

## 10. Tests

```bash
python -m pytest -q
```

The tests use a temporary store, the offline embedder and the mock LLMs, so they never call external APIs, even if `.env` contains keys.

## 11. Project layout

```
config.yaml                 all parameters
.env.example                API key template
data/sample/                sample curriculum (+ generated CSVs)
data/store/                 vector DB, concepts.json, question_bank.json, learner reports   (generated)
data/models/                ekt.pt / dkt.pt, kt_history.csv, experiments.json               (generated)
docs/                       Project_Report.pdf / .docx (final-year report), report_source/ (scripts that build it),
                            Project_Step_by_Step_Guide.pdf (plain-language guide and review Q&A)
PPTS/                       First and Final Review slide decks
sample_uploads/             example OBE result CSVs for the web upload, and an example study-material PDF
plrs/                       the package (see section 7)
plrs/static/index.html      web frontend (plain HTML + JavaScript, no build step)
tests/                      pytest suite
```

## 12. Project report

The complete final-year project report is in `docs/`, in two formats with the same content:

- `docs/Project_Report.pdf`: ready to print or submit.
- `docs/Project_Report.docx`: editable in Word. Fill in the placeholders in square brackets (student names, roll numbers,
  college, guide, logos), then right-click the Table of Contents and choose **Update Field** (Word also offers to update
  fields when the file opens) so the contents, list of figures and list of tables show page numbers.

A 6-page **research paper** in IEEE two-column format is in `docs/Research_Paper.pdf` and `docs/Research_Paper.docx`.
It reports new experiments: EKT ablation over 3 seeds, calibration of the mastery output, the exam-ordering
artifact, retrieval purity, and a 40-learner closed-loop simulation. `docs/paper_source/` holds the scripts that
produced it, and `paper_results.json` holds the raw numbers.

All result numbers in the report are read from this project's own output files. `docs/report_source/` holds the scripts
that rebuild it after new experiments: `make_figures.py` (charts and diagrams), `make_screens.py` and `make_screens_materials.py` (screenshots of the
running web app), `content.py` (the text), `render_pdf.py` and `render_docx.js`.

## 13. Limitations

- Only **EKT and DKT** are implemented. The paper also compares DKVMN, AKT and SimpleKT.
- The **mock provider** builds questions and recommendations from sentences in the curriculum. It is useful for
  testing the pipeline offline, but the content quality depends on real models (Groq, OpenRouter and Gemini by
  default; OpenAI or DeepSeek optionally).
- The bundled data is **synthetic** (IRT-simulated learners), not the AMPLE LMS data used in the paper.
- The mock verifiers apply structural checks only. Real pedagogical verification needs the Gemini and DeepSeek keys,
  plus instructor review of flagged items, as the paper recommends.
