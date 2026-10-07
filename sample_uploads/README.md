# Sample upload files

Use these with the **Upload CSV** tab in the web app (`python -m plrs serve`, then http://127.0.0.1:8000/).
They use the same columns as `data/sample/current_batch.csv` and the questions from its question bank.

| File | Learners | What it shows |
|---|---|---|
| `1_single_student_weak_in_strings.csv` | STU001 | Strong overall; Strings (CO4) 0%, Data Structures (CO5) 61% -> two gaps |
| `2_single_student_strong.csv` | STU002 | 83-100% everywhere -> few or no gaps |
| `3_single_student_struggling.csv` | STU003 | 0-17% everywhere -> gaps in all five concepts |
| `4_class_of_5_students.csv` | STU101-STU105 | One file, five profiles; pick the learner from the drop-down after upload |

Class file profiles (percentage of marks, CO1 to CO5):

| Learner | CO1 | CO2 | CO3 | CO4 | CO5 | Pattern |
|---|---|---|---|---|---|---|
| STU101 | 100 | 87 | 83 | 0 | 61 | Weak in strings |
| STU102 | 100 | 100 | 83 | 100 | 87 | Strong |
| STU103 | 17 | 0 | 11 | 0 | 17 | Struggling |
| STU104 | 87 | 17 | 0 | 100 | 87 | Weak in pointers and arrays |
| STU105 | 61 | 61 | 48 | 61 | 61 | Average (Intermediate) everywhere |

## CSV format

One row = one learner's answer to one exam question.

```
learner_id,exam,order,question_id,question_text,concept,course_outcome,difficulty,bloom_level,max_marks,marks
```

`order` is the answer sequence (used by knowledge tracing), `marks` is out of `max_marks`.
To make your own file, copy one of these and change `learner_id` and `marks`.

## Curriculum file

The syllabus for the demo course is `data/sample/curriculum_c_programming.txt`.
Load it with `python -m plrs ingest data/sample/curriculum_c_programming.txt`.
