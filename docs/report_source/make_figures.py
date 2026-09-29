"""Diagrams and result charts for the project report. Charts read the project's real result files."""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

NAVY, BLUE, SOFT, GREY, LINE = "#16233A", "#2A5FC0", "#E6EEFC", "#5F6878", "#C5CCD9"
ORANGE, ORANGE_SOFT, PAPER, LIGHT = "#A9540F", "#FBF1DF", "#F7F6F1", "#A9B7CF"
plt.rcParams.update({"font.family": "Arial", "font.size": 11, "axes.edgecolor": GREY,
                     "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY})

SHORT = {
    "Algorithmic Problem Solving and Array Manipulation": "Algorithmic problem\nsolving & arrays",
    "Data Structures, Memory Management, and File Handling": "Data structures,\nmemory & files",
    "Fundamentals and System Architecture": "Fundamentals &\nsystem architecture",
    "Operators, Pointers and Control Structures": "Operators, pointers\n& control",
    "String Manipulation and Text Processing": "Strings & text\nprocessing",
}
ORDER = list(SHORT)


# ------------------------------------------------------------------ diagram helpers
class Diagram:
    def __init__(self, w, h):
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.ax.set_xlim(0, w * 10)
        self.ax.set_ylim(0, h * 10)
        self.ax.axis("off")
        self.boxes = {}

    def box(self, key, x, y, w, h, title, sub=None, fill="white", edge=LINE, tcolor=NAVY, size=11):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.2",
                                         fc=fill, ec=edge, lw=1.4))
        if sub:
            self.ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=size,
                         fontweight="bold", color=tcolor, wrap=True)
            self.ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=size - 2,
                         color=GREY if tcolor == NAVY else tcolor)
        else:
            self.ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=size,
                         fontweight="bold", color=tcolor)
        self.boxes[key] = (x, y, w, h)

    def _pt(self, key, side):
        x, y, w, h = self.boxes[key]
        return {"l": (x, y + h / 2), "r": (x + w, y + h / 2), "t": (x + w / 2, y + h),
                "b": (x + w / 2, y)}[side]

    def arrow(self, a, sa, b, sb, color=BLUE, style="-", label=None, rad=0.0, lpos=0.5):
        p1, p2 = self._pt(a, sa), self._pt(b, sb)
        self.ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, color=color, lw=1.6,
                                          linestyle=style, connectionstyle=f"arc3,rad={rad}"))
        if label:
            mx, my = p1[0] + (p2[0] - p1[0]) * lpos, p1[1] + (p2[1] - p1[1]) * lpos
            self.ax.text(mx + 0.8, my + 0.8, label, fontsize=9, color=color, fontweight="bold")

    def path(self, pts, color=ORANGE, style="--", label=None, lxy=None):
        xs, ys = zip(*pts)
        self.ax.plot(xs[:-1] + (xs[-1],), ys[:-1] + (ys[-1],), color=color, lw=1.6, linestyle=style)
        self.ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=16, color=color,
                                          lw=1.6))
        if label:
            self.ax.text(*lxy, label, fontsize=9.5, color=color, fontweight="bold")

    def save(self, name):
        self.fig.tight_layout()
        self.fig.savefig(OUT / name, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(self.fig)


def vertical_flow(name, steps, w=6.2, highlight=(), loop=None):
    """steps: list of (title, sub). loop: (from_index, to_index, label)."""
    n = len(steps)
    h = n * 1.25 + 0.4
    d = Diagram(w, h)
    bw, bh, x = 36, 8.5, (w * 10 - 36) / 2
    for i, (t, s) in enumerate(steps):
        y = h * 10 - 3 - (i + 1) * 12.5 + 2
        hl = i in highlight
        d.box(i, x, y, bw, bh, t, s, fill=SOFT if hl else "white", edge=BLUE if hl else LINE)
        if i:
            d.arrow(i - 1, "b", i, "t")
    if loop:
        a, b, label = loop
        xa, ya, _, ha = d.boxes[a]
        xb, yb, _, hb = d.boxes[b]
        right = x + bw + 6
        d.path([(x + bw, ya + ha / 2), (right, ya + ha / 2), (right, yb + hb / 2), (x + bw, yb + hb / 2)],
               label=label, lxy=(right + 0.8, (ya + yb) / 2 + 4))
    d.save(name)


# ------------------------------------------------------------------ diagrams
def existing_system():
    vertical_flow("fig_existing.png", [("Student", "same for everyone"), ("Common study material", None),
                                       ("Common test", "same questions for all"), ("Total marks", None),
                                       ("Generic feedback", "no concept-level detail")], w=5.2)


def proposed_system():
    vertical_flow("fig_proposed.png", [
        ("Course PDF", "curriculum uploaded by teacher"), ("PDF processing", "text extraction (pypdf)"),
        ("Concept extraction", "one concept per unit (LLM)"), ("RAG vector database", "chunks + embeddings"),
        ("OBE exam results / answers", "marks per question"), ("Knowledge tracing (EKT)", "mastery per concept"),
        ("Identify weak concepts", "mastery < 70% target"), ("Remedial recommendation", "5-part plan, cross-verified"),
        ("Question generation (RAG)", "new concept MCQs"), ("Adaptive re-test", "Easy > Medium > Hard"),
        ("Updated knowledge state", "progress report")],
        w=7.0, highlight=(5, 7), loop=(9, 5, "answers fed back"))


def architecture():
    d = Diagram(9.6, 7.4)
    W = 96
    layers = [("Users", [("Teacher / Admin", "uploads curriculum, sets rules"), ("Learner", "takes tests, reads plans")], 62),
              ("Interfaces", [("Web frontend", "static/index.html"), ("REST API", "api.py (FastAPI)"),
                              ("Command line", "cli.py")], 49),
              ("Orchestration", [("Personalized learning pipeline", "pipeline.py - Algorithm 1 loop")], 36),
              ("Modules", [("Curriculum RAG", "rag.py"), ("Knowledge tracing", "kt.py"), ("Gap analysis", "gap.py"),
                           ("Remedial plans", "remedial.py"), ("MCQ generation", "qgen.py"),
                           ("Adaptive test", "evaluation.py")], 21),
              ("Foundation", [("LLM providers", "llm.py"), ("Embeddings", "embeddings.py"), ("OBE data", "data.py"),
                              ("Metrics", "metrics.py"), ("Settings", "config.yaml")], 6)]
    for label, items, y in layers:
        d.ax.text(1, y + 11.5, label.upper(), fontsize=9, fontweight="bold", color=GREY)
        n = len(items)
        gap = 1.6
        bw = (W - 2 - gap * (n - 1)) / n
        for i, (t, s) in enumerate(items):
            mod = label == "Modules"
            d.box(f"{label}{i}", 1 + i * (bw + gap), y, bw, 10, t, s, fill=SOFT if mod else "white",
                  edge=BLUE if mod else LINE, size=10 if n < 6 else 9)
    for a, b in [("Users0", "Interfaces0"), ("Users1", "Interfaces0"), ("Interfaces1", "Orchestration0"),
                 ("Orchestration0", "Modules2")]:
        d.arrow(a, "b", b, "t")
    d.ax.add_patch(FancyBboxPatch((70, 49), 26, 10, boxstyle="round,pad=0.02,rounding_size=1.2",
                                  fc="none", ec="none"))
    d.save("fig_architecture.png")


def rag_pipeline():
    d = Diagram(9.6, 3.2)
    steps = [("Course material", "PDF / text"), ("Text chunks", "by unit, 1000 chars"), ("Embeddings", "MiniLM, 384-d"),
             ("Vector DB", "FAISS"), ("Retrieve", "concept's own unit"), ("LLM", "question / plan")]
    bw, gap = 13.6, 2.8
    for i, (t, s) in enumerate(steps):
        d.box(i, 1 + i * (bw + gap), 11, bw, 11, t, s, fill=SOFT if i in (3, 5) else "white",
              edge=BLUE if i in (3, 5) else LINE, size=10)
        if i:
            d.arrow(i - 1, "r", i, "l")
    d.ax.text(48, 4, "The LLM only sees retrieved syllabus text, so generated content stays inside the curriculum.",
              ha="center", fontsize=10, color=GREY)
    d.save("fig_rag.png")


def ekt_model():
    d = Diagram(9.6, 5.0)
    d.box("q", 1, 34, 20, 10, "Question text", "MiniLM embedding (384)")
    d.box("c", 1, 20, 20, 10, "Concept id", "concept embedding (100)")
    d.box("r", 1, 6, 20, 10, "Response", "right / wrong")
    d.box("e", 27, 27, 17, 10, "Linear + tanh", "exercise vector (100)")
    d.box("x", 27, 8, 17, 12, "Response split", "[e, 0] if right\n[0, e] if wrong")
    d.box("l", 50, 15, 16, 14, "LSTM", "100 hidden units\nknowledge state h", fill=SOFT, edge=BLUE)
    d.box("m", 73, 30, 22, 11, "Mastery head", "sigmoid(W h + b)\none value per concept", fill=SOFT, edge=BLUE)
    d.box("p", 73, 6, 22, 13, "Next-answer head", "attention over similar\npast questions", fill="white")
    d.arrow("q", "r", "e", "l")
    d.arrow("c", "r", "e", "l")
    d.arrow("e", "b", "x", "t")
    d.arrow("r", "r", "x", "l")
    d.arrow("x", "r", "l", "l")
    d.arrow("l", "r", "m", "l")
    d.arrow("l", "r", "p", "l")
    d.save("fig_ekt.png")


def remedial_flow():
    d = Diagram(9.6, 3.6)
    items = [("Weak concept", "mastery + attempts"), ("Retrieve context", "concept's unit (RAG)"),
             ("Generate plan", "OpenAI, 5 sections"), ("Cross-verify", "Gemini + DeepSeek, 1-5"),
             ("Deliver to learner", "markdown / web page")]
    bw, gap = 16.6, 2.8
    for i, (t, s) in enumerate(items):
        d.box(i, 1 + i * (bw + gap), 17, bw, 11, t, s, fill=SOFT if i == 3 else "white",
              edge=BLUE if i == 3 else LINE, size=10)
        if i:
            d.arrow(i - 1, "r", i, "l")
    x3 = d.boxes[3]
    x2 = d.boxes[2]
    d.path([(x3[0] + x3[2] / 2, 17), (x3[0] + x3[2] / 2, 8), (x2[0] + x2[2] / 2, 8), (x2[0] + x2[2] / 2, 17)],
           label="average score < 3: regenerate with feedback (max 2 retries)", lxy=(x2[0] - 12, 3))
    d.save("fig_remedial.png")


def mcq_pipeline():
    d = Diagram(9.6, 3.6)
    items = [("Retrieve", "concept's unit"), ("Generate 3x", "at Bloom level"), ("Refine", "grammar, format"),
             ("De-duplicate", "cosine > 0.85"), ("Cross-verify", "score >= 3"), ("Question bank", "JSON memory")]
    bw, gap = 13.6, 2.8
    for i, (t, s) in enumerate(items):
        d.box(i, 1 + i * (bw + gap), 17, bw, 11, t, s, fill=SOFT if i == 5 else "white",
              edge=BLUE if i == 5 else LINE, size=10)
        if i:
            d.arrow(i - 1, "r", i, "l")
    for i, lbl in [(3, "duplicate: dropped"), (4, "low score: rejected")]:
        x, y, w, h = d.boxes[i]
        d.ax.annotate("", xy=(x + w / 2, 6), xytext=(x + w / 2, 17),
                      arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
        d.ax.text(x + w / 2, 3, lbl, ha="center", fontsize=9, color=ORANGE, fontweight="bold")
    d.save("fig_mcq.png")


def adaptive_flow():
    d = Diagram(9.6, 3.8)
    d.box("s", 1, 20, 12, 10, "Start", None)
    d.box("e", 19, 20, 16, 10, "Easy level", "Remember / Understand")
    d.box("m", 42, 20, 16, 10, "Medium level", "Apply / Analyze")
    d.box("h", 65, 20, 16, 10, "Hard level", "Evaluate / Create")
    d.box("x", 86, 20, 9, 10, "Expert", None, fill="#E3F4EA", edge="#1B6B3C", tcolor="#1B6B3C")
    d.box("nc", 19, 2, 16, 9, "Not Competent", None, fill="#FBE6E3", edge="#8E2A1E", tcolor="#8E2A1E", size=10)
    d.box("b", 42, 2, 16, 9, "Beginner", None, fill=ORANGE_SOFT, edge=ORANGE, tcolor=ORANGE, size=10)
    d.box("i", 65, 2, 16, 9, "Intermediate", None, fill=ORANGE_SOFT, edge=ORANGE, tcolor=ORANGE, size=10)
    d.arrow("s", "r", "e", "l")
    d.arrow("e", "r", "m", "l", label="pass")
    d.arrow("m", "r", "h", "l", label="pass")
    d.arrow("h", "r", "x", "l", label="pass")
    for a, b in [("e", "nc"), ("m", "b"), ("h", "i")]:
        d.arrow(a, "b", b, "t", color=ORANGE, label="fail")
    d.ax.text(48, 34, 'Pass = at least the rule\'s percentage correct, e.g. "5 and 80" = 4 of 5', ha="center",
              fontsize=10, color=GREY)
    d.save("fig_adaptive.png")


def concept_tree():
    d = Diagram(9.6, 4.2)
    d.box("root", 33, 30, 30, 9, "Procedural Programming using C", None, fill=SOFT, edge=BLUE)
    names = [("CO1", "Fundamentals &\nsystem architecture"), ("CO2", "Operators, pointers\n& control structures"),
             ("CO3", "Algorithmic problem\nsolving & arrays"), ("CO4", "String manipulation\n& text processing"),
             ("CO5", "Data structures,\nmemory & files")]
    bw, gap = 17.2, 2.0
    for i, (co, n) in enumerate(names):
        d.box(i, 1 + i * (bw + gap), 4, bw, 14, n, co, size=9)
        d.arrow("root", "b", i, "t", color=GREY)
    d.save("fig_concepts.png")


def data_design():
    d = Diagram(9.6, 6.0)
    tables = [
        ("records", 1, 30, "OBE exam record (CSV)", ["learner_id", "exam, order", "question_id, question_text",
                                                     "concept, course_outcome", "difficulty, bloom_level",
                                                     "max_marks, marks"]),
        ("concept", 34, 38, "Concept (concepts.json)", ["id (C1..C5)", "name", "course_outcome",
                                                        "learning_objectives[]"]),
        ("chunk", 67, 38, "Curriculum chunk (vector DB)", ["text", "embedding (384)", "source, section", "chunk no."]),
        ("mcq", 34, 2, "Question (question_bank.json)", ["id, question", "options[4], answer", "concept, difficulty",
                                                         "bloom_level, explanation", "verification"]),
        ("state", 1, 2, "Knowledge state (ekt.pt)", ["concept list", "EKT weights", "mastery per concept",
                                                     "(computed per learner)"]),
        ("cycle", 67, 2, "Learner report (cycle.json)", ["round", "knowledge_state", "gaps",
                                                         "recommendations", "evaluation"]),
    ]
    for key, x, y, title, fields in tables:
        h = 5 + 3.6 * len(fields)
        d.ax.add_patch(FancyBboxPatch((x, y), 30, h, boxstyle="round,pad=0.02,rounding_size=0.8", fc="white",
                                      ec=BLUE, lw=1.3))
        d.ax.add_patch(FancyBboxPatch((x, y + h - 5), 30, 5, boxstyle="round,pad=0.02,rounding_size=0.8", fc=SOFT,
                                      ec=BLUE, lw=1.3))
        d.ax.text(x + 15, y + h - 2.5, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=NAVY)
        for j, f in enumerate(fields):
            d.ax.text(x + 2, y + h - 8 - j * 3.6, f, fontsize=9, color=NAVY, va="center")
        d.boxes[key] = (x, y, 30, h)
    d.arrow("records", "r", "concept", "l", color=GREY, label="concept")
    d.arrow("concept", "r", "chunk", "l", color=GREY, label="section")
    d.arrow("mcq", "t", "concept", "b", color=GREY, label="concept")
    d.arrow("records", "b", "state", "t", color=GREY, label="traced by EKT")
    d.arrow("state", "r", "mcq", "l", color=GREY)
    d.arrow("mcq", "r", "cycle", "l", color=GREY)
    d.save("fig_data.png")


# ------------------------------------------------------------------ charts from real results
def bar_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E3E6EC", lw=0.8)
    ax.set_axisbelow(True)


def kt_charts(exp):
    rows = list(csv.DictReader(open(PROJECT / "data/models/kt_history.csv", encoding="utf-8")))
    colors = {"ekt": BLUE, "dkt": ORANGE}
    for metric, fname, ylabel in [("auc", "fig_kt_auc.png", "AUC"), ("loss", "fig_kt_loss.png", "Loss (BCE)")]:
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for m in ("ekt", "dkt"):
            rs = [r for r in rows if r["model"] == m]
            ep = [int(r["epoch"]) for r in rs]
            ax.plot(ep, [float(r[f"train_{metric}"]) for r in rs], "--", color=colors[m], label=f"{m.upper()} train")
            ax.plot(ep, [float(r[f"test_{metric}"]) for r in rs], "-", color=colors[m], lw=2, label=f"{m.upper()} test")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=2)
        bar_style(ax)
        fig.tight_layout()
        fig.savefig(OUT / fname, dpi=200)
        plt.close(fig)
    fin = exp["knowledge_tracing"]["final_epoch"]
    names = ["accuracy", "precision", "recall", "f1", "auc"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(names))
    for k, (m, off) in enumerate([("ekt", -0.2), ("dkt", 0.2)]):
        vals = [fin[m][f"test_{n}"] for n in names]
        bars = ax.bar(x + off, vals, 0.38, color=colors[m], label=m.upper())
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5, color=NAVY)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    bar_style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kt_metrics.png", dpi=200)
    plt.close(fig)


def study_charts(exp):
    tt = exp["remedial_effect"]["paired_t_test"]
    pre = [tt[c]["mean_pre"] for c in ORDER]
    post = [tt[c]["mean_post"] for c in ORDER]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    x = np.arange(len(ORDER))
    b1 = ax.bar(x - 0.2, pre, 0.38, color=LIGHT, label="Before remediation")
    b2 = ax.bar(x + 0.2, post, 0.38, color=BLUE, label="After remediation")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center",
                    fontsize=8.5, color=NAVY)
    ax.set_xticks(x, [SHORT[c] for c in ORDER], fontsize=9)
    ax.set_ylabel("Mean score (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    bar_style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_before_after.png", dpi=200)
    plt.close(fig)

    ang = np.linspace(0, 2 * np.pi, len(ORDER), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(5.6, 5.0), subplot_kw={"polar": True})
    for vals, col, lab in [(pre, GREY, "Before"), (post, BLUE, "After")]:
        v = vals + vals[:1]
        ax.plot(ang + ang[:1], v, color=col, lw=2, label=lab)
        ax.fill(ang + ang[:1], v, color=col, alpha=0.15)
    ax.set_xticks(ang, [SHORT[c] for c in ORDER], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.12), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_radar.png", dpi=200)
    plt.close(fig)

    cats = ["Not Competent", "Beginner", "Intermediate", "Expert"]
    ccol = ["#C0392B", "#E08A2E", "#8FB4F5", BLUE]
    before, after = exp["remedial_effect"]["competency_before"], exp["remedial_effect"]["competency_after"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    pos, labels = [], []
    for i, c in enumerate(ORDER):
        for j, (src, tag) in enumerate([(before, "Before"), (after, "After")]):
            p = i * 2.6 + j
            bottom = 0
            for k, cat in enumerate(cats):
                v = src[c][cat]
                ax.bar(p, v, 0.85, bottom=bottom, color=ccol[k], label=cat if (i == 0 and j == 0) else None)
                if v >= 8:
                    ax.text(p, bottom + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=7.5,
                            color="white" if k in (0, 3) else NAVY)
                bottom += v
            pos.append(p)
            labels.append(tag)
    ax.set_xticks(pos, labels, fontsize=8)
    for i, c in enumerate(ORDER):
        ax.text(i * 2.6 + 0.5, -17, SHORT[c], ha="center", va="top", fontsize=8.5, color=NAVY)
    ax.set_ylabel("Learners (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_competency.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def demo_rounds():
    cyc = json.load(open(PROJECT / "data/store/learners/CSE230001/cycle.json", encoding="utf-8"))
    r1, r2 = cyc[0]["knowledge_state"], cyc[1]["knowledge_state"]
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    y = np.arange(len(ORDER))
    ax.barh(y + 0.2, [100 * r1[c] for c in ORDER], 0.38, color=LIGHT, label="Round 1 (after exam)")
    ax.barh(y - 0.2, [100 * r2[c] for c in ORDER], 0.38, color=BLUE, label="Round 2 (after remediation + test)")
    for i, c in enumerate(ORDER):
        ax.text(100 * r1[c] + 1, i + 0.2, f"{100 * r1[c]:.0f}%", va="center", fontsize=8.5)
        ax.text(100 * r2[c] + 1, i - 0.2, f"{100 * r2[c]:.0f}%", va="center", fontsize=8.5)
    ax.axvline(70, color=ORANGE, lw=2, ls="--")
    ax.text(70.5, len(ORDER) - 0.45, "70% target", color=ORANGE, fontsize=9, fontweight="bold")
    ax.axvline(50, color=GREY, lw=1, ls=":")
    ax.text(50.5, len(ORDER) - 0.45, "50% threshold", color=GREY, fontsize=9)
    ax.set_yticks(y, [SHORT[c] for c in ORDER], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("EKT mastery (%)")
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_demo_rounds.png", dpi=200)
    plt.close(fig)


def gap_matrix(exp):
    gm = exp["datasets"]["knowledge_gap_matrix"]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    y = np.arange(len(ORDER))
    att = [gm[c]["attained_pct"] for c in ORDER]
    ax.barh(y, att, color=BLUE, label="Attained (>= 70% target)")
    ax.barh(y, [100 - a for a in att], left=att, color=LIGHT, label="Not attained")
    for i, a in enumerate(att):
        ax.text(a / 2, i, f"{a:.1f}%", va="center", ha="center", color="white", fontsize=9, fontweight="bold")
        ax.text(a + (100 - a) / 2, i, f"{100 - a:.1f}%", va="center", ha="center", color=NAVY, fontsize=9)
    ax.set_yticks(y, [SHORT[c] for c in ORDER], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Learners in the current batch (%)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18), fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_gap_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def mcq_stats(exp):
    g = exp["mcq_generation"]
    names = list(g)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(names))
    keys = [("generated", LIGHT, "Generated"), ("duplicates", ORANGE, "Duplicates removed"),
            ("accepted", BLUE, "Accepted")]
    for k, (key, col, lab) in enumerate(keys):
        vals = [g[n]["pipeline_stats"][key] for n in names]
        bars = ax.bar(x + (k - 1) * 0.26, vals, 0.25, color=col, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", fontsize=8.5)
    ax.set_xticks(x, [n.replace("mock:", "") + " (offline)" for n in names])
    ax.set_ylabel("Questions")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    bar_style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mcq_stats.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    exp = json.load(open(PROJECT / "data/models/experiments.json", encoding="utf-8"))
    for f in (existing_system, proposed_system, architecture, rag_pipeline, ekt_model, remedial_flow, mcq_pipeline,
              adaptive_flow, concept_tree, data_design, demo_rounds):
        f()
    kt_charts(exp)
    study_charts(exp)
    gap_matrix(exp)
    mcq_stats(exp)
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))
