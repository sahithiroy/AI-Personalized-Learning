"""Column-width figures for the paper, drawn from paper_results.json and the project's cycle report."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

R = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 9, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})
DARK, MID, LIGHT, ACC = "#1F3A68", "#5B7DB8", "#B9C8E4", "#B8611A"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- 1. system loop (column diagram)
fig, ax = plt.subplots(figsize=(3.4, 3.3))
ax.set_xlim(0, 34)
ax.set_ylim(0, 33)
ax.axis("off")
boxes = {}


def box(k, x, y, w, h, t, s=None, hl=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.6",
                                fc="#E9EEF7" if hl else "white", ec=DARK if hl else "#8A96AA", lw=0.8))
    ax.text(x + w / 2, y + h * (0.63 if s else 0.5), t, ha="center", va="center", fontsize=7.6, fontweight="bold")
    if s:
        ax.text(x + w / 2, y + h * 0.27, s, ha="center", va="center", fontsize=6.4, color="#444444")
    boxes[k] = (x, y, w, h)


def arr(p, q, color=DARK, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=color, lw=0.8, linestyle=ls))


box("cur", 1, 27, 14, 5, "Curriculum PDF", "unit-aware chunks")
box("vdb", 19, 27, 14, 5, "Vector index", "MiniLM + FAISS")
box("obe", 1, 19, 14, 5, "OBE exam records", "marks per item")
box("ekt", 19, 19, 14, 5, "Calibrated EKT", "mastery per concept", True)
box("gap", 19, 11, 14, 5, "Gap rule", "mastery < target")
box("rem", 19, 3, 14, 5, "Grounded remediation", "5 parts, verified", True)
box("mcq", 1, 3, 14, 5, "Item generation", "3x, dedup 0.85")
box("ada", 1, 11, 14, 5, "Adaptive re-test", "easy > medium > hard")
arr((15, 29.5), (19, 29.5))
arr((15, 21.5), (19, 21.5))
arr((26, 19), (26, 16))
arr((26, 11), (26, 8))
arr((19, 5.5), (15, 5.5))
arr((8, 8), (8, 11))
arr((15, 13.5), (19, 20.4), ACC, "--")
ax.text(15.2, 17.1, "responses", fontsize=6.4, color=ACC, rotation=42)
save(fig, "p_system.png")

# ---------------------------------------------------------------- 2. KT ablation
ab = R["kt_ablation"]
names = list(ab)
short = {"DKT": "DKT", "EKT (full)": "EKT\n(full)", "EKT w/o attention": "w/o\nattention",
         "EKT w/o mastery loss": "w/o mastery\nloss", "EKT + exercise dropout 0.5": "+ exercise\ndropout"}
x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(3.4, 2.2))
for k, (key, col, lab) in enumerate([("test_auc", DARK, "Next-answer AUC"), ("mastery_auc", LIGHT, "Mastery-output AUC")]):
    m = [ab[n]["mean"][key] for n in names]
    s = [ab[n]["std"][key] for n in names]
    ax.bar(x + (k - 0.5) * 0.36, m, 0.34, yerr=s, color=col, edgecolor=DARK, lw=0.4, capsize=2,
           error_kw={"lw": 0.6}, label=lab)
ax.set_xticks(x, [short[n] for n in names], fontsize=7)
ax.set_ylim(0.5, 0.9)
ax.set_ylabel("AUC (mean of 3 seeds)")
ax.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "p_ablation.png")

# ---------------------------------------------------------------- 3. ordering artifact
od = R["ordering"]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
groups = [("EKT", "EKT, shuffled exams", "EKT, ordered exams"), ("DKT", "DKT, shuffled exams", "DKT, ordered exams")]
for i, (g, sh, orr) in enumerate(groups):
    for j, (key, col, lab) in enumerate([(sh, MID, "shuffled exams"), (orr, ACC, "ordered exams")]):
        m = od[key]["mean"]["delta_after_correct"]
        s = od[key]["std"]["delta_after_correct"]
        ax.bar(i + (j - 0.5) * 0.36, 100 * m, 0.34, yerr=100 * s, color=col, capsize=2, error_kw={"lw": 0.6},
               label=lab if i == 0 else None)
ax.axhline(0, color="black", lw=0.6)
ax.set_xticks([0, 1], ["EKT", "DKT"])
ax.set_ylabel("Mastery change after\n5 correct answers (points)")
ax.legend(frameon=False, fontsize=7)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "p_ordering.png")

# ---------------------------------------------------------------- 4. retrieval purity
rp = R["retrieval"]
strategies = ["Fixed windows + similarity", "Unit-aware chunks + similarity", "Unit-aware chunks + unit filter (ours)"]
lab = ["Fixed\nwindows", "Unit-aware\nchunks", "Unit-aware +\nunit filter"]
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for k, (size, col) in enumerate([(500, LIGHT), (1000, DARK)]):
    v = [100 * rp[f"{s} | {size}"]["purity"] for s in strategies]
    b = ax.bar(np.arange(3) + (k - 0.5) * 0.36, v, 0.34, color=col, edgecolor=DARK, lw=0.4,
               label=f"{size}-character chunks")
    for bb, vv in zip(b, v):
        ax.text(bb.get_x() + bb.get_width() / 2, vv + 1.5, f"{vv:.0f}", ha="center", fontsize=6.5)
ax.set_xticks(np.arange(3), lab, fontsize=7)
ax.set_ylim(0, 112)
ax.set_ylabel("Context purity (%)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
save(fig, "p_retrieval.png")

# ---------------------------------------------------------------- 5. closed loop
cl = R["closed_loop"]
fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
rounds = np.arange(1, 5)
for cond, col, ls in [("with remediation", DARK, "-"), ("without remediation", ACC, "--")]:
    axes[0].plot(rounds, cl[cond]["mean_gaps_per_round"], ls, color=col, marker="o", ms=3, lw=1, label=cond)
    axes[1].plot(rounds, [100 * v for v in cl[cond]["mean_mastery_per_round"]], ls, color=col, marker="o", ms=3, lw=1)
axes[0].set_ylabel("Mean open gaps")
axes[1].set_ylabel("Mean mastery (%)")
axes[1].axhline(70, color="#888888", lw=0.6, ls=":")
for a in axes:
    a.set_xlabel("Round")
    a.set_xticks(rounds)
    a.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=6.3, loc="upper right")
save(fig, "p_loop.png")
print("figures written to", OUT)
