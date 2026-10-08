"""Personal study-material PDF for a learner (extension of paper Sec. III-C).

For every knowledge gap the PDF collects, in study order (weakest concept first):
the questions the learner got wrong, the five-section remedial recommendation,
and the matching pages of the curriculum retrieved from the vector DB (RAG), so
the learner has the actual course material to read next to the plan.
"""
from __future__ import annotations

import io
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from .gap import obe_levels
from .rag import CurriculumKB

NAVY = colors.HexColor("#16233A")
BLUE = colors.HexColor("#2A5FC0")
GRID = colors.HexColor("#C9D1DE")
TINT = {"bad": colors.HexColor("#FBE9E7"), "warn": colors.HexColor("#FFF4E0"), "good": colors.HexColor("#E8F3EA"),
        "info": colors.HexColor("#EEF2F8"), "read": colors.HexColor("#FBF8F1")}

_ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=10.5, leading=15, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=9.5, leading=13, spaceAfter=0)
READ = ParagraphStyle("read", parent=BODY, fontSize=10, leading=14.5, spaceAfter=0)
TITLE = ParagraphStyle("title", parent=_ss["Title"], fontName="Helvetica-Bold", fontSize=24, leading=30, textColor=NAVY)
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=17, leading=22,
                    textColor=NAVY, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5, textColor=BLUE,
                    spaceBefore=10, spaceAfter=4)
META = ParagraphStyle("meta", parent=BODY, textColor=colors.HexColor("#5A6475"))
WIDTH = 16.6 * cm


def _t(x) -> str:
    """Plain text -> safe reportlab markup (keeps C code like a[i] < n or '\\0' intact)."""
    return escape(str(x if x is not None else "")).replace("\n", "<br/>")


def _box(flowables, tint: str):
    t = Table([[flowables]], colWidths=[WIDTH])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), TINT[tint]), ("BOX", (0, 0), (-1, -1), 0.6, GRID),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def _table(rows, widths, tints=None):
    data = [[Paragraph(f"<b>{_t(c)}</b>" if i == 0 else _t(c), SMALL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.5, GRID), ("BACKGROUND", (0, 0), (-1, 0), TINT["info"]),
             ("VALIGN", (0, 0), (-1, -1), "TOP")]
    for i, tint in enumerate(tints or [], start=1):
        style.append(("BACKGROUND", (-1, i), (-1, i), TINT[tint]))
    t.setStyle(TableStyle(style))
    return t


def _bullets(items) -> list:
    out = []
    for x in items or []:
        out.append(Paragraph(f"&bull; {_t(x if isinstance(x, str) else x.get('topic') or x.get('title') or x)}", BODY))
    return out


def _course_outcome(kb: CurriculumKB | None, concept: str) -> str:
    try:
        return kb.concept(concept).course_outcome if kb else ""
    except KeyError:
        return ""


def _reading(kb: CurriculumKB | None, concept: str) -> list[str]:
    if kb is None:
        return []
    try:
        return kb.context_for(kb.concept(concept))
    except KeyError:
        return kb.retrieve(concept)


def _concept_section(no: int, gap: dict, rec: dict, kb: CurriculumKB | None) -> list:
    concept = gap["concept"]
    co = _course_outcome(kb, concept)
    exam = "not attempted" if gap.get("obe_score") is None else f"{gap['obe_score']:.0f}%"
    story = [Paragraph(f"Priority {no}: {_t(concept)}", H1),
             Paragraph(f"{_t(co) + ' &middot; ' if co else ''}Mastery {gap['mastery']:.0%} &middot; "
                       f"exam score {exam} &middot; level {_t(gap.get('level', ''))}", META)]

    wrong = [a for a in gap.get("attempts", []) if not a.get("correct") and a.get("question")]
    if wrong:
        story += [Paragraph("Questions you got wrong", H2),
                  _box([Paragraph(f"&#10007; {_t(a['question'])} <font color='#5A6475'>({_t(a.get('difficulty', ''))})"
                                  f"</font>", SMALL) for a in wrong], "bad")]

    story += [Paragraph("1. What you will be able to do", H2), *_bullets(rec.get("learning_objectives"))]

    story.append(Paragraph("2. Topics to revise", H2))
    for i, t in enumerate(rec.get("topics_to_revise") or [], 1):
        if isinstance(t, dict):
            story.append(Paragraph(f"{i}. <b>{_t(t.get('topic', ''))}</b>", BODY))
            story += [Paragraph(f"&nbsp;&nbsp;&nbsp;&ndash; {_t(p)}", BODY) for p in t.get("points", [])]
        else:
            story.append(Paragraph(f"{i}. {_t(t)}", BODY))

    if rec.get("remedial_explanation"):
        story += [Paragraph("3. Explanation", H2), Paragraph(_t(rec["remedial_explanation"]), BODY)]

    chunks = _reading(kb, concept)
    if chunks:
        boxes = [_box([Paragraph(_t(c), READ)], "read") for c in chunks]
        # keep the heading on the same page as the first passage
        story.append(KeepTogether([Paragraph("Reading material from your syllabus", H2),
                                   Paragraph("These are the parts of the course notes that cover this concept. "
                                             "Read them before you try the practice activities.", META),
                                   boxes[0], Spacer(1, 5)]))
        for b in boxes[1:]:
            story += [b, Spacer(1, 5)]

    story.append(Paragraph("4. Practice activities", H2))
    for a in rec.get("practice_activities") or []:
        if isinstance(a, dict):
            part = [Paragraph(f"<b>{_t(a.get('title', 'Practice'))}</b>: {_t(a.get('description', ''))}", BODY)]
            if a.get("input") or a.get("expected_output"):
                part.append(Paragraph(f"<font face='Courier'>Input: {_t(a.get('input', '-'))} &rarr; Expected output: "
                                      f"{_t(a.get('expected_output', '-'))}</font>", SMALL))
            story.append(KeepTogether(part))
        else:
            story.append(Paragraph(f"&bull; {_t(a)}", BODY))

    if rec.get("concept_gap_rationale"):
        story += [Paragraph("5. Why this was recommended", H2), Paragraph(_t(rec["concept_gap_rationale"]), BODY)]

    topics = [t.get("topic") if isinstance(t, dict) else t for t in rec.get("topics_to_revise") or []]
    if topics:
        story += [Paragraph("Self-check: tick when you can explain it without notes", H2),
                  _box([Paragraph(f"&#9744;&nbsp; {_t(t)}", BODY) for t in topics if t], "good")]
    return story


def build_study_pdf(learner_id: str, concepts: list[dict], recommendations: list[dict],
                    kb: CurriculumKB | None = None, only: str | None = None) -> bytes:
    """Return the PDF bytes. ``concepts`` = /analyze output, ``recommendations`` = /recommend recommendation dicts.

    ``only`` limits the study sections to one concept (the summary table still lists all of them).
    """
    threshold, target = obe_levels()
    recs = {r.get("concept"): r for r in recommendations}
    gaps = sorted((c for c in concepts if c.get("is_gap")), key=lambda c: c["mastery"])
    if only:
        gaps = [c for c in gaps if c["concept"] == only]

    story = [Paragraph("Personal Study Material", TITLE),
             Paragraph(f"Learner: <b>{_t(learner_id)}</b> &middot; {date.today():%d %B %Y}", META), Spacer(1, 8)]
    rows, tints = [["Concept", "Mastery", "Level", "Status"]], []
    for c in sorted(concepts, key=lambda c: c["mastery"]):
        m = 100 * c["mastery"]
        rows.append([c["concept"], f"{m:.0f}%", c.get("level", ""), "Study this" if c.get("is_gap") else "On target"])
        tints.append("good" if m >= target else "warn" if m >= threshold else "bad")
    story += [Paragraph("Your knowledge state", H2), _table(rows, [8.6, 2.2, 3.0, 2.8], tints),
              Paragraph(f"A concept needs study when mastery is below the course target of {target:.0f}% "
                        f"(the pass threshold is {threshold:.0f}%).", META)]
    if not gaps:
        story.append(_box([Paragraph("Every concept is at or above the target. No remedial study is needed; "
                                     "keep practising to stay there.", BODY)], "good"))
    else:
        steps = ["Start with Priority 1, your weakest concept.",
                 "Look at the questions you got wrong and find out why the right answer is right.",
                 "Read the topics to revise and the explanation, then the syllabus pages that follow them.",
                 "Do the practice activities without looking at the notes.",
                 "Tick the self-check list, then take the adaptive test in the app."]
        story += [Paragraph("How to use this material", H2),
                  *[Paragraph(f"{i}. {s}", BODY) for i, s in enumerate(steps, 1)]]
        for i, g in enumerate(gaps, 1):
            story += [PageBreak(), *_concept_section(i, g, recs.get(g["concept"], {}), kb)]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(colors.HexColor("#7A8394"))
        canvas.drawString(2.2 * cm, 1.2 * cm, f"Personal study material - {learner_id}")
        canvas.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, str(doc.page))
        canvas.restoreState()

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                      bottomMargin=2 * cm, title=f"Study material - {learner_id}").build(
        story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
