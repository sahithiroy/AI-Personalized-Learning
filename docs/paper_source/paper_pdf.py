"""Render paper_blocks.json as an IEEE-style two-column paper (US Letter, Times 10 pt)."""
import html
import json
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, FrameBreak, Image, KeepTogether, NextPageTemplate,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

BLOCKS = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]
F = "C:/Windows/Fonts/"
for n, f in [("T", "times.ttf"), ("T-B", "timesbd.ttf"), ("T-I", "timesi.ttf"), ("T-BI", "timesbi.ttf"),
             ("M", "consola.ttf")]:
    pdfmetrics.registerFont(TTFont(n, F + f))
pdfmetrics.registerFontFamily("T", normal="T", bold="T-B", italic="T-I", boldItalic="T-BI")

W, H = letter
ML = MR = 0.625 * inch
MT, MB = 0.75 * inch, 1.0 * inch
GAP = 0.25 * inch
CW = (W - ML - MR - GAP) / 2
TOP_H = 2.35 * inch

S = {
    "title": ParagraphStyle("title", fontName="T", fontSize=22, leading=26, alignment=TA_CENTER, spaceAfter=10),
    "author": ParagraphStyle("author", fontName="T", fontSize=10, leading=12, alignment=TA_CENTER),
    "abs": ParagraphStyle("abs", fontName="T-B", fontSize=9, leading=10.8, alignment=TA_JUSTIFY, spaceAfter=6),
    "p": ParagraphStyle("p", fontName="T", fontSize=10, leading=11.8, alignment=TA_JUSTIFY, firstLineIndent=12,
                        spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="T", fontSize=10, leading=12, alignment=TA_CENTER, spaceBefore=8,
                         spaceAfter=4, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="T-I", fontSize=10, leading=12, spaceBefore=5, spaceAfter=2,
                         keepWithNext=1),
    "li": ParagraphStyle("li", fontName="T", fontSize=10, leading=11.8, alignment=TA_JUSTIFY, leftIndent=12,
                         bulletIndent=2, spaceAfter=1.5),
    "cap": ParagraphStyle("cap", fontName="T", fontSize=8, leading=9.5, alignment=TA_JUSTIFY, spaceBefore=3,
                          spaceAfter=8),
    "tcap": ParagraphStyle("tcap", fontName="T", fontSize=8, leading=9.5, alignment=TA_CENTER, spaceBefore=6,
                           spaceAfter=3, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName="T", fontSize=8, leading=9.2),
    "cellh": ParagraphStyle("cellh", fontName="T-B", fontSize=8, leading=9.2),
    "eq": ParagraphStyle("eq", fontName="T-I", fontSize=10, leading=13, alignment=TA_CENTER),
    "eqn": ParagraphStyle("eqn", fontName="T", fontSize=10, leading=13, alignment=2),
    "ref": ParagraphStyle("ref", fontName="T", fontSize=8, leading=9.6, alignment=TA_JUSTIFY, leftIndent=16,
                          firstLineIndent=-16, spaceAfter=2),
    "ack": ParagraphStyle("ack", fontName="T", fontSize=10, leading=11.8, alignment=TA_JUSTIFY),
}


GREEK = {"theta": "θ", "tau": "τ", "lambda": "λ", "sigma": "σ"}
VAR = "A-Za-zθτλσ"


def greek(t):
    for k, v in GREEK.items():
        t = re.sub(rf"(?<![A-Za-z]){k}(?![A-Za-z])", v, t)
    return t.replace("&lt;=", "≤").replace("&gt;=", "≥")


def mathify(t):
    """x_t, h_{t-1}, theta_k -> italic with subscripts; Greek names -> symbols (t is already escaped)."""
    t = greek(t)
    t = re.sub(rf"([{VAR}]'?)_\{{([^}}]*)\}}", r"<i>\1<sub>\2</sub></i>", t)
    t = re.sub(rf"([{VAR}]'?)_([A-Za-z0-9+]+)", r"<i>\1<sub>\2</sub></i>", t)
    return t


def md(t):
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([A-Za-z][^*\n]*?)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='M' size='8.5'>\1</font>", t)
    return mathify(t)


def eq_markup(t):
    t = greek(html.escape(t, quote=False))
    t = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", t)
    t = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", t)
    t = re.sub(r"\^\{([^}]*)\}", r"<super>\1</super>", t)
    return t


def smallcaps(text, size=10):
    """IEEE section headings: small capitals."""
    out = []
    for ch in text:
        if ch.islower():
            out.append(f"<font size='{size * 0.8:.1f}'>{ch.upper()}</font>")
        else:
            out.append(html.escape(ch))
    return "".join(out)


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


def on_page(c, d):
    c.saveState()
    c.setFont("T", 8)
    c.setFillColor(colors.HexColor("#444444"))
    c.drawCentredString(W / 2, 0.55 * inch, str(d.page))
    c.restoreState()


doc = BaseDocTemplate(OUT, pagesize=letter, leftMargin=ML, rightMargin=MR, topMargin=MT, bottomMargin=MB,
                      title=BLOCKS[0]["title"], author="[Authors]")
colh = H - MT - MB
first = [Frame(ML, H - MT - TOP_H, W - ML - MR, TOP_H, id="top", leftPadding=0, rightPadding=0, topPadding=0,
               bottomPadding=0),
         Frame(ML, MB, CW, colh - TOP_H, id="c1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
         Frame(ML + CW + GAP, MB, CW, colh - TOP_H, id="c2", leftPadding=0, rightPadding=0, topPadding=0,
               bottomPadding=0)]
rest = [Frame(ML, MB, CW, colh, id="r1", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
        Frame(ML + CW + GAP, MB, CW, colh, id="r2", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)]
doc.addPageTemplates([PageTemplate("first", first, onPage=on_page), PageTemplate("rest", rest, onPage=on_page)])

story = [NextPageTemplate("rest")]
sec = 0
sub = 0
fig_n = 0
tab_n = 0
eq_n = 0
for b in BLOCKS:
    t = b["t"]
    if t == "title":
        story.append(Paragraph(md(b["title"]), S["title"]))
        cells = []
        for a in b["authors"]:
            lines = [f"<i>{md(a['name'])}</i>"] + [md(x) for x in a["lines"]]
            cells.append(Paragraph("<br/>".join(lines), S["author"]))
        story.append(Table([cells], colWidths=[(W - ML - MR) / len(cells)] * len(cells)))
        story.append(FrameBreak())
    elif t == "abstract":
        story.append(Paragraph(f"<i>Abstract</i>\u2014{md(b['text'])}", S["abs"]))
    elif t == "keywords":
        story.append(Paragraph(f"<i>Index Terms</i>\u2014{md(b['text'])}", S["abs"]))
    elif t == "h1":
        sec += 1
        sub = 0
        label = b["text"] if b.get("unnumbered") else f"{ROMAN[sec - 1]}. {b['text']}"
        story.append(Paragraph(smallcaps(label), S["h1"]))
    elif t == "h2":
        sub += 1
        story.append(Paragraph(f"{chr(64 + sub)}. {md(b['text'])}", S["h2"]))
    elif t == "p":
        story.append(Paragraph(md(b["text"]), S["p"]))
    elif t == "bullets":
        for i in b["items"]:
            story.append(Paragraph(md(i), S["li"], bulletText="\u2022"))
    elif t == "numbered":
        for k, i in enumerate(b["items"], 1):
            story.append(Paragraph(md(i), S["li"], bulletText=f"{k})"))
    elif t == "equation":
        eq_n += 1
        tb = Table([[Paragraph(eq_markup(b["text"]), S["eq"]), Paragraph(f"({eq_n})", S["eqn"])]],
                   colWidths=[CW - 0.35 * inch, 0.35 * inch])
        tb.setStyle(TableStyle([("ALIGN", (0, 0), (0, 0), "CENTER"), ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 2),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        story.append(tb)
    elif t == "figure":
        fig_n += 1
        iw, ih = ImageReader(b["path"]).getSize()
        w = CW * b.get("width", 1.0)
        h = w * ih / iw
        story.append(KeepTogether([Spacer(1, 4), Image(b["path"], width=w, height=h),
                                   Paragraph(f"Fig. {fig_n}. {md(b['caption'])}", S["cap"])]))
    elif t == "table":
        tab_n += 1
        head = [Paragraph(md(c), S["cellh"]) for c in b["header"]]
        rows = [[Paragraph(md(c), S["cell"]) for c in r] for r in b["rows"]]
        tb = Table([head] + rows, colWidths=[w * CW for w in b["widths"]], repeatRows=1)
        tb.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
                                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
                                ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
                                ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5), ("TOPPADDING", (0, 0), (-1, -1), 1.8),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8)]))
        cap = Paragraph(f"TABLE {ROMAN[tab_n - 1]}<br/>{smallcaps(b['caption'], 8)}", S["tcap"])
        story.append(KeepTogether([cap, tb, Spacer(1, 6)]))
    elif t == "ack":
        story.append(Paragraph(smallcaps("Acknowledgment"), S["h1"]))
        story.append(Paragraph(md(b["text"]), S["ack"]))
    elif t == "references":
        story.append(Paragraph(smallcaps("References"), S["h1"]))
        for i, r in enumerate(b["items"], 1):
            story.append(Paragraph(f"[{i}]&nbsp;&nbsp;{md(r)}", S["ref"]))

doc.build(story)
print("pages:", doc.page)
