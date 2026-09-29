"""Render blocks.json to a university-style PDF report (reportlab)."""
import html
import json
import re
import sys
import textwrap

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Flowable, Frame, Image, KeepTogether, ListFlowable, ListItem,
                                PageBreak, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle,
                                NextPageTemplate)
from reportlab.platypus.tableofcontents import TableOfContents

BLOCKS = json.load(open(sys.argv[1], encoding="utf-8"))
OUT = sys.argv[2]

F = "C:/Windows/Fonts/"
for name, file in [("TNR", "times.ttf"), ("TNR-B", "timesbd.ttf"), ("TNR-I", "timesi.ttf"),
                   ("TNR-BI", "timesbi.ttf"), ("Mono", "consola.ttf"), ("Mono-B", "consolab.ttf")]:
    try:
        pdfmetrics.registerFont(TTFont(name, F + file))
    except Exception:
        if name == "Mono-B":
            pdfmetrics.registerFont(TTFont(name, F + "consola.ttf"))
pdfmetrics.registerFontFamily("TNR", normal="TNR", bold="TNR-B", italic="TNR-I", boldItalic="TNR-BI")
pdfmetrics.registerFontFamily("Mono", normal="Mono", bold="Mono-B", italic="Mono", boldItalic="Mono-B")

NAVY = colors.HexColor("#16233A")
BLUE = colors.HexColor("#1F4E99")
SOFT = colors.HexColor("#E8EEF8")
GRID = colors.HexColor("#9AA6B8")
CODEBG = colors.HexColor("#F4F5F7")
WARN = colors.HexColor("#FBF1DF")
WARNB = colors.HexColor("#A9540F")

LM, RM, TM, BM = 2.5 * cm, 2.0 * cm, 2.2 * cm, 2.2 * cm
TW = A4[0] - LM - RM
TH = A4[1] - TM - BM

S = {
    "p": ParagraphStyle("p", fontName="TNR", fontSize=12, leading=18, alignment=TA_JUSTIFY, spaceAfter=8),
    "li": ParagraphStyle("li", fontName="TNR", fontSize=12, leading=18, alignment=TA_JUSTIFY),
    "h1a": ParagraphStyle("h1a", fontName="TNR-B", fontSize=16, leading=22, alignment=TA_CENTER, textColor=NAVY),
    "h1b": ParagraphStyle("h1b", fontName="TNR-B", fontSize=20, leading=26, alignment=TA_CENTER, textColor=NAVY,
                          spaceAfter=22),
    "front": ParagraphStyle("front", fontName="TNR-B", fontSize=18, leading=24, alignment=TA_CENTER,
                            textColor=NAVY, spaceAfter=20),
    "h2": ParagraphStyle("h2", fontName="TNR-B", fontSize=14, leading=19, textColor=NAVY, spaceBefore=12,
                         spaceAfter=6, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="TNR-B", fontSize=12.5, leading=17, textColor=NAVY, spaceBefore=8,
                         spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", fontName="TNR-B", fontSize=11, leading=14, alignment=TA_CENTER, spaceBefore=4,
                          spaceAfter=10),
    "tcap": ParagraphStyle("tcap", fontName="TNR-B", fontSize=11, leading=14, alignment=TA_CENTER, spaceBefore=6,
                           spaceAfter=5, keepWithNext=1),
    "cell": ParagraphStyle("cell", fontName="TNR", fontSize=10, leading=12.5),
    "cellh": ParagraphStyle("cellh", fontName="TNR-B", fontSize=10, leading=12.5),
    "code": ParagraphStyle("code", fontName="Mono", fontSize=8.2, leading=10.4),
    "codet": ParagraphStyle("codet", fontName="TNR-B", fontSize=10.5, leading=13, spaceBefore=4, spaceAfter=2,
                            keepWithNext=1),
    "eq": ParagraphStyle("eq", fontName="TNR-I", fontSize=12, leading=17, alignment=TA_CENTER, spaceAfter=6),
    "note": ParagraphStyle("note", fontName="TNR", fontSize=11.5, leading=16.5, alignment=TA_JUSTIFY),
    "ref": ParagraphStyle("ref", fontName="TNR", fontSize=11, leading=15, alignment=TA_JUSTIFY, leftIndent=26,
                          firstLineIndent=-26, spaceAfter=5),
    "center": ParagraphStyle("center", fontName="TNR", fontSize=12, leading=18, alignment=TA_CENTER),
    "sig": ParagraphStyle("sig", fontName="TNR", fontSize=11.5, leading=15, alignment=TA_CENTER),
}


def md(text):
    t = html.escape(text, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([A-Za-z][^*\n]*?)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"`(.+?)`", r"<font face='Mono'>\1</font>", t)
    return t


class Marker(Flowable):
    """Zero-size flowable that switches page numbering or records a TOC entry when drawn."""

    def __init__(self, action, **kw):
        super().__init__()
        self.action, self.kw = action, kw
        self.width = self.height = 0

    def draw(self):
        doc = self.canv._doctemplate
        if self.action == "main":
            doc.main_start = self.canv.getPageNumber()
        elif self.action == "front":
            doc.front_start = self.canv.getPageNumber()


class Heading(Paragraph):
    def __init__(self, text, style, toc_kind=None, toc_text=None, level=0):
        super().__init__(text, style)
        self.toc_kind, self.toc_text, self.level = toc_kind, toc_text, level


class Caption(Paragraph):
    def __init__(self, text, style, kind):
        super().__init__(md(text), style)
        self.kind, self.plain = kind, text


class ListTOC(TableOfContents):
    def __init__(self, kind, **kw):
        super().__init__(**kw)
        self.kind = kind

    def notify(self, kind, stuff):
        if kind == self.kind:
            self.addEntry(*stuff)


def roman(n):
    vals = [(10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, pagesize=A4, leftMargin=LM, rightMargin=RM, topMargin=TM, bottomMargin=BM, **kw)
        self.main_start = None
        self.front_start = 2
        frame = Frame(LM, BM, TW, TH, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate("title", [frame], onPage=lambda c, d: None),
                               PageTemplate("body", [frame], onPageEnd=self.decorate)])

    def label(self, page):
        if self.main_start and page >= self.main_start:
            return str(page - self.main_start + 1)
        return roman(max(page - self.front_start + 1, 1))

    def decorate(self, canv, doc):
        canv.saveState()
        canv.setFont("TNR", 10.5)
        canv.setFillColor(colors.HexColor("#444444"))
        canv.drawCentredString(A4[0] / 2, 1.2 * cm, self.label(canv.getPageNumber()))
        if self.main_start and canv.getPageNumber() >= self.main_start:
            canv.setFont("TNR-I", 9)
            canv.drawRightString(A4[0] - RM, A4[1] - 1.3 * cm,
                                 "AI-Driven Personalized Learning and Remedial Recommendation")
            canv.setStrokeColor(colors.HexColor("#B8C0CC"))
            canv.setLineWidth(0.5)
            canv.line(LM, A4[1] - 1.45 * cm, A4[0] - RM, A4[1] - 1.45 * cm)
        canv.restoreState()

    def afterFlowable(self, f):
        page = self.page
        if isinstance(f, Heading) and f.toc_kind:
            self.notify("TOCEntry", (f.level, f.toc_text, page, None))
        elif isinstance(f, Caption):
            self.notify(f.kind, (0, f.plain, page, None))


def table_flowable(b):
    fs = b.get("font") or 10
    cs = ParagraphStyle("c", parent=S["cell"], fontSize=fs, leading=fs * 1.25)
    ch = ParagraphStyle("h", parent=S["cellh"], fontSize=fs, leading=fs * 1.25)
    data = [[Paragraph(md(c), ch) for c in b["header"]]]
    data += [[Paragraph(md(c), cs) for c in r] for r in b["rows"]]
    widths = [w * TW for w in b["widths"]]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, GRID), ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def boxed(flowables, bg, border):
    t = Table([[flowables]], colWidths=[TW])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg), ("BOX", (0, 0), (-1, -1), 0.8, border),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def code_flowable(text):
    maxc = int((TW - 18) / (0.55 * 8.2))
    lines = []
    for line in text.split("\n"):
        if len(line) <= maxc:
            lines.append(line)
        else:
            ind = len(line) - len(line.lstrip())
            lines += textwrap.wrap(line, maxc, subsequent_indent=" " * (ind + 4), drop_whitespace=False) or [""]
    return boxed(Preformatted("\n".join(lines), S["code"]), CODEBG, colors.HexColor("#D5DAE3"))


def logo_box(label):
    t = Table([[Paragraph(label, S["sig"])]], colWidths=[4 * cm], rowHeights=[3.2 * cm])
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.8, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
    return t


story = [NextPageTemplate("title")]


def page_break():
    """Add a page break unless the story already ends with one (avoids blank pages)."""
    last = next((f for f in reversed(story) if not isinstance(f, (Marker, NextPageTemplate))), None)
    if not isinstance(last, PageBreak):
        story.append(PageBreak())


pending_main = False
for b in BLOCKS:
    t = b["t"]
    if t == "titlepage":
        story += [Spacer(1, 0.3 * cm),
                  Table([[logo_box("College logo"), "", logo_box("Department logo")]],
                        colWidths=[4.4 * cm, TW - 8.8 * cm, 4.4 * cm]),
                  Spacer(1, 0.9 * cm),
                  Paragraph(md(b["title"]), ParagraphStyle("tt", parent=S["h1b"], fontSize=21, leading=28,
                                                          spaceAfter=18)),
                  Paragraph(md(b["kind"]), S["center"]), Spacer(1, 4),
                  Paragraph(f"<b>{md(b['degree'])}</b>", ParagraphStyle("dg", parent=S["center"], fontSize=13.5,
                                                                        leading=19)),
                  Spacer(1, 0.8 * cm), Paragraph(md(b["by"]), S["center"]), Spacer(1, 4)]
        story += [Paragraph(f"<b>{md(s)}</b>", S["center"]) for s in b["students"]]
        story += [Spacer(1, 0.8 * cm), Paragraph(md(b["guide"]), S["center"]), Spacer(1, 1.3 * cm),
                  Paragraph(f"<b>{md(b['dept'])}</b>", ParagraphStyle("dp", parent=S["center"], fontSize=14,
                                                                     leading=20)),
                  Paragraph(f"<b>{md(b['college'])}</b>", ParagraphStyle("cl", parent=S["center"], fontSize=15,
                                                                        leading=21)),
                  Paragraph(md(b["univ"]), S["center"]), Spacer(1, 0.4 * cm),
                  Paragraph(f"<b>{md(b['year'])}</b>", S["center"]),
                  NextPageTemplate("body"), PageBreak(), Marker("front")]
    elif t == "frontheading":
        page_break()
        story += [Heading(md(b["text"]), S["front"], "TOCEntry", b["text"], 0)]
    elif t == "h1":
        page_break()
        if pending_main:
            story.append(Marker("main"))
            pending_main = False
        if b["num"]:
            story += [Spacer(1, 1.2 * cm), Paragraph(b["text"].upper(), S["h1a"]), Spacer(1, 6),
                      Heading(md(b["sub"]), S["h1b"], "TOCEntry", f"{b['num']}. {b['sub']}", 0)]
        else:
            story += [Spacer(1, 1.2 * cm), Heading(md(b["text"]), S["h1b"], "TOCEntry", b["text"], 0)]
    elif t == "h2":
        story.append(Heading(md(b["text"]), S["h2"], "TOCEntry", b["text"], 1))
    elif t == "h3":
        story.append(Heading(md(b["text"]), S["h3"], "TOCEntry", b["text"], 2))
    elif t == "p":
        story.append(Paragraph(md(b["text"]), S["p"]))
    elif t in ("bullets", "numbered"):
        items = [ListItem(Paragraph(md(i), S["li"]), leftIndent=18, value=None) for i in b["items"]]
        story += [ListFlowable(items, bulletType="bullet" if t == "bullets" else "1", start=None if t == "bullets"
                               else 1, bulletFontName="TNR", bulletFontSize=11, leftIndent=18,
                               bulletFormat="%s." if t == "numbered" else None), Spacer(1, 8)]
    elif t == "table":
        story += [Caption(b["caption"], S["tcap"], "LOTEntry"), table_flowable(b), Spacer(1, 10)]
    elif t == "figure":
        from reportlab.lib.utils import ImageReader

        iw, ih = ImageReader(b["path"]).getSize()
        w = TW * b.get("width", 1.0)
        h = w * ih / iw
        maxh = TH * 0.78
        if h > maxh:
            h, w = maxh, maxh * iw / ih
        story.append(KeepTogether([Image(b["path"], width=w, height=h),
                                   Caption(b["caption"], S["cap"], "LOFEntry")]))
    elif t == "code":
        items = [code_flowable(b["text"]), Spacer(1, 8)]
        if b.get("title"):
            items.insert(0, Paragraph(md(b["title"]), S["codet"]))
        story += items if len(b["text"].split("\n")) > 30 else [KeepTogether(items)]
    elif t == "equation":
        eq = html.escape(b["text"])
        eq = re.sub(r"_\{([^}]*)\}", r"<sub>\1</sub>", eq)
        eq = re.sub(r"_([A-Za-z0-9]+)", r"<sub>\1</sub>", eq)
        story.append(Paragraph(eq, S["eq"]))
    elif t == "note":
        warn = b.get("kind") == "warn"
        story += [boxed(Paragraph(md(b["text"]), S["note"]), WARN if warn else SOFT, WARNB if warn else BLUE),
                  Spacer(1, 10)]
    elif t == "pagebreak":
        page_break()
    elif t == "signatures":
        cells = []
        for title, name in b["items"]:
            parts = [Spacer(1, 1.4 * cm), Paragraph("____________________", S["sig"]),
                     Paragraph(f"<b>{md(title)}</b>", S["sig"])]
            parts += [Paragraph(md(n), S["sig"]) for n in name.split("\n") if n]
            cells.append(parts)
        story += [Table([cells], colWidths=[TW / len(cells)] * len(cells)), Spacer(1, 0.6 * cm)]
    elif t == "abbreviations":
        rows = [[Paragraph(f"<b>{a}</b>", S["cell"]), Paragraph(md(d), S["cell"])] for a, d in b["rows"]]
        tb = Table(rows, colWidths=[3.2 * cm, TW - 3.2 * cm])
        tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 11), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDE2EA"))]))
        story.append(tb)
    elif t in ("toc", "lof", "lot"):
        title = {"toc": "Table of Contents", "lof": "List of Figures", "lot": "List of Tables"}[t]
        page_break()
        story.append(Heading(title, S["front"], "TOCEntry", title, 0))
        lvl = [ParagraphStyle("t0", fontName="TNR-B", fontSize=11.5, leading=16, leftIndent=0, spaceBefore=3),
               ParagraphStyle("t1", fontName="TNR", fontSize=11, leading=14.5, leftIndent=18),
               ParagraphStyle("t2", fontName="TNR", fontSize=10.5, leading=13.5, leftIndent=36)]
        if t == "toc":
            toc = TableOfContents(levelStyles=lvl, dotsMinLevel=0, formatter=lambda p: DOC.label(p))
        else:
            toc = ListTOC("LOFEntry" if t == "lof" else "LOTEntry",
                          levelStyles=[ParagraphStyle("l", fontName="TNR", fontSize=11, leading=15)], dotsMinLevel=0,
                          formatter=lambda p: DOC.label(p))
        story.append(toc)
    elif t == "mainmatter":
        pending_main = True
    elif t == "references":
        story += [Paragraph(f"[{i}]&nbsp;&nbsp;{md(r)}", S["ref"]) for i, r in enumerate(b["items"], 1)]

DOC = doc = Doc(OUT, title="AI-Driven Personalized Learning and Remedial Recommendation - Project Report",
          author="[Student names]", subject="Final year project report")
doc.multiBuild(story)
print("pages:", doc.page)
