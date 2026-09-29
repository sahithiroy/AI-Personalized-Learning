// Render blocks.json to a Word report (docx-js). Same content and structure as the PDF.
const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  BorderStyle, ShadingType, ImageRun, TableOfContents, StyleLevel, Footer, Header, PageNumber, NumberFormat,
  LevelFormat, PageBreak, VerticalAlign,
} = require("docx");

const blocks = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3];

const NAVY = "16233A", SOFT = "E8EEF8", GRID = "9AA6B8", CODEBG = "F4F5F7", WARN = "FBF1DF", WARNB = "A9540F",
  BLUE = "1F4E99";
const PAGE_W = 11906, PAGE_H = 16838, ML = 1418, MR = 1134, MT = 1247, MB = 1247;
const TW = PAGE_W - ML - MR; // text width in DXA
const TWPX = Math.round(TW / 1440 * 96);
const THPX = Math.round((PAGE_H - MT - MB) / 1440 * 96);
const FONT = "Times New Roman", MONO = "Consolas";

// ------------------------------------------------------------------ inline markup
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+?\*\*|`[^`]+?`|(?<![\w*])\*[A-Za-z][^*\n]*?\*(?![\w*]))/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: MONO, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function eqRuns(text) {
  // x_e, h_{t-1} -> subscripts
  const out = [];
  const re = /_\{([^}]*)\}|_([A-Za-z0-9]+)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), italics: true }));
    out.push(new TextRun({ text: m[1] ?? m[2], italics: true, subScript: true }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), italics: true }));
  return out;
}

const P = (text, opts = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 120, line: 360 }, ...opts });
const C = (text, opts = {}, run = {}) => new Paragraph({ children: runs(text, run), alignment: AlignmentType.CENTER,
  spacing: { after: 60, line: 300 }, ...opts });

// ------------------------------------------------------------------ building blocks
const border = { style: BorderStyle.SINGLE, size: 4, color: GRID };
const borders = { top: border, bottom: border, left: border, right: border };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: none, bottom: none, left: none, right: none };

function dataTable(b) {
  const sz = Math.round((b.font || 10) * 2);
  const widths = b.widths.map(w => Math.floor(w * TW));
  widths[widths.length - 1] = TW - widths.slice(0, -1).reduce((a, c) => a + c, 0);
  const mkRow = (cells, header) => new TableRow({
    tableHeader: header, cantSplit: true,
    children: cells.map((c, i) => new TableCell({
      borders, width: { size: widths[i], type: WidthType.DXA },
      shading: header ? { fill: SOFT, type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 50, bottom: 50, left: 90, right: 90 },
      children: [new Paragraph({ children: runs(c, { size: sz, bold: header || undefined }),
        spacing: { after: 0, line: 260 } })],
    })),
  });
  return new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: widths,
    rows: [mkRow(b.header, true), ...b.rows.map(r => mkRow(r, false))] });
}

function pngSize(path) {
  const buf = fs.readFileSync(path);
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20), buf };
}

function figure(b) {
  const { w, h, buf } = pngSize(b.path);
  let width = Math.round(TWPX * (b.width || 1));
  let height = Math.round(width * h / w);
  const maxh = Math.round(THPX * 0.78);
  if (height > maxh) { height = maxh; width = Math.round(maxh * w / h); }
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
        altText: { title: b.caption, description: b.caption, name: b.caption } })] }),
    new Paragraph({ style: "FigureCaption", children: runs(b.caption) }),
  ];
}

function wrapCode(text, max = 100) {
  const out = [];
  for (const line of text.split("\n")) {
    if (line.length <= max) { out.push(line); continue; }
    const ind = line.length - line.trimStart().length;
    let rest = line;
    let first = true;
    while (rest.length > max) {
      let cut = rest.lastIndexOf(" ", max);
      if (cut <= ind + 4) cut = max;
      out.push(rest.slice(0, cut));
      rest = " ".repeat(ind + 4) + rest.slice(cut).trimStart();
      first = false;
    }
    out.push(rest);
  }
  return out;
}

function codeBlock(b) {
  const out = [];
  if (b.title) out.push(new Paragraph({ children: runs(b.title, { bold: true, size: 21 }), keepNext: true,
    spacing: { before: 120, after: 60 } }));
  const lines = wrapCode(b.text);
  lines.forEach((l, i) => out.push(new Paragraph({ style: "Code", keepNext: lines.length < 30 && i < lines.length - 1,
    children: [new TextRun({ text: l.length ? l : " " })] })));
  out.push(new Paragraph({ children: [], spacing: { after: 80 } }));
  return out;
}

function note(b) {
  const warn = b.kind === "warn";
  const bd = { style: BorderStyle.SINGLE, size: 6, color: warn ? WARNB : BLUE, space: 6 };
  return new Paragraph({ children: runs(b.text, { size: 23 }), alignment: AlignmentType.JUSTIFIED,
    shading: { fill: warn ? WARN : SOFT, type: ShadingType.CLEAR, color: "auto" },
    border: { top: bd, bottom: bd, left: bd, right: bd }, spacing: { before: 120, after: 200, line: 320 },
    indent: { left: 120, right: 120 } });
}

function signatures(b) {
  const n = b.items.length;
  const w = Math.floor(TW / n);
  const widths = Array(n).fill(w); widths[n - 1] = TW - w * (n - 1);
  return [new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: widths,
    rows: [new TableRow({ children: b.items.map(([title, name], i) => new TableCell({
      borders: noBorders, width: { size: widths[i], type: WidthType.DXA },
      children: [
        new Paragraph({ children: [], spacing: { before: 700 } }),
        C("____________________"),
        C(`**${title}**`),
        ...name.split("\n").filter(Boolean).map(x => C(x)),
      ] })) })] }), new Paragraph({ children: [], spacing: { after: 240 } })];
}

function titlePage(b) {
  const logo = label => new TableCell({ borders, width: { size: 2268, type: WidthType.DXA },
    verticalAlign: VerticalAlign.CENTER, children: [C(label, { spacing: { before: 600, after: 600 } })] });
  const gap = TW - 2 * 2268;
  const out = [new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: [2268, gap, 2268],
    rows: [new TableRow({ children: [logo("College logo"),
      new TableCell({ borders: noBorders, width: { size: gap, type: WidthType.DXA }, children: [new Paragraph("")] }),
      logo("Department logo")] })] })];
  const big = (t, size, after = 120) => C(t, { spacing: { after, line: 300 } }, { bold: true, size, color: NAVY });
  out.push(new Paragraph({ children: [], spacing: { after: 500 } }));
  out.push(big(b.title, 40, 360));
  out.push(C(b.kind), big(b.degree, 27, 400), C(b.by));
  b.students.forEach(s => out.push(C(`**${s}**`)));
  out.push(new Paragraph({ children: [], spacing: { after: 300 } }), C(b.guide),
    new Paragraph({ children: [], spacing: { after: 700 } }), big(b.dept, 28, 60), big(b.college, 30, 60),
    C(b.univ), new Paragraph({ children: [], spacing: { after: 200 } }), C(`**${b.year}**`));
  return out;
}

// ------------------------------------------------------------------ walk blocks into sections
const sections = { title: [], front: [], main: [] };
let target = "title";
let numberedInstance = 0;
let pendingBreak = false;

function push(...items) { sections[target].push(...items); }

function heading(text, level, opts = {}) {
  return new Paragraph({ heading: level, children: [new TextRun({ text })], ...opts });
}

for (const b of blocks) {
  switch (b.t) {
    case "titlepage": push(...titlePage(b)); target = "front"; break;
    case "frontheading":
      push(heading(b.text, HeadingLevel.HEADING_1, { alignment: AlignmentType.CENTER,
        pageBreakBefore: sections.front.length > 0 && !pendingBreak }));
      pendingBreak = false; break;
    case "pagebreak": push(new Paragraph({ children: [new PageBreak()] })); pendingBreak = true; break;
    case "toc":
    case "lof":
    case "lot": {
      const title = { toc: "Table of Contents", lof: "List of Figures", lot: "List of Tables" }[b.t];
      push(heading(title, HeadingLevel.HEADING_1, { alignment: AlignmentType.CENTER, pageBreakBefore: !pendingBreak }));
      pendingBreak = false;
      if (b.t === "toc") push(new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }));
      else push(new TableOfContents(title, { hyperlink: true,
        stylesWithLevels: [new StyleLevel(b.t === "lof" ? "Figure Caption" : "Table Caption", 1)] }));
      push(P("*Right-click and choose Update Field (or press F9) if page numbers are not shown.*",
        { spacing: { before: 120 } }));
      break;
    }
    case "abbreviations": {
      const w1 = 1700, w2 = TW - w1;
      const bottom = { style: BorderStyle.SINGLE, size: 2, color: "DDE2EA" };
      push(new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: [w1, w2],
        rows: b.rows.map(([a, d]) => new TableRow({ children: [
          new TableCell({ borders: { ...noBorders, bottom }, width: { size: w1, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: a, bold: true, size: 22 })], spacing: { after: 20 } })] }),
          new TableCell({ borders: { ...noBorders, bottom }, width: { size: w2, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: d, size: 22 })], spacing: { after: 20 } })] }),
        ] })) }));
      break;
    }
    case "mainmatter": target = "main"; break;
    case "h1": {
      const first = sections.main.length === 0;
      if (b.num) {
        push(new Paragraph({ alignment: AlignmentType.CENTER, pageBreakBefore: !first, spacing: { before: 600, after: 60 },
          keepNext: true, children: [new TextRun({ text: b.text.toUpperCase(), bold: true, size: 32, color: NAVY })] }));
        push(heading(`${b.num}. ${b.sub}`, HeadingLevel.HEADING_1, { alignment: AlignmentType.CENTER }));
      } else {
        push(heading(b.text, HeadingLevel.HEADING_1, { alignment: AlignmentType.CENTER, pageBreakBefore: !first,
          spacing: { before: 600, after: 360 } }));
      }
      break;
    }
    case "h2": push(heading(b.text, HeadingLevel.HEADING_2)); break;
    case "h3": push(heading(b.text, HeadingLevel.HEADING_3)); break;
    case "p": push(P(b.text)); break;
    case "bullets":
      b.items.forEach(i => push(new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(i),
        alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 340 } })));
      push(new Paragraph({ children: [], spacing: { after: 60 } }));
      break;
    case "numbered":
      numberedInstance += 1;
      b.items.forEach(i => push(new Paragraph({ numbering: { reference: "numbers", level: 0, instance: numberedInstance },
        children: runs(i), alignment: AlignmentType.JUSTIFIED, spacing: { after: 60, line: 340 } })));
      push(new Paragraph({ children: [], spacing: { after: 60 } }));
      break;
    case "table":
      push(new Paragraph({ style: "TableCaption", children: runs(b.caption) }), dataTable(b),
        new Paragraph({ children: [], spacing: { after: 160 } }));
      break;
    case "figure": push(...figure(b)); break;
    case "code": push(...codeBlock(b)); break;
    case "equation": push(new Paragraph({ alignment: AlignmentType.CENTER, children: eqRuns(b.text),
      spacing: { after: 100, line: 320 } })); break;
    case "note": push(note(b)); break;
    case "signatures": push(...signatures(b)); break;
    case "references":
      b.items.forEach((r, i) => push(new Paragraph({ children: [new TextRun({ text: `[${i + 1}]\t` }), ...runs(r, { size: 22 })],
        alignment: AlignmentType.JUSTIFIED, indent: { left: 720, hanging: 720 }, tabStops: [{ type: "left", position: 720 }],
        spacing: { after: 100, line: 300 } })));
      break;
    default: console.warn("unknown block", b.t);
  }
}

// ------------------------------------------------------------------ document
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 21 })] })] });
const header = new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "B8C0CC", space: 2 } },
  children: [new TextRun({ text: "AI-Driven Personalized Learning and Remedial Recommendation", italics: true, size: 18 })] })] });
const page = { size: { width: PAGE_W, height: PAGE_H }, margin: { top: MT, bottom: MB, left: ML, right: MR } };

const doc = new Document({
  creator: "[Student names]",
  title: "AI-Driven Personalized Learning and Remedial Recommendation - Project Report",
  description: "Final year project report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 120, after: 360 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: FONT, color: NAVY }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 2, keepNext: true } },
      { id: "FigureCaption", name: "Figure Caption", basedOn: "Normal", next: "Normal",
        run: { size: 22, bold: true, font: FONT }, paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 } } },
      { id: "TableCaption", name: "Table Caption", basedOn: "Normal", next: "Normal",
        run: { size: 22, bold: true, font: FONT }, paragraph: { alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160, after: 100 } } },
      { id: "Code", name: "Code", basedOn: "Normal", next: "Normal",
        run: { size: 16, font: MONO }, paragraph: { spacing: { before: 0, after: 0, line: 220 },
          shading: { fill: CODEBG, type: ShadingType.CLEAR, color: "auto" }, indent: { left: 120, right: 120 } } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [
    { properties: { page }, children: sections.title },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.LOWER_ROMAN } } },
      footers: { default: footer }, children: sections.front },
    { properties: { page: { ...page, pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL } } },
      headers: { default: header }, footers: { default: footer }, children: sections.main },
  ],
});

Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("written", OUT, buf.length, "bytes"); });
