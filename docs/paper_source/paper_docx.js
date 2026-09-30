// Render paper_blocks.json as an IEEE-style two-column Word paper (US Letter).
const fs = require("fs");
const path = require("path");
const docx = require(path.join(process.argv[4], "node_modules", "docx"));
const { Document, Packer, Paragraph, TextRun, AlignmentType, Table, TableRow, TableCell, WidthType, BorderStyle,
  ImageRun, SectionType, Footer, PageNumber, TabStopType } = docx;

const blocks = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const OUT = process.argv[3];
const FONT = "Times New Roman";
const PW = 12240, PH = 15840, ML = 900, MR = 900, MT = 1080, MB = 1440, GAP = 360;
const FULL = PW - ML - MR;
const CW = Math.floor((FULL - GAP) / 2);
const CWPX = Math.round(CW / 1440 * 96);
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"];

const GREEK = { theta: "θ", tau: "τ", lambda: "λ", sigma: "σ" };
const greek = t => t.replace(/(?<![A-Za-z])(theta|tau|lambda|sigma)(?![A-Za-z])/g, g => GREEK[g])
  .replace(/<=/g, "≤").replace(/>=/g, "≥");

// plain text -> runs, with x_t, h_{t-1}, theta_k as italic letters with subscripts
function mathRuns(text, base = {}) {
  text = greek(text);
  const out = [];
  const re = /([A-Za-zθτλσ]'?)_(?:\{([^}]*)\}|([A-Za-z0-9+]+))/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    out.push(new TextRun({ text: m[1], ...base, italics: true }));
    out.push(new TextRun({ text: m[2] ?? m[3], ...base, italics: true, subScript: true }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+?\*\*|`[^`]+?`|(?<![\w*])\*[A-Za-z][^*\n]*?\*(?![\w*]))/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(...mathRuns(text.slice(last, m.index), base));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else if (tok.startsWith("`")) out.push(new TextRun({ text: tok.slice(1, -1), font: "Consolas", ...base, size: 17 }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(...mathRuns(text.slice(last), base));
  return out;
}

function eqRuns(text) {
  text = greek(text);
  const out = [];
  const re = /_\{([^}]*)\}|_([A-Za-z0-9]+)|\^\{([^}]*)\}/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), italics: true }));
    if (m[3] !== undefined) out.push(new TextRun({ text: m[3], italics: true, superScript: true }));
    else out.push(new TextRun({ text: m[1] ?? m[2], italics: true, subScript: true }));
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), italics: true }));
  return out;
}

const smallCaps = (text, size = 20) => new TextRun({ text, smallCaps: true, size });
const line = { style: BorderStyle.SINGLE, size: 8, color: "000000" };
const thin = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

const top = [], body = [];
let sec = 0, sub = 0, fig = 0, tab = 0, eqn = 0;

for (const b of blocks) {
  switch (b.t) {
    case "title": {
      top.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new TextRun({ text: b.title, size: 44 })] }));
      const n = b.authors.length, w = Math.floor(FULL / n);
      top.push(new Table({ width: { size: FULL, type: WidthType.DXA }, columnWidths: Array(n).fill(w),
        rows: [new TableRow({ children: b.authors.map(a => new TableCell({
          borders: { top: none, bottom: none, left: none, right: none }, width: { size: w, type: WidthType.DXA },
          children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: a.name, italics: true, size: 20 })] }),
            ...a.lines.map(l => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 },
              children: [new TextRun({ text: l, size: 20 })] }))],
        })) })] }));
      top.push(new Paragraph({ children: [], spacing: { after: 240 } }));
      break;
    }
    case "abstract":
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 228 },
        children: [new TextRun({ text: "Abstract", bold: true, italics: true, size: 18 }),
          new TextRun({ text: "—", bold: true, size: 18 }), ...runs(b.text, { bold: true, size: 18 })] }));
      break;
    case "keywords":
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 228 },
        children: [new TextRun({ text: "Index Terms", bold: true, italics: true, size: 18 }),
          new TextRun({ text: "—", bold: true, size: 18 }), ...runs(b.text, { bold: true, size: 18 })] }));
      break;
    case "h1": {
      sec += 1; sub = 0;
      const label = b.unnumbered ? b.text : `${ROMAN[sec - 1]}. ${b.text}`;
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
        children: [smallCaps(label)] }));
      break;
    }
    case "h2":
      sub += 1;
      body.push(new Paragraph({ keepNext: true, spacing: { before: 120, after: 60 },
        children: [new TextRun({ text: `${String.fromCharCode(64 + sub)}. ${b.text}`, italics: true, size: 20 })] }));
      break;
    case "p":
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { firstLine: 240 },
        spacing: { after: 40, line: 240 }, children: runs(b.text, { size: 20 }) }));
      break;
    case "bullets":
      b.items.forEach(i => body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 300, hanging: 200 },
        spacing: { after: 30, line: 240 }, children: [new TextRun({ text: "•\t", size: 20 }), ...runs(i, { size: 20 })],
        tabStops: [{ type: TabStopType.LEFT, position: 300 }] })));
      break;
    case "numbered":
      b.items.forEach((i, k) => body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, indent: { left: 340, hanging: 280 },
        spacing: { after: 30, line: 240 }, children: [new TextRun({ text: `${k + 1})\t`, size: 20 }), ...runs(i, { size: 20 })],
        tabStops: [{ type: TabStopType.LEFT, position: 340 }] })));
      break;
    case "equation":
      eqn += 1;
      body.push(new Paragraph({ spacing: { before: 60, after: 60 },
        tabStops: [{ type: TabStopType.CENTER, position: Math.floor(CW / 2) }, { type: TabStopType.RIGHT, position: CW }],
        children: [new TextRun({ text: "\t", size: 20 }), ...eqRuns(b.text).map(r => r), new TextRun({ text: `\t(${eqn})`, size: 20 })] }));
      break;
    case "figure": {
      fig += 1;
      const buf = fs.readFileSync(b.path);
      const w0 = buf.readUInt32BE(16), h0 = buf.readUInt32BE(20);
      const width = Math.round(CWPX * (b.width || 1)), height = Math.round(width * h0 / w0);
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 100, after: 40 },
        children: [new ImageRun({ type: "png", data: buf, transformation: { width, height },
          altText: { title: b.caption, description: b.caption, name: `Figure ${fig}` } })] }));
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 160, line: 216 },
        children: [new TextRun({ text: `Fig. ${fig}. `, size: 16 }), ...runs(b.caption, { size: 16 })] }));
      break;
    }
    case "table": {
      tab += 1;
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 0 },
        children: [new TextRun({ text: `TABLE ${ROMAN[tab - 1]}`, size: 16 })] }));
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 60 },
        children: [smallCaps(b.caption, 16)] }));
      const widths = b.widths.map(w => Math.floor(w * CW));
      widths[widths.length - 1] = CW - widths.slice(0, -1).reduce((a, c) => a + c, 0);
      const nrows = b.rows.length;
      const mk = (cells, ri) => new TableRow({ tableHeader: ri === 0, cantSplit: true, children: cells.map((c, i) =>
        new TableCell({ width: { size: widths[i], type: WidthType.DXA },
          borders: { left: none, right: none, top: ri === 0 ? line : none,
            bottom: ri === 0 ? thin : (ri === nrows ? line : none) },
          margins: { top: 20, bottom: 20, left: 50, right: 50 },
          children: [new Paragraph({ spacing: { after: 0, line: 200 }, children: runs(c, { size: 16, bold: ri === 0 || undefined }) })] })) });
      body.push(new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: widths,
        rows: [mk(b.header, 0), ...b.rows.map((r, k) => mk(r, k + 1))] }));
      body.push(new Paragraph({ children: [], spacing: { after: 120 } }));
      break;
    }
    case "ack":
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
        children: [smallCaps("Acknowledgment")] }));
      body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 40, line: 240 }, children: runs(b.text, { size: 20 }) }));
      break;
    case "references":
      body.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 100 },
        children: [smallCaps("References")] }));
      b.items.forEach((r, i) => body.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED,
        indent: { left: 360, hanging: 360 }, tabStops: [{ type: TabStopType.LEFT, position: 360 }],
        spacing: { after: 30, line: 200 }, children: [new TextRun({ text: `[${i + 1}]\t`, size: 16 }), ...runs(r, { size: 16 })] })));
      break;
    default: console.warn("unknown", b.t);
  }
}

const page = { size: { width: PW, height: PH }, margin: { top: MT, bottom: MB, left: ML, right: MR } };
const footer = new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
  children: [new TextRun({ children: [PageNumber.CURRENT], size: 16 })] })] });
const doc = new Document({
  creator: "[Authors]", title: blocks[0].title,
  styles: { default: { document: { run: { font: FONT, size: 20 } } } },
  sections: [
    { properties: { page }, footers: { default: footer }, children: top },
    { properties: { page, type: SectionType.CONTINUOUS, column: { count: 2, space: GAP, equalWidth: true } },
      footers: { default: footer }, children: body },
  ],
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log("written", OUT, buf.length); });
