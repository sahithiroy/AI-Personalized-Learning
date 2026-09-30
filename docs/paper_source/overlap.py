import re, sys
from pypdf import PdfReader
def words(path):
    t = " ".join(p.extract_text() or "" for p in PdfReader(path).pages)
    t = re.sub(r"-\s*\n\s*", "", t)
    return re.findall(r"[a-z0-9]+", t.lower())
def grams(w, n): return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
paper = " ".join(p.extract_text() for p in PdfReader("Research_Paper.pdf").pages)
body = paper.split("REFERENCES")[0] if "REFERENCES" in paper else paper.split("References")[0]
pw_all = re.findall(r"[a-z0-9]+", re.sub(r"-\s*\n\s*", "", paper).lower())
pw_body = re.findall(r"[a-z0-9]+", re.sub(r"-\s*\n\s*", "", body).lower())
srcs = {"Base IEEE paper": sys.argv[1], "Your project report": sys.argv[2], "Project_Explained.pdf": sys.argv[3]}
union8 = set()
for name, path in srcs.items():
    sw = words(path)
    for n in (8, 5):
        g = grams(sw, n)
        pb = grams(pw_body, n); pa = grams(pw_all, n)
        hb = pb & g
        if n == 8: union8 |= hb
        print(f"{name:24} {n}-word: body {100*len(hb)/len(pb):5.1f}%   with references {100*len(pa & g)/len(pa):5.1f}%")
pb8 = grams(pw_body, 8)
print(f"{'ALL SOURCES (8-word, body)':24} {100*len(union8)/len(pb8):5.1f}%")
print("paper words:", len(pw_all))
