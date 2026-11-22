"""Build the written lecture notes (lecture-notes.qmd) from the slides.

For every slide of slides/ml4acc.qmd we keep: the title, the main figure (for animations, the
last frame, which contains everything) and the speaker notes, cleaned of "[click]" marks.
The anticipated questions & self-check quiz are included from _questions.qmd.

Usage:  python scripts/make_handout.py   (run before `quarto render`)
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "slides" / "ml4acc.qmd"
OUT = ROOT / "lecture-notes.qmd"

HEADER = """---
title: "Lecture notes"
subtitle: "Machine Learning for Accelerator Physics — the slides, explained"
format:
  html:
    theme: cosmo
    css: styles.css
    toc: true
    toc-depth: 2
---

::: {.callout-note}
These notes follow the [slides](slides/ml4acc.html) one by one: each section shows the main
figure of the slide and the explanation given orally. They are generated automatically from the
speaker notes of the slides (`python scripts/make_handout.py`). At the end you will find
[frequently asked questions](#frequently-asked-questions) and a [self-check quiz](#self-check-quiz).
:::

"""

IMG = re.compile(r"!\[[^\]]*\]\((\.\./assets/[^)]+\.svg)\)")


def clean_title(line):
    t = line.lstrip("#").strip()
    t = re.sub(r"\{[^}]*\}\s*$", "", t).strip()        # drop {background-color=...}
    return t


def split_slides(text):
    body = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    slides, cur = [], None
    in_code = False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,2} ", line):
            if cur:
                slides.append(cur)
            cur = {"level": len(line) - len(line.lstrip("#")), "title": clean_title(line), "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        slides.append(cur)
    return slides


def notes_and_figure(lines):
    text = "\n".join(lines)
    m = re.search(r"::: notes\n(.*?)\n:::", text, re.S)
    notes = m.group(1).strip() if m else ""
    notes = re.sub(r"[ \t]*\[click\][ \t]*", " ", notes)
    notes = re.sub(r" {2,}", " ", notes)
    notes = "\n".join(l.strip() for l in notes.splitlines())
    body = text[:m.start()] if m else text
    imgs = IMG.findall(body)
    # For an r-stack animation, the last frame is the most complete; the learning-rate figure
    # that ends the gradient-descent stack is a different figure, so keep both in that case.
    figs = []
    if imgs:
        anim = [i for i in imgs if "/anim/" in i]
        other = [i for i in imgs if "/anim/" not in i]
        if anim:
            figs.append(anim[-1])
        figs += other
    figs = [f.replace("../", "") for f in figs]
    table = "\n".join(l for l in body.splitlines() if l.startswith("|"))
    return notes, figs, table


def main():
    slides = split_slides(SRC.read_text())
    out = [HEADER]
    for s in slides:
        notes, figs, table = notes_and_figure(s["lines"])
        if s["title"].startswith("Questions?") or s["title"].startswith("Check your understanding"):
            continue
        if s["level"] == 1 or (not notes and not figs):
            out.append(f"# {s['title']}\n")
            continue
        out.append(f"## {s['title']}\n")
        for f in figs:
            out.append(f"![]({f}){{width=\"85%\" fig-align=\"center\"}}\n")
        if table:
            out.append(table + "\n")
        if notes and not notes.startswith("Backup slide"):
            out.append(notes + "\n")
    out.append("{{< include _questions.qmd >}}\n")
    OUT.write_text("\n".join(out))
    print("wrote", OUT.relative_to(ROOT), f"({len(slides)} slides)")


if __name__ == "__main__":
    main()
