"""Build the exercise and solution notebooks from the plain-text sources.

Sources live in notebooks/src/*.py and use a light "percent" format:

    # %% [markdown]            -> markdown cell (lines start with "# ")
    # %% [markdown] solution   -> markdown cell kept only in the solution notebook
    # %%                       -> code cell
    # %% notest                -> code cell (skipped by the offline test runner)

Inside code cells:

    ### BEGIN SOLUTION
    ...                        -> kept in solutions, removed from exercises
    ### END SOLUTION
    ### EXERCISE: <code line>  -> only in the exercise notebook

The token {{COLAB}} in markdown is replaced by an "Open in Colab" badge.

Usage:  python scripts/build_notebooks.py [--test-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = "hayg25/Lectures_ML4Accelerators"
BRANCH = "main"
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "src"
OUT_EX = ROOT / "notebooks"
OUT_SOL = ROOT / "notebooks" / "solutions"


def colab_badge(rel_path: str) -> str:
    url = f"https://colab.research.google.com/github/{REPO}/blob/{BRANCH}/{rel_path}"
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({url})"


def parse(text: str):
    cells, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^# %%(.*)$", line)
        if m:
            if cur:
                cells.append(cur)
            tag = m.group(1).strip()
            if tag.startswith("[markdown]"):
                cur = {"type": "markdown", "solution": "solution" in tag, "lines": []}
            else:
                cur = {"type": "code", "notest": "notest" in tag, "lines": []}
            continue
        if cur is None:
            continue
        cur["lines"].append(line)
    if cur:
        cells.append(cur)
    return cells


def strip_blank(lines):
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def md_text(lines):
    out = []
    for ln in lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln.strip() == "#":
            out.append("")
        else:
            out.append(ln)
    return strip_blank(out)


def code_text(lines, solution: bool):
    out, in_sol = [], False
    for ln in lines:
        s = ln.strip()
        if s == "### BEGIN SOLUTION":
            in_sol = True
            continue
        if s == "### END SOLUTION":
            in_sol = False
            continue
        m = re.match(r"^(\s*)### EXERCISE: ?(.*)$", ln)
        if m:
            if not solution:
                out.append(m.group(1) + m.group(2))
            continue
        if in_sol and not solution:
            continue
        out.append(ln)
    return strip_blank(out)


def to_source(lines):
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines else [])


def build(src: Path, solution: bool):
    rel = ("notebooks/solutions/" if solution else "notebooks/") + src.stem + ".ipynb"
    cells = []
    for c in parse(src.read_text(encoding="utf-8")):
        if c["type"] == "markdown":
            if c["solution"] and not solution:
                continue
            lines = md_text(c["lines"])
            lines = [ln.replace("{{COLAB}}", colab_badge(rel)) for ln in lines]
            if solution and lines and lines[0].startswith("# "):
                lines[0] += " — solutions"
            cells.append({"cell_type": "markdown", "metadata": {}, "source": to_source(lines)})
        else:
            lines = code_text(c["lines"], solution)
            if not lines:
                continue
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": to_source(lines)})
    nb = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return rel, nb


def test_script(src: Path) -> str:
    """Solution code as a flat script (no shell magics, no 'notest' cells)."""
    chunks = []
    for c in parse(src.read_text(encoding="utf-8")):
        if c["type"] != "code" or c["notest"]:
            continue
        lines = [(ln[: len(ln) - len(ln.lstrip())] + "pass") if ln.strip().startswith("!") else ln
                 for ln in code_text(c["lines"], True)]
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", type=Path, default=None)
    args = ap.parse_args()
    OUT_SOL.mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC.glob("*.py")):
        for sol in (False, True):
            rel, nb = build(src, sol)
            (ROOT / rel).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            print("wrote", rel)
        if args.test_dir:
            args.test_dir.mkdir(parents=True, exist_ok=True)
            (args.test_dir / (src.stem + ".py")).write_text(test_script(src), encoding="utf-8")


if __name__ == "__main__":
    main()
