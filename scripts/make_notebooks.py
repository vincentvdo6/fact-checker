"""
Generate the three training notebooks from one template.

The three variants must differ only in their config. Three hand-maintained .ipynb files would
drift on the first edit, and the drift would be invisible -- a notebook is JSON, so a diff
between two of them is unreadable. Generating them from `notebooks/verdict_notebook.py` makes
identity structural: the template is ordinary Python that ruff checks and a test can import,
and the only thing this script varies is three values in one cell.

Cells are split on the `# %%` markers the jupytext and VS Code conventions already use, so the
template opens as a notebook in an editor without any of this running.

Notebooks are written with no outputs and no execution counts, so a commit carries source only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TEMPLATE = Path("notebooks/verdict_notebook.py")
OUT = Path("notebooks")

# Only these three values differ. claim_only needs a fraction of the sequence budget because it
# sees no evidence, which is why it trains in about a tenth of the time and should be run first:
# a strong evidence-free baseline is a reason to stop and think, not a footnote.
VARIANTS = {
    "retrieved": {"variant": "retrieved", "max_length": 512, "time_budget_h": 10.5},
    "claim_only": {"variant": "claim_only", "max_length": 64, "time_budget_h": 6.0},
    "gold": {"variant": "gold", "max_length": 512, "time_budget_h": 10.5},
}

HEADINGS = {
    "retrieved": "retrieved evidence",
    "claim_only": "claim only (evidence-free baseline)",
    "gold": "gold evidence (reasoning ceiling)",
}


def cells(source: str) -> list[tuple[str, str]]:
    """Split the template into (kind, body) on its cell markers."""
    out: list[tuple[str, str]] = []
    kind, buffer = "code", []
    for line in source.splitlines():
        if line.startswith("# %%"):
            if buffer:
                out.append((kind, "\n".join(buffer).strip("\n")))
            kind = "markdown" if line.strip() == "# %% [markdown]" else "code"
            buffer = []
            continue
        buffer.append(line)
    if buffer:
        out.append((kind, "\n".join(buffer).strip("\n")))
    return [(k, b) for k, b in out if b.strip()]


def undecorate(markdown: str) -> str:
    """Markdown cells live behind `# ` in the template; strip it to get the prose back."""
    return "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in markdown.splitlines())


def notebook(source: str, variant: str, *, smoke: bool = False) -> dict:
    config = VARIANTS[variant]
    body = source
    body = body.replace("# # Verdict model -- retrieved", f"# # Verdict model -- {HEADINGS[variant]}")
    body = body.replace('    variant="retrieved",', f'    variant="{config["variant"]}",')
    body = body.replace("    max_length=512,", f"    max_length={config['max_length']},")
    body = body.replace("    time_budget_h=10.5,", f"    time_budget_h={config['time_budget_h']},")
    # The committed notebook is the real run. Smoke is a rehearsal of the failure paths -- 2,048
    # rows, 60 steps, one deliberate kill and resume -- so it is generated on demand into a
    # scratch directory rather than being the artifact the repository ships.
    body = body.replace("    smoke=True,", f"    smoke={smoke},")

    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": (undecorate(text) if kind == "markdown" else text).splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, text in cells(body)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the notebooks are stale")
    parser.add_argument("--smoke", action="store_true", help="rehearsal notebooks; requires --out")
    parser.add_argument("--out", default=str(OUT), help="destination directory (default notebooks/)")
    args = parser.parse_args()
    if args.smoke and Path(args.out) == OUT:
        raise SystemExit("--smoke writes rehearsal notebooks; point --out at a scratch directory")

    source = TEMPLATE.read_text(encoding="utf-8")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stale: list[str] = []
    for variant in VARIANTS:
        path = out / f"kaggle_verdict_{variant}.ipynb"
        rendered = json.dumps(notebook(source, variant, smoke=args.smoke), indent=1) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                stale.append(path.name)
            continue
        # Explicit LF so the working tree matches what git stores; on Windows the default
        # would dirty all three files with line-ending churn on every regeneration.
        path.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"  wrote {path.name}  ({len(rendered) / 1024:.0f} KB)")

    if args.check:
        if stale:
            print(f"stale: {', '.join(stale)}\nrun: python -m scripts.make_notebooks")
            return 1
        print("notebooks match the template")
    return 0


if __name__ == "__main__":
    sys.exit(main())
