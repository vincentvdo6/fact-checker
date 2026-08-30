"""The three training notebooks must stay one notebook with three configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_notebooks import VARIANTS, cells, notebook, undecorate

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "notebooks" / "verdict_notebook.py"


def source_cells(variant: str) -> list[str]:
    path = ROOT / "notebooks" / f"kaggle_verdict_{variant}.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in payload["cells"]]


def test_the_committed_notebooks_match_the_template():
    """
    A stale notebook is the failure mode this generator exists to prevent: it would train on a
    config nobody can see in the diff, and a .ipynb diff is unreadable by hand.
    """
    template = TEMPLATE.read_text(encoding="utf-8")
    for variant in VARIANTS:
        path = ROOT / "notebooks" / f"kaggle_verdict_{variant}.ipynb"
        assert path.exists(), f"{path.name} has not been generated"
        expected = json.dumps(notebook(template, variant), indent=1) + "\n"
        assert path.read_text(encoding="utf-8") == expected, (
            f"{path.name} is stale; run python -m scripts.make_notebooks"
        )


def test_the_variants_differ_only_in_the_title_and_config():
    variants = {name: source_cells(name) for name in VARIANTS}
    counts = {len(v) for v in variants.values()}
    assert len(counts) == 1, f"cell counts differ: {counts}"

    differing = [
        i for i in range(counts.pop())
        if len({v[i] for v in variants.values()}) > 1
    ]
    assert len(differing) == 2, f"expected the title and config cells to differ, got {differing}"
    assert "Verdict model" in variants["retrieved"][differing[0]]
    assert "CFG = dict(" in variants["retrieved"][differing[1]]


@pytest.mark.parametrize("variant,expected", [
    ("retrieved", 512),
    ("claim_only", 64),      # no evidence, so the sequence budget is a fraction
    ("gold", 512),
])
def test_each_notebook_carries_its_own_budget(variant, expected):
    config = [c for c in source_cells(variant) if "CFG = dict(" in c][0]
    assert f'variant="{variant}"' in config
    assert f"max_length={expected}," in config


def test_notebooks_are_committed_without_outputs():
    """Outputs would dwarf the source in the diff and leak whatever the last run printed."""
    for variant in VARIANTS:
        payload = json.loads((ROOT / "notebooks" / f"kaggle_verdict_{variant}.ipynb").read_text(encoding="utf-8"))
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None


def test_cells_split_on_the_markers():
    parsed = cells("# %% [markdown]\n# a heading\n\n# %%\nx = 1\n\n# %%\ny = 2\n")
    assert [kind for kind, _ in parsed] == ["markdown", "code", "code"]
    assert parsed[1][1] == "x = 1"


def test_empty_cells_are_dropped():
    assert cells("# %%\n\n# %%\nx = 1\n") == [("code", "x = 1")]


def test_markdown_loses_its_comment_prefix():
    assert undecorate("# # Title\n#\n# body") == "# Title\n\nbody"


def test_the_template_is_importable_python():
    """It is a real module, which is why ruff checks it and this test can parse it."""
    import ast

    ast.parse(TEMPLATE.read_text(encoding="utf-8"))


def test_the_template_names_the_dataset_to_attach():
    """The asserts are the notebook's error messages; they have to say what to do."""
    text = TEMPLATE.read_text(encoding="utf-8")
    assert "attach the fever-verdict-v1 dataset" in text
    assert "smoke" in text and "resume" in text
