"""The three training notebooks must stay one notebook with three configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.make_notebooks import DATASETS, VARIANTS, cells, kernel_metadata, notebook, undecorate

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


def test_the_committed_notebooks_are_full_runs_not_rehearsals():
    """
    Smoke trains 2,048 rows for 60 steps. A notebook committed with smoke on would run the whole
    phase as a rehearsal and export an artifact the installer then refuses, after the GPU hours
    are already spent.
    """
    source = TEMPLATE.read_text(encoding="utf-8")
    for variant in VARIANTS:
        cfg = json.dumps(notebook(source, variant))
        assert "smoke=False" in cfg, f"{variant} is committed as a rehearsal"
        assert "smoke=True" not in cfg


def test_rehearsal_mode_turns_smoke_on():
    source = TEMPLATE.read_text(encoding="utf-8")
    cfg = json.dumps(notebook(source, "retrieved", smoke=True))
    assert "smoke=True" in cfg and "smoke=False" not in cfg


def test_every_kernel_points_at_a_notebook_that_exists():
    """A wrong code_file fails at push time, after the dataset has already been uploaded."""
    for name in DATASETS:
        code_file = kernel_metadata(name, "owner")["code_file"]
        assert (ROOT / "notebooks" / code_file).exists(), f"{name} -> {code_file}"


def test_the_regrounded_kernel_trains_on_v2_and_the_rest_on_v1():
    """
    The whole point of Phase 05 is a comparison, and it evaporates if the new kernel is pointed at
    the old data -- the run would succeed and produce a second copy of the control.
    """
    assert kernel_metadata("retrieved_grounded", "o")["dataset_sources"] == ["o/fever-verdict-v2"]
    for name in ("retrieved", "claim_only", "gold"):
        assert kernel_metadata(name, "o")["dataset_sources"] == ["o/fever-verdict-v1"], name


def test_the_regrounded_kernel_reuses_the_retrieved_notebook():
    """Same encoder, same config, different data. A fourth copy would only invite drift."""
    assert kernel_metadata("retrieved_grounded", "o")["code_file"] == "kaggle_verdict_retrieved.ipynb"
    assert kernel_metadata("retrieved", "o")["code_file"] == "kaggle_verdict_retrieved.ipynb"


def test_kernel_ids_are_distinct():
    """Two kernels sharing an id means the second push overwrites the first one's run."""
    ids = [kernel_metadata(name, "owner")["id"] for name in DATASETS]
    assert len(set(ids)) == len(ids)


def test_kernels_stay_private():
    for name in DATASETS:
        assert kernel_metadata(name, "owner")["is_private"] is True, name


def test_the_committed_kernel_definitions_match_the_generator():
    """Hand-editing one would make the committed run and the generated one disagree silently."""
    for name in DATASETS:
        path = ROOT / "notebooks" / "kernels" / f"{name}.json"
        assert path.exists(), f"{path} is missing; run python -m scripts.make_notebooks"
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored == kernel_metadata(name, stored["id"].split("/")[0]), name
