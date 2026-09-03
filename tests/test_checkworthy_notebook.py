"""
The check-worthiness notebook, and the metric it reports from across the Kaggle boundary.

`scores()` runs on Kaggle and produces the numbers that come back in metrics.json. Nothing
downstream re-derives them, so a swapped precision and recall, or a binarization that counts the
wrong classes, would be reported as fact and never contradicted. The function is lifted out of the
template and executed here rather than reimplemented -- reimplementing it would test the copy.

The generator is separate from `make_notebooks.py` deliberately. That one hard-codes a
`fever-verdict-` prefix into every kernel id, which is already wrong for the AVeriTeC runs and
would be wronger here: this is a different task, on a different corpus, with a different label
space. The test below pins that this kernel is not named after FEVER.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.make_checkworthy_notebook import (
    export_metadata,
    kernel_metadata,
    to_notebook,
)

TEMPLATE = Path("notebooks/checkworthy_notebook.py")


def lifted(name: str):
    """Execute one top-level function out of the notebook template, with numpy in scope."""
    tree = ast.parse(TEMPLATE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            namespace: dict = {"np": np}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<template>", "exec"), namespace)
            return namespace[name]
    raise AssertionError(f"{name} is not a top-level function in {TEMPLATE}")


# --- the metric that crosses the boundary -------------------------------------------------------

def test_the_two_binarizations_count_the_classes_they_claim_to():
    """
    Classes are 0 non-factual, 1 unimportant-factual, 2 check-worthy. "factual" is 1 and 2
    together; "check-worthy" is 2 alone. Counting the wrong set changes the headline number
    without changing anything that would raise.
    """
    scores = lifted("scores")
    actual = np.array([0, 0, 1, 1, 2, 2, 2])
    perfect = scores(actual.copy(), actual)
    assert perfect["accuracy"] == pytest.approx(1.0)
    assert perfect["factual"]["n_positive"] == 5, "1s and 2s"
    assert perfect["check_worthy"]["n_positive"] == 3, "2s only"


def test_precision_and_recall_are_not_interchangeable():
    """
    Predict check-worthy everywhere: recall is 1.0 and precision is the base rate. Equal values
    would let a swap pass, so the case is chosen to separate them.
    """
    scores = lifted("scores")
    actual = np.array([0, 0, 0, 0, 0, 0, 2, 2])
    got = scores(np.full_like(actual, 2), actual)
    assert got["check_worthy"]["recall"] == pytest.approx(1.0)
    assert got["check_worthy"]["precision"] == pytest.approx(0.25)
    assert got["check_worthy"]["f1"] == pytest.approx(0.4)


def test_a_model_that_predicts_nothing_positive_scores_zero_not_one():
    """The 0/0 guard must not resolve to a perfect score, which is the flattering direction."""
    scores = lifted("scores")
    actual = np.array([0, 0, 2, 2])
    got = scores(np.zeros_like(actual), actual)
    assert got["check_worthy"]["precision"] == 0.0
    assert got["check_worthy"]["recall"] == 0.0
    assert got["check_worthy"]["f1"] == pytest.approx(0.0)


def test_unimportant_factual_counts_as_factual_but_not_as_check_worthy():
    """
    The class the two label definitions disagree about. Predicting UFS everywhere should look
    perfect on recall for "factual" and score nothing at all on "check-worthy".
    """
    scores = lifted("scores")
    actual = np.array([1, 1, 1, 1])
    got = scores(np.ones_like(actual), actual)
    assert got["factual"]["recall"] == pytest.approx(1.0)
    assert got["check_worthy"]["n_positive"] == 0


# --- the notebook and its kernel ------------------------------------------------------------------

def test_the_template_survives_conversion_as_the_same_program():
    built = to_notebook(TEMPLATE.read_text(encoding="utf-8"))
    assert built["nbformat"] == 4
    kinds = {c["cell_type"] for c in built["cells"]}
    assert kinds == {"markdown", "code"}
    code = "".join("".join(c["source"]) for c in built["cells"] if c["cell_type"] == "code")
    assert "def scores(" in code
    assert "GradScaler" in code


def test_markdown_cells_lose_their_comment_prefix():
    built = to_notebook(TEMPLATE.read_text(encoding="utf-8"))
    markdown = "".join("".join(c["source"]) for c in built["cells"] if c["cell_type"] == "markdown")
    assert markdown.lstrip().startswith("#"), "the heading survives"
    assert "# # Check-worthiness" not in markdown, "but the template's comment prefix does not"


def test_the_kernel_is_not_named_after_fever():
    """
    make_notebooks.py prefixes every id with fever-verdict-, which already mislabels the AVeriTeC
    runs. This is a different task on a different corpus and must not inherit that name.
    """
    metadata = kernel_metadata("someone")
    assert "fever" not in metadata["id"]
    assert metadata["id"] == "someone/checkworthy-detector"


def test_the_kernel_attaches_the_dataset_it_needs_and_asks_for_a_gpu():
    metadata = kernel_metadata("someone")
    assert metadata["dataset_sources"] == ["someone/checkworthy-v1"]
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True


def test_the_committed_kernel_definition_matches_the_generator():
    """
    The definition is version-controlled so a run can be reproduced from the repo. If the two
    drift, the committed file is a description of a kernel that no longer exists.
    """
    path = Path("notebooks/kernels/checkworthy.json")
    if not path.exists():
        pytest.skip("kernel definition not generated yet")
    assert json.loads(path.read_text(encoding="utf-8")) == kernel_metadata("vincentvdo6")


# --- the ONNX export kernel ---------------------------------------------------------------------

def test_the_export_mounts_the_trained_kernel_rather_than_a_dataset():
    """
    The weights are another kernel's output, so mounting them directly avoids a 738 MB round trip
    through a dataset upload. Attaching a dataset instead would find no model_v1 to export.
    """
    export = export_metadata("someone")
    assert export["kernel_sources"] == ["someone/checkworthy-detector"]
    assert export["dataset_sources"] == []


def test_the_export_does_not_ask_for_a_gpu():
    """Tracing a graph is CPU work, and Kaggle allows only two concurrent GPU sessions."""
    assert export_metadata("someone")["enable_gpu"] is False


def test_the_export_is_not_named_after_the_trainer_or_after_fever():
    export = export_metadata("someone")
    assert export["id"] == "someone/checkworthy-onnx"
    assert "fever" not in export["id"]
    assert export["id"] != kernel_metadata("someone")["id"], "two kernels, two ids"


def test_the_committed_export_definition_matches_the_generator():
    path = Path("notebooks/kernels/checkworthy-onnx.json")
    if not path.exists():
        pytest.skip("export kernel definition not generated yet")
    assert json.loads(path.read_text(encoding="utf-8")) == export_metadata("vincentvdo6")


def test_the_export_notebook_is_the_same_program_as_the_committed_template():
    """
    export_onnx.py is task-agnostic -- it globs for model_v1 and reads max_length from the contract
    beside it -- so the same template serves the verdict model and this one. That only holds if the
    conversion does not quietly alter it.
    """
    built = to_notebook(Path("notebooks/export_onnx.py").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in built["cells"] if c["cell_type"] == "code")
    assert "torch.onnx.export" in code
    assert "dynamo=False" in code, "the TorchScript exporter, which Kaggle's image can run"
    assert "**/model_v1" in code, "the model is discovered, not hard-coded to one task"
