"""
The CI definition, checked against the repository it claims to check.

A workflow drifts from its project quietly: someone changes the Python version, or the marker, or
the lint command, and CI keeps reporting green on something other than what the repo asks for. The
failure mode is a badge that means nothing, which is worse than no badge -- and this repository's
argument is entirely about numbers meaning what they claim.

The specific thing guarded is the skip trap. Every artifact CI cannot have is gitignored, so a
suite run without them still passes; if the marker or the guards changed such that everything
skipped, the job would go green having checked nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WORKFLOW = Path(".github/workflows/checks.yml")


def workflow() -> str:
    if not WORKFLOW.exists():
        pytest.skip("no CI workflow")
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_runs_the_marker_the_project_documents():
    """
    CLAUDE.md and the layout both treat `-m "not slow"` as the suite a machine without artifacts
    runs. CI running the full suite would be permanently red on the 738 MB graphs it cannot have.
    """
    assert 'pytest -m "not slow"' in workflow()


def test_ci_pins_the_python_the_project_targets():
    """3.12 is what the code is written against; 3.11 would fail on the match statements alone."""
    assert 'python-version: "3.12"' in workflow()


def test_ci_lints_as_well_as_tests():
    assert "ruff check ." in workflow()


def test_ci_varies_the_hash_seed():
    """
    Dict and set ordering has decided a result in this repo before. One seed can hide that; two
    make it show up on someone else's machine rather than in a reported number.
    """
    assert "PYTHONHASHSEED" in workflow()


def test_ci_reports_what_was_skipped():
    """
    Every artifact CI lacks is gitignored, so a suite that skipped everything would still be
    green. Printing the skip reasons is what makes "passed" legible rather than assumed.
    """
    text = workflow()
    assert "-rs" in text
    assert re.search(r"skipped", text, re.I)


def test_ci_does_not_pretend_to_run_what_it_cannot():
    """
    A job that fetched the corpus would take hours and be red whenever the mirror moved. The
    workflow says what it is not doing, so nobody reads the badge as full coverage.
    """
    text = workflow().lower()
    assert "fetch_wiki" not in text and "build_index" not in text
    assert "gitignored" in text


def test_the_requirements_file_is_what_ci_installs():
    assert "pip install -r requirements.txt" in workflow()
    assert Path("requirements.txt").exists()
