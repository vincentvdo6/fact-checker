"""
Kernel names, which are the only thing telling ten near-identical runs apart.

Every id used to be built as `fever-verdict-<variant>` regardless of what the kernel trained on,
which produced `fever-verdict-claim-only-averitec`: a kernel named after FEVER, attached to
AVeriTeC, that never touches FEVER at all. On a Kaggle listing the name is the whole interface --
there is no column for the dataset -- so a name that lies is worse than one that is long.

The fix is to derive the prefix from the attached dataset, which makes the name correct by
construction rather than by discipline: a kernel cannot be renamed without moving its data.

Reconciling against Kaggle: `claim_only_averitec` and `gold_averitec` ran under the old
FEVER-prefixed ids and produced the Phase 06 results. Re-pushing makes correctly-named kernels;
the numbers are unaffected, since neither the notebook nor the dataset changed.
"""

from __future__ import annotations

# --- kernel names describe their data ------------------------------------------------------------

def test_a_kernel_is_named_after_the_corpus_it_trains_on():
    """
    Every id used to begin fever-verdict-, which produced fever-verdict-claim-only-averitec: a
    kernel named after FEVER that trains on AVeriTeC and never touches FEVER. The name is the only
    thing separating ten near-identical runs on the Kaggle listing, so one that lies is worse than
    one that is long.
    """
    from scripts.make_notebooks import DATASETS, kernel_metadata

    for name, dataset in DATASETS.items():
        kernel = kernel_metadata(name, "owner")["id"].split("/", 1)[1]
        corpus = dataset.split("-")[0]
        assert kernel.startswith(corpus), f"{name}: {kernel} does not name {corpus}"


def test_the_averitec_kernels_are_no_longer_named_after_fever():
    from scripts.make_notebooks import kernel_metadata

    for name in ("claim_only_averitec", "gold_averitec"):
        assert "fever" not in kernel_metadata(name, "owner")["id"]


def test_the_corpus_is_not_repeated_in_the_variant_suffix():
    """averitec-verdict-claim-only-averitec names the same corpus twice and reads like a typo."""
    from scripts.make_notebooks import DATASETS, kernel_metadata

    for name in DATASETS:
        kernel = kernel_metadata(name, "owner")["id"].split("/", 1)[1]
        for token in kernel.split("-"):
            assert kernel.count(f"-{token}") <= 1 or token in ("v1", "v2", "v3"), (
                f"{kernel} repeats {token!r}"
            )


def test_a_variant_qualifier_that_is_not_a_corpus_survives():
    """`grounded` and `dropped` distinguish training arms, not datasets, and must not be stripped."""
    from scripts.make_notebooks import kernel_metadata

    assert kernel_metadata("retrieved_grounded", "o")["id"].endswith("retrieved-grounded")
    assert kernel_metadata("retrieved_dropped", "o")["id"].endswith("retrieved-dropped")


def test_every_id_is_unique():
    """Two kernels sharing an id means the second push silently overwrites the first."""
    from scripts.make_notebooks import DATASETS, kernel_metadata

    ids = [kernel_metadata(name, "owner")["id"] for name in DATASETS]
    assert len(set(ids)) == len(ids)


def test_the_committed_definitions_match_the_generator():
    import json
    from pathlib import Path

    from scripts.make_notebooks import DATASETS, kernel_metadata

    for name in DATASETS:
        path = Path("notebooks/kernels") / f"{name}.json"
        if not path.exists():
            continue
        assert json.loads(path.read_text(encoding="utf-8")) == kernel_metadata(name, "vincentvdo6")
