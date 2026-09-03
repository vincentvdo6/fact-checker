"""
Generate the check-worthiness kernel: notebook plus its Kaggle definition.

Separate from `make_notebooks.py` on purpose. That generator builds the three verdict variants
from one template and hard-codes a `fever-verdict-` prefix into every kernel id -- already wrong
for the AVeriTeC runs it produced, and wronger still here, since check-worthiness is a different
task on a different dataset with a different label space. Reusing it would mean a fifth kernel
carrying a name that describes neither its data nor its job. Unifying the two generators is worth
doing; quietly widening the broken one is not.

The cell-splitting is shared rather than reimplemented, because that logic is what keeps the
committed `.py` and the pushed `.ipynb` the same program.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.make_notebooks import cells, undecorate

TEMPLATE = Path("notebooks/checkworthy_notebook.py")
EXPORT_TEMPLATE = Path("notebooks/export_onnx.py")
KERNELS = Path("notebooks/kernels")
DATASET = "checkworthy-v1"
SLUG = "checkworthy-detector"
EXPORT_SLUG = "checkworthy-onnx"


def to_notebook(source: str) -> dict:
    body = []
    for kind, text in cells(source):
        if kind == "markdown":
            body.append({"cell_type": "markdown", "metadata": {},
                         "source": undecorate(text).splitlines(keepends=True)})
        else:
            body.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                         "outputs": [], "source": text.splitlines(keepends=True)})
    return {
        "cells": body,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def kernel_metadata(owner: str) -> dict:
    """
    `enable_gpu` does not choose the accelerator -- that happens at push time with
    `--accelerator NvidiaTeslaT4`, capital N. The CLI validates nothing there and silently falls
    back to a P100, which cannot run Kaggle's installed torch at all.
    """
    return {
        "id": f"{owner}/{SLUG}",
        "title": "checkworthy detector",
        "code_file": "kaggle_checkworthy.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{owner}/{DATASET}"],
        "competition_sources": [],
        "kernel_sources": [],
    }


def export_metadata(owner: str) -> dict:
    """
    The ONNX export, which mounts the trained kernel's output rather than a dataset.

    `export_onnx.py` globs for `**/model_v1` and reads max_length from the contract beside it, so
    it is already task-agnostic -- only what gets attached differs. Local inference has no torch,
    which is why every model this project ships crosses through ONNX.
    """
    return {
        "id": f"{owner}/{EXPORT_SLUG}",
        "title": "checkworthy onnx",
        "code_file": "kaggle_export_onnx.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,          # a CPU export; the graph is traced, not trained
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [f"{owner}/{SLUG}"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="vincentvdo6")
    parser.add_argument("--out", default="notebooks/build")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    KERNELS.mkdir(parents=True, exist_ok=True)

    source = TEMPLATE.read_text(encoding="utf-8")
    built = to_notebook(source)
    (out / "kaggle_checkworthy.ipynb").write_text(json.dumps(built, indent=1), encoding="utf-8")

    metadata = kernel_metadata(args.owner)
    (KERNELS / "checkworthy.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    exported = to_notebook(EXPORT_TEMPLATE.read_text(encoding="utf-8"))
    # Nested inside the build directory rather than beside it: .gitignore already excludes
    # build/, and a sibling would need its own rule to stay out of the tree.
    export_dir = out / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    (export_dir / "kaggle_export_onnx.ipynb").write_text(json.dumps(exported, indent=1),
                                                        encoding="utf-8")
    export = export_metadata(args.owner)
    (KERNELS / "checkworthy-onnx.json").write_text(json.dumps(export, indent=2), encoding="utf-8")
    (export_dir / "kernel-metadata.json").write_text(json.dumps(export, indent=2), encoding="utf-8")

    code = sum(1 for c in built["cells"] if c["cell_type"] == "code")
    print(f"{len(built['cells'])} cells ({code} code) -> {out / 'kaggle_checkworthy.ipynb'}")
    print(f"kernel {metadata['id']}  attaching {metadata['dataset_sources']}")
    print(f"export kernel {export['id']}  mounting {export['kernel_sources']}")
    print(f"\npush with:\n  kaggle kernels push -p {out} --accelerator NvidiaTeslaT4")
    print(f"  kaggle kernels push -p {export_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
