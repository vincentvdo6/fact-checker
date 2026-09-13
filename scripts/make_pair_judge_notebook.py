"""
Generate the pair-judge kernel: notebook plus its Kaggle definition, and the ONNX export kernel.

The check-worthiness generator is copied rather than widened for the reason it gave itself: each
task has its own dataset, label space and kernel name, and one generator carrying all of them
would describe none. The cell splitting is shared, which is what keeps the committed `.py` and
the pushed `.ipynb` the same program.

The export kernel is the shared `export_onnx.py`: it globs `**/model_v1` and reads max_length
from the contract beside it, so only what it mounts differs. Build to a separate directory so the
check-worthiness build is not overwritten.

    python -m scripts.make_pair_judge_notebook --version 2
    kaggle datasets create -p data/kaggle/pair-judge-v2        # then wait for `kaggle datasets status` = ready
    kaggle kernels push -p notebooks/build/pair-judge-v2 --accelerator NvidiaTeslaT4
    kaggle kernels push -p notebooks/build/pair-judge-v2/export
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from scripts.make_checkworthy_notebook import to_notebook

TEMPLATE = Path("notebooks/pair_judge_notebook.py")
EXPORT_TEMPLATE = Path("notebooks/export_onnx.py")
KERNELS = Path("notebooks/kernels")


def names(version: int, dataset_version: int | None = None) -> dict[str, str]:
    """v1 keeps its original kernel names. A kernel slug may not equal a dataset slug of the same
    owner -- Kaggle answers 409 -- so later kernels are `pair-judge-train-vN` against `pair-judge-vN`.
    A run may attach an earlier dataset (same pairs, a different encoder) via `dataset_version`."""
    slug = "pair-judge" if version == 1 else f"pair-judge-train-v{version}"
    return {"dataset": f"pair-judge-v{dataset_version or version}", "slug": slug, "export": f"{slug}-onnx",
            "title": "pair judge" if version == 1 else f"pair judge train v{version}"}


def configure(template: str, overrides: dict[str, object]) -> str:
    """Rewrite CFG entries in the notebook source; every key must exist there, so a typo cannot silently train the default."""
    for key, value in overrides.items():
        pattern = rf"^(\s+{re.escape(key)}=)([^,]+)(,.*)$"
        template, count = re.subn(pattern, lambda m: f"{m[1]}{value!r}{m[3]}", template, count=1, flags=re.M)
        if count != 1:
            raise ValueError(f"CFG has no entry {key!r}")
    return template


def kernel_metadata(owner: str, version: int, dataset_version: int | None = None) -> dict:
    """`enable_gpu` does not choose the accelerator; `--accelerator NvidiaTeslaT4` at push time does."""
    name = names(version, dataset_version)
    return {"id": f"{owner}/{name['slug']}", "title": name["title"], "code_file": "kaggle_pair_judge.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [f"{owner}/{name['dataset']}"], "competition_sources": [],
            "kernel_sources": []}


def export_metadata(owner: str, version: int) -> dict:
    name = names(version)
    return {"id": f"{owner}/{name['export']}", "title": f"{name['title']} onnx", "code_file": "kaggle_export_onnx.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [f"{owner}/{name['slug']}"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", default="vincentvdo6")
    parser.add_argument("--version", type=int, default=1, help="dataset and kernel version suffix")
    parser.add_argument("--dataset-version", type=int, default=None, help="attach an earlier dataset version")
    parser.add_argument("--base-model", default=None, help="encoder to fine-tune, e.g. microsoft/deberta-v3-base")
    parser.add_argument("--base-revision", default=None, help="pinned revision of --base-model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    name = names(args.version, args.dataset_version)
    out = Path(args.out or f"notebooks/build/{name['slug']}")
    out.mkdir(parents=True, exist_ok=True)
    KERNELS.mkdir(parents=True, exist_ok=True)

    overrides = {key: value for key, value in (("base_model", args.base_model), ("base_revision", args.base_revision),
                                               ("epochs", args.epochs), ("lr", args.lr), ("batch", args.batch),
                                               ("grad_accum", args.grad_accum)) if value is not None}
    if (args.base_model is None) != (args.base_revision is None):
        raise SystemExit("--base-model and --base-revision go together: a new encoder needs its pinned revision")
    source = configure(TEMPLATE.read_text(encoding="utf-8"), overrides)
    built = to_notebook(source)
    (out / "kaggle_pair_judge.ipynb").write_text(json.dumps(built, indent=1), encoding="utf-8")
    (out / "overrides.json").write_text(json.dumps(overrides, indent=2), encoding="utf-8")
    metadata = kernel_metadata(args.owner, args.version, args.dataset_version)
    (KERNELS / f"{name['slug']}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    export_dir = out / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = to_notebook(EXPORT_TEMPLATE.read_text(encoding="utf-8"))
    (export_dir / "kaggle_export_onnx.ipynb").write_text(json.dumps(exported, indent=1), encoding="utf-8")
    export = export_metadata(args.owner, args.version)
    (KERNELS / f"{name['export']}.json").write_text(json.dumps(export, indent=2), encoding="utf-8")
    (export_dir / "kernel-metadata.json").write_text(json.dumps(export, indent=2), encoding="utf-8")

    code = sum(1 for cell in built["cells"] if cell["cell_type"] == "code")
    print(f"{len(built['cells'])} cells ({code} code) -> {out / 'kaggle_pair_judge.ipynb'}")
    print(f"kernel {metadata['id']}  attaching {metadata['dataset_sources']}")
    print(f"export kernel {export['id']}  mounting {export['kernel_sources']}")
    print(f"\npush with:\n  kaggle kernels push -p {out} --accelerator NvidiaTeslaT4\n  kaggle kernels push -p {export_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
