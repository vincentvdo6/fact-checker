"""A rejected or interrupted install cannot expose a partial model to the host."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts import install_context_ranker as installer
from src.verdict import context_ranker


@pytest.mark.parametrize("failure", [None, "source", "copy", "corrupt_copy", "existing"])
def test_installer_checks_all_files_and_publishes_only_complete_bundle(tmp_path, monkeypatch, failure):
    bundle, destination = tmp_path / "bundle", tmp_path / "installed/ranker"
    bundle.mkdir()
    for name in ("contract.json", "tokenizer.json", "ranker.onnx", "weights"):
        (bundle / name).write_bytes(name.encode())
    def digest(name):
        return hashlib.sha256((bundle / name).read_bytes()).hexdigest()

    (bundle / "source-copy.json").write_text(json.dumps({
        "files_sha256": {name: digest(name) for name in ("ranker.onnx", "weights")}}))
    for name, constant in (("contract.json", "CONTRACT_SHA256"), ("tokenizer.json", "TOKENIZER_SHA256"),
                           ("source-copy.json", "MANIFEST_SHA256")):
        monkeypatch.setattr(installer, constant, digest(name))
    monkeypatch.setattr(context_ranker, "MANIFEST_SHA256", digest("source-copy.json"))
    if failure == "source":
        (bundle / "weights").write_bytes(b"corrupt")
    elif failure in ("copy", "corrupt_copy"):
        copy = installer.shutil.copy2

        def interrupt(source, target):
            if source.name == "weights" and failure == "copy":
                raise OSError("interrupted copy")
            result = copy(source, target)
            if source.name == "weights" and failure == "corrupt_copy":
                target.write_bytes(b"corrupted during copy")
            return result

        monkeypatch.setattr(installer.shutil, "copy2", interrupt)
    elif failure == "existing":
        destination.mkdir(parents=True)
        (destination / "keep").write_bytes(b"old model")
    if failure:
        with pytest.raises((ValueError, OSError)):
            installer.install(bundle, destination)
        assert destination.exists() == (failure == "existing")
        assert not list(destination.parent.glob("context-ranker-*"))
        if failure == "existing":
            assert (destination / "keep").read_bytes() == b"old model"
    else:
        (bundle / "unrelated").write_bytes(b"not part of the model")
        installer.install(bundle, destination)
        assert {p.name for p in destination.iterdir()} == {
            "contract.json", "tokenizer.json", "source-copy.json", "ranker.onnx", "weights"}
        for path in destination.iterdir():
            assert path.read_bytes() == (bundle / path.name).read_bytes()
