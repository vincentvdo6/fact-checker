"""News pairs enter training shaped as the judge is asked them, and never from the measurement set."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from scripts.add_news_pairs import fresh_pairs, load_pairs, main, news_pair, write_pairs


def item(**overrides):
    return {"id": "c1/assertion-1/s1:p1:u1", "case": "c1", "assertion": "We have a labor shortage.", "sentence": "Employers cannot hire.",
            "definitions": ["A shortage means unfilled jobs."], "url": "https://a.example/x", "relation": "states",
            "qualifiers": []} | overrides


def test_a_negated_assertion_is_asked_in_its_positive_form_with_the_label_inverted():
    plain = news_pair(item(), "labels/x.json")
    assert plain.premise == "Employers cannot hire. A shortage means unfilled jobs."
    assert plain.hypothesis == "We have a labor shortage." and plain.relation == "states" and plain.source == "news"
    assert plain.group == "news:c1" and plain.id == "news:c1/assertion-1/s1:p1:u1"
    denial = news_pair(item(assertion="We don't have a labor shortage.", relation="states_negation"), "labels/x.json")
    assert denial.hypothesis == "We have a labor shortage." and denial.relation == "states", "inverted with the form"
    assert denial.provenance["labelled_relation"] == "states_negation"
    bearing = news_pair(item(assertion="We don't have a labor shortage.", relation="bears_on"), "labels/x.json")
    assert bearing.relation == "bears_on", "only counted labels invert"
    with pytest.raises(ValueError):
        news_pair(item(assertion="We don't have a labor shortage. We have a good job shortage."), "labels/x.json")


def test_duplicates_count_once_and_conflicting_labels_are_rejected():
    pair = news_pair(item(), "labels/x.json")
    duplicate = replace(pair, id="news:duplicate")
    assert fresh_pairs([pair, duplicate], []) == ([pair], 1, 0)
    assert fresh_pairs([pair, duplicate], [pair]) == ([], 1, 1)
    conflict = replace(duplicate, relation="unrelated")
    with pytest.raises(ValueError, match="conflicting relation"):
        fresh_pairs([pair, conflict], [])
    with pytest.raises(ValueError, match="conflicting relation"):
        fresh_pairs([conflict], [pair])


@pytest.fixture
def build(tmp_path, monkeypatch):
    base, dest = tmp_path / "base", tmp_path / "news"
    base.mkdir()
    for split in ("train", "trainval", "calibration", "test"):
        pair = replace(news_pair(item(), "base"), id=split, group=split, source="fever", premise=f"Base {split} text.")
        write_pairs([pair], base / f"pairs_{split}.jsonl.gz")
    (base / "dataset_manifest.json").write_text("{}", encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labeller": "fixture", "items": [item(), item(id="duplicate")]}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["add_news_pairs", "--base", str(base), "--dest", str(dest),
                                    "--labels", str(labels), "--repeat", "3"])
    return base, dest


def test_only_train_changes_and_each_unique_news_pair_gets_the_requested_weight(build):
    base, dest = build
    assert main() == 0
    for split in ("trainval", "calibration", "test"):
        name = f"pairs_{split}.jsonl.gz"
        assert (dest / name).read_bytes() == (base / name).read_bytes()
    train = load_pairs(dest / "pairs_train.jsonl.gz")
    assert len(train) == 4 and len({pair.id for pair in train}) == 4
    assert sum(pair.source == "news" for pair in train) == 3
    news = json.loads((dest / "dataset_manifest.json").read_text(encoding="utf-8"))["news"]
    assert (news["unique_pairs"], news["duplicate_rows"], news["already_in_base"], news["repeat"]) == (1, 1, 0, 3)


def test_excluded_pair_blocks_the_build_without_using_its_label(build, monkeypatch):
    import sys

    base, dest = build
    excluded = base / "excluded.json"
    excluded.write_text(json.dumps({"items": [item(assertion="We don't have a labor shortage.",
                                                  relation=["not a training label"])]}), encoding="utf-8")
    monkeypatch.setattr("sys.argv", sys.argv + ["--exclude", str(excluded)])
    with pytest.raises(SystemExit, match="excluded label file"):
        main()
    assert not dest.exists()


@pytest.mark.parametrize("repeat", ["0", "-1"])
def test_nonpositive_weight_writes_nothing(build, monkeypatch, repeat):
    import sys

    _, dest = build
    monkeypatch.setattr("sys.argv", sys.argv[:-1] + [repeat])
    with pytest.raises(SystemExit):
        main()
    assert not dest.exists()


def test_existing_destination_is_not_overwritten(build):
    _, dest = build
    dest.mkdir()
    sentinel = dest / "keep.txt"
    sentinel.write_text("preserved", encoding="utf-8")
    with pytest.raises(SystemExit):
        main()
    assert sentinel.read_text(encoding="utf-8") == "preserved"
    assert list(dest.iterdir()) == [sentinel]
