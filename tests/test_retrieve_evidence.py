"""Resume, torn-line repair, and shard partitioning for the long retrieval run."""

from __future__ import annotations

import json

import pytest

from scripts.retrieve_evidence import completed, merge, repair


def write(path, rows, trailing_newline=True):
    text = "\n".join(json.dumps(r) for r in rows)
    path.write_text(text + ("\n" if trailing_newline else ""), encoding="utf-8")


def test_repair_leaves_a_complete_file_alone(tmp_path):
    path = tmp_path / "r.jsonl"
    write(path, [{"id": 1, "evidence": []}, {"id": 2, "evidence": []}])
    before = path.read_bytes()
    assert repair(path) == 0
    assert path.read_bytes() == before


def test_repair_drops_a_torn_final_row(tmp_path):
    """
    A truncated row is the dangerous case: it can still parse as valid JSON with a short evidence
    list, and a claim that quietly kept three sentences instead of twenty-five reads as a
    retrieval failure rather than a torn file.
    """
    path = tmp_path / "r.jsonl"
    write(path, [{"id": 1, "evidence": [["A", 0]]}])
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"id": 2, "evidence": [["B"')

    assert repair(path) > 0
    assert completed(path) == {1}
    # The survivor must be intact, not merely parseable.
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": 1, "evidence": [["A", 0]]}]


def test_repair_handles_a_file_with_no_newline_at_all(tmp_path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"id": 1, "evi', encoding="utf-8")
    assert repair(path) > 0
    assert path.read_text(encoding="utf-8") == ""
    assert completed(path) == set()


def test_repair_on_missing_or_empty_file(tmp_path):
    assert repair(tmp_path / "absent.jsonl") == 0
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert repair(empty) == 0


def test_completed_reads_every_id(tmp_path):
    path = tmp_path / "r.jsonl"
    write(path, [{"id": 7, "evidence": []}, {"id": 9, "evidence": []}])
    assert completed(path) == {7, 9}


@pytest.mark.parametrize("shards", [1, 2, 3, 4])
def test_shards_partition_the_claims(shards):
    """Disjoint and exhaustive, decided by id alone so no process needs to know about the others."""
    ids = list(range(1000))
    buckets = [[i for i in ids if i % shards == s] for s in range(shards)]
    assert sum(len(b) for b in buckets) == len(ids)
    assert set().union(*(set(b) for b in buckets)) == set(ids)
    for a in range(shards):
        for b in range(a + 1, shards):
            assert not set(buckets[a]) & set(buckets[b])


class FakeClaim:
    def __init__(self, id_):
        self.id = id_


def test_merge_orders_by_split_not_by_shard(tmp_path):
    write(tmp_path / "retrieved.part0.jsonl", [{"id": 0, "evidence": [["A", 0]]}, {"id": 2, "evidence": []}])
    write(tmp_path / "retrieved.part1.jsonl", [{"id": 1, "evidence": [["B", 1]]}])

    assert merge(tmp_path, [FakeClaim(i) for i in (0, 1, 2)], 2) == 0
    rows = [json.loads(line) for line in (tmp_path / "retrieved.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [0, 1, 2]
    assert rows[1]["evidence"] == [["B", 1]]


def test_merge_refuses_a_missing_shard(tmp_path):
    write(tmp_path / "retrieved.part0.jsonl", [{"id": 0, "evidence": []}])
    with pytest.raises(SystemExit, match="part1.jsonl is missing"):
        merge(tmp_path, [FakeClaim(0), FakeClaim(1)], 2)


def test_merge_refuses_a_duplicated_claim(tmp_path):
    write(tmp_path / "retrieved.part0.jsonl", [{"id": 5, "evidence": []}])
    write(tmp_path / "retrieved.part1.jsonl", [{"id": 5, "evidence": []}])
    with pytest.raises(SystemExit, match="more than one shard"):
        merge(tmp_path, [FakeClaim(5)], 2)


def test_merge_refuses_an_incomplete_run(tmp_path):
    """Better to fail than to hand a short file to the dataset builder."""
    write(tmp_path / "retrieved.part0.jsonl", [{"id": 0, "evidence": []}])
    with pytest.raises(SystemExit, match="have no evidence"):
        merge(tmp_path, [FakeClaim(0), FakeClaim(2)], 1)
