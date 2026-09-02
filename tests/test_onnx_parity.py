"""
Checkpoint 1's own tests: the parity check has to be able to fail.

`scripts/check_onnx_parity.py` is the only thing standing between "the export loaded" and "the
export is the model every calibration artifact was fitted on". A gate that cannot fail is worse
than no gate, because it is mistaken for evidence -- so these guard the three ways this one could
quietly stop checking: comparing something weaker than band assignment, a tolerance loose enough
to admit any export, and an evidence stream that drifts out of step with the notebook's under
--limit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_onnx_parity import read_cache

PARITY = Path("scripts/check_onnx_parity.py")


def test_parity_compares_band_assignment_not_only_logits():
    """
    Bands are thresholds on calibrated confidence. Comparing raw logits alone would pass an export
    that drifts a claim across a band edge -- which is exactly the failure that turns "strong:
    right nine times in ten" into a false promise.
    """
    source = PARITY.read_text(encoding="utf-8")
    assert "policy.assign" in source
    assert "band agreement" in source


def test_parity_tolerance_is_the_recorded_one():
    from scripts.check_onnx_parity import LOGIT_TOLERANCE

    assert LOGIT_TOLERANCE == 1e-3


def test_parity_advances_the_rng_over_every_row_even_when_limited():
    """
    The notebook advanced one Random across the whole split in file order. Skipping rows under
    --limit would desynchronise the evidence shuffle and compare against logits computed from
    different text, making any disagreement uninterpretable.
    """
    source = PARITY.read_text(encoding="utf-8")
    body = source[source.index("for row in rows:"):source.index("print(f\"scoring")]
    assert "select_evidence(" in body
    assert body.index("select_evidence(") < body.index("args.limit"), (
        "the limit must be applied after the evidence is selected, not before"
    )


def test_parity_fails_loudly_rather_than_returning_zero():
    source = PARITY.read_text(encoding="utf-8")
    assert "return 1" in source
    assert "FAIL" in source


# --- the cache that lets a crashed run resume ---------------------------------------------------

def test_an_absent_cache_is_an_empty_start_not_an_error(tmp_path):
    assert read_cache(tmp_path / "nothing.jsonl") == {}


def test_the_cache_round_trips_what_the_run_wrote(tmp_path):
    path = tmp_path / "logits.jsonl"
    lines = [json.dumps({"id": i, "logits": [1.0, 2.0, 3.0]}) for i in ("a", "b")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert read_cache(path) == {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}


def test_a_line_truncated_by_the_crash_is_dropped_rather_than_parsed(tmp_path):
    """
    The failure this cache exists to survive is a hard reset, which can land mid-write. Half a
    JSON object must cost one re-scored claim, never a malformed row that flows into the
    comparison as if it were real logits.
    """
    path = tmp_path / "logits.jsonl"
    whole = json.dumps({"id": "a", "logits": [1.0, 2.0, 3.0]})
    path.write_text(whole + "\n" + '{"id": "b", "logi', encoding="utf-8")
    assert read_cache(path) == {"a": [1.0, 2.0, 3.0]}


def test_the_run_fsyncs_each_batch_rather_than_trusting_the_page_cache():
    """
    flush() alone moves bytes into the OS page cache, which a power cut discards. Without fsync
    the cache would look written and come back empty, which is the failure it exists to prevent.
    """
    source = PARITY.read_text(encoding="utf-8")
    assert "os.fsync(cache.fileno())" in source
    assert source.index("cache.flush()") < source.index("os.fsync(cache.fileno())")


def test_a_fully_cached_batch_is_skipped_rather_than_rescored():
    source = PARITY.read_text(encoding="utf-8")
    assert "if all(i in done for i in chunk):" in source
    assert "continue" in source


def test_the_comparison_reads_the_cache_not_a_separate_in_memory_list():
    """
    Comparing a list built during this run would silently exclude everything a previous run
    scored -- the resume would appear to work while checking only the new claims.
    """
    source = PARITY.read_text(encoding="utf-8")
    assert "local = np.asarray([done[i] for i in ids], dtype=np.float64)" in source


@pytest.mark.parametrize("split", ["test", "calibration"])
def test_each_split_caches_under_its_own_name(split):
    from scripts.check_onnx_parity import CACHE

    assert CACHE.format(split=split) == f"onnx_logits_{split}.jsonl"
