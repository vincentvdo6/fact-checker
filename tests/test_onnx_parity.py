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

from pathlib import Path

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
