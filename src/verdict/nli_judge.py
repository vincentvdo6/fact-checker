"""Judge each (assertion, sentence) pair with the calibrated FEVER cross-encoder.

This is the one component in the repository with a measured calibration, so it is the
first judge tried behind the pair contract. It is out of distribution here twice over:
the sentences are news prose rather than Wikipedia, and it was trained on packed
evidence rather than a single sentence with its definitions. Its bands were promised
on FEVER's test split and promise nothing about these pairs. A pair earns a relation
only when the calibrated confidence reaches a band; everything below abstains to
`unrelated`, with the raw label kept so the abstention can be inspected.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from src.calibration.bands import BAND_ORDER, Band, BandPolicy
from src.calibration.scaling import from_dict as calibrator_from_dict
from src.verdict.assertions import positive_form
from src.verdict.encode import LABELS
from src.verdict.pair_judgment import nli_judgment
from src.verdict.runtime import MODELS, VerdictRuntime

INVERTED = {"supported": "contradicted", "contradicted": "supported"}


def _title(source: dict) -> str:
    """The page's own heading stands in for a Wikipedia title; the host when there is none."""
    for line in source.get("reading_context", []):
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    host = urlsplit(source.get("url", "")).hostname or "source"
    return host[4:] if host.startswith("www.") else host


class NliJudge:
    def __init__(self, sources: list[dict], *, variant: str = "retrieved", models: str | Path = MODELS,
                 min_band: Band | None = Band.WEAK) -> None:
        root = Path(models) / variant
        payload = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        self.runtime = VerdictRuntime(variant, models=models)
        self.calibrator = calibrator_from_dict(payload["calibrator"])
        self.bands = BandPolicy.from_dict(payload["bands"])
        self.min_band = min_band
        self.titles = {source["id"]: _title(source) for source in sources}

    def _accepts(self, band: Band | None) -> bool:
        if band is None or self.min_band is None:
            return band is not None
        return BAND_ORDER.index(band) <= BAND_ORDER.index(self.min_band)

    def __call__(self, assertions: list[dict], units: list[dict]) -> list[dict]:
        pairs, keys = [], []
        for assertion in assertions:
            # FEVER taught the encoder that a negation word in the claim predicts REFUTES; judging the
            # positive clause and inverting the relation in code keeps that tell out of the verdict.
            probe = positive_form(assertion["text"]) if assertion["negated"] else None
            for unit in units:
                title = self.titles.get(unit["source_id"], "source")
                evidence = [(title, 0, unit["text"])] + [(title, index + 1, item["text"])
                                                         for index, item in enumerate(unit["definitions"])]
                pairs.append((probe or assertion["text"], evidence))
                keys.append((assertion, unit, probe))
        if not pairs:
            return []
        scored = self.runtime.score_batch(pairs)
        probabilities = self.calibrator.transform(np.stack([row.logits for row in scored]))
        judgments = []
        for (assertion, unit, probe), row, tokens in zip(keys, probabilities, scored, strict=True):
            index = int(np.argmax(row))
            label, confidence = LABELS[index], float(row[index])
            band = self.bands.assign(confidence)
            accepted = self._accepts(band)
            effective = INVERTED.get(label, label) if probe else label
            judgment = nli_judgment(assertion, unit, effective if accepted else "not_enough_evidence", confidence)
            judgments.append(judgment | {"raw_label": label, "judged_text": probe or assertion["text"],
                                         "band": band.value if band else None,
                                         "probabilities": [float(value) for value in row],
                                         "token_len": tokens.token_len})
        return judgments
