"""Compare fixed development queries; title hits are not evidence sufficiency or accuracy.

The saved transcript supplies text, not real ASR timestamps. Every retrieval arm
packs evidence against the unchanged claim with the shipped encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from src.pipeline.context import SpeechMetadata, build_query
from src.pipeline.transcript import TranscriptUpdate, TranscriptWindow
from src.pipeline.verify import PAGES, SENTENCES, Verifier
from src.retrieval.search import DEFAULT_MAX_TITLES
from src.verdict.encode import pack

CASES = (50, 89, 91, 149)
TARGET_TITLE = "Patient_Protection_and_Affordable_Care_Act"
SOTU_SOURCE = "https://www.govinfo.gov/content/pkg/DCPD-201600012/html/DCPD-201600012.htm"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript_results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.transcript_results.read_bytes()
    transcript = json.loads(source)
    if transcript.get("transcript") != "sotu-2016" or transcript.get("source") != SOTU_SOURCE:
        raise ValueError("Expected the 2016 SOTU transcript and source")
    indices = [row["index"] for row in transcript["rows"]]
    if indices != list(range(len(indices))) or not set(CASES).issubset(indices):
        raise ValueError("Expected contiguous transcript rows including every selected case")
    metadata = SpeechMetadata(country="United States", spoken_at="2016-01-12")
    window, verifier = TranscriptWindow(use_references=True), Verifier()
    results, cache = [], {}
    try:
        for row in transcript["rows"]:
            index = row["index"]
            context = window.accept(TranscriptUpdate(str(index), 0, row["text"], index * 2, index * 2 + 1, True))
            if index not in CASES:
                continue
            arms = {
                "claim": build_query(context, metadata),
                "previous_context": build_query(replace(context, reference_context=()), metadata, mode="context"),
                "law_context": build_query(context, metadata, mode="context"),
            }
            item = {"index": index, "context": asdict(context), "arms": {}}
            for name, query in arms.items():
                if query.query not in cache:
                    cache[query.query] = verifier.retrieve(query.query)
                evidence, scores = cache[query.query]
                used = pack(query.claim, evidence, verifier.runtime.contract.max_length, verifier.runtime.measure)
                item["arms"][name] = {
                    "query": asdict(query), "retrieved": evidence, "scores": scores,
                    "packed": evidence[:used], "packed_count": used,
                    "aca_first_packed_rank": next((i + 1 for i, e in enumerate(evidence[:used])
                                                   if e[0] == TARGET_TITLE), None),
                }
            results.append(item)
            print(f"Compared segment {index}", flush=True)
    finally:
        verifier.close()
    payload = {
        "source_sha256": hashlib.sha256(source).hexdigest(), "indices": CASES,
        "metadata": asdict(metadata), "corpus": "FEVER Wikipedia June 2017; retrospective diagnostic",
        "scope": "Fixed selected cases; no verdict accuracy or sufficiency assessment. Timestamps are synthetic.",
        "retrieval": {"pages": PAGES, "sentences": SENTENCES, "injector_cap": DEFAULT_MAX_TITLES},
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
