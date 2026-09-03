"""
Run a transcript through the whole pipeline and write what happened to every sentence.

This is the demo's data step: segment, filter, retrieve, score, calibrate, gate. The output keeps
**every** sentence, not just the ones that produced a verdict, because the rejections are the
demo's honesty. A page showing four verdicts out of forty sentences is telling the truth only if a
reader can see the other thirty-six and why each was skipped.

Three distinct reasons a sentence carries no verdict, and they are kept apart end to end:

  not check-worthy   a question, an imperative, an opinion, or nothing retrieval can grip
  declined           the model's confidence earned no band, or the evidence was inadequate
  answered NEI       a *verdict* -- we looked and found nothing that settles it

The middle one is a statement about the model; the last is a statement about the world. Collapsing
them would make the system's honesty unfalsifiable, which is the whole thing this project is for.

**Expect heavy abstention, and do not treat it as a bug.** The verdict model was trained on FEVER
and the corpus is a June 2017 Wikipedia dump, so political speech is out of distribution in both
directions at once: the wording is unlike FEVER's claims, and the evidence often is not in the
corpus at all. The abstention rate is a headline number of this demo. Tuning the gate until the
page looks busier would void every promise Phases 03 and 04 measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.pipeline.detector import DetectorFilter
from src.pipeline.segment import check_worthy, segment
from src.pipeline.verify import Verifier

TRANSCRIPTS = Path("data/transcripts")
RUNS = Path("runs/demo")


def select(sentences, args) -> list:
    """
    Which filter decides what reaches the model.

    The default is the learned detector, because it was measured: F1 0.7619 against the rules'
    0.3103 on 120 hand labels neither system was fitted to. The rules stay reachable, and not out
    of sentiment -- they are the only filter whose rejections name a clause a reader can argue
    with, which makes them the better instrument when the question is *why* something was skipped.
    """
    if args.filter == "rules":
        return [check_worthy(s.text) for s in sentences]
    chosen = DetectorFilter(args.binarization)
    print(f"filter: learned detector, {args.binarization} >= {chosen.threshold} "
          f"(threshold frozen on ClaimBuster's calibration debates)", flush=True)
    return chosen.decide_batch([s.text for s in sentences])


def to_row(sentence, decision, judgement) -> dict:
    """
    One sentence's full record, whatever happened to it.

    Every sentence gets a row, including the ones no model ever saw. A page that lists only the
    verdicts is not showing its work: a reader cannot tell a claim the system answered from one
    the filter silently dropped, and the abstention rate -- the headline number here -- would have
    no denominator.
    """
    row: dict = {
        "index": sentence.index, "text": sentence.text,
        "start": sentence.start, "end": sentence.end,
        "check_worthy": decision.worthy, "filter_reason": decision.reason,
    }
    # The detector reports a score where the rules report a clause. Carrying it lets the page say
    # which filter ran and how sure it was, rather than implying a rule fired.
    if hasattr(decision, "score"):
        row["filter_score"] = float(decision.score)
    if judgement is not None:
        row |= {
            "outcome": str(judgement.outcome),
            "verdict": str(judgement.verdict) if judgement.verdict else None,
            "predicted": str(judgement.predicted),
            "band": str(judgement.band) if judgement.band else None,
            "confidence": judgement.confidence,
            "sufficiency": judgement.sufficiency,
            "evidence": [list(e) for e in judgement.evidence],
        }
    elif decision.worthy:
        row["outcome"] = "not_verified"       # only reachable under --limit
    return row


def coverage_of(rows: list[dict]) -> tuple[list[dict], list[dict], float]:
    """
    Answered, declined, and the share answered *among claims that reached the model*.

    The denominator is deliberately not the sentence count. Coverage is what the gate governs, and
    dividing by every sentence in the transcript would fold the check-worthiness filter's
    behaviour into a number that is supposed to describe the abstention policy.
    """
    answered = [r for r in rows if r.get("outcome") == "answered"]
    declined = [r for r in rows if str(r.get("outcome", "")).startswith("declined")]
    return answered, declined, len(answered) / max(len(answered) + len(declined), 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="sotu-2016")
    parser.add_argument("--variant", default="retrieved")
    parser.add_argument("--limit", type=int, default=0, help="verify at most N claims")
    parser.add_argument("--filter", default="detector", choices=("detector", "rules"),
                        help="which check-worthiness filter selects the claims")
    parser.add_argument("--binarization", default="factual",
                        choices=("factual", "check_worthy"),
                        help="detector only: which definition of check-worthy to gate on")
    parser.add_argument("--out", default=str(RUNS))
    args = parser.parse_args()

    text = (TRANSCRIPTS / f"{args.transcript}.txt").read_text(encoding="utf-8")
    sentences = segment(text)
    decisions = list(zip(sentences, select(sentences, args), strict=True))
    all_worthy = [s for s, d in decisions if d.worthy]
    # --limit truncates what gets verified, never what gets counted: reporting the truncated
    # number as the check-worthy total would understate how much the filter admitted.
    worthy = all_worthy[: args.limit] if args.limit else all_worthy

    print(f"{args.transcript}: {len(sentences):,} sentences, {len(all_worthy):,} check-worthy "
          f"({len(all_worthy) / len(sentences):.1%})")
    if len(worthy) != len(all_worthy):
        print(f"  --limit {args.limit}: verifying {len(worthy):,} of them")
    print("retrieving and scoring ... first claim is slow while the index pages in", flush=True)

    verifier = Verifier(args.variant)
    judged: dict[int, object] = {}
    started = time.time()
    for n, sentence in enumerate(worthy, 1):
        judged[sentence.index] = verifier.verify(sentence.text)
        if n % 10 == 0 or n == len(worthy):
            print(f"  {n}/{len(worthy)}  {(time.time() - started) / n:.2f}s/claim", flush=True)
    verifier.close()

    rows = [to_row(s, d, judged.get(s.index)) for s, d in decisions]
    answered, declined, coverage = coverage_of(rows)

    payload = {
        "transcript": args.transcript,
        "variant": args.variant,
        "filter": args.filter,
        "binarization": args.binarization if args.filter == "detector" else None,
        "source": "https://www.govinfo.gov/content/pkg/DCPD-201600012/html/DCPD-201600012.htm",
        "sentences": len(sentences),
        "check_worthy": len(all_worthy),
        "verified": len(worthy),
        "answered": len(answered),
        "declined": len(declined),
        # Coverage among claims that reached the model, which is the number the gate governs.
        "coverage": coverage,
        "calibration": {
            "fitted_on": "FEVER calibration split (13,332 claims)",
            "measured_on": "FEVER test split (2,000 claims)",
            "bands": {b: dict(zip(("promised", "measured", "n"),
                                  (v["promised"], v["measured"], v["n"]), strict=True))
                      for b, v in json.loads(
                          (Path("runs/sufficiency-test") / "metrics.json").read_text(
                              encoding="utf-8")
                      )["bands"].items()},
            "out_of_domain": (
                "These promises were measured on FEVER, whose claims are short synthetic "
                "statements about Wikipedia entities. Political speech is out of distribution in "
                "both directions -- the wording is unlike FEVER's, and the June 2017 corpus often "
                "does not contain the evidence at all. The band accuracies below describe FEVER, "
                "not this transcript, and are shown with that provenance or not at all."
            ),
        },
        "rows": rows,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdicts.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print(f"\n{'outcome':<34} {'n':>5}")
    print(f"{'not check-worthy':<34} {len(sentences) - len(all_worthy):>5}")
    print(f"{'answered':<34} {len(answered):>5}")
    print(f"{'declined':<34} {len(declined):>5}")
    print(f"\ncoverage among claims reaching the model: {payload['coverage']:.1%}")
    print(f"wrote {out / 'verdicts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
