"""
Checkpoint 3: how good is the check-worthiness filter, measured rather than asserted?

The filter decides what the sidebar shows. Left unmeasured it would quietly set the demo's content
-- a claim that never appears is indistinguishable from a claim the system got right, and a
transcript that yields four verdicts out of forty looks the same whether the filter was selective
or simply broken.

**Precision and recall are not equally important here, and the asymmetry is deliberate.** A false
positive costs a retrieval and a forward pass, and the sufficiency gate is likely to decline it
anyway -- the claim appears with an honest "not enough evidence" beside it. A false negative is
invisible: the claim is silently absent and no downstream check can recover it. So recall is the
number to watch, and the per-reason table exists to say *which rule* is dropping real claims.

**The known bias, stated up front.** The heuristic and the hand labels share an author, so this
measures one person's judgement against their own rules rather than agreement between two
independent judges. It is a sanity check with a known direction of error, not validation. The
labels carry a written rubric precisely so a second annotator can redo them; `labels/*.json`
records the limitation alongside the data.

The rubric labels a sentence check-worthy if it asserts something about the world that could be
looked up. It deliberately does **not** encode the heuristic's anchor requirement -- needing a
name, number or date is a limitation of a BM25-over-Wikipedia retrieval stack, not part of what
makes a claim worth checking. Recall is expected to suffer there, and measuring that gap is most
of the point.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from src.pipeline.segment import check_worthy, segment

TRANSCRIPTS = Path("data/transcripts")
LABELS = Path("labels")


def missed_by_reason(labelled: dict[int, bool], decisions: dict[int, object]) -> Counter:
    """
    Which rule dropped each real claim.

    Recall says the filter is losing claims; this says which rule to go and fix. Without it the
    measurement is a grade rather than a diagnosis, and on this transcript it is the whole finding
    -- seven of fourteen misses are the anchor requirement, which is a limitation of the retrieval
    stack rather than a judgement about the claim.
    """
    return Counter(
        decisions[i].reason for i in sorted(labelled) if labelled[i] and not decisions[i].worthy
    )


def counts(predicted: list[bool], actual: list[bool]) -> dict[str, int]:
    pairs = Counter(zip(predicted, actual, strict=True))
    return {
        "tp": pairs[(True, True)], "fp": pairs[(True, False)],
        "fn": pairs[(False, True)], "tn": pairs[(False, False)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript", default="sotu-2016")
    parser.add_argument("--show-errors", action="store_true", help="print every disagreement")
    args = parser.parse_args()

    text = (TRANSCRIPTS / f"{args.transcript}.txt").read_text(encoding="utf-8")
    truth = json.loads(
        (LABELS / f"checkworthy-{args.transcript}.json").read_text(encoding="utf-8")
    )
    sentences = {s.index: s for s in segment(text)}
    labelled = {int(i): v for i, v in truth["labels"].items()}

    missing = sorted(set(labelled) - set(sentences))
    if missing:
        raise SystemExit(
            f"{len(missing)} labelled indices are not in the segmentation ({missing[:5]}...). "
            "The transcript or the segmenter changed since the labels were made; re-label or "
            "pin the transcript, but do not silently score against a different set of sentences."
        )

    decisions = {i: check_worthy(sentences[i].text) for i in labelled}
    predicted = [decisions[i].worthy for i in sorted(labelled)]
    actual = [labelled[i] for i in sorted(labelled)]

    c = counts(predicted, actual)
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else float("nan")
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")

    print(f"{args.transcript}: {len(sentences):,} sentences, {len(labelled)} hand-labelled "
          f"({truth['sample']})")
    print(f"base rate: {sum(actual)}/{len(actual)} = {sum(actual) / len(actual):.1%} check-worthy")
    print(f"\n{'':>14} {'kept':>8} {'dropped':>8}")
    print(f"{'check-worthy':>14} {c['tp']:>8} {c['fn']:>8}")
    print(f"{'not':>14} {c['fp']:>8} {c['tn']:>8}")
    print(f"\nprecision {precision:.4f}   recall {recall:.4f}   F1 {f1:.4f}   "
          f"accuracy {(c['tp'] + c['tn']) / len(actual):.4f}")

    # --- which rule drops a real claim ---------------------------------------------------------
    dropped = missed_by_reason(labelled, decisions)
    if dropped:
        print(f"\n{sum(dropped.values())} real claims dropped, by the rule that fired:")
        for reason, n in dropped.most_common():
            print(f"  {reason:<16} {n:>3}")
    kept_wrongly = Counter(
        "check_worthy" for i in sorted(labelled) if not labelled[i] and decisions[i].worthy
    )
    if kept_wrongly:
        print(f"\n{sum(kept_wrongly.values())} non-claims kept -- these reach retrieval and are "
              "expected to be declined by the sufficiency gate rather than answered")

    if args.show_errors:
        print("\ndisagreements:")
        for i in sorted(labelled):
            if labelled[i] != decisions[i].worthy:
                kind = "MISSED" if labelled[i] else "spurious"
                print(f"  [{kind:>8}] {decisions[i].reason:<14} {sentences[i].text[:96]}")

    print(f"\nlimitation: {truth['limitation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
