"""Write the truth file for a batch of recorded clicks, blind: every researched claim, no verdict.

`measure_directions` scores the reading's direction against a truth value per claim, and the
value must come from someone who has not seen the verdict. This lists the claims the checker
actually researched in a click directory and writes them with `"true": null`, so the person
fills in true / false (or leaves null for a claim they cannot settle) before running the
measurement; an existing file keeps every value already filled and only gains new claims.

    python -m scripts.claim_truth_template --clicks runs/clicks --out labels/claim-truth-clicks-2026-09-20.json
    # fill in "true": true / false, then
    python -m scripts.measure_directions --clicks runs/clicks --truth labels/claim-truth-clicks-2026-09-20.json
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


def researched_claims(click_dir: Path) -> list[dict]:
    """(claim, video, position) for every researched claim, in click order; verdicts are not read."""
    found: list[dict] = []
    seen: set[str] = set()
    for path in sorted(click_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        result = record.get("result") or {}
        for row in result.get("rows") or []:
            outcome = row.get("result") or {}
            decomposed = outcome.get("decomposed") or {}
            claim = decomposed.get("claim") or row.get("text")
            if "research" not in outcome or not claim or claim in seen:
                continue
            seen.add(claim)
            found.append({"claim": claim, "video": result.get("video_id", ""), "title": result.get("title", ""),
                          "at_seconds": result.get("clicked_at"), "country": result.get("claim_country") or ""})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clicks", type=Path, default=Path("runs/clicks"))
    parser.add_argument("--out", type=Path, default=Path(f"labels/claim-truth-clicks-{date.today().isoformat()}.json"))
    parser.add_argument("--labeller", default="", help="who fills the values in; recorded on the file")
    args = parser.parse_args()
    existing = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    claims = dict(existing.get("claims", {}))
    added = 0
    for item in researched_claims(args.clicks):
        if item["claim"] not in claims:
            claims[item["claim"]] = {"true": None, "note": "", "video": item["video"], "title": item["title"],
                                     "at_seconds": item["at_seconds"], "country": item["country"]}
            added += 1
    payload = {"labeller": args.labeller or existing.get("labeller", ""),
               "written": existing.get("written", date.today().isoformat()),
               "instructions": "set \"true\" to true or false for each claim as you understand it, from your own knowledge or "
                               "sources you check yourself, without reading the checker's verdict; leave null when you cannot "
                               "settle it; a note is optional",
               "batches": existing.get("batches", {}) | {str(args.clicks): f"recorded clicks, template written {date.today().isoformat()}"},
               "claims": claims}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    unfilled = sum(1 for value in claims.values() if value.get("true") is None)
    print(f"{len(claims)} claims in {args.out} ({added} new, {unfilled} still null)")


if __name__ == "__main__":
    main()
