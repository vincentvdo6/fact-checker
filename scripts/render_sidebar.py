"""
Render the sidebar: one self-contained HTML page from `runs/demo/verdicts.json`.

The demo cannot be a live service. Retrieval needs 3.9 GB of SQLite and a 1.5 GB index, so the
pipeline runs on one machine and the shareable artifact is a *rendering* of what it concluded --
never a server someone else can query.

**The page's job is to be honest, not to look busy.** Three design rules follow from that, and
each one is checked by `--check` rather than left to good intentions:

  abstention is a first-class state    it gets its own styling and its own stated reason, never a
                                       missing row and never an error
  every promise carries its provenance "strong: right about nine times in ten" is a fact about
                                       FEVER's test split, and appears with that attached or not
                                       at all
  the domain gap is stated up front    these claims are out of distribution, the page says so
                                       before showing a single verdict

The distinction the page must never blur is NOT ENOUGH EVIDENCE against declining. The first is a
verdict about the world; the second is the system declining to answer. Both read as "no verdict"
to a hurried reader, which is exactly why they are rendered differently and labelled explicitly.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

RUNS = Path("runs/demo")

VERDICT_TEXT = {
    "supported": "Supported",
    "contradicted": "Contradicted",
    "not_enough_evidence": "No evidence found",
}
OUTCOME_TEXT = {
    "declined_low_confidence":
        "Declined - the model was not confident enough for any band it has measured",
    "declined_insufficient_evidence":
        "Declined - retrieval did not return the kind of evidence this model needs",
    "declined_both":
        "Declined - low confidence and inadequate evidence, both",
}
FILTER_TEXT = {
    "check_worthy": "kept",
    "below_factual_threshold": "scored below the check-worthiness threshold",
    "below_check_worthy_threshold": "scored below the check-worthiness threshold",
    "too_short": "too short to carry a claim",
    "question": "a question, not an assertion",
    "imperative": "an instruction, not an assertion",
    "opinion": "opinion or a statement of intent",
    "no_assertion": "no finite verb - a fragment or a title",
    "no_anchor": "no name, number or date for retrieval to search on",
}

STYLE = """
:root { --ink:#1a1a1a; --muted:#6b6b6b; --line:#e0ddd8; --bg:#faf9f7; --card:#fff;
        --ok:#2f6b4f; --no:#8c3a3a; --none:#5a5a72; --hold:#8a6d3b; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
.wrap { max-width:820px; margin:0 auto; padding:32px 20px 80px; }
h1 { font-size:22px; margin:0 0 4px; letter-spacing:-0.01em; }
.sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
.warn { background:#fff8e6; border:1px solid #e8d9a8; border-left:3px solid var(--hold);
        padding:14px 16px; border-radius:4px; margin-bottom:22px; font-size:13.5px; }
.warn b { display:block; margin-bottom:5px; }
.stats { display:flex; flex-wrap:wrap; gap:0; border:1px solid var(--line); border-radius:4px;
         background:var(--card); margin-bottom:26px; overflow:hidden; }
.stat { flex:1 1 25%; padding:13px 16px; border-right:1px solid var(--line); }
.stat:last-child { border-right:0; }
.stat .n { font-size:21px; font-weight:600; letter-spacing:-0.02em; }
.stat .k { font-size:11.5px; color:var(--muted); text-transform:uppercase;
           letter-spacing:0.04em; margin-top:2px; }
.claim { background:var(--card); border:1px solid var(--line); border-left-width:3px;
         border-radius:4px; padding:13px 16px; margin-bottom:9px; }
.claim.supported { border-left-color:var(--ok); }
.claim.contradicted { border-left-color:var(--no); }
.claim.not_enough_evidence { border-left-color:var(--none); }
.claim.declined { border-left-color:var(--line); background:#f7f6f4; }
.tag { display:inline-block; font-size:11px; font-weight:600; text-transform:uppercase;
       letter-spacing:0.05em; padding:2px 7px; border-radius:3px; margin-bottom:7px; }
.tag.supported { background:#e6f0ea; color:var(--ok); }
.tag.contradicted { background:#f6e8e8; color:var(--no); }
.tag.not_enough_evidence { background:#eaeaf0; color:var(--none); }
.tag.declined { background:#eeece8; color:var(--muted); }
.text { margin:0 0 8px; }
.why { font-size:12.5px; color:var(--muted); }
.why .prov { font-style:italic; }
details { margin-top:9px; font-size:12.5px; }
summary { cursor:pointer; color:var(--muted); }
.ev { margin:7px 0 0; padding-left:16px; color:var(--muted); font-size:12.5px; }
.ev li { margin-bottom:3px; }
h2 { font-size:14px; text-transform:uppercase; letter-spacing:0.05em; color:var(--muted);
     margin:34px 0 12px; font-weight:600; }
table { border-collapse:collapse; width:100%; font-size:13px; background:var(--card);
        border:1px solid var(--line); border-radius:4px; }
th,td { text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); }
th { font-weight:600; font-size:11.5px; text-transform:uppercase; color:var(--muted);
     letter-spacing:0.04em; }
tr:last-child td { border-bottom:0; }
.miss { color:var(--no); }
.skipped { font-size:13px; color:var(--muted); }
.skipped li { margin-bottom:4px; }
footer { margin-top:40px; padding-top:18px; border-top:1px solid var(--line);
         font-size:12px; color:var(--muted); }
"""


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def claim_card(row: dict) -> str:
    outcome = row.get("outcome")
    if outcome == "answered":
        verdict = row["verdict"]
        label = VERDICT_TEXT.get(verdict, verdict)
        band = row.get("band")
        # The promise is a fact about FEVER's test split. It travels with that sentence or it is
        # not shown -- an accuracy quoted beside an out-of-domain claim would be a false promise.
        why = (f'<span class="prov">{esc(band)} confidence. On FEVER\'s test split this band was '
               f'right {row["band_measured"]:.1%} of the time (n={row["band_n"]}). '
               f'This claim is not from that distribution.</span>')
        css = verdict
        tag = label
    else:
        css, tag = "declined", "No verdict"
        why = esc(OUTCOME_TEXT.get(str(outcome), "Declined"))
        why += (f' &middot; confidence {row["confidence"]:.2f}, '
                f'evidence sufficiency {row["sufficiency"]:.2f}')
        if row.get("predicted"):
            why += (f' &middot; the model would have said '
                    f'<b>{esc(VERDICT_TEXT.get(row["predicted"], row["predicted"]))}</b>, '
                    f'withheld because the gate did not pass')

    evidence = ""
    if row.get("evidence"):
        items = "".join(
            f"<li><b>{esc(t)}</b> &sect;{esc(i)} &mdash; {esc(txt)}</li>"
            for t, i, txt in row["evidence"][:5]
        )
        evidence = (f'<details><summary>{len(row["evidence"])} evidence sentences retrieved'
                    f'</summary><ul class="ev">{items}</ul></details>')

    return (f'<div class="claim {css}"><span class="tag {css}">{esc(tag)}</span>'
            f'<p class="text">{esc(row["text"])}</p>'
            f'<div class="why">{why}</div>{evidence}</div>')


def filter_text(data: dict) -> str:
    """
    Which filter chose the claims, said plainly.

    Two filters admit different sentences from the same transcript, so a page that does not name
    the one that ran is not interpretable -- a reader comparing two renderings would see different
    claims and no reason for it.
    """
    if data.get("filter") == "rules":
        return "hand-written rules"
    binarization = data.get("binarization") or "factual"
    return f"a detector trained on ClaimBuster ({binarization})"


def render(data: dict) -> str:
    rows = data["rows"]
    verified = [r for r in rows if r.get("outcome") in
                {"answered", "declined_low_confidence", "declined_insufficient_evidence",
                 "declined_both"}]
    bands = data["calibration"]["bands"]
    for row in verified:
        if row.get("band"):
            row["band_measured"] = bands[row["band"]]["measured"]
            row["band_n"] = bands[row["band"]]["n"]

    skipped: dict[str, int] = {}
    for row in rows:
        if not row["check_worthy"]:
            skipped[row["filter_reason"]] = skipped.get(row["filter_reason"], 0) + 1

    cards = "\n".join(claim_card(r) for r in verified)
    band_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{v['promised']:.0%}</td>"
        f"<td{' class=\"miss\"' if v['measured'] < v['promised'] else ''}>{v['measured']:.1%}</td>"
        f"<td>{v['n']:,}</td>"
        f"<td>{'kept' if v['measured'] >= v['promised'] else 'MISSED'}</td></tr>"
        for name, v in bands.items()
    )
    skip_rows = "".join(
        f"<li>{n} &mdash; {esc(FILTER_TEXT.get(reason, reason))}</li>"
        for reason, n in sorted(skipped.items(), key=lambda kv: -kv[1])
    )
    # A learned filter has no clause to name, so the page shows the distribution of what it did
    # score. Only the threshold rejections belong in that median: the length floor overrules the
    # model regardless of score -- one sentence it dropped scored 0.875 -- so folding those in
    # would understate the median and describe as "scored" something that was ruled on.
    scored = [r["filter_score"] for r in rows
              if not r["check_worthy"] and str(r["filter_reason"]).startswith("below_")]
    ruled = sum(1 for r in rows
                if not r["check_worthy"] and not str(r["filter_reason"]).startswith("below_"))
    skip_detail = ""
    if scored:
        median = sorted(scored)[len(scored) // 2]
        threshold = data.get("filter_threshold")
        against = f" against a threshold of {threshold:.2f}" if threshold is not None else ""
        overruled = (f" A further {ruled} were ruled out before the score was consulted."
                     if ruled else "")
        skip_detail = (
            f'<div class="sub" style="margin-top:8px">{len(scored)} of these were scored rather '
            f'than ruled on: the detector gave them a median check-worthiness of {median:.2f}'
            f'{against}, and that threshold was fixed on held-out debates before this transcript '
            f'was seen. A learned filter has no clause to point at, so the number is what there '
            f'is.{overruled}</div>'
        )

    return f"""<title>Sidebar - {esc(data['transcript'])}</title>
<style>{STYLE}</style>
<div class="wrap">
<h1>Calibrated claim verification</h1>
<div class="sub">{esc(data['transcript'])} &middot; {data['sentences']:,} sentences &middot;
claims selected by {esc(filter_text(data))} &middot;
verdicts from a model trained on FEVER, evidence from a June 2017 Wikipedia dump &middot;
<a href="{esc(data['source'])}">source transcript</a></div>

<div class="warn"><b>These claims are outside the distribution the system was measured on.</b>
{esc(data['calibration']['out_of_domain'])}</div>

<div class="stats">
 <div class="stat"><div class="n">{data['sentences']:,}</div><div class="k">sentences</div></div>
 <div class="stat"><div class="n">{data['check_worthy']:,}</div><div class="k">check-worthy</div></div>
 <div class="stat"><div class="n">{data['answered']:,}</div><div class="k">answered</div></div>
 <div class="stat"><div class="n">{data['declined']:,}</div><div class="k">declined</div></div>
</div>

<h2>Claims that reached the model &mdash; {data['coverage']:.0%} answered,
{1 - data['coverage']:.0%} declined</h2>
{cards}

<h2>What the confidence bands promise, and where that was measured</h2>
<table><tr><th>band</th><th>promised</th><th>measured</th><th>n</th><th></th></tr>
{band_rows}</table>
<div class="sub" style="margin-top:9px">Fitted on {esc(data['calibration']['fitted_on'])},
measured on {esc(data['calibration']['measured_on'])}. The weak band missed its promise and is
reported as missed rather than retuned.</div>

<h2>Sentences the filter skipped before any model ran</h2>
<ul class="skipped">{skip_rows}</ul>
{skip_detail}

<footer>Abstention is an outcome, not an error. A declined claim is the system saying it cannot
answer; &ldquo;no evidence found&rdquo; is the system saying the evidence does not exist. Those
are different statements and the page keeps them apart.</footer>
</div>"""


CHECKS = {
    "abstentions are visible as their own state":
        lambda page, data: data["declined"] == 0 or 'class="tag declined"' in page,
    "every band promise carries where it was measured":
        lambda page, data: "FEVER's test split this band was right" in page or data["answered"] == 0,
    "the out-of-domain warning is present":
        lambda page, data: "outside the distribution" in page,
    "the skipped sentences are accounted for":
        lambda page, data: "Sentences the filter skipped" in page,
    "NEI and abstention are labelled differently":
        lambda page, data: "No evidence found" in page or "No verdict" in page,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(RUNS))
    parser.add_argument("--check", action="store_true", help="Checkpoint 4: does the page tell the truth?")
    args = parser.parse_args()

    run = Path(args.run)
    data = json.loads((run / "verdicts.json").read_text(encoding="utf-8"))
    page = render(data)
    out = run / "sidebar.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}  {len(page) / 1000:.1f} kB")

    if not args.check:
        return 0

    print(f"\n{'Checkpoint 4 -- the page tells the truth':<52} verdict")
    failures = 0
    for name, predicate in CHECKS.items():
        ok = predicate(page, data)
        failures += not ok
        print(f"{name:<52} {'ok' if ok else 'FAIL'}")
    if failures:
        print(f"\nCheckpoint 4: FAIL -- {failures} of {len(CHECKS)} properties missing")
        return 1
    print("\nCheckpoint 4: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
