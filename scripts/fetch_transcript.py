"""
Fetch a real political transcript for the demo, from an official public-domain source.

The check-worthiness filter decides what the sidebar shows, so it has to be measured against
something. A transcript written for the occasion would not do: the filter and the text would share
an author, and the resulting precision and recall would measure nothing but agreement with itself.
This fetches speech that was actually delivered, by someone with no knowledge of the heuristic.

**govinfo, not a news site.** The Daily Compilation of Presidential Documents is prepared by the
Office of the Federal Register and is a work of the United States Government, so the text is in
the public domain -- no licence question about redistributing what the demo quotes back. News
transcripts of the same speech carry the outlet's own rights in the surrounding page.

`DCPD-201600012` is the 2016 State of the Union. It suits the demo for a reason worth stating:
FEVER's corpus is a June 2017 Wikipedia dump, so a January 2016 speech is one of the few political
texts where retrieval has any chance at all. Newer speech is further out of distribution, not
closer -- and the sidebar says so rather than hiding it.

The compilation's editorial apparatus is stripped: `[Laughter]`, `[Applause]`, speaker labels,
and the trailing index. None of it was asserted by anyone, and leaving it in hands the segmenter
sentences nobody said -- the index alone is a single 514-word "sentence" of semicolon-separated
subject headings, which the check-worthiness filter then has to rule on.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import urllib.request
from pathlib import Path

DEST = Path("data/transcripts")
GOVINFO = "https://www.govinfo.gov/content/pkg/{id}/html/{id}.htm"
DEFAULT_ID = "DCPD-201600012"

# Editorial insertions, not speech. Audience reaction and the compilation's own speaker labels.
STAGE = re.compile(r"\[(Laughter|Applause|Inaudible|At this point[^\]]*)\]", re.I)
SPEAKER = re.compile(r"^(The President|The Vice President|Q|Audience members?)\.\s*", re.I)
# The compilation closes with an editorial note and a subject index. Everything from the first of
# these markers onward is apparatus, not speech.
APPARATUS = re.compile(
    r"^(NOTE:|Names:|Subjects:|Categories:|Locations:|DCPD Number:|In his remarks, |"
    r"In her remarks, |The President spoke at |The transcript released )", re.I
)
PARAGRAPH = re.compile(r"<P[^>]*>(.*?)</P", re.S | re.I)
TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


def to_text(markup: str) -> tuple[str, list[str]]:
    """The spoken text, one paragraph per line, plus the header lines that precede it."""
    paragraphs: list[str] = []
    for raw in PARAGRAPH.findall(markup):
        line = WHITESPACE.sub(" ", html.unescape(TAG.sub("", raw))).strip()
        line = SPEAKER.sub("", STAGE.sub("", line))
        line = WHITESPACE.sub(" ", line).strip()
        if APPARATUS.match(line):
            break
        if line:
            paragraphs.append(line)
    # The compilation opens with the administration, the title and the date, each its own
    # paragraph. They are provenance, not speech, so they are returned separately.
    return "\n\n".join(paragraphs[3:]), paragraphs[:3]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", default=DEFAULT_ID, help="govinfo DCPD package id")
    parser.add_argument("--name", default="sotu-2016", help="output stem under data/transcripts")
    parser.add_argument("--dest", default=str(DEST))
    args = parser.parse_args()

    url = GOVINFO.format(id=args.id)
    print(f"fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response:      # noqa: S310 - fixed govinfo host
        markup = response.read().decode("utf-8", errors="replace")

    text, header = to_text(markup)
    if len(text) < 5_000:
        raise SystemExit(f"{args.id} yielded {len(text):,} characters; wrong package id?")

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{args.name}.txt"
    path.write_text(text, encoding="utf-8")

    for line in header:
        print(f"  {line}")
    print(f"\nwrote {path}  {len(text):,} characters, {text.count(chr(10) * 2) + 1:,} paragraphs")
    print(f"source: {url}  (US Government work, public domain)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
