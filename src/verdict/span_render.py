"""Render a composed verdict from verbatim sentences so nothing can be paraphrased into it.

Every figure, definition and attribution a reader sees is copied from a source
sentence, a source record or the claim. Attribution comes from the source record, so a
passage cannot be credited to the wrong publisher. The figure check is the contract for
any later generated explanation: a number that appears in prose but in no quoted
sentence, definition, claim or source date is rejected, because that is exactly where
earlier models changed a population or a denominator without touching a citation.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

MAX_RELEVANT = 6        # a page shows the strongest bearing sentences; the verdict data keeps them all

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_FIGURE = re.compile(r"[$€£]\s?\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d{1,3}(?:,\d{3})*\.\d+%?"
                     r"|\d{1,3}(?:,\d{3})*%|\d{1,3}(?:,\d{3})+|\d{4,}")

STATUS_TEXT = {"supported": "supported", "contradicted": "contradicted",
               "qualified_support": "supported but not established",
               "qualified_contradiction": "contradicted but not established",
               "contested": "contested by the sources", "insufficient": "not established by the sources"}
SUMMARY_TEXT = {"established": "Established", "partially_established": "Partially established",
                "partially_contradicted": "Partially contradicted", "qualified_support": "Supported, not established",
                "qualified_contradiction": "Contradicted, not established", "contested": "Contested",
                "unresolved": "Not established"}


def summary_text(verdict: dict) -> str:
    """The whole-claim line; a qualified direction on a contrast names the half it applies to."""
    text = SUMMARY_TEXT[verdict["summary"]]
    if verdict["summary"] in ("qualified_support", "qualified_contradiction") and len(verdict["assertions"]) > 1:
        return "One part " + text[0].lower() + text[1:]
    return text


def figures(text: str) -> list[str]:
    return _FIGURE.findall(text)


def ungrounded_figures(prose: str, grounds: list[str]) -> list[str]:
    """Figures in the prose that occur in none of the permitted texts."""
    return [figure for figure in figures(prose) if not any(figure in ground for ground in grounds)]


def display_text(text: str) -> str:
    """A quoted sentence as a reader sees it: link labels without their destinations, no bare URLs.

    The evidence record keeps the exact span; only the page shows the label. The panel applies
    the same two replacements in `displayText`.
    """
    return _BARE_URL.sub("", _MARKDOWN_LINK.sub(r"\1", text))


def _publisher(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host[4:] if host.startswith("www.") else host or "unnamed source"


READ_AS = {"states": "read as stating it", "states_negation": "read as denying it"}


def _row(row: dict) -> list[str]:
    where = f" (published {row['published_at']})" if row["published_at"] else " (publication date unconfirmed)"
    scope = f" Scope stated in the sentence: {'; '.join(row['qualifiers'])}." if row["qualifiers"] else ""
    # A counted sentence says which way it was read; a contested assertion is unreadable otherwise.
    direction = f" [{READ_AS[row['relation']]}]" if row["relation"] in READ_AS else ""
    lines = [f"  - {_publisher(row['url'])}{where}{direction}: \"{display_text(row['text'])}\"{scope}"]
    lines.extend(f"    Definition in the same paragraph: \"{display_text(item['text'])}\"" for item in row["definitions"])
    if row.get("figures"):
        lines.append("    Gives the claim's figure: " + "; ".join(f"{item['read_as']} fits \"{item['claimed']}\""
                                                             if item["relation"] != "exact" else f"{item['read_as']} as claimed"
                                                             for item in row["figures"]) + ".")
    if row.get("period"):
        lines.append(f"    Dated to {', '.join(row['period'])}, outside the claim's stated period; shown, not counted.")
    if row.get("origin") == "context":
        lines.append("    Found by caption-concept research, which does not resolve the claim.")
    elif row.get("not_counted") and not row.get("period"):
        lines.append(f"    Shown, not counted: {row['not_counted']}.")
    return lines


def _figures_line(row: dict, visible: set[str]) -> str | None:
    """The claim's figures found in bearing sentences quoted under this assertion, by the sentence's own
    publisher: the number checked, the claim not stated. A figure from a sentence the page does not
    show is not cited."""
    found: dict[str, tuple[list[str], list[str]]] = {}
    for item in row.get("figures_confirmed", []):
        if item["unit_id"] not in visible:
            continue
        readings, publishers = found.setdefault(f"{item['claimed']}|{item['relation']}", ([], []))
        readings.append(item["read_as"])
        publishers.append(_publisher(item["url"]))
    if not found:
        return None
    parts = []
    for key, (readings, publishers) in found.items():
        claimed, relation = key.split("|")
        given, who = " and ".join(dict.fromkeys(readings)), " and ".join(dict.fromkeys(publishers))
        parts.append(f"\"{claimed}\" is given as {given} by {who}" if relation != "exact" else f"{claimed} is given by {who}")
    return "  Its figure checks: " + "; ".join(parts) + " -- in sentences that bear on the claim without stating it."


def render(verdict: dict, claim: str) -> str:
    """Plain text built only from the claim, quoted sentences and source records."""
    lines = [f"Claim: \"{claim}\"",
             f"Verdict: {summary_text(verdict)} ({verdict['relationship']})."]
    lines.extend(f"Limit: {limit}" for limit in verdict["scope_limits"])
    lines.extend(f"Scope: {note}" for note in verdict.get("scope_notes", []))
    if verdict.get("judge_note"):
        lines.append(f"Reliability: {verdict['judge_note']}")
    shown: set[str] = set()      # a sentence quoted under one half of a contrast is not quoted again
    rendered: list[dict] = []    # every row the page quotes; the only sentences a figure may be grounded in
    for row in verdict["assertions"]:
        lines.append("")
        limits = f": {'; '.join(row['limits'])}" if row.get("limits") else ""
        basis = f" ({row['basis']})" if row.get("basis") else ""
        # The basis names dates the figure check cannot see (a day is not a figure): each must be a counted row's own.
        if stray_dates := set(_DATE.findall(basis)) - {item["published_at"] for item in row["evidence"]}:
            raise ValueError(f"Basis names a publication date no counted sentence carries: {sorted(stray_dates)}")
        lines.append(f"\"{row['text']}\" is {STATUS_TEXT[row['status']]}{basis}{limits}.")
        fresh = [item for item in row["relevant"] if item["unit_id"] not in shown][:MAX_RELEVANT]
        if row["evidence"]:
            lines.append("  Counted:")
            for item in row["evidence"]:
                lines.extend(_row(item))
        elif figures := _figures_line(row, {item["unit_id"] for item in fresh}):
            lines.append(figures)
        if fresh:
            lines.append("  Relevant, not counted:")
            for item in fresh:
                lines.extend(_row(item))
        if row["relevant"]:
            if repeated := sum(item["unit_id"] in shown for item in row["relevant"]):
                lines.append(f"    {repeated} relevant sentence{'s' if repeated != 1 else ''} already shown above.")
            if (more := len(row["relevant"]) - repeated - len(fresh)) > 0:
                lines.append(f"    ... and {more} more relevant sentences not shown.")
        if not row["evidence"] and not row["relevant"]:
            lines.append(f"  No eligible sentence addresses it ({row['judged']} of {row['eligible']} judged).")
        rendered.extend(row["evidence"] + fresh)
        shown.update(item["unit_id"] for item in row["evidence"] + fresh)
    if verdict["withheld"]:
        withheld = ", ".join(f"{count} {reason}" for reason, count in sorted(verdict["withheld"].items()))
        lines.append("")
        lines.append(f"Read as context, never as evidence: {withheld}.")
    text = "\n".join(lines)
    quoted = [item["text"] for item in rendered] + [definition["text"] for item in rendered for definition in item["definitions"]]
    # The reliability line's figures come from the judge's measurement record, not from any source or model prose.
    grounds = [claim, *quoted, *map(display_text, quoted), *(item["published_at"] for item in rendered),
               verdict.get("judge_note", "")]
    stray = ungrounded_figures(text, grounds)
    if stray:
        raise ValueError(f"Rendered prose contains figures found in no quoted sentence: {stray}")
    return text
