"""Type what a source sentence does, by named closed-class cues, with no claim in view.

The gate withholds a sentence that reports an opinion, an expectation, an illustration or
navigation text, and keeps reported observations. A language model did that typing in the
diagnostics, and users will not run one. The cues below are the words that carry those
readings -- speech verbs beside a quotation, "in our view", "expects", "for example",
"is defined as", "click here" -- and each rule is named, so a sentence the gate withheld can
be traced to the cue that fired rather than to a score. Roles may co-occur; the gate reads
the set.

This is a heuristic and is measured as one: `scripts/measure_role_rules.py` scores it
against the hand-labelled roles in `labels/roles-*.json` and, until those are filled,
reports its agreement with the archived model annotations as a comparison, not a truth.
Errors are expected in both directions; the gate is conservative when they are in the
withholding direction and the pair judge still has to be wrong for a kept sentence to
count.

One cue does look outside the source: the speaker's own words. A search for a spoken claim
returns transcripts of the speech and articles quoting it, and a sentence that repeats the
claim is not evidence for it -- on recorded clicks it was the commonest way a false claim
was counted as supported. A sentence that repeats `REPRINT_RUN` consecutive words of the
captions is typed `speech_reprint`, and so is one sharing most of its word triples with them
when it opens a quotation or another sentence on its page matches too: captions and reprints
differ by disfluencies and spellings, while a lone unquoted near match is how a news report
restates a short claim. A page
whose URL names a transcript or the text of a speech is someone speaking throughout. Short
runs are left alone, because independent reporting of the same event in plain words is evidence.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from urllib.parse import urlsplit

from src.pipeline.segment import AUXILIARIES, IRREGULAR_PAST, QUESTION_OPENERS
from src.retrieval.visible_text import inline_links

_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)|https?://\S+|www\.\S+")
_QUOTE = re.compile(r"[\"“”][^\"“”]{6,}[\"“”]")
# A quotation whose opening mark sat in an earlier sentence still ends in its attribution.
_QUOTE_TAIL = re.compile(r"[\"”]\s*,?\s*(?:(?:(?:[A-Z][\w.'’-]*|he|she|they|we|I)\s+){1,6}"
                         r"(?:said|says|told|added|wrote|asked|stated|noted|explained|recalled|insisted|remarked|argued|claimed)\b"
                         r"|(?:said|says|added|wrote|asked)\s+[A-Z])")
_ROLE_PREFIX = re.compile(r"^[A-Z][A-Z ]{2,}:\s")
_FUTURE = re.compile(r"\b(?:will|won't|expects?|expected|expecting|forecasts?|forecasting|projected|projects|projections?|"
                     r"predicts?|predicted|predictions?|anticipates?|anticipated|outlook|next (?:week|month|quarter|year)|"
                     r"by 20\d\d|is set to|are set to|plans? to|poised to)\b", re.I)
# "For example" alone is not a cue: it introduces findings as often as worked cases. The worked case
# shows in what follows -- a supposed value, a "would", a bare "a ratio of 0.39".
_HYPOTHETICAL = re.compile(r"\b(?:suppose|supposing|imagine|hypothetical(?:ly)?|illustrat\w*|"
                           r"would (?:mean|indicate|imply|have|be|represent)|if [^,]+ would|consider a|a ratio of \d)\b", re.I)
_DEFINITION = re.compile(r"\b(?:is defined as|are defined as|defined as|refers? to|is a measure of|"
                         r"a measure of|measures? (?:what|the)|indicates the|is the (?:sum|share|ratio|number|portion|"
                         r"percentage|proportion) of|consists? of|comprises?|is calculated|are calculated|"
                         r"captures?|includes? the|includes|the (?:term|index|rate|ratio|measure) [^,]*?\bis\b)\b", re.I)
# A purpose statement says what something is for -- about-page and methodology prose -- not what was found.
_PURPOSE = re.compile(r"\b(?:the (?:mission|purpose|goal|aim|objective) of|aims? to|aimed at|is designed to|are designed to|"
                      r"seeks? to|is intended to|are intended to|is meant to|are meant to|strives? to|is dedicated to)\b", re.I)
# "According to X" attributes a finding, not a view; it is deliberately not a speech verb here.
_SPEECH = re.compile(r"\b(?:said|says|saying|told|tells|stated|wrote|writes|added|noted|argued|argues|remarked|"
                     r"claimed|claims|insisted|declared|explained|explains|warned|warns|urged|urges)\b", re.I)
_OPINION = re.compile(r"\b(?:in (?:the |our |my |his |her |their |its )(?:[\w'’-]+ ){0,3}(?:view|opinion|assessment|judgment)|we (?:believe|think|"
                      r"interpret|argue|view|see|read|consider|suspect)|i (?:believe|think|argue)|believes?|argues?|"
                      r"contends?|maintains?|insists?|suggests? that|the results suggest|this suggests|could help explain|"
                      r"may (?:reflect|explain|suggest)|likely reflects?|seems? to|appears? to|in (?:our|my) opinion|"
                      r"we would (?:argue|say)|opinion|viewpoint)\b", re.I)
_NAVIGATION = re.compile(r"\b(?:can be viewed|available (?:at|on|here)|read more|click here|learn more|for more information|"
                         r"sign up|subscribe|follow us|all rights reserved|copyright|terms of (?:use|service)|"
                         r"privacy policy|photo(?:graph)?:|credit:|on x:|on twitter|contact us|see also|related:|"
                         r"advertisement|share this|your instructions)\b", re.I)
_CONTINUATION = re.compile(r"^(?:instead|put differently|in other words|that is|rather|likewise|similarly|"
                           r"in short|simply put|to put it another way)\b", re.I)
# First-person stance and spoken fillers: someone advocating or talking, not a report of what was found.
_STANCE = re.compile(r"\b(?:we|i)\s+(?:need|should|must|have to|ought to|want|hope|urge|call on|think|believe|feel)\b|"
                     r"\byou know\b|\bi mean\b|\bsort of\b|\bkind of\b|\blet me\b|\blook,|\bokay,|\bok,", re.I)
# A page that is a transcript -- stage directions, speaker labels, Q/A turns -- is someone speaking throughout.
_TRANSCRIPT = re.compile(r"\((?:applause|laughter|pause|inaudible|crosstalk|cheers)\.?\)|^(?:Q|A|MR\.|MS\.|DR\.)\s*[.:]|"
                         r"^[A-Z][a-z]+(?: [A-Z][a-z]+){1,2}:\s", re.I | re.M)
_NUMBER = re.compile(r"\d")
# Publishers label opinion sections in the path; every sentence on such a page is someone's argument.
_OPINION_SECTION = re.compile(r"/(?:opinions?|op-eds?|columnists?|columns?|editorials?|commentary|perspectives?|"
                              r"letters(?:-to-the-editor)?|blogs?)(?:/|$)", re.I)
# A photo caption ends in its credit -- "Molly Riley/AP via The White House  hide caption", "(AP Photo/Evan Vucci)",
# "REUTERS/Kevin Lamarque", "(Image/PTI)", "Photo: Getty Images" -- and describes a picture, not the event; the
# tail is read on the whole passage, so every sentence the caption carries is context.
_CREDIT_TAIL = re.compile(r"(?:(?i:hide caption)|\b(?i:photo|photograph|image|illustration|credit|source)s?\s*[:/]\s*[^.!?]{2,80}|"
                          r"\b(?:AP|AFP|Reuters|REUTERS|EPA|PTI|ANI|Xinhua|Bloomberg|Shutterstock|Getty Images|Getty)\b"
                          r"(?:\s*/\s*[A-Z][^.!?]{1,60}|\s+via\s+[^.!?]{2,60})?)\s*\)?\s*$")
IMPERATIVES = frozenset("""
    click see read visit follow subscribe sign download learn contact ignore output cite view check watch listen
    call email join register buy order note please treat return use try get find go tell say write
    """.split())

ROLES = ("reported_observation", "definition", "hypothetical", "forecast", "attributed_opinion", "instruction", "caption",
         "speech_reprint", "unknown")

# Long enough that independent reporting almost never repeats a speaker's phrasing by chance,
# short enough to catch a quoted clause.
REPRINT_RUN = 10
# Captions and reprints differ by disfluencies ("over a 100,000", "out of of") and spellings
# ("Katherine"/"Catherine"), so a sentence most of whose word triples the speaker said is a reprint too.
REPRINT_TRIGRAMS, REPRINT_SHARE = 5, 0.6
_OPENS_QUOTE = re.compile(r"\s*[“\"]")
_SPEECH_WORD = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")
_SPEECH_TEXT_KINDS = frozenset("speech speeches address remarks debate interview testimony".split())


def _finite_verb(words: list[str]) -> bool:
    return any(word in AUXILIARIES or word in IRREGULAR_PAST or (len(word) > 4 and word.endswith("ed")) or
               (len(word) > 3 and word.endswith("s") and not word.endswith("ss")) for word in words)


def transcript_source(context: list[str]) -> bool:
    """Whether a source's context lines read as a transcript: stage directions, speaker labels, Q/A turns."""
    return any(_TRANSCRIPT.search(line) for line in context)


def opinion_source(url: str) -> bool:
    """Whether a source URL sits in a publisher's opinion, column, editorial or blog section."""
    return bool(_OPINION_SECTION.search(urlsplit(url or "").path))


def caption_passage(text: str) -> bool:
    """Whether a passage is a photo caption: it ends in an image credit, read on its last 120 characters."""
    return bool(_CREDIT_TAIL.search(text.strip()[-120:]))


def speech_text_source(url: str) -> bool:
    """Whether a URL path names a transcript, "remarks by" someone, or the text of a speech or debate.

    The text of a bill or a report is a primary source and is not matched: "text" counts only
    beside a kind of spoken event.
    """
    tokens = [token for token in re.split(r"[/_.\-]+", urlsplit(url or "").path.lower()) if token]
    if {"transcript", "transcripts"} & set(tokens):
        return True
    if any(first == "remarks" and second == "by" for first, second in zip(tokens, tokens[1:])):
        return True
    return bool({"text", "full"} & set(tokens)) and bool(_SPEECH_TEXT_KINDS & set(tokens))


def _speech_words(text: str) -> list[str]:
    parts, cursor = [], 0
    for link in inline_links(text):
        parts.extend((text[cursor:link["start"]], link["label"]))
        cursor = link["end"]
    plain = _LINK.sub(" ", "".join(parts) + text[cursor:])
    plain = plain.lower().replace("’", "'").replace("‘", "'").replace("%", " percent ")
    return _SPEECH_WORD.findall(plain)


def _grams(words: list[str], size: int) -> set[tuple[str, ...]]:
    return {tuple(words[start:start + size]) for start in range(len(words) - size + 1)}


def speech_runs(speech: Sequence[str]) -> tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]:
    """The speaker's runs of `REPRINT_RUN` words and word triples, read in the order given."""
    words = [word for text in speech for word in _speech_words(text)]
    return frozenset(_grams(words, REPRINT_RUN)), frozenset(_grams(words, 3))


def speech_match(text: str, runs: tuple[frozenset[tuple[str, ...]], frozenset[tuple[str, ...]]]) -> str | None:
    """How a source sentence repeats the speaker's words, ignoring case, punctuation and links.

    "run" for a whole run of `REPRINT_RUN` words; "near" for at least `REPRINT_TRIGRAMS` of the
    sentence's word triples making up `REPRINT_SHARE` of them, which a news sentence restating a
    short claim also reaches, so `annotate` needs a second matching sentence on the page.
    """
    long_runs, triples = runs
    words = _speech_words(text)
    if long_runs & _grams(words, REPRINT_RUN):
        return "run"
    own = _grams(words, 3)
    shared = len(own & triples)
    return "near" if own and shared >= REPRINT_TRIGRAMS and shared / len(own) >= REPRINT_SHARE else None


def type_sentence(text: str, *, previous: dict | None = None, spoken: bool = False, caption: bool = False,
                  reprint: bool = False) -> dict:
    """Roles for one sentence and the named cues that produced them.

    `previous` is the typing of the sentence before it in the same paragraph: a sentence that
    opens with a quotation mark, or with "Instead" or "Put differently", inherits an opinion the
    previous sentence carried, because the attribution was made once for the run. `spoken` marks
    a sentence from a transcript or an opinion-section page, where every sentence is someone's
    statement or argument. `caption` marks a sentence from a passage that ends in an image credit,
    where every sentence describes a picture. `reprint` marks a sentence that repeats the speaker's
    own words, which restates the claim rather than evidencing it.
    """
    # Linked words still carry role cues. Keep the original prose-only observation
    # checks: a linked headline or navigation label alone does not establish a finding.
    parts, prose, cursor = [], [], 0
    for link in inline_links(text):
        parts.extend((text[cursor:link["start"]], link["label"]))
        prose.extend((text[cursor:link["start"]], " "))
        cursor = link["end"]
    visible = _LINK.sub(" ", "".join(parts) + text[cursor:]).strip()
    cue_prose = _LINK.sub(" ", "".join(prose) + text[cursor:]).strip()
    stripped = _LINK.sub(" ", text).strip()
    words = [word.lower() for word in _WORD.findall(stripped)]
    roles: list[str] = []
    cues: list[str] = []

    def add(role: str, cue: str) -> None:
        cues.append(cue)
        if role not in roles:
            roles.append(role)

    if len(words) < 3:
        add("instruction", "link_or_fragment")
    if words and words[0] in IMPERATIVES and not (len(words) > 1 and words[1] in AUXILIARIES):
        add("instruction", "imperative_opener")
    if _NAVIGATION.search(text) or _NAVIGATION.search(visible) or _ROLE_PREFIX.match(text.strip()):
        add("instruction", "navigation_phrase")
    if (_QUOTE.search(text) and _SPEECH.search(text)) or _QUOTE_TAIL.search(text):
        add("attributed_opinion", "quoted_speech")
    if _OPINION.search(cue_prose) or _OPINION.search(visible):
        add("attributed_opinion", "opinion_marker")
    if _STANCE.search(cue_prose) or _STANCE.search(visible):
        add("attributed_opinion", "first_person_stance")
    if spoken:
        add("attributed_opinion", "spoken_or_opinion_source")
    if caption:
        add("caption", "image_credit_tail")
    if reprint:
        add("speech_reprint", "reprints_speech")
    if previous and "attributed_opinion" in previous.get("roles", ()):
        if cue_prose[:1] in "\"“" or visible[:1] in "\"“":
            add("attributed_opinion", "quote_continues")
        elif _CONTINUATION.match(cue_prose) or _CONTINUATION.match(visible):
            add("attributed_opinion", "opinion_continues")
    if _FUTURE.search(cue_prose) or _FUTURE.search(visible):
        add("forecast", "future_cue")
    if _HYPOTHETICAL.search(cue_prose) or _HYPOTHETICAL.search(visible):
        add("hypothetical", "example_cue")
    if (_DEFINITION.search(cue_prose) or _DEFINITION.search(visible)
            or re.match(r"^about\s+\S", cue_prose, re.I) or re.match(r"^about\s+\S", visible, re.I)):
        add("definition", "definition_cue")
    if _PURPOSE.search(cue_prose) or _PURPOSE.search(visible):
        add("definition", "purpose_cue")
    question = stripped.endswith("?") or (words and words[0] in QUESTION_OPENERS)
    if question:
        roles, cues = ["unknown"], ["question"]
    elif "instruction" in roles and len(roles) == 1:
        pass
    elif _finite_verb(words) and ("definition" not in roles or _NUMBER.search(stripped)):
        # A definition reports nothing unless a measurement sits in the same sentence.
        if "instruction" not in roles:
            roles.insert(0, "reported_observation")
            cues.append("finite_verb")
    elif not roles:
        roles, cues = ["unknown"], ["no_finite_verb"]
    return {"roles": roles, "cues": cues}


def annotate(reading: dict, *, speech: Sequence[str] = ()) -> list[dict]:
    """Role rows for every sentence of a reading packet, in the shape the gate validates.

    `speech` is the speaker's own text in spoken order -- the captions before the claim, then the
    claim -- against which reprints are found; without it no sentence is typed a reprint. A near
    match is a reprint only when it opens a quotation or another distinct sentence of the same
    source matches too.
    """
    spoken = {source["id"]: transcript_source(source.get("reading_context", [])) or opinion_source(source.get("url", ""))
              or speech_text_source(source.get("url", "")) for source in reading["sources"]}
    runs = speech_runs(speech)
    matches, matched = {}, {}
    for passage in reading["passages"]:
        for unit in passage["units"]:
            if kind := speech_match(unit["text"], runs):
                matches[unit["id"]] = kind
                matched.setdefault(passage["source_id"], set()).add(tuple(_speech_words(unit["text"])))
    rows = []
    for passage in reading["passages"]:
        previous, caption = None, caption_passage(passage["text"])
        page = len(matched.get(passage["source_id"], ())) > 1
        for unit in passage["units"]:
            kind = matches.get(unit["id"])
            near = kind == "near" and (page or bool(_OPENS_QUOTE.match(unit["text"])))
            previous = type_sentence(unit["text"], previous=previous, spoken=spoken.get(passage["source_id"], False),
                                     caption=caption, reprint=kind == "run" or near)
            rows.append({"unit_id": unit["id"], **previous})
    return rows
