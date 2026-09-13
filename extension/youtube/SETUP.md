# Try the YouTube extension locally

The extension adds a **Fact checker** tab where YouTube shows its transcript panel.
The checker starts closed. Open the tab when needed and close it with × or Escape.
Chrome or Edge starts
the Python model host when needed; no local website or audio listener is required.

1. From the repository, register the local connection:

   ```powershell
   .venv\Scripts\python.exe -m scripts.install_youtube_extension
   ```

2. Open `chrome://extensions` or `edge://extensions`, enable **Developer mode**,
   choose **Load unpacked**, and select this `extension/youtube` directory.
3. Pin **Fact checker for YouTube** in the browser's extensions menu.
4. Open or refresh a YouTube video. Play the claim and let the sentence finish,
   then open **Fact checker** and click **Check what I just heard**.
   The extension icon opens the checker and starts a check directly.

Visible **CC can stay off** and the transcript panel can stay closed. The checker
requests available English subtitle data in the background when you check a claim.

The first check loads the local models. Existing detector/verdict exports and the
Wikipedia index must be installed in this repository. Reload the extension after
editing its files, then refresh YouTube. Registration is per Windows user and tied
to this checkout's path; rerun the installer after moving the repository.

## Experimental news research

News research is enabled by default. It uses Google News RSS to discover public
articles, then reads those pages locally. No search account, provider key, hosted
backend or research browser tab is required.

Each check-worthy claim sends a bounded search query to Google News. The query contains
the claim, relevant preceding caption context and any supplied country or date
metadata. The full transcript, video URL and model output are not sent to Google News.
The checker fetches discovered public HTTPS pages directly without browser cookies.
Search snippets never become evidence. The same discovery rules apply across topics;
there are no publisher routes or Firecrawl dependencies.

This is an experiment with an externally controlled RSS endpoint, not a guaranteed
search service. Coverage is unmeasured and articles
may be inaccessible. A recent article can describe an older event. Feed timestamps
are displayed separately from publication dates found on the article, and neither
establishes the date of the event or claim. Ambiguous publication dates stay unknown.
Search uses the English US news edition, which does not establish the claim's country.
When a speech or video publication boundary is supplied, discovery requests articles
through that day. Each article's own date is checked separately because search filters
can return later material. Google article links are resolved in the background before
reading the publisher page; a failed resolver produces an error, never landing-page
text as evidence. Links resolving to the same publisher URL count once.

Research shares a 45-second budget per click. When the first query produces no
eligible passage, one fallback searches compact words from the original assertion
with the same supplied country and year. It adds no synonyms or publisher routes.
For assertions containing digits or common number words, the fallback removes only
preceding context, preserving the entire assertion's units and qualifications.
The original claim stays unchanged. Both attempts share a limit of five distinct
pages per assertion for two candidate passages. Caption context can help rank passages,
but cannot make a passage eligible without matching the assertion itself. Visibly
truncated teasers are excluded. HTML pages are supported; PDFs and pages
requiring browser execution may be unavailable. Research candidates give the
verifier **No verdict**: matching words do not establish truth, semantic relevance or
applicability. When the sentence-level reading below reaches a direction, the row's badge
shows it -- "Contradicted, not established (reading)", "Partially contradicted (reading)",
"Supported (reading)" -- dashed, marked as a reading, because it is composed by rules from
quoted sentences and is a draft; "not established" means every count is qualified (one
source, an undated source, a narrower scope, or an undeclared country) and "partially" is
said only when one part of a contrast is established; an unresolved reading keeps the plain
**No verdict**. Old local Wikipedia matches do not
suppress a fresh news search. Errors and unknown dates stay visible.

To disable network research, set `FACT_CHECKER_WEB_SEARCH` to `off` in the Windows
user environment, then fully exit and restart the browser. Remove that variable
(or set it to `news`) to restore the default. Local-only checks use the June 2017
Wikipedia corpus, which cannot establish recent events.

## Claims are about

The checker never infers which country a claim is about: neither the news edition it
searches nor a source's own scope can establish that, and a video rarely says. Without a
country nothing can be established -- every count is reported as qualified. The panel's
**Claims are about** field lets you declare it for the video you are watching (a short
place name; suggestions are offered). It is sent with each check, used in the searches,
and shown on every card as supplied by you. Leave it unset and the card says so. The
setting is kept in the extension's local storage (the `storage` permission) so it survives
reloads; it goes nowhere but the local checker. **Spoken on** works the same way for the date
the claims were made, when you know it: it becomes the search boundary (a debate uploaded
weeks after it was recorded is then not checked against pages published in between), can
never be later than the video's publication date, and is shown on every card as supplied by
you. It is not stored.

## Sentence-level reading

Beside the research, each researched claim gets a **Sentence-level reading (draft)**.
Every source sentence is typed by named rules first -- attributed opinion, forecasts,
worked examples, definitions and navigation text are read as context and can never
count -- and a local model (a 739 MB DeBERTa-v3-base pair judge under `models/pair_judge`)
then answers one narrow question per remaining sentence: does it state the assertion, state its negation, bear on it, or neither. A
verdict is composed from those answers by fixed rules: only exact quoted sentences
count; a qualifier in the sentence, an unknown claim country or an undated page can
qualify but never establish; an assertion is established by one independent source
with a confirmed publication date, which the card names ("one source: rbc.com
(published 2025-07-09)"); a contrast is established only when both halves are. The
reliability line above the counted sentences is the measurement that matches what you
see: on 153 unseen claims (2026-09-14) the reading pointed a direction 47 times and
pointed the right way 47 times -- a number decided on those same clicks, so the next
batch of recorded clicks is its test. The page shows the
counted sentences and the strongest bearing ones verbatim with any definition from
the same paragraph; nothing is paraphrased and no figure appears that is not in a
quoted sentence. When nothing counts but a bearing sentence gives a figure the claim
states -- "around 25%" against a published 24.3% -- the card says so as a number check,
by the sentence's own publisher, and says that the sentence does not state the claim.
Pages found by caption-concept research (phrases from the captions,
such as the name of a measure the speaker cites) are shown as context and marked;
they never resolve the claim. The reading is a draft over a model whose reading of a
single news sentence, when it counts one, was wrong about half the time on unseen claims
(measured 2026-09-12 on labels an assistant wrote, not a person; the card prints the
record beside every count), which is why one counted sentence never establishes anything
and why every counted sentence is quoted with its direction for you to read yourself.

Set `FACT_CHECKER_DECOMPOSED` to `off` to remove the reading, or
`FACT_CHECKER_CONCEPTS` to `off` to skip caption-concept research. Each needs a local
artifact -- the pair judge under `models/pair_judge` and the Wikipedia store for
phrase specificity -- and a missing one is reported on the card, not raised.

## Collecting clicks for labelling

The pair judge's error rate on news pages is unmeasured, and the only way to measure it
is on the sentences real clicks retrieve, read by a person. Set `FACT_CHECKER_RECORD_CLICKS`
to a directory (the repository's `runs/clicks`, by absolute path, is gitignored) in the
Windows user environment, fully exit and restart the browser, and every check writes
`click-<time>-<video>.json` there -- the caption payload and the complete result, nothing
else, on this machine only. A failed write is printed to the host's stderr and never fails
the check. Remove the variable to stop recording.

Aim for ten to twenty clicks on different videos and topics -- economics, crime, health,
foreign affairs, sport, whatever you watch -- with **Claims are about** set, since that
changes what is retrieved. Clicks on one video seconds apart are rate-limited by Google
News (the card records the HTTP 403), so spread them out. Then, in the repository:

    python -m scripts.harvest_clicks --name clicks-2026-09-13
    python -m scripts.label_pairs
    python -m scripts.eval_pair_judge --labels labels/pairs-clicks-2026-09-13.json --judge pair --probe-negation --probe-hedges
    python -m scripts.measure_role_rules --labels labels/roles-clicks-2026-09-13.json

The harvest rebuilds exactly what the chain read for every researched claim and writes
`labels/pairs-<name>.json` (one item per assertion and eligible sentence, the judge's
own answer kept beside it as `judged`, not shown while labelling) and
`labels/roles-<name>.json` (every sentence, for the rule typer). The labelling page
serves every file under `labels/` on localhost with keyboard labelling -- 1 to 4 for the
relation, Enter to save, `q` for a scope qualifier chosen from the sentence, `n` for a
note -- and writes only the label fields back; label from the sentence and the rubric,
not from `judged`. The evaluator then reports the judge's accuracy, its counting error
at the shipped band, and a sweep of thresholds over its readings, on your labels; add
`--model-dir models/pair_judge/v2` to score another judge on the same file. Around sixty
pairs a claim and a few minutes a claim to label.

## Current scope

- Regular YouTube watch pages use the transcript area beside the video, moving
  below it with YouTube's layout in narrow windows and theater mode. The extension
  only inserts its own controls and panel. Styles stay inside its shadow root;
  opening the checker does not hide YouTube panels or scroll the page.
  Shorts retain an icon-opened floating panel; reads only YouTube pages.
- Checks up to three check-worthy claims from completed captions in the preceding
  30 seconds, counting back from the playback position; sentences the detector skips
  are shown as skipped and do not use a slot. Short corrective contrasts such as
  "We don't have a labor shortage. We have a good job shortage." stay together with
  their original caption wording.
  The panel shows the captured excerpt and its interval, not exact sentence timing.
- Earlier captions with shared topic terms guide a content-focused search. A lexical
  phrase filter removes obvious distractors before verification; it does not prove
  semantic relevance or that evidence matches the speech's place and time.
  Corrective contrasts receive **No verdict** from the verifier with retrieved
  background; the sentence-level reading reads each half against every eligible
  sentence and reports them apart.
- English captions, plus YouTube-provided English translations when accessible.
  Automatic and translated captions may contain errors.
- Caption tracks, browser text tracks, loaded transcript data, background transcript
  requests, and already-observed on-screen captions are attempted. The checker never
  enables CC, opens the transcript, or expands the video description. No universal video-coverage
  claim has been measured. Videos without accessible captions cannot be checked.
- Captions are the claim input. Default research displays article passages for review.
  Local-only mode uses Wikipedia from June 2017. YouTube verdict accuracy is unmeasured.
- Only recent visible caption text is buffered in memory. No microphone or system
  audio is captured, and no transcript is saved by the extension or native host.
- One local inference request runs at a time. Seeking or changing videos clears
  the panel and prevents stale results from appearing under the new position.

The unpacked extension's stable ID is `abjbodonhgkdgkhalhdpjfhmlfcpdegf`.
The manifest key is public identification material, not a secret.

## Development regression

The primary manual case is [this video at 52:53](https://www.youtube.com/watch?v=2S-WJN3L5eo&t=3173s):
"We don't have a labor shortage, we have a good job shortage."
The captured English-caption fixture is `tests/youtube_labor_case.json`; caption
punctuation separates the two clauses with a period. Check that they stay in one
card and that the evidence discusses the labor market rather than song titles. With
the sentence-level reading on and nothing declared, expect the speaker's "around 25% of the
population" sentence checked beside the contrast with its figure confirmed against LISEP's
24.3% by prnewswire.com and finance.yahoo.com, **Not established** for both halves, the
Wisconsin labor-shortage report and LISEP's functional-unemployment figures with
their definitions quoted as bearing, the LISEP chair's quotes and a congressman's
remarks read as opinion, and no counted sentence. With **Claims are about** set to United
States, the outcome depends on what Google News returns that hour: when RBC's, SHRM's or
the Chamber's labor-shortage pages are among the results, expect **Partially
contradicted (reading)** on the badge and, on the card, "Partially contradicted" -- the
denial half "contradicted (one source: rbc.com (published 2025-07-09))" or the SHRM
equivalent, each counted sentence marked "[read as denying it]" with its publisher and
date, the Chamber's "Several states ..." sentence carrying its scope, any page the judge
read both ways shown with "Shown, not counted: ... a misreading, not a dispute", and the
reliability line ("When this reading pointed a direction on 153 unseen claims ... it
pointed the right way 47 of 47 times") above them; the good-job half stays not
established with LISEP quoted and the speaker's "around 25%" confirmed as a number
check. That is what the sources say: mainstream labor-market reporting states a labor
shortage in 2025, the speaker's figure is LISEP's published figure, and the
characterisation built on it is stated nowhere. When none of those pages is returned,
both halves stay **Not established**. Either is the reading of the sources actually found. A
three-claim click takes about 40 seconds, most of it research.
`runs/live-click/click.py` runs
this click through the same host call the extension makes and records the timing.
This is a development case, not an independent accuracy benchmark or a truth label.
