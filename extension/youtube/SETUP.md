# Try the YouTube extension locally

The extension adds a **Fact checker** tab where YouTube shows its transcript panel.
The checker starts closed. Open the tab when needed and close it with × or Escape.
Chrome or Edge starts
the Python model host when needed; no local website or audio listener is required.

Before registering, check the local prerequisites from this checkout:

```powershell
.venv\Scripts\python.exe -m scripts.check_youtube_setup
```

Errors identify missing files or invalid settings to fix before checking a video.
Warnings identify optional features that are unavailable or disabled. Use `--json`
for a structured report; errors return a nonzero exit code. This check does not
download anything, load models, change browser registration, or contact websites.
It checks the selected news/offline configuration and local file inventory;
passing does not establish model integrity, browser connectivity or accuracy.

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
If the player has only a partial caption response, it tries video-bound page metadata
and refreshes the transcript request metadata without opening YouTube's panels.

The first check loads the local models. News checks need the claim detector;
sentence-level reading also needs the pair judge and its calibration, and uses the optional
support-confirming NLI model when it is installed (the first check then takes about 20 seconds
longer while its 1.7 GB graph loads; later checks ask it about a pair or two per claim). The
Wikipedia corpus enables optional caption-concept research. Offline checks
additionally need the Wikipedia index and the offline verdict artifacts.
The setup check reports missing prerequisites for the selected configuration.
Reload the extension after
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
applicability. The card's badge comes from the sentence-level reading below --
"Supported", "Contradicted", "Supported, not established", "Partially contradicted",
"Contested" -- dashed, with the card stating that it is a draft composed by rules from
quoted sentences; "not established" in a badge means every count is qualified (an undated
source, a narrower scope, or an undeclared country) and "partially" is said only when one
part of a contrast is established. A reading that finds no direction shows **Not
established**; a check with no reading shows **No verdict**. Old local Wikipedia matches do not
suppress a fresh news search. Errors and unknown dates stay visible.

To disable network research, set `FACT_CHECKER_WEB_SEARCH` to `off` in the Windows
user environment, then fully exit and restart the browser. Remove that variable
(or set it to `news`) to restore the default. Local-only checks use the June 2017
Wikipedia corpus, which cannot establish recent events.

## Automatic context and corrections

Each check looks for a single explicit country in its recent captions, falling back to
the video title. "America", "American" and "Americans" count as the United States; Latin,
South, North and Central America do not. Conflicting countries in the excerpt stay unknown. This is a detected
context hint, shown with its origin; it can be wrong when a place is only mentioned in
passing. The viewer's location, channel location and search edition never supply it.

The publication date comes from metadata belonging to the current video. The recording
date is separate: it requires an explicitly labelled date in the title or description,
or a completed broadcast wholly within one UTC day. Unknown recording dates stay unknown;
an upload date alone does not say when a speech happened. These dates bound source searches.

**Context** contains optional **Claims are about** and **Spoken on** corrections. Leave
them blank for automatic detection. Corrections are labelled as supplied by you and saved
locally for that video only. Old global country settings are ignored. Country and date
context reach the local checker and can be included in its public search queries.

## Sentence-level reading

Each checked claim gets a card, newest first: a badge (Supported, Contradicted, Not
established, or a qualified form of these), the claim, one line per assertion, and the
sentences that were counted, each quoted with its source. Related sentences that were not
counted, and how the claim was checked (limits, withheld sentences, searches, failed pages,
every source read), are folded under the card; sentences the detector passed over are
listed once at the end. The reading is a draft composed by rules, and the card says so.

Every source sentence is typed by named rules first -- attributed opinion, forecasts,
worked examples, definitions and navigation text are read as context and can never
count. A sentence that repeats ten or more consecutive words of the captions is the
speaker's own words reprinted -- a transcript of the speech, or an article quoting it --
and never counts as evidence for them. So is one sharing most of its word triples with the
captions (caption disfluencies and spellings differ from a printed text) when it opens a
quotation or another sentence on the same page matches too; a lone unquoted near match is
how a news report restates a short claim, and stays evidence. A page whose address names a transcript or the text of a
speech or debate is read as speech throughout.

Some caption sentences are listed under "not checked" with their reason instead of being
researched: one that names nothing but pronouns, numbers and times ("Last month it was
32,000"), because its subject was said earlier and a search for its words finds any page
with that number; one whose subject is "he", "she" or "they" with no name after it ("They
don't have equity in their home"), for the same reason; one recounting what someone said
("He said they're just having headaches"), because quotations never count; and, in
punctuated captions, the last sentence when it has not finished at the click.

A local model (a 739 MB DeBERTa-v3-base pair judge under `models/pair_judge`)
then answers one narrow question per remaining sentence: does it state the assertion, state its negation, bear on it, or neither. A
verdict is composed from those answers by fixed rules: only exact quoted sentences
count; a qualifier in the sentence, an unknown claim country or an undated page can
qualify but never establish; an assertion is established by one independent source
with a confirmed publication date, which the card names ("one source: rbc.com
(published 2025-07-09)"); a contrast is established only when both halves are, and so is
a claim made of two clauses joined by "but".

A second, larger model confirms the pair judge's near misses. A sentence the pair judge reads
as stating the assertion, but below its counting band, counts when an NLI model
(DeBERTa-v3-large trained on MNLI, FEVER, ANLI, LingNLI and WANLI; 1.7 GB under
`models/nli_confirm`) agrees with at least 0.95 entailment. It only adds supports both models
agree on: it never vetoes a count, never proposes one from a sentence the pair judge read as
context, and never touches contradictions. Those variants were tried and added wrong readings.
Install it with `.venv\Scripts\python.exe -m scripts.install_nli_confirm` (the pinned revision,
hash-checked); without it, checks run as before and say so. `FACT_CHECKER_NLI_CONFIRM=off`
disables it.

The reliability line above the counted sentences is the measurement that matches what you
see: on 105 claims written blind with their truth and clicked on 2026-10-20, the reading
pointed a direction 30 times and pointed the right way 29 times (the pair judge alone: 21 of
22); on held-out real captions it pointed the right way 5 of 6 times. Earlier misses were
mostly single-source contradictions of true claims, so read the quoted sentence before
trusting a "contradicted". The page shows the
counted sentences and the strongest bearing ones verbatim with any definition from
the same paragraph; nothing is paraphrased and no figure appears that is not in a
quoted sentence. Similar numbers are not labelled as confirming a claim's figure:
arithmetic alone cannot establish that they measure the same population, outcome or period.
Pages found by caption-concept research (phrases from the captions,
such as the name of a measure the speaker cites) are shown as context and marked;
they never resolve the claim. A context page must also share a nonnumeric claim term
in its selected reading window, including headings and scope notes; link targets do not
count. This is a conservative lexical filter, not a semantic relevance check: synonyms
can be missed and incidental shared words can still pass. The reading is a draft over a model whose reading of a
single news sentence, when it counts one, was wrong about half the time on unseen claims
(measured 2026-09-12 on labels an assistant wrote, not a person; the card prints the
record beside every count), which is why a single counted sentence establishes only from
an independent, dated source, and why every counted sentence is quoted with its direction
for you to read yourself.

Set `FACT_CHECKER_CONTEXT_REVIEW` to `off` to skip the second opinion on contextual
sentences. When the local v5 pair judge is installed, it reviews only sentences the
primary judge calls contextual and withholds ones it considers unrelated. It cannot
change support or contradiction decisions or turn context into proof. Disagreements
remain in the saved check's `context_review` fields. This can also withhold useful
context and adds model loading and inference time. It uses another local model
(about 739 MB on disk), without downloading one automatically; a missing model is
reported in the check's notes. A model runtime error fails the check visibly.

An optional local Qwen3 ranker orders the remaining contextual quotations after that
review. It keeps the original assertion and source text, preserves every candidate,
and cannot change the verdict or counted evidence. Its scores are ordering signals,
not truth probabilities. It can still rank unrelated material highly; the feature
does not establish that every displayed source is relevant.

Install the validated full-precision bundle with:

```powershell
.venv\Scripts\python.exe -m scripts.install_context_ranker --bundle path\to\validated-bundle
```

The bundle is an experimental artifact, not downloaded automatically. It contains
the pinned graph, external tensors, contract, manifest and tokenizer; the installer
checks their hashes and refuses to replace an existing installation. Once present in
`models/context_ranker`, ordering is enabled by default. Set
`FACT_CHECKER_CONTEXT_ORDERING=off` to retain the previous order. Missing artifacts
or an input that cannot fit in full retain that order, with a note in the saved
check. Other runtime errors remain errors. Scores and actual runtime providers are
recorded separately from the verdict judge's confidence.

Windows requirements use ONNX Runtime DirectML for this ranker, with CPU fallback;
the existing judges still use CPU. For an existing Windows environment, uninstall
the old `onnxruntime` package before installing `requirements.txt`: it shares the
same Python module with `onnxruntime-directml`. The ranker adds model loading,
memory use and inference time. Reload the extension to start a fresh model host.

Set `FACT_CHECKER_DECOMPOSED` to `off` to remove the reading, or
`FACT_CHECKER_CONCEPTS` to `off` to skip caption-concept research. Each needs a local
artifact -- the pair judge under `models/pair_judge` and the Wikipedia store for
phrase specificity -- and a missing one is reported on the card, not raised.

## Collecting clicks for labelling

Record checks explicitly through the native host installer:

    .venv\Scripts\python.exe -m scripts.install_youtube_extension --record

Reload the extension to use the updated launcher. Checks are saved in the ignored
`runs/clicks` directory as `click-<time>-<video>.json`, containing the caption payload
and complete result. Use `--record DIR` for a different location. Recording failures
are logged to the host's stderr without failing the check. Run the installer with
`--no-record` (or no recording flag), then reload the extension, to stop recording.
The launcher clears inherited recording settings so recording stays opt-in.

Every earlier measurement was made on claims written as single clean sentences; what
viewers click are captions. Aim for ten to twenty clicks on different videos and topics
-- economics, crime, health, foreign affairs, sport, whatever you watch -- and inspect
the detected context, correcting it when needed, since it changes what is retrieved.
Clicks on one video seconds apart are rate-limited by Google News (the card records the
HTTP 403), so spread them out. Two measurements follow, and the first needs nothing but
your own knowledge:

    python -m scripts.claim_truth_template --clicks runs/clicks --out labels/claim-truth-clicks-2026-09-20.json
    # fill in "true": true / false for each claim without reading its card, then
    python -m scripts.measure_directions --clicks runs/clicks --truth labels/claim-truth-clicks-2026-09-20.json

reports how often the reading's direction was right on your clicks; with `--record` it
becomes the record the card's reliability line quotes. The second is the judge's
per-sentence record, a few minutes of labelling per claim:

    python -m scripts.harvest_clicks --name clicks-2026-09-20
    python -m scripts.label_pairs
    python -m scripts.eval_pair_judge --labels labels/pairs-clicks-2026-09-20.json --judge pair --probe-negation --probe-hedges
    python -m scripts.measure_role_rules --labels labels/roles-clicks-2026-09-20.json

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
card and that the evidence discusses the labor market rather than song titles.
Do not require a predetermined verdict: sources and their availability change.
The neighboring "around 25%" claim must not be labelled confirmed merely because a
source contains 24% or 24.3%. The measure, population and period may differ.
Context-only pages must connect to the original claim, not merely repeat an incidental
caption phrase such as "people together". Neither that lexical connection nor a quoted
number establishes the claim. Check the quoted evidence and each assertion's limitations.
`runs/live-click/click.py` runs
this click through the same host call the extension makes and records the timing.
This is a development case, not an independent accuracy benchmark or a truth label.
