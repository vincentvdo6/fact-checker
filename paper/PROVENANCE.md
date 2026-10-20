# Where each number in the paper comes from

Every figure in `main.tex` was read from a saved record or regenerated from saved artifacts on
2026-10-19; none was copied from notes. Records under `runs/` and `models/` are local and
gitignored: datasets, weights, indexes and per-claim outputs are not published. The three
aggregate files under `results/` and the truth files under `labels/` are tracked.

Regenerations that wrote nothing outside `runs/paper-2026-10-19/` are marked *regen*; their
scripts are saved beside the outputs in that directory.

## FEVER (Sections 3-4)

| Paper | Value | Source |
| --- | --- | --- |
| Corpus | 5,416,536 pages | `results/retrieval/metrics.json` |
| Strict recall@25, test / calibration | 0.811 / 0.801 | `runs/bm25-test/metrics.json`, `runs/bm25-calibration/metrics.json` (`cutoffs[3]`) |
| Document recall at 25 pages | 0.896 | `runs/bm25-test/metrics.json` (`doc_cutoffs[3]`) |
| Packed ceiling; gold read 1,045 of 1,350 | 0.774 | `runs/verdict-test/metrics.json` (`retrieved.ceiling`, `by_retrieval`) |
| Table 1 accuracy, macro F1, NEI recall | | `runs/verdict-test/metrics.json`; `results/verdict/metrics.json` |
| Claim-only right on 72.4% of verifiable | 0.7237 | `runs/verdict-test/metrics.json` (`claim_only.by_retrieval.without_gold`) |
| 252 of 341 NEI errors predicted supported | | `runs/verdict-test/metrics.json` (`retrieved.confusion[2]`) |
| Evidence lift +0.115 [0.096, 0.135], p 6.4e-30; gold +0.176 [0.157, 0.196] | | *regen* `abstention-denominators-test.json` (paired bootstrap seed 0, McNemar from `src/eval/stats.py`) |
| NEI inputs identical, 650/650 | | *regen* `scripts/checks.py` over `models/verdict/{retrieved,gold}/predictions_test.jsonl` |
| Out-of-fold NLL per calibrator | 0.673 / 0.728 / 0.728 / 0.823 | `models/verdict/retrieved/calibration.json` (`oof_nll`) |
| Table 2, Figure 1, bands, risk-coverage | | `runs/calibration-test/metrics.json`; figure `runs/calibration-test/calibration_retrieved.png` |
| Gold read / missed accuracy and mean confidence; NEI overconfidence 0.066 | | `runs/calibration-test/metrics.json` (`retrieved.subgroups`) |
| Sufficiency AUCs 0.748 / 0.586 / 0.748 | | `runs/sufficiency/checkpoint1.json` |
| Correctness AUCs 0.531 / 0.717; r 0.030; E-AURC 0.114 to 0.149; 65.4% of 298 | | *regen* `sufficiency-correctness-calibration.json` (installed sufficiency model, out-of-fold calibration, seed 42) |
| Table 3 | | *regen* `abstention-denominators-test.json` |
| Gate coverage 0.9355 to 0.758, accuracy 0.751 to 0.755 | | `runs/sufficiency-test/metrics.json` |
| 29,309 of 106,297 ungrounded rows | 0.2757 | `data/kaggle/fever-verdict-v2/dataset_manifest.json` (`regrounded.train`) |
| Table 4 | | `python -m scripts.eval_regrounded --split test --grounded retrieved_dropped` and `--grounded retrieved_grounded`, copied to `runs/regrounded-test-retrieved_{dropped,grounded}/metrics.json`; McNemar p from the printed output (0.725, 2.84e-06) |
| Naive Bayes claim-only lifts | +0.068 / +0.063 | `runs/claim-artifact/metrics.json` |
| AVeriTeC DeBERTa results (majority collapse at ~2,100 claims; a few points above majority) | | Kaggle run outputs reported in the project log; **not retained locally** |

## Live video (Section 5)

| Paper | Value | Source |
| --- | --- | --- |
| Detector F1 0.792 vs 0.310, 120 sentences | | `runs/checkworthy-sotu/metrics.json` |
| Pair judge held-out accuracy 0.886; counting band 69.1% at 0.0512 | | `models/pair_judge/v4/calibration_report.json` (`test_raw.accuracy`, `test_calibrated.counting_bands.strong`) |
| Probe: 35 claims, 1,060 pairs, 10 wrong of 18 | | `models/pair_judge/v4/news_measurement.json` |
| Retrain v5: 15 of 22 | | `runs/pair-judge-v5/result.json` |
| Retrain v6: 7 of 14; untouched 41 claims: 20 of 46 vs 24 of 45 | | `runs/pair-judge-v6/result.json` (`probe`, `count_b2`) |
| Directions, 153 claims: 47 of 47 | | *regen* `directions-insample-153.json` (`measure_directions` on `runs/clicks-probe` + `runs/clicks-count`, truth `labels/claim-truth-2026-09-13.json`) |
| Directions, 142 claims: 45 directional, 42 right, 35 established | | *regen* `directions-heldout-142.json`; matches `runs/direction-measurement/test-batch-report.json` and `models/pair_judge/v4/direction_measurement.json`; registration `runs/direction-measurement/test-batch-registration.json` (truth sha256 `0a20a7c9...`, verified against `labels/claim-truth-test-2026-09-15.json`) |
| For 57/57, against 32/35 | | the two files above (`readings[].direction`, `right`) |
| The three held-out errors | | counted sentences in `runs/clicks-test/click-2026-09-15T073015Z-test0000080.json`, `...073905Z-test0000470.json`, `...074534Z-test0000770.json` |
| Real captions: 7 checks, 17 claims, 10 with truth values, 3 scored, 1 right | | *regen* `directions-real-clicks-7.json` over the seven clicks recorded before 2026-09-16T02:00Z in `runs/clicks`, truth `labels/claim-truth-clicks-2026-09-20.json` (labeller: Vincent). Matches `runs/direction-measurement/report.json` |
| Reprinted speaker sentences (dailysignal.com 2026-06-13; PBS full text of the 2025 joint address) | | `runs/clicks/click-2026-09-16T000037Z-6Q_Xk66gsRU.json`, `runs/clicks/click-2026-09-16T000227Z-XkFKNkAEzQ8.json` |
| SOTU sufficiency transfer: 40 claims, 14 relevant, AUC 0.58 [0.39, 0.75], declined 30% vs 24.2% | | `runs/sufficiency-transfer/metrics.json`; labels `labels/relevance-sotu-2016.json` |
| Wilson intervals | | computed from the counts above (z = 1.96) |

## Corrections to earlier project notes

- **Abstention selectivity.** The project log and `scripts/eval_sufficiency.py` compare the
  gold-missed share of *all* declined claims (NEI included) against the gold-missed rate among
  *verifiable* claims. With matched denominators (Table 3), the confidence bands decline
  gold-missed claims at 1.5 times the rate of gold-read ones, so confidence-only abstention does
  not under-select ungrounded claims, as the log stated; it over-selects them weakly. The
  sufficiency gate's improvement stands (2.4 times).
- **NEI overconfidence.** NEI is the most overconfident subgroup after calibration (+0.066), not
  the only one: refuted claims are also slightly overconfident (+0.026).
- **`runs/direction-measurement/report.json`** no longer holds the in-sample 47/47 record; it was
  overwritten by the real-caption measurement. The in-sample record is regenerated above. That
  report lists 16 claims because it was written before the seventh click; the regeneration over the
  same seven clicks lists 17. The directional counts agree.

Found by the independent review of the first draft (all re-checked against the records before the
paper was changed):

- **In-sample truth values** (`labels/claim-truth-2026-09-13.json`, `labeller`): the probe
  batch's values were written after the clicks by an annotator who had seen which claims received
  a direction. Only the held-out batch was registered before it ran.
- **Composition rules.** Place and quantified-group qualifiers keep a count but block
  "established" (`src/verdict/composition.py` `_corroborating`, `_limits`); only a period outside
  the claim's demotes a count (`_evidence_row`).
- **Band seed.** The bands of Section 4.2 used cross-fitting seed 0
  (`models/verdict/retrieved/calibration.json`), the refitted bands seed 42 (`fit_sufficiency.py`
  default). The seed alone moves test coverage from 0.906 to 0.936.
- **Sufficiency transfer.** `scripts/measure_sufficiency_transfer.py` compares sufficiency alone on
  the SOTU sample (30%) with `FEVER_DECLINE = 0.242`, which is the full gate's decline rate, not
  sufficiency alone; its comment says otherwise. Sufficiency alone declines 382 of 2,000 FEVER
  test claims (19.1%): *regen* `scripts/review_checks.py`.
- **Real captions.** Five of the six directional real-caption readings, including the one scored
  right, rest on a reprint of the speech (PBS advance text twice, PBS full text twice,
  dailysignal.com quoting it); the sixth on an unrelated New Yorker sentence: *regen*
  `scripts/review_checks.py`.
- **Calibrator.** `VectorScaling` is `logits / T + b` (`src/calibration/scaling.py`), a restricted
  form of Guo et al.'s vector scaling.
- **Probe countries.** Nine, not seven (`labels/pairs-probe-2026-09-12.json`, `cases[].country`).
