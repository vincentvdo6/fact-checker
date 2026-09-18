# Saved offline benchmark results

This record makes the project's earlier component evaluations inspectable. It
summarizes saved Wikipedia retrieval and verdict-model calibration artifacts;
it does not evaluate the current YouTube extension as a complete system.

The figures were extracted on September 18, 2026. No training, inference,
retrieval, or calibration experiment was rerun for this documentation update.
Exact values, selected configuration fields, and source-file SHA-256 hashes are
in [saved_component_metrics.json](benchmarks/saved_component_metrics.json).

## Results

| Measurement | Saved value | Scope |
| --- | ---: | --- |
| Wikipedia pages | 5,416,536 | Corpus metadata |
| BM25 term-document postings | 292,551,543 | Index metadata |
| Expected calibration error, before → after | 0.128668 → 0.031385 | Retrieved-evidence verdict model; 2,000 test claims |
| Multiclass Brier score, before → after | 0.440891 → 0.374174 | Same model and 2,000 test claims |
| Strict evidence recall at 25 sentences | 0.811111 | 1,350 verifiable claims from a 2,000-claim test sample |

Lower calibration error and Brier score are better. The changes compare the
uncalibrated model probabilities with vector-scaled probabilities on the same
test sample. They describe probability quality on this benchmark, not the
truthfulness of arbitrary video assessments.

## Evaluation design

FEVER's released development set is partitioned into calibration and test groups
by hashed claim keys, stratified by label. Repeated claim keys remain in the same
partition. These results use the project's held-out test sample, not FEVER's
official hidden test benchmark. The split construction is implemented in
[src/data/splits.py](../src/data/splits.py).

The saved calibrator record identifies vector scaling fitted on 2,000 separate
calibration claims. Calibrator selection uses five-fold out-of-fold negative log
likelihood; the selected method is then refitted on the calibration sample and
applied to test predictions. The saved calibration and test prediction files each
contain 2,000 rows and share no claim IDs, verified by a read-only check during
this documentation update. That check is narrower than a full training-leakage
audit.

Expected calibration error uses 15 equal-width confidence bins, weighted by
their sample counts. Multiclass Brier score averages the sum of squared errors
across the three class probabilities. Definitions are in
[src/eval/calibration.py](../src/eval/calibration.py).

Strict evidence recall requires the first 25 retrieved sentence references to
contain every sentence in at least one complete gold evidence group. Its
denominator excludes the 650 claims labeled as having insufficient evidence;
recall is evaluated on the remaining 1,350 verifiable claims. Finding a relevant
page or a single sentence from an incomplete group does not count as a strict
hit. The definition is in [src/eval/retrieval.py](../src/eval/retrieval.py).

## Implementation entry points

| Stage | Source |
| --- | --- |
| Build the Wikipedia store and memory-mapped index | [build_wiki.py](../scripts/build_wiki.py), [build_index.py](../scripts/build_index.py), [bm25.py](../src/retrieval/bm25.py) |
| Retrieve and score sentence evidence | [eval_retrieval.py](../scripts/eval_retrieval.py) |
| Fit and freeze calibration | [calibrate.py](../scripts/calibrate.py) |
| Apply frozen calibration and score test predictions | [eval_calibration.py](../scripts/eval_calibration.py) |

A fresh clone does not include the datasets, indexes, model weights, or saved
prediction files required to rerun those stages. The public JSON contains
aggregate results and hashes only. Reproducing a result requires acquiring the
relevant data and model artifacts and recording the exact code and environment
used; these documentation changes do not claim that a fresh checkout reproduces
the reported numbers.

## Provenance and limits

The JSON records each original artifact's repository-relative path, byte count,
and SHA-256 hash. Those paths identify local, ignored experiment files; their
contents are not published here. The inspected source-code commit is recorded
separately from run provenance and is not presented as the commit that produced
every result.

The retrieval folder retains both original and rescored configurations. The
rescored configuration records a dirty working tree, so those files alone cannot
bind the saved metrics to an exactly reproducible clean commit. Hashes identify
the files from which this summary was extracted; they do not independently
validate the experiments.

The current extension researches caption claims using public web sources, while
these benchmarks concern older Wikipedia evidence and an earlier verdict-model
pipeline. Accuracy on political videos, recent news, and the complete interactive
workflow remains unestablished. The [README](../README.md) describes the current
workflow, setup requirements, and external requests.
