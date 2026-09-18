# Fact Checker

Fact Checker connects YouTube captions to a local Python claim-checking pipeline,
combining DeBERTa classifiers, ONNX inference, and evidence retrieval. Its saved
FEVER benchmark pipeline uses a SQLite-backed Wikipedia store and a memory-mapped
BM25 index built in two streaming passes.

The browser extension lets a viewer request a check and inspect supporting source
quotations and links. It remains a development prototype; the offline results
below measure individual components, not reliability on political videos.

## Evaluation

These saved experiments evaluate three DeBERTa-v3-base verdict variants, confidence
calibration, and Wikipedia evidence retrieval. The test sample contains **2,000
held-out FEVER claims** from this project's partition of the released development
set. It is separate from calibration data and is not FEVER's official hidden test
benchmark. Results were extracted on September 18, 2026; experiments were not rerun.

| Measurement | Saved result | Scope and result record |
| --- | --- | --- |
| Wikipedia corpus and BM25 index | **5,416,536 pages; 292,551,543 postings** | [Recorded corpus/index metadata](results/retrieval/metrics.json) |
| Expected calibration error | **0.129 → 0.031** | Retrieved-evidence model; [vector scaling fitted on 2,000 separate calibration claims](results/calibration/metrics.json) |
| Multiclass Brier score | **0.441 → 0.374** | Same model and test sample; lower is better |
| Uncalibrated verdict accuracy | **Retrieved: 70.2%; claim-only: 58.7%; gold evidence: 87.8%** | [Three model variants on the same 2,000 claims](results/verdict/metrics.json) |
| Strict evidence recall@25 | **81.1%** | [1,350 verifiable claims](results/retrieval/metrics.json); excludes 650 insufficient-evidence claims |

The claim-only baseline measures how much can be predicted without evidence; the
gold-evidence variant provides an oracle reference for the retrieved-evidence
model. Strict recall requires a complete gold evidence group among the first 25
retrieved sentences. Calibration measures probability quality, not the truthfulness
of arbitrary video assessments.

The three small JSON files under [`results/`](results/) publish **aggregate extracts**
with exact values, denominators, source paths, and SHA-256 hashes. They exclude raw
claims, per-example predictions, datasets, and weights. The
[benchmark methods and provenance](docs/benchmark_results.md) explain the splits,
metric definitions, and reproduction limits.

## How it works

1. Play a YouTube video with available English captions and let a claim finish.
2. Open the **Fact checker** panel and select **Check what I just heard**.
3. The extension reads recent caption text and passes it to the local Python
   checker, which selects claims and searches for source material.
4. The panel displays source quotations, links, and an experimental assessment
   for the viewer to inspect.

Checks run when requested. The current workflow uses caption data; visible CC and
YouTube's transcript panel can stay closed. Caption availability and errors can
affect what is captured.

## Current status

This is a development prototype. Its reliability on real-world video claims has
not been established. Retrieval can return irrelevant or incomplete material,
and the checker can misinterpret a source. A quotation or citation alone does not
show that a claim is supported.

Some claims depend on definitions, geography, or dates that are missing from the
video. Others combine factual statements with opinion. The checker can leave
claims unresolved, and its assessments should be read alongside the cited
material. Improving those distinctions is ongoing work.

## Trying it locally

The current installation path is for **Windows with Chrome or Edge**. It requires
Python 3.12 or later, a repository-local Python environment with the
[dependencies](requirements.txt), and the local model and retrieval artifacts.
Model weights, datasets, and generated indexes are excluded from Git; cloning the
repository alone does not provide a ready-to-run checker.

The [extension setup guide](extension/youtube/SETUP.md) describes how to register
the local host and load the extension once those prerequisites are in place. It
also covers configuration and troubleshooting. Installation and artifact setup
are still developer-oriented.

## Local processing and web access

Model inference runs on the user's computer. Web research is enabled by default
and makes external requests: it sends claim-related search queries to Google News
and fetches public source pages. Queries may include nearby caption context and
country or date information. Source availability and external service limits can
affect a check.

The setup guide explains how to disable network research. Local-only checks use
an older Wikipedia corpus, which limits their usefulness for recent claims.

## Development direction

Current priorities are to:

- Retrieve sources that address the actual claim and its context.
- Distinguish relevant evidence from material that merely shares its wording.
- Handle missing context, conflicting sources, and uncertainty more clearly.
- Evaluate the complete workflow across a broader range of videos and topics.

## Repository guide

| Location | Contents |
| --- | --- |
| [extension/youtube](extension/youtube/) | Browser extension and local setup instructions |
| [src/pipeline](src/pipeline/) | Claim processing and evidence retrieval |
| [src/verdict](src/verdict/) | Models and rules for interpreting evidence |
| [scripts](scripts/) | Setup, data preparation, and evaluation tools |
| [results](results/) | Three small aggregate metric records with source-file hashes |
| [docs/benchmark_results.md](docs/benchmark_results.md) | Saved offline results, metric definitions, and artifact provenance |
| [tests](tests/) | Pipeline and extension tests |

For a reproducible problem, include the video URL and timestamp, the exact claim,
and what the checker displayed. Reports of irrelevant sources, missed context,
and incorrect interpretations help guide development.
