# Fact Checker

Fact Checker is an experimental project exploring how to help people investigate
claims while watching online video. Development currently focuses on a YouTube
browser extension that connects to a checker running on the user's computer.

The goal is to bring source material and context closer to the moment a claim is
heard. A viewer can request a check from the video page and inspect the sources
behind the result. The project explores claim detection, evidence retrieval, and
how to communicate uncertainty when the available material cannot settle a claim.

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
| [tests](tests/) | Pipeline and extension tests |

For a reproducible problem, include the video URL and timestamp, the exact claim,
and what the checker displayed. Reports of irrelevant sources, missed context,
and incorrect interpretations help guide development.
