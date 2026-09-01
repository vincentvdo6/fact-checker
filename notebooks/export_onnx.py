# %% [markdown]
# # Export the verdict model to ONNX
#
# The trained weights live on this machine only because Kaggle produced them. Locally there is no
# torch -- that was Phase 02's decision after ROCm proved a dead end -- so the model cannot run
# where the demo needs it. This exports it once, to a runtime that has no torch dependency.
#
# The model arrives as another kernel's output rather than a dataset: `kernel_sources` mounts
# `fever-verdict-retrieved` at `/kaggle/input/`, which avoids a 738 MB round trip through a dataset
# upload.
#
# **What matters downstream is not that the export succeeds but that it is faithful.** Every
# calibration artifact -- temperature, three per-class biases, band thresholds, the sufficiency
# gate -- was fitted on the logits this model produced. If ONNX drifts, the bands silently stop
# meaning what they measured, and a sidebar that says "strong: right nine times in ten" starts
# lying. This notebook does a first parity check in place; the authoritative one runs locally
# against `predictions_test.jsonl`.

# %%
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

WORK = Path("/kaggle/working")
SOURCE = next(Path("/kaggle/input").glob("**/model_v1"), None)
assert SOURCE is not None, "attach the fever-verdict-retrieved kernel output"
print(f"model from {SOURCE}")
print(f"torch {torch.__version__}")

CONTRACT = json.loads((SOURCE.parent / "contract.json").read_text())
MAX_LENGTH = CONTRACT["max_length"]
print(f"contract: {CONTRACT['labels']}  max_length {MAX_LENGTH}  {CONTRACT['template_id']}")

# %%
tokenizer = AutoTokenizer.from_pretrained(str(SOURCE))
model = AutoModelForSequenceClassification.from_pretrained(str(SOURCE)).float().eval()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# %%
# Dynamic on batch and sequence. A fixed 512 would force every claim to pad to the maximum, which
# costs roughly 4x on the short inputs that dominate a transcript.
DUMMY = tokenizer(
    "a claim", "some evidence", return_tensors="pt", truncation=True, max_length=MAX_LENGTH
)
TARGET = WORK / "verdict.onnx"

# torch 2.10's default exporter imports onnxscript, which Kaggle's image does not carry. It is
# installed here so the newer path stays available as a fallback, even though the export below
# asks for the older one.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxscript", "onnxruntime"], check=False)

# dynamo=False selects the TorchScript exporter. torch 2.10 defaults to dynamo=True, which
# imports onnxscript and fails on Kaggle's image with ModuleNotFoundError; the package is
# installed above so the newer path stays available, but DeBERTa-v3's disentangled attention is
# far better travelled through the older exporter and this only has to run once.
torch.onnx.export(
    model,
    (DUMMY["input_ids"], DUMMY["attention_mask"]),
    str(TARGET),
    dynamo=False,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    },
    opset_version=17,
    do_constant_folding=True,
)
print(f"wrote {TARGET}  {TARGET.stat().st_size / 1e6:.0f} MB")

# %%
# First parity check, here, on inputs of varying length and padding. The authoritative check is
# local and covers all 2,000 test claims; this one exists so a broken export fails in the five
# minutes it took to make rather than after a download.
import onnxruntime as ort  # noqa: E402

session = ort.InferenceSession(str(TARGET), providers=["CPUExecutionProvider"])

pairs = [
    ("Barack Obama was born in Hawaii.", "Obama was born in Honolulu, Hawaii, in 1961."),
    ("The Eiffel Tower is in Berlin.", "The Eiffel Tower is a tower in Paris, France."),
    ("A short one.", "Evidence."),
    ("A claim with a much longer body of evidence attached to it for padding behaviour.",
     " ".join(["Sentence about something."] * 60)),
]
batch = tokenizer(
    [c for c, _ in pairs], [e for _, e in pairs],
    return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True,
)
with torch.no_grad():
    reference = model(**{k: batch[k] for k in ("input_ids", "attention_mask")}).logits.numpy()
got = session.run(None, {k: batch[k].numpy() for k in ("input_ids", "attention_mask")})[0]

difference = np.abs(reference - got).max()
print(f"max |logit difference| over {len(pairs)} pairs: {difference:.3e}")
print(f"argmax identical: {(reference.argmax(1) == got.argmax(1)).all()}")
assert difference < 1e-3, f"ONNX drifted by {difference:.3e}; the calibration no longer applies"

# %%
# The tokenizer travels with the model. Local inference must tokenize exactly as training did, and
# a mismatched vocabulary is the failure that produces plausible logits for the wrong input.
for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spm.model"):
    source = SOURCE / name
    if source.exists():
        shutil.copy(source, WORK / name)
(WORK / "contract.json").write_text(json.dumps(CONTRACT, indent=2))

print("\nexported:")
for path in sorted(WORK.iterdir()):
    if path.is_file():
        print(f"  {path.name:<28} {path.stat().st_size / 1e6:>8.1f} MB")
print(f"\nonnxruntime {ort.__version__}  |  opset 17")
print("download verdict.onnx and the tokenizer, then run the local parity check")
os.listdir(WORK)
