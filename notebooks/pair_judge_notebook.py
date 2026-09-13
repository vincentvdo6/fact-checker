# %% [markdown]
# # Pair judge: one sentence against one assertion, four relations
#
# The decomposed checker judges evidence one sentence at a time: does this sentence *state* the
# assertion, state its *negation*, *bear on* its subject without settling it, or concern
# something else. Every off-the-shelf candidate failed that task on real news sentences in
# opposite directions -- the FEVER-tuned verdict model reads a negation word in the claim as a
# REFUTES tell and cannot read "no shutdowns" as denying a shutdown; SNLI/MNLI cross-encoders read
# negation but call "LISEP issues TRU one to two weeks after the BLS report" a contradiction of
# "we have a good job shortage" at 0.99. Neither was ever measured on this task. This notebook
# trains the smallest DeBERTa that can carry it, so the judge can be calibrated and banded with
# the Phase 03 machinery and shipped at a size an extension can hold.
#
# **The hypothesis-only baseline is the artifact detector.** FEVER's REFUTES claims carry a
# negation word ten times as often as SUPPORTS claims; in this dataset `states_negation`
# hypotheses are negated 29% of the time against 6% for `states`. A model that sees only the
# assertion and still beats the majority floor is reading that tell. It is computed first, and
# the transformer's lift is reported against it, not against chance.
#
# **Model selection is on trainval; calibration is never looked at while training.** The
# calibration split exists to fit the temperature and the bands afterwards, and fitting on
# anything the model was selected on roughly halves the reported ECE.

# %%
import gzip
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

CFG = dict(
    task="pair_judge",
    base_model="microsoft/deberta-v3-xsmall",
    base_revision="4b419818330868dff6a60ad3e6b1c730f8b8c0c6",
    max_length=192,          # one sentence with its own-paragraph definitions, plus an assertion
    template_id="premise_hypothesis_v1",
    epochs=2,
    lr=3e-5,                 # xsmall tolerates a higher rate than base
    batch=64,
    grad_accum=1,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
    max_per_class=60_000,    # cap per (source, relation) in train; FEVER bears_on/unrelated are 140k each
    seed=42,
    smoke=False,
    time_budget_h=5.0,
    contract_version=1,
)

LABELS = ("states", "states_negation", "bears_on", "unrelated")
WORK = Path("/kaggle/working")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

# %%
DATA = next(Path("/kaggle/input").glob("**/pairs_train.jsonl*")).parent
MANIFEST = json.loads((DATA / "dataset_manifest.json").read_text())
assert MANIFEST["relations"] == list(LABELS), f"relation order drifted: {MANIFEST['relations']}"
print(f"data from {DATA}")
print(f"quotas {MANIFEST['quotas']}")


def read(split: str) -> list[dict]:
    # Kaggle decompresses .gz on dataset ingest, so both spellings are accepted.
    path = DATA / f"pairs_{split}.jsonl.gz"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    with open(DATA / f"pairs_{split}.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


SPLITS = {name: read(name) for name in ("train", "trainval", "calibration", "test")}

# Cap the train split per (source, relation) with a seeded shuffle, so no class is 140k rows while
# another is 29k. The cap is recorded in the contract; the held-out splits are never capped.
rng = random.Random(CFG["seed"])
rng.shuffle(SPLITS["train"])
kept, taken = [], {}
for row in SPLITS["train"]:
    key = (row["source"], row["relation"])
    if taken.get(key, 0) < CFG["max_per_class"]:
        taken[key] = taken.get(key, 0) + 1
        kept.append(row)
SPLITS["train"] = kept
if CFG["smoke"]:
    SPLITS["train"] = SPLITS["train"][:4096]
for name, rows in SPLITS.items():
    counts = {}
    for row in rows:
        counts[row["relation"]] = counts.get(row["relation"], 0) + 1
    print(f"  {name:<12} {len(rows):>8,} pairs  {dict(sorted(counts.items()))}")

INDEX = {name: i for i, name in enumerate(LABELS)}

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], revision=CFG["base_revision"])


class Pairs(Dataset):
    """(premise, hypothesis) in, relation index out. Premise first: that is how the judge asks."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        encoded = tokenizer(
            [r["premise"] for r in rows], [r["hypothesis"] for r in rows],
            truncation=True, max_length=CFG["max_length"], padding="max_length", return_tensors="np",
        )
        self.input_ids = encoded["input_ids"].astype(np.int64)
        self.attention_mask = encoded["attention_mask"].astype(np.int64)
        self.labels = np.asarray([INDEX[r["relation"]] for r in rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return {"input_ids": torch.from_numpy(self.input_ids[i]),
                "attention_mask": torch.from_numpy(self.attention_mask[i]),
                "labels": torch.tensor(self.labels[i])}


DATASETS = {name: Pairs(rows) for name, rows in SPLITS.items()}

# %%
# --- baselines first, so the transformer has something to clear ------------------------------
train_counts = np.bincount(DATASETS["train"].labels, minlength=len(LABELS))
majority = int(train_counts.argmax())
print(f"train prior: {dict(zip(LABELS, (train_counts / train_counts.sum()).round(4)))}")

COUNTED = {INDEX["states"], INDEX["states_negation"]}


def scores(predicted: np.ndarray, actual: np.ndarray, rows: list[dict]) -> dict:
    """Accuracy, macro F1, the counting error, and the two subgroups that matter here."""
    per = {}
    f1s = []
    for k, name in enumerate(LABELS):
        tp = int(((predicted == k) & (actual == k)).sum())
        p = tp / max(int((predicted == k).sum()), 1)
        r = tp / max(int((actual == k).sum()), 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        per[name] = {"precision": p, "recall": r, "f1": f1, "support": int((actual == k).sum())}
        f1s.append(f1)
    counted = np.isin(predicted, list(COUNTED))
    wrong = counted & (predicted != actual)
    source = np.asarray([r["source"] for r in rows])
    negated = np.asarray([bool(r["negated_hypothesis"]) for r in rows])
    return {
        "accuracy": float((predicted == actual).mean()),
        "macro_f1": float(np.mean(f1s)),
        "per_relation": per,
        "counting_error": {"wrong": int(wrong.sum()), "counted": int(counted.sum()),
                           "rate": float(wrong.sum() / max(counted.sum(), 1))},
        "by_source": {s: float((predicted[source == s] == actual[source == s]).mean()) for s in ("fever", "mnli")},
        "negated_hypothesis": {
            "accuracy": float((predicted[negated] == actual[negated]).mean()) if negated.any() else None,
            "n": int(negated.sum()),
            "predicted_negation_rate": float((predicted[negated] == INDEX["states_negation"]).mean()) if negated.any() else None,
        },
    }


BASELINES = {}
for name in ("trainval", "test"):
    actual = DATASETS[name].labels
    BASELINES[f"majority/{name}"] = scores(np.full_like(actual, majority), actual, SPLITS[name])


def bag(row: dict, view: str) -> list[str]:
    words = []
    if view in ("hypothesis", "pair"):
        words += ["h:" + w for w in row["hypothesis"].lower().split()]
    if view in ("premise", "pair"):
        words += ["p:" + w for w in row["premise"].lower().split()]
    return words


def naive_bayes(view: str) -> dict:
    vocab: dict[str, int] = {}
    for row in SPLITS["train"]:
        for word in bag(row, view):
            vocab.setdefault(word, len(vocab))
    counts = np.ones((len(LABELS), len(vocab) + 1))
    for row in SPLITS["train"]:
        k = INDEX[row["relation"]]
        for word in bag(row, view):
            counts[k, vocab.get(word, len(vocab))] += 1
    log_likelihood = np.log(counts / counts.sum(axis=1, keepdims=True))
    log_prior = np.log(train_counts / train_counts.sum())

    def predict(rows: list[dict]) -> np.ndarray:
        out = np.empty(len(rows), dtype=np.int64)
        for i, row in enumerate(rows):
            total = log_prior.copy()
            for word in bag(row, view):
                total += log_likelihood[:, vocab.get(word, len(vocab))]
            out[i] = int(total.argmax())
        return out

    return {name: scores(predict(SPLITS[name]), DATASETS[name].labels, SPLITS[name]) for name in ("trainval", "test")}


# hypothesis-only is the artifact detector: it sees the assertion and never the sentence.
for view in ("hypothesis", "premise", "pair"):
    for name, result in naive_bayes(view).items():
        BASELINES[f"nb_{view}/{name}"] = result

print(f"\n{'baseline':<26} {'accuracy':>9} {'macro F1':>9} {'count err':>10} {'neg acc':>8}")
for name, s in BASELINES.items():
    print(f"{name:<26} {s['accuracy']:>9.4f} {s['macro_f1']:>9.4f} {s['counting_error']['rate']:>10.4f} "
          f"{(s['negated_hypothesis']['accuracy'] or 0):>8.4f}")

# %%
def build_model():
    config = AutoConfig.from_pretrained(CFG["base_model"], revision=CFG["base_revision"], num_labels=len(LABELS))
    model = AutoModelForSequenceClassification.from_pretrained(
        CFG["base_model"], revision=CFG["base_revision"], config=config
    )
    # Kaggle's transformers honours the dtype in the checkpoint config and hands back fp16 master
    # weights, which GradScaler then refuses with "Attempting to unscale FP16 gradients". Calling
    # .float() is explicit and does not depend on how the installed version spells the argument.
    return model.float().to(DEVICE)


model = build_model()
print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {DEVICE}")


@torch.no_grad()
def evaluate(model, name: str, *, precise: bool = False) -> tuple[dict, np.ndarray]:
    """
    precise=True runs fp32. Logits born under autocast are fp16 and casting them wider afterwards
    widens the dtype without recovering the precision -- a trap that never raises and quietly
    changes what any calibrator downstream is fitted on.
    """
    model.eval()
    dataset = DATASETS[name]
    loader = DataLoader(dataset, batch_size=CFG["batch"] * (1 if precise else 2))
    collected = []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=not precise and DEVICE == "cuda"):
            out = model(input_ids=ids, attention_mask=mask).logits
        collected.append(out.float().cpu().numpy())
    logits = np.concatenate(collected)
    return scores(logits.argmax(axis=1), dataset.labels, SPLITS[name]), logits


# %%
loader = DataLoader(DATASETS["train"], batch_size=CFG["batch"], shuffle=True, drop_last=True)
total_steps = (len(loader) // CFG["grad_accum"]) * CFG["epochs"]
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
schedule = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CFG["lr"], total_steps=total_steps, pct_start=CFG["warmup_ratio"], anneal_strategy="linear",
)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

print(f"{total_steps:,} optimizer steps over {CFG['epochs']} epochs")
started = time.time()
step = 0
log = []
best = {"macro_f1": -1.0}

for epoch in range(CFG["epochs"]):
    # An odd number of batches leaves an incomplete accumulation group; discard it between epochs.
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for i, batch in enumerate(loader):
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        scaler.scale(loss / CFG["grad_accum"]).backward()
        if (i + 1) % CFG["grad_accum"] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if step < total_steps - 1:
                schedule.step()
            step += 1
            if step % 200 == 0:
                log.append({"step": step, "loss": float(loss), "lr": schedule.get_last_lr()[0]})
                print(f"  step {step:>6}/{total_steps}  loss {float(loss):.4f}", flush=True)
        if (time.time() - started) / 3600 > CFG["time_budget_h"]:
            break

    # Model selection on trainval, never on calibration, by macro F1: the counting relations are
    # the minority classes and accuracy would reward ignoring them.
    picked, _ = evaluate(model, "trainval", precise=True)
    print(f"epoch {epoch + 1}: trainval accuracy {picked['accuracy']:.4f}  macro F1 {picked['macro_f1']:.4f}  "
          f"counting error {picked['counting_error']['rate']:.4f}  "
          f"negated acc {picked['negated_hypothesis']['accuracy']:.4f}", flush=True)
    if picked["macro_f1"] > best["macro_f1"]:
        best = {"macro_f1": picked["macro_f1"], "epoch": epoch + 1}
        model.save_pretrained(WORK / "best", safe_serialization=True)
    if (time.time() - started) / 3600 > CFG["time_budget_h"]:
        print("time budget reached")
        break

elapsed = time.time() - started
print(f"\ntrained in {elapsed / 60:.1f} min, best epoch {best.get('epoch')}")

# %%
# Reload the selected epoch, then score every held-out split in fp32 and save its logits: the
# calibrator and the bands are fitted locally on the calibration file and reported on test.
model = AutoModelForSequenceClassification.from_pretrained(WORK / "best").float().to(DEVICE)

RESULTS = {}
for name in ("trainval", "calibration", "test"):
    RESULTS[name], logits = evaluate(model, name, precise=True)
    with open(WORK / f"predictions_{name}.jsonl", "w", encoding="utf-8") as handle:
        for row, vector in zip(SPLITS[name], logits):
            handle.write(json.dumps({"id": row["id"], "source": row["source"], "relation": row["relation"],
                                     "negated_hypothesis": row["negated_hypothesis"],
                                     "logits": [float(x) for x in vector]}) + "\n")

print(f"\n{'model':<26} {'accuracy':>9} {'macro F1':>9} {'count err':>10} {'neg acc':>8} {'fever':>7} {'mnli':>7}")
for name, s in list(BASELINES.items()) + [("deberta/" + k, v) for k, v in RESULTS.items()]:
    print(f"{name:<26} {s['accuracy']:>9.4f} {s['macro_f1']:>9.4f} {s['counting_error']['rate']:>10.4f} "
          f"{(s['negated_hypothesis']['accuracy'] or 0):>8.4f} {s['by_source']['fever']:>7.4f} {s['by_source']['mnli']:>7.4f}")

lift = RESULTS["test"]["macro_f1"] - BASELINES["nb_pair/test"]["macro_f1"]
artifact = BASELINES["nb_hypothesis/test"]["accuracy"] - BASELINES["majority/test"]["accuracy"]
print(f"\nmacro F1 over the lexical pair floor: {lift:+.4f}")
print(f"hypothesis-only lift over majority (the artifact): {artifact:+.4f}")
print("A judge that cannot clear a bag of words is not a judge; a hypothesis-only model that clears the"
      "\nfloor is reading the assertion's wording, and the negated-hypothesis column says how much.")

# %%
model_dir = WORK / "model_v1"
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)

tokenizer_sha = hashlib.sha256()
for name in sorted(p.name for p in model_dir.iterdir()):
    if name.startswith(("tokenizer", "special_tokens", "spm")):
        tokenizer_sha.update((model_dir / name).read_bytes())

contract = {
    "base_model": CFG["base_model"], "base_revision": CFG["base_revision"],
    "tokenizer_sha256": tokenizer_sha.hexdigest(), "labels": list(LABELS),
    "max_length": CFG["max_length"], "template_id": CFG["template_id"],
    "variant": CFG["task"], "seed": CFG["seed"], "max_per_class": CFG["max_per_class"],
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "contract_version": CFG["contract_version"],
}
(WORK / "contract.json").write_text(json.dumps(contract, indent=2))
(WORK / "metrics.json").write_text(json.dumps({
    "task": CFG["task"], "cfg": CFG, "steps": step, "seconds": round(elapsed, 1),
    "train_counts": dict(zip(LABELS, train_counts.tolist())), "baselines": BASELINES, "model": RESULTS,
    "best_epoch": best.get("epoch"), "macro_f1_over_lexical": lift, "hypothesis_only_artifact": artifact,
    "dataset_manifest": MANIFEST,
}, indent=2))
with open(WORK / "train_log.jsonl", "w", encoding="utf-8") as handle:
    for entry in log:
        handle.write(json.dumps(entry) + "\n")

print("\nwritten:")
for path in sorted(WORK.iterdir()):
    if path.is_file():
        print(f"  {path.name:<28} {path.stat().st_size / 1e6:>8.2f} MB")
print(os.listdir(WORK))
