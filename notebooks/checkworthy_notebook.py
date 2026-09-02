# %% [markdown]
# # Check-worthiness: learn the filter Checkpoint 3 measured at F1 0.31
#
# Phase 07's hand-written filter scored precision 0.2571 and recall 0.3913 on 120 hand-labelled
# sentences of the 2016 State of the Union. The error table said the causes were structural, not
# tunable: an anchor requirement that belongs to the retrieval stack, a closed-class verb test that
# misses present-tense lexical verbs, and an `-ed` rule that fires on "United" inside a proper
# noun. Patching those against the same 120 sentences would be fitting on the test set.
#
# So the task is learned from ClaimBuster instead -- 22,501 debate sentences labelled by
# crowdworkers who had never heard of this project. That independence is what makes the eventual
# comparison mean anything.
#
# **Three classes, not two.** ClaimBuster separates factual-but-unimportant from check-worthy
# factual, and the difference is importance rather than verifiability. The Phase 07 rubric labelled
# by verifiability alone, so the two definitions disagree exactly on that middle band. A binary
# head would bury the disagreement; a three-class head lets the evaluation report both readings.
#
# **Phase 06 is the warning hanging over this.** FEVER needed ~145k rows before a transformer
# extracted anything, and at 2,103 rows the same architecture was degenerate -- it predicted one
# class for all 2,000 test claims. This has 15,512 train rows. That is 7x the degenerate size and
# a ninth of what FEVER needed, so the majority and lexical baselines below are not ceremony: if
# the transformer does not clear them, the honest reading is that the dataset is too small for
# this architecture, exactly as Phase 06 concluded.

# %%
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
    task="checkworthy",
    base_model="microsoft/deberta-v3-base",
    base_revision="8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
    max_length=128,          # a debate sentence, not a claim plus 25 evidence sentences
    template_id="sentence_v1",
    epochs=3,                # 15,512 rows is small; two epochs underfit in the smoke run
    lr=2e-5,
    batch=32,
    grad_accum=1,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    smoke=False,
    time_budget_h=3.0,
    contract_version=1,
)

LABELS = ("non_factual", "unimportant_factual", "check_worthy")
WORK = Path("/kaggle/working")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

# %%
DATA = next(Path("/kaggle/input").glob("**/checkworthy_train.jsonl")).parent
MANIFEST = json.loads((DATA / "dataset_manifest.json").read_text())
assert MANIFEST["labels"] == list(LABELS), f"label order drifted: {MANIFEST['labels']}"
print(f"data from {DATA}")
print(f"split by {MANIFEST['split_by']}  |  licence {MANIFEST['licence']}")


def read(split: str) -> list[dict]:
    with open(DATA / f"checkworthy_{split}.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


SPLITS = {name: read(name) for name in ("train", "calibration", "test")}
if CFG["smoke"]:
    SPLITS["train"] = SPLITS["train"][:2048]
for name, rows in SPLITS.items():
    print(f"  {name:<12} {len(rows):>7,} sentences, {len({r['debate'] for r in rows}):>3} debates")

INDEX = {name: i for i, name in enumerate(LABELS)}

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], revision=CFG["base_revision"])


class Sentences(Dataset):
    """Sentence in, class index out. No evidence, no template -- the task is about the sentence."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        encoded = tokenizer(
            [r["text"] for r in rows],
            truncation=True, max_length=CFG["max_length"], padding="max_length",
            return_tensors="np",
        )
        self.input_ids = encoded["input_ids"].astype(np.int64)
        self.attention_mask = encoded["attention_mask"].astype(np.int64)
        self.labels = np.asarray([INDEX[r["label"]] for r in rows], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": torch.from_numpy(self.input_ids[i]),
            "attention_mask": torch.from_numpy(self.attention_mask[i]),
            "labels": torch.tensor(self.labels[i]),
        }


DATASETS = {name: Sentences(rows) for name, rows in SPLITS.items()}

# %%
# --- baselines first, so the transformer has something to clear ------------------------------
# Phase 06's lesson: a model near its majority class looks like a result until you print the floor
# beside it. Both baselines are computed before training so neither can be quietly skipped.

train_counts = np.bincount(DATASETS["train"].labels, minlength=len(LABELS))
majority = int(train_counts.argmax())
print(f"train prior: {dict(zip(LABELS, (train_counts / train_counts.sum()).round(4)))}")


def scores(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """Accuracy plus both binarizations, which is what the demo actually cares about."""
    factual_p = predicted >= 1
    factual_a = actual >= 1
    worthy_p = predicted == 2
    worthy_a = actual == 2

    def prf(p: np.ndarray, a: np.ndarray) -> dict:
        tp = int((p & a).sum())
        precision = tp / max(int(p.sum()), 1)
        recall = tp / max(int(a.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        return {"precision": precision, "recall": recall, "f1": f1, "n_positive": int(a.sum())}

    return {
        "accuracy": float((predicted == actual).mean()),
        "factual": prf(factual_p, factual_a),
        "check_worthy": prf(worthy_p, worthy_a),
    }


BASELINES = {}
for name in ("calibration", "test"):
    actual = DATASETS[name].labels
    BASELINES[f"majority/{name}"] = scores(np.full_like(actual, majority), actual)

# Multinomial naive Bayes over unigrams -- the lexical floor. If DeBERTa cannot beat a bag of
# words on 15,512 rows, the finding is about dataset size, not about the task.
vocab: dict[str, int] = {}
for row in SPLITS["train"]:
    for word in row["text"].lower().split():
        vocab.setdefault(word, len(vocab))
counts = np.ones((len(LABELS), len(vocab) + 1))
for row in SPLITS["train"]:
    k = INDEX[row["label"]]
    for word in row["text"].lower().split():
        counts[k, vocab.get(word, len(vocab))] += 1
log_likelihood = np.log(counts / counts.sum(axis=1, keepdims=True))
log_prior = np.log(train_counts / train_counts.sum())


def naive_bayes(rows: list[dict]) -> np.ndarray:
    out = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        total = log_prior.copy()
        for word in row["text"].lower().split():
            total += log_likelihood[:, vocab.get(word, len(vocab))]
        out[i] = int(total.argmax())
    return out


for name in ("calibration", "test"):
    BASELINES[f"naive_bayes/{name}"] = scores(naive_bayes(SPLITS[name]), DATASETS[name].labels)

print(f"\n{'baseline':<26} {'accuracy':>9} {'factual F1':>11} {'worthy F1':>10}")
for name, s in BASELINES.items():
    print(f"{name:<26} {s['accuracy']:>9.4f} {s['factual']['f1']:>11.4f} "
          f"{s['check_worthy']['f1']:>10.4f}")

# %%
def build_model():
    config = AutoConfig.from_pretrained(
        CFG["base_model"], revision=CFG["base_revision"], num_labels=len(LABELS)
    )
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
def evaluate(model, dataset, *, precise: bool = False) -> tuple[dict, np.ndarray]:
    """
    precise=True runs fp32. Logits born under autocast are fp16 and casting them wider afterwards
    widens the dtype without recovering the precision -- a trap that never raises and quietly
    changes what any calibrator downstream is fitted on.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=CFG["batch"] * (1 if precise else 2))
    collected = []
    for batch in loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        with torch.autocast("cuda", dtype=torch.float16,
                            enabled=not precise and DEVICE == "cuda"):
            out = model(input_ids=ids, attention_mask=mask).logits
        collected.append(out.float().cpu().numpy())
    logits = np.concatenate(collected)
    return scores(logits.argmax(axis=1), dataset.labels), logits


# %%
loader = DataLoader(DATASETS["train"], batch_size=CFG["batch"], shuffle=True, drop_last=True)
total_steps = (len(loader) // CFG["grad_accum"]) * CFG["epochs"]
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
schedule = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=CFG["lr"], total_steps=total_steps,
    pct_start=CFG["warmup_ratio"], anneal_strategy="linear",
)
scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

print(f"{total_steps:,} optimizer steps over {CFG['epochs']} epochs")
started = time.time()
step = 0
log = []
best = {"f1": -1.0}

for epoch in range(CFG["epochs"]):
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
            if step % 50 == 0:
                log.append({"step": step, "loss": float(loss), "lr": schedule.get_last_lr()[0]})
                print(f"  step {step:>5}/{total_steps}  loss {float(loss):.4f}", flush=True)

    # Model selection on calibration, by the metric the demo turns on: the factual/non-factual
    # split, which is the one the Phase 07 hand labels are comparable to.
    picked, _ = evaluate(model, DATASETS["calibration"], precise=True)
    print(f"epoch {epoch + 1}: calibration accuracy {picked['accuracy']:.4f}  "
          f"factual F1 {picked['factual']['f1']:.4f}  worthy F1 {picked['check_worthy']['f1']:.4f}",
          flush=True)
    if picked["factual"]["f1"] > best["f1"]:
        best = {"f1": picked["factual"]["f1"], "epoch": epoch + 1}
        model.save_pretrained(WORK / "best", safe_serialization=True)
    model.train()
    if (time.time() - started) / 3600 > CFG["time_budget_h"]:
        print("time budget reached")
        break

elapsed = time.time() - started
print(f"\ntrained in {elapsed / 60:.1f} min, best epoch {best.get('epoch')}")

# %%
# Reload the selected epoch, then score every split in fp32.
model = AutoModelForSequenceClassification.from_pretrained(WORK / "best").float().to(DEVICE)

RESULTS = {}
for name in ("calibration", "test"):
    RESULTS[name], logits = evaluate(model, DATASETS[name], precise=True)
    with open(WORK / f"predictions_{name}.jsonl", "w", encoding="utf-8") as handle:
        for row, vector in zip(SPLITS[name], logits):
            handle.write(json.dumps({
                "id": row["id"], "debate": row["debate"], "label": row["label"],
                "logits": [float(x) for x in vector],
            }) + "\n")

print(f"\n{'model':<26} {'accuracy':>9} {'factual F1':>11} {'worthy F1':>10}")
for name, s in BASELINES.items():
    print(f"{name:<26} {s['accuracy']:>9.4f} {s['factual']['f1']:>11.4f} "
          f"{s['check_worthy']['f1']:>10.4f}")
for name, s in RESULTS.items():
    print(f"{'deberta/' + name:<26} {s['accuracy']:>9.4f} {s['factual']['f1']:>11.4f} "
          f"{s['check_worthy']['f1']:>10.4f}")

lift = RESULTS["test"]["factual"]["f1"] - BASELINES["naive_bayes/test"]["factual"]["f1"]
print(f"\nfactual F1 over the lexical floor: {lift:+.4f}")
print("A transformer that cannot clear a bag of words on 15,512 rows is Phase 06's finding again,"
      "\nnot a check-worthiness result -- report it that way if that is what happened.")

# %%
model_dir = WORK / "model_v1"
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)

# Hashed over the tokenizer files as saved, which is the real identity check: a moving tag can
# change them under a fixed model name, and a mismatched vocabulary produces plausible logits for
# the wrong input.
tokenizer_sha = hashlib.sha256()
for name in sorted(p.name for p in model_dir.iterdir()):
    if name.startswith(("tokenizer", "special_tokens", "spm")):
        tokenizer_sha.update((model_dir / name).read_bytes())

contract = {
    "base_model": CFG["base_model"], "base_revision": CFG["base_revision"],
    "tokenizer_sha256": tokenizer_sha.hexdigest(), "labels": list(LABELS),
    "max_length": CFG["max_length"], "template_id": CFG["template_id"],
    "variant": CFG["task"], "seed": CFG["seed"],
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "contract_version": CFG["contract_version"],
}
(WORK / "contract.json").write_text(json.dumps(contract, indent=2))
(WORK / "metrics.json").write_text(json.dumps({
    "task": CFG["task"], "cfg": CFG, "steps": step, "seconds": round(elapsed, 1),
    "train_prior": MANIFEST["train_prior"], "baselines": BASELINES, "model": RESULTS,
    "best_epoch": best.get("epoch"), "factual_f1_over_lexical": lift,
}, indent=2))
with open(WORK / "train_log.jsonl", "w", encoding="utf-8") as handle:
    for entry in log:
        handle.write(json.dumps(entry) + "\n")

print("\nwritten:")
for path in sorted(WORK.iterdir()):
    if path.is_file():
        print(f"  {path.name:<28} {path.stat().st_size / 1e6:>8.2f} MB")
print(os.listdir(WORK))
