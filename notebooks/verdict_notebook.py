# %% [markdown]
# # Verdict model -- retrieved
#
# Fine-tunes DeBERTa-v3-base three ways on FEVER. This notebook is one of three that differ only
# in their CFG cell; the per-variant logic lives in `encode_spec.select_evidence`, shipped inside
# the dataset so training runs the same code the repository scores with.
#
# **Attach:** the `fever-verdict-v1` dataset. Nothing else, and nothing is downloaded at runtime.
#
# **Before a full run:** set `smoke = True` and run to the end. It rehearses every failure path
# in under ten minutes, including a deliberate kill and resume, and prints the extrapolated time
# for the real run. Do not clear `smoke` until that passes.

# %%
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))

# %%
CFG = dict(
    variant="retrieved",
    base_model="microsoft/deberta-v3-base",
    base_revision="main",
    max_length=512,
    template_id="per_page_grouped_v1",
    epochs=2,
    lr=2e-5,
    batch=16,
    grad_accum=2,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    smoke=True,
    time_budget_h=10.5,
    ckpt_every_steps=500,
    eval_every_steps=1000,
    contract_version=1,
)

WORK = Path("/kaggle/working")
SMOKE_ROWS, SMOKE_STEPS = 512, 60

# %%
# Inputs arrive as an attached dataset, discovered by glob so the mount name cannot break the
# notebook. Every assert says what to attach rather than what went wrong.
hits = glob.glob("/kaggle/input/**/dataset_manifest.json", recursive=True)
assert hits, "attach the fever-verdict-v1 dataset"
DATA = Path(hits[0]).parent
MANIFEST = json.loads((DATA / "dataset_manifest.json").read_text())

sys.path.insert(0, str(DATA))
import encode_spec  # noqa: E402

assert encode_spec.TEMPLATE_ID == MANIFEST["template_id"] == CFG["template_id"], (
    f"template drift: spec {encode_spec.TEMPLATE_ID}, manifest {MANIFEST['template_id']}, "
    f"cfg {CFG['template_id']} -- rebuild the dataset"
)
assert list(encode_spec.LABELS) == MANIFEST["labels"], "label order drift between spec and manifest"
assert MANIFEST["contract_version"] == CFG["contract_version"], "dataset predates this notebook"

LABELS = list(encode_spec.LABELS)
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

# The manifest hash is part of RUN_HASH: without it a re-uploaded dataset would resume a
# checkpoint trained on different rows, and nothing would say so.
MANIFEST_SHA = hashlib.sha256(json.dumps(MANIFEST["files"], sort_keys=True).encode()).hexdigest()
RUN_HASH = hashlib.sha256(
    json.dumps({**CFG, "manifest": MANIFEST_SHA}, sort_keys=True).encode()
).hexdigest()[:16]
print(f"run {RUN_HASH}  variant {CFG['variant']}  smoke {CFG['smoke']}")

# %%
def seed_everything(seed: int) -> None:
    """
    Seeded, not deterministic. torch.use_deterministic_algorithms would cost throughput and
    force CUBLAS workspace flags; saying runs are reproducible when they are only seed-controlled
    would be worse than saying neither.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %%
def read_rows(split: str) -> list[dict]:
    import gzip

    path = DATA / f"verdict_{split}.jsonl.gz"
    assert path.exists(), f"{path.name} missing from the attached dataset"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if CFG["smoke"]:
        rows = rows[: SMOKE_ROWS if split == "train" else min(256, len(rows))]
    return rows


tokenizer = AutoTokenizer.from_pretrained(CFG["base_model"], revision=CFG["base_revision"])


def measure(first: str, second: str) -> int:
    return len(tokenizer(first, second)["input_ids"])


def encode(rows: list[dict]) -> TensorDataset:
    """Build inputs through the shipped spec, so training and scoring cannot diverge."""
    firsts, seconds, labels, used = [], [], [], []
    rng = random.Random(CFG["seed"])
    for row in rows:
        evidence = encode_spec.select_evidence(row, CFG["variant"], CFG["max_length"], measure, rng)
        first, second = encode_spec.build_input(row["claim"], evidence)
        firsts.append(first)
        seconds.append(second)
        labels.append(LABEL_TO_ID[row["label"]])
        used.append(len(evidence))
    batch = tokenizer(
        firsts, seconds, truncation=True, max_length=CFG["max_length"], padding="max_length",
        return_tensors="pt",
    )
    return TensorDataset(
        batch["input_ids"], batch["attention_mask"],
        torch.tensor(labels), torch.tensor(used),
    )


SPLITS = {name: read_rows(name) for name in ("train", "trainval", "calibration", "test")}
ENCODED = {name: encode(rows) for name, rows in SPLITS.items()}
for name, rows in SPLITS.items():
    print(f"  {name:<12} {len(rows):>7,} rows")

# %%
def build_model():
    config = AutoConfig.from_pretrained(
        CFG["base_model"], revision=CFG["base_revision"], num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID,
    )
    return AutoModelForSequenceClassification.from_pretrained(
        CFG["base_model"], revision=CFG["base_revision"], config=config
    ).to(DEVICE)


def make_optimizer(model, total_steps: int):
    decay = [p for n, p in model.named_parameters() if not any(s in n for s in ("bias", "LayerNorm.weight"))]
    no_decay = [p for n, p in model.named_parameters() if any(s in n for s in ("bias", "LayerNorm.weight"))]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": CFG["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=CFG["lr"],
    )
    warmup = int(total_steps * CFG["warmup_ratio"])

    def schedule(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


# %%
CKPT = WORK / f"ckpt_{CFG['variant']}_{RUN_HASH}.pt"


def provenance() -> dict:
    return {"run_hash": RUN_HASH, "manifest_sha": MANIFEST_SHA, "cfg": CFG}


def save_checkpoint(model, optimizer, scheduler, scaler, step: int, epoch: int) -> None:
    """
    Written to .tmp then replaced, because a kill during a 700 MB save would otherwise leave a
    truncated file where the resume expects its rescue.
    """
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "epoch": epoch,
        "cpu_rng": torch.get_rng_state(), "python_rng": random.getstate(),
        "provenance": provenance(),
    }
    tmp = CKPT.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, CKPT)


def load_checkpoint(model, optimizer, scheduler, scaler):
    """A checkpoint from a different config or dataset is deleted, not adapted."""
    for candidate in [CKPT, *(Path(p) for p in glob.glob(f"/kaggle/input/**/{CKPT.name}", recursive=True))]:
        if not candidate.exists():
            continue
        payload = torch.load(candidate, map_location=DEVICE, weights_only=False)
        got = payload.get("provenance", {})
        if got.get("run_hash") != RUN_HASH or got.get("manifest_sha") != MANIFEST_SHA:
            print(f"  stale checkpoint at {candidate} (run {got.get('run_hash')}) -- ignoring")
            continue
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        torch.set_rng_state(payload["cpu_rng"].cpu())
        random.setstate(payload["python_rng"])
        print(f"  resumed from step {payload['step']}")
        return payload["step"], payload["epoch"]
    return 0, 0


# %%
def evaluate(model, dataset, rows: list[dict]) -> tuple[float, list[dict]]:
    """Accuracy plus one prediction row per claim, carrying fp32 logits for Phase 03."""
    model.eval()
    loader = DataLoader(dataset, batch_size=CFG["batch"] * 2)
    predictions, correct, offset = [], 0, 0
    with torch.no_grad():
        for input_ids, mask, labels, used in loader:
            input_ids, mask = input_ids.to(DEVICE), mask.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=mask).logits
            # float32 on the way out: Phase 03 fits a temperature on these, and a softmax at
            # fp16 and back loses information nothing downstream can recover.
            logits = logits.float().cpu()
            for i in range(logits.size(0)):
                row = rows[offset + i]
                predicted = int(logits[i].argmax())
                correct += predicted == int(labels[i])
                predictions.append({
                    "id": row["id"],
                    "label": row["label"],
                    "pred": LABELS[predicted],
                    "logits": [float(v) for v in logits[i]],
                    "n_evidence_used": int(used[i]),
                    "gold_resolved": row["gold_resolved"],
                })
            offset += logits.size(0)
    model.train()
    return correct / len(rows), predictions


# %%
def train():
    loader = DataLoader(ENCODED["train"], batch_size=CFG["batch"], shuffle=True, drop_last=True)
    steps_per_epoch = max(1, len(loader) // CFG["grad_accum"])
    total_steps = SMOKE_STEPS if CFG["smoke"] else steps_per_epoch * CFG["epochs"]

    model = build_model()
    optimizer, scheduler = make_optimizer(model, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    step, start_epoch = load_checkpoint(model, optimizer, scheduler, scaler)

    started = time.time()
    log, best, nonfinite = [], -1.0, 0
    model.train()

    for epoch in range(start_epoch, CFG["epochs"]):
        for i, (input_ids, mask, labels, _) in enumerate(loader):
            if step >= total_steps:
                break
            input_ids, mask, labels = input_ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss

            # DeBERTa-v3 disentangled attention is known to overflow under fp16, and neither T4
            # nor P100 offers bf16. Dying loudly after a run of NaNs beats five hours spent
            # training a model whose weights stopped meaning anything at step 300.
            if not torch.isfinite(loss):
                nonfinite += 1
                assert nonfinite < 20, "loss non-finite 20 steps running; rerun in fp32"
                optimizer.zero_grad(set_to_none=True)
                continue
            nonfinite = 0

            scaler.scale(loss / CFG["grad_accum"]).backward()
            if (i + 1) % CFG["grad_accum"] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                log.append({"step": step, "loss": float(loss), "lr": scheduler.get_last_lr()[0]})

                checkpoint_due = step % CFG["ckpt_every_steps"] == 0 or (CFG["smoke"] and step in (20, 40))
                eval_due = step % CFG["eval_every_steps"] == 0 or (CFG["smoke"] and step in (20, 40))
                if checkpoint_due:
                    save_checkpoint(model, optimizer, scheduler, scaler, step, epoch)
                if eval_due:
                    accuracy, _ = evaluate(model, ENCODED["trainval"], SPLITS["trainval"])
                    print(f"  step {step:>6}  loss {float(loss):.4f}  trainval {accuracy:.4f}", flush=True)
                    # Selection happens on trainval, never on calibration -- calibration has to
                    # stay unseen in every sense for Phase 03 to mean anything.
                    if accuracy > best:
                        best = accuracy
                        model.save_pretrained(WORK / "best", safe_serialization=True)

                if (time.time() - started) / 3600 > CFG["time_budget_h"]:
                    print(f"  time budget reached at step {step}; saving and scoring what exists")
                    save_checkpoint(model, optimizer, scheduler, scaler, step, epoch)
                    return model, log, step, total_steps, started, True
        if step >= total_steps:
            break

    return model, log, step, total_steps, started, False


# %%
def smoke_resume_check():
    """
    Rehearse the failure that actually loses a session: a kill mid-run.

    Trains a few steps, throws model and optimizer away, reloads from the checkpoint and asserts
    the step counter and the loss on a fixed batch both come back. Nothing else here exercises
    the resume path, and a resume that has never been run is not a resume.
    """
    model = build_model()
    optimizer, scheduler = make_optimizer(model, 40)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    fixed = next(iter(DataLoader(ENCODED["train"], batch_size=CFG["batch"], shuffle=False)))

    for _ in range(3):
        input_ids, mask, labels, _ = (t.to(DEVICE) for t in fixed)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            loss = model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    input_ids, mask, labels, _ = (t.to(DEVICE) for t in fixed)
    model.eval()
    with torch.no_grad():
        before = float(model(input_ids=input_ids, attention_mask=mask, labels=labels).loss)
    model.train()
    save_checkpoint(model, optimizer, scheduler, scaler, step=7, epoch=0)

    del model, optimizer, scheduler, scaler
    torch.cuda.empty_cache()

    model = build_model()
    optimizer, scheduler = make_optimizer(model, 40)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    step, epoch = load_checkpoint(model, optimizer, scheduler, scaler)

    model.eval()
    with torch.no_grad():
        after = float(model(input_ids=input_ids, attention_mask=mask, labels=labels).loss)

    assert step == 7 and epoch == 0, f"resumed at step {step}, epoch {epoch}"
    assert abs(before - after) < 1e-3, f"loss moved across the resume: {before:.6f} -> {after:.6f}"
    CKPT.unlink(missing_ok=True)
    print(f"  resume ok: step {step}, loss {before:.6f} -> {after:.6f}")


if CFG["smoke"]:
    print("smoke: rehearsing kill and resume")
    smoke_resume_check()

# %%
model, log, step, total_steps, started, truncated = train()
elapsed = time.time() - started
per_step = elapsed / max(1, step)
print(f"\n{step} steps in {elapsed / 60:.1f} min  ({per_step:.2f} s/step)")

if CFG["smoke"]:
    rows_full = MANIFEST["splits"]["train"]["rows"]
    full_steps = (rows_full // CFG["batch"] // CFG["grad_accum"]) * CFG["epochs"]
    print(f"full run extrapolates to roughly {per_step * full_steps / 3600:.1f} h")
    print("clear CFG['smoke'] only if that fits time_budget_h")

# %%
results = {}
for split in ("calibration", "test"):
    accuracy, predictions = evaluate(model, ENCODED[split], SPLITS[split])
    results[split] = accuracy
    with open(WORK / f"predictions_{split}.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")
    ceiling = MANIFEST["splits"][split].get("ceiling_curve", {}).get("25")
    print(f"{split:<12} accuracy {accuracy:.4f}   retrieval ceiling@25 {ceiling}")

# %%
import zipfile  # noqa: E402

model_dir = WORK / "model_v1"
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)

# The tokenizer identity is the real contract token: a transformer has no feature-name list, so
# what must not drift is the thing that turns text into ids.
tokenizer_sha = hashlib.sha256()
for name in sorted(p.name for p in model_dir.iterdir() if p.suffix in (".json", ".model")):
    tokenizer_sha.update((model_dir / name).read_bytes())

contract = {
    "base_model": CFG["base_model"], "base_revision": CFG["base_revision"],
    "tokenizer_sha256": tokenizer_sha.hexdigest(), "labels": LABELS,
    "max_length": CFG["max_length"], "template_id": CFG["template_id"],
    "variant": CFG["variant"], "seed": CFG["seed"],
    "torch_version": torch.__version__,
    "transformers_version": __import__("transformers").__version__,
    "contract_version": CFG["contract_version"],
}
(WORK / "contract.json").write_text(json.dumps(contract, indent=2))

metrics = {
    "variant": CFG["variant"], "run_hash": RUN_HASH, "manifest_sha": MANIFEST_SHA,
    "cfg": CFG, "steps": step, "planned_steps": total_steps, "truncated": truncated,
    "seconds": round(elapsed, 1), "accuracy": results,
    "train_prior": MANIFEST.get("train_prior"), "smoke": CFG["smoke"],
}
(WORK / "metrics.json").write_text(json.dumps(metrics, indent=2))
with open(WORK / "train_log.jsonl", "w", encoding="utf-8") as handle:
    for entry in log:
        handle.write(json.dumps(entry) + "\n")

archive = WORK / f"artifacts_{CFG['variant']}_v1.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
    for path in model_dir.iterdir():
        zf.write(path, f"model_v1/{path.name}")
    for name in ("contract.json", "metrics.json", "train_log.jsonl",
                 "predictions_calibration.jsonl", "predictions_test.jsonl"):
        zf.write(WORK / name, name)

# Re-open it here rather than discovering on the laptop that the archive is short.
with zipfile.ZipFile(archive) as zf:
    present = set(zf.namelist())
    required = {"contract.json", "metrics.json", "model_v1/model.safetensors"}
    assert required <= present, f"archive is missing {sorted(required - present)}"
print(f"\nwrote {archive.name}  {archive.stat().st_size / 1e6:.0f} MB")
print("download it, then: python -m scripts.install_artifacts <path>")
