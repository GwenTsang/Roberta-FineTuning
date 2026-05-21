# Notebook de fine-tuning d'un modèle RoBERTa


```python
import os, psutil, platform, subprocess, textwrap, sys

print("Python:", sys.version)
print("Platform:", platform.platform())
print("CPU cores:", os.cpu_count())
print("RAM (GB):", round(psutil.virtual_memory().total / (1024**3), 2))

import jax
devs = jax.devices()
print("JAX devices:", devs)
print("Num devices:", len(devs))
```
```output
Python: 3.12.13 (main, Mar  4 2026, 09:23:07) [GCC 11.4.0]
Platform: Linux-6.6.122+-x86_64-with-glibc2.35
CPU cores: 24
RAM (GB): 47.05
/usr/local/lib/python3.12/dist-packages/jax/_src/cloud_tpu_init.py:86: UserWarning: Transparent hugepages are not enabled. TPU runtime startup and shutdown time should be significantly improved on TPU v5e and newer. If not already set, you may need to enable transparent hugepages in your VM image (sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled")
  warnings.warn(
JAX devices: [TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
Num devices: 1
```
cell
```python
!pip install evaluate>=0.4 -q
```

Tous les autres packages sont déjà pré-installés.
La cellule suivante fait le fine-tuning. Elle s'est exécutée intégralement en 7 minutes.


```python
#!/usr/bin/env python3
"""
================================================================================
Fine-tune RoBERTa-base for Sequence Classification (SST-2) on TPU v5e
================================================================================
  • Fixed-length tokenization → static XLA graph (no recompilation)
  • drop_last=True on DataLoaders → uniform batch shape every step
  • MpDeviceLoader for asynchronous host-to-device data prefetch
  • Minimal CPU↔TPU synchronization (logging sync only every N steps)
  • xm.mark_step() at the correct position to flush the lazy graph
================================================================================
"""
import os
os.environ.pop("XLA_USE_BF16", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import time
import math

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

from torch.utils.data import DataLoader, Dataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
)
from datasets import load_dataset
import evaluate

device = torch_xla.device()
print(f"[INFO] XLA device acquired: {device}")
print(f"[INFO] PyTorch: {torch.__version__} | torch_xla: {torch_xla.__version__}")

# HYPERPARAMETERS
MODEL_NAME       = "roberta-base"
TASK_NAME        = "sst2"           # Sentiment binary classification (GLUE)
MAX_SEQ_LENGTH   = 96               # Fixed length → static XLA graph
BATCH_SIZE       = 128              # Each TPU v5e core has 16 GiB HBM
NUM_EPOCHS       = 3
LEARNING_RATE    = 3e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.06             # 6% warmup of total steps
MAX_GRAD_NORM    = 1.0
LOGGING_STEPS    = 50               # Sync to CPU only every N steps
NUM_WORKERS      = 4                # DataLoader workers (of 24 available cores)

# Reproducibility
torch.manual_seed(11)

# SECTION 4: DATA PREPARATION
print("[INFO] Loading tokenizer and dataset...")
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

# Load SST-2 (Stanford Sentiment Treebank, binary) from GLUE benchmark.
raw_datasets = load_dataset("glue", TASK_NAME)

def tokenize_fn(examples):
    """
    Tokenize with FIXED padding to MAX_SEQ_LENGTH.

    WHY: XLA compiles a separate graph for each unique tensor shape it
    encounters.  Variable-length sequences cause repeated recompilations,
    destroying performance.  By always padding to the same length, the
    graph compiles ONCE and is reused every step.
    """
    return tokenizer(
        examples["sentence"],
        padding="max_length",        # ← ALWAYS pad to MAX_SEQ_LENGTH
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_attention_mask=True,
    )

# Tokenize all splits in parallel (batched map).
tokenized_datasets = raw_datasets.map(
    tokenize_fn,
    batched=True,
    num_proc=NUM_WORKERS,
    remove_columns=["sentence", "idx"],
    desc="Tokenizing",
)
# Tell the HF dataset to return PyTorch tensors.
tokenized_datasets.set_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)


class FixedShapeDataset(Dataset):
    """
    Thin wrapper guaranteeing every __getitem__ returns the same dict schema
    and tensor shapes.  Avoids any accidental ragged/dynamic shapes.
    """

    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids":      item["input_ids"],        # shape: [MAX_SEQ_LENGTH]
            "attention_mask": item["attention_mask"],    # shape: [MAX_SEQ_LENGTH]
            "labels":         item["label"],             # shape: scalar
        }


train_dataset = FixedShapeDataset(tokenized_datasets["train"])
eval_dataset  = FixedShapeDataset(tokenized_datasets["validation"])

# SECTION 5: DATALOADERS + XLA PARALLEL LOADER
# drop_last=True is CRITICAL:
#   The last batch is often smaller than BATCH_SIZE.  A different batch size
#   means a different tensor shape → XLA recompiles the graph.  Dropping it
#   keeps the shape uniform and avoids a costly extra compilation.

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True,               # ← Prevent ragged final batch
    persistent_workers=True,      # ← Avoid worker restart between epochs
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    drop_last=True,               # ← Same reasoning; small accuracy trade-off
    persistent_workers=True,
)

# MpDeviceLoader wraps the host DataLoader and:
#   1. Prefetches batches on a background thread.
#   2. Transfers them to the XLA device asynchronously.
# This overlaps data transfer with TPU computation (pipelining).
train_device_loader = pl.MpDeviceLoader(train_loader, device)
eval_device_loader  = pl.MpDeviceLoader(eval_loader, device)

# MODEL
print(f"[INFO] Loading {MODEL_NAME} for sequence classification (2 labels)...")
model = RobertaForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,        # SST-2 is binary
)
# the float32 weights are
model = model.to(device)
model.train()

# OPTIMIZER & LEARNING-RATE SCHEDULER
# Standard AdamW with decoupled weight decay.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-6,
    weight_decay=WEIGHT_DECAY,
)

total_steps  = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

def lr_lambda(current_step: int) -> float:
    """Linear warmup → linear decay to 0."""
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    return max(
        0.0,
        float(total_steps - current_step)
        / float(max(1, total_steps - warmup_steps)),
    )

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# TRAINING LOOP
print("  TRAINING CONFIGURATION")
print("=" * 70)
print(f"  Model             : {MODEL_NAME}")
print(f"  Task              : GLUE/{TASK_NAME}")
print(f"  Epochs            : {NUM_EPOCHS}")
print(f"  Batch size        : {BATCH_SIZE}")
print(f"  Sequence length   : {MAX_SEQ_LENGTH}")
print(f"  Total steps       : {total_steps}")
print(f"  Warmup steps      : {warmup_steps}")
print(f"  Learning rate     : {LEARNING_RATE}")
print(f"  Weight decay      : {WEIGHT_DECAY}")
print(f"  Device            : {device}")

global_step    = 0
log_step_time  = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()

    # ──────────────────────────────────────────────────────────────────────
    # Iterate via MpDeviceLoader: each `batch` is already on the XLA device.
    # DO NOT call batch.to(device) — that would be a redundant no-op at best
    # and could trigger an unwanted host-device sync at worst.
    # ──────────────────────────────────────────────────────────────────────
    for batch in train_device_loader:

        # ┌──────────────────────────────────────────────────────────────┐
        # │  FORWARD PASS                                                 │
        # │  All tensors (input_ids, attention_mask, labels, weights,     │
        # │  activations) live on the XLA device.  Operations are         │
        # │  recorded as a lazy IR graph — nothing executes yet.          │
        # └──────────────────────────────────────────────────────────────┘
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss

        # ┌──────────────────────────────────────────────────────────────┐
        # │  BACKWARD PASS                                                │
        # │  Appends gradient computation nodes to the same IR graph.     │
        # └──────────────────────────────────────────────────────────────┘
        loss.backward()

        # ┌──────────────────────────────────────────────────────────────┐
        # │  GRADIENT CLIPPING                                            │
        # │  Still lazy — adds norm computation + clipping ops to graph.  │
        # └──────────────────────────────────────────────────────────────┘
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # ┌──────────────────────────────────────────────────────────────┐
        # │  OPTIMIZER STEP + xm.mark_step()                             │
        # │                                                               │
        # │  xm.optimizer_step(optimizer) does TWO things:                │
        # │    1. optimizer.step() — adds param update ops to the graph.  │
        # │    2. xm.mark_step()  — FLUSHES the entire accumulated IR    │
        # │       graph to the XLA compiler.  The compiler optimizes,     │
        # │       compiles (first time only), and dispatches execution    │
        # │       to TPU ASYNCHRONOUSLY.                                  │
        # │                                                               │
        # │  After this call, the host is free to start building the      │
        # │  next step's graph while the TPU executes the current one.    │
        # └──────────────────────────────────────────────────────────────┘
        xm.optimizer_step(optimizer)

        # ┌──────────────────────────────────────────────────────────────┐
        # │  HOUSEKEEPING (runs on host; next graph starts here)          │
        # └──────────────────────────────────────────────────────────────┘
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)  # Free grad memory immediately

        global_step += 1

        # ┌──────────────────────────────────────────────────────────────┐
        # │  LOGGING (intentional CPU sync — only every LOGGING_STEPS)   │
        # │                                                               │
        # │  loss.item() forces a host↔device synchronization:  the host │
        # │  WAITS until the dispatched graph finishes executing on TPU,  │
        # │  then copies the scalar loss value back to CPU.               │
        # │                                                               │
        # │  We tolerate this latency only once every LOGGING_STEPS       │
        # │  iterations.  NEVER call .item()/.cpu()/print(tensor) inside  │
        # │  the hot path — it serializes host and device, killing the    │
        # │  pipelining advantage.                                        │
        # └──────────────────────────────────────────────────────────────┘
        if global_step % LOGGING_STEPS == 0:
            loss_val  = loss.item()
            elapsed   = time.time() - log_step_time
            steps_sec = LOGGING_STEPS / elapsed
            current_lr = scheduler.get_last_lr()[0]

            print(
                f"  [Epoch {epoch+1}/{NUM_EPOCHS}] "
                f"Step {global_step:>5d}/{total_steps} │ "
                f"Loss: {loss_val:.4f} │ "
                f"LR: {current_lr:.2e} │ "
                f"Speed: {steps_sec:.1f} steps/s"
            )
            log_step_time = time.time()

    epoch_elapsed = time.time() - epoch_start
    print(f"\n  ✓ Epoch {epoch+1} finished in {epoch_elapsed:.1f}s\n")

print("[INFO] Training complete.\n")

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 9: EVALUATION                                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
print("[INFO] Running evaluation on validation set...")
metric = evaluate.load("glue", TASK_NAME)
model.eval()

eval_start = time.time()

with torch.no_grad():
    for batch in eval_device_loader:
        # Forward only — no loss computation needed for predictions.
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # argmax is added to the lazy graph.
        predictions = torch.argmax(outputs.logits, dim=-1)

        # Flush the graph so predictions are materialized on device.
        xm.mark_step()

        # Transfer materialized predictions + labels to CPU for metric.
        # .cpu() after mark_step() is a simple D2H memcpy (no recomputation).
        metric.add_batch(
            predictions=predictions.cpu(),
            references=batch["labels"].cpu(),
        )

results = metric.compute()
eval_elapsed = time.time() - eval_start

print(f"  EVALUATION RESULTS (GLUE/{TASK_NAME})")
print(f"{'=' * 70}")
print(f"  Accuracy : {results['accuracy']:.4f}")
print(f"  Eval time: {eval_elapsed:.1f}s")

# SAVE MODEL
SAVE_DIR = "./roberta-sst2-finetuned"
print(f"[INFO] Saving model to {SAVE_DIR} ...")

# Move model to CPU before serialization.
# XLA tensors cannot be directly pickled by safetensors/torch.save.
model_cpu = model.cpu()
model_cpu.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"[INFO] Model + tokenizer saved to {SAVE_DIR}")
print("[INFO] Done.")
```
```output
[INFO] XLA device acquired: xla:0
[INFO] PyTorch: 2.9.0+cpu | torch_xla: 2.9.0
[INFO] Loading tokenizer and dataset...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
tokenizer_config.json:   0%|          | 0.00/25.0 [00:00<?, ?B/s]vocab.json: 0.00B [00:00, ?B/s]merges.txt: 0.00B [00:00, ?B/s]tokenizer.json: 0.00B [00:00, ?B/s]README.md: 0.00B [00:00, ?B/s]sst2/train-00000-of-00001.parquet:   0%|          | 0.00/3.11M [00:00<?, ?B/s]sst2/validation-00000-of-00001.parquet:   0%|          | 0.00/72.8k [00:00<?, ?B/s]sst2/test-00000-of-00001.parquet:   0%|          | 0.00/148k [00:00<?, ?B/s]Generating train split:   0%|          | 0/67349 [00:00<?, ? examples/s]Generating validation split:   0%|          | 0/872 [00:00<?, ? examples/s]Generating test split:   0%|          | 0/1821 [00:00<?, ? examples/s]Tokenizing (num_proc=4):   0%|          | 0/67349 [00:00<?, ? examples/s]/usr/local/lib/python3.12/dist-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/usr/local/lib/python3.12/dist-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
Tokenizing (num_proc=4):   0%|          | 0/872 [00:00<?, ? examples/s]/usr/local/lib/python3.12/dist-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/usr/local/lib/python3.12/dist-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
Tokenizing (num_proc=4):   0%|          | 0/1821 [00:00<?, ? examples/s]/usr/local/lib/python3.12/dist-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/usr/local/lib/python3.12/dist-packages/multiprocess/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
[INFO] Loading roberta-base for sequence classification (2 labels)...
config.json:   0%|          | 0.00/481 [00:00<?, ?B/s]model.safetensors:   0%|          | 0.00/499M [00:00<?, ?B/s]Loading weights:   0%|          | 0/197 [00:00<?, ?it/s]RobertaForSequenceClassification LOAD REPORT from: roberta-base
Key                             | Status     | 
--------------------------------+------------+-
lm_head.layer_norm.weight       | UNEXPECTED | 
lm_head.dense.bias              | UNEXPECTED | 
lm_head.dense.weight            | UNEXPECTED | 
roberta.embeddings.position_ids | UNEXPECTED | 
lm_head.layer_norm.bias         | UNEXPECTED | 
lm_head.bias                    | UNEXPECTED | 
classifier.dense.weight         | MISSING    | 
classifier.out_proj.bias        | MISSING    | 
classifier.out_proj.weight      | MISSING    | 
classifier.dense.bias           | MISSING    | 

Notes:
- UNEXPECTED	:can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING	:those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

======================================================================
  TRAINING CONFIGURATION
======================================================================
  Model             : roberta-base
  Task              : GLUE/sst2
  Epochs            : 3
  Batch size        : 128
  Sequence length   : 96
  Total steps       : 1578
  Warmup steps      : 94
  Learning rate     : 3e-05
  Weight decay      : 0.01
  Device            : xla:0
======================================================================

/usr/lib/python3.12/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
  [Epoch 1/3] Step    50/1578 │ Loss: 0.7062 │ LR: 1.60e-05 │ Speed: 0.4 steps/s
  [Epoch 1/3] Step   100/1578 │ Loss: 0.6707 │ LR: 2.99e-05 │ Speed: 3.9 steps/s
  [Epoch 1/3] Step   150/1578 │ Loss: 0.6848 │ LR: 2.89e-05 │ Speed: 3.9 steps/s
  [Epoch 1/3] Step   200/1578 │ Loss: 0.6316 │ LR: 2.79e-05 │ Speed: 3.9 steps/s
  [Epoch 1/3] Step   250/1578 │ Loss: 0.6359 │ LR: 2.68e-05 │ Speed: 3.8 steps/s
  [Epoch 1/3] Step   300/1578 │ Loss: 0.6033 │ LR: 2.58e-05 │ Speed: 3.8 steps/s
  [Epoch 1/3] Step   350/1578 │ Loss: 0.6266 │ LR: 2.48e-05 │ Speed: 3.9 steps/s
  [Epoch 1/3] Step   400/1578 │ Loss: 0.5481 │ LR: 2.38e-05 │ Speed: 3.9 steps/s
  [Epoch 1/3] Step   450/1578 │ Loss: 0.5792 │ LR: 2.28e-05 │ Speed: 3.9 steps/s
  [Epoch 1/3] Step   500/1578 │ Loss: 0.5345 │ LR: 2.18e-05 │ Speed: 3.9 steps/s

  ✓ Epoch 1 finished in 260.7s

  [Epoch 2/3] Step   550/1578 │ Loss: 0.5673 │ LR: 2.08e-05 │ Speed: 3.8 steps/s
  [Epoch 2/3] Step   600/1578 │ Loss: 0.5360 │ LR: 1.98e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   650/1578 │ Loss: 0.5172 │ LR: 1.88e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   700/1578 │ Loss: 0.5438 │ LR: 1.77e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   750/1578 │ Loss: 0.4999 │ LR: 1.67e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   800/1578 │ Loss: 0.5021 │ LR: 1.57e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   850/1578 │ Loss: 0.4975 │ LR: 1.47e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   900/1578 │ Loss: 0.5143 │ LR: 1.37e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step   950/1578 │ Loss: 0.4727 │ LR: 1.27e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step  1000/1578 │ Loss: 0.5693 │ LR: 1.17e-05 │ Speed: 3.9 steps/s
  [Epoch 2/3] Step  1050/1578 │ Loss: 0.5618 │ LR: 1.07e-05 │ Speed: 3.9 steps/s

  ✓ Epoch 2 finished in 135.7s

  [Epoch 3/3] Step  1100/1578 │ Loss: 0.4724 │ LR: 9.66e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1150/1578 │ Loss: 0.3917 │ LR: 8.65e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1200/1578 │ Loss: 0.4183 │ LR: 7.64e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1250/1578 │ Loss: 0.4595 │ LR: 6.63e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1300/1578 │ Loss: 0.4919 │ LR: 5.62e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1350/1578 │ Loss: 0.4431 │ LR: 4.61e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1400/1578 │ Loss: 0.4645 │ LR: 3.60e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1450/1578 │ Loss: 0.4591 │ LR: 2.59e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1500/1578 │ Loss: 0.4670 │ LR: 1.58e-06 │ Speed: 3.8 steps/s
  [Epoch 3/3] Step  1550/1578 │ Loss: 0.4111 │ LR: 5.66e-07 │ Speed: 3.8 steps/s

  ✓ Epoch 3 finished in 137.5s

[INFO] Training complete.

[INFO] Running evaluation on validation set...
Downloading builder script: 0.00B [00:00, ?B/s]/tmp/ipykernel_346/163808786.py:330: DeprecationWarning: Use torch_xla.sync instead
  xm.mark_step()

======================================================================
  EVALUATION RESULTS (GLUE/sst2)
======================================================================
  Accuracy : 0.7461
  Eval time: 9.1s
======================================================================

[INFO] Saving model to ./roberta-sst2-finetuned ...
Writing model shards:   0%|          | 0/1 [00:00<?, ?it/s][INFO] Model + tokenizer saved to ./roberta-sst2-finetuned
[INFO] Done.
```
