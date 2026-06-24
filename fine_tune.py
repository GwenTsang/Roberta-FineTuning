#!/usr/bin/env python3
"""
CamemBERT MLM fine-tuning on TPU v5e 1-core.

Uses 100% of the corpus for training (no eval split) because evaluation is
done downstream on a classification task with a linear head over the hidden
states of the last layer.

Usage in notebook:
    %run -i tokenize_data.py   # only the first time
    %run -i fine_tuning.py
"""

import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Must be set before importing torch_xla.
os.environ.setdefault("XLA_USE_BF16", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import (
    CamembertForMaskedLM,
    CamembertTokenizerFast,
    DataCollatorForLanguageModeling,
)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


@dataclass(frozen=True)
class Config:
    model_name: str = os.getenv("MODEL_NAME", "camembert-base")
    max_seq_length: int = env_int("MAX_SEQ_LENGTH", 128)

    batch_size: int = env_int("BATCH_SIZE", 128)
    gradient_accumulation_steps: int = env_int("GRADIENT_ACCUMULATION_STEPS", 1)

    num_epochs: int = env_int("NUM_EPOCHS", 3)
    learning_rate: float = env_float("LEARNING_RATE", 5e-5)
    weight_decay: float = env_float("WEIGHT_DECAY", 0.01)
    warmup_ratio: float = env_float("WARMUP_RATIO", 0.01)
    max_grad_norm: float = env_float("MAX_GRAD_NORM", 1.0)
    mlm_probability: float = env_float("MLM_PROBABILITY", 0.15)

    logging_steps: int = env_int("LOGGING_STEPS", 50)
    dataloader_num_workers: int = env_int("DATALOADER_NUM_WORKERS", 0)
    prefetch_factor: int = env_int("PREFETCH_FACTOR", 2)
    torch_num_threads: int = env_int("TORCH_NUM_THREADS", 4)

    reuse_tokenizer: bool = env_bool("REUSE_TOKENIZER", True)
    reuse_model: bool = env_bool("REUSE_MODEL", True)
    gradient_checkpointing: bool = env_bool("GRADIENT_CHECKPOINTING", False)

    seed: int = env_int("SEED", 11)
    save_dir: str = os.getenv("SAVE_DIR", "./camembert-base-french-comments-tweets-mlm")
    tokenized_cache_dir: str = os.getenv("TOKENIZED_CACHE_DIR", "./tokenized_cache")


class FixedShapeMLMDataset(Dataset):
    """Stable sample schema; labels are created by the MLM collator."""

    def __init__(self, hf_dataset: HFDataset):
        self.data = hf_dataset

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.data[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "special_tokens_mask": item["special_tokens_mask"],
        }


def set_reproducibility(seed: int, torch_num_threads: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, torch_num_threads))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = env_int("SEED", 11) + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def get_global(name: str) -> Any | None:
    return globals().get(name)


def get_tokenizer(cfg: Config) -> CamembertTokenizerFast:
    existing = get_global("tokenizer")
    if cfg.reuse_tokenizer and existing is not None:
        print("[INFO] Reusing tokenizer from notebook globals.")
        return existing
    print(f"[INFO] Loading tokenizer: {cfg.model_name}")
    return CamembertTokenizerFast.from_pretrained(cfg.model_name)


def load_tokenized_dataset(cfg: Config) -> HFDataset:
    existing = get_global("tokenized_dataset")
    if isinstance(existing, HFDataset) and len(existing) > 0:
        try:
            if len(existing[0]["input_ids"]) == cfg.max_seq_length:
                print("[INFO] Reusing tokenized_dataset from notebook globals.")
                existing.set_format(
                    type="torch",
                    columns=["input_ids", "attention_mask", "special_tokens_mask"],
                )
                return existing
        except Exception:
            pass

    cache_path = Path(cfg.tokenized_cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Tokenized cache not found at {cache_path}. Run tokenize_data.py first."
        )
    print(f"[INFO] Loading tokenized dataset from cache: {cache_path}")
    dataset = load_from_disk(str(cache_path))
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "special_tokens_mask"],
    )
    return dataset


def make_loader(
    dataset: Dataset,
    cfg: Config,
    collator: DataCollatorForLanguageModeling,
    generator: torch.Generator | None = None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "collate_fn": collator,
        "num_workers": cfg.dataloader_num_workers,
        "drop_last": True,
        "worker_init_fn": seed_worker if cfg.dataloader_num_workers > 0 else None,
    }
    if generator is not None:
        kwargs["generator"] = generator
    if cfg.dataloader_num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def get_model(cfg: Config, device: torch.device) -> CamembertForMaskedLM:
    existing = get_global("model")
    if cfg.reuse_model and isinstance(existing, CamembertForMaskedLM):
        print("[INFO] Reusing CamembertForMaskedLM model from notebook globals.")
        model_obj = existing
    else:
        print(f"[INFO] Loading {cfg.model_name} for Masked Language Modeling.")
        model_obj = CamembertForMaskedLM.from_pretrained(
            cfg.model_name,
            use_safetensors=True,
        )

    if cfg.gradient_checkpointing:
        print("[INFO] Enabling gradient checkpointing.")
        model_obj.gradient_checkpointing_enable()

    model_obj = model_obj.to(device)
    model_obj.train()
    return model_obj


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_update_steps: int,
    cfg: Config,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_update_steps * cfg.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_update_steps - current_step)
            / float(max(1, total_update_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def print_config(
    cfg: Config,
    train_dataset: Dataset,
    train_loader: DataLoader,
    update_steps_per_epoch: int,
    total_update_steps: int,
    device: torch.device,
) -> None:
    print("\nTRAINING CONFIGURATION")
    print(f"Model                    : {cfg.model_name}")
    print("Objective                : Masked Language Modeling (full corpus)")
    print(f"XLA BF16                 : {os.getenv('XLA_USE_BF16')}")
    print(f"Epochs                   : {cfg.num_epochs}")
    print(f"Batch size               : {cfg.batch_size}")
    print(f"Gradient accumulation    : {cfg.gradient_accumulation_steps}")
    print(f"Effective batch          : {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print(f"Sequence length          : {cfg.max_seq_length}")
    print(f"MLM probability          : {cfg.mlm_probability}")
    print(f"Train examples           : {len(train_dataset):,}")
    print(f"Micro-batches / epoch    : {len(train_loader):,}")
    print(f"Update steps / epoch     : {update_steps_per_epoch:,}")
    print(f"Total update steps       : {total_update_steps:,}")
    print(f"Warmup steps             : {int(total_update_steps * cfg.warmup_ratio):,}")
    print(f"Learning rate            : {cfg.learning_rate}")
    print(f"Weight decay             : {cfg.weight_decay}")
    print(f"DataLoader workers       : {cfg.dataloader_num_workers}")
    print(f"Torch CPU threads        : {cfg.torch_num_threads}")
    print(f"Device                   : {device}")


def train() -> None:
    cfg = Config()
    if cfg.gradient_accumulation_steps < 1:
        raise ValueError("GRADIENT_ACCUMULATION_STEPS must be >= 1.")

    loader_generator = set_reproducibility(cfg.seed, cfg.torch_num_threads)

    print("Python:", sys.version)
    print("Platform:", platform.platform())
    print("CPU cores:", os.cpu_count())

    tokenizer = get_tokenizer(cfg)
    tokenized_dataset = load_tokenized_dataset(cfg)
    train_dataset = FixedShapeMLMDataset(tokenized_dataset)

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.mlm_probability,
        return_tensors="pt",
    )

    train_loader = make_loader(
        train_dataset,
        cfg,
        mlm_collator,
        generator=loader_generator,
    )

    if len(train_loader) == 0:
        raise ValueError(
            f"Training DataLoader has 0 full batches. Reduce BATCH_SIZE={cfg.batch_size}."
        )
    if len(train_loader) < cfg.gradient_accumulation_steps:
        raise ValueError(
            "GRADIENT_ACCUMULATION_STEPS is larger than the number of train batches."
        )

    # Import XLA only after dataset setup is complete.
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl

    device = torch_xla.device()
    print(f"[INFO] XLA device acquired: {device}")
    print(f"[INFO] PyTorch: {torch.__version__} | torch_xla: {torch_xla.__version__}")

    model = get_model(cfg, device)
    optimizer = make_optimizer(model, cfg)

    update_steps_per_epoch = len(train_loader) // cfg.gradient_accumulation_steps
    total_update_steps = update_steps_per_epoch * cfg.num_epochs
    scheduler = make_scheduler(optimizer, total_update_steps, cfg)

    print_config(
        cfg,
        train_dataset,
        train_loader,
        update_steps_per_epoch,
        total_update_steps,
        device,
    )

    global_update_step = 0
    running_loss_sum = 0.0
    running_loss_count = 0
    optimizer.zero_grad(set_to_none=True)
    log_start = time.time()

    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        model.train()
        train_device_loader = pl.MpDeviceLoader(train_loader, device)

        usable_micro_batches = update_steps_per_epoch * cfg.gradient_accumulation_steps

        for micro_step, batch in enumerate(train_device_loader, start=1):
            if micro_step > usable_micro_batches:
                break

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / cfg.gradient_accumulation_steps
            loss.backward()

            if micro_step % cfg.gradient_accumulation_steps != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            xm.optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_update_step += 1

            if global_update_step % cfg.logging_steps == 0:
                # .item() on XLA forces a sync; only do it at logging boundaries.
                loss_val = (loss * cfg.gradient_accumulation_steps).item()
                elapsed = time.time() - log_start
                updates_per_sec = cfg.logging_steps / max(elapsed, 1e-9)
                examples_per_sec = (
                    cfg.logging_steps
                    * cfg.batch_size
                    * cfg.gradient_accumulation_steps
                    / max(elapsed, 1e-9)
                )
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"[Epoch {epoch + 1}/{cfg.num_epochs}] "
                    f"Update {global_update_step:>6d}/{total_update_steps} | "
                    f"MLM loss: {loss_val:.4f} | "
                    f"perplexity: {math.exp(min(loss_val, 20)):.2f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Speed: {updates_per_sec:.2f} upd/s | "
                    f"{examples_per_sec:.0f} ex/s"
                )
                log_start = time.time()

        torch_xla.sync()
        epoch_elapsed = time.time() - epoch_start
        print(f"\n[INFO] Epoch {epoch + 1} finished in {epoch_elapsed:.1f}s\n")

    print("[INFO] Training complete.")
    print(f"[INFO] Saving model and tokenizer to {cfg.save_dir} ...")
    torch_xla.sync()
    model_cpu = model.cpu()
    model_cpu.save_pretrained(cfg.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.save_dir)
    print(f"[INFO] Saved CamemBERT MLM checkpoint to: {cfg.save_dir}")


if __name__ == "__main__":
    train()
