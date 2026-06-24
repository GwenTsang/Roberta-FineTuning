#!/usr/bin/env python3
"""
Highly optimized single-core TPU v5e CamemBERT MLM fine-tuning.

Assumptions:
- TOKENIZED_CACHE_DIR already exists.
- Dataset is already cleaned.
- Dataset already contains fixed-shape tensors:
  input_ids, attention_mask, special_tokens_mask.
- All sequences already have MAX_SEQ_LENGTH.
- Single TPU core / one XLA device.
"""

import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

# Set before importing torch_xla.
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import CamembertForMaskedLM, CamembertTokenizerFast

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from torch_xla.amp import syncfree


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True, slots=True)
class Config:
    model_name: str = env_str("MODEL_NAME", "camembert-base")
    tokenized_cache_dir: str = env_str("TOKENIZED_CACHE_DIR", "./tokenized_cache")
    save_dir: str = "."  
    max_seq_length: int = env_int("MAX_SEQ_LENGTH", 128)
    batch_size: int = env_int("BATCH_SIZE", 248)
    epochs: int = env_int("NUM_EPOCHS", 3)

    learning_rate: float = env_float("LEARNING_RATE", 5e-5)
    weight_decay: float = env_float("WEIGHT_DECAY", 0.01)
    warmup_ratio: float = env_float("WARMUP_RATIO", 0.01)
    max_grad_norm: float = env_float("MAX_GRAD_NORM", 1.0)
    mlm_probability: float = env_float("MLM_PROBABILITY", 0.15)

    seed: int = env_int("SEED", 11)

    # 24 vCPU machine: leave a few cores for the TPU runtime and Python host.
    dataloader_workers: int = env_int("DATALOADER_NUM_WORKERS", 16)
    prefetch_factor: int = env_int("PREFETCH_FACTOR", 8)
    torch_num_threads: int = env_int("TORCH_NUM_THREADS", 4)

    xla_loader_prefetch_size: int = env_int("XLA_LOADER_PREFETCH_SIZE", 32)
    xla_device_prefetch_size: int = env_int("XLA_DEVICE_PREFETCH_SIZE", 16)
    xla_transfer_threads: int = env_int("XLA_TRANSFER_THREADS", 4)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")

    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay)
    ]
    nodecay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay)
    ]

    return syncfree.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-6,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        warmup = step / max(1, warmup_steps)
        decay = (total_steps - step) / max(1, total_steps - warmup_steps)
        return min(warmup, max(0.0, decay))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_mlm_batch(
    input_ids: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mlm_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rand_mask = torch.rand(input_ids.shape, device=input_ids.device)
    masked = (rand_mask < mlm_probability) & ~special_tokens_mask.bool()

    labels = torch.where(masked, input_ids, torch.full_like(input_ids, -100))

    replace_rand = torch.rand(input_ids.shape, device=input_ids.device)
    replace_with_mask = masked & (replace_rand < 0.8)
    replace_with_random = masked & (replace_rand >= 0.8) & (replace_rand < 0.9)

    random_words = torch.randint(
        low=0,
        high=vocab_size,
        size=input_ids.shape,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )

    masked_input_ids = torch.where(
        replace_with_mask,
        torch.full_like(input_ids, mask_token_id),
        input_ids,
    )
    masked_input_ids = torch.where(
        replace_with_random,
        random_words,
        masked_input_ids,
    )

    return masked_input_ids, labels


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def train() -> None:
    cfg = Config()

    seed_everything(cfg.seed)
    torch.set_num_threads(cfg.torch_num_threads)

    device = torch_xla.device()

    tokenizer = CamembertTokenizerFast.from_pretrained(cfg.model_name)
    mask_token_id = int(tokenizer.mask_token_id)
    vocab_size = int(tokenizer.vocab_size)

    dataset = load_from_disk(str(Path(cfg.tokenized_cache_dir)))
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "special_tokens_mask"],
    )

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.dataloader_workers,
        persistent_workers=True,
        prefetch_factor=cfg.prefetch_factor,
    )

    model = CamembertForMaskedLM.from_pretrained(
        cfg.model_name,
        use_safetensors=True,
    ).to(device)
    model.train()

    optimizer = make_optimizer(model, cfg)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = make_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    print(
        "\n"
        f"Model              : {cfg.model_name}\n"
        f"Device             : {device}\n"
        f"Epochs             : {cfg.epochs}\n"
        f"Batch size         : {cfg.batch_size}\n"
        f"Seq length         : {cfg.max_seq_length}\n"
        f"Train examples     : {len(dataset):,}\n"
        f"Steps / epoch      : {steps_per_epoch:,}\n"
        f"Total steps        : {total_steps:,}\n"
        f"Workers            : {cfg.dataloader_workers}\n"
        f"Prefetch factor    : {cfg.prefetch_factor}\n"
        f"XLA prefetch       : loader={cfg.xla_loader_prefetch_size}, "
        f"device={cfg.xla_device_prefetch_size}, transfer={cfg.xla_transfer_threads}\n"
        f"AMP                : torch.autocast('xla', dtype=torch.bfloat16)\n"
    )

    xla_loader_kwargs = {
        "loader_prefetch_size": cfg.xla_loader_prefetch_size,
        "device_prefetch_size": cfg.xla_device_prefetch_size,
        "host_to_device_transfer_threads": cfg.xla_transfer_threads,
    }

    global_step = 0
    run_start = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss = torch.zeros((), device=device)

        device_loader = pl.MpDeviceLoader(
            train_loader,
            device,
            **xla_loader_kwargs,
        )

        for batch in device_loader:
            input_ids, labels = make_mlm_batch(
                input_ids=batch["input_ids"],
                special_tokens_mask=batch["special_tokens_mask"],
                mask_token_id=mask_token_id,
                vocab_size=vocab_size,
                mlm_probability=cfg.mlm_probability,
            )

            with torch.autocast("xla", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                )
                loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            xm.optimizer_step(optimizer)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.detach()
            global_step += 1

        torch_xla.sync()

        avg_loss = (epoch_loss / steps_per_epoch).item()
        epoch_seconds = time.perf_counter() - epoch_start
        epoch_examples = steps_per_epoch * cfg.batch_size
        examples_per_second = epoch_examples / max(epoch_seconds, 1e-9)

        print(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"loss={avg_loss:.4f} "
            f"ppl={math.exp(min(avg_loss, 20.0)):.2f} "
            f"time={format_seconds(epoch_seconds)} "
            f"throughput={examples_per_second:.0f} ex/s"
        )

    torch_xla.sync()

    total_seconds = time.perf_counter() - run_start
    print(f"\nTraining finished in {format_seconds(total_seconds)}.")
    print(f"Saving to {cfg.save_dir} ...")

    model_cpu = model.cpu()
    model_cpu.save_pretrained(cfg.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(cfg.save_dir)

    print(f"Saved checkpoint: {cfg.save_dir}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()
