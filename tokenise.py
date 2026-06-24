#!/usr/bin/env python3
"""
Tokenize the full corpus once and cache it to disk.

Usage in notebook:
    %run -i load_data.py
    %run -i tokenize_data.py

All data is placed into the training corpus (no train/test split) because the
goal is to compare a vanilla CamemBERT vs a fine-tuned CamemBERT on a separate
downstream classification task.
"""

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import polars as pl
from datasets import Dataset as HFDataset
from transformers import CamembertTokenizerFast


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


MODEL_NAME = os.getenv("MODEL_NAME", "camembert-base")
MAX_SEQ_LENGTH = env_int("MAX_SEQ_LENGTH", 128)
TOKENIZE_NUM_PROC = env_int("TOKENIZE_NUM_PROC", 4)
TOKENIZED_CACHE_DIR = os.getenv("TOKENIZED_CACHE_DIR", "./tokenized_cache")


def get_global(name: str) -> Any | None:
    return globals().get(name)


def get_texts_df() -> Any:
    existing = get_global("texts_df")
    if existing is not None:
        print("[INFO] Reusing texts_df from notebook globals.")
        return existing
    print("[INFO] texts_df not found in globals; importing load_data.py.")
    from load_data import texts_df as loaded_texts_df
    return loaded_texts_df


def main() -> None:
    cache_path = Path(TOKENIZED_CACHE_DIR)
    if cache_path.exists():
        print(f"[INFO] Cache already exists at {cache_path}. Delete it to re-tokenize.")
        return

    print(f"[INFO] Loading tokenizer: {MODEL_NAME}")
    tokenizer = CamembertTokenizerFast.from_pretrained(MODEL_NAME)

    texts_df = get_texts_df()
    texts_df = (
        texts_df
        .select(
            pl.col("text")
            .cast(pl.Utf8, strict=False)
            .str.strip_chars()
            .alias("text")
        )
        .filter(pl.col("text").is_not_null() & (pl.col("text") != ""))
    )
    print(f"[INFO] Total texts after cleaning: {len(texts_df):,}")

    raw_dataset = HFDataset.from_polars(texts_df)

    def tokenize_fn(examples: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )

    print("[INFO] Tokenizing dataset...")
    tokenized = raw_dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=max(1, TOKENIZE_NUM_PROC),
        remove_columns=["text"],
        desc="Tokenizing",
    )

    print(f"[INFO] Saving tokenized dataset to {cache_path}")
    tokenized.save_to_disk(str(cache_path))
    print(f"[INFO] Done. {len(tokenized):,} examples cached.")


if __name__ == "__main__":
    main()
