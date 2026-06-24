from pathlib import Path
import polars as pl
import kagglehub

# Download Kaggle dataset
path = Path(kagglehub.dataset_download("gwendaltsang/frenchcomments-youtube-twitter"))

# Locate parquet file(s)
if path.is_file() and path.suffix == ".parquet":
    parquet_files = [path]
else:
    parquet_files = sorted(path.rglob("*.parquet"))

if not parquet_files:
    raise FileNotFoundError(f"No .parquet file found under: {path}")

# Read only the text column using Polars.
# Lazy scan is memory-friendly, then collect after filtering.
texts_df = (
    pl.concat(
        [
            pl.scan_parquet(str(parquet_file)).select("text")
            for parquet_file in parquet_files
        ]
    )
    .with_columns(
        pl.col("text").cast(pl.Utf8).str.strip_chars()
    )
    .filter(
        pl.col("text").is_not_null() & (pl.col("text") != "")
    )
    .collect()
)

print(f"Loaded {texts_df.height:,} text rows from {len(parquet_files)} parquet file(s).")
texts_df.head()
