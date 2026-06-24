```py
import kagglehub
path = kagglehub.dataset_download("gwendaltsang/frenchyoutubecomments-frenchtweets")
```

```py
import polars as pl
pl.read_parquet(path)
```

```py
import polars as pl
from transformers import AutoTokenizer

texts_df = pl.read_parquet(path)
tokenizer = AutoTokenizer.from_pretrained('camembert-base')
```

```py
max_chunk_size = 126
min_last_chunk_size = 16
chunked_texts = []

for i in tqdm(range(0, len(texts), batch_size), desc="Chunking Texts"):
    batch = texts[i : i + batch_size]
    encodings = tokenizer(batch, truncation=False, add_special_tokens=False)["input_ids"]
    
    for tokens in encodings:
        n = len(tokens)
        
        # Case 1: Text is already very short (<= 5 tokens)
        if n <= min_last_chunk_size:
            chunked_texts.append(tokenizer.decode(tokens))
            
        # Case 2: Fits in one standard chunk
        elif n <= max_chunk_size:
            chunked_texts.append(tokenizer.decode(tokens))
            
        # Case 3: Requires splitting
        else:
            for j in range(0, n, max_chunk_size):
                start = j
                end = j + max_chunk_size
                
                # Check if this is the last chunk and if it's too small
                if end >= n:
                    remaining = n - start
                    if remaining < min_last_chunk_size:
                        # Sliding window: shift 'start' back to capture context and ensure 5 tokens
                        start = max(0, n - min_last_chunk_size)
                    chunk = tokens[start:n]
                    chunked_texts.append(tokenizer.decode(chunk))
                    break
                else:
                    # Normal chunk
                    chunk = tokens[start:end]
                    chunked_texts.append(tokenizer.decode(chunk))

print(f"Original number of texts: {len(texts)}")
print(f"Number of texts after chunking: {len(chunked_texts)}")
```

```py
import polars as pl

def get_token_counts(batch_texts):
    # Tokenize without special tokens to get 'content' length directly
    encodings = tokenizer(batch_texts, truncation=False, add_special_tokens=False)["input_ids"]
    return [len(tokens) for tokens in encodings]

# We process in batches to keep memory usage low while being much faster than .apply()
all_counts = []
batch_size = 5000
for i in tqdm(range(0, len(texts), batch_size), desc="Calculating Token Lengths"):
    batch = texts[i : i + batch_size]
    all_counts.extend(get_token_counts(batch))

# Add counts to the dataframe and filter
texts_df = texts_df.with_columns(n_tokens = pl.Series(all_counts))

original_count = texts_df.height
texts_df = texts_df.filter(pl.col("n_tokens") >= 6)
new_count = texts_df.height

print(f"Rows before filtering: {original_count}")
print(f"Rows after filtering (len >= 6): {new_count}")
print(f"Removed {original_count - new_count} rows.")

# Update the 'texts' list for any subsequent chunking steps
texts = texts_df["text"].to_list()
display(texts_df.head())
```
