# polars-sgt

## Sequence Graph Transform for Polars



[![PyPI version](https://badge.fury.io/py/polars-sgt.svg)](https://badge.fury.io/py/polars-sgt)

Transform sequential data into powerful n-gram representations with [Polars](https://www.pola.rs/).

**polars-sgt** brings Sequence Graph Transform (SGT) to Polars, enabling you to:
- ✅ Transform sequences into weighted n-gram features
- ✅ Capture temporal patterns with time-based weighting
- ✅ Apply flexible normalization strategies (L1, L2, or none)
- ✅ Handle datetime, date, duration, and numeric time columns
- ✅ Blazingly fast, written in Rust
- ✅ Compatible with Polars lazy evaluation and streaming

## What is SGT?

Sequence Graph Transform converts sequential data (like user clickstreams, sensor readings, or transaction histories) into weighted n-gram representations. It captures:

- **Sequential patterns**: Unigrams, bigrams, trigrams, and higher-order n-grams
- **Temporal dynamics**: Time-based weighting with multiple decay functions
- **Normalized features**: L1/L2 normalization for comparable feature spaces

Perfect for:
- User behavior analysis
- Time series feature engineering
- Sequential pattern mining
- Anomaly detection in sequences

## Installation



Then install `polars-sgt`:

```console
pip install polars-sgt
```

## Quick Start

### Basic Example

```python
import polars as pl
import polars_sgt as sgt

# User clickstream data
df = pl.DataFrame({
    "user_id": [1, 1, 1, 2, 2, 2],
    "action": ["login", "view_product", "purchase", "login", "view_product", "logout"],
    "timestamp": [0, 10, 20, 0, 5, 15],
})

# Generate bigrams with exponential time decay
result = df.select(
    sgt.sgt_transform(
        "user_id",
        "action",
        time_col="timestamp",
        kappa=2,  # bigrams
        time_penalty="exponential",
        alpha=0.1,
        mode="l1"  # L1 normalization
    ).alias("sgt_features")
)

# Extract features
features = result.select([
    pl.col("sgt_features").struct.field("sequence_id"),
    pl.col("sgt_features").struct.field("ngram_keys").alias("ngrams"),
    pl.col("sgt_features").struct.field("ngram_values").alias("weights"),
]).explode(["ngrams", "weights"])

print(features)

#OR 
result = df.select(
    sgt.sgt_transform(
        "session_id",
        "event",
        time_col="time",
        deltatime="m",  # minutes
        kappa=3,  # trigrams
        time_penalty="inverse",
        mode="l2",
        alpha=0.5
    ).alias("struct_type")
)
out = (
    result
    .unnest("struct_type")
    .explode(["ngram_keys", "ngram_values"])
    .filter(pl.col("ngram_keys").str.split("->").list.len() > 0)
)
```

### With DateTime Columns

```python
from datetime import datetime

df = pl.DataFrame({
    "session_id": ["A", "A", "A", "A"],
    "event": ["start", "click", "scroll", "exit"],
    "time": [
        datetime(2024, 1, 1, 10, 0),
        datetime(2024, 1, 1, 10, 5),
        datetime(2024, 1, 1, 10, 7),
        datetime(2024, 1, 1, 10, 15),
    ],
})

result = df.select(
    sgt.sgt_transform(
        "session_id",
        "event",
        time_col="time",
        deltatime="m",  # minutes
        kappa=3,  # trigrams
        time_penalty="inverse",
    )
)
```

### Lazy Evaluation & Streaming

```python
result = (
    pl.scan_csv("large_sequences.csv")
    .with_columns(pl.col("timestamp").str.to_datetime())
    .select(
        sgt.sgt_transform(
            "user_id",
            "action",
            time_col="timestamp",
            kappa=2,
            deltatime="h",
        )
    )
    .collect(streaming=True)
)
```

## Parameters

### Required
- `sequence_id_col`: Column with sequence identifiers (groups)
- `state_col`: Column with state/event values

### Optional
- `time_col`: Timestamp column (datetime, date, duration, or numeric)
- `kappa`: Maximum n-gram size (default: 1)
  - 1 = unigrams only
  - 2 = unigrams + bigrams
  - 3 = unigrams + bigrams + trigrams, etc.
  
- `time_penalty`: Time decay function (default: "inverse")
  - `"inverse"`: weight = alpha / time_diff
  - `"exponential"`: weight = exp(-alpha × time_diff)
  - `"linear"`: weight = max(0, 1 - alpha × time_diff)
  - `"power"`: weight = 1 / time_diff^beta
  - `"none"`: No time penalty

- `mode`: Normalization mode (default: "l1")
  - `"l1"`: Sum of weights = 1
  - `"l2"`: L2 norm = 1
  - `"none"`: No normalization

- `length_sensitive`: Apply length normalization (default: False)
- `alpha`: Time penalty scale parameter (default: 1.0)
- `beta`: Power parameter for "power" penalty (default: 2.0)
- `deltatime`: Time unit for datetime columns
  - `"s"`, `"m"`, `"h"`, `"d"`, `"w"`, `"month"`, `"q"`, `"y"`

## Output

Returns a Struct with three fields:
- `sequence_id`: Original sequence identifier
- `ngram_keys`: List of n-gram strings (e.g., "login -> view -> purchase")
- `ngram_values`: List of corresponding weights

## Additional DateTime Utilities

While SGT is the primary focus, polars-sgt also includes helpful datetime utilities from the original polars-xdt:
- Timezone conversions
- Localized date formatting  
- Julian date conversion
- Month delta calculations

See the [full API documentation](https://github.com/Zedd-L/polars-sgt) for details.

## Author & Acknowledgments

**Author:** Zedd (lytran14789@gmail.com)

**Special Thanks:** This project is built upon [polars-xdt](https://github.com/MarcoGorelli/polars-xdt) 
created by [Marco Gorelli](https://github.com/MarcoGorelli). We are grateful for his excellent foundation.

## License

MIT
