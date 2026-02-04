# polars-sgt

## Sequence Graph Transform for Polars

[![PyPI version](https://badge.fury.io/py/polars-sgt.svg)](https://badge.fury.io/py/polars-sgt)

Transform sequential data into powerful n-gram representations with [Polars](https://www.pola.rs/).

**polars-sgt** brings Sequence Graph Transform (SGT) to Polars, enabling you to:
- ✅ **Transform** sequences into weighted n-gram features
- ✅ **Grouped Analysis**: Apply SGT across subsets (e.g., by direction, metric) and merge into a single wide DataFrame
- ✅ **Billion-Row Scale**: Optimized Rust implementation with O(1) time weight lookups
- ✅ **Temporal Dynamics**: Capture patterns with multiple decay functions across all n-gram transitions
- ✅ **Flexible**: Support for datetime, date, duration, and numeric time columns
- ✅ **Lazy & Parallel**: Fully compatible with Polars lazy evaluation and Rayon-backed parallel processing

## What is SGT?

Sequence Graph Transform converts sequential data (like user clickstreams, sensor readings, or transaction histories) into weighted n-gram representations. Unlike traditional n-grams, SGT captures:

- **Sequential patterns**: Multi-transition dependencies (Unigrams, bigrams, trigrams...)
- **Temporal dynamics**: Weights decay based on time gaps between events.
- **Normalized features**: L1/L2 normalization for machine-learning-ready feature spaces.

---

## Performance at Scale

Optimized for processing billions of rows:
- **O(1) Weight Calculation**: Uses cumulative product prefix arrays to calculate multi-transition time weights in constant time.
- **Zero-Cost Abstraction**: Written in Rust with Rayon for automatic multi-core utilization.
- **Memory Efficient**: Leverages Polars' arrow-backed memory management.

---

## Installation

```console
pip install polars-sgt
```

## Quick Start

### 1. High-Level API: `sgt_transform_df`

The `sgt_transform_df` function is the easiest way to generate SGT features. It handles unnesting, exploding, and pivoting into a wide format automatically.

#### Single Group (Default)
```python
import polars as pl
import polars_sgt as sgt

df = pl.DataFrame({
    "user_id": ["A", "A", "A", "B", "B"],
    "action": ["login", "view", "purchase", "login", "view"],
    "time": [1, 2, 10, 1, 5],
})

# Generate wide-format features merged into one DataFrame
features = sgt.sgt_transform_df(
    df, 
    sequence_id_col="user_id", 
    state_col="action", 
    time_col="time",
    kappa=2
)
```

#### Grouped Sequence Analysis
Calculate separate SGT features for different groups (e.g., event types or directions) and merge them into one wide DataFrame.

```python
# Calculate SGT features for each 'direction' and 'metric'
result = sgt.sgt_transform_df(
    df,
    sequence_id_col="user_id",
    state_col="action",
    time_col="time",
    group_cols=["direction", "metric"],
    kappa=3,
    time_penalty="exponential",
    alpha=0.7,
    group_name="analysis"
)
# Columns: ['user_id', 'analysis-buy-p_login', 'analysis-sell-p_login', ...]
```

### 2. Expression API: `sgt_transform`

For more control or integration into complex pipelines, use the expression-based API.

```python
# Basic expression usage (returns a struct)
result = df.select(
    sgt.sgt_transform(
        "user_id",
        "action",
        time_col="time",
        kappa=2,
        time_penalty="exponential",
        alpha=0.1,
        mode="l1"
    ).alias("sgt_features")
)

# Extract and explode
features = result.select([
    pl.col("sgt_features").struct.field("sequence_id"),
    pl.col("sgt_features").struct.field("ngram_keys").alias("ngrams"),
    pl.col("sgt_features").struct.field("value").alias("weights"),
]).explode(["ngrams", "weights"])
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
        deltatime="m",  # unit: minutes
        kappa=3,
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
    .collect(engine="streaming")
)
```

---

## API Reference

### `sgt.sgt_transform_df`
The recommended high-level entry point. Returns a wide-format DataFrame.

- `df`: Input DataFrame or LazyFrame.
- `sequence_id_col`: Column(s) identifying sequences.
- `state_col`: Column containing states/events.
- `time_col`: Optional timestamp column.
- `group_cols`: Optional column(s) to group by before SGT.
- `kappa`: Maximum n-gram size.
- `mode`: Normalization (`"l1"`, `"l2"`, `"none"`).
- `time_penalty`: Decay function (`"inverse"`, `"exponential"`, `"linear"`, `"power"`, `"none"`).

### `sgt.sgt_transform` (Expression)
Returns a struct with `sequence_id`, `ngram_keys`, and `value`.

```python
df.select(
    sgt.sgt_transform("user", "action", kappa=2).alias("sgt")
).unnest("sgt")
```

---

## Author & Acknowledgments

**Author:** Zedd (lytran14789@gmail.com)

**Special Thanks:** Built upon [polars-xdt](https://github.com/MarcoGorelli/polars-xdt) by [Marco Gorelli](https://github.com/MarcoGorelli).

## License

MIT
