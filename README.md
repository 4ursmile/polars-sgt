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
The recommended high-level entry point. Automatically handles unnesting and pivoting into a wide-format DataFrame.

**Parameters:**
- `df`: Input Polars `DataFrame` or `LazyFrame`.
- `sequence_id_col`: String or list of strings. Identifies unique sequences (e.g., `user_id`). Multiple columns are concatenated for processing and can be restored in the output.
- `state_col`: String. Column containing the elements of the sequence (e.g., `action`).
- `time_col`: Optional string. Column containing timestamps or numeric time values.
- `group_cols`: Optional string or list of strings. Subsets to split data before applying SGT (e.g., `direction`, `metric`). Features from each group are prefixed and merged into a single wide DataFrame.
- `kappa`: Integer (default=`1`). Maximum n-gram length to consider. `kappa=1` for unigrams, `2` for bigrams, etc.
- `mode`: String (default=`"l1"`). Normalization strategy:
    - `"l1"`: Sum of weights equals 1.
    - `"l2"`: Euclidean norm of weights equals 1.
    - `"none"`: Raw cumulative weights.
- `time_penalty`: String (default=`"inverse"`). Decay function for temporal weighting:
    - `"inverse"`: `alpha / time_diff`
    - `"exponential"`: `exp(-alpha * time_diff)`
    - `"linear"`: `max(0, 1 - alpha * time_diff)`
    - `"power"`: `1 / time_diff^beta`
    - `"none"`: Ignores time intervals.
- `alpha`: Float (default=`1.0`). Scaling/decay rate for time penalties.
- `beta`: Float (default=`2.0`). Exponent for the `"power"` penalty.
- `deltatime`: String. Unit for temporal columns: `"s"`, `"m"`, `"h"`, `"d"`, `"w"`, `"month"`, `"q"`, `"y"`.
- `group_name`: String (default=`"sgt_"`). Prefix used for feature columns when `group_cols` is provided.
- `use_tqdm`: Boolean (default=`True`). Enable/disable progress bar during computation.
- `keep_original_name`: Boolean (default=`True`). Restore original column names if `sequence_id_col` was a list.
- `length_sensitive`: Boolean (default=`False`). Normalize weights by sequence length.

**Returns:**
- A wide `pl.DataFrame` containing sequence ID columns and n-gram feature columns.

---

### `sgt.sgt_transform` (Expression)
Polars plugin expression for use within `select`, `with_columns`, or `group_by`.

**Parameters:**
Identical to `sgt_transform_df`, but returns a `pl.Struct` column.

**Output Struct Fields:**
- `sequence_id`: Identifier for the sequence.
- `ngram_keys`: List of n-gram labels (e.g., `"A -> B"`).
- `value`: List of corresponding n-gram weights.

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
