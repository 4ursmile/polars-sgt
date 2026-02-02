"""Tests for SGT (Sequence Graph Transform) function."""

from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

import polars_sgt as xdt


def test_sgt_basic_unigrams() -> None:
    """Test basic unigram generation."""
    df = pl.DataFrame({
        "user_id": ["A", "A", "A", "B", "B"],
        "action": ["login", "view", "purchase", "login", "view"],
    })
    
    result = df.select(
        xdt.sgt_transform("user_id", "action", kappa=1).alias("sgt")
    )
    
    assert result.shape[0] == 2  # Two unique sequences (A and B)
    assert "sgt" in result.columns


def test_sgt_bigrams() -> None:
    """Test bigram generation."""
    df = pl.DataFrame({
        "seq_id": ["1", "1", "1", "1"],
        "state": ["A", "B", "A", "C"],
    })
    
    result = df.select(
        xdt.sgt_transform("seq_id", "state", kappa=2, mode="none").alias("sgt")
    )
    
    # Should have both unigrams and bigrams
    ngrams = result.select(
        pl.col("sgt").struct.field("ngram_keys")
    ).to_series().to_list()[0]
    
    assert "A" in ngrams  # Unigram
    assert "A -> B" in ngrams  # Bigram


def test_sgt_with_numeric_time() -> None:
    """Test SGT with numeric time column."""
    df = pl.DataFrame({
        "user_id": ["U1", "U1", "U1"],
        "action": ["start", "middle", "end"],
        "timestamp": [0, 10, 20],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "user_id",
            "action",
            time_col="timestamp",
            kappa=2,
            time_penalty="inverse",
            alpha=1.0,
        ).alias("sgt")
    )
    
    assert result.shape[0] == 1


def test_sgt_with_datetime() -> None:
    """Test SGT with datetime column."""
    df = pl.DataFrame({
        "session_id": ["S1", "S1", "S1"],
        "event": ["start", "click", "end"],
        "time": [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 0, 10),
        ],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "session_id",
            "event",
            time_col="time",
            deltatime="m",
            kappa=2,
        ).alias("sgt")
    )
    
    assert result.shape[0] == 1


def test_sgt_time_penalty_exponential() -> None:
    """Test exponential time penalty."""
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["A", "B", "C"],
        "time": [0, 1, 2],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            time_col="time",
            kappa=2,
            time_penalty="exponential",
            alpha=0.5,
            mode="none",
        ).alias("sgt")
    )
    
    weights = result.select(
        pl.col("sgt").struct.field("value")
    ).to_series().to_list()[0]
    
    # Weights should be positive
    assert all(w > 0 for w in weights)


def test_sgt_time_penalty_linear() -> None:
    """Test linear time penalty."""
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["X", "Y", "Z"],
        "time": [0, 1, 5],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            time_col="time",
            kappa=2,
            time_penalty="linear",
            alpha=0.1,
            mode="none",
        ).alias("sgt")
    )
    
    assert result.shape[0] == 1


def test_sgt_time_penalty_power() -> None:
    """Test power time penalty."""
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["P", "Q", "R"],
        "time": [0, 2, 4],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            time_col="time",
            kappa=2,
            time_penalty="power",
            beta=2.0,
            mode="none",
        ).alias("sgt")
    )
    
    assert result.shape[0] == 1


def test_sgt_time_penalty_none() -> None:
    """Test no time penalty."""
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["A", "B", "C"],
        "time": [0, 100, 1000],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            time_col="time",
            kappa=2,
            time_penalty="none",
            mode="none",
        ).alias("sgt")
    )
    
    # With no penalty, all weights should be integer counts
    weights = result.select(
        pl.col("sgt").struct.field("value")
    ).to_series().to_list()[0]
    
    assert all(w > 0 for w in weights)


def test_sgt_l1_normalization() -> None:
    """Test L1 normalization."""
    df = pl.DataFrame({
        "id": ["1", "1", "1", "1"],
        "state": ["A", "B", "A", "B"],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            kappa=2,
            mode="l1",
        ).alias("sgt")
    )
    
    weights = result.select(
        pl.col("sgt").struct.field("value")
    ).to_series().to_list()[0]
    
    # L1 normalization: sum should be 1.0
    assert abs(sum(weights) - 1.0) < 1e-10


def test_sgt_l2_normalization() -> None:
    """Test L2 normalization."""
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["X", "Y", "Z"],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            kappa=2,
            mode="l2",
        ).alias("sgt")
    )
    
    weights = result.select(
        pl.col("sgt").struct.field("value")
    ).to_series().to_list()[0]
    
    # L2 normalization: sum of squares should be 1.0
    sum_of_squares = sum(w * w for w in weights)
    assert abs(sum_of_squares - 1.0) < 1e-10


def test_sgt_length_sensitive() -> None:
    """Test length-sensitive normalization."""
    df = pl.DataFrame({
        "id": ["1", "1", "1", "1", "1"],
        "state": ["A", "B", "C", "D", "E"],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            kappa=2,
            length_sensitive=True,
            mode="none",
        ).alias("sgt")
    )
    
    weights = result.select(
        pl.col("sgt").struct.field("value")
    ).to_series().to_list()[0]
    
    # With length normalization, weights should be divided by sequence length
    assert all(w > 0 for w in weights)


def test_sgt_multiple_sequences() -> None:
    """Test SGT with multiple sequences."""
    df = pl.DataFrame({
        "user": ["A", "A", "A", "B", "B", "C", "C", "C", "C"],
        "event": ["e1", "e2", "e3", "e1", "e2", "e4", "e5", "e6", "e7"],
    })
    
    result = df.select(
        xdt.sgt_transform("user", "event", kappa=2).alias("sgt")
    )
    
    # Should have 3 rows (one per unique user)
    assert result.shape[0] == 3
    
    # Check that sequence_ids are correct
    seq_ids = result.select(
        pl.col("sgt").struct.field("sequence_id")
    ).to_series().to_list()
    
    assert set(seq_ids) == {"A", "B", "C"}


def test_sgt_trigrams() -> None:
    """Test trigram generation (kappa=3)."""
    df = pl.DataFrame({
        "id": ["1", "1", "1", "1"],
        "state": ["A", "B", "C", "D"],
    })
    
    result = df.select(
        xdt.sgt_transform("id", "state", kappa=3, mode="none").alias("sgt")
    )
    
    ngrams = result.select(
        pl.col("sgt").struct.field("ngram_keys")
    ).to_series().to_list()[0]
    
    # Should have unigrams, bigrams, and trigrams
    assert "A" in ngrams
    assert "A -> B" in ngrams
    assert "A -> B -> C" in ngrams


def test_sgt_empty_sequence() -> None:
    """Test handling of empty sequences."""
    # Testing with a dataframe that has no rows for a sequence is tricky
    # This test just ensures no crash with minimal data
    df = pl.DataFrame({
        "id": ["1"],
        "state": ["A"],
    })
    
    result = df.select(
        xdt.sgt_transform("id", "state", kappa=1).alias("sgt")
    )
    
    assert result.shape[0] == 1


def test_sgt_struct_output() -> None:
    """Test that output is a proper struct with expected fields."""
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["A", "B", "C"],
    })
    
    result = df.select(
        xdt.sgt_transform("id", "state", kappa=2).alias("sgt")
    )
    
    # Extract fields
    expanded = result.select([
        pl.col("sgt").struct.field("sequence_id").alias("seq_id"),
        pl.col("sgt").struct.field("ngram_keys").alias("keys"),
        pl.col("sgt").struct.field("value").alias("values"),
    ])
    
    assert "seq_id" in expanded.columns
    assert "keys" in expanded.columns
    assert "values" in expanded.columns
    
    # Keys and values should be lists
    keys = expanded["keys"].to_list()[0]
    values = expanded["values"].to_list()[0]
    
    assert isinstance(keys, list)
    assert isinstance(values, list)
    assert len(keys) == len(values)


def test_sgt_explode_pattern() -> None:
    """Test the common pattern of exploding n-grams."""
    df = pl.DataFrame({
        "user": ["A", "A", "A"],
        "action": ["login", "view", "logout"],
    })
    
    result = df.select(
        xdt.sgt_transform("user", "action", kappa=2).alias("sgt")
    )
    
    # Explode the results
    exploded = result.select([
        pl.col("sgt").struct.field("sequence_id"),
        pl.col("sgt").struct.field("ngram_keys").alias("ngram"),
        pl.col("sgt").struct.field("value").alias("weight"),
    ]).explode(["ngram", "weight"])
    
    assert exploded.shape[0] > 0
    assert "ngram" in exploded.columns
    assert "weight" in exploded.columns


def test_sgt_with_date_column() -> None:
    """Test SGT with date column."""
    from datetime import date
    df = pl.DataFrame({
        "id": ["1", "1", "1"],
        "state": ["A", "B", "C"],
        "date": [
            date(2024, 1, 1),
            date(2024, 1, 2),
            date(2024, 1, 3),
        ],
    })
    
    result = df.select(
        xdt.sgt_transform(
            "id",
            "state",
            time_col="date",
            deltatime="d",
            kappa=2,
        ).alias("sgt")
    )
    
    assert result.shape[0] == 1
