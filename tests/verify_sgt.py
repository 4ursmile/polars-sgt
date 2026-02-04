
import polars as pl
import polars_sgt as xdt
import pytest
from datetime import date

def test_sgt_transform_df_no_group():
    df = pl.DataFrame({
        "user_id": [1, 1, 2, 2],
        "action": ["A", "B", "A", "C"],
        "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 3)]
    })
    
    result = xdt.sgt_transform_df(
        df,
        sequence_id_col="user_id",
        state_col="action",
        time_col="date",
        group_cols=None,
        use_tqdm=False
    )
    
    assert isinstance(result, pl.DataFrame)
    assert "user_id" in result.columns
    assert "A" in result.columns  # Assuming unigrams are generated as column names

def test_sgt_transform_df_with_group():
    df = pl.DataFrame({
        "group": ["X", "X", "Y", "Y"],
        "user_id": [1, 1, 2, 2],
        "action": ["A", "B", "A", "C"],
        "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 3)]
    })
    
    result = xdt.sgt_transform_df(
        df,
        sequence_id_col="user_id",
        state_col="action",
        time_col="date",
        group_cols="group",
        use_tqdm=False,
        group_name="test_group_"
    )
    
    assert isinstance(result, dict)
    assert "test_group_X" in result
    assert "test_group_Y" in result
    assert isinstance(result["test_group_X"], pl.DataFrame)

def test_sgt_transform_df_multi_seq_id():
    df = pl.DataFrame({
        "id_part1": ["u1", "u1", "u2", "u2"],
        "id_part2": ["01", "01", "02", "02"],
        "action": ["A", "B", "A", "C"],
        "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 3)]
    })
    
    result = xdt.sgt_transform_df(
        df,
        sequence_id_col=["id_part1", "id_part2"],
        state_col="action",
        time_col="date",
        group_cols=None,
        use_tqdm=False,
        keep_original_name=True
    )
    
    assert isinstance(result, pl.DataFrame)
    assert "id_part1" in result.columns
    assert "id_part2" in result.columns
    # Verify values are correct
    assert result.filter(pl.col("id_part1") == "u1").height > 0

def test_sgt_transform_df_keep_original_name_false():
    df = pl.DataFrame({
        "user_id": [1, 1, 2, 2],
        "action": ["A", "B", "A", "C"],
        "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1), date(2023, 1, 3)]
    })
    
    result = xdt.sgt_transform_df(
        df,
        sequence_id_col="user_id",
        state_col="action",
        time_col="date",
        group_cols=None,
        use_tqdm=False,
        keep_original_name=False
    )
    
    assert isinstance(result, pl.DataFrame)
    assert "sequence_id" in result.columns
    # user_id should NOT be in columns unless it's an alias, but function renames it if keep_original_name is True.
    # If False, it should stay as sequence_id (or whatever the pivot output is). 
    # Current implementation outputs "sequence_id" if keep_original_name is False.

if __name__ == "__main__":
    # Manually running tests
    test_sgt_transform_df_no_group()
    test_sgt_transform_df_with_group()
    test_sgt_transform_df_multi_seq_id()
    test_sgt_transform_df_keep_original_name_false()
    print("All verification tests passed!")
