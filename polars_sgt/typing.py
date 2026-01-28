from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Union

if sys.version_info >= (3, 10):
    from typing import TypeAlias
else:
    from typing_extensions import TypeAlias

if TYPE_CHECKING:
    import polars as pl
    from polars.type_aliases import PolarsDataType

IntoExprColumn: TypeAlias = Union["pl.Expr", "pl.Series", str]

__all__ = ["IntoExprColumn", "PolarsDataType"]
