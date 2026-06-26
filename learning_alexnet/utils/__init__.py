"""Utilities: data loading and data-type helpers.

Public API — import directly from the package, e.g.::

    from utils import DataLoader, validate_dtype, promote_dtype, cast_array
"""

from .dataloader import DataLoader

from .dtype_utils import (
    is_float,
    is_int,
    is_numeric,
    validate_dtype,
    promote_dtype,
    cast_array,
    ensure_float,
    ensure_int,
    check_dtype_compatibility,
    broadcast_dtypes,
    get_dtype_info,
    safe_divide,
    clip_to_dtype,
    DtypeContext,
)

__all__ = [
    "DataLoader",
    "is_float",
    "is_int",
    "is_numeric",
    "validate_dtype",
    "promote_dtype",
    "cast_array",
    "ensure_float",
    "ensure_int",
    "check_dtype_compatibility",
    "broadcast_dtypes",
    "get_dtype_info",
    "safe_divide",
    "clip_to_dtype",
    "DtypeContext",
]
