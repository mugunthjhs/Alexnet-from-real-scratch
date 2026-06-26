"""
Data type utilities for Tensor operations.
Handles dtype conversion, validation, and compatibility checks.
"""

import numpy as np
from typing import Union, Tuple, Optional, List

# Supported dtypes
DTYPE_FLOAT32 = np.float32
DTYPE_FLOAT64 = np.float64
DTYPE_INT32 = np.int32
DTYPE_INT64 = np.int64
DTYPE_UINT8 = np.uint8

FLOAT_TYPES = (np.float32, np.float64)
INT_TYPES = (np.int32, np.int64, np.int16)
NUMERIC_TYPES = FLOAT_TYPES + INT_TYPES

DTYPE_MAP = {
    'float32': np.float32,
    'float64': np.float64,
    'int32': np.int32,
    'int64': np.int64,
    'uint8': np.uint8,
}


def is_float(dtype: np.dtype) -> bool:
    """Check if dtype is a floating-point type."""
    return dtype in FLOAT_TYPES or np.issubdtype(dtype, np.floating)


def is_int(dtype: np.dtype) -> bool:
    """Check if dtype is an integer type."""
    return dtype in INT_TYPES or np.issubdtype(dtype, np.integer)


def is_numeric(dtype: np.dtype) -> bool:
    """Check if dtype is numeric (float or int)."""
    return is_float(dtype) or is_int(dtype)


def validate_dtype(dtype: Union[str, np.dtype]) -> np.dtype:
    """Validate and convert dtype string to numpy dtype."""
    if isinstance(dtype, str):
        if dtype not in DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype: {dtype}. "
                f"Supported: {list(DTYPE_MAP.keys())}"
            )
        return DTYPE_MAP[dtype]
    if isinstance(dtype, type):
        return np.dtype(dtype)
    return dtype


def promote_dtype(*dtypes: np.dtype) -> np.dtype:
    """
    Promote multiple dtypes to a common dtype.

    Rules:
    - float64 > float32 > int64 > int32
    - Floats > Ints
    """
    if not dtypes:
        return np.float32

    dtypes = [validate_dtype(d) for d in dtypes]

    float_types = [d for d in dtypes if is_float(d)]
    int_types = [d for d in dtypes if is_int(d)]

    if float_types:
        return max(float_types, key=lambda d: 8 if d == np.float64 else 4)

    if int_types:
        return max(int_types, key=lambda d: 8 if d == np.int64 else 4)

    return np.float32


def cast_array(array: np.ndarray, dtype: Union[str, np.dtype]) -> np.ndarray:
    """Cast array to target dtype safely."""
    target_dtype = validate_dtype(dtype)
    if array.dtype == target_dtype:
        return array
    return array.astype(target_dtype)


def ensure_float(array: np.ndarray, dtype: np.dtype = np.float32) -> np.ndarray:
    """Ensure array is floating-point type."""
    if is_float(array.dtype):
        return array
    return cast_array(array, dtype)


def ensure_int(array: np.ndarray, dtype: np.dtype = np.int32) -> np.ndarray:
    """Ensure array is integer type."""
    if is_int(array.dtype):
        return array
    return cast_array(array, dtype)


def check_dtype_compatibility(
    *arrays: np.ndarray,
    allow_cast: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Check if arrays have compatible dtypes for operations.

    Returns:
        (is_compatible, error_message)
    """
    if not arrays:
        return True, None

    dtypes = [a.dtype for a in arrays]

    all_numeric = all(is_numeric(d) for d in dtypes)
    if not all_numeric:
        msg = f"Not all dtypes are numeric: {dtypes}"
        return False, msg

    if not allow_cast:
        return len(set(dtypes)) <= 1, f"Dtype mismatch: {dtypes}"

    return True, None


def broadcast_dtypes(*dtypes: np.dtype) -> np.dtype:
    """Broadcast multiple dtypes to result dtype."""
    return promote_dtype(*dtypes)


def get_dtype_info(dtype: Union[str, np.dtype]) -> dict:
    """Get information about a dtype."""
    dtype = validate_dtype(dtype)
    dt = np.dtype(dtype)

    return {
        'dtype': str(dtype),
        'kind': dt.kind,
        'itemsize': dt.itemsize,
        'is_float': is_float(dtype),
        'is_int': is_int(dtype),
        'is_numeric': is_numeric(dtype),
    }


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Perform safe division with epsilon to avoid division by zero."""
    denom = np.where(
        np.abs(denominator) < eps,
        eps,
        denominator
    )
    return numerator / denom


def clip_to_dtype(array: np.ndarray, dtype: Union[str, np.dtype]) -> np.ndarray:
    """Clip array values to valid range for target dtype."""
    dtype = validate_dtype(dtype)
    dt = np.dtype(dtype)

    if np.issubdtype(dt, np.floating):
        return array

    info = np.iinfo(dt)
    return np.clip(array, info.min, info.max)


class DtypeContext:
    """Context manager for temporary dtype operations."""

    def __init__(self, default_dtype: Union[str, np.dtype] = 'float32'):
        self.default_dtype = validate_dtype(default_dtype)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def cast(self, array: np.ndarray) -> np.ndarray:
        """Cast array to context's default dtype."""
        return cast_array(array, self.default_dtype)
