"""TensorFlow dtype helpers for linear Gaussian backends."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import tensorflow as tf


SUPPORTED_FLOAT_DTYPES = (tf.float32, tf.float64)


def _as_dtype_or_none(value: Any) -> tf.DType | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    try:
        return tf.as_dtype(dtype)
    except TypeError:
        return None


def common_floating_dtype(
    *values: Any,
    default: tf.DType = tf.float64,
    allowed: Iterable[tf.DType] = SUPPORTED_FLOAT_DTYPES,
    context: str = "floating tensor inputs",
) -> tf.DType:
    """Return the shared floating dtype of explicitly typed inputs.

    Python literals and untyped containers do not determine the dtype.  This
    preserves the historical default for callers that pass only literals while
    allowing TensorFlow tensors, variables, tensor specs, and NumPy arrays with
    explicit floating dtypes to select the computation dtype.
    """

    default_dtype = tf.as_dtype(default)
    allowed_dtypes = tuple(tf.as_dtype(dtype) for dtype in allowed)
    observed: list[tf.DType] = []
    for value in values:
        dtype = _as_dtype_or_none(value)
        if dtype is None or not dtype.is_floating:
            continue
        if dtype not in allowed_dtypes:
            allowed_names = ", ".join(dtype.name for dtype in allowed_dtypes)
            raise TypeError(
                f"{context} use unsupported dtype {dtype.name}; "
                f"supported dtypes are {allowed_names}"
            )
        if dtype not in observed:
            observed.append(dtype)
    if not observed:
        if default_dtype not in allowed_dtypes:
            allowed_names = ", ".join(dtype.name for dtype in allowed_dtypes)
            raise TypeError(
                f"{context} default dtype {default_dtype.name} is unsupported; "
                f"supported dtypes are {allowed_names}"
            )
        return default_dtype
    if len(observed) > 1:
        names = ", ".join(dtype.name for dtype in observed)
        raise TypeError(f"{context} must share one floating dtype; observed {names}")
    return observed[0]


def as_float_tensor(value: Any, dtype: tf.DType, name: str | None = None) -> tf.Tensor:
    """Convert ``value`` to a TensorFlow tensor with the chosen floating dtype."""

    dtype = tf.as_dtype(dtype)
    if dtype not in SUPPORTED_FLOAT_DTYPES:
        allowed_names = ", ".join(item.name for item in SUPPORTED_FLOAT_DTYPES)
        raise TypeError(
            f"{name or 'value'} requested unsupported dtype {dtype.name}; "
            f"supported dtypes are {allowed_names}"
        )
    return tf.convert_to_tensor(value, dtype=dtype, name=name)


__all__ = ["SUPPORTED_FLOAT_DTYPES", "as_float_tensor", "common_floating_dtype"]
