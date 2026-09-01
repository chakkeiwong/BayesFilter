"""TensorFlow wrapper for the BayesFilter symmetric Sylvester custom op."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import tensorflow as tf


_LIB_DIR = Path(os.environ.get("BAYESFILTER_OP_LIB_DIR", Path(__file__).parent))
_LIB_PATH = _LIB_DIR / "_symmetric_sylvester_ops.so"
try:
    _lib = tf.load_op_library(str(_LIB_PATH))
except Exception as exc:  # pragma: no cover - exercised when build is missing.
    raise ImportError(
        f"Could not load BayesFilter symmetric Sylvester op at {_LIB_PATH}. "
        "Build it with the BayesFilter CMakeLists.txt before importing this "
        f"module. Original error: {exc}"
    ) from exc


@dataclass(frozen=True)
class SymmetricSylvesterStatus:
    """Shape and backend status for a symmetric Sylvester request."""

    batch_size: int | None
    parameter_count: int | None
    dimension: int | None
    backend: str
    implemented: bool


def _static_dim(tensor: tf.Tensor, axis: int) -> int | None:
    dim = tensor.shape[axis]
    return None if dim is None else int(dim)


def _validate_shapes(
    symmetric_factor: tf.Tensor,
    rhs: tf.Tensor,
) -> SymmetricSylvesterStatus:
    if symmetric_factor.shape.rank != 3:
        raise ValueError("symmetric_factor must have shape [batch, n, n]")
    if rhs.shape.rank != 4:
        raise ValueError("rhs must have shape [batch, parameter, n, n]")

    batch = _static_dim(symmetric_factor, 0)
    n_rows = _static_dim(symmetric_factor, 1)
    n_cols = _static_dim(symmetric_factor, 2)
    rhs_batch = _static_dim(rhs, 0)
    parameter_count = _static_dim(rhs, 1)
    rhs_rows = _static_dim(rhs, 2)
    rhs_cols = _static_dim(rhs, 3)

    if n_rows is not None and n_cols is not None and n_rows != n_cols:
        raise ValueError("symmetric_factor must be square on its trailing axes")
    if rhs_rows is not None and rhs_cols is not None and rhs_rows != rhs_cols:
        raise ValueError("rhs must be square on its trailing axes")
    if batch is not None and rhs_batch is not None and batch != rhs_batch:
        raise ValueError("symmetric_factor and rhs batch dimensions must match")
    if n_rows is not None and rhs_rows is not None and n_rows != rhs_rows:
        raise ValueError("symmetric_factor and rhs matrix dimensions must match")

    return SymmetricSylvesterStatus(
        batch_size=batch,
        parameter_count=parameter_count,
        dimension=n_rows,
        backend="compiled_symmetric_sylvester",
        implemented=True,
    )


def symmetric_sylvester_solve(
    symmetric_factor: tf.Tensor,
    rhs: tf.Tensor,
    *,
    backend: str = "mkl_symmetric_sylvester",
) -> tf.Tensor:
    """Solve ``S X + X S = RHS`` for batched symmetric ``S``.

    Parameters
    ----------
    symmetric_factor:
        Tensor with shape ``[batch, n, n]``.  Later phases require this to be
        the principal square root used by the sigma-point placement backend.
    rhs:
        Tensor with shape ``[batch, parameter, n, n]``.
    backend:
        Reserved backend selector.  The only planned implementation backend is
        a BayesFilter-owned compiled custom op.

    Returns
    -------
    Tensor
        Tensor with shape ``[batch, parameter, n, n]``.

    Notes
    -----
    This wrapper assumes finite symmetric positive-definite ``symmetric_factor``
    and finite ``rhs``.  Regularization and covariance repair policy belong to
    the placement backend that calls this primitive.
    """

    symmetric_factor = tf.cast(tf.convert_to_tensor(symmetric_factor), tf.float64)
    rhs = tf.cast(tf.convert_to_tensor(rhs), tf.float64)
    _validate_shapes(symmetric_factor, rhs)
    if backend != "mkl_symmetric_sylvester":
        raise ValueError(f"unknown symmetric Sylvester backend: {backend!r}")
    return _lib.symmetric_sylvester(s=symmetric_factor, rhs=rhs)
