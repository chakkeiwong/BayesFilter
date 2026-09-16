"""TensorFlow wrapper for the BayesFilter symmetric principal-sqrt custom op."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.ops.symmetric_sylvester_tf import _lib


@dataclass(frozen=True)
class SymmetricPrincipalSqrtStatus:
    """Shape and backend status for a symmetric principal-sqrt request."""

    batch_size: int | None
    dimension: int | None
    backend: str
    implemented: bool


def _static_dim(tensor: tf.Tensor, axis: int) -> int | None:
    dim = tensor.shape[axis]
    return None if dim is None else int(dim)


def _validate_shapes(covariance: tf.Tensor) -> SymmetricPrincipalSqrtStatus:
    if covariance.shape.rank != 3:
        raise ValueError("covariance must have shape [batch, n, n]")

    batch = _static_dim(covariance, 0)
    n_rows = _static_dim(covariance, 1)
    n_cols = _static_dim(covariance, 2)
    if n_rows is not None and n_cols is not None and n_rows != n_cols:
        raise ValueError("covariance must be square on its trailing axes")

    return SymmetricPrincipalSqrtStatus(
        batch_size=batch,
        dimension=n_rows,
        backend="compiled_symmetric_principal_sqrt",
        implemented=True,
    )


def symmetric_principal_sqrt(
    covariance: tf.Tensor,
    *,
    backend: str = "mkl_symmetric_principal_sqrt",
) -> tf.Tensor:
    """Return the strict-SPD symmetric principal square root.

    The compiled op is the value path for principal-square-root sigma-point
    placement. It assumes finite symmetric positive-definite covariance
    matrices; covariance repair policy belongs to the caller.
    """

    covariance = tf.cast(tf.convert_to_tensor(covariance), tf.float64)
    _validate_shapes(covariance)
    if backend != "mkl_symmetric_principal_sqrt":
        raise ValueError(f"unknown symmetric principal-sqrt backend: {backend!r}")
    return _lib.symmetric_principal_sqrt(covariance=covariance)
