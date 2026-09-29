"""Direct joint-stack QR for square-root conditional updates.

The kernel factors a joint observation/state residual stack directly.  The
lower-right triangular block is the conditional state factor, so the runtime
does not form a Schur-complement covariance or apply covariance downdates.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf

from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower


def _as_float(value: Any, name: str) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(value, name=name), tf.float64)
    tf.debugging.assert_all_finite(tensor, f"{name} contains NaN or Inf")
    return tensor


def _right_lower_solve(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Solve ``X factor = rhs`` for a lower triangular factor."""

    solved = tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(factor),
        tf.linalg.matrix_transpose(rhs),
        lower=False,
    )
    return tf.linalg.matrix_transpose(solved)


def _right_lower_solve_batch(factor: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
    """Batched version for ``rhs`` with a parameter axis."""

    if rhs.shape.rank != 4 or factor.shape.rank != 3:
        raise ValueError("batch solve expects factor [B,N,N] and rhs [B,P,M,N]")
    b, p, m, n = rhs.shape.as_list()
    if None in (b, p, m, n) or factor.shape.as_list() != [b, n, n]:
        raise ValueError("batch solve dimensions are incompatible")
    repeated = tf.repeat(factor, p, axis=0)
    flat_rhs = tf.reshape(rhs, [b * p, m, n])
    solved = tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(repeated),
        tf.linalg.matrix_transpose(flat_rhs),
        lower=False,
    )
    return tf.reshape(tf.linalg.matrix_transpose(solved), [b, p, m, n])


def batched_block_qr_conditional(
    observation_stack: tf.Tensor,
    state_stack: tf.Tensor,
    observation_derivative_stack: tf.Tensor | None = None,
    state_derivative_stack: tf.Tensor | None = None,
    *,
    compute_covariance_diagnostics: bool = True,
    relative_pivot_tolerance: float = 1.0e-12,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Factor a joint residual stack and return innovation/gain/conditional factors.

    ``observation_stack`` has shape ``[B, Ny, K]`` and ``state_stack`` has
    shape ``[B, Nx, K]``.  Both stacks share the same point columns; callers
    append zero-padded observation-noise columns to the state stack.  The
    returned factors are ``Ly`` with ``S=Ly Ly.T``, ``K`` and ``Lf`` with the
    conditional covariance ``Lf Lf.T``.  Derivative stacks have shape
    ``[B, P, dimension, K]``.
    """

    observation_stack = _as_float(observation_stack, "observation_stack")
    state_stack = _as_float(state_stack, "state_stack")
    if observation_stack.shape.rank != 3 or state_stack.shape.rank != 3:
        raise ValueError("stacks must have shape [B, dimension, columns]")
    b, ny, k = observation_stack.shape.as_list()
    bs, nx, ks = state_stack.shape.as_list()
    if None in (b, ny, nx, k) or (b, ny, nx, k) != (bs, ny, nx, ks):
        raise ValueError("observation/state stack dimensions are incompatible")
    if k < ny + nx:
        raise ValueError("joint stack must have at least Ny+Nx columns")

    joint = tf.concat([observation_stack, state_stack], axis=1)
    d_joint = None
    if (observation_derivative_stack is None) != (state_derivative_stack is None):
        raise ValueError("both derivative stacks must be supplied together")
    if observation_derivative_stack is not None:
        observation_derivative_stack = _as_float(
            observation_derivative_stack, "observation_derivative_stack"
        )
        state_derivative_stack = _as_float(state_derivative_stack, "state_derivative_stack")
        if observation_derivative_stack.shape.rank != 4 or state_derivative_stack.shape.rank != 4:
            raise ValueError("derivative stacks must have shape [B,P,dimension,columns]")
        if observation_derivative_stack.shape[0] != b or state_derivative_stack.shape[0] != b:
            raise ValueError("derivative batch dimensions do not match stacks")
        if observation_derivative_stack.shape[2:] != (ny, k) or state_derivative_stack.shape[2:] != (nx, k):
            raise ValueError("derivative spatial dimensions do not match stacks")
        if observation_derivative_stack.shape[1] != state_derivative_stack.shape[1]:
            raise ValueError("derivative parameter dimensions do not match")
        d_joint = tf.concat([observation_derivative_stack, state_derivative_stack], axis=2)

    factor, d_factor, qr = batched_stack_qr_lower(
        joint,
        d_joint,
        compute_covariance_diagnostics=compute_covariance_diagnostics,
        relative_pivot_tolerance=relative_pivot_tolerance,
    )
    ly = factor[:, :ny, :ny]
    lxy = factor[:, ny:, :ny]
    lf = factor[:, ny:, ny:]
    gain = _right_lower_solve(ly, lxy)

    d_ly = d_lxy = d_lf = d_gain = None
    if d_factor is not None:
        d_ly = d_factor[:, :, :ny, :ny]
        d_lxy = d_factor[:, :, ny:, :ny]
        d_lf = d_factor[:, :, ny:, ny:]
        d_gain = _right_lower_solve_batch(
            ly,
            d_lxy - tf.einsum("bik,bpkj->bpij", gain, d_ly),
        )

    diagnostics = dict(qr)
    diagnostics.update({
        "minimum_innovation_pivot": tf.reduce_min(tf.linalg.diag_part(ly), axis=-1),
        "minimum_conditional_pivot": tf.reduce_min(tf.linalg.diag_part(lf), axis=-1),
    })
    if compute_covariance_diagnostics:
        diagnostics["conditional_factor_reconstruction_residual"] = tf.linalg.norm(
            lf @ tf.linalg.matrix_transpose(lf)
            - tf.einsum("bij,bkj->bik", state_stack, state_stack)
            + tf.einsum("bij,bkj->bik", lxy, lxy),
            axis=[-2, -1],
        )
    else:
        diagnostics["conditional_factor_reconstruction_residual"] = diagnostics[
            "factor_reconstruction_residual"
        ]
    if d_factor is not None:
        if compute_covariance_diagnostics:
            d_cond = tf.einsum("bpik,bkj->bpij", state_derivative_stack, tf.linalg.matrix_transpose(state_stack))
            d_cond += tf.einsum("bik,bpkj->bpij", state_stack, tf.linalg.matrix_transpose(state_derivative_stack))
            d_cond -= tf.einsum("bpik,bkj->bpij", d_lxy, tf.linalg.matrix_transpose(lxy))
            d_cond -= tf.einsum("bik,bpkj->bpij", lxy, tf.linalg.matrix_transpose(d_lxy))
            d_reconstructed = tf.einsum("bpik,bkj->bpij", d_lf, tf.linalg.matrix_transpose(lf))
            d_reconstructed += tf.einsum("bik,bpkj->bpij", lf, tf.linalg.matrix_transpose(d_lf))
            diagnostics["conditional_factor_derivative_reconstruction_residual"] = tf.linalg.norm(
                d_reconstructed - d_cond, axis=[-2, -1]
            )
        else:
            diagnostics["conditional_factor_derivative_reconstruction_residual"] = diagnostics[
                "factor_derivative_reconstruction_residual"
            ]
    return ly, gain, lf, d_ly, d_gain, d_lf, diagnostics


__all__ = ["batched_block_qr_conditional"]
