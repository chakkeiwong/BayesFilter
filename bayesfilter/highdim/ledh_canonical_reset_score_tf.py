"""S6/S7: reset (Sinkhorn + Contract-E) with analytical tangent (Q1.1).

Single-cloud float64, single score direction. Semantic authorities:
- Sinkhorn barycentric form: the hand-derived batch value/JVP of the
  historical lane (recovered from git 43de3cb6^; the JVP there is the
  derivation reference — this module re-derives the same product-rule
  tangent for the single-cloud unrolled iterations).
- Contract-E restore: `ledh_contract_e_reset_tf._contract_e_chol_cloud_
  forward_core` primal semantics; tangents via the Phi-operator Cholesky
  differential and adjoint-solve differentials already oracle-gated in
  the score stages.
- Dual-cap correction (S7): `higher_moment_contract_e.higher_moment_
  shape_jvp` — the general implementation's own hand-derived tangents,
  called with the score direction's input tangents.

Validity note: the gap-eigenvalue check of the production value lane is a
VALIDITY diagnostic (eigvalsh), not part of the particle-value program;
the score lane records finiteness-based validity instead (eigvalsh has no
forward tangent here and never touches the value path). NO autodiff
(C-9).
"""

from __future__ import annotations

import tensorflow as tf

Tensor = tf.Tensor


def _chol_diff(chol: Tensor, d_matrix: Tensor) -> Tensor:
    inv_d = tf.linalg.triangular_solve(chol, d_matrix)
    inv_d_inv_t = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(chol, tf.linalg.matrix_transpose(inv_d))
    )
    lower = tf.linalg.band_part(inv_d_inv_t, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(inv_d_inv_t))
    return tf.linalg.matmul(chol, phi)


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def sinkhorn_contract_e_reset_with_tangent(
    children: Tensor,
    d_children: Tensor,
    weights: Tensor,
    d_weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> tuple[Tensor, Tensor]:
    """Return reset particles and their score-direction tangent.

    children [N, d]; weights [N] (normalized); design [N, d].
    """

    dtype = children.dtype
    n = tf.shape(children)[0]
    dim = tf.shape(children)[1]
    n_f = tf.cast(n, dtype)

    # --- Sinkhorn kernel (vendored form) with tangent
    deltas = children[:, None, :] - children[None, :, :]
    d_deltas = d_children[:, None, :] - d_children[None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=2)
    d_cost = 2.0 * tf.reduce_sum(deltas * d_deltas, axis=2)
    mean_cost = tf.reduce_mean(cost)
    floor = tf.cast(1.0e-3, dtype)
    cost_scale = tf.maximum(mean_cost, floor)
    d_cost_scale = tf.where(
        mean_cost > floor, tf.reduce_mean(d_cost), tf.zeros([], dtype)
    )
    eps_c = tf.cast(epsilon, dtype)
    exponent = -cost / (cost_scale * eps_c)
    d_exponent = -(d_cost / cost_scale - cost * d_cost_scale / tf.square(cost_scale)) / eps_c
    kernel = tf.exp(exponent)
    d_kernel = kernel * d_exponent

    uniform = tf.fill([n], 1.0 / n_f)
    left = tf.ones([n], dtype)
    right = tf.ones([n], dtype)
    d_left = tf.zeros([n], dtype)
    d_right = tf.zeros([n], dtype)
    tiny = tf.cast(1.0e-7, dtype)
    for _ in range(sinkhorn_steps + balance_steps):
        kr = tf.linalg.matvec(kernel, right) + tiny
        d_kr = tf.linalg.matvec(d_kernel, right) + tf.linalg.matvec(
            kernel, d_right
        )
        left_new = uniform / kr
        d_left_new = -uniform * d_kr / tf.square(kr)
        kl = tf.linalg.matvec(kernel, left_new, transpose_a=True) + tiny
        d_kl = tf.linalg.matvec(
            d_kernel, left_new, transpose_a=True
        ) + tf.linalg.matvec(kernel, d_left_new, transpose_a=True)
        right = weights / kl
        d_right = d_weights / kl - weights * d_kl / tf.square(kl)
        left, d_left = left_new, d_left_new

    coupling = left[:, None] * kernel * right[None, :]
    d_coupling = (
        d_left[:, None] * kernel * right[None, :]
        + left[:, None] * d_kernel * right[None, :]
        + left[:, None] * kernel * d_right[None, :]
    )
    row_mass = tf.reduce_sum(coupling, axis=1)
    d_row_mass = tf.reduce_sum(d_coupling, axis=1)
    numer = tf.linalg.matmul(coupling, children)
    d_numer = tf.linalg.matmul(d_coupling, children) + tf.linalg.matmul(
        coupling, d_children
    )
    barycentric = numer / row_mass[:, None]
    d_barycentric = (
        d_numer / row_mass[:, None]
        - numer * d_row_mass[:, None] / tf.square(row_mass)[:, None]
    )

    # --- Contract-E restore with tangent
    target_mean = tf.reduce_sum(weights[:, None] * children, axis=0)
    d_target_mean = tf.reduce_sum(
        d_weights[:, None] * children + weights[:, None] * d_children, axis=0
    )
    centered_src = children - target_mean[None, :]
    d_centered_src = d_children - d_target_mean[None, :]
    target_cov = _sym(
        tf.einsum("n,ni,nj->ij", weights, centered_src, centered_src)
    )
    d_target_cov = _sym(
        tf.einsum("n,ni,nj->ij", d_weights, centered_src, centered_src)
        + tf.einsum("n,ni,nj->ij", weights, d_centered_src, centered_src)
        + tf.einsum("n,ni,nj->ij", weights, centered_src, d_centered_src)
    )
    eye = tf.eye(dim, dtype=dtype)
    ridge_eye = tf.cast(ridge, dtype) * eye

    plus_mean = tf.reduce_mean(barycentric, axis=0)
    d_plus_mean = tf.reduce_mean(d_barycentric, axis=0)
    centered_plus = barycentric - plus_mean[None, :]
    d_centered_plus = d_barycentric - d_plus_mean[None, :]
    plus_cov = _sym(
        tf.einsum("ni,nj->ij", centered_plus, centered_plus) / n_f
    )
    d_plus_cov = _sym(
        (
            tf.einsum("ni,nj->ij", d_centered_plus, centered_plus)
            + tf.einsum("ni,nj->ij", centered_plus, d_centered_plus)
        )
        / n_f
    )
    gap = _sym(target_cov - plus_cov) + ridge_eye
    d_gap = _sym(d_target_cov - d_plus_cov)
    gap_chol = tf.linalg.cholesky(gap)
    d_gap_chol = _chol_diff(gap_chol, d_gap)

    injected = barycentric + tf.linalg.matmul(
        design, gap_chol, transpose_b=True
    )
    d_injected = d_barycentric + tf.linalg.matmul(
        design, d_gap_chol, transpose_b=True
    )
    inj_mean = tf.reduce_mean(injected, axis=0)
    d_inj_mean = tf.reduce_mean(d_injected, axis=0)
    centered_inj = injected - inj_mean[None, :]
    d_centered_inj = d_injected - d_inj_mean[None, :]
    inj_cov = _sym(
        tf.einsum("ni,nj->ij", centered_inj, centered_inj) / n_f
    ) + ridge_eye
    d_inj_cov = _sym(
        (
            tf.einsum("ni,nj->ij", d_centered_inj, centered_inj)
            + tf.einsum("ni,nj->ij", centered_inj, d_centered_inj)
        )
        / n_f
    )
    target_chol = tf.linalg.cholesky(target_cov + ridge_eye)
    d_target_chol = _chol_diff(target_chol, d_target_cov)
    inj_chol = tf.linalg.cholesky(inj_cov)
    d_inj_chol = _chol_diff(inj_chol, d_inj_cov)

    # affine^T = L_inj^{-T} @ target_chol^T  (adjoint triangular solve)
    rhs = tf.linalg.matrix_transpose(target_chol)
    d_rhs = tf.linalg.matrix_transpose(d_target_chol)
    solved = tf.linalg.triangular_solve(inj_chol, rhs, adjoint=True)
    # dX = L^{-T} (dB - dL^T X)
    d_solved = tf.linalg.triangular_solve(
        inj_chol,
        d_rhs - tf.linalg.matmul(d_inj_chol, solved, transpose_a=True),
        adjoint=True,
    )
    affine = tf.linalg.matrix_transpose(solved)
    d_affine = tf.linalg.matrix_transpose(d_solved)

    particles = target_mean[None, :] + tf.linalg.matmul(
        centered_inj, affine, transpose_b=True
    )
    d_particles = (
        d_target_mean[None, :]
        + tf.linalg.matmul(d_centered_inj, affine, transpose_b=True)
        + tf.linalg.matmul(centered_inj, d_affine, transpose_b=True)
    )
    return particles, d_particles


__all__ = ["sinkhorn_contract_e_reset_with_tangent"]
