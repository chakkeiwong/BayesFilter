"""Shared batched Contract-E reset and covariance-carry kernel.

Cloud reductions retain explicit B and tangent directions retain explicit K:
primal tensors are [B, N, ...], tangents are [K, B, N, ...]. Pointwise
flattening is deliberately absent from this reduction-bearing stage.
"""

from __future__ import annotations

import tensorflow as tf

Tensor = tf.Tensor


def _sym(value: Tensor) -> Tensor:
    return 0.5 * (value + tf.linalg.matrix_transpose(value))


def _chol_diff(chol: Tensor, d_matrix: Tensor) -> Tensor:
    """Cholesky JVP with chol [B,D,D] and tangent [K,B,D,D]."""
    chol_k = chol[None, ...]
    inv_d = tf.linalg.triangular_solve(chol_k, d_matrix)
    inv_d_inv_t = tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(chol_k, tf.linalg.matrix_transpose(inv_d))
    )
    lower = tf.linalg.band_part(inv_d_inv_t, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(inv_d_inv_t))
    return tf.linalg.matmul(chol_k, phi)


def _validate_shapes(
    children: Tensor,
    d_children: Tensor,
    covariances: Tensor,
    d_covariances: Tensor,
    weights: Tensor,
    d_weights: Tensor,
    design: Tensor,
) -> Tensor:
    if children.shape.rank != 3:
        raise ValueError("children must have shape [B,N,D]")
    if d_children.shape.rank != 4:
        raise ValueError("d_children must have shape [K,B,N,D]")
    if covariances.shape.rank != 4:
        raise ValueError("covariances must have shape [B,N,D,D]")
    if d_covariances.shape.rank != 5:
        raise ValueError("d_covariances must have shape [K,B,N,D,D]")
    if weights.shape.rank != 2 or d_weights.shape.rank != 3:
        raise ValueError("weights must be [B,N] and d_weights [K,B,N]")
    if design.shape.rank == 2:
        design = tf.broadcast_to(design[None, ...], tf.shape(children))
    elif design.shape.rank != 3:
        raise ValueError("design must have shape [N,D] or [B,N,D]")
    return design


def batched_sinkhorn_contract_e_reset_triple_with_tangent(
    children: Tensor,
    d_children: Tensor,
    covariances: Tensor,
    d_covariances: Tensor,
    weights: Tensor,
    d_weights: Tensor,
    design: Tensor,
    *,
    epsilon: float,
    sinkhorn_steps: int,
    balance_steps: int,
    ridge: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Reset B clouds for K tangent directions without cross-row reductions."""
    children = tf.convert_to_tensor(children)
    dtype = children.dtype
    d_children = tf.cast(d_children, dtype)
    covariances = tf.cast(covariances, dtype)
    d_covariances = tf.cast(d_covariances, dtype)
    weights = tf.cast(weights, dtype)
    d_weights = tf.cast(d_weights, dtype)
    design = _validate_shapes(
        children, d_children, covariances, d_covariances,
        weights, d_weights, tf.cast(design, dtype),
    )
    n = tf.shape(children)[1]
    dim = tf.shape(children)[2]
    n_f = tf.cast(n, dtype)

    deltas = children[:, :, None, :] - children[:, None, :, :]
    d_deltas = d_children[:, :, :, None, :] - d_children[:, :, None, :, :]
    cost = tf.reduce_sum(tf.square(deltas), axis=3)
    d_cost = 2.0 * tf.reduce_sum(deltas[None, ...] * d_deltas, axis=4)
    mean_cost = tf.reduce_mean(cost, axis=(1, 2))
    floor = tf.cast(1.0e-3, dtype)
    cost_scale = tf.maximum(mean_cost, floor)
    d_cost_scale = tf.where(
        mean_cost[None, :] > floor,
        tf.reduce_mean(d_cost, axis=(2, 3)),
        tf.zeros_like(d_cost[:, :, 0, 0]),
    )
    eps_c = tf.cast(epsilon, dtype)
    exponent = -cost / (cost_scale[:, None, None] * eps_c)
    d_exponent = -(
        d_cost / cost_scale[None, :, None, None]
        - cost[None, ...] * d_cost_scale[:, :, None, None]
        / tf.square(cost_scale)[None, :, None, None]
    ) / eps_c
    kernel = tf.exp(exponent)
    d_kernel = kernel[None, ...] * d_exponent

    batch = tf.shape(children)[0]
    uniform = tf.fill([batch, n], 1.0 / n_f)
    left = tf.ones_like(uniform)
    right = tf.ones_like(uniform)
    d_left = tf.zeros_like(d_weights)
    d_right = tf.zeros_like(d_weights)
    tiny = tf.cast(1.0e-7, dtype)
    for _ in range(sinkhorn_steps + balance_steps):
        kr = tf.einsum("bij,bj->bi", kernel, right) + tiny
        d_kr = tf.einsum("kbij,bj->kbi", d_kernel, right) + tf.einsum(
            "bij,kbj->kbi", kernel, d_right
        )
        left_new = uniform / kr
        d_left_new = -uniform[None, ...] * d_kr / tf.square(kr)[None, ...]
        kl = tf.einsum("bij,bi->bj", kernel, left_new) + tiny
        d_kl = tf.einsum("kbij,bi->kbj", d_kernel, left_new) + tf.einsum(
            "bij,kbi->kbj", kernel, d_left_new
        )
        right = weights / kl
        d_right = d_weights / kl[None, ...] - (
            weights[None, ...] * d_kl / tf.square(kl)[None, ...]
        )
        left, d_left = left_new, d_left_new

    coupling = left[:, :, None] * kernel * right[:, None, :]
    d_coupling = (
        d_left[:, :, :, None] * kernel[None, ...] * right[None, :, None, :]
        + left[None, :, :, None] * d_kernel * right[None, :, None, :]
        + left[None, :, :, None] * kernel[None, ...] * d_right[:, :, None, :]
    )
    row_mass = tf.reduce_sum(coupling, axis=2)
    d_row_mass = tf.reduce_sum(d_coupling, axis=3)
    transport = coupling / row_mass[:, :, None]
    d_transport = (
        d_coupling / row_mass[None, :, :, None]
        - coupling[None, ...] * d_row_mass[:, :, :, None]
        / tf.square(row_mass)[None, :, :, None]
    )
    barycentric = tf.einsum("bij,bjd->bid", transport, children)
    d_barycentric = tf.einsum(
        "kbij,bjd->kbid", d_transport, children
    ) + tf.einsum("bij,kbjd->kbid", transport, d_children)

    target_mean = tf.reduce_sum(weights[:, :, None] * children, axis=1)
    d_target_mean = tf.reduce_sum(
        d_weights[:, :, :, None] * children[None, ...]
        + weights[None, :, :, None] * d_children,
        axis=2,
    )
    centered_src = children - target_mean[:, None, :]
    d_centered_src = d_children - d_target_mean[:, :, None, :]
    target_cov = _sym(
        tf.einsum("bn,bni,bnj->bij", weights, centered_src, centered_src)
    )
    d_target_cov = _sym(
        tf.einsum("kbn,bni,bnj->kbij", d_weights, centered_src, centered_src)
        + tf.einsum(
            "bn,kbni,bnj->kbij", weights, d_centered_src, centered_src
        )
        + tf.einsum(
            "bn,bni,kbnj->kbij", weights, centered_src, d_centered_src
        )
    )
    eye = tf.eye(dim, batch_shape=[batch], dtype=dtype)
    ridge_eye = tf.cast(ridge, dtype) * eye

    plus_mean = tf.reduce_mean(barycentric, axis=1)
    d_plus_mean = tf.reduce_mean(d_barycentric, axis=2)
    centered_plus = barycentric - plus_mean[:, None, :]
    d_centered_plus = d_barycentric - d_plus_mean[:, :, None, :]
    plus_cov = _sym(
        tf.einsum("bni,bnj->bij", centered_plus, centered_plus) / n_f
    )
    d_plus_cov = _sym(
        (
            tf.einsum(
                "kbni,bnj->kbij", d_centered_plus, centered_plus
            )
            + tf.einsum(
                "bni,kbnj->kbij", centered_plus, d_centered_plus
            )
        )
        / n_f
    )
    gap = _sym(target_cov - plus_cov) + ridge_eye
    d_gap = _sym(d_target_cov - d_plus_cov)
    gap_chol = tf.linalg.cholesky(gap)
    d_gap_chol = _chol_diff(gap_chol, d_gap)

    injected = barycentric + tf.einsum("bnj,bij->bni", design, gap_chol)
    d_injected = d_barycentric + tf.einsum(
        "bnj,kbij->kbni", design, d_gap_chol
    )
    injected_mean = tf.reduce_mean(injected, axis=1)
    d_injected_mean = tf.reduce_mean(d_injected, axis=2)
    centered_injected = injected - injected_mean[:, None, :]
    d_centered_injected = d_injected - d_injected_mean[:, :, None, :]
    injected_cov = _sym(
        tf.einsum(
            "bni,bnj->bij", centered_injected, centered_injected
        )
        / n_f
    ) + ridge_eye
    d_injected_cov = _sym(
        (
            tf.einsum(
                "kbni,bnj->kbij", d_centered_injected, centered_injected
            )
            + tf.einsum(
                "bni,kbnj->kbij", centered_injected, d_centered_injected
            )
        )
        / n_f
    )

    target_chol = tf.linalg.cholesky(target_cov + ridge_eye)
    d_target_chol = _chol_diff(target_chol, d_target_cov)
    injected_chol = tf.linalg.cholesky(injected_cov)
    d_injected_chol = _chol_diff(injected_chol, d_injected_cov)
    rhs = tf.linalg.matrix_transpose(target_chol)
    d_rhs = tf.linalg.matrix_transpose(d_target_chol)
    solved = tf.linalg.triangular_solve(injected_chol, rhs, adjoint=True)
    d_solved = tf.linalg.triangular_solve(
        injected_chol[None, ...],
        d_rhs
        - tf.linalg.matmul(
            d_injected_chol, solved[None, ...], transpose_a=True
        ),
        adjoint=True,
    )
    affine = tf.linalg.matrix_transpose(solved)
    d_affine = tf.linalg.matrix_transpose(d_solved)

    particles = (
        tf.einsum("bnj,bij->bni", centered_injected, affine)
        + target_mean[:, None, :]
    )
    d_particles = (
        d_target_mean[:, :, None, :]
        + tf.einsum(
            "kbnj,bij->kbni", d_centered_injected, affine
        )
        + tf.einsum(
            "bnj,kbij->kbni", centered_injected, d_affine
        )
    )
    carried_covariances = tf.einsum(
        "bij,bjmn->bimn", transport, covariances
    )
    d_carried_covariances = (
        tf.einsum(
            "kbij,bjmn->kbimn", d_transport, covariances
        )
        + tf.einsum(
            "bij,kbjmn->kbimn", transport, d_covariances
        )
    )
    return (
        particles,
        d_particles,
        _sym(carried_covariances),
        _sym(d_carried_covariances),
        transport,
        d_transport,
    )


__all__ = ["batched_sinkhorn_contract_e_reset_triple_with_tangent"]
