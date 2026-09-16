"""Rectangular direct-stack factors and singular-support likelihoods.

SVD is intentionally confined to this value/diagnostic module and is applied
to the residual stack itself, never to a materialized covariance.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf


_DEFAULT_PIVOT_TOLERANCE = 1.0e-12
_DEFAULT_CHART_TOLERANCE = 1.0e-10
_LOG_TWO_PI = tf.math.log(tf.constant(2.0 * 3.141592653589793, tf.float64))


def _finite(value: Any, name: str) -> tf.Tensor:
    value = tf.cast(tf.convert_to_tensor(value, name=name), tf.float64)
    tf.debugging.assert_all_finite(value, f"{name} contains NaN or Inf")
    return value


def _batched_qr_with_derivative(
    matrix: tf.Tensor,
    derivative: tf.Tensor | None = None,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None]:
    """Thin QR and its fixed positive-diagonal branch derivative.

    ``matrix`` is ``[B,N,R]`` with ``N >= R``.  The derivative uses the
    standard fixed-chart identity ``dQ=(I-QQ')dA R^-1+Q Omega`` where the
    skew matrix Omega is selected so that ``dR`` remains upper triangular.
    """
    matrix = _finite(matrix, "qr_matrix")
    if matrix.shape.rank != 3:
        raise ValueError("qr_matrix must have shape [B,N,R]")
    b, n, r = matrix.shape.as_list()
    if None in (b, n, r) or n < r:
        raise ValueError("qr_matrix must have static N >= R dimensions")
    q, upper = tf.linalg.qr(matrix, full_matrices=False)
    diagonal = tf.linalg.diag_part(upper)
    signs = tf.where(diagonal >= 0.0, tf.ones_like(diagonal), -tf.ones_like(diagonal))
    q = q * signs[:, tf.newaxis, :]
    upper = signs[:, :, tf.newaxis] * upper
    if derivative is None:
        return q, upper, None, None
    derivative = _finite(derivative, "qr_derivative")
    if derivative.shape.rank != 4 or derivative.shape[0] != b or derivative.shape[2:] != (n, r):
        raise ValueError("qr_derivative must have shape [B,P,N,R]")
    p = derivative.shape[1]
    d_q_rows = []
    d_r_rows = []
    for parameter_index in range(int(p)):
        d_matrix = derivative[:, parameter_index]
        projected = tf.einsum("bni,bnj->bij", q, d_matrix)
        scaled = tf.linalg.triangular_solve(
            tf.linalg.matrix_transpose(upper),
            tf.linalg.matrix_transpose(projected),
            lower=True,
        )
        scaled = tf.linalg.matrix_transpose(scaled)
        lower = tf.linalg.band_part(scaled, -1, 0) - tf.linalg.diag(
            tf.linalg.diag_part(scaled)
        )
        omega = lower - tf.linalg.matrix_transpose(lower)
        d_upper = tf.einsum("bij,bjk->bik", scaled - omega, upper)
        normal = d_matrix - tf.einsum("bni,bij->bnj", q, projected)
        d_q = tf.linalg.matrix_transpose(
            tf.linalg.triangular_solve(
                tf.linalg.matrix_transpose(upper),
                tf.linalg.matrix_transpose(normal),
                lower=True,
            )
        ) + tf.einsum("bni,bij->bnj", q, omega)
        d_q_rows.append(d_q)
        d_r_rows.append(d_upper)
    return q, upper, tf.stack(d_q_rows, axis=1), tf.stack(d_r_rows, axis=1)


def _fixed_chart_decomposition(
    stack: tf.Tensor,
    permutation: tf.Tensor,
    rank: int,
    derivative_stack: tf.Tensor | None = None,
    *,
    residual_tolerance: float = _DEFAULT_CHART_TOLERANCE,
    pivot_tolerance: float = _DEFAULT_PIVOT_TOLERANCE,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor | None, tf.Tensor | None, tf.Tensor | None, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Return ``stack = G Q'`` on one fixed row-pivot QR chart."""
    stack = _finite(stack, "stack")
    if stack.shape.rank != 3:
        raise ValueError("stack must have shape [B,N,K]")
    b, n, k = stack.shape.as_list()
    if None in (b, n, k):
        raise ValueError("stack dimensions must be static")
    permutation = tf.convert_to_tensor(permutation, tf.int32)
    if permutation.shape.rank != 1 or permutation.shape[0] != n:
        raise ValueError("permutation must contain one index per state coordinate")
    perm_static = tf.get_static_value(permutation)
    if perm_static is None:
        raise ValueError("permutation must be statically known before tracing")
    perm_values = perm_static.tolist()
    if sorted(perm_values) != list(range(n)):
        raise ValueError("permutation must be a bijection of state coordinates")
    rank = int(rank)
    if rank <= 0 or rank > min(n, k):
        raise ValueError("rank must satisfy 1 <= rank <= min(N,K)")
    if derivative_stack is not None:
        derivative_stack = _finite(derivative_stack, "derivative_stack")
        if derivative_stack.shape.rank != 4 or derivative_stack.shape[0] != b or derivative_stack.shape[2:] != (n, k):
            raise ValueError("derivative_stack must have shape [B,P,N,K]")
    permuted = tf.gather(stack, permutation, axis=1)
    base = tf.linalg.matrix_transpose(permuted[:, :rank, :])
    tail = tf.linalg.matrix_transpose(permuted[:, rank:, :])
    d_base = d_tail = None
    if derivative_stack is not None:
        d_permuted = tf.gather(derivative_stack, permutation, axis=2)
        d_base = tf.linalg.matrix_transpose(d_permuted[:, :, :rank, :])
        d_tail = tf.linalg.matrix_transpose(d_permuted[:, :, rank:, :])
    q, r11, d_q, d_r11 = _batched_qr_with_derivative(base, d_base)
    r12 = tf.einsum("bkr,bks->brs", q, tail)
    d_r12 = None
    if d_q is not None:
        d_r12 = tf.einsum("bpkr,bks->bprs", d_q, tail) + tf.einsum(
            "bkr,bpks->bprs", q, d_tail
        )
    g_permuted = tf.linalg.matrix_transpose(tf.concat([r11, r12], axis=2))
    inverse_permutation = tf.argsort(permutation)
    g = tf.gather(g_permuted, inverse_permutation, axis=1)
    d_g = None
    if d_r12 is not None:
        d_g = tf.gather(
            tf.linalg.matrix_transpose(tf.concat([d_r11, d_r12], axis=3)),
            inverse_permutation,
            axis=2,
        )
    residual_matrix = tail - tf.einsum("bkr,brs->bks", q, r12)
    norm = tf.linalg.norm(permuted, axis=[-2, -1])
    residual = tf.linalg.norm(residual_matrix, axis=[-2, -1])
    pivots = tf.abs(tf.linalg.diag_part(r11))
    diagnostics = {
        "rank": tf.fill([b], tf.cast(rank, tf.int32)),
        "chart_residual": residual,
        "minimum_chart_pivot": tf.reduce_min(pivots, axis=-1),
        "relative_chart_pivot": tf.reduce_min(pivots, axis=-1) / tf.maximum(norm, 1.0e-300),
        "chart_valid": tf.logical_and(
            residual <= tf.cast(residual_tolerance, tf.float64) * tf.maximum(norm, 1.0),
            tf.reduce_all(pivots > tf.cast(pivot_tolerance, tf.float64) * tf.maximum(norm[:, None], 1.0e-300), axis=-1),
        ),
        "pivot_permutation": permutation,
        "chart_kind": tf.constant("fixed_row_pivot_qr"),
    }
    if derivative_stack is not None:
        diagnostics["derivative_chart_valid"] = diagnostics["chart_valid"]
    return g, q, r12, d_g, d_q, d_r11, d_r12, diagnostics


def batched_fixed_pivot_rectangular_qr(
    stack: tf.Tensor,
    permutation: tf.Tensor,
    rank: int,
    derivative_stack: tf.Tensor | None = None,
    residual_tolerance: float = 1.0e-12,
    pivot_tolerance: float = _DEFAULT_PIVOT_TOLERANCE,
) -> tuple[tf.Tensor, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Construct a fixed-rank rectangular factor from a selected QR chart."""
    if rank == 0:
        stack = _finite(stack, "stack")
        if stack.shape.rank != 3:
            raise ValueError("stack must have shape [B, dimension, columns]")
        b, n, _ = stack.shape.as_list()
        if None in (b, n):
            raise ValueError("stack dimensions must be static")
        d_factor = None
        if derivative_stack is not None:
            derivative_stack = _finite(derivative_stack, "derivative_stack")
            d_factor = tf.zeros([b, derivative_stack.shape[1], n, 0], tf.float64)
        return tf.zeros([b, n, 0], tf.float64), d_factor, {
            "rank": tf.zeros([b], tf.int32),
            "chart_residual": tf.linalg.norm(stack, axis=[-2, -1]),
            "minimum_chart_pivot": tf.zeros([b], tf.float64),
            "relative_chart_pivot": tf.zeros([b], tf.float64),
            "chart_valid": tf.linalg.norm(stack, axis=[-2, -1]) <= residual_tolerance,
            "chart_kind": tf.constant("rank_zero_value_only"),
        }
    factor, _, _, d_factor, _, _, _, diagnostics = _fixed_chart_decomposition(
        stack,
        permutation,
        rank,
        derivative_stack,
        residual_tolerance=residual_tolerance,
        pivot_tolerance=pivot_tolerance,
    )
    return factor, d_factor, diagnostics


def batched_direct_stack_svd_factor(
    stack: tf.Tensor,
    relative_cutoff: float = 1.0e-12,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    """Return a padded direct-stack SVD factor and effective ranks."""

    stack = _finite(stack, "stack")
    if stack.shape.rank != 3:
        raise ValueError("stack must have shape [B, dimension, columns]")
    if relative_cutoff < 0.0:
        raise ValueError("relative_cutoff must be nonnegative")
    singular, u, _ = tf.linalg.svd(stack, full_matrices=False, compute_uv=True)
    scale = tf.maximum(singular[:, :1], tf.constant(1.0e-300, tf.float64))
    active = singular > tf.cast(relative_cutoff, tf.float64) * scale
    rank = tf.reduce_sum(tf.cast(active, tf.int32), axis=-1)
    factor = u * singular[:, tf.newaxis, :] * tf.cast(active[:, tf.newaxis, :], tf.float64)
    diagnostics = {
        "rank": rank,
        "singular_values": singular,
        "active_mask": active,
        "discarded_tail_norm": tf.linalg.norm(
            singular * tf.cast(tf.logical_not(active), tf.float64), axis=-1
        ),
        "value_only": tf.constant(True),
    }
    return factor, singular, u, diagnostics


def batched_support_gaussian_log_likelihood(
    innovation: tf.Tensor,
    observation_stack: tf.Tensor,
    relative_cutoff: float = 1.0e-12,
    support_tolerance: float = 1.0e-10,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    """Evaluate a Gaussian density on the support of a direct residual stack."""

    innovation = _finite(innovation, "innovation")
    stack = _finite(observation_stack, "observation_stack")
    if innovation.shape.rank != 2 or stack.shape.rank != 3 or innovation.shape[0] != stack.shape[0] or innovation.shape[1] != stack.shape[1]:
        raise ValueError("innovation and observation_stack dimensions are incompatible")
    factor, singular, u, diagnostics = batched_direct_stack_svd_factor(stack, relative_cutoff)
    active = diagnostics["active_mask"]
    coordinates = tf.einsum("bnk,bn->bk", u, innovation)
    projected = tf.einsum("bnk,bk->bn", u, coordinates * tf.cast(active, tf.float64))
    support_residual = tf.linalg.norm(innovation - projected, axis=-1)
    scale = tf.maximum(tf.constant(1.0, tf.float64), tf.linalg.norm(innovation, axis=-1))
    on_support = support_residual <= support_tolerance * scale
    safe = tf.maximum(singular, tf.constant(1.0e-300, tf.float64))
    coordinates = tf.einsum("bnk,bn->bk", u, innovation) / safe
    mask = tf.cast(active, tf.float64)
    log_likelihood = -0.5 * (
        tf.cast(diagnostics["rank"], tf.float64) * tf.math.log(2.0 * tf.constant(3.141592653589793, tf.float64))
        + 2.0 * tf.reduce_sum(mask * tf.math.log(safe), axis=-1)
        + tf.reduce_sum(mask * coordinates * coordinates, axis=-1)
    )
    log_likelihood = tf.where(
        on_support,
        log_likelihood,
        tf.fill(tf.shape(log_likelihood), tf.constant(-float("inf"), tf.float64)),
    )
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "support_residual": support_residual,
            "on_support": on_support,
            "likelihood_measure": tf.constant("affine_support_gaussian"),
        }
    )
    return log_likelihood, diagnostics["rank"], diagnostics


def batched_direct_support_conditional(
    observation_stack: tf.Tensor,
    state_stack: tf.Tensor,
    innovation: tf.Tensor,
    relative_cutoff: float = 1.0e-12,
    support_tolerance: float = 1.0e-10,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    """Value-only conditional update for a possibly singular observation stack.

    The observation SVD is applied to the stack itself.  Its right singular
    vectors define the observed column subspace, so projecting the state stack
    onto the orthogonal complement gives the conditional loading directly.
    No covariance or Schur complement is materialized.
    """

    observation_stack = _finite(observation_stack, "observation_stack")
    state_stack = _finite(state_stack, "state_stack")
    innovation = _finite(innovation, "innovation")
    if (
        observation_stack.shape.rank != 3
        or state_stack.shape.rank != 3
        or innovation.shape.rank != 2
    ):
        raise ValueError("stacks must be rank three and innovation rank two")
    b, ny, k = observation_stack.shape.as_list()
    bs, nx, ks = state_stack.shape.as_list()
    if None in (b, ny, nx, k) or (b, ny, k) != (bs, ny, ks) or innovation.shape.as_list() != [b, ny]:
        raise ValueError("observation/state/innovation dimensions are incompatible")
    singular, u, v = tf.linalg.svd(observation_stack, full_matrices=False, compute_uv=True)
    scale = tf.maximum(singular[:, :1], tf.constant(1.0e-300, tf.float64))
    active = singular > tf.cast(relative_cutoff, tf.float64) * scale
    mask = tf.cast(active, tf.float64)
    u_active = u * mask[:, tf.newaxis, :]
    v_active = v * mask[:, tf.newaxis, :]
    coordinates = tf.einsum("bnk,bn->bk", u, innovation)
    projected = tf.einsum("bnk,bk->bn", u, coordinates * mask)
    support_residual = tf.linalg.norm(innovation - projected, axis=-1)
    on_support = support_residual <= support_tolerance * tf.maximum(tf.constant(1.0, tf.float64), tf.linalg.norm(innovation, axis=-1))
    rank = tf.reduce_sum(tf.cast(active, tf.int32), axis=-1)
    safe = tf.maximum(singular, tf.constant(1.0e-300, tf.float64))
    state_coordinates = tf.einsum("bnk,bkr->bnr", state_stack, v_active)
    observed_state = tf.einsum("bnr,bkr->bnk", state_coordinates, v_active)
    conditional_stack = state_stack - observed_state
    conditional_factor, _, _, conditional_diagnostics = batched_direct_stack_svd_factor(
        conditional_stack, relative_cutoff
    )
    gain = tf.einsum(
        "bnk,bk,bmk->bnm",
        state_coordinates,
        mask / safe,
        u,
    )
    log_likelihood = -0.5 * (
        tf.cast(rank, tf.float64) * tf.math.log(2.0 * tf.constant(3.141592653589793, tf.float64))
        + 2.0 * tf.reduce_sum(mask * tf.math.log(safe), axis=-1)
        + tf.reduce_sum(mask * (coordinates / safe) * (coordinates / safe), axis=-1)
    )
    log_likelihood = tf.where(
        on_support,
        log_likelihood,
        tf.fill(tf.shape(log_likelihood), tf.constant(-float("inf"), tf.float64)),
    )
    diagnostics = {
        "innovation_rank": rank,
        "innovation_singular_values": singular,
        "innovation_active_mask": active,
        "support_residual": support_residual,
        "on_support": on_support,
        "conditional_stack": conditional_stack,
        "conditional_rank": conditional_diagnostics["rank"],
        "likelihood_measure": tf.constant("affine_support_gaussian"),
        "value_only": tf.constant(True),
    }
    return log_likelihood, gain, conditional_factor, rank, diagnostics


def batched_fixed_support_qr_likelihood(
    innovation: tf.Tensor,
    factor: tf.Tensor,
    derivative_innovation: tf.Tensor | None = None,
    derivative_factor: tf.Tensor | None = None,
    *,
    support_tolerance: float = 1.0e-10,
) -> tuple[tf.Tensor, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Support likelihood on a fixed rectangular QR factor branch.

    The factor is ``G`` in ``P = G G'`` with shape ``[B,N,R]``.  The returned
    score is ``d log p`` with parameter axis ``P`` and is valid only while the
    rank, QR signs, and support chart remain fixed.  No covariance is formed.
    """
    innovation = _finite(innovation, "innovation")
    factor = _finite(factor, "factor")
    if innovation.shape.rank != 2 or factor.shape.rank != 3:
        raise ValueError("innovation must be [B,N] and factor [B,N,R]")
    b, n = innovation.shape.as_list()
    if factor.shape[0] != b or factor.shape[1] != n:
        raise ValueError("innovation/factor dimensions are incompatible")
    if (derivative_innovation is None) != (derivative_factor is None):
        raise ValueError("both derivative_innovation and derivative_factor are required together")
    d_innovation = d_factor = None
    if derivative_innovation is not None:
        d_innovation = _finite(derivative_innovation, "derivative_innovation")
        d_factor = _finite(derivative_factor, "derivative_factor")
        if d_innovation.shape.rank != 3 or d_factor.shape.rank != 4:
            raise ValueError("derivatives must have shapes [B,P,N] and [B,P,N,R]")
        if d_innovation.shape[0] != b or d_innovation.shape[2] != n or d_factor.shape[0] != b or d_factor.shape[2:] != factor.shape[1:]:
            raise ValueError("derivative dimensions are incompatible")
    q, r, d_q, d_r = _batched_qr_with_derivative(factor, d_factor)
    diagonal = tf.linalg.diag_part(r)
    projected = tf.einsum("bnr,bn->br", q, innovation)
    z = tf.linalg.triangular_solve(r, projected[..., None], lower=False)[..., 0]
    residual = innovation - tf.einsum("bnr,br->bn", q, projected)
    on_support = tf.linalg.norm(residual, axis=-1) <= tf.cast(support_tolerance, tf.float64) * tf.maximum(tf.linalg.norm(innovation, axis=-1), 1.0)
    value = -0.5 * (
        tf.cast(tf.shape(r)[-1], tf.float64) * _LOG_TWO_PI
        + 2.0 * tf.reduce_sum(tf.math.log(diagonal), axis=-1)
        + tf.reduce_sum(z * z, axis=-1)
    )
    value = tf.where(on_support, value, tf.fill(tf.shape(value), tf.constant(-float("inf"), tf.float64)))
    score = None
    if d_innovation is not None:
        d_projected = tf.einsum("bpnr,bn->bpr", d_q, innovation) + tf.einsum("bnr,bpn->bpr", q, d_innovation)
        dz_rhs = d_projected - tf.einsum("bpri,bi->bpr", d_r, z)
        dz = tf.linalg.triangular_solve(
            tf.repeat(r, d_innovation.shape[1], axis=0),
            tf.reshape(dz_rhs, [b * d_innovation.shape[1], -1, 1]),
            lower=False,
        )
        dz = tf.reshape(dz[..., 0], [b, d_innovation.shape[1], -1])
        score = -tf.reduce_sum(tf.linalg.diag_part(d_r) / diagonal[:, None, :], axis=-1) - tf.reduce_sum(z[:, None, :] * dz, axis=-1)
        score = tf.where(on_support[:, None], score, tf.fill(tf.shape(score), tf.constant(float("nan"), tf.float64)))
    diagnostics = {
        "rank": tf.fill([b], tf.shape(r)[-1]),
        "support_residual": tf.linalg.norm(residual, axis=-1),
        "on_support": on_support,
        "likelihood_measure": tf.constant("affine_support_gaussian_fixed_qr"),
        "score_valid": on_support,
        "value_only": tf.constant(d_innovation is None),
    }
    return value, score, diagnostics


def batched_fixed_support_qr_conditional(
    observation_stack: tf.Tensor,
    state_stack: tf.Tensor,
    innovation: tf.Tensor,
    observation_permutation: tf.Tensor,
    observation_rank: int,
    conditional_permutation: tf.Tensor,
    conditional_rank: int,
    observation_derivative_stack: tf.Tensor | None = None,
    state_derivative_stack: tf.Tensor | None = None,
    derivative_innovation: tf.Tensor | None = None,
    *,
    chart_tolerance: float = _DEFAULT_CHART_TOLERANCE,
    pivot_tolerance: float = _DEFAULT_PIVOT_TOLERANCE,
    support_tolerance: float = 1.0e-10,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor, tf.Tensor | None, tf.Tensor, tf.Tensor | None, tf.Tensor, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Fixed-support QR conditional update and first derivatives.

    ``Q`` is the retained column-space basis of ``observation_stack.T``.  The
    posterior loading is the fixed-chart QR factor of ``state_stack(I-QQ')``.
    """
    y = _finite(observation_stack, "observation_stack")
    x = _finite(state_stack, "state_stack")
    e = _finite(innovation, "innovation")
    if y.shape.rank != 3 or x.shape.rank != 3 or e.shape.rank != 2:
        raise ValueError("stacks must be rank three and innovation rank two")
    b, ny, k = y.shape.as_list()
    if x.shape[0] != b or x.shape[2] != k or e.shape.as_list() != [b, ny]:
        raise ValueError("observation/state/innovation dimensions are incompatible")
    if (observation_derivative_stack is None) != (state_derivative_stack is None) or (state_derivative_stack is None) != (derivative_innovation is None):
        raise ValueError("all derivative inputs are required together")
    dy = dx = de = None
    if observation_derivative_stack is not None:
        dy = _finite(observation_derivative_stack, "observation_derivative_stack")
        dx = _finite(state_derivative_stack, "state_derivative_stack")
        de = _finite(derivative_innovation, "derivative_innovation")
    # Observation chart: Y = G V' where V is the retained sigma-column basis
    # and G is the rectangular observation-support loading.
    y_factor, v, _, d_y_factor, d_v, _, _, y_diag = _fixed_chart_decomposition(
        y, observation_permutation, observation_rank, dy,
        residual_tolerance=chart_tolerance, pivot_tolerance=pivot_tolerance,
    )
    u_obs, r_obs, d_u_obs, d_r_obs = _batched_qr_with_derivative(y_factor, d_y_factor)
    projected = tf.einsum("bnr,bn->br", u_obs, e)
    residual_x = x - tf.einsum("bnk,bkr,bmr->bnm", x, v, v)
    d_residual_x = None
    if dx is not None:
        d_residual_x = dx - tf.einsum("bpnk,bkr,bmr->bpnm", dx, v, v)
        d_residual_x -= tf.einsum("bnk,bpkr,bmr->bpnm", x, d_v, v)
        d_residual_x -= tf.einsum("bnk,bkr,bpmr->bpnm", x, v, d_v)
    conditional_factor, d_conditional_factor, conditional_diag = batched_fixed_pivot_rectangular_qr(
        residual_x, conditional_permutation, conditional_rank, d_residual_x,
        residual_tolerance=chart_tolerance,
        pivot_tolerance=pivot_tolerance,
    )
    x_v = tf.einsum("bnk,bkr->bnr", x, v)
    solved = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(r_obs), tf.linalg.matrix_transpose(x_v), lower=True
    ))
    gain = tf.einsum("bnr,bsr->bns", solved, u_obs)
    filtered_increment = tf.einsum("bnr,br->bn", solved, projected)
    d_gain = d_increment = d_likelihood = None
    if dy is not None:
        p = dy.shape[1]
        d_projected = tf.einsum("bpnr,bn->bpr", d_u_obs, e) + tf.einsum("bnr,bpn->bpr", u_obs, de)
        d_x_v = tf.einsum("bpnk,bkr->bpnr", dx, v) + tf.einsum("bnk,bpkr->bpnr", x, d_v)
        d_solved_rhs = d_x_v - tf.einsum("bnj,bpjr->bpnr", solved, d_r_obs)
        d_solved = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(
            tf.repeat(tf.linalg.matrix_transpose(r_obs), p, axis=0),
            tf.linalg.matrix_transpose(tf.reshape(d_solved_rhs, [b * p, x.shape[1], observation_rank])),
            lower=True,
        ))
        d_solved = tf.reshape(d_solved, [b, p, x.shape[1], observation_rank])
        d_gain = tf.einsum("bpni,bsi->bpns", d_solved, u_obs) + tf.einsum("bni,bpsi->bpns", solved, d_u_obs)
        d_increment = tf.einsum("bpnr,br->bpn", d_solved, projected) + tf.einsum("bnr,bpr->bpn", solved, d_projected)
    likelihood, d_likelihood, likelihood_diag = batched_fixed_support_qr_likelihood(
        e, y_factor, de, d_y_factor, support_tolerance=support_tolerance
    )
    diagnostics = {**y_diag, **{"conditional_" + key: value for key, value in conditional_diag.items()}, **likelihood_diag}
    diagnostics["value_only"] = tf.constant(dy is None)
    return likelihood, d_likelihood, gain, d_gain, conditional_factor, d_conditional_factor, filtered_increment, d_increment, diagnostics


def batched_fixed_support_qr_update(
    observation_stack: tf.Tensor,
    state_stack: tf.Tensor,
    innovation: tf.Tensor,
    observation_permutation: tf.Tensor,
    observation_rank: int,
    conditional_permutation: tf.Tensor,
    conditional_rank: int,
    observation_derivative_stack: tf.Tensor | None = None,
    state_derivative_stack: tf.Tensor | None = None,
    derivative_innovation: tf.Tensor | None = None,
    *,
    chart_tolerance: float = _DEFAULT_CHART_TOLERANCE,
    pivot_tolerance: float = _DEFAULT_PIVOT_TOLERANCE,
    support_tolerance: float = 1.0e-10,
) -> tuple[tf.Tensor, tf.Tensor | None, tf.Tensor, tf.Tensor | None, tf.Tensor, tf.Tensor | None, dict[str, tf.Tensor]]:
    """Compact public wrapper returning support update quantities."""
    result = batched_fixed_support_qr_conditional(
        observation_stack,
        state_stack,
        innovation,
        observation_permutation,
        observation_rank,
        conditional_permutation,
        conditional_rank,
        observation_derivative_stack,
        state_derivative_stack,
        derivative_innovation,
        chart_tolerance=chart_tolerance,
        pivot_tolerance=pivot_tolerance,
        support_tolerance=support_tolerance,
    )
    likelihood, _, gain, d_gain, factor, d_factor, increment, d_increment, diagnostics = result
    score = result[1]
    return likelihood, score, increment, d_increment, factor, d_factor, {
        **diagnostics,
        "gain": gain,
        "d_gain": d_gain if d_gain is not None else tf.zeros([0], tf.float64),
    }


__all__ = [
    "batched_fixed_pivot_rectangular_qr",
    "batched_direct_stack_svd_factor",
    "batched_support_gaussian_log_likelihood",
    "batched_direct_support_conditional",
    "batched_fixed_support_qr_likelihood",
    "batched_fixed_support_qr_conditional",
    "batched_fixed_support_qr_update",
]
