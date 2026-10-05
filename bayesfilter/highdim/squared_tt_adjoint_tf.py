"""Manual adjoint (reverse-sweep) node primitives for the squared-TT score.

UB-1 Addendum A (2026-08-17) implementation, node classes A.2.1-A.2.7.
Every function here is the explicit TRANSPOSE of a forward-linear tangent
map already derived and FD-tested in the forward chain; no autodiff is
used (Method A manual-backend discipline). Validation contract:

- U-ADJ-NODE-1: inner-product identity
  <bar_out, F[d_in]> == <F^T[bar_out], d_in> per node class;
- U-ADJ-SOLVE-1: solve-node adjoint vs FD through the actual scaled
  augmented solver.

Selected by the P2A mode decision
(docs/plans/bayesfilter-p2a-cost-prototype-result-2026-08-17.md):
forward tangent replay is disqualified at p=300; this adjoint sweep is
the P2 score mode.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64


def solve_node_adjoint(
    design: tf.Tensor,
    weights: tf.Tensor,
    target: tf.Tensor,
    coefficients: tf.Tensor,
    ridge: float,
    bar_coefficients: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Adjoint of (A, g) -> c solving (A'WA + rho I) c = A'Wg  (A.2.1).

    Returns (bar_g, bar_A) for cotangent bar_c:
        lambda = N^{-1} bar_c              (N symmetric)
        bar_g  = W A lambda
        bar_A  = W (g - A c) lambda' - (W A lambda) c'
    """

    design = tf.convert_to_tensor(design, DTYPE)
    weights = tf.convert_to_tensor(weights, DTYPE)
    target = tf.convert_to_tensor(target, DTYPE)
    c = tf.convert_to_tensor(coefficients, DTYPE)
    bar_c = tf.convert_to_tensor(bar_coefficients, DTYPE)
    lam = scaled_normal_solve(design, weights, ridge, bar_c)
    w_a_lam = weights * tf.linalg.matvec(design, lam)
    residual = target - tf.linalg.matvec(design, c)
    bar_g = w_a_lam
    bar_a = (weights * residual)[:, None] * lam[None, :] - w_a_lam[:, None] * c[None, :]
    return bar_g, bar_a


def _core_matrices(
    product_basis: ProductBasis, points: tf.Tensor, cores: Sequence[TTCore]
) -> list[tf.Tensor]:
    from bayesfilter.highdim.tt_native_control_tf import core_matrices

    return list(core_matrices(product_basis, points, cores))


def design_assembly_adjoint(
    product_basis: ProductBasis,
    points: tf.Tensor,
    cores: Sequence[TTCore],
    core_index: int,
    bar_design: tf.Tensor,
) -> tuple[TTCore, ...]:
    """Adjoint of dot_cores -> dot_A_i (A.2.2): distribute bar_A onto cores.

    Transpose of `differentiate_design_matrix`: reverse accumulation of the
    left/right dot-environment recursions. Core `core_index` receives a
    zero adjoint (A_i does not contain core i).
    """

    from bayesfilter.highdim.tt_native_control_tf import row_chain_adjoint

    values = row_chain_adjoint(cores, product_basis, tf.convert_to_tensor(points, DTYPE),
                                core_index=core_index, bar_design=bar_design)
    return tuple(TTCore(value) for value in values)


def sqrt_target_adjoint(
    sqrt_target: tf.Tensor,
    argmax_index: tf.Tensor,
    bar_sqrt_target: tf.Tensor,
) -> tf.Tensor:
    """Adjoint of dot_g = 0.5 g (dot_logf - dot_s), dot_s = dot_logf[j*] (A.2.3).

    PRE-v0.3 FORM (argmax max-shift), retained for the node pairing test
    only. The v0.3 engines use the smooth logsumexp shift and implement
    its cotangent INLINE (softmax-weighted; see
    `run_adjoint_score_filter`), so this function is NOT on the active
    score path. Do not wire it into an engine without updating it to the
    v0.3 shift semantics.

    Returns bar_logf.
    """

    g = tf.convert_to_tensor(sqrt_target, DTYPE)
    bar_g = tf.convert_to_tensor(bar_sqrt_target, DTYPE)
    half = 0.5 * g * bar_g
    total = tf.reduce_sum(half)
    return half - tf.scatter_nd(
        tf.reshape(tf.cast(argmax_index, tf.int32), [1, 1]), [total], tf.shape(g)
    )


def retained_evaluator_adjoint(
    v: tf.Tensor,
    gram: tf.Tensor,
    tau: tf.Tensor,
    z_complete: tf.Tensor,
    bar_log_rows: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Adjoint of (v rows, E, Zc) -> log p_ret_ref rows (A.2.5).

    Forward tangent: dot_log = (2 v E dot_v' + v dot_E v') / (q + tau) - dot_Zc / Zc.
    Returns (bar_v [N,r], bar_E [r,r] symmetric, bar_Zc scalar).
    """

    v = tf.convert_to_tensor(v, DTYPE)
    gram = tf.convert_to_tensor(gram, DTYPE)
    bar_rows = tf.convert_to_tensor(bar_log_rows, DTYPE)
    q = tf.einsum("na,ab,nb->n", v, gram, v)
    scale = bar_rows / (q + tau)
    bar_v = 2.0 * scale[:, None] * tf.einsum("nb,ab->na", v, gram)
    bar_e = tf.einsum("n,na,nb->ab", scale, v, v)
    bar_e = 0.5 * (bar_e + tf.transpose(bar_e))
    bar_zc = -tf.reduce_sum(bar_rows) / z_complete
    return bar_v, bar_e, bar_zc


def prefix_rows_adjoint(
    cores: Sequence[TTCore],
    product_basis: ProductBasis,
    points: tf.Tensor,
    bar_rows: tf.Tensor,
) -> tuple[TTCore, ...]:
    """Adjoint of prefix cores -> H_L rows (transpose of the forward
    product-rule propagation in `prefix_row_vectors_tangent`)."""

    from bayesfilter.highdim.tt_native_control_tf import row_chain_adjoint

    values = row_chain_adjoint(cores, product_basis, tf.convert_to_tensor(points, DTYPE), bar_rows=bar_rows)
    return tuple(TTCore(value) for value in values)


def gram_chain_adjoint(
    cores: Sequence[TTCore],
    product_basis: ProductBasis,
    *,
    axis_offset: int,
    bar_gram: tf.Tensor,
) -> tuple[TTCore, ...]:
    """Adjoint of suffix cores -> E (transpose of the suffix Gram chain;
    A.2.7/A.2.8 pattern). Forward tangent is the sum over axes of the
    chain with one core pair replaced; the adjoint mirrors it."""

    from bayesfilter.highdim.tt_native_control_tf import basis_masses, gram_adjoint

    masses = basis_masses(product_basis, cores, axis_offset=axis_offset)
    return tuple(TTCore(value) for value in gram_adjoint(cores, masses, bar_gram, reverse=True))


def cholesky_vjp(chol: tf.Tensor, bar_chol: tf.Tensor) -> tf.Tensor:
    """VJP of A -> L = chol(A): returns symmetric bar_A (A.2.6).

    Standard Phi-operator formula:
        M = L' bar_L;  Phi = lower(M) with halved diagonal
        bar_A = 0.5 * L^{-T} (Phi + Phi') L^{-1}, symmetrized.
    """

    l_matrix = tf.convert_to_tensor(chol, DTYPE)
    bar_l = tf.convert_to_tensor(bar_chol, DTYPE)
    m = tf.matmul(l_matrix, bar_l, transpose_a=True)
    lower = tf.linalg.band_part(m, -1, 0)
    phi = lower - 0.5 * tf.linalg.diag(tf.linalg.diag_part(m))
    inner = phi + tf.transpose(phi)
    step1 = tf.linalg.triangular_solve(l_matrix, inner, lower=True, adjoint=True)
    step2 = tf.linalg.triangular_solve(
        l_matrix, tf.transpose(step1), lower=True, adjoint=True
    )
    bar_a = 0.5 * tf.transpose(step2)  # = 0.5 * L^{-T} inner L^{-1}
    return 0.5 * (bar_a + tf.transpose(bar_a))


__all__ = [
    "solve_node_adjoint",
    "design_assembly_adjoint",
    "sqrt_target_adjoint",
    "retained_evaluator_adjoint",
    "prefix_rows_adjoint",
    "gram_chain_adjoint",
    "cholesky_vjp",
]


# ---------------------------------------------------------------------------
# Scaled-factorization solves for derivative chains (conditioning repair,
# 2026-08-17): N = A'WA + rho I factors as S (M'M) S with M the scaled
# augmented matrix of the VALUE solver; N^{-1} r = S^{-1} R^{-1} R^{-T} S^{-1} r
# via QR of M. Derivative solves thereby inherit the value path's
# conditioning instead of squaring it through raw normal equations.
# ---------------------------------------------------------------------------


def scaled_normal_solve(
    design: tf.Tensor,
    weights: tf.Tensor,
    ridge: float,
    rhs: tf.Tensor,
) -> tf.Tensor:
    """Solve (A'WA + rho I) x = rhs through the scaled augmented QR."""

    from bayesfilter.highdim.fitting import _weighted_column_scales, _DEFAULT_COLUMN_SCALE_FLOOR

    design = tf.convert_to_tensor(design, DTYPE)
    weights = tf.convert_to_tensor(weights, DTYPE)
    rhs = tf.convert_to_tensor(rhs, DTYPE)
    scales, _norms, _floor = _weighted_column_scales(
        design, weights, _DEFAULT_COLUMN_SCALE_FLOOR
    )
    scaled_design = design / tf.reshape(scales, [1, -1])
    sqrt_w = tf.sqrt(weights)
    augmented = tf.concat(
        [
            scaled_design * tf.reshape(sqrt_w, [-1, 1]),
            tf.linalg.diag(tf.sqrt(tf.constant(float(ridge), DTYPE)) / scales),
        ],
        axis=0,
    )
    r_factor = tf.linalg.qr(augmented).r
    y = rhs / scales
    w = tf.linalg.triangular_solve(
        tf.transpose(r_factor), y[:, None], lower=True
    )
    z = tf.linalg.triangular_solve(r_factor, w, lower=False)[:, 0]
    return z / scales


def forward_jvp_replay_scaled(
    updates: Sequence[dict],
    initial_cores: Sequence[TTCore],
    initial_dot_cores: Sequence[TTCore],
    dot_target: tf.Tensor,
) -> tuple[tuple[TTCore, ...], tuple[TTCore, ...]]:
    """Independent test-reference JVP over traced updates with scaled solves.

    This unrolled replay is a parity oracle only. Runtime value/score consumers
    use the native forward/reverse recurrence in squared_tt_native_adjoint_engine_tf.

    Value cores are the traced solutions (bit-identical to the value
    program); only the tangent solves run here, each through
    `scaled_normal_solve`. Same recursion as the donor
    `fixed_als_value_jvp`, differing only in the solve backend.
    """

    cores = list(initial_cores)
    dots = list(initial_dot_cores)
    dot_target = tf.convert_to_tensor(dot_target, DTYPE)
    for update in updates:
        idx = update["core_index"]
        dot_design = None
        from bayesfilter.highdim.derivatives import differentiate_design_matrix

        dot_design = differentiate_design_matrix(
            update["basis"], update["rows"], tuple(cores), tuple(dots), idx
        )
        design = update["design"]
        weights = update["weights"]
        target = update["target"]
        solution = update["solution"]
        a_c = tf.linalg.matvec(design, solution)
        dot_b = tf.linalg.matvec(dot_design, weights * target, transpose_a=True) + tf.linalg.matvec(
            design, weights * dot_target, transpose_a=True
        )
        dot_n_c = tf.linalg.matvec(dot_design, weights * a_c, transpose_a=True) + tf.linalg.matvec(
            design, weights * tf.linalg.matvec(dot_design, solution), transpose_a=True
        )
        dot_solution = scaled_normal_solve(
            design, weights, update["ridge"], dot_b - dot_n_c
        )
        cores[idx] = TTCore(tf.reshape(solution, cores[idx].values.shape))
        dots[idx] = TTCore(tf.reshape(dot_solution, dots[idx].values.shape))
    return tuple(cores), tuple(dots)
