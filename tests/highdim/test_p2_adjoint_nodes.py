"""U-ADJ-NODE-1 / U-ADJ-SOLVE-1: adjoint node primitives vs forward tangents.

Every node: <bar_out, F[d_in]> == <F^T[bar_out], d_in> on random
cotangents/tangents (exact transpose identity, 1e-12 scale), plus FD for
the solve node through the actual scaled augmented solver.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.derivatives import differentiate_design_matrix
from bayesfilter.highdim.fitting import _solve_scaled_augmented_ridge
from bayesfilter.highdim.retained_quadratic_form_tf import (
    prefix_row_vectors_tangent,
    suffix_gram_matrix_tangent,
)
from bayesfilter.highdim.squared_tt_adjoint_tf import (
    cholesky_vjp,
    design_assembly_adjoint,
    gram_chain_adjoint,
    prefix_rows_adjoint,
    retained_evaluator_adjoint,
    solve_node_adjoint,
    sqrt_target_adjoint,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64
SEED = 20260817


def _basis(dimension: int, degree: int = 5) -> highdim.ProductBasis:
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    return highdim.ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree)
            for _ in range(dimension)
        ],
        convention,
    )


def _cores(dimension: int, degree: int, ranks: tuple[int, ...], seed: int):
    basis_dim = int(_basis(1, degree).bases[0].basis_dim)
    rng = np.random.default_rng(seed)
    return tuple(
        TTCore(tf.constant(0.5 * rng.standard_normal((ranks[a], basis_dim, ranks[a + 1])), DTYPE))
        for a in range(dimension)
    )


def _inner(a, b) -> float:
    return float(tf.reduce_sum(tf.convert_to_tensor(a, DTYPE) * tf.convert_to_tensor(b, DTYPE)).numpy())


def _inner_cores(bar_cores, dot_cores) -> float:
    return sum(_inner(b.values, d.values) for b, d in zip(bar_cores, dot_cores))


def test_u_adj_solve_1_solve_node_adjoint_matches_fd_and_pairing() -> None:
    rng = np.random.default_rng(SEED)
    n_rows, cols = 60, 12
    design = tf.constant(rng.standard_normal((n_rows, cols)), DTYPE)
    weights = tf.constant(rng.uniform(0.5, 1.5, size=n_rows), DTYPE)
    target = tf.constant(rng.standard_normal(n_rows), DTYPE)
    ridge = 1e-8
    solve = _solve_scaled_augmented_ridge(
        design=design, target_values=target, weights=weights, ridge=ridge
    )
    c = solve.solution
    bar_c = tf.constant(rng.standard_normal(cols), DTYPE)
    bar_g, bar_a = solve_node_adjoint(design, weights, target, c, ridge, bar_c)

    # Pairing identity against random forward tangents (dot_A, dot_g).
    dot_a = tf.constant(0.1 * rng.standard_normal((n_rows, cols)), DTYPE)
    dot_g = tf.constant(0.1 * rng.standard_normal(n_rows), DTYPE)
    step = 1e-6

    def solve_at(eps: float) -> tf.Tensor:
        return _solve_scaled_augmented_ridge(
            design=design + eps * dot_a,
            target_values=target + eps * dot_g,
            weights=weights,
            ridge=ridge,
        ).solution

    dot_c_fd = (solve_at(step) - solve_at(-step)) / (2.0 * step)
    lhs = _inner(bar_c, dot_c_fd)
    rhs = _inner(bar_g, dot_g) + _inner(bar_a, dot_a)
    assert abs(lhs - rhs) <= 1e-6 * max(1.0, abs(lhs))


def test_u_adj_node_1_design_assembly_pairing() -> None:
    dimension, degree = 3, 5
    ranks = (1, 3, 2, 1)
    basis = _basis(dimension, degree)
    cores = _cores(dimension, degree, ranks, SEED + 1)
    dot_cores = _cores(dimension, degree, ranks, SEED + 2)
    rng = np.random.default_rng(SEED + 3)
    points = tf.constant(rng.uniform(-1.0, 1.0, size=(40, dimension)), DTYPE)
    for core_index in range(dimension):
        dots = tuple(
            TTCore(tf.zeros_like(d.values)) if a == core_index else d
            for a, d in enumerate(dot_cores)
        )
        forward = differentiate_design_matrix(basis, points, cores, dots, core_index)
        bar_design = tf.constant(rng.standard_normal(forward.shape), DTYPE)
        bar_cores = design_assembly_adjoint(basis, points, cores, core_index, bar_design)
        lhs = _inner(bar_design, forward)
        rhs = _inner_cores(bar_cores, dots)
        assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs)), f"core {core_index}: {lhs} vs {rhs}"
        assert float(tf.reduce_max(tf.abs(bar_cores[core_index].values)).numpy()) == 0.0


def test_u_adj_node_1_sqrt_target_pairing() -> None:
    rng = np.random.default_rng(SEED + 4)
    n = 50
    log_f = tf.constant(rng.standard_normal(n), DTYPE)
    j_star = int(tf.argmax(log_f).numpy())
    shift = log_f[j_star]
    g = tf.exp(0.5 * (log_f - shift))
    dot_logf = tf.constant(rng.standard_normal(n), DTYPE)
    dot_g = 0.5 * g * (dot_logf - dot_logf[j_star])
    bar_g = tf.constant(rng.standard_normal(n), DTYPE)
    bar_logf = sqrt_target_adjoint(g, j_star, bar_g)
    assert abs(_inner(bar_g, dot_g) - _inner(bar_logf, dot_logf)) <= 1e-12 * n


def test_u_adj_node_1_retained_evaluator_pairing() -> None:
    rng = np.random.default_rng(SEED + 5)
    n, r = 30, 4
    v = tf.constant(rng.standard_normal((n, r)), DTYPE)
    raw = rng.standard_normal((r, r))
    gram = tf.constant(raw @ raw.T + 0.5 * np.eye(r), DTYPE)
    tau = tf.constant(0.3, DTYPE)
    zc = tf.constant(2.7, DTYPE)
    dot_v = tf.constant(rng.standard_normal((n, r)), DTYPE)
    raw_de = rng.standard_normal((r, r))
    dot_e = tf.constant(0.5 * (raw_de + raw_de.T), DTYPE)
    dot_zc = tf.constant(0.37, DTYPE)
    q = tf.einsum("na,ab,nb->n", v, gram, v)
    dot_log = (
        2.0 * tf.einsum("na,ab,nb->n", v, gram, dot_v)
        + tf.einsum("na,ab,nb->n", dot_v * 0.0 + v, dot_e, v)
    ) / (q + tau) - dot_zc / zc
    bar_rows = tf.constant(rng.standard_normal(n), DTYPE)
    bar_v, bar_e, bar_zc = retained_evaluator_adjoint(v, gram, tau, zc, bar_rows)
    lhs = _inner(bar_rows, dot_log)
    rhs = _inner(bar_v, dot_v) + _inner(bar_e, dot_e) + float(bar_zc.numpy()) * float(dot_zc.numpy())
    assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs))


def test_u_adj_node_1_prefix_rows_pairing() -> None:
    dimension, degree = 2, 5
    ranks = (1, 3, 2)
    basis = _basis(dimension, degree)
    cores = _cores(dimension, degree, ranks, SEED + 6)
    dot_cores = _cores(dimension, degree, ranks, SEED + 7)
    rng = np.random.default_rng(SEED + 8)
    points = tf.constant(rng.uniform(-1.0, 1.0, size=(25, dimension)), DTYPE)
    _rows, dot_rows = prefix_row_vectors_tangent(cores, dot_cores, basis, points)
    bar_rows = tf.constant(rng.standard_normal(dot_rows.shape), DTYPE)
    bar_cores = prefix_rows_adjoint(cores, basis, points, bar_rows)
    lhs = _inner(bar_rows, dot_rows)
    rhs = _inner_cores(bar_cores, dot_cores)
    assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs))


def test_u_adj_node_1_gram_chain_pairing() -> None:
    dimension, degree = 3, 5
    ranks = (2, 3, 2, 1)  # boundary rank 2 at the left end of the suffix
    basis = _basis(dimension, degree)
    cores = _cores(dimension, degree, ranks, SEED + 9)
    dot_cores = _cores(dimension, degree, ranks, SEED + 10)
    _gram, dot_gram = suffix_gram_matrix_tangent(
        cores, dot_cores, basis, axis_offset=0
    )
    rng = np.random.default_rng(SEED + 11)
    raw = rng.standard_normal(dot_gram.shape)
    bar_gram = tf.constant(0.5 * (raw + raw.T), DTYPE)
    bar_cores = gram_chain_adjoint(cores, basis, axis_offset=0, bar_gram=bar_gram)
    lhs = _inner(bar_gram, dot_gram)
    rhs = _inner_cores(bar_cores, dot_cores)
    assert abs(lhs - rhs) <= 1e-10 * max(1.0, abs(lhs))


def test_u_adj_node_1_cholesky_vjp_matches_fd() -> None:
    rng = np.random.default_rng(SEED + 12)
    r = 4
    raw = rng.standard_normal((r, r))
    a = raw @ raw.T + r * np.eye(r)
    raw_da = rng.standard_normal((r, r))
    dot_a = 0.5 * (raw_da + raw_da.T)
    chol = tf.linalg.cholesky(tf.constant(a, DTYPE))
    step = 1e-6
    dot_l_fd = (
        tf.linalg.cholesky(tf.constant(a + step * dot_a, DTYPE))
        - tf.linalg.cholesky(tf.constant(a - step * dot_a, DTYPE))
    ) / (2.0 * step)
    bar_l = tf.constant(np.tril(rng.standard_normal((r, r))), DTYPE)
    bar_a = cholesky_vjp(chol, bar_l)
    lhs = _inner(bar_l, dot_l_fd)
    rhs = _inner(bar_a, tf.constant(dot_a, DTYPE))
    assert abs(lhs - rhs) <= 1e-6 * max(1.0, abs(lhs))
