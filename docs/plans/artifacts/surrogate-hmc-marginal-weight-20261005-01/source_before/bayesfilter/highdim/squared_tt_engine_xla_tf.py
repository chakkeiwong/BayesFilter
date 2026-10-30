"""Complete XLA branch-axis TT value recurrence with native time and ALS loops.

The scaled CholeskyQR2 solver and eigvalsh condition estimator are unchanged.
Frozen random designs have their own compiled preparation preserving their
streams. The initial fit, first rank-changing transition, and remaining fixed-
shape dates run in one compiled endpoint. Condition and finite vetoes are
applied at its return boundary; Python formats only the diagnostic records.
"""

from __future__ import annotations

import math
import weakref

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.fitting import (
    _DEFAULT_COLUMN_SCALE_FLOOR,
    FixedTTFitter,
    _weighted_column_scales,
)
from bayesfilter.highdim.retained_quadratic_form_tf import (
    prefix_gram_matrix,
    prefix_row_vectors,
    suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _gauss_rows,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_native_control_tf import random_core_start_program
from bayesfilter.highdim.tt_preparation_tf import frozen_design_rows

DTYPE = tf.float64

# compiled-step cache: adapter (weak) -> {config: {"init": fn, branch_count: fn}}.
# Adapter closures are baked into traced graphs, so the cache must not
# outlive the adapter object; EngineConfig is a frozen dataclass (hashable).
_STEP_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _tall_r_factor(matrix):
    """(Q, R, Gram) of a tall matrix by CholeskyQR2 (matmul-dominated).

    XLA's blocked-Householder QR on CPU is both slow AND inaccurate for
    wide fits (measured P3.3: 2.4e-4 value error at 208 columns);
    CholeskyQR2 gives near-Householder solution accuracy for scaled
    systems (post-scaling condition measured <= ~1e4 in fit regimes;
    second pass restores orthogonality to O(eps)).
    """

    gram = tf.matmul(matrix, matrix, transpose_a=True)
    r1 = tf.linalg.cholesky(gram)  # lower; R1 = r1^T
    q1 = tf.transpose(
        tf.linalg.triangular_solve(r1, tf.transpose(matrix), lower=True)
    )
    gram2 = tf.matmul(q1, q1, transpose_a=True)
    r2 = tf.linalg.cholesky(gram2)
    q2 = tf.transpose(tf.linalg.triangular_solve(r2, tf.transpose(q1), lower=True))
    r_factor = tf.matmul(tf.transpose(r2), tf.transpose(r1))
    return q2, r_factor, gram


def _solve_scaled_qr(design, weights, target, ridge):
    """Scaled augmented ridge solve by CholeskyQR2 + eigvalsh condition.

    Same system as `_solve_scaled_augmented_ridge`; different (equivalent)
    least-squares backend — see module docstring. Backend equivalence is
    measured end-to-end by the P3.3 parity gate (2.8e-14 worst).

    Condition estimate: sqrt of the Gram eigenvalue ratio == the repo's
    SVD condition (validated 1.4e-14 rel at cond ~1e4); the tall-matrix
    SVD OOMs under XLA and the [cols, cols] SVD costs 6.6 s/call vs
    92 ms for eigvalsh (measured). ESTIMATOR CEILING ~1e8: beyond it the
    Gram Cholesky itself fails and the step goes non-finite — the host
    loop's finite check is the fail-closed backstop in that regime.
    """

    scales, _norms, _floor = _weighted_column_scales(
        design, weights, _DEFAULT_COLUMN_SCALE_FLOOR
    )
    scaled = design / scales[None, :]
    sqrt_w = tf.sqrt(weights)
    augmented = tf.concat(
        [scaled * sqrt_w[:, None], tf.linalg.diag(tf.sqrt(ridge) / scales)], axis=0
    )
    rhs = tf.concat([target * sqrt_w, tf.zeros_like(scales)], axis=0)
    q, r_factor, gram = _tall_r_factor(augmented)
    y = tf.linalg.matvec(q, rhs, transpose_a=True)
    z = tf.linalg.triangular_solve(r_factor, y[:, None], lower=False)[:, 0]
    eigenvalues = tf.linalg.eigvalsh(gram)
    condition = tf.sqrt(
        eigenvalues[-1] / tf.maximum(eigenvalues[0], tf.constant(1e-300, DTYPE))
    )
    return z / scales, condition


def _fit_als_graph(fitter, basis, rows, target, weights, core_values, shapes, sweeps, ridge):
    """Native fixed-schedule ALS; static branches preserve heterogeneous cores."""

    count = len(core_values)
    def branch(axis, values, worst):
        def update():
            cores = tuple(TTCore(tf.reshape(v, shape)) for v, shape in zip(values, shapes))
            design = fitter._build_design_matrix(basis, rows, cores, axis)
            solution, condition = _solve_scaled_qr(design, weights, target, ridge)
            updated = values[:axis] + (tf.reshape(solution, shapes[axis]),) + values[axis + 1:]
            return updated, tf.maximum(worst, condition)
        return update
    def body(index, values, worst):
        branches = tuple(branch(axis, values, worst) for axis in range(count))
        values, worst = tf.switch_case(index % count, branches)
        return index + 1, values, worst
    _, values, worst = tf.while_loop(lambda i, *_: i < sweeps * count, body,
        (tf.constant(0), tuple(core_values), tf.constant(0.0, DTYPE)),
        parallel_iterations=1, maximum_iterations=sweeps * count)
    cores = [TTCore(value) for value in values]
    design = fitter._build_design_matrix(basis, rows, tuple(cores), len(cores) - 1)
    residual = tf.linalg.matvec(design, tf.reshape(cores[-1].values, [-1])) - target
    rms = tf.sqrt(tf.reduce_sum(weights * tf.square(residual)) / tf.reduce_sum(weights))
    return cores, worst, rms


def make_value_filter_branch_axis_xla(
    adapter, observation_shape, config: EngineConfig, *, jit_compile=True,
):
    """Bind a complete fixed-signature TT value recurrence.

    Rows and random starting cores are prepared once with native TF loops, so
    the random streams match the existing non-XLA preparation exactly. Each
    core's static shape is a branch specialization, not a numerical Python loop.
    The returned numerical endpoint emits all veto inputs as tensor histories.
    """
    n = adapter.state_dim
    horizon = int(observation_shape[0])
    if horizon < 1:
        raise ValueError("positive observation horizon required")
    cache = _STEP_CACHE.setdefault(adapter, {}).setdefault(config, {})
    key = ("complete", tuple(observation_shape), bool(jit_compile))
    if key in cache:
        return cache[key]
    basis = _product_basis(n, config.basis_degree)
    extended = _product_basis(n + 1, config.basis_degree)
    basis_dim = int(basis.bases[0].basis_dim)
    half = tf.constant(config.coordinate_half_width, DTYPE)
    conversion = tf.cast(n, DTYPE) * (tf.math.log(half) + tf.math.log(tf.constant(2., DTYPE)))
    tau = tf.constant(config.tau, DTYPE)
    ridge = tf.constant(config.ridge, DTYPE)
    fitter = FixedTTFitter()
    initial = tuple(c.values for c in _initial_tt_cores(n, basis_dim, config.rank))
    initial_shapes = tuple(tuple(v.shape) for v in initial)
    if config.quadrature_order is not None:
        initial_rows, initial_weights = _gauss_rows(n, config.quadrature_order, jit_compile=jit_compile)
        rows, z_weights = _gauss_rows(2 * n, config.quadrature_order, jit_compile=jit_compile)
        all_rows = tf.repeat(rows[None], max(1, horizon - 1), axis=0)
    else:
        initial_rows = frozen_design_rows(config, config.row_count, n, tf.constant([0]), 17,
            jit_compile=jit_compile)[0]
        initial_weights = tf.fill([config.row_count], tf.constant(1. / config.row_count, DTYPE))
        all_rows = frozen_design_rows(config, config.row_count, 2 * n,
            tf.range(1, max(2, horizon)), 100, jit_compile=jit_compile)
        z_weights = tf.fill([config.row_count], tf.constant(1. / config.row_count, DTYPE))

    def phase(previous_shapes, dates):
        branch_count = previous_shapes[-1][-1] + 1
        mixed_basis = ProductBasis(list(basis.bases) + [DiscreteIndicatorBasis1D(branch_count)]
                                   + list(basis.bases), basis.convention)
        dims = (basis_dim,) * n + (branch_count,) + (basis_dim,) * n
        shapes = tuple((1 if a == 0 else config.rank, dim,
                        1 if a == 2 * n else config.rank) for a, dim in enumerate(dims))
        starts = random_core_start_program(shapes, dates.shape[0], jit_compile=jit_compile)(
            dates, tf.constant(config.seed, tf.int32))

        def transition(t, prefix, gram, previous_zc, y, packed):
            prefix_cores = tuple(TTCore(value) for value in prefix)
            eigenvalues = tf.linalg.eigvalsh(gram)
            gram_condition = eigenvalues[-1] / tf.maximum(eigenvalues[0], tf.constant(1e-300, DTYPE))
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(gram + tf.constant(config.branch_gram_floor, DTYPE)
                                      * floor_scale * tf.eye(tf.shape(gram)[0], dtype=DTYPE))
            rows = all_rows[t - 1]
            x_current, z_previous = rows[:, :n] * half, rows[:, n:]
            log_g = adapter.transition_log_density(x_current, z_previous * half) + adapter.observation_log_density(x_current, y) + conversion
            v_prev = tf.einsum("na,ab->nb", prefix_row_vectors(prefix_cores, basis, z_previous), chol)
            tau_abs = tau * (previous_zc / (1. + tau))
            log_f = tf.math.log(tf.reduce_sum(tf.square(v_prev), axis=1) + tau_abs) + log_g
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
            sqrt_g = tf.exp(.5 * (log_g - shift))
            amplitudes = tf.concat((v_prev, tf.ones([rows.shape[0], 1], DTYPE) * tf.sqrt(tau_abs)), axis=1)
            target = tf.reshape(amplitudes * sqrt_g[:, None], [-1])
            codes = tf.tile(tf.range(branch_count, dtype=DTYPE)[None], [rows.shape[0], 1])
            full_rows = tf.concat((tf.repeat(rows[:, :n], branch_count, axis=0),
                                  tf.reshape(codes, [-1, 1]), tf.repeat(rows[:, n:], branch_count, axis=0)), axis=1)
            weights = tf.reshape(tf.repeat(z_weights, branch_count, axis=0), [-1])
            start = tuple(tf.reshape(packed[a, :math.prod(shape)], shape) for a, shape in enumerate(shapes))
            cores, worst, rms = _fit_als_graph(fitter, mixed_basis, full_rows, target, weights,
                                              start, shapes, config.sweeps, ridge)
            new_gram = suffix_gram_matrix(tuple(cores[n:]), mixed_basis, axis_offset=n)
            prefix_gram = prefix_gram_matrix(tuple(cores[:n]), mixed_basis)
            zc = (1. + tau) * tf.einsum("ab,ab->", prefix_gram, new_gram)
            increment = shift + tf.math.log(zc) - tf.math.log(previous_zc)
            return tuple(core.values for core in cores[:n]), new_gram, zc, tf.stack((increment, worst, rms, gram_condition))
        return transition, starts, shapes[:n]

    first_transition, first_starts, prefix_shapes = phase(initial_shapes, tf.constant([1], tf.int32))
    transition, starts, _ = phase(prefix_shapes, tf.range(2, max(3, horizon)))

    def evaluate(observations):
        log_f = adapter.initial_log_density(initial_rows * half) + adapter.observation_log_density(initial_rows * half, observations[0]) + conversion
        shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
        target = tf.exp(.5 * (log_f - shift))
        cores, worst, rms = _fit_als_graph(fitter, basis, initial_rows, target, initial_weights,
                                          initial, initial_shapes, config.sweeps, ridge)
        suffix = tf.tensor_scatter_nd_update(tf.zeros([cores[-1].right_rank, basis_dim, 1], DTYPE), [[0, 0, 0]], [1.])
        gram = suffix_gram_matrix((TTCore(suffix),), extended, axis_offset=n)
        zc = (1. + tau) * tf.einsum("ab,ab->", prefix_gram_matrix(tuple(cores), extended), gram)
        total = shift + tf.math.log(zc)
        history = tf.TensorArray(DTYPE, size=horizon, element_shape=[4], clear_after_read=False)
        history = history.write(0, tf.stack((total, worst, rms, tf.constant(1., DTYPE))))
        if horizon > 1:
            prefix, gram, zc, diagnostic = first_transition(tf.constant(1), tuple(c.values for c in cores), gram, zc, observations[1], first_starts[0])
            total += diagnostic[0]
            history = history.write(1, diagnostic)
            def body(t, prefix, gram, zc, total, history):
                prefix, gram, zc, diagnostic = transition(t, prefix, gram, zc, observations[t], starts[t - 2])
                return t + 1, prefix, gram, zc, total + diagnostic[0], history.write(t, diagnostic)
            if horizon > 2:
                _, _, _, _, total, history = tf.while_loop(lambda t, *_: t < horizon, body,
                    (tf.constant(2), prefix, gram, zc, total, history),
                    maximum_iterations=horizon - 2, parallel_iterations=1)
        return total, history.stack()
    compiled = tf.function(evaluate, input_signature=[tf.TensorSpec(observation_shape, DTYPE)],
                           jit_compile=jit_compile, autograph=False)
    cache[key] = compiled
    return compiled


def run_value_filter_branch_axis_xla(
    adapter, observations: tf.Tensor, config: EngineConfig, *, gram_condition_veto=None, jit_compile=True,
) -> tuple[tf.Tensor, list[dict]]:
    """Complete XLA value filter, followed by host veto/reporting only."""
    observations = tf.convert_to_tensor(observations, DTYPE)
    call = make_value_filter_branch_axis_xla(adapter, observations.shape, config, jit_compile=jit_compile)
    value, history = call(observations)
    # Fail closed at the return boundary: invalid states never reach callers.
    if not bool(tf.reduce_all(tf.math.is_finite(history[:, :3])).numpy()):
        raise ValueError("non-finite step increment or ALS diagnostic (fail-closed)")
    if bool(tf.reduce_any(history[:, 1] > config.condition_number_veto).numpy()):
        raise ValueError("condition number veto in fixed ALS fit")
    if gram_condition_veto is not None and bool(tf.reduce_any(history[1:, 3] > gram_condition_veto).numpy()):
        raise ValueError("retained Gram conditioning veto requested by caller")
    diagnostics = []
    for t, row in enumerate(history.numpy().tolist()):
        record = {"time_index": t, "log_increment": row[0], "tie_flag": False,
                  "worst_condition": row[1], "weighted_fit_rms": row[2]}
        if t:
            record["gram_condition"] = row[3]
        diagnostics.append(record)
    return value, diagnostics


__all__ = ["make_value_filter_branch_axis_xla", "run_value_filter_branch_axis_xla"]
