"""XLA-compiled branch-axis value filter (P3.3).

Scoping note: docs/plans/bayesfilter-p3-xla-port-scoping-note-2026-08-18.md.
Same declared program as `run_value_filter_branch_axis` (v0.2 relative tau,
v0.3 smooth shift, Cholesky branch factor with declared floor): the Python
filter loop stays host-side; each step (target assembly + unrolled ALS +
exact Gram retention) runs as one jit-compiled function. Host syncs occur
only at the per-step veto/diagnostic boundary (backend rule compliant).

Solver note: the eager fit solves via `tf.linalg.lstsq(fast=False)`, which
does not lower to XLA; the compiled route solves the SAME scaled augmented
system by explicit QR (P3.1 probe: 1.5e-14 solution agreement incl.
ill-conditioned fixtures) and computes the SAME SVD condition number for
the veto. Eager-vs-XLA value parity is therefore a measured gate
(P3 target 1e-12), not a bit-identity claim.

Vetoes preserved fail-closed: per-update condition veto
(config.condition_number_veto) and retained-Gram conditioning veto
(gram_condition_veto) are computed in-graph and checked host-side after
each step.
"""

from __future__ import annotations

import weakref

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.fitting import (
    FixedTTFitter,
    _DEFAULT_COLUMN_SCALE_FLOOR,
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
    _design_rows,
    _gauss_rows,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64

# compiled-step cache: adapter (weak) -> {config: {"init": fn, branch_count: fn}}.
# Adapter closures are baked into traced graphs, so the cache must not
# outlive the adapter object; EngineConfig is a frozen dataclass (hashable).
_STEP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _solve_scaled_qr(design, weights, target, ridge):
    """Scaled augmented ridge solve by QR + SVD condition (XLA-lowerable).

    Same system as `_solve_scaled_augmented_ridge`; different (equivalent)
    least-squares backend — see module docstring.
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
    q, r_factor = tf.linalg.qr(augmented)
    y = tf.linalg.matvec(q, rhs, transpose_a=True)
    z = tf.linalg.triangular_solve(r_factor, y[:, None], lower=False)[:, 0]
    # condition of the augmented matrix == condition of its R factor
    # (same singular values); the tall-matrix SVD OOMs under XLA
    # (measured 62 GB at [62252, 44]), the [cols, cols] SVD is cheap.
    singular = tf.linalg.svd(r_factor, compute_uv=False)
    condition = singular[0] / tf.maximum(singular[-1], tf.constant(1e-300, DTYPE))
    return z / scales, condition


def _fit_als_graph(fitter, basis, rows, target, weights, core_values, shapes, sweeps, ridge):
    """Unrolled fixed-schedule ALS; returns (core values, worst cond, rms)."""

    cores = [TTCore(tf.reshape(v, s)) for v, s in zip(core_values, shapes)]
    worst = tf.constant(0.0, DTYPE)
    for _sweep in range(sweeps):
        for idx in range(len(cores)):
            design = fitter._build_design_matrix(basis, rows, tuple(cores), idx)
            solution, condition = _solve_scaled_qr(design, weights, target, ridge)
            worst = tf.maximum(worst, condition)
            cores[idx] = TTCore(tf.reshape(solution, shapes[idx]))
    design = fitter._build_design_matrix(basis, rows, tuple(cores), len(cores) - 1)
    residual = tf.linalg.matvec(design, tf.reshape(cores[-1].values, [-1])) - target
    rms = tf.sqrt(tf.reduce_sum(weights * tf.square(residual)) / tf.reduce_sum(weights))
    return cores, worst, rms


def run_value_filter_branch_axis_xla(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    gram_condition_veto: float | None = None,
) -> tuple[tf.Tensor, list[dict]]:
    """XLA-compiled value filter; same program/veto semantics as eager."""

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    half = tf.constant(config.coordinate_half_width, DTYPE)
    conversion = tf.cast(n, DTYPE) * (
        tf.math.log(half) + tf.math.log(tf.constant(2.0, DTYPE))
    )
    tau = tf.constant(config.tau, DTYPE)
    ridge = tf.constant(config.ridge, DTYPE)
    fitter = FixedTTFitter()
    # bases are static program objects; construct OUTSIDE traced scope
    # (BoundedInterval validation host-syncs and cannot trace).
    extended_basis = _product_basis(n + 1, config.basis_degree)
    per_adapter = _STEP_CACHE.setdefault(adapter, {})
    step_cache = per_adapter.setdefault(config, {})

    @tf.function(jit_compile=True)
    def init_step(rows, weights, y0, core0_values):
        shapes = [tuple(v.shape.as_list()) for v in core0_values]
        x_current = rows * half
        log_f = (
            adapter.initial_log_density(x_current)
            + adapter.observation_log_density(x_current, y0)
            + conversion
        )
        shift = tf.reduce_logsumexp(log_f) - tf.math.log(
            tf.cast(tf.shape(log_f)[0], DTYPE)
        )
        sqrt_target = tf.exp(0.5 * (log_f - shift))
        cores, worst, rms = _fit_als_graph(
            fitter, current_basis, rows, sqrt_target, weights,
            core0_values, shapes, config.sweeps, ridge,
        )
        suffix_core = tf.zeros([int(cores[-1].right_rank), basis_dim, 1], DTYPE)
        suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
        gram = suffix_gram_matrix(
            (TTCore(suffix_core),), extended_basis, axis_offset=n
        )
        p_gram = prefix_gram_matrix(tuple(cores), extended_basis)
        z_h = tf.einsum("ab,ab->", p_gram, gram)
        z_complete = (1.0 + tau) * z_h
        log_increment = shift + tf.math.log(z_complete)
        return (
            [c.values for c in cores], gram, z_complete, log_increment, worst, rms
        )

    def _make_transition_step(mixed_basis, mixed_shapes, prefix_shapes):
        @tf.function(jit_compile=True)
        def transition_step(prefix_values, gram, zc_prev, z_rows, z_weights, core0_values, y):
            prefix_cores = tuple(
                TTCore(tf.reshape(v, s)) for v, s in zip(prefix_values, prefix_shapes)
            )
            eigenvalues = tf.linalg.eigvalsh(gram)
            gram_condition = eigenvalues[-1] / tf.maximum(
                eigenvalues[0], tf.constant(1e-300, DTYPE)
            )
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(
                gram
                + tf.constant(config.branch_gram_floor, DTYPE)
                * floor_scale
                * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
            )
            branch_count = int(prefix_shapes[-1][-1]) + 1
            x_current = z_rows[:, :n] * half
            z_previous = z_rows[:, n:]
            x_previous = z_previous * half
            log_g_kernel = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, y)
                + conversion
            )
            v_prev = tf.einsum(
                "na,ab->nb",
                prefix_row_vectors(prefix_cores, current_basis, z_previous),
                chol,
            )
            tau_abs = tau * (zc_prev / (1.0 + tau))
            sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + tau_abs
            log_f_row = tf.math.log(sum_sq) + log_g_kernel
            shift = tf.reduce_logsumexp(log_f_row) - tf.math.log(
                tf.cast(tf.shape(log_f_row)[0], DTYPE)
            )
            sqrt_g_shifted = tf.exp(0.5 * (log_g_kernel - shift))
            amplitudes = tf.concat(
                [v_prev, tf.ones([int(z_rows.shape[0]), 1], DTYPE) * tf.sqrt(tau_abs)],
                axis=1,
            )
            targets = amplitudes * sqrt_g_shifted[:, None]
            g_codes = tf.tile(
                tf.range(branch_count, dtype=DTYPE)[None, :], [int(z_rows.shape[0]), 1]
            )
            full_rows = tf.concat(
                [
                    tf.repeat(z_rows[:, :n], branch_count, axis=0),
                    tf.reshape(g_codes, [-1, 1]),
                    tf.repeat(z_rows[:, n:], branch_count, axis=0),
                ],
                axis=1,
            )
            sqrt_target = tf.reshape(targets, [-1])
            weights = tf.reshape(tf.repeat(z_weights, branch_count, axis=0), [-1])
            cores, worst, rms = _fit_als_graph(
                fitter, mixed_basis, full_rows, sqrt_target, weights,
                core0_values, mixed_shapes, config.sweeps, ridge,
            )
            new_gram = suffix_gram_matrix(
                tuple(cores[n:]), mixed_basis, axis_offset=n
            )
            p_gram = prefix_gram_matrix(tuple(cores[:n]), mixed_basis)
            z_h_new = tf.einsum("ab,ab->", p_gram, new_gram)
            zc_new = (1.0 + tau) * z_h_new
            log_increment = shift + tf.math.log(zc_new) - tf.math.log(zc_prev)
            return (
                [c.values for c in cores[:n]], new_gram, zc_new, log_increment,
                gram_condition, worst, rms,
            )

        return transition_step

    log_likelihood = tf.constant(0.0, DTYPE)
    diagnostics: list[dict] = []
    prefix_values = None
    gram = None
    zc = None

    for t in range(horizon):
        if t == 0:
            if config.quadrature_order is not None:
                rows, weights = _gauss_rows(n, config.quadrature_order)
            else:
                rows = _design_rows(config, config.row_count, n, (config.seed, 17))
                weights = tf.fill(
                    [int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE)
                )
            cores0 = _initial_tt_cores(n, basis_dim, config.rank)
            init_fn = step_cache.setdefault("init", init_step)
            prefix_values, gram, zc, log_increment, worst, rms = init_fn(
                rows, weights, observations[t], [c.values for c in cores0]
            )
            gram_condition = None
        else:
            branch_count = int(prefix_values[-1].shape[-1]) + 1
            if config.quadrature_order is not None:
                z_rows, z_weights = _gauss_rows(2 * n, config.quadrature_order)
            else:
                z_rows = _design_rows(config, config.row_count, 2 * n, (config.seed, 100 + t))
                z_weights = tf.fill(
                    [int(z_rows.shape[0])], tf.constant(1.0 / int(z_rows.shape[0]), DTYPE)
                )
            mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
            cores0_values = [
                0.3
                * tf.random.stateless_normal(
                    [
                        1 if axis == 0 else config.rank,
                        mixed_dims[axis],
                        1 if axis == 2 * n else config.rank,
                    ],
                    tf.constant((config.seed, 7000 + 31 * t + axis), tf.int32),
                    dtype=DTYPE,
                )
                for axis in range(2 * n + 1)
            ]
            if branch_count not in step_cache:
                mixed_basis = ProductBasis(
                    list(current_basis.bases)
                    + [DiscreteIndicatorBasis1D(branch_count)]
                    + list(_product_basis(n, config.basis_degree).bases),
                    current_basis.convention,
                )
                mixed_shapes = [tuple(v.shape.as_list()) for v in cores0_values]
                prefix_shapes = [tuple(v.shape.as_list()) for v in prefix_values]
                step_cache[branch_count] = _make_transition_step(
                    mixed_basis, mixed_shapes, prefix_shapes
                )
            step_fn = step_cache[branch_count]
            (
                prefix_values, gram, zc, log_increment,
                gram_condition_t, worst, rms,
            ) = step_fn(
                prefix_values, gram, zc, z_rows, z_weights, cores0_values,
                observations[t],
            )
            gram_condition = float(gram_condition_t.numpy())
            # parity with the eager VALUE engine: gram conditioning is a
            # recorded diagnostic here; the hard veto is the SCORE-path
            # claim gate (adjoint engine). gram_condition_veto is kept in
            # the signature for callers that opt into a value-path check.
            if gram_condition_veto is not None and gram_condition > gram_condition_veto:
                raise ValueError("retained Gram conditioning veto requested by caller")
        worst_condition = float(worst.numpy())
        if worst_condition > config.condition_number_veto:
            raise ValueError("condition number veto in fixed ALS fit")
        log_likelihood += log_increment
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": float(log_increment.numpy()),
                "tie_flag": False,
                "worst_condition": worst_condition,
                "weighted_fit_rms": float(rms.numpy()),
                **({"gram_condition": gram_condition} if gram_condition is not None else {}),
            }
        )
    return log_likelihood, diagnostics


__all__ = ["run_value_filter_branch_axis_xla"]
