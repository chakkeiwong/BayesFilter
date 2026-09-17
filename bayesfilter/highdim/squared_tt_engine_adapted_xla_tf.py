"""Complete native adapted-map TT value recurrence.

Preserves the triangular containment and truncation-mass formulas in the
August 20 design note, Sections 9--13. Frozen designs are prepared once;
initial fitting, maps, recurrent fitting and mass correction compile together.
This mechanical execution repair makes no new source-faithfulness claim.
"""

from __future__ import annotations

import weakref
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.retained_quadratic_form_tf import (
    prefix_gram_matrix,
    prefix_row_vectors,
    suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_engine_xla_tf import _fit_als_graph
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.tt_preparation_tf import frozen_design_rows

DTYPE = tf.float64
_STEP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def make_value_filter_branch_axis_adapted_xla(
    adapter,
    observation_shape,
    config: EngineConfig,
    *,
    map_kappa_prev: float = 3.0,
    map_kappa_current: float = 4.0,
    jit_compile: bool = True,
):
    if config.quadrature_order is not None:
        raise ValueError("adapted maps are defined for scattered rows only")

    n = adapter.state_dim
    horizon = int(observation_shape[0])
    if horizon < 1:
        raise ValueError("positive observation horizon required")
    current_basis = _product_basis(n, config.basis_degree)
    extended_basis = _product_basis(n+1, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    hw = tf.constant(config.coordinate_half_width, DTYPE)
    kappa_p_const = tf.constant(map_kappa_prev, DTYPE)
    kappa_c_const = tf.constant(map_kappa_current, DTYPE)
    tau = tf.constant(config.tau, DTYPE)
    log2 = tf.math.log(tf.constant(2.0, DTYPE))
    ridge = tf.constant(config.ridge, DTYPE)
    fitter = FixedTTFitter()
    per_adapter = _STEP_CACHE.setdefault(adapter, {})
    cache_key = (config, tuple(observation_shape), map_kappa_prev, map_kappa_current, jit_compile)
    if cache_key in per_adapter:
        return per_adapter[cache_key]

    def _make_transition_fit(mixed_basis, mixed_shapes, prefix_shapes):
        def transition_fit(
            prefix_values, gram, zc_prev, z_rows, z_weights, core0_values,
            y, m_c, l_cc, m_p_base, l_pc, l_pp, l_old, m_old,
        ):
            prefix_cores = tuple(
                TTCore(tf.reshape(v, s)) for v, s in zip(prefix_values, prefix_shapes)
            )
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(
                gram
                + tf.constant(config.branch_gram_floor, DTYPE)
                * floor_scale
                * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
            )
            branch_count = int(prefix_shapes[-1][-1]) + 1
            z_c = z_rows[:, :n]
            z_p = z_rows[:, n:]
            x_current = m_c[None, :] + tf.einsum("ij,nj->ni", l_cc, z_c)
            x_previous = (
                m_p_base[None, :]
                + tf.einsum("ij,nj->ni", l_pc, z_c)
                + tf.einsum("ij,nj->ni", l_pp, z_p)
            )
            z_old = tf.transpose(
                tf.linalg.triangular_solve(
                    l_old, tf.transpose(x_previous - m_old[None, :]), lower=True
                )
            )
            max_excess = tf.reduce_max(tf.abs(z_old))
            logdet_c = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_cc))))
            logdet_p = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_pp))))
            logdet_old = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_old))))
            conversion = logdet_c + tf.cast(n, DTYPE) * log2 + logdet_p - logdet_old
            log_g = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, y)
                + conversion
            )
            v_prev = tf.einsum(
                "na,ab->nb",
                prefix_row_vectors(prefix_cores, current_basis, z_old),
                chol,
            )
            tau_abs = tau * (zc_prev / (1.0 + tau))
            sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + tau_abs
            log_f = tf.math.log(sum_sq) + log_g
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_g_shifted = tf.exp(0.5 * (log_g - shift))
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
            new_gram = suffix_gram_matrix(tuple(cores[n:]), mixed_basis, axis_offset=n)
            p_gram = prefix_gram_matrix(tuple(cores[:n]), mixed_basis)
            z_h_new = tf.einsum("ab,ab->", p_gram, new_gram)
            zc_new = (1.0 + tau) * z_h_new
            log_increment_in = shift + tf.math.log(zc_new) - tf.math.log(zc_prev)
            return (
                [c.values for c in cores[:n]], new_gram, zc_new,
                log_increment_in, worst, rms, max_excess,
            )

        return transition_fit

    initial = tuple(c.values for c in _initial_tt_cores(n, basis_dim, config.rank))
    initial_shapes = tuple(tuple(v.shape) for v in initial)
    initial_rows = frozen_design_rows(config, config.row_count, n, tf.constant([0]), 17,
        jit_compile=jit_compile)[0]
    weights = tf.fill([config.row_count], tf.constant(1. / config.row_count, DTYPE))
    dates = tf.range(1, max(2, horizon))
    rows = frozen_design_rows(config, config.row_count, 2 * n, dates, 100, jit_compile=jit_compile)
    correction_rows = frozen_design_rows(config, config.row_count, 2 * n, dates, 500, jit_compile=jit_compile)

    def phase(previous_shapes, dates):
        from bayesfilter.highdim.tt_native_control_tf import random_core_starts
        branch_count = previous_shapes[-1][-1] + 1
        mixed_basis = ProductBasis(list(current_basis.bases) + [DiscreteIndicatorBasis1D(branch_count)]
            + list(current_basis.bases), current_basis.convention)
        dims = (basis_dim,) * n + (branch_count,) + (basis_dim,) * n
        shapes = tuple((1 if axis == 0 else config.rank, dim,
            1 if axis == 2*n else config.rank) for axis, dim in enumerate(dims))
        starts = random_core_starts(shapes, dates, config.seed, jit_compile=jit_compile)
        fit = _make_transition_fit(mixed_basis, shapes, previous_shapes)

        def step(t, state, y, joint_mean, joint_cov, core_starts):
            prefix, gram, zc, m_old, l_old = state
            joint_chol = tf.linalg.cholesky(joint_cov)
            m_c, m_p = joint_mean[:n], joint_mean[n:]
            l_cc = joint_chol[:n, :n] * kappa_c_const
            l_pc = joint_chol[n:, :n] * kappa_c_const
            l_pp = joint_chol[n:, n:] * kappa_p_const
            center = tf.linalg.triangular_solve(l_old, (m_p-m_old)[:, None], lower=True)[:, 0]
            transfer = tf.linalg.triangular_solve(l_old, tf.concat((l_pc, l_pp), axis=1), lower=True)
            rowsum = tf.reduce_sum(tf.abs(transfer), axis=1)
            slack = 1. - tf.constant(1e-9, DTYPE) - tf.abs(center)
            shrink = tf.minimum(tf.reduce_min(tf.where(rowsum > 0.,
                slack / tf.maximum(rowsum, 1e-300), tf.ones_like(rowsum))), tf.constant(1., DTYPE))
            l_pc, l_pp = l_pc * shrink, l_pp * shrink
            new_prefix, new_gram, new_zc, increment_in, worst, rms, excess = fit(
                prefix, gram, zc, rows[t-1], weights, core_starts,
                y, m_c, l_cc, m_p, l_pc, l_pp, l_old, m_old)
            corr = correction_rows[t-1]
            xc = m_c[None] + tf.einsum("ij,nj->ni", l_cc, corr[:, :n])
            xp = m_old[None] + tf.einsum("ij,nj->ni", l_old, corr[:, n:])
            zp = tf.transpose(tf.linalg.triangular_solve(l_pp,
                tf.transpose(xp-m_p[None]-tf.einsum("ij,nj->ni", l_pc, corr[:, :n])), lower=True))
            outside = tf.cast(tf.reduce_max(tf.abs(zp), axis=1) > 1., DTYPE)
            kernel = adapter.transition_log_density(xc, xp) + adapter.observation_log_density(xc, y)
            old_reference = tf.transpose(tf.linalg.solve(l_old,tf.transpose(xp-m_old[None])))
            vectors = prefix_row_vectors(tuple(TTCore(v) for v in prefix),current_basis,old_reference)
            density = (tf.einsum("na,ab,nb->n",vectors,gram,vectors) + tau*(zc/(1.+tau))) / zc
            # Every map here is lower triangular with a positive diagonal.
            density *= tf.exp(-tf.cast(n,DTYPE)*log2-tf.reduce_sum(tf.math.log(tf.linalg.diag_part(l_old))))
            integrand = outside * density * tf.exp(kernel)
            logdet_c = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_cc))))
            logdet_old = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_old))))
            volume = tf.exp(tf.cast(2*n, DTYPE) * log2 + logdet_c + logdet_old)
            log_out = tf.math.log(tf.maximum(volume * tf.reduce_mean(integrand), tf.constant(1e-300, DTYPE)))
            increment = tf.reduce_logsumexp(tf.stack((increment_in, log_out)))
            valid = (tf.reduce_all(tf.math.is_finite(joint_chol)) & tf.reduce_all(tf.abs(center) < 1.)
                & _retained_valid(new_gram,new_zc,tau))
            report = tf.stack((increment, worst, rms, shrink, excess,
                tf.exp(log_out - increment_in), tf.cast(valid, DTYPE)))
            return (tuple(new_prefix), new_gram, new_zc, m_c, l_cc), report
        return step, starts, shapes[:n]

    first_step, first_starts, prefix_shapes = phase(initial_shapes, tf.constant([1], tf.int32))
    later_step, later_starts, _ = phase(prefix_shapes, tf.range(2, max(3, horizon)))

    def evaluate(observations, joint_means, joint_covariances):
        x0 = initial_rows * hw
        log_f = (adapter.initial_log_density(x0) + adapter.observation_log_density(x0, observations[0])
            + tf.cast(n, DTYPE) * (tf.math.log(hw) + log2))
        shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
        cores, worst, rms = _fit_als_graph(fitter, current_basis, initial_rows,
            tf.exp(.5*(log_f-shift)), weights, initial, initial_shapes, config.sweeps, ridge)
        suffix = tf.tensor_scatter_nd_update(tf.zeros([cores[-1].right_rank, basis_dim, 1], DTYPE), [[0,0,0]], [1.])
        gram = suffix_gram_matrix((TTCore(suffix),),extended_basis,axis_offset=n)
        zc = (1.+tau)*tf.einsum("ab,ab->",prefix_gram_matrix(tuple(cores),current_basis),gram)
        total = shift + tf.math.log(zc)
        history = tf.TensorArray(DTYPE, size=horizon, element_shape=[7])
        history = history.write(0, tf.stack((total, worst, rms, tf.constant(1.,DTYPE),
            tf.constant(0.,DTYPE), tf.constant(0.,DTYPE), tf.cast(_retained_valid(gram,zc,tau),DTYPE))))
        if horizon > 1:
            state = (tuple(c.values for c in cores), gram, zc, tf.zeros([n],DTYPE), tf.eye(n,dtype=DTYPE)*hw)
            state, report = first_step(tf.constant(1), state, observations[1], joint_means[0], joint_covariances[0],
                tuple(v[0] for v in first_starts))
            total, history = total + report[0], history.write(1, report)
            def advance(t, state, total, history):
                state, report = later_step(t, state, observations[t], joint_means[t-1], joint_covariances[t-1],
                    tuple(v[t-2] for v in later_starts))
                return t+1, state, total+report[0], history.write(t, report)
            if horizon > 2:
                _, _, total, history = tf.while_loop(lambda t,*_: t < horizon, advance,
                    (tf.constant(2), state, total, history), maximum_iterations=horizon-2, parallel_iterations=1)
        return total, history.stack()
    call = tf.function(evaluate, input_signature=[tf.TensorSpec(observation_shape, DTYPE),
        tf.TensorSpec([horizon-1,2*n], DTYPE), tf.TensorSpec([horizon-1,2*n,2*n], DTYPE)],
        jit_compile=jit_compile, autograph=False)
    per_adapter[cache_key] = call
    return call


def run_value_filter_branch_axis_adapted_xla(adapter, observations, config, *,
    predictive_moment_hint: Callable, map_kappa_prev=3., map_kappa_current=4., jit_compile=True):
    """Evaluate tensor-native frozen hints, then one compiled numerical filter.

    Hints must accept a tensor date. Stateful Python/NumPy hint generators are
    reference-only and must prepare their frozen arrays before this endpoint.
    """
    observations = tf.convert_to_tensor(observations, DTYPE)
    n, horizon = adapter.state_dim, observations.shape[0]
    means, covariances = tf.map_fn(lambda t: predictive_moment_hint(t, observations[t]),
        tf.range(1,horizon), fn_output_signature=(tf.TensorSpec([2*n],DTYPE),
            tf.TensorSpec([2*n,2*n],DTYPE)), parallel_iterations=1)
    call = make_value_filter_branch_axis_adapted_xla(adapter, observations.shape, config,
        map_kappa_prev=map_kappa_prev, map_kappa_current=map_kappa_current, jit_compile=jit_compile)
    value, history = call(observations, means, covariances)
    finite = tf.reduce_all(tf.math.is_finite(history[:,0]))
    if not bool(finite.numpy()):
        raise ValueError("non-finite step increment or moment hint (fail-closed)")
    if bool(tf.reduce_any(history[:,1] > config.condition_number_veto).numpy()):
        raise ValueError("condition number veto in fixed ALS fit")
    if bool(tf.reduce_any((history[:,4] > 1.+1e-12) | (history[:,6] != 1.)).numpy()):
        raise ValueError("adapted-map containment violated or invalid moment hint")
    diagnostics = []
    for t,row in enumerate(history.numpy().tolist()):
        report = dict(time_index=t, log_increment=row[0], tie_flag=False,
            worst_condition=row[1], weighted_fit_rms=row[2])
        if t:
            report.update(map_shrink=row[3], z_old_max=row[4], truncation_mass_ratio=row[5])
        diagnostics.append(report)
    return value, diagnostics


__all__ = ["make_value_filter_branch_axis_adapted_xla", "run_value_filter_branch_axis_adapted_xla"]


def _retained_valid(gram,zc,tau):
    scale = tf.maximum(tf.reduce_max(tf.abs(gram)),tf.constant(1.,DTYPE))
    return ((tf.reduce_max(tf.abs(gram-tf.transpose(gram))) <= tf.constant(1e-12,DTYPE)*scale)
        & (tau >= 0.) & (zc > 0.) & tf.math.is_finite(zc))
