"""XLA-compiled C2 Gaussian-reference value filter (Phase A2 of the
reviewed completion-campaign plan, 2026-08-24).

Program identical to `run_value_filter_branch_axis_gaussian` (the
REVIEWED C2 note incl. §3b row law and clamped τ); execution identical
in structure to the adapted XLA engine: host filter loop + one
jit-compiled transition step per branch-count signature, `_fit_als_graph`
solve backend. Host side: hints, Christoffel rows/weights, τ clamp,
increment accounting, diagnostics. Compiled side: target assembly with
the row-dependent η-ratio conversion, retained re-expression, ALS fit,
Gram contractions. Parity vs the eager Gaussian engine is a MEASURED
Gate-A criterion (≤ 1e-12), not a bit-identity claim.
"""

from __future__ import annotations

import math
import weakref
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_gram_matrix,
    prefix_row_vectors,
    retained_quadratic_form_from_squared_tt,
    suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    _check_hint,
    _christoffel_rows,
    _clamped_tau,
    _hermite_product_basis,
    _log_eta,
    _log_student_t_ratio,
    _logdet_lower,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _fixed_als_fit,
    _initial_tt_cores,
)
from bayesfilter.highdim.squared_tt_engine_xla_tf import _fit_als_graph
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.bases import ProductBasis

DTYPE = tf.float64
_STEP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def run_value_filter_branch_axis_gaussian_xla(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_moment_hint: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    defensive_nu: float | None = None,
) -> tuple[tf.Tensor, list[dict]]:
    if config.quadrature_order is not None:
        raise ValueError("gaussian engine is defined for scattered rows only")
    if predictive_moment_hint is None or initial_moment_hint is None:
        raise ValueError("gaussian engine requires frozen moment hints")

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _hermite_product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    ridge = tf.constant(config.ridge, DTYPE)
    fitter = FixedTTFitter()
    per_adapter = _STEP_CACHE.setdefault(adapter, {})
    step_cache = per_adapter.setdefault(config, {})

    def _make_transition_fit(mixed_basis, mixed_shapes, prefix_shapes):
        @tf.function(jit_compile=True)
        def transition_fit(
            prefix_values, gram, tau_abs_prev, u_rows, u_weights, core0_values,
            y, m_c, l_cc, m_p, l_pc, l_pp, l_old, m_old,
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
            u_c = u_rows[:, :n]
            u_p = u_rows[:, n:]
            x_current = m_c[None, :] + tf.einsum("ij,nj->ni", l_cc, u_c)
            x_previous = (
                m_p[None, :]
                + tf.einsum("ij,nj->ni", l_pc, u_c)
                + tf.einsum("ij,nj->ni", l_pp, u_p)
            )
            u_old = tf.transpose(
                tf.linalg.triangular_solve(
                    l_old, tf.transpose(x_previous - m_old[None, :]), lower=True
                )
            )
            u_old_max = tf.reduce_max(tf.abs(u_old))
            logdet_c = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_cc))))
            logdet_p = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_pp))))
            logdet_old = tf.reduce_sum(
                tf.math.log(tf.abs(tf.linalg.diag_part(l_old)))
            )
            # reviewed note §1: row-dependent eta-ratio conversion
            conversion = (
                logdet_c
                + logdet_p
                - logdet_old
                + _log_eta(u_old)
                - _log_eta(u_rows)
            )
            log_g = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, y)
                + conversion
            )
            v_prev = tf.einsum(
                "na,ab->nb",
                prefix_row_vectors(prefix_cores, current_basis, u_old),
                chol,
            )
            if defensive_nu is None:
                floor_values = tau_abs_prev * tf.ones(
                    [int(u_rows.shape[0])], DTYPE
                )
            else:
                floor_values = tau_abs_prev * tf.exp(
                    _log_student_t_ratio(u_old, defensive_nu)
                )
            sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + floor_values
            log_f = tf.math.log(sum_sq) + log_g
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_g_shifted = tf.exp(0.5 * (log_g - shift))
            amplitudes = tf.concat(
                [v_prev, tf.sqrt(floor_values)[:, None]],
                axis=1,
            )
            targets = amplitudes * sqrt_g_shifted[:, None]
            g_codes = tf.tile(
                tf.range(branch_count, dtype=DTYPE)[None, :],
                [int(u_rows.shape[0]), 1],
            )
            full_rows = tf.concat(
                [
                    tf.repeat(u_rows[:, :n], branch_count, axis=0),
                    tf.reshape(g_codes, [-1, 1]),
                    tf.repeat(u_rows[:, n:], branch_count, axis=0),
                ],
                axis=1,
            )
            sqrt_target = tf.reshape(targets, [-1])
            weights = tf.reshape(tf.repeat(u_weights, branch_count, axis=0), [-1])
            cores, worst, rms = _fit_als_graph(
                fitter, mixed_basis, full_rows, sqrt_target, weights,
                core0_values, mixed_shapes, config.sweeps, ridge,
            )
            new_gram = suffix_gram_matrix(
                tuple(cores[n:]), mixed_basis, axis_offset=n
            )
            p_gram = prefix_gram_matrix(tuple(cores[:n]), mixed_basis)
            z_h_new = tf.einsum("ab,ab->", p_gram, new_gram)
            return (
                [c.values for c in cores[:n]], new_gram, z_h_new,
                shift, worst, rms, u_old_max,
            )

        return transition_fit

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[dict] = []
    prefix_values = None
    gram = None
    zc = None

    for t in range(horizon):
        if t == 0:
            # t=0: eager (cheap n-axis fit), identical to the eager engine
            m_c0, l_cc0 = _check_hint(*initial_moment_hint(observations[0]), n)
            map_c = AffineCoordinateMap(offset=m_c0, matrix=l_cc0)
            rows, weights, row_ess = _christoffel_rows(
                config, config.row_count, n, (config.seed, 17), config.basis_degree
            )
            x0 = m_c0[None, :] + tf.einsum("ij,nj->ni", l_cc0, rows)
            conversion = _logdet_lower(l_cc0) - _log_eta(rows)
            log_f = (
                adapter.initial_log_density(x0)
                + adapter.observation_log_density(x0, observations[t])
                + conversion
            )
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_target = tf.exp(0.5 * (log_f - shift))
            cores, fit_diag = _fixed_als_fit(
                current_basis, rows, sqrt_target, weights,
                _initial_tt_cores(n, basis_dim, config.rank), config,
            )
            suffix_core = tf.zeros([int(cores[-1].right_rank), basis_dim, 1], DTYPE)
            suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
            extended = tuple(cores) + (TTCore(suffix_core),)
            extended_basis = _hermite_product_basis(n + 1, config.basis_degree)
            base = retained_quadratic_form_from_squared_tt(
                extended, extended_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=map_c,
            )
            z_h_new = base.z_complete_ref
            tau_t, eps_rel_sq = _clamped_tau(fit_diag["weighted_fit_rms"], z_h_new)
            retained = RetainedQuadraticForm(
                prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
                tau=tau_t * z_h_new, z_complete_ref=(1.0 + tau_t) * z_h_new,
                prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
            )
            prefix_values = [c.values for c in base.prefix_cores]
            gram = base.suffix_gram
            zc = retained.z_complete_ref
            log_increment = shift + tf.math.log(retained.z_complete_ref)
            log_likelihood += log_increment
            diagnostics.append(
                {
                    "time_index": 0,
                    "log_increment": float(log_increment.numpy()),
                    "tau_t": float(tau_t.numpy()),
                    "eps_rel_sq": eps_rel_sq,
                    "row_ess": row_ess,
                    "tie_flag": False,
                    **fit_diag,
                }
            )
            continue

        # ---- host-side step setup (maps, rows, cache) ----
        map_old = retained.coordinate_map
        l_old = tf.convert_to_tensor(map_old.matrix, DTYPE)
        m_old = tf.convert_to_tensor(map_old.offset, DTYPE)
        joint_mean, joint_chol = _check_hint(
            *predictive_moment_hint(t, observations[t]), 2 * n
        )
        m_c = joint_mean[:n]
        m_p = joint_mean[n:]
        l_cc = joint_chol[:n, :n]
        l_pc = joint_chol[n:, :n]
        l_pp = joint_chol[n:, n:]
        map_c_new = AffineCoordinateMap(offset=m_c, matrix=l_cc)
        u_rows, u_weights, row_ess = _christoffel_rows(
            config, config.row_count, 2 * n, (config.seed, 100 + t),
            config.basis_degree,
        )
        branch_count = retained.boundary_rank + 1
        cache_key = (branch_count, defensive_nu)
        if cache_key not in step_cache:
            mixed_basis = ProductBasis(
                list(current_basis.bases)
                + [DiscreteIndicatorBasis1D(branch_count)]
                + list(_hermite_product_basis(n, config.basis_degree).bases),
                current_basis.convention,
            )
            mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
            mixed_shapes = [
                (
                    1 if axis == 0 else config.rank,
                    mixed_dims[axis],
                    1 if axis == 2 * n else config.rank,
                )
                for axis in range(2 * n + 1)
            ]
            prefix_shapes = [tuple(v.shape.as_list()) for v in prefix_values]
            step_cache[cache_key] = (
                _make_transition_fit(mixed_basis, mixed_shapes, prefix_shapes),
                mixed_shapes,
            )
        transition_fit, mixed_shapes = step_cache[cache_key]
        core0_values = [
            0.3
            * tf.random.stateless_normal(
                list(shape),
                tf.constant((config.seed, 7000 + 31 * t + axis), tf.int32),
                dtype=DTYPE,
            )
            for axis, shape in enumerate(mixed_shapes)
        ]
        prefix_values_new, gram_new, z_h_new, shift, worst, rms, u_old_max = (
            transition_fit(
                prefix_values, gram, retained.tau, u_rows, u_weights,
                core0_values, observations[t], m_c, l_cc, m_p, l_pc, l_pp,
                l_old, m_old,
            )
        )
        tau_t, eps_rel_sq = _clamped_tau(float(rms.numpy()), z_h_new)
        zc_new = (1.0 + tau_t) * z_h_new
        log_increment = shift + tf.math.log(zc_new) - tf.math.log(zc)
        increment_value = float(log_increment.numpy())
        if not math.isfinite(increment_value):
            raise ValueError("non-finite step increment (fail-closed)")
        retained = RetainedQuadraticForm(
            prefix_cores=tuple(TTCore(v) for v in prefix_values_new),
            suffix_gram=gram_new,
            tau=tau_t * z_h_new,
            z_complete_ref=zc_new,
            prefix_basis=current_basis,
            coordinate_map=map_c_new,
        )
        prefix_values = list(prefix_values_new)
        gram = gram_new
        zc = zc_new
        log_likelihood += log_increment
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": increment_value,
                "tau_t": float(tau_t.numpy()),
                "eps_rel_sq": eps_rel_sq,
                "row_ess": row_ess,
                "tie_flag": False,
                "worst_condition": float(worst.numpy()),
                "weighted_fit_rms": float(rms.numpy()),
                "u_old_max": float(u_old_max.numpy()),
            }
        )
    return log_likelihood, diagnostics


__all__ = ["run_value_filter_branch_axis_gaussian_xla"]
