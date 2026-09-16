"""XLA-compiled adapted (triangular + truncation-corrected) value filter.

attempt04 engine (plan: bayesfilter-p1b-attempt04-plan-2026-08-21.md).
Program identical to `run_value_filter_branch_axis_adapted` (design note
Sections 9-13); execution identical in structure to the P3.3 XLA engine:
host filter loop + one jit-compiled transition step per branch-count
signature, CholeskyQR2 solve backend, eigvalsh conditioning. Parity is a
MEASURED gate vs the eager adapted engine (1e-12 target on the n=2
fixture) — same discipline as P3.3, not a bit-identity claim.

Map/moment/correction work (joint Cholesky, containment shrink, retained
moments telemetry, truncation-mass MC) is cheap host-side setup per step;
only the fit + retention tensors compile.
"""

from __future__ import annotations

import weakref
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.fitting import FixedTTFitter
from bayesfilter.highdim.retained_moments_tf import retained_reference_moments
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_gram_matrix,
    prefix_row_vectors,
    retained_quadratic_form_from_squared_tt,
    suffix_gram_matrix,
)
from bayesfilter.highdim.squared_tt_engine_xla_tf import _fit_als_graph
from bayesfilter.highdim.squared_tt_engine_adapted_tf import _check_hint
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _design_rows,
    _fixed_als_fit,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64
_STEP_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def run_value_filter_branch_axis_adapted_xla(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    map_kappa_prev: float = 3.0,
    map_kappa_current: float = 4.0,
) -> tuple[tf.Tensor, list[dict]]:
    if config.quadrature_order is not None:
        raise ValueError("adapted maps are defined for scattered rows only")

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    hw = tf.constant(config.coordinate_half_width, DTYPE)
    kappa_p_const = tf.constant(map_kappa_prev, DTYPE)
    kappa_c_const = tf.constant(map_kappa_current, DTYPE)
    tau = tf.constant(config.tau, DTYPE)
    log2 = tf.math.log(tf.constant(2.0, DTYPE))
    ridge = tf.constant(config.ridge, DTYPE)
    fitter = FixedTTFitter()
    per_adapter = _STEP_CACHE.setdefault(adapter, {})
    step_cache = per_adapter.setdefault((config, map_kappa_prev, map_kappa_current), {})

    def _make_transition_fit(mixed_basis, mixed_shapes, prefix_shapes):
        @tf.function(jit_compile=True)
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

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[dict] = []
    prefix_values = None
    gram = None
    zc = None

    for t in range(horizon):
        if t == 0:
            # t=0: global box, eager (cheap n-axis fit; XLA gain negligible)
            rows = _design_rows(config, config.row_count, n, (config.seed, 17))
            weights = tf.fill(
                [int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE)
            )
            map_c = AffineCoordinateMap(
                offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE) * hw
            )
            x0 = rows * hw
            conversion = tf.cast(n, DTYPE) * (tf.math.log(hw) + log2)
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
            extended_basis = _product_basis(n + 1, config.basis_degree)
            base = retained_quadratic_form_from_squared_tt(
                extended, extended_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=map_c,
            )
            z_h_new = base.z_complete_ref
            retained = RetainedQuadraticForm(
                prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
                tau=tau * z_h_new, z_complete_ref=(1.0 + tau) * z_h_new,
                prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
            )
            prefix_values = [c.values for c in base.prefix_cores]
            gram = base.suffix_gram
            zc = retained.z_complete_ref
            log_likelihood += shift + tf.math.log(retained.z_complete_ref)
            diagnostics.append(
                {"time_index": 0, "log_increment": float(
                    (shift + tf.math.log(retained.z_complete_ref)).numpy()),
                 "tie_flag": False, **fit_diag}
            )
            continue

        # ---- host-side map construction (triangular; Sections 9-11) ----
        map_old = retained.coordinate_map
        l_old = tf.convert_to_tensor(map_old.matrix, DTYPE)
        m_old = tf.convert_to_tensor(map_old.offset, DTYPE)
        joint_mean, joint_chol = _check_hint(
            *predictive_moment_hint(t, observations[t]), 2 * n
        )
        m_c = joint_mean[:n]
        m_p_base = joint_mean[n:]
        l_cc = joint_chol[:n, :n] * kappa_c_const
        l_pc = joint_chol[n:, :n] * kappa_c_const
        l_pp = joint_chol[n:, n:] * kappa_p_const
        center = tf.linalg.triangular_solve(
            l_old, (m_p_base - m_old)[:, None], lower=True
        )[:, 0]
        if bool(tf.reduce_any(tf.abs(center) >= 1.0).numpy()):
            raise ValueError(
                "adapted-map containment: hint previous mean outside previous box"
            )
        transfer = tf.linalg.triangular_solve(
            l_old, tf.concat([l_pc, l_pp], axis=1), lower=True
        )
        rowsum = tf.reduce_sum(tf.abs(transfer), axis=1)
        slack = 1.0 - tf.constant(1e-9, DTYPE) - tf.abs(center)
        shrink = tf.minimum(
            tf.reduce_min(
                tf.where(rowsum > 0.0, slack / tf.maximum(rowsum, 1e-300),
                         tf.ones_like(rowsum))
            ),
            tf.constant(1.0, DTYPE),
        )
        l_pc = l_pc * shrink
        l_pp = l_pp * shrink
        map_c_new = AffineCoordinateMap(offset=m_c, matrix=l_cc)

        branch_count = int(prefix_values[-1].shape[-1]) + 1
        z_rows = _design_rows(config, config.row_count, 2 * n, (config.seed, 100 + t))
        z_weights = tf.fill(
            [int(z_rows.shape[0])], tf.constant(1.0 / int(z_rows.shape[0]), DTYPE)
        )
        mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
        cores0_values = [
            0.3
            * tf.random.stateless_normal(
                [1 if a == 0 else config.rank, mixed_dims[a],
                 1 if a == 2 * n else config.rank],
                tf.constant((config.seed, 7000 + 31 * t + a), tf.int32),
                dtype=DTYPE,
            )
            for a in range(2 * n + 1)
        ]
        if branch_count not in step_cache:
            mixed_basis = ProductBasis(
                list(current_basis.bases)
                + [DiscreteIndicatorBasis1D(branch_count)]
                + list(_product_basis(n, config.basis_degree).bases),
                current_basis.convention,
            )
            step_cache[branch_count] = _make_transition_fit(
                mixed_basis,
                [tuple(v.shape.as_list()) for v in cores0_values],
                [tuple(v.shape.as_list()) for v in prefix_values],
            )
        step_fn = step_cache[branch_count]
        (
            prefix_values, gram, zc_new, log_increment_in, worst, rms, max_excess_t,
        ) = step_fn(
            prefix_values, gram, zc, z_rows, z_weights, cores0_values,
            observations[t], m_c, l_cc, m_p_base, l_pc, l_pp, l_old, m_old,
        )
        max_excess = float(max_excess_t.numpy())
        if max_excess > 1.0 + 1e-12:
            raise ValueError(
                f"adapted-map containment violated post-shrink ({max_excess})"
            )
        if float(worst.numpy()) > config.condition_number_veto:
            raise ValueError("condition number veto in fixed ALS fit")

        # ---- truncation-mass correction (host-side; Section 12) ----
        corr_rows = _design_rows(config, config.row_count, 2 * n, (config.seed, 500 + t))
        xc_corr = m_c[None, :] + tf.einsum("ij,nj->ni", l_cc, corr_rows[:, :n])
        xp_corr = m_old[None, :] + tf.einsum("ij,nj->ni", l_old, corr_rows[:, n:])
        zp_new = tf.transpose(
            tf.linalg.triangular_solve(
                l_pp,
                tf.transpose(
                    xp_corr - m_p_base[None, :]
                    - tf.einsum("ij,nj->ni", l_pc, corr_rows[:, :n])
                ),
                lower=True,
            )
        )
        outside = tf.cast(tf.reduce_max(tf.abs(zp_new), axis=1) > 1.0, DTYPE)
        log_kernel_corr = (
            adapter.transition_log_density(xc_corr, xp_corr)
            + adapter.observation_log_density(xc_corr, observations[t])
        )
        integrand = (
            outside
            * retained.evaluate_physical_density(xp_corr)
            * tf.exp(log_kernel_corr)
        )
        logdet_c = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_cc))))
        logdet_old = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_old))))
        volume = tf.exp(tf.cast(2 * n, DTYPE) * log2 + logdet_c + logdet_old)
        m_out = volume * tf.reduce_mean(integrand)
        log_m_out = tf.math.log(tf.maximum(m_out, tf.constant(1e-300, DTYPE)))
        log_increment = tf.reduce_logsumexp(
            tf.stack([log_increment_in, log_m_out])
        )
        increment_value = float(log_increment.numpy())
        import math as _math
        if not _math.isfinite(increment_value):
            raise ValueError("non-finite step increment (fail-closed)")

        # rebuild the typed retained object for the next step's host-side work
        retained = RetainedQuadraticForm(
            prefix_cores=tuple(TTCore(v) for v in prefix_values),
            suffix_gram=gram,
            tau=tau * (zc_new / (1.0 + tau)),
            z_complete_ref=zc_new,
            prefix_basis=current_basis,
            coordinate_map=map_c_new,
        )
        zc = zc_new
        log_likelihood += log_increment
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": increment_value,
                "tie_flag": False,
                "worst_condition": float(worst.numpy()),
                "weighted_fit_rms": float(rms.numpy()),
                "map_shrink": float(shrink.numpy()),
                "z_old_max": max_excess,
                "truncation_mass_ratio": float(
                    tf.exp(log_m_out - log_increment_in).numpy()
                ),
            }
        )
    return log_likelihood, diagnostics


__all__ = ["run_value_filter_branch_axis_adapted_xla"]
