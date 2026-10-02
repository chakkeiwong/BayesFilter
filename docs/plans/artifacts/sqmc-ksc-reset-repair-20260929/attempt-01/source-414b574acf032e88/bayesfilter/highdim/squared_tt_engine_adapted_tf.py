"""Adapted-coordinate-map branch-axis value filter (linear bridging v1).

Design note: docs/plans/bayesfilter-adapted-coordinate-maps-design-note-2026-08-20.md
(Sections 1-3, 7-8; Zhao-Cui JMLR 2024 Section 5.2 linear instance).

Separate function from `run_value_filter_branch_axis` ON PURPOSE: the
fixed-map program stays byte-for-byte on its existing code path (rung-3
bit-identity is structural). This engine differs only in:
- per-step, per-block frozen affine maps (m, L) computed BEFORE rows;
- previous block map from exact retained moments (M1, measured sound)
  with the closed-form containment shrinkage into the old box image;
- current block map from the adapter-supplied predictive moment hint
  (M2, promoted by probe arm M1) — fail-closed if absent;
- per-block reference-typed conversion terms (Section 1 audited form).

Scattered rows only (quadrature_order rejected fail-closed).
Value path only; the adjoint engine port is a follow-up under V5.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_moments_tf import retained_reference_moments
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_row_vectors,
    retained_quadratic_form_from_squared_tt,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _design_rows,
    _fixed_als_fit,
    _gauss_rows,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64


def _containment_shrink(
    m_new: tf.Tensor, l_new: tf.Tensor, map_old: AffineCoordinateMap
) -> tf.Tensor:
    """Shrink l_new so map_new([-1,1]^n) lies inside map_old([-1,1]^n).

    z_old(z) = L_old^{-1}(m_new - m_old) + L_old^{-1} L_new z; over the
    box, |z_old_i| <= |c_i| + sum_j |T_ij| with c the center image and
    T = L_old^{-1} L_new. The returned scale s <= 1 enforces
    |c_i| + s * rowsum_i <= 1 - eps for all i (exact containment for
    ALL box points, not just sampled rows). Fails closed if the center
    itself is outside the old box.
    """

    eps = tf.constant(1e-9, DTYPE)
    l_old = tf.convert_to_tensor(map_old.matrix, DTYPE)
    m_old = tf.convert_to_tensor(map_old.offset, DTYPE)
    center = tf.linalg.triangular_solve(
        l_old, (m_new - m_old)[:, None], lower=True
    )[:, 0]
    if bool(tf.reduce_any(tf.abs(center) >= 1.0).numpy()):
        raise ValueError(
            "adapted-map containment: retained mean outside previous box"
        )
    transfer = tf.linalg.triangular_solve(l_old, l_new, lower=True)
    rowsum = tf.reduce_sum(tf.abs(transfer), axis=1)
    slack = 1.0 - eps - tf.abs(center)
    scale = tf.reduce_min(
        tf.where(rowsum > 0.0, slack / tf.maximum(rowsum, 1e-300), tf.ones_like(rowsum))
    )
    return tf.minimum(scale, tf.constant(1.0, DTYPE))


def _check_hint(mean: tf.Tensor, cov: tf.Tensor, n: int) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.convert_to_tensor(mean, DTYPE)
    cov = tf.convert_to_tensor(cov, DTYPE)
    if mean.shape.as_list() != [n] or cov.shape.as_list() != [n, n]:
        raise ValueError("moment hint shape mismatch")
    if not bool(
        (tf.reduce_all(tf.math.is_finite(mean)) & tf.reduce_all(tf.math.is_finite(cov))).numpy()
    ):
        raise ValueError("moment hint nonfinite")
    chol = tf.linalg.cholesky(cov)  # raises on non-PD
    if not bool(tf.reduce_all(tf.math.is_finite(chol)).numpy()):
        raise ValueError("moment hint covariance not positive definite")
    return mean, chol


def run_value_filter_branch_axis_adapted(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    map_kappa_prev: float = 3.0,
    map_kappa_current: float = 4.0,
) -> tuple[tf.Tensor, list[dict]]:
    """Adapted-map value filter (v2 triangular). `predictive_moment_hint(t, y_t)`
    must return the JOINT moments (mean [2n], cov [2n,2n]) of
    (x_t, x_{t-1}) in (current, previous) order (M2-joint; design note
    Section 9). Cholesky of the joint covariance yields (L_cc, L_pc, L_pp)."""

    if config.quadrature_order is not None:
        raise ValueError("adapted maps are defined for scattered rows only")
    if predictive_moment_hint is None:
        raise ValueError("adapted maps require a predictive moment hint (M2)")

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    hw = tf.constant(config.coordinate_half_width, DTYPE)
    # kappa_current > kappa_prev (design note Section 3): the current-block
    # box must leave room for the NEXT step's previous-block box, or the
    # containment shrink binds and truncates retained tails (measured
    # 2026-08-20: equal kappas -> shrink 0.74-0.83, -1 nat/step at n=4).
    kappa_p_const = tf.constant(map_kappa_prev, DTYPE)
    kappa_c_const = tf.constant(map_kappa_current, DTYPE)
    tau = tf.constant(config.tau, DTYPE)
    log2 = tf.math.log(tf.constant(2.0, DTYPE))

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[dict] = []

    for t in range(horizon):
        if t == 0:
            # t=0 stays on the global box (target not concentrated;
            # step-localization evidence 2026-08-19).
            rows = _design_rows(config, config.row_count, n, (config.seed, 17))
            weights = tf.fill(
                [int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE)
            )
            map_c = AffineCoordinateMap(
                offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE) * hw
            )
            x_current = rows * hw
            conversion = tf.cast(n, DTYPE) * (tf.math.log(hw) + log2)
            log_f = (
                adapter.initial_log_density(x_current)
                + adapter.observation_log_density(x_current, observations[t])
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
            step_extra: dict = {}
        else:
            # ---- v2 TRIANGULAR joint map (design note Section 9):
            #   x_c = m_c + L_cc z_c
            #   x_p = m_p + L_pc z_c + L_pp z_p
            # (m, Sigma) joint from the M2-JOINT hint; Cholesky in (c, p)
            # order yields (L_cc, L_pc, L_pp). The z_p box then tracks the
            # CONDITIONAL x_p | x_c — the collapsed direction of the
            # block-diagonal instance (kappa grid 2026-08-20).
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
            # containment (Section 9 point 3): z_old over the box satisfies
            # |z_old_i| <= |center_i| + rowsum_i(L_old^{-1}[L_pc | L_pp]);
            # shrink scales (L_pc, L_pp) jointly.
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
                    tf.where(
                        rowsum > 0.0,
                        slack / tf.maximum(rowsum, 1e-300),
                        tf.ones_like(rowsum),
                    )
                ),
                tf.constant(1.0, DTYPE),
            )
            l_pc = l_pc * shrink
            l_pp = l_pp * shrink
            map_c = AffineCoordinateMap(offset=m_c, matrix=l_cc)
            # telemetry cross-check: retained p-marginal moments (M1)
            mean_z, _cov_z = retained_reference_moments(retained)
            retained_mean_p = m_old + tf.linalg.matvec(l_old, mean_z)
            hint_prev_gap = float(
                tf.norm(retained_mean_p - m_p_base).numpy()
            )

            gram = retained.suffix_gram
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(
                gram
                + tf.constant(config.branch_gram_floor, DTYPE)
                * floor_scale
                * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
            )
            branch_count = retained.boundary_rank + 1
            z_rows = _design_rows(config, config.row_count, 2 * n, (config.seed, 100 + t))
            z_weights = tf.fill(
                [int(z_rows.shape[0])], tf.constant(1.0 / int(z_rows.shape[0]), DTYPE)
            )
            z_c = z_rows[:, :n]
            z_p = z_rows[:, n:]
            x_current = m_c[None, :] + tf.einsum("ij,nj->ni", l_cc, z_c)
            x_previous = (
                m_p_base[None, :]
                + tf.einsum("ij,nj->ni", l_pc, z_c)
                + tf.einsum("ij,nj->ni", l_pp, z_p)
            )
            # re-express previous rows in the OLD box coordinates (exact)
            z_old = tf.transpose(
                tf.linalg.triangular_solve(
                    l_old, tf.transpose(x_previous - m_old[None, :]), lower=True
                )
            )
            max_excess = float(tf.reduce_max(tf.abs(z_old)).numpy())
            if max_excess > 1.0 + 1e-12:
                raise ValueError(
                    f"adapted-map containment violated post-shrink ({max_excess})"
                )
            # per-block reference-typed conversions (Section 1 audited form;
            # block-triangular Jacobian: |det| = |det L_cc| |det L_pp|)
            logdet_c = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_cc))))
            logdet_p = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_pp))))
            logdet_old = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_old))))
            conversion = logdet_c + tf.cast(n, DTYPE) * log2 + logdet_p - logdet_old
            log_g = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            # ---- truncation-mass correction (design note Section 12) ----
            # The increment computes predictive mass over the NEW box image;
            # the containment fixed point (Section 11) makes the missed mass
            # a deterministic per-step bias. It is a SCALAR (never enters the
            # retained object), so frozen-Sobol MC over (new c-box) x (full
            # OLD p-box), keeping only points OUTSIDE the new image:
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
            outside = tf.cast(
                tf.reduce_max(tf.abs(zp_new), axis=1) > 1.0, DTYPE
            )
            log_kernel_corr = (
                adapter.transition_log_density(xc_corr, xp_corr)
                + adapter.observation_log_density(xc_corr, observations[t])
            )
            integrand = (
                outside
                * retained.evaluate_physical_density(xp_corr)
                * tf.exp(log_kernel_corr)
            )
            volume = tf.exp(
                tf.cast(2 * n, DTYPE) * log2 + logdet_c + logdet_old
            )
            m_out = volume * tf.reduce_mean(integrand)
            outside_fraction = float(tf.reduce_mean(outside).numpy())

            v_prev = tf.einsum(
                "na,ab->nb",
                prefix_row_vectors(retained.prefix_cores, retained.prefix_basis, z_old),
                chol,
            )
            tau_abs = tau * (retained.z_complete_ref / (1.0 + tau))
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
            mixed_basis = ProductBasis(
                list(current_basis.bases)
                + [DiscreteIndicatorBasis1D(branch_count)]
                + list(_product_basis(n, config.basis_degree).bases),
                current_basis.convention,
            )
            mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
            cores0 = tuple(
                TTCore(
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
                )
                for axis in range(2 * n + 1)
            )
            cores, fit_diag = _fixed_als_fit(
                mixed_basis, full_rows, sqrt_target, weights, cores0, config
            )
            base = retained_quadratic_form_from_squared_tt(
                tuple(cores), mixed_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=map_c,
            )
            step_extra = {
                "map_shrink": float(shrink.numpy()),
                "map_logdet_c": float(logdet_c.numpy()),
                "map_logdet_p": float(logdet_p.numpy()),
                "z_old_max": max_excess,
                "hint_prev_mean_gap": hint_prev_gap,
            }
        z_h_new = base.z_complete_ref
        retained_new = RetainedQuadraticForm(
            prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
            tau=tau * z_h_new, z_complete_ref=(1.0 + tau) * z_h_new,
            prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
        )
        if t == 0:
            log_increment = shift + tf.math.log(retained_new.z_complete_ref)
        else:
            log_increment_in = (
                shift
                + tf.math.log(retained_new.z_complete_ref)
                - tf.math.log(retained.z_complete_ref)
            )
            # correction: increment = log(M_in + M_out); M_out already
            # carries the retained normalization through the physical
            # evaluator, matching the telescoped M_in (Section 12).
            log_m_out = tf.math.log(tf.maximum(m_out, tf.constant(1e-300, DTYPE)))
            log_increment = tf.reduce_logsumexp(
                tf.stack([log_increment_in, log_m_out])
            )
            step_extra["truncation_mass_ratio"] = float(
                tf.exp(log_m_out - log_increment_in).numpy()
            )
            step_extra["outside_fraction"] = outside_fraction
        log_likelihood += log_increment
        retained = retained_new
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": float(log_increment.numpy()),
                "tie_flag": False,
                **fit_diag,
                **step_extra,
            }
        )
    return log_likelihood, diagnostics


__all__ = ["run_value_filter_branch_axis_adapted"]
