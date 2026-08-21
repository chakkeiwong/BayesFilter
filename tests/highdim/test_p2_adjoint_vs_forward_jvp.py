"""I-P2-4: full-path forward-JVP vs adjoint gradient (FD-independent).

Two independently implemented MANUAL derivative chains of the identical
frozen program must agree to near machine precision:
- forward: donor ordered ALS value+JVP replay (`fixed_als_value_jvp`,
  moment_teacher_als) + P1A retained tangents, chained over steps;
- reverse: the P2 adjoint engine.

This is the decisive score gate where FD is resolution-limited (n>=2:
the value program's curvature/roughness makes centered FD inconsistent
with itself below ~1e-3; diagnosed 2026-08-17).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_row_vectors,
    prefix_row_vectors_tangent,
    retained_quadratic_form_from_squared_tt,
    retained_quadratic_form_tangent_from_squared_tt,
)
from bayesfilter.highdim.squared_tt_adjoint_engine_tf import run_adjoint_score_filter
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    _fixed_als_fit_traced,
    _frozen_rows,
    _gauss_rows,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.squared_tt_adjoint_tf import forward_jvp_replay_scaled
from bayesfilter.highdim.fitting import FixedTTFitConfig
from bayesfilter.highdim.tt import TTCore
from bayesfilter.highdim.bases import ProductBasis
from tests.highdim.test_p2_adjoint_engine_fd import _family, _config

DTYPE = tf.float64


def _fit_config(dims_count: int, config) -> FixedTTFitConfig:
    return FixedTTFitConfig(
        ranks=tuple([1] + [config.rank] * (dims_count - 1) + [1]),
        ridge=config.ridge, max_sweeps=config.sweeps,
        sweep_order=tuple(range(dims_count)), row_budget=10**9,
        column_budget=4096, dense_matrix_byte_budget=1 << 30,
        normal_matrix_byte_budget=1 << 30, condition_number_warning=1e12,
        condition_number_veto=1e18,  # cross-check instrument: donor uses UNSCALED normal equations; the program's scaled solver is the claim path
        holdout_tolerance=1e30,
    )


def run_forward_jvp_filter(
    adapter, observations, config, *, transition_jvp, observation_jvp,
    initial_jvp,
):
    """(value, directional derivative) by the forward tangent chain.

    JVP adapter contract (directional): fn(x_c[, x_p], y?) -> dot log rows.
    Mirrors the score engine's forward program exactly.
    """

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    half = tf.constant(config.coordinate_half_width, DTYPE)
    current_map = AffineCoordinateMap(
        offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE) * half
    )
    conversion = tf.cast(n, DTYPE) * (
        tf.math.log(half) + tf.math.log(tf.constant(2.0, DTYPE))
    )
    tau = tf.constant(config.tau, DTYPE)

    value = tf.constant(0.0, DTYPE)
    dot_value = tf.constant(0.0, DTYPE)
    retained = None
    dot_state = None  # (dot_prefix cores, dot_E, dot_Zh)
    for t in range(horizon):
        if t == 0:
            if config.quadrature_order is not None:
                rows, weights = _gauss_rows(n, config.quadrature_order)
            else:
                rows = _frozen_rows(config.row_count, n, (config.seed, 17))
                weights = tf.fill([int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE))
            x_current = rows * half
            log_f = (
                adapter.initial_log_density(x_current)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            dot_logf = initial_jvp(x_current) + observation_jvp(x_current, observations[t])
            softmax = tf.nn.softmax(log_f)
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
            dot_shift = tf.reduce_sum(softmax * dot_logf)
            sqrt_target = tf.exp(0.5 * (log_f - shift))
            dot_target = 0.5 * sqrt_target * (dot_logf - dot_shift)
            cores0 = _initial_tt_cores(n, basis_dim, config.rank)
            dot_cores0 = tuple(TTCore(tf.zeros_like(c.values)) for c in cores0)
            _cores_v, updates = _fixed_als_fit_traced(
                current_basis, rows, sqrt_target, weights, cores0, config
            )
            cores_fit, dots_fit = forward_jvp_replay_scaled(
                updates, cores0, dot_cores0, dot_target
            )
            suffix_core = tf.zeros([int(cores_fit[-1].right_rank), basis_dim, 1], DTYPE)
            suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
            cores_full = tuple(cores_fit) + (TTCore(suffix_core),)
            dots_full = tuple(dots_fit) + (TTCore(tf.zeros_like(suffix_core)),)
            split_basis = _product_basis(n + 1, config.basis_degree)
        else:
            gram = retained.suffix_gram
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            floored = gram + tf.constant(config.branch_gram_floor, DTYPE) * floor_scale * tf.eye(
                tf.shape(gram)[0], dtype=DTYPE
            )
            chol = tf.linalg.cholesky(floored)
            dot_prefix, dot_gram, dot_zh_prev = dot_state
            dot_floored = dot_gram + tf.constant(config.branch_gram_floor, DTYPE) * (
                tf.linalg.trace(dot_gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            ) * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
            # Cholesky JVP: dL = L Phi(L^-1 dM L^-T), Phi = tril with half diag
            # (FD-checked standalone 2026-08-17; L @ Phi' is wrong: rel ~1.3)
            inner = tf.linalg.triangular_solve(chol, tf.transpose(
                tf.linalg.triangular_solve(chol, dot_floored, lower=True)
            ), lower=True)
            phi = tf.linalg.band_part(inner, -1, 0) - 0.5 * tf.linalg.diag(
                tf.linalg.diag_part(inner)
            )
            dot_chol = tf.matmul(chol, phi)
            branch_count = retained.boundary_rank + 1
            if config.quadrature_order is not None:
                z_rows, z_weights = _gauss_rows(2 * n, config.quadrature_order)
            else:
                z_rows = _frozen_rows(config.row_count, 2 * n, (config.seed, 100 + t))
                z_weights = tf.fill([int(z_rows.shape[0])], tf.constant(1.0 / int(z_rows.shape[0]), DTYPE))
            x_current = z_rows[:, :n] * half
            z_previous = z_rows[:, n:]
            x_previous = z_previous * half
            log_g = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            dot_log_g = transition_jvp(x_current, x_previous) + observation_jvp(
                x_current, observations[t]
            )
            v_rows, dot_v = prefix_row_vectors_tangent(
                retained.prefix_cores, dot_prefix, retained.prefix_basis, z_previous
            )
            u = tf.einsum("na,ab->nb", v_rows, chol)
            dot_u = tf.einsum("na,ab->nb", dot_v, chol) + tf.einsum(
                "na,ab->nb", v_rows, dot_chol
            )
            z_h_prev = retained.z_complete_ref / (1.0 + tau)
            tau_abs = tau * z_h_prev
            dot_tau_abs = tau * dot_zh_prev
            sum_sq = tf.reduce_sum(tf.square(u), axis=1) + tau_abs
            dot_sum_sq = 2.0 * tf.reduce_sum(u * dot_u, axis=1) + dot_tau_abs
            log_f = tf.math.log(sum_sq) + log_g
            dot_logf = dot_sum_sq / sum_sq + dot_log_g
            softmax = tf.nn.softmax(log_f)
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(tf.cast(tf.shape(log_f)[0], DTYPE))
            dot_shift = tf.reduce_sum(softmax * dot_logf)
            sqrt_g_shifted = tf.exp(0.5 * (log_g - shift))
            dot_sqrt_g = 0.5 * sqrt_g_shifted * (dot_log_g - dot_shift)
            sqrt_tau_abs = tf.sqrt(tau_abs)
            dot_sqrt_tau = dot_tau_abs / (2.0 * sqrt_tau_abs)
            amplitudes = tf.concat(
                [u, tf.ones([int(z_rows.shape[0]), 1], DTYPE) * sqrt_tau_abs], axis=1
            )
            dot_amplitudes = tf.concat(
                [dot_u, tf.ones([int(z_rows.shape[0]), 1], DTYPE) * dot_sqrt_tau], axis=1
            )
            targets = amplitudes * sqrt_g_shifted[:, None]
            dot_targets = dot_amplitudes * sqrt_g_shifted[:, None] + amplitudes * dot_sqrt_g[:, None]
            g_codes = tf.tile(tf.range(branch_count, dtype=DTYPE)[None, :], [int(z_rows.shape[0]), 1])
            full_rows = tf.concat(
                [
                    tf.repeat(z_rows[:, :n], branch_count, axis=0),
                    tf.reshape(g_codes, [-1, 1]),
                    tf.repeat(z_rows[:, n:], branch_count, axis=0),
                ],
                axis=1,
            )
            sqrt_target = tf.reshape(targets, [-1])
            dot_target = tf.reshape(dot_targets, [-1])
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
                        [1 if a == 0 else config.rank, mixed_dims[a], 1 if a == 2 * n else config.rank],
                        tf.constant((config.seed, 7000 + 31 * t + a), tf.int32),
                        dtype=DTYPE,
                    )
                )
                for a in range(2 * n + 1)
            )
            dot_cores0 = tuple(TTCore(tf.zeros_like(c.values)) for c in cores0)
            _cores_v, updates = _fixed_als_fit_traced(
                mixed_basis, full_rows, sqrt_target, weights, cores0, config
            )
            cores_full, dots_full = forward_jvp_replay_scaled(
                updates, cores0, dot_cores0, dot_target
            )
            split_basis = mixed_basis
        base = retained_quadratic_form_from_squared_tt(
            cores_full, split_basis, split_index=n, tau=0.0,
            prefix_basis=current_basis, coordinate_map=current_map,
        )
        dp, dg, dzh = retained_quadratic_form_tangent_from_squared_tt(
            cores_full, dots_full, split_basis, split_index=n
        )
        z_h_new = base.z_complete_ref
        retained_new = RetainedQuadraticForm(
            prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
            tau=tau * z_h_new, z_complete_ref=(1.0 + tau) * z_h_new,
            prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
        )
        if t == 0:
            log_increment = shift + tf.math.log(retained_new.z_complete_ref)
            dot_increment = dot_shift + dzh / z_h_new
        else:
            log_increment = (
                shift + tf.math.log(retained_new.z_complete_ref)
                - tf.math.log(retained.z_complete_ref)
            )
            dot_increment = dot_shift + dzh / z_h_new - dot_state[2] / (
                retained.z_complete_ref / (1.0 + tau)
            )
        value += log_increment
        dot_value += dot_increment
        retained = retained_new
        dot_state = (dp, dg, dzh)
    return value, dot_value


def test_i_p2_4_adjoint_matches_forward_jvp_n2() -> None:
    n, seed, T = 2, 62, 4
    theta0 = np.zeros(n)
    adapter, tvjp, ovjp, ivjp, ys_full = _family(n, theta0, seed)
    ys = ys_full[:T]
    config = _config(n)
    _value, grad = run_adjoint_score_filter(
        adapter, ys, config,
        transition_vjp=tvjp, observation_vjp=ovjp, initial_vjp=ivjp,
        parameter_dim=n,
    )
    # shift-family JVPs: dot log p(x_c|x_p) along e_k = [Q^{-1}(x_c-Ax_p-theta)]_k
    q_inv = np.linalg.inv(0.4 * np.eye(n))
    a_matrix = 0.7 * np.eye(n)
    for k in range(n):
        e_k = tf.constant(np.eye(n)[k], DTYPE)

        def transition_jvp(xc, xp):
            residual = xc - tf.linalg.matvec(tf.constant(a_matrix, DTYPE), xp)
            return tf.einsum("nd,d->n", tf.einsum("nd,de->ne", residual, tf.constant(q_inv, DTYPE)), e_k)

        value_f, dot_f = run_forward_jvp_filter(
            adapter, ys, config,
            transition_jvp=transition_jvp,
            observation_jvp=lambda xc, y: tf.zeros([int(xc.shape[0])], DTYPE),
            initial_jvp=lambda xc: tf.zeros([int(xc.shape[0])], DTYPE),
        )
        rel = abs(float(grad[k].numpy()) - float(dot_f.numpy())) / max(
            1.0, abs(float(dot_f.numpy()))
        )
        assert rel <= 1e-9, (
            f"axis {k}: adjoint {float(grad[k].numpy())} vs forward-JVP "
            f"{float(dot_f.numpy())} rel {rel}"
        )
