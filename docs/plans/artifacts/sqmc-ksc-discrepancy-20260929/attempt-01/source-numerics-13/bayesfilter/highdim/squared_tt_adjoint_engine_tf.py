"""Full-path manual adjoint score engine for the branch-axis squared-TT filter.

P2 implementation of UB-1 Addendum A: forward pass identical to
`run_value_filter_branch_axis` (same frozen program) with per-update
checkpoints, then one reverse sweep chaining the node adjoints of
`squared_tt_adjoint_tf`. Gradient cost is O(1) x value in flops
(one transposed solve per update; no per-parameter environment work) —
the P2A-selected mode.

Adapter VJP contract (per UB-1 A.2.4, transposes of the JVP obligations):
    transition_vjp(x_c, x_p, cotangent_rows) -> bar_theta [p]
    observation_vjp(x_c, y, cotangent_rows) -> bar_theta [p]
    initial_vjp(x_c, cotangent_rows)        -> bar_theta [p]

Telescoping note: for interior steps the retained-normalizer cotangent
cancels exactly (increment_t carries +1/Zc_t, increment_{t+1} carries
-1/Zc_t); the implementation keeps both terms explicit so the identity is
an emergent check, not an assumption.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_gram_matrix,
    prefix_row_vectors,
    retained_quadratic_form_from_squared_tt,
    suffix_gram_matrix,
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
from bayesfilter.highdim.fitting import FixedTTFitConfig, FixedTTFitter
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _design_rows,
    _fixed_als_fit_traced,
    _gauss_rows,
    _initial_tt_cores,
    _product_basis,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64


def _prefix_gram_adjoint(prefix_cores, product_basis, bar_gram):
    """Adjoint of prefix cores -> P = int H_L' H_L d mu (chain transpose)."""

    checked = tuple(prefix_cores)
    count = len(checked)
    mass = [
        product_basis.bases[j].mass_matrix(product_basis.convention.mass_measure)
        for j in range(count)
    ]
    above = [tf.ones([1, 1], DTYPE)]
    for j in range(count):
        above.append(
            tf.einsum(
                "akb,AlB,kl,aA->bB", checked[j].values, checked[j].values, mass[j], above[-1]
            )
        )
    bar_cores = [tf.zeros_like(c.values) for c in checked]
    bar_state = tf.convert_to_tensor(bar_gram, DTYPE)
    for j in range(count - 1, -1, -1):
        core_j = checked[j].values
        bar_cores[j] += tf.einsum(
            "bB,AlB,kl,aA->akb", bar_state, core_j, mass[j], above[j]
        ) + tf.einsum("bB,akb,kl,aA->AlB", bar_state, core_j, mass[j], above[j])
        bar_state = tf.einsum("bB,akb,AlB,kl->aA", bar_state, core_j, core_j, mass[j])
    return tuple(TTCore(values) for values in bar_cores)


def _random_cores(dims, rank: int, seed: tuple[int, int]):
    cores = []
    for axis, dim in enumerate(dims):
        left = 1 if axis == 0 else rank
        right = 1 if axis == len(dims) - 1 else rank
        cores.append(
            TTCore(
                0.3
                * tf.random.stateless_normal(
                    [left, dim, right],
                    tf.constant((seed[0], seed[1] + axis), tf.int32),
                    dtype=DTYPE,
                )
            )
        )
    return tuple(cores)


def _fit_config_for(config: EngineConfig, core_count: int, row_count: int) -> FixedTTFitConfig:
    """The exact FixedTTFitConfig `_fixed_als_fit_traced` uses (for design rebuild)."""

    return FixedTTFitConfig(
        ranks=tuple([1] + [config.rank] * (core_count - 1) + [1])[: core_count + 1],
        ridge=config.ridge,
        max_sweeps=config.sweeps,
        sweep_order=tuple(range(core_count)),
        row_budget=row_count,
        column_budget=4096,
        dense_matrix_byte_budget=1 << 30,
        normal_matrix_byte_budget=1 << 30,
        condition_number_warning=1e12,
        condition_number_veto=config.condition_number_veto,
        holdout_tolerance=1e30,
    )


def run_adjoint_score_filter(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    transition_vjp: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    observation_vjp: Callable[[tf.Tensor, tf.Tensor, tf.Tensor], tf.Tensor],
    initial_vjp: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    parameter_dim: int,
    gram_condition_veto: float = 1e12,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return (log-likelihood, full-theta gradient) for the frozen program."""

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

    # ------------------------- forward with trace -------------------------
    trace: list[dict] = []
    log_likelihood = tf.constant(0.0, DTYPE)
    retained = None
    for t in range(horizon):
        step: dict = {"t": t}
        if t == 0:
            if config.quadrature_order is not None:
                rows, weights = _gauss_rows(n, config.quadrature_order)
            else:
                rows = _design_rows(config, config.row_count, n, (config.seed, 17))
                weights = tf.fill(
                    [int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE)
                )
            x_current = rows * half
            log_f = (
                adapter.initial_log_density(x_current)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            softmax = tf.nn.softmax(log_f)
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_target = tf.exp(0.5 * (log_f - shift))
            cores0 = _initial_tt_cores(n, basis_dim, config.rank)
            cores, updates = _fixed_als_fit_traced(
                current_basis, rows, sqrt_target, weights, cores0, config
            )
            suffix_core = tf.zeros([int(cores[-1].right_rank), basis_dim, 1], DTYPE)
            suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
            cores_full = tuple(cores) + (TTCore(suffix_core),)
            split_basis = _product_basis(n + 1, config.basis_degree)
            base = retained_quadratic_form_from_squared_tt(
                cores_full, split_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=current_map,
            )
            z_h_new = base.z_complete_ref
            retained_new = RetainedQuadraticForm(
                prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
                tau=tau * z_h_new, z_complete_ref=(1.0 + tau) * z_h_new,
                prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
            )
            log_increment = shift + tf.math.log(retained_new.z_complete_ref)
            step.update(
                mode="initial", rows=rows, weights=weights, x_current=x_current,
                sqrt_target=sqrt_target, softmax=softmax,
                fit_basis=current_basis, updates=updates,
                cores_full=cores_full, split_basis=split_basis,
                retained_new=retained_new, fitted_core_count=len(cores),
            )
        else:
            gram = retained.suffix_gram
            eigenvalues = tf.linalg.eigvalsh(gram)
            condition = float(
                (eigenvalues[-1] / tf.maximum(eigenvalues[0], tf.constant(1e-300, DTYPE))).numpy()
            )
            if condition > gram_condition_veto:
                raise ValueError("retained Gram conditioning veto (P2 claim gate)")
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(
                gram
                + tf.constant(config.branch_gram_floor, DTYPE)
                * floor_scale
                * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
            )
            branch_count = retained.boundary_rank + 1
            if config.quadrature_order is not None:
                z_rows, z_weights = _gauss_rows(2 * n, config.quadrature_order)
            else:
                z_rows = _design_rows(config, config.row_count, 2 * n, (config.seed, 100 + t))
                z_weights = tf.fill(
                    [int(z_rows.shape[0])], tf.constant(1.0 / int(z_rows.shape[0]), DTYPE)
                )
            x_current = z_rows[:, :n] * half
            z_previous = z_rows[:, n:]
            x_previous = z_previous * half
            log_g = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            v_rows = prefix_row_vectors(
                retained.prefix_cores, retained.prefix_basis, z_previous
            )
            u = tf.einsum("na,ab->nb", v_rows, chol)
            tau_abs = tau * (retained.z_complete_ref / (1.0 + tau))  # = tau * Z_h_prev
            sum_sq = tf.reduce_sum(tf.square(u), axis=1) + tau_abs
            log_f = tf.math.log(sum_sq) + log_g
            softmax = tf.nn.softmax(log_f)
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_g_shifted = tf.exp(0.5 * (log_g - shift))
            amplitudes = tf.concat(
                [u, tf.ones([int(z_rows.shape[0]), 1], DTYPE) * tf.sqrt(tau_abs)], axis=1
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
            cores0 = _random_cores(mixed_dims, config.rank, (config.seed, 7000 + 31 * t))
            cores, updates = _fixed_als_fit_traced(
                mixed_basis, full_rows, sqrt_target, weights, cores0, config
            )
            cores_full = tuple(cores)
            base = retained_quadratic_form_from_squared_tt(
                cores_full, mixed_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=current_map,
            )
            z_h_new = base.z_complete_ref
            retained_new = RetainedQuadraticForm(
                prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
                tau=tau * z_h_new, z_complete_ref=(1.0 + tau) * z_h_new,
                prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
            )
            log_increment = (
                shift
                + tf.math.log(retained_new.z_complete_ref)
                - tf.math.log(retained.z_complete_ref)
            )
            step.update(
                mode="transition", z_rows=z_rows, x_current=x_current, tau_abs=tau_abs,
                x_previous=x_previous, z_previous=z_previous, chol=chol,
                v_rows=v_rows, u=u, sum_sq=sum_sq, sqrt_g_shifted=sqrt_g_shifted,
                amplitudes=amplitudes, branch_count=branch_count,
                softmax=softmax, sqrt_target=sqrt_target,
                fit_basis=mixed_basis, updates=updates, cores_full=cores_full,
                split_basis=mixed_basis, retained_new=retained_new,
                prev_retained=retained, fitted_core_count=len(cores),
            )
        log_likelihood += log_increment
        retained = step["retained_new"]
        # T=120 trace-memory repair (stress plan 2026-08-17, attempt01 gate
        # failure at 19.5 GiB): the stored design matrices dominate trace
        # memory (O(rows x cols) per update, linear in T). Designs are a
        # deterministic function of (basis, rows, weights, target, cores
        # before the update, core_index) and are rebuilt bit-identically in
        # the reverse sweep; drop them from the trace.
        for update in step["updates"]:
            update["design"] = None
        trace.append(step)

    # ---------------------------- reverse sweep ----------------------------
    grad = tf.zeros([parameter_dim], DTYPE)
    bar_prefix = None
    bar_gram_downstream = None
    bar_zc_downstream = tf.constant(0.0, DTYPE)
    for t in range(horizon - 1, -1, -1):
        step = trace[t]
        retained_new = step["retained_new"]
        split = n
        split_basis = step["split_basis"]
        cores_full = step["cores_full"]
        prefix_cores_full = cores_full[:split]
        suffix_cores_full = cores_full[split:]

        # v0.2 semantics: Zc = (1+tau) Z_h with Z_h = <prefixGram, suffixGram>.
        # Cotangent on Z_h,t: +1/Z_h from increment_t, plus downstream terms
        # (-1/Z_h from increment_{t+1}, tau-branch and sum_sq couplings).
        z_h_new = retained_new.z_complete_ref / (1.0 + tau)
        bar_zh = 1.0 / z_h_new + bar_zc_downstream
        gram_suffix = suffix_gram_matrix(suffix_cores_full, split_basis, axis_offset=split)
        gram_prefix = prefix_gram_matrix(prefix_cores_full, split_basis)
        bar_gram_suffix = bar_zh * gram_prefix
        bar_gram_prefix = bar_zh * gram_suffix
        if bar_gram_downstream is not None:
            bar_gram_suffix = bar_gram_suffix + bar_gram_downstream

        bar_cores = [tf.zeros_like(c.values) for c in cores_full]
        for k, adj in enumerate(
            gram_chain_adjoint(
                suffix_cores_full, split_basis, axis_offset=split, bar_gram=bar_gram_suffix
            )
        ):
            bar_cores[split + k] += adj.values
        for k, adj in enumerate(
            _prefix_gram_adjoint(prefix_cores_full, split_basis, bar_gram_prefix)
        ):
            bar_cores[k] += adj.values
        if bar_prefix is not None:
            for k in range(split):
                bar_cores[k] += bar_prefix[k]

        # Reverse the traced ALS updates (only the fitted cores; a trivial
        # appended suffix core at t=0 is constant and its cotangent is
        # dropped by never entering the update list).
        fitted_count = step["fitted_core_count"]
        cores_state = list(cores_full[:fitted_count])
        bar_target = tf.zeros_like(step["sqrt_target"])
        fitter = FixedTTFitter()
        rebuild_config = _fit_config_for(
            config, fitted_count, int(step["updates"][0]["rows"].shape[0])
        )
        for update in reversed(step["updates"]):
            idx = update["core_index"]
            bar_c = tf.reshape(bar_cores[idx], [-1])
            # Rebuild the design dropped from the trace. cores_state at this
            # point differs from the pre-update state only at core idx, which
            # its own design matrix does not contain -> bit-identical rebuild.
            design = fitter.build_core_update_system(
                update["basis"], update["rows"], update["target"],
                update["weights"], tuple(cores_state), idx, rebuild_config,
            ).design_matrix
            bar_g_rows, bar_a = solve_node_adjoint(
                design, update["weights"], update["target"],
                update["solution"], update["ridge"], bar_c,
            )
            bar_target += bar_g_rows
            for k, adj in enumerate(
                design_assembly_adjoint(
                    update["basis"], update["rows"], tuple(cores_state), idx, bar_a
                )
            ):
                bar_cores[k] += adj.values
            bar_cores[idx] = tf.zeros_like(bar_cores[idx])
            cores_state[idx] = update["previous_core"]

        # sqrt-target adjoint and theta-direct VJPs
        if step["mode"] == "initial":
            half = 0.5 * step["sqrt_target"] * bar_target
            bar_shift0 = -tf.reduce_sum(half)
            bar_logf = half + (tf.constant(1.0, DTYPE) + bar_shift0) * step["softmax"]
            grad += initial_vjp(step["x_current"], bar_logf)
            grad += observation_vjp(step["x_current"], observations[t], bar_logf)
            bar_prefix, bar_gram_downstream = None, None
            bar_zc_downstream = tf.constant(0.0, DTYPE)
        else:
            n_z = int(step["z_rows"].shape[0])
            branch_count = step["branch_count"]
            bar_targets = tf.reshape(bar_target, [n_z, branch_count])
            bar_amplitudes = bar_targets * step["sqrt_g_shifted"][:, None]
            bar_sqrt_g = tf.reduce_sum(bar_targets * step["amplitudes"], axis=1)
            bar_log_g = 0.5 * step["sqrt_g_shifted"] * bar_sqrt_g
            bar_shift = -tf.reduce_sum(0.5 * step["sqrt_g_shifted"] * bar_sqrt_g)
            bar_logf = (tf.constant(1.0, DTYPE) + bar_shift) * step["softmax"]
            bar_sum_sq = bar_logf / step["sum_sq"]
            bar_log_g += bar_logf
            bar_u = bar_amplitudes[:, : branch_count - 1] + 2.0 * step["u"] * bar_sum_sq[:, None]
            # v0.2 tau couplings: tau_abs = tau * Z_h_prev enters sum_sq and
            # the tau-branch amplitude sqrt(tau_abs).
            tau_abs = step["tau_abs"]
            bar_tau_abs = tf.reduce_sum(bar_sum_sq) + tf.reduce_sum(
                bar_amplitudes[:, branch_count - 1]
            ) / (2.0 * tf.sqrt(tau_abs))
            bar_v = tf.einsum("nb,ab->na", bar_u, step["chol"])
            bar_chol = tf.einsum("na,nb->ab", step["v_rows"], bar_u)
            prev = step["prev_retained"]
            bar_gram_prev = cholesky_vjp(step["chol"], bar_chol)
            prev_prefix_adj = prefix_rows_adjoint(
                prev.prefix_cores, prev.prefix_basis, step["z_previous"], bar_v
            )
            grad += transition_vjp(step["x_current"], step["x_previous"], bar_log_g)
            grad += observation_vjp(step["x_current"], observations[t], bar_log_g)
            bar_prefix = [adj.values for adj in prev_prefix_adj]
            bar_gram_downstream = bar_gram_prev
            z_h_prev = prev.z_complete_ref / (1.0 + tau)
            bar_zc_downstream = -1.0 / z_h_prev + tau * bar_tau_abs
    return log_likelihood, grad


__all__ = ["run_adjoint_score_filter"]
