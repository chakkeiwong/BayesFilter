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
import hashlib
import json
import weakref
from dataclasses import dataclass
from typing import Callable, Mapping, NamedTuple, Sequence

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
_FROZEN_EVALUATOR_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


class _TransitionTarget(NamedTuple):
    physical_rows: tf.Tensor
    expanded_rows: tf.Tensor
    sqrt_target: tf.Tensor
    fit_weights: tf.Tensor
    shift: tf.Tensor
    computed_shift: tf.Tensor
    log_g: tf.Tensor
    log_f: tf.Tensor
    floor_values: tf.Tensor
    amplitudes: tf.Tensor
    u_old_max: tf.Tensor


_TARGET_SUMMARY_NAMES = (
    "all_finite",
    "log_g_min",
    "log_g_max",
    "log_f_min",
    "log_f_max",
    "sqrt_target_min",
    "sqrt_target_max",
    "branch_closure_relative_max",
    "floor_dominance_fraction",
)


@dataclass(frozen=True)
class GaussianXLAFrozenTransitionSnapshot:
    """Frozen transition state and fitted TT for read-only diagnostics.

    The snapshot is emitted only by the diagnostic entry point. It contains
    enough state to evaluate the exact production target on disjoint rows
    without refitting or recomputing the training shift.
    """

    run_identity: str
    time_index: int
    state_dim: int
    basis_degree: int
    rank: int
    row_count: int
    sweeps: int
    ridge: float
    configured_tau: float
    coordinate_half_width: float
    config_seed: int
    condition_number_veto: float
    branch_gram_floor: float
    row_design: str
    training_row_seed: tuple[int, int]
    defensive_nu: float | None
    branch_count: int
    basis_identity: str
    mixed_shapes: tuple[tuple[int, int, int], ...]
    prefix_values: tuple[tf.Tensor, ...]
    suffix_gram: tf.Tensor
    tau_abs_previous: tf.Tensor
    z_complete_previous: tf.Tensor
    old_coordinate_offset: tf.Tensor
    old_coordinate_matrix: tf.Tensor
    joint_mean: tf.Tensor
    joint_chol: tf.Tensor
    observation: tf.Tensor
    training_rows: tf.Tensor
    training_weights: tf.Tensor
    frozen_shift: tf.Tensor
    fitted_core_values: tuple[tf.Tensor, ...]
    z_h: tf.Tensor
    raw_increment: tf.Tensor
    corrected_increment: tf.Tensor
    worst_condition: tf.Tensor
    weighted_fit_rms: tf.Tensor
    u_old_max: tf.Tensor
    target_summary: Mapping[str, float]


@dataclass(frozen=True)
class GaussianXLARetainedProposalSnapshot:
    """Post-update retained state captured from the seven-output fit graph.

    Unlike ``GaussianXLAFrozenTransitionSnapshot``, this snapshot never asks
    the XLA fit graph to return the complete fitted TT.  The retained prefix
    cores and suffix Gram are already production outputs and fully define the
    next-state proposal quadratic form.
    """

    run_identity: str
    time_index: int
    state_dim: int
    basis_degree: int
    rank: int
    row_count: int
    sweeps: int
    ridge: float
    configured_tau: float
    coordinate_half_width: float
    config_seed: int
    row_design: str
    defensive_nu: float | None
    basis_identity: str
    prefix_core_values: tuple[tf.Tensor, ...]
    suffix_gram: tf.Tensor
    z_h: tf.Tensor
    tau_abs: tf.Tensor
    z_complete: tf.Tensor
    coordinate_offset: tf.Tensor
    coordinate_matrix: tf.Tensor
    raw_increment: tf.Tensor
    corrected_increment: tf.Tensor


def _assemble_transition_target(
    *,
    adapter,
    current_basis: ProductBasis,
    prefix_shapes: Sequence[tuple[int, int, int]],
    branch_gram_floor: float,
    defensive_nu: float | None,
    prefix_values: Sequence[tf.Tensor],
    gram: tf.Tensor,
    tau_abs_prev: tf.Tensor,
    u_rows: tf.Tensor,
    u_weights: tf.Tensor,
    y: tf.Tensor,
    m_c: tf.Tensor,
    l_cc: tf.Tensor,
    m_p: tf.Tensor,
    l_pc: tf.Tensor,
    l_pp: tf.Tensor,
    l_old: tf.Tensor,
    m_old: tf.Tensor,
    frozen_shift: tf.Tensor | None = None,
) -> _TransitionTarget:
    """Assemble the production transition target on a supplied row design."""

    n = int(m_c.shape[0])
    prefix_cores = tuple(
        TTCore(tf.reshape(v, s)) for v, s in zip(prefix_values, prefix_shapes)
    )
    floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
    chol = tf.linalg.cholesky(
        gram
        + tf.constant(branch_gram_floor, DTYPE)
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
    logdet_old = tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(l_old))))
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
        floor_values = tau_abs_prev * tf.ones([tf.shape(u_rows)[0]], DTYPE)
    else:
        floor_values = tau_abs_prev * tf.exp(
            _log_student_t_ratio(u_old, defensive_nu)
        )
    sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + floor_values
    log_f = tf.math.log(sum_sq) + log_g
    computed_shift = tf.reduce_logsumexp(log_f) - tf.math.log(
        tf.cast(tf.shape(log_f)[0], DTYPE)
    )
    shift = computed_shift if frozen_shift is None else tf.convert_to_tensor(
        frozen_shift, DTYPE
    )
    sqrt_g_shifted = tf.exp(0.5 * (log_g - shift))
    amplitudes = tf.concat([v_prev, tf.sqrt(floor_values)[:, None]], axis=1)
    targets = amplitudes * sqrt_g_shifted[:, None]
    g_codes = tf.tile(
        tf.range(branch_count, dtype=DTYPE)[None, :],
        [tf.shape(u_rows)[0], 1],
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
    fit_weights = tf.reshape(
        tf.repeat(u_weights, branch_count, axis=0), [-1]
    )
    return _TransitionTarget(
        physical_rows=u_rows,
        expanded_rows=full_rows,
        sqrt_target=sqrt_target,
        fit_weights=fit_weights,
        shift=shift,
        computed_shift=computed_shift,
        log_g=log_g,
        log_f=log_f,
        floor_values=floor_values,
        amplitudes=amplitudes,
        u_old_max=u_old_max,
    )


def _transition_target_summary(target: _TransitionTarget) -> tf.Tensor:
    branch_count = tf.shape(target.amplitudes)[1]
    branch_target = tf.reshape(
        target.sqrt_target, [tf.shape(target.physical_rows)[0], branch_count]
    )
    branch_energy = tf.reduce_sum(tf.square(branch_target), axis=1)
    expected_energy = tf.exp(target.log_f - target.shift)
    closure_scale = tf.maximum(
        tf.reduce_max(tf.abs(expected_energy)), tf.constant(1e-300, DTYPE)
    )
    closure = tf.reduce_max(tf.abs(branch_energy - expected_energy)) / closure_scale
    finite = tf.reduce_all(
        tf.concat(
            [
                tf.math.is_finite(target.log_g),
                tf.math.is_finite(target.log_f),
                tf.math.is_finite(target.sqrt_target),
            ],
            axis=0,
        )
    )
    floor_dominates = tf.reduce_mean(
        tf.cast(
            target.floor_values
            >= tf.reduce_max(tf.square(target.amplitudes[:, :-1]), axis=1),
            DTYPE,
        )
    )
    return tf.stack(
        [
            tf.cast(finite, DTYPE),
            tf.reduce_min(target.log_g),
            tf.reduce_max(target.log_g),
            tf.reduce_min(target.log_f),
            tf.reduce_max(target.log_f),
            tf.reduce_min(target.sqrt_target),
            tf.reduce_max(target.sqrt_target),
            closure,
            floor_dominates,
        ]
    )


def _transition_input_signature(
    prefix_shapes: Sequence[tuple[int, int, int]],
    mixed_shapes: Sequence[tuple[int, int, int]],
    *,
    n: int,
    row_count: int,
) -> tuple:
    boundary_rank = int(prefix_shapes[-1][-1])
    return (
        tuple(tf.TensorSpec(shape, DTYPE) for shape in prefix_shapes),
        tf.TensorSpec([boundary_rank, boundary_rank], DTYPE),
        tf.TensorSpec([], DTYPE),
        tf.TensorSpec([row_count, 2 * n], DTYPE),
        tf.TensorSpec([row_count], DTYPE),
        tuple(tf.TensorSpec(shape, DTYPE) for shape in mixed_shapes),
        tf.TensorSpec([n], DTYPE),
        tf.TensorSpec([n], DTYPE),
        tf.TensorSpec([n, n], DTYPE),
        tf.TensorSpec([n], DTYPE),
        tf.TensorSpec([n, n], DTYPE),
        tf.TensorSpec([n, n], DTYPE),
        tf.TensorSpec([n, n], DTYPE),
        tf.TensorSpec([n], DTYPE),
    )


def _run_value_filter_branch_axis_gaussian_xla(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_moment_hint: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    defensive_nu: float | None = None,
    capture_steps: frozenset[int] = frozenset(),
    run_identity: str = "",
    retained_proposal_capture: bool = False,
) -> tuple[
    tf.Tensor,
    list[dict],
    dict[
        int,
        GaussianXLAFrozenTransitionSnapshot | GaussianXLARetainedProposalSnapshot,
    ],
]:
    if config.quadrature_order is not None:
        raise ValueError("gaussian engine is defined for scattered rows only")
    if predictive_moment_hint is None or initial_moment_hint is None:
        raise ValueError("gaussian engine requires frozen moment hints")

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    invalid_capture_steps = sorted(
        step for step in capture_steps if step <= 0 or step >= horizon
    )
    if invalid_capture_steps:
        raise ValueError(
            "transition capture steps must be in [1, horizon); "
            f"got {invalid_capture_steps}"
        )
    if capture_steps and not run_identity:
        raise ValueError("diagnostic capture requires a non-empty run_identity")
    current_basis = _hermite_product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    ridge = tf.constant(config.ridge, DTYPE)
    fitter = FixedTTFitter()
    per_adapter = _STEP_CACHE.setdefault(adapter, {})
    step_cache = per_adapter.setdefault(config, {})

    def _make_transition_fit(
        mixed_basis,
        mixed_shapes,
        prefix_shapes,
        *,
        capture_full: bool,
    ):
        input_signature = _transition_input_signature(
            prefix_shapes,
            mixed_shapes,
            n=n,
            row_count=config.row_count,
        )

        @tf.function(input_signature=input_signature, jit_compile=True)
        def transition_fit(
            prefix_values, gram, tau_abs_prev, u_rows, u_weights, core0_values,
            y, m_c, l_cc, m_p, l_pc, l_pp, l_old, m_old,
        ):
            target = _assemble_transition_target(
                adapter=adapter,
                current_basis=current_basis,
                prefix_shapes=prefix_shapes,
                branch_gram_floor=config.branch_gram_floor,
                defensive_nu=defensive_nu,
                prefix_values=prefix_values,
                gram=gram,
                tau_abs_prev=tau_abs_prev,
                u_rows=u_rows,
                u_weights=u_weights,
                y=y,
                m_c=m_c,
                l_cc=l_cc,
                m_p=m_p,
                l_pc=l_pc,
                l_pp=l_pp,
                l_old=l_old,
                m_old=m_old,
            )
            cores, worst, rms = _fit_als_graph(
                fitter,
                mixed_basis,
                target.expanded_rows,
                target.sqrt_target,
                target.fit_weights,
                core0_values, mixed_shapes, config.sweeps, ridge,
            )
            new_gram = suffix_gram_matrix(
                tuple(cores[n:]), mixed_basis, axis_offset=n
            )
            p_gram = prefix_gram_matrix(tuple(cores[:n]), mixed_basis)
            z_h_new = tf.einsum("ab,ab->", p_gram, new_gram)
            production_outputs = (
                [c.values for c in cores[:n]],
                new_gram,
                z_h_new,
                target.shift,
                worst,
                rms,
                target.u_old_max,
            )
            if capture_full:
                return production_outputs + (
                    _transition_target_summary(target),
                    tuple(c.values for c in cores),
                )
            # Preserve the original production graph exactly: even adding an
            # otherwise-unused observability output can perturb this highly
            # ill-conditioned ALS/XLA computation.  Retained capture copies
            # these seven tensors on the host; full capture is the separate
            # observability route above.
            return production_outputs

        return transition_fit

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[dict] = []
    snapshots: dict[
        int,
        GaussianXLAFrozenTransitionSnapshot | GaussianXLARetainedProposalSnapshot,
    ] = {}
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
        capture_retained = retained_proposal_capture and t in capture_steps
        capture_full = not retained_proposal_capture and t in capture_steps
        # Retained capture copies the original seven production outputs on the
        # host.  It deliberately shares this graph/cache entry with the normal
        # value path so capture cannot change the fitted finite program.
        cache_key = (branch_count, defensive_nu, capture_full)
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
                _make_transition_fit(
                    mixed_basis,
                    mixed_shapes,
                    prefix_shapes,
                    capture_full=capture_full,
                ),
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
        if capture_full:
            prefix_values_previous = tuple(tf.identity(v) for v in prefix_values)
            gram_previous = tf.identity(gram)
            tau_abs_previous = tf.identity(retained.tau)
            z_complete_previous = tf.identity(zc)
        transition_outputs = transition_fit(
            tuple(prefix_values),
            gram,
            retained.tau,
            u_rows,
            u_weights,
            tuple(core0_values),
            observations[t],
            m_c,
            l_cc,
            m_p,
            l_pc,
            l_pp,
            l_old,
            m_old,
        )
        if capture_full:
            (
                prefix_values_new,
                gram_new,
                z_h_new,
                shift,
                worst,
                rms,
                u_old_max,
                target_summary_tensor,
                fitted_core_values,
            ) = transition_outputs
        else:
            (
                prefix_values_new,
                gram_new,
                z_h_new,
                shift,
                worst,
                rms,
                u_old_max,
            ) = transition_outputs
            target_summary = {}
            fitted_core_values = None
        if capture_full:
            target_summary = {
                name: float(value.numpy())
                for name, value in zip(
                    _TARGET_SUMMARY_NAMES, tf.unstack(target_summary_tensor)
                )
            }
        # Class-A observability (n=4 diagnosis 2026-08-27): the retained-Gram
        # spectrum decides whether the unpropagated branch_gram_floor can
        # matter. A 1e-12 relative ridge only perturbs the quadratic form
        # materially when cond(E) >~ 1e12; cond(E) was previously unrecorded
        # (the manifest's cond_max is the ALS design-matrix condition).
        # Computed eagerly, outside the jit_compile kernel, per the TF policy
        # on keeping non-kernel validation out of the compiled hot path.
        gram_new_eigvals = tf.linalg.eigvalsh(gram_new)
        gram_lambda_min = float(gram_new_eigvals[0].numpy())
        gram_lambda_max = float(gram_new_eigvals[-1].numpy())
        gram_cond = gram_lambda_max / max(abs(gram_lambda_min), 1e-300)
        gram_sym_err = float(
            tf.linalg.norm(gram_new - tf.transpose(gram_new))
            / tf.maximum(tf.linalg.norm(gram_new), tf.constant(1e-300, DTYPE))
        )

        tau_t, eps_rel_sq = _clamped_tau(float(rms.numpy()), z_h_new)

        # Class-A observability: the branch axis repeats each physical row
        # branch_count times and its Gram mass matrix is counting measure,
        # while `rms` is normalized by the repeated-row weight total. The
        # counting-measure residual is therefore branch_count x larger.
        # RECORDED ONLY -- tau_t above is unchanged. Re-calibrating tau on
        # eps_rel_sq_counting is a Class-C numerics-altering change that would
        # move the veto threshold and break comparability with the n=2 verdict
        # (r*(2)=6 was established under the current calibration), so it needs
        # its own no-harm evaluation and owner decision. See the fix plan.
        branch_count_diag = retained.boundary_rank + 1
        eps_rel_sq_counting = eps_rel_sq * branch_count_diag

        zc_new = (1.0 + tau_t) * z_h_new
        log_increment = shift + tf.math.log(zc_new) - tf.math.log(zc)
        corrected_increment = log_increment - tf.math.log1p(tau_t)
        increment_value = float(log_increment.numpy())
        if not math.isfinite(increment_value):
            raise ValueError("non-finite step increment (fail-closed)")
        if capture_retained:
            snapshots[t] = GaussianXLARetainedProposalSnapshot(
                run_identity=run_identity,
                time_index=t,
                state_dim=n,
                basis_degree=config.basis_degree,
                rank=config.rank,
                row_count=config.row_count,
                sweeps=config.sweeps,
                ridge=config.ridge,
                configured_tau=config.tau,
                coordinate_half_width=config.coordinate_half_width,
                config_seed=config.seed,
                row_design=config.row_design,
                defensive_nu=defensive_nu,
                basis_identity="hermite_retained_quadratic_form_v1",
                prefix_core_values=tuple(
                    tf.identity(value) for value in prefix_values_new
                ),
                suffix_gram=tf.identity(gram_new),
                z_h=tf.identity(z_h_new),
                tau_abs=tf.identity(tau_t * z_h_new),
                z_complete=tf.identity(zc_new),
                coordinate_offset=tf.identity(m_c),
                coordinate_matrix=tf.identity(l_cc),
                raw_increment=tf.identity(log_increment),
                corrected_increment=tf.identity(corrected_increment),
            )
        elif capture_full:
            if target_summary["all_finite"] != 1.0:
                raise ValueError("non-finite captured transition target (fail-closed)")
            snapshots[t] = GaussianXLAFrozenTransitionSnapshot(
                run_identity=run_identity,
                time_index=t,
                state_dim=n,
                basis_degree=config.basis_degree,
                rank=config.rank,
                row_count=config.row_count,
                sweeps=config.sweeps,
                ridge=config.ridge,
                configured_tau=config.tau,
                coordinate_half_width=config.coordinate_half_width,
                config_seed=config.seed,
                condition_number_veto=config.condition_number_veto,
                branch_gram_floor=config.branch_gram_floor,
                row_design=config.row_design,
                training_row_seed=(config.seed, 100 + t),
                defensive_nu=defensive_nu,
                branch_count=branch_count,
                basis_identity="hermite_reference_counting_branch_v1",
                mixed_shapes=tuple(tuple(shape) for shape in mixed_shapes),
                prefix_values=prefix_values_previous,
                suffix_gram=gram_previous,
                tau_abs_previous=tau_abs_previous,
                z_complete_previous=z_complete_previous,
                old_coordinate_offset=tf.identity(m_old),
                old_coordinate_matrix=tf.identity(l_old),
                joint_mean=tf.identity(joint_mean),
                joint_chol=tf.identity(joint_chol),
                observation=tf.identity(observations[t]),
                training_rows=tf.identity(u_rows),
                training_weights=tf.identity(u_weights),
                frozen_shift=tf.identity(shift),
                fitted_core_values=tuple(
                    tf.identity(v) for v in fitted_core_values
                ),
                z_h=tf.identity(z_h_new),
                raw_increment=tf.identity(log_increment),
                corrected_increment=tf.identity(corrected_increment),
                worst_condition=tf.identity(worst),
                weighted_fit_rms=tf.identity(rms),
                u_old_max=tf.identity(u_old_max),
                target_summary=dict(target_summary),
            )
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
                # n=4 diagnosis 2026-08-27: Gram health and counting-measure RMS
                "gram_lambda_min": gram_lambda_min,
                "gram_lambda_max": gram_lambda_max,
                "gram_cond": gram_cond,
                "gram_sym_err": gram_sym_err,
                "eps_rel_sq_counting": eps_rel_sq_counting,
                "branch_count": branch_count_diag,
                **{f"target_{key}": value for key, value in target_summary.items()},
            }
        )
    return log_likelihood, diagnostics, snapshots


_SNAPSHOT_SCALAR_TENSORS = (
    "tau_abs_previous",
    "z_complete_previous",
    "frozen_shift",
    "z_h",
    "raw_increment",
    "corrected_increment",
    "worst_condition",
    "weighted_fit_rms",
    "u_old_max",
)
_SNAPSHOT_ARRAY_TENSORS = (
    "suffix_gram",
    "old_coordinate_offset",
    "old_coordinate_matrix",
    "joint_mean",
    "joint_chol",
    "observation",
    "training_rows",
    "training_weights",
)


def gaussian_xla_frozen_snapshot_parts(
    snapshot: GaussianXLAFrozenTransitionSnapshot,
) -> tuple[dict, dict[str, tf.Tensor]]:
    """Return JSON metadata and TensorFlow tensors for lossless persistence."""

    tensors = {
        name: tf.convert_to_tensor(getattr(snapshot, name), DTYPE)
        for name in _SNAPSHOT_SCALAR_TENSORS + _SNAPSHOT_ARRAY_TENSORS
    }
    for index, value in enumerate(snapshot.prefix_values):
        tensors[f"prefix_{index:03d}"] = tf.convert_to_tensor(value, DTYPE)
    for index, value in enumerate(snapshot.fitted_core_values):
        tensors[f"fitted_{index:03d}"] = tf.convert_to_tensor(value, DTYPE)
    metadata = {
        "schema_id": "gaussian_xla_frozen_transition_snapshot_v1",
        "run_identity": snapshot.run_identity,
        "time_index": snapshot.time_index,
        "state_dim": snapshot.state_dim,
        "basis_degree": snapshot.basis_degree,
        "rank": snapshot.rank,
        "row_count": snapshot.row_count,
        "sweeps": snapshot.sweeps,
        "ridge": snapshot.ridge,
        "configured_tau": snapshot.configured_tau,
        "coordinate_half_width": snapshot.coordinate_half_width,
        "config_seed": snapshot.config_seed,
        "condition_number_veto": snapshot.condition_number_veto,
        "branch_gram_floor": snapshot.branch_gram_floor,
        "row_design": snapshot.row_design,
        "training_row_seed": list(snapshot.training_row_seed),
        "defensive_nu": snapshot.defensive_nu,
        "branch_count": snapshot.branch_count,
        "basis_identity": snapshot.basis_identity,
        "mixed_shapes": [list(shape) for shape in snapshot.mixed_shapes],
        "prefix_count": len(snapshot.prefix_values),
        "fitted_core_count": len(snapshot.fitted_core_values),
        "target_summary": dict(snapshot.target_summary),
        "tensor_shapes": {
            name: tensor.shape.as_list() for name, tensor in tensors.items()
        },
        "tensor_dtypes": {name: tensor.dtype.name for name, tensor in tensors.items()},
    }
    return metadata, tensors


def gaussian_xla_frozen_snapshot_from_parts(
    metadata: Mapping[str, object],
    tensors: Mapping[str, tf.Tensor],
) -> GaussianXLAFrozenTransitionSnapshot:
    """Reconstruct and validate a snapshot loaded from persisted parts."""

    if metadata.get("schema_id") != "gaussian_xla_frozen_transition_snapshot_v1":
        raise ValueError("unknown frozen-transition snapshot schema")
    shapes = metadata.get("tensor_shapes")
    dtypes = metadata.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        raise ValueError("snapshot metadata is missing tensor shape/dtype identity")

    def _read(name: str) -> tf.Tensor:
        if name not in tensors or name not in shapes or name not in dtypes:
            raise ValueError(f"snapshot tensor {name!r} is missing")
        if dtypes[name] != DTYPE.name:
            raise ValueError(
                f"snapshot tensor {name!r} has dtype {dtypes[name]!r}, "
                f"expected {DTYPE.name!r}"
            )
        return tf.ensure_shape(tf.convert_to_tensor(tensors[name], DTYPE), shapes[name])

    prefix_count = int(metadata["prefix_count"])
    fitted_core_count = int(metadata["fitted_core_count"])
    prefix_values = tuple(_read(f"prefix_{index:03d}") for index in range(prefix_count))
    fitted_core_values = tuple(
        _read(f"fitted_{index:03d}") for index in range(fitted_core_count)
    )
    mixed_shapes = tuple(
        tuple(int(value) for value in shape) for shape in metadata["mixed_shapes"]
    )
    n = int(metadata["state_dim"])
    branch_count = int(metadata["branch_count"])
    if prefix_count != n or fitted_core_count != 2 * n + 1:
        raise ValueError("snapshot core count is inconsistent with state dimension")
    fitted_shapes = tuple(
        tuple(tensor.shape.as_list()) for tensor in fitted_core_values
    )
    if fitted_shapes != mixed_shapes:
        raise ValueError("snapshot fitted-core shapes do not match mixed_shapes")
    if int(prefix_values[-1].shape[-1]) + 1 != branch_count:
        raise ValueError("snapshot branch count does not close the prefix boundary")

    return GaussianXLAFrozenTransitionSnapshot(
        run_identity=str(metadata["run_identity"]),
        time_index=int(metadata["time_index"]),
        state_dim=n,
        basis_degree=int(metadata["basis_degree"]),
        rank=int(metadata["rank"]),
        row_count=int(metadata["row_count"]),
        sweeps=int(metadata["sweeps"]),
        ridge=float(metadata["ridge"]),
        configured_tau=float(metadata["configured_tau"]),
        coordinate_half_width=float(metadata["coordinate_half_width"]),
        config_seed=int(metadata["config_seed"]),
        condition_number_veto=float(metadata["condition_number_veto"]),
        branch_gram_floor=float(metadata["branch_gram_floor"]),
        row_design=str(metadata["row_design"]),
        training_row_seed=tuple(int(value) for value in metadata["training_row_seed"]),
        defensive_nu=(
            None if metadata["defensive_nu"] is None else float(metadata["defensive_nu"])
        ),
        branch_count=branch_count,
        basis_identity=str(metadata["basis_identity"]),
        mixed_shapes=mixed_shapes,
        prefix_values=prefix_values,
        suffix_gram=_read("suffix_gram"),
        tau_abs_previous=_read("tau_abs_previous"),
        z_complete_previous=_read("z_complete_previous"),
        old_coordinate_offset=_read("old_coordinate_offset"),
        old_coordinate_matrix=_read("old_coordinate_matrix"),
        joint_mean=_read("joint_mean"),
        joint_chol=_read("joint_chol"),
        observation=_read("observation"),
        training_rows=_read("training_rows"),
        training_weights=_read("training_weights"),
        frozen_shift=_read("frozen_shift"),
        fitted_core_values=fitted_core_values,
        z_h=_read("z_h"),
        raw_increment=_read("raw_increment"),
        corrected_increment=_read("corrected_increment"),
        worst_condition=_read("worst_condition"),
        weighted_fit_rms=_read("weighted_fit_rms"),
        u_old_max=_read("u_old_max"),
        target_summary={
            str(name): float(value)
            for name, value in metadata["target_summary"].items()
        },
    )


def gaussian_xla_frozen_snapshot_fingerprint(
    snapshot: GaussianXLAFrozenTransitionSnapshot,
) -> str:
    """Hash metadata and serialized TensorFlow values for state identity."""

    metadata, tensors = gaussian_xla_frozen_snapshot_parts(snapshot)
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(bytes(tf.io.serialize_tensor(tensors[name]).numpy()))
    return digest.hexdigest()


_RETAINED_SNAPSHOT_SCALAR_TENSORS = (
    "z_h",
    "tau_abs",
    "z_complete",
    "raw_increment",
    "corrected_increment",
)
_RETAINED_SNAPSHOT_ARRAY_TENSORS = (
    "suffix_gram",
    "coordinate_offset",
    "coordinate_matrix",
)


def gaussian_xla_retained_proposal_snapshot_parts(
    snapshot: GaussianXLARetainedProposalSnapshot,
) -> tuple[dict, dict[str, tf.Tensor]]:
    """Return lossless parts for a post-update retained proposal snapshot."""

    tensors = {
        name: tf.convert_to_tensor(getattr(snapshot, name), DTYPE)
        for name in (
            _RETAINED_SNAPSHOT_SCALAR_TENSORS
            + _RETAINED_SNAPSHOT_ARRAY_TENSORS
        )
    }
    for index, value in enumerate(snapshot.prefix_core_values):
        tensors[f"prefix_{index:03d}"] = tf.convert_to_tensor(value, DTYPE)
    metadata = {
        "schema_id": "gaussian_xla_retained_proposal_snapshot_v1",
        "run_identity": snapshot.run_identity,
        "time_index": snapshot.time_index,
        "state_dim": snapshot.state_dim,
        "basis_degree": snapshot.basis_degree,
        "rank": snapshot.rank,
        "row_count": snapshot.row_count,
        "sweeps": snapshot.sweeps,
        "ridge": snapshot.ridge,
        "configured_tau": snapshot.configured_tau,
        "coordinate_half_width": snapshot.coordinate_half_width,
        "config_seed": snapshot.config_seed,
        "row_design": snapshot.row_design,
        "defensive_nu": snapshot.defensive_nu,
        "basis_identity": snapshot.basis_identity,
        "prefix_count": len(snapshot.prefix_core_values),
        "capture_semantics": "host_copy_of_original_seven_output_fit_graph",
        "tensor_shapes": {
            name: tensor.shape.as_list() for name, tensor in tensors.items()
        },
        "tensor_dtypes": {
            name: tensor.dtype.name for name, tensor in tensors.items()
        },
    }
    return metadata, tensors


def gaussian_xla_retained_proposal_snapshot_from_parts(
    metadata: Mapping[str, object],
    tensors: Mapping[str, tf.Tensor],
) -> GaussianXLARetainedProposalSnapshot:
    """Reconstruct and validate a persisted retained proposal snapshot."""

    if metadata.get("schema_id") != "gaussian_xla_retained_proposal_snapshot_v1":
        raise ValueError("unknown retained-proposal snapshot schema")
    shapes = metadata.get("tensor_shapes")
    dtypes = metadata.get("tensor_dtypes")
    if not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        raise ValueError("snapshot metadata is missing tensor shape/dtype identity")

    def _read(name: str) -> tf.Tensor:
        if name not in tensors or name not in shapes or name not in dtypes:
            raise ValueError(f"snapshot tensor {name!r} is missing")
        if dtypes[name] != DTYPE.name:
            raise ValueError(
                f"snapshot tensor {name!r} has dtype {dtypes[name]!r}, "
                f"expected {DTYPE.name!r}"
            )
        value = tf.ensure_shape(
            tf.convert_to_tensor(tensors[name], DTYPE), shapes[name]
        )
        if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
            raise ValueError(f"snapshot tensor {name!r} contains non-finite values")
        return value

    n = int(metadata["state_dim"])
    prefix_count = int(metadata["prefix_count"])
    if prefix_count != n:
        raise ValueError("snapshot prefix count is inconsistent with state dimension")
    prefix_values = tuple(
        _read(f"prefix_{index:03d}") for index in range(prefix_count)
    )
    boundary_rank = int(prefix_values[-1].shape[-1])
    suffix_gram = _read("suffix_gram")
    if suffix_gram.shape != tf.TensorShape([boundary_rank, boundary_rank]):
        raise ValueError("snapshot suffix Gram does not close the prefix boundary")
    coordinate_offset = _read("coordinate_offset")
    coordinate_matrix = _read("coordinate_matrix")
    if coordinate_offset.shape != tf.TensorShape([n]):
        raise ValueError("snapshot coordinate offset has the wrong dimension")
    if coordinate_matrix.shape != tf.TensorShape([n, n]):
        raise ValueError("snapshot coordinate matrix has the wrong dimension")
    z_h = _read("z_h")
    tau_abs = _read("tau_abs")
    z_complete = _read("z_complete")
    closure_scale = max(1.0, abs(float(z_complete.numpy())))
    if abs(float((z_complete - z_h - tau_abs).numpy())) > 2e-12 * closure_scale:
        raise ValueError("snapshot complete mass does not equal z_h + tau_abs")

    return GaussianXLARetainedProposalSnapshot(
        run_identity=str(metadata["run_identity"]),
        time_index=int(metadata["time_index"]),
        state_dim=n,
        basis_degree=int(metadata["basis_degree"]),
        rank=int(metadata["rank"]),
        row_count=int(metadata["row_count"]),
        sweeps=int(metadata["sweeps"]),
        ridge=float(metadata["ridge"]),
        configured_tau=float(metadata["configured_tau"]),
        coordinate_half_width=float(metadata["coordinate_half_width"]),
        config_seed=int(metadata["config_seed"]),
        row_design=str(metadata["row_design"]),
        defensive_nu=(
            None
            if metadata["defensive_nu"] is None
            else float(metadata["defensive_nu"])
        ),
        basis_identity=str(metadata["basis_identity"]),
        prefix_core_values=prefix_values,
        suffix_gram=suffix_gram,
        z_h=z_h,
        tau_abs=tau_abs,
        z_complete=z_complete,
        coordinate_offset=coordinate_offset,
        coordinate_matrix=coordinate_matrix,
        raw_increment=_read("raw_increment"),
        corrected_increment=_read("corrected_increment"),
    )


def gaussian_xla_retained_proposal_snapshot_fingerprint(
    snapshot: GaussianXLARetainedProposalSnapshot,
) -> str:
    """Hash metadata and tensors defining a retained proposal snapshot."""

    metadata, tensors = gaussian_xla_retained_proposal_snapshot_parts(snapshot)
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(bytes(tf.io.serialize_tensor(tensors[name]).numpy()))
    return digest.hexdigest()


def _make_frozen_transition_evaluator(
    adapter,
    snapshot: GaussianXLAFrozenTransitionSnapshot,
    *,
    row_count: int,
    jit_compile: bool,
):
    n = snapshot.state_dim
    prefix_shapes = tuple(tuple(v.shape.as_list()) for v in snapshot.prefix_values)
    mixed_shapes = snapshot.mixed_shapes
    boundary_rank = int(prefix_shapes[-1][-1])
    current_basis = _hermite_product_basis(n, snapshot.basis_degree)
    branch_basis = DiscreteIndicatorBasis1D(snapshot.branch_count)
    mixed_basis = ProductBasis(
        list(current_basis.bases)
        + [branch_basis]
        + list(_hermite_product_basis(n, snapshot.basis_degree).bases),
        current_basis.convention,
    )
    input_signature = (
        tuple(tf.TensorSpec(shape, DTYPE) for shape in prefix_shapes),
        tf.TensorSpec([boundary_rank, boundary_rank], DTYPE),
        tf.TensorSpec([], DTYPE),
        tf.TensorSpec([row_count, 2 * n], DTYPE),
        tf.TensorSpec([row_count], DTYPE),
        tf.TensorSpec([n], DTYPE),
        tf.TensorSpec([2 * n], DTYPE),
        tf.TensorSpec([2 * n, 2 * n], DTYPE),
        tf.TensorSpec([n, n], DTYPE),
        tf.TensorSpec([n], DTYPE),
        tf.TensorSpec([], DTYPE),
        tuple(tf.TensorSpec(shape, DTYPE) for shape in mixed_shapes),
        tf.TensorSpec([], DTYPE),
    )

    @tf.function(input_signature=input_signature, jit_compile=jit_compile)
    def evaluate(
        prefix_values,
        gram,
        tau_abs_previous,
        rows,
        weights,
        observation,
        joint_mean,
        joint_chol,
        old_coordinate_matrix,
        old_coordinate_offset,
        frozen_shift,
        fitted_core_values,
        expected_z_h,
    ):
        target = _assemble_transition_target(
            adapter=adapter,
            current_basis=current_basis,
            prefix_shapes=prefix_shapes,
            branch_gram_floor=snapshot.branch_gram_floor,
            defensive_nu=snapshot.defensive_nu,
            prefix_values=prefix_values,
            gram=gram,
            tau_abs_prev=tau_abs_previous,
            u_rows=rows,
            u_weights=weights,
            y=observation,
            m_c=joint_mean[:n],
            l_cc=joint_chol[:n, :n],
            m_p=joint_mean[n:],
            l_pc=joint_chol[n:, :n],
            l_pp=joint_chol[n:, n:],
            l_old=old_coordinate_matrix,
            m_old=old_coordinate_offset,
            frozen_shift=frozen_shift,
        )
        full_cores = tuple(
            TTCore(tf.reshape(value, shape))
            for value, shape in zip(fitted_core_values, mixed_shapes)
        )
        prediction = prefix_row_vectors(
            full_cores, mixed_basis, target.expanded_rows
        )[:, 0]
        residual = prediction - target.sqrt_target
        branch_target = tf.reshape(
            tf.square(target.sqrt_target), [row_count, snapshot.branch_count]
        )
        branch_residual = tf.reshape(
            tf.square(residual), [row_count, snapshot.branch_count]
        )
        branch_prediction = tf.reshape(
            tf.square(prediction), [row_count, snapshot.branch_count]
        )
        row_target_energy = tf.reduce_sum(branch_target, axis=1)
        row_residual_energy = tf.reduce_sum(branch_residual, axis=1)
        row_prediction_energy = tf.reduce_sum(branch_prediction, axis=1)
        z_t = tf.reduce_sum(weights * row_target_energy)
        z_h_qmc = tf.reduce_sum(weights * row_prediction_energy)
        counting_residual = tf.reduce_sum(weights * row_residual_energy)
        emitted_rms = tf.sqrt(
            counting_residual / tf.reduce_sum(target.fit_weights)
        )
        full_gram = prefix_gram_matrix(full_cores, mixed_basis)
        z_h_direct = full_gram[0, 0]
        prefix_gram = prefix_gram_matrix(full_cores[:n], mixed_basis)
        suffix_gram = suffix_gram_matrix(
            full_cores[n:], mixed_basis, axis_offset=n
        )
        z_h_factored = tf.einsum("ab,ab->", prefix_gram, suffix_gram)
        rho_h_qmc = counting_residual / z_h_qmc
        rho_h_exact_denominator = counting_residual / expected_z_h
        sqrt_rho = tf.sqrt(
            tf.maximum(rho_h_qmc, tf.constant(0.0, DTYPE))
        )
        reverse_bound_valid = sqrt_rho < 1.0
        reverse_lower = -2.0 * tf.math.log1p(sqrt_rho)
        reverse_upper_value = -2.0 * tf.math.log(
            tf.maximum(1.0 - sqrt_rho, tf.constant(1e-300, DTYPE))
        )
        reverse_upper = tf.where(
            reverse_bound_valid,
            reverse_upper_value,
            tf.constant(0.0, DTYPE),
        )
        return (
            z_t,
            z_h_direct,
            z_h_factored,
            z_h_qmc,
            counting_residual,
            emitted_rms,
            rho_h_qmc,
            rho_h_exact_denominator,
            tf.cast(reverse_bound_valid, DTYPE),
            reverse_lower,
            reverse_upper,
            tf.math.log(expected_z_h) - tf.math.log(z_t),
            tf.math.log(z_h_qmc) - tf.math.log(z_t),
            tf.math.log(expected_z_h) - tf.math.log(z_h_qmc),
            target.computed_shift - frozen_shift,
            tf.reduce_sum(target.fit_weights),
            row_target_energy,
            row_residual_energy,
            row_prediction_energy,
            tf.reduce_max(tf.abs(rows), axis=1),
            prediction,
            target.sqrt_target,
            _transition_target_summary(target),
        )

    return evaluate


def evaluate_gaussian_xla_frozen_transition(
    snapshot: GaussianXLAFrozenTransitionSnapshot,
    adapter,
    rows: tf.Tensor,
    weights: tf.Tensor,
    *,
    shift_offset: float = 0.0,
    jit_compile: bool = True,
) -> dict[str, tf.Tensor]:
    """Evaluate a captured fitted TT and its frozen production target."""

    if adapter.state_dim != snapshot.state_dim:
        raise ValueError("snapshot and adapter state dimensions differ")
    if snapshot.basis_identity != "hermite_reference_counting_branch_v1":
        raise ValueError("snapshot basis identity is not supported")
    rows = tf.convert_to_tensor(rows, DTYPE)
    weights = tf.convert_to_tensor(weights, DTYPE)
    if rows.shape.rank != 2 or int(rows.shape[1]) != 2 * snapshot.state_dim:
        raise ValueError("diagnostic rows must have shape [N, 2 * state_dim]")
    if rows.shape[0] is None:
        raise ValueError("diagnostic evaluator requires a setup-static row count")
    row_count = int(rows.shape[0])
    if weights.shape != tf.TensorShape([row_count]):
        raise ValueError("diagnostic weights must have shape [N]")
    if not bool(tf.reduce_all(tf.math.is_finite(rows)).numpy()):
        raise ValueError("diagnostic rows contain non-finite values")
    if not bool(tf.reduce_all(tf.math.is_finite(weights)).numpy()):
        raise ValueError("diagnostic weights contain non-finite values")
    if bool((tf.reduce_min(weights) < 0.0).numpy()):
        raise ValueError("diagnostic weights must be nonnegative")

    prefix_shapes = tuple(tuple(v.shape.as_list()) for v in snapshot.prefix_values)
    key = (
        snapshot.state_dim,
        snapshot.basis_degree,
        snapshot.branch_count,
        prefix_shapes,
        snapshot.mixed_shapes,
        snapshot.branch_gram_floor,
        snapshot.defensive_nu,
        row_count,
        bool(jit_compile),
    )
    per_adapter = _FROZEN_EVALUATOR_CACHE.setdefault(adapter, {})
    if key not in per_adapter:
        per_adapter[key] = _make_frozen_transition_evaluator(
            adapter,
            snapshot,
            row_count=row_count,
            jit_compile=bool(jit_compile),
        )
    evaluate = per_adapter[key]
    outputs = evaluate(
        tuple(snapshot.prefix_values),
        snapshot.suffix_gram,
        snapshot.tau_abs_previous,
        rows,
        weights,
        snapshot.observation,
        snapshot.joint_mean,
        snapshot.joint_chol,
        snapshot.old_coordinate_matrix,
        snapshot.old_coordinate_offset,
        snapshot.frozen_shift + tf.constant(shift_offset, DTYPE),
        tuple(snapshot.fitted_core_values),
        snapshot.z_h,
    )
    names = (
        "z_t",
        "z_h_direct",
        "z_h_factored",
        "z_h_qmc",
        "counting_residual",
        "emitted_rms",
        "rho_h_qmc",
        "rho_h_exact_denominator",
        "reverse_triangle_bound_valid",
        "reverse_triangle_log_lower",
        "reverse_triangle_log_upper",
        "fit_log_ratio_exact",
        "fit_log_ratio_qmc",
        "gram_vs_qmc_log_gap",
        "recomputed_shift_delta",
        "expanded_weight_sum",
        "row_target_energy",
        "row_residual_energy",
        "row_prediction_energy",
        "row_u_abs_max",
        "prediction",
        "sqrt_target",
        "target_summary_tensor",
    )
    result = dict(zip(names, outputs))
    summary_values = tf.unstack(result.pop("target_summary_tensor"))
    result.update(
        {
            f"target_{name}": value
            for name, value in zip(_TARGET_SUMMARY_NAMES, summary_values)
        }
    )
    return result


def run_value_filter_branch_axis_gaussian_xla(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_moment_hint: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    defensive_nu: float | None = None,
) -> tuple[tf.Tensor, list[dict]]:
    """Run the production XLA value path without diagnostic capture."""

    value, diagnostics, _ = _run_value_filter_branch_axis_gaussian_xla(
        adapter,
        observations,
        config,
        predictive_moment_hint=predictive_moment_hint,
        initial_moment_hint=initial_moment_hint,
        defensive_nu=defensive_nu,
    )
    return value, diagnostics


def run_value_filter_branch_axis_gaussian_xla_diagnostic(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_moment_hint: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    capture_steps: Sequence[int],
    run_identity: str,
    defensive_nu: float | None = None,
) -> tuple[
    tf.Tensor,
    list[dict],
    dict[int, GaussianXLAFrozenTransitionSnapshot],
]:
    """Run the unchanged value path and capture selected transition states."""

    normalized_steps = frozenset(int(step) for step in capture_steps)
    if len(normalized_steps) != len(tuple(capture_steps)):
        raise ValueError("capture_steps must not contain duplicates")
    return _run_value_filter_branch_axis_gaussian_xla(
        adapter,
        observations,
        config,
        predictive_moment_hint=predictive_moment_hint,
        initial_moment_hint=initial_moment_hint,
        defensive_nu=defensive_nu,
        capture_steps=normalized_steps,
        run_identity=run_identity,
    )


def run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_moment_hint: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    capture_steps: Sequence[int],
    run_identity: str,
    defensive_nu: float | None = None,
) -> tuple[
    tf.Tensor,
    list[dict],
    dict[int, GaussianXLARetainedProposalSnapshot],
]:
    """Capture retained proposal state without changing the fitted graph.

    Each snapshot is assembled on the host from the seven outputs returned by
    the original production transition graph.  No additional graph output,
    refit, or derivative route is introduced.
    """

    normalized_steps = frozenset(int(step) for step in capture_steps)
    if len(normalized_steps) != len(tuple(capture_steps)):
        raise ValueError("capture_steps must not contain duplicates")
    return _run_value_filter_branch_axis_gaussian_xla(
        adapter,
        observations,
        config,
        predictive_moment_hint=predictive_moment_hint,
        initial_moment_hint=initial_moment_hint,
        defensive_nu=defensive_nu,
        capture_steps=normalized_steps,
        run_identity=run_identity,
        retained_proposal_capture=True,
    )


__all__ = [
    "GaussianXLAFrozenTransitionSnapshot",
    "GaussianXLARetainedProposalSnapshot",
    "evaluate_gaussian_xla_frozen_transition",
    "gaussian_xla_frozen_snapshot_fingerprint",
    "gaussian_xla_frozen_snapshot_from_parts",
    "gaussian_xla_frozen_snapshot_parts",
    "gaussian_xla_retained_proposal_snapshot_fingerprint",
    "gaussian_xla_retained_proposal_snapshot_from_parts",
    "gaussian_xla_retained_proposal_snapshot_parts",
    "run_value_filter_branch_axis_gaussian_xla",
    "run_value_filter_branch_axis_gaussian_xla_diagnostic",
    "run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic",
]
