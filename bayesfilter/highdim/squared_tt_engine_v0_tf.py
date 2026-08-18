"""Generic squared-TT filtering engine v0: value-only, quadratic-form retention.

P1B implementation of the generic program
(docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md).
density_kernel mode only. One filter step (UB-1 Section 1):

  (V1) reference-measure target on frozen scattered rows of the adjacent
       block (current axes first, previous axes last):
       log f_ref = log p_ret_ref(z_prev) + log p(x_c|x_p) + log p(y|x_c)
                   + log J_curr - log omega_curr
  (V2) sqrt target with max-shift s_t
  (V3) frozen-schedule ALS fit (FixedTTFitter systems + scaled augmented
       ridge solves; frozen rows, weights, ridge, ranks, sweep order/count)
  (V4) log Zhat_t = s_t + log(Z_h + tau)   [complete normalizer, uniform
       reference defensive density, tau per-scope declared]
  (V5) retention as RetainedQuadraticForm over the current axes (exact
       suffix-Gram marginalization; nothing densifies -> program veto V2)

No dense q^n object appears anywhere: rows are N scattered frozen points,
retention is the exact quadratic form. Ties in the max shift raise a status
flag (claim veto per UB-1 Sec. 5; value path proceeds for diagnostics).

Engineering-only caveats: this is the value path (no score), eager float64,
and the initial step fits p0(x_c) * p(y0|x_c) on the current block with a
trivial suffix axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import tensorflow as tf

from bayesfilter.highdim.bases import ProductBasis
from bayesfilter.highdim.diagnostics import MassMeasure, MeasureConvention
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.fitting import FixedTTFitConfig, FixedTTFitter, _solve_scaled_augmented_ridge
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    retained_quadratic_form_from_squared_tt,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64


@dataclass(frozen=True)
class DensityKernelAdapter:
    """Minimal density_kernel adapter: physical-coordinate log densities."""

    state_dim: int
    transition_log_density: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]  # (x_c, x_p) -> [N]
    observation_log_density: Callable[[tf.Tensor, tf.Tensor], tf.Tensor]  # (x_c, y) -> [N]
    initial_log_density: Callable[[tf.Tensor], tf.Tensor]  # (x_c,) -> [N]


@dataclass(frozen=True)
class EngineConfig:
    basis_degree: int
    rank: int
    row_count: int
    sweeps: int
    ridge: float
    tau: float
    coordinate_half_width: float
    seed: int
    condition_number_veto: float = 1e14
    quadrature_order: int | None = None  # tensor GL rows for small-n rungs


def _product_basis(dimension: int, degree: int) -> ProductBasis:
    import bayesfilter.highdim as highdim

    convention = MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    return ProductBasis(
        [
            highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree)
            for _ in range(dimension)
        ],
        convention,
    )


def _frozen_rows(count: int, dimension: int, seed: tuple[int, int]) -> tf.Tensor:
    return tf.random.stateless_uniform(
        [count, dimension], tf.constant(seed, tf.int32), minval=-1.0, maxval=1.0, dtype=DTYPE
    )


def _gauss_rows(dimension: int, order: int) -> tuple[tf.Tensor, tf.Tensor]:
    import numpy as _np

    nodes, weights = _np.polynomial.legendre.leggauss(order)
    mesh = _np.meshgrid(*([nodes] * dimension), indexing="ij")
    points = _np.stack([m.reshape(-1) for m in mesh], axis=1)
    weight_mesh = _np.meshgrid(*([weights / 2.0] * dimension), indexing="ij")
    row_weights = _np.prod(_np.stack([w.reshape(-1) for w in weight_mesh], axis=1), axis=1)
    return tf.constant(points, DTYPE), tf.constant(row_weights, DTYPE)


def _initial_tt_cores(dimension: int, basis_dim: int, rank: int) -> tuple[TTCore, ...]:
    cores = []
    for axis in range(dimension):
        left = 1 if axis == 0 else rank
        right = 1 if axis == dimension - 1 else rank
        values = tf.zeros([left, basis_dim, right], DTYPE)
        eye = tf.eye(left, right, batch_shape=[1], dtype=DTYPE)[0]
        values = tf.tensor_scatter_nd_update(
            values, [[l, 0, r] for l in range(left) for r in range(right)],
            tf.reshape(eye, [-1]),
        )
        cores.append(TTCore(values))
    return tuple(cores)


def _fixed_als_fit(
    product_basis: ProductBasis,
    points: tf.Tensor,
    sqrt_target: tf.Tensor,
    weights: tf.Tensor,
    cores: tuple[TTCore, ...],
    config: EngineConfig,
) -> tuple[tuple[TTCore, ...], Mapping[str, float]]:
    fitter = FixedTTFitter()
    fit_config = FixedTTFitConfig(
        ranks=tuple([1] + [config.rank] * (len(cores) - 1) + [1])[: len(cores) + 1],
        ridge=config.ridge,
        max_sweeps=config.sweeps,
        sweep_order=tuple(range(len(cores))),
        row_budget=int(points.shape[0]),
        column_budget=4096,
        dense_matrix_byte_budget=1 << 30,
        normal_matrix_byte_budget=1 << 30,
        condition_number_warning=1e12,
        condition_number_veto=config.condition_number_veto,
        holdout_tolerance=1e30,
    )
    worst_condition = 0.0
    current = cores
    for _sweep in range(config.sweeps):
        for core_index in range(len(current)):
            system = fitter.build_core_update_system(
                product_basis, points, sqrt_target, weights, current, core_index, fit_config
            )
            solve = _solve_scaled_augmented_ridge(
                design=system.design_matrix,
                target_values=sqrt_target,
                weights=weights,
                ridge=config.ridge,
            )
            condition = float(solve.scaled_augmented_condition_number)
            worst_condition = max(worst_condition, condition)
            if condition > config.condition_number_veto:
                raise ValueError("condition number veto in fixed ALS fit")
            updated = list(current)
            updated[core_index] = TTCore(
                tf.reshape(solve.solution, current[core_index].values.shape)
            )
            current = tuple(updated)
    residual = tf.linalg.matvec(
        fitter.build_core_update_system(
            product_basis, points, sqrt_target, weights, current, len(current) - 1, fit_config
        ).design_matrix,
        tf.reshape(current[-1].values, [-1]),
    ) - sqrt_target
    rms = float(
        tf.sqrt(tf.reduce_sum(weights * tf.square(residual)) / tf.reduce_sum(weights)).numpy()
    )
    return current, {"worst_condition": worst_condition, "weighted_fit_rms": rms}


@dataclass(frozen=True)
class EngineStepResult:
    log_increment: tf.Tensor
    retained: RetainedQuadraticForm
    diagnostics: Mapping[str, object]


def run_value_filter(
    adapter: DensityKernelAdapter,
    observations: tf.Tensor,
    config: EngineConfig,
) -> tuple[tf.Tensor, list[Mapping[str, object]]]:
    """Run the value-only squared-TT filter; return (log-lik, per-step diags)."""

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    joint_basis = _product_basis(2 * n, config.basis_degree)
    current_basis = _product_basis(n, config.basis_degree)
    basis_dim = int(joint_basis.bases[0].basis_dim)
    half = tf.constant(config.coordinate_half_width, DTYPE)
    current_map = AffineCoordinateMap(
        offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE) * half
    )
    # log J_curr (constant for affine box map) and log omega_curr = -n log 2.
    log_j_curr = tf.cast(n, DTYPE) * tf.math.log(half)
    log_omega_curr = -tf.cast(n, DTYPE) * tf.math.log(tf.constant(2.0, DTYPE))
    conversion = log_j_curr - log_omega_curr
    tau = tf.constant(config.tau, DTYPE)

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[Mapping[str, object]] = []

    for t in range(horizon):
        if t == 0:
            rows = _frozen_rows(config.row_count, n, (config.seed, 17))
            x_current = rows * half
            log_f = (
                adapter.initial_log_density(x_current)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            fit_basis, split = current_basis, None
        else:
            rows = _frozen_rows(config.row_count, 2 * n, (config.seed, 100 + t))
            x_current = rows[:, :n] * half
            z_previous = rows[:, n:]
            x_previous = z_previous * half
            log_f = (
                tf.math.log(retained.evaluate_reference_density(z_previous))
                + adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            fit_basis, split = joint_basis, n
        shift = tf.reduce_max(log_f)
        sorted_f = tf.sort(log_f, direction="DESCENDING")
        tie_flag = bool(((sorted_f[0] - sorted_f[1]) < 1e-12).numpy())
        sqrt_target = tf.exp(0.5 * (log_f - shift))
        weights = tf.fill([int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE))
        dimension = int(rows.shape[1])
        cores0 = _initial_tt_cores(dimension, basis_dim, config.rank)
        cores, fit_diag = _fixed_als_fit(fit_basis, rows, sqrt_target, weights, cores0, config)

        if split is None:
            # Initial step: append a trivial suffix axis so retention is uniform.
            suffix_core = tf.zeros([int(cores[-1].right_rank), basis_dim, 1], DTYPE)
            suffix_core = tf.tensor_scatter_nd_update(
                suffix_core, [[0, 0, 0]], [1.0]
            )
            extended = tuple(cores) + (TTCore(suffix_core),)
            extended_basis = _product_basis(n + 1, config.basis_degree)
            retained_new = retained_quadratic_form_from_squared_tt(
                extended, extended_basis, split_index=n, tau=float(config.tau),
                prefix_basis=current_basis, coordinate_map=current_map,
            )
        else:
            retained_new = retained_quadratic_form_from_squared_tt(
                cores, joint_basis, split_index=n, tau=float(config.tau),
                prefix_basis=current_basis, coordinate_map=current_map,
            )
        log_increment = shift + tf.math.log(retained_new.z_complete_ref)
        log_likelihood += log_increment
        retained = retained_new
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": float(log_increment.numpy()),
                "tie_flag": tie_flag,
                "suffix_gram_condition": float(
                    retained_new.suffix_gram_condition_estimate().numpy()
                ),
                **fit_diag,
            }
        )
    return log_likelihood, diagnostics


__all__ = ["DensityKernelAdapter", "EngineConfig", "run_value_filter"]


@dataclass(frozen=True)
class DiscreteIndicatorBasis1D:
    """Indicator basis on integer codes {0..B-1} with counting-measure mass.

    Branch axis for the branch-axis target assembly (design note
    bayesfilter-squared-tt-engine-branch-axis-design-2026-08-16.md).
    """

    cardinality: int

    @property
    def basis_dim(self) -> int:
        return int(self.cardinality)

    @property
    def dtype(self) -> tf.DType:
        return DTYPE

    def evaluate(self, points: tf.Tensor) -> tf.Tensor:
        codes = tf.cast(tf.round(tf.convert_to_tensor(points, DTYPE)), tf.int32)
        return tf.one_hot(codes, self.cardinality, dtype=DTYPE)

    def mass_matrix(self, measure) -> tf.Tensor:
        return tf.eye(self.cardinality, dtype=DTYPE)

    def integral_vector(self, measure) -> tf.Tensor:
        return tf.ones([self.cardinality], DTYPE)

    def manifest_payload(self):
        return {"family": "DiscreteIndicatorBasis1D", "cardinality": self.cardinality}


def run_value_filter_branch_axis(
    adapter: DensityKernelAdapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    gram_condition_veto: float = 1e12,
) -> tuple[tf.Tensor, list[Mapping[str, object]]]:
    """Branch-axis value filter: smooth branch targets, no |h| kinks.

    Axis order per step (t>=1): (z_curr axes..., branch axis g, z_prev
    axes...); retention splits after the z_curr block, so g and z_prev are
    integrated/summed out and the retained object stays a standard
    RetainedQuadraticForm over the current state.
    """

    import bayesfilter.highdim as highdim

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)
    half = tf.constant(config.coordinate_half_width, DTYPE)
    current_map = AffineCoordinateMap(
        offset=tf.zeros([n], DTYPE), matrix=tf.eye(n, dtype=DTYPE) * half
    )
    log_j_curr = tf.cast(n, DTYPE) * tf.math.log(half)
    log_omega_curr = -tf.cast(n, DTYPE) * tf.math.log(tf.constant(2.0, DTYPE))
    conversion = log_j_curr - log_omega_curr
    tau = tf.constant(config.tau, DTYPE)

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[Mapping[str, object]] = []

    for t in range(horizon):
        if t == 0:
            if config.quadrature_order is not None:
                rows, weights = _gauss_rows(n, config.quadrature_order)
            else:
                rows = _frozen_rows(config.row_count, n, (config.seed, 17))
                weights = tf.fill(
                    [int(rows.shape[0])], tf.constant(1.0 / int(rows.shape[0]), DTYPE)
                )
            x_current = rows * half
            log_f = (
                adapter.initial_log_density(x_current)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            shift = tf.reduce_max(log_f)
            sqrt_target = tf.exp(0.5 * (log_f - shift))
            cores0 = _initial_tt_cores(n, basis_dim, config.rank)
            cores, fit_diag = _fixed_als_fit(current_basis, rows, sqrt_target, weights, cores0, config)
            suffix_core = tf.zeros([int(cores[-1].right_rank), basis_dim, 1], DTYPE)
            suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
            extended = tuple(cores) + (TTCore(suffix_core),)
            extended_basis = _product_basis(n + 1, config.basis_degree)
            retained_new = retained_quadratic_form_from_squared_tt(
                extended, extended_basis, split_index=n, tau=float(config.tau),
                prefix_basis=current_basis, coordinate_map=current_map,
            )
            log_increment = shift + tf.math.log(retained_new.z_complete_ref)
            tie_flag = False
        else:
            gram = retained.suffix_gram
            eigenvalues, eigenvectors = tf.linalg.eigh(gram)
            gram_condition = float(
                (eigenvalues[-1] / tf.maximum(eigenvalues[0], tf.constant(1e-300, DTYPE))).numpy()
            )
            # Near-singular E is expected for near-Gaussian retained laws and
            # is benign for the VALUE path (any factor L with L L' = E gives
            # the exact branch sum). The smoothness guard (Cholesky + hard
            # conditioning veto) is a SCORE-path claim gate, applied at P2.
            chol = eigenvectors * tf.sqrt(tf.maximum(eigenvalues, 0.0))[None, :]
            branch_count = retained.boundary_rank + 1  # + tau branch
            if config.quadrature_order is not None:
                z_rows, z_weights = _gauss_rows(2 * n, config.quadrature_order)
            else:
                z_rows = _frozen_rows(config.row_count, 2 * n, (config.seed, 100 + t))
                z_weights = tf.fill(
                    [int(z_rows.shape[0])],
                    tf.constant(1.0 / int(z_rows.shape[0]), DTYPE),
                )
            x_current = z_rows[:, :n] * half
            z_previous = z_rows[:, n:]
            x_previous = z_previous * half
            log_g_kernel = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            v_prev = tf.einsum(
                "na,ab->nb",
                _prefix_rows_for(retained, z_previous),
                chol,
            )  # [N, r_c] signed branch amplitudes u_g
            sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + tau
            log_f_row = tf.math.log(sum_sq) + log_g_kernel  # = log f + log Zc_prev
            shift = tf.reduce_max(log_f_row)
            sorted_f = tf.sort(log_f_row, direction="DESCENDING")
            tie_flag = bool(((sorted_f[0] - sorted_f[1]) < 1e-12).numpy())
            sqrt_g_shifted = tf.exp(0.5 * (log_g_kernel - shift))
            amplitudes = tf.concat(
                [v_prev, tf.fill([int(z_rows.shape[0]), 1], tf.sqrt(tau))], axis=1
            )  # [N, B]
            targets = amplitudes * sqrt_g_shifted[:, None]  # [N, B] smooth signed
            n_rows_total = int(z_rows.shape[0]) * branch_count
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
            # mu (x) counting inner product: each branch row carries the z-row
            # mu-weight (counting measure sums branches with weight 1).
            weights = tf.reshape(tf.repeat(z_weights, branch_count, axis=0), [-1])
            convention = current_basis.convention
            mixed_basis = ProductBasis(
                list(current_basis.bases)
                + [DiscreteIndicatorBasis1D(branch_count)]
                + list(_product_basis(n, config.basis_degree).bases),
                convention,
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
            retained_new = retained_quadratic_form_from_squared_tt(
                cores, mixed_basis, split_index=n, tau=float(config.tau),
                prefix_basis=current_basis, coordinate_map=current_map,
            )
            # increment: log int f = shift + log Z_h_new - log Zc_prev; the
            # engine's declared increment additionally carries the new tau
            # mass inside z_complete_ref, matching the v0 program semantics.
            log_increment = (
                shift
                + tf.math.log(retained_new.z_complete_ref)
                - tf.math.log(retained.z_complete_ref)
            )
            fit_diag = {**fit_diag, "gram_condition": gram_condition}
        log_likelihood += log_increment
        retained = retained_new
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": float(log_increment.numpy()),
                "tie_flag": tie_flag,
                **fit_diag,
            }
        )
    return log_likelihood, diagnostics


def _prefix_rows_for(retained: RetainedQuadraticForm, points: tf.Tensor) -> tf.Tensor:
    from bayesfilter.highdim.retained_quadratic_form_tf import prefix_row_vectors

    return prefix_row_vectors(retained.prefix_cores, retained.prefix_basis, points)
