"""Consolidated three-route actual-SV simulation benchmark.

On common simulated exact actual-SV paths, this benchmark compares, in one
artifact with strictly separated sections:

1. the fixed-variant actual-SV batch TT route (scalar, fixed sigma=1,
   UKF-center-frozen cores rebuilt on the simulated dataset) against the
   exact-transformed dense same-target reference;
2. the exact-transformed Zhao-Cui factorized TT route against its own dense
   same-target reference;
3. the KSC-surrogate Zhao-Cui factorized TT route against its own dense KSC
   mixture reference;
4. the dense Gaussian-mixture / Kalman approximation under a fitted
   7 / 14 / 28-component refinement ladder with the 1% stabilization screen.

Cross-family comparisons are reported only in the raw-`y` representation
after exact transformation/Jacobian correction and are descriptive only.

Plan: docs/plans/bayesfilter-actual-sv-three-route-simulation-benchmark-plan-2026-08-13.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
import tensorflow_probability as tfp

import bayesfilter.highdim as highdim

_STD_NORMAL = tfp.distributions.Normal(
    loc=tf.constant(0.0, dtype=tf.float64),
    scale=tf.constant(1.0, dtype=tf.float64),
)

PLAN = (
    "docs/plans/"
    "bayesfilter-actual-sv-three-route-simulation-benchmark-plan-2026-08-13.md"
)

NONCLAIMS = (
    "single simulated path per dimension: no statistical ranking of approximation families",
    "same-target gaps are internal-consistency veto diagnostics only",
    "cross-family raw-y gaps are descriptive approximation-family differences, not correctness scores",
    "the 1% refinement rule is an empirical stabilization screen, not a convergence proof",
    "not a production timing benchmark and not HMC convergence evidence",
    "KSC-surrogate rows target a different (offset log-square KSC mixture) density than exact rows",
    "batch TT route is scalar fixed-sigma=1 and reported for dim 1 only",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dims", default="1,2,3")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=83120)
    parser.add_argument("--dense-order", type=int, default=401)
    parser.add_argument("--dense-radius", type=float, default=8.0)
    parser.add_argument("--ksc-transform-offset", type=float, default=1e-8)
    parser.add_argument("--fd-step", type=float, default=1e-5)
    parser.add_argument("--mixture-components", default="7,14,28")
    parser.add_argument("--skip-batch-tt", action="store_true")
    parser.add_argument("--skip-scores", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args()


def _parse_int_list(value: str, *, name: str) -> list[int]:
    items = [int(part) for part in value.split(",") if part.strip()]
    if not items:
        raise ValueError(f"{name} must not be empty")
    return items


# ---------------------------------------------------------------------------
# Simulated data and parameters
# ---------------------------------------------------------------------------


def _observations(dim: int, *, seed_base: int, horizon: int) -> tf.Tensor:
    """Simulate independent exact actual-SV coordinate paths (gamma=.6, beta=.4)."""

    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
    )

    columns = []
    for axis in range(int(dim)):
        _states, observations = generate_frozen_exact_sv_dataset_tf(
            seed=seed_base + dim + 1000 * axis, horizon=horizon
        )
        columns.append(observations[:, 0])
    return tf.stack(columns, axis=1)


def _physical_parameters(dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    gamma = tf.constant([0.60, 0.52, 0.47], dtype=tf.float64)[: int(dim)]
    beta = tf.constant([0.40, 0.35, 0.45], dtype=tf.float64)[: int(dim)]
    sigma = tf.constant([1.00, 0.85, 0.75], dtype=tf.float64)[: int(dim)]
    return gamma, beta, sigma


def _theta_from_physical(gamma: tf.Tensor, beta: tf.Tensor) -> tf.Tensor:
    return tf.reshape(
        tf.stack([_STD_NORMAL.quantile(gamma), tf.math.log(beta)], axis=1), [-1]
    )


def _physical_from_theta(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    theta_matrix = tf.reshape(tf.convert_to_tensor(theta, dtype=tf.float64), [-1, 2])
    return _STD_NORMAL.cdf(theta_matrix[:, 0]), tf.exp(theta_matrix[:, 1])


# ---------------------------------------------------------------------------
# Jacobian corrections into raw-y representation
# ---------------------------------------------------------------------------


def _offset_log_square_jacobian_log_abs_det(
    observations: tf.Tensor, offset: float
) -> tf.Tensor:
    """Return the additive raw-`y` correction for z = log(y^2 + offset).

    The map y -> z is two-to-one and sign-symmetric, so
    ``p_z(z) = 2 p_y(y) |dy/dz|`` with ``|dz/dy| = 2|y|/(y^2+offset)``, giving
    ``log p_y = log p_z + sum[log|y| - log(y^2+offset)]``.  At offset=0 this
    reduces to the repository convention ``log p_y = log p_z - sum log|y|``
    (see ``exact_transformed_sv_jacobian_log_abs_det`` and the p41 raw-native
    equality test).
    """

    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    return tf.reduce_sum(
        tf.math.log(tf.abs(y)) - tf.math.log(tf.square(y) + tf.constant(offset, tf.float64))
    )


def _exact_log_square_jacobian_log_abs_det(observations: tf.Tensor) -> tf.Tensor:
    """Additive raw-`y` correction for z = log(y^2): ``- sum log|y|``."""

    y = tf.convert_to_tensor(observations, dtype=tf.float64)
    return -tf.reduce_sum(tf.math.log(tf.abs(y)))


# ---------------------------------------------------------------------------
# Fitted log-chi-square Gaussian mixtures (7 / 14 / 28)
# ---------------------------------------------------------------------------


def _split_mixture(
    weights: tf.Tensor, means: tf.Tensor, variances: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Deterministically split each component into two along its mean axis."""

    offset = 0.5 * tf.sqrt(variances)
    new_weights = tf.concat([0.5 * weights, 0.5 * weights], axis=0)
    new_means = tf.concat([means - offset, means + offset], axis=0)
    new_variances = tf.concat([0.75 * variances, 0.75 * variances], axis=0)
    return new_weights, new_means, new_variances


def _fit_log_chi_square_mixture(
    component_count: int,
    *,
    grid_order: int = 4001,
    grid_left: float = -30.0,
    grid_right: float = 8.0,
    em_iterations: int = 400,
) -> tuple[highdim.SVLogChiSquareGaussianMixture, dict[str, Any]]:
    """Fit a K-component Gaussian mixture to the exact log(chi^2_1) density.

    Deterministic quadrature-weighted EM on a fixed trapezoid grid, initialized
    from the pinned KSC 1998 7-component mixture (split deterministically for
    K=14 and K=28).  float64 TensorFlow throughout.
    """

    base = highdim.ksc_1998_log_chi_square_mixture()
    weights = tf.identity(base.weights)
    means = tf.identity(base.means)
    variances = tf.identity(base.variances)
    while int(weights.shape[0]) < component_count:
        weights, means, variances = _split_mixture(weights, means, variances)
    if int(weights.shape[0]) != component_count:
        raise ValueError(
            f"component count {component_count} is not 7*2^k; got {int(weights.shape[0])}"
        )

    grid = tf.linspace(
        tf.constant(grid_left, tf.float64), tf.constant(grid_right, tf.float64), grid_order
    )
    step = (grid_right - grid_left) / (grid_order - 1)
    trapezoid = tf.concat(
        [
            tf.constant([0.5], tf.float64),
            tf.ones([grid_order - 2], tf.float64),
            tf.constant([0.5], tf.float64),
        ],
        axis=0,
    ) * tf.constant(step, tf.float64)
    target_log_density = highdim.exact_log_chi_square_log_density(grid)
    target_density = tf.exp(target_log_density)
    target_mass = trapezoid * target_density

    def _component_log_probs(m: tf.Tensor, v: tf.Tensor) -> tf.Tensor:
        return -0.5 * (
            tf.math.log(2.0 * tf.constant(3.141592653589793, tf.float64) * v)[:, None]
            + tf.square(grid[None, :] - m[:, None]) / v[:, None]
        )

    for _ in range(em_iterations):
        log_probs = _component_log_probs(means, variances)
        log_joint = tf.math.log(weights)[:, None] + log_probs
        log_mix = tf.reduce_logsumexp(log_joint, axis=0)
        responsibilities = tf.exp(log_joint - log_mix[None, :])
        weighted = responsibilities * target_mass[None, :]
        component_mass = tf.reduce_sum(weighted, axis=1)
        weights = component_mass / tf.reduce_sum(component_mass)
        means = tf.reduce_sum(weighted * grid[None, :], axis=1) / component_mass
        centered_sq = tf.square(grid[None, :] - means[:, None])
        variances = tf.maximum(
            tf.reduce_sum(weighted * centered_sq, axis=1) / component_mass,
            tf.constant(1e-6, tf.float64),
        )

    log_probs = _component_log_probs(means, variances)
    fitted_log_density = tf.reduce_logsumexp(
        tf.math.log(weights)[:, None] + log_probs, axis=0
    )
    l1_error = float(
        tf.reduce_sum(
            trapezoid * tf.abs(tf.exp(fitted_log_density) - target_density)
        ).numpy()
    )
    weights = weights / tf.reduce_sum(weights)
    mixture = highdim.SVLogChiSquareGaussianMixture(
        weights=weights,
        means=means,
        variances=variances,
        source=(
            f"fitted_em_k{component_count}_log_chi_square_grid{grid_order}"
            f"_[{grid_left},{grid_right}]_iters{em_iterations}_ksc7_split_init"
        ),
    )
    fit_diagnostics = {
        "component_count": component_count,
        "weighted_l1_density_error": l1_error,
        "em_iterations": em_iterations,
        "grid_order": grid_order,
        "grid_interval": [grid_left, grid_right],
        "initialization": "ksc_1998_pinned_7_component_deterministic_split",
    }
    return mixture, fit_diagnostics


# ---------------------------------------------------------------------------
# Coordinate-wise mixture Kalman (factorized panel evaluation)
# ---------------------------------------------------------------------------


def _panel_mixture_kalman_log_likelihood(
    observations: tf.Tensor,
    *,
    gamma: tf.Tensor,
    beta: tf.Tensor,
    sigma: tf.Tensor,
    mixture: highdim.SVLogChiSquareGaussianMixture,
    transform_offset: float,
) -> tf.Tensor:
    """Sum of per-coordinate scalar mixture-Kalman log-likelihoods.

    Valid because the independent panel target factorizes exactly across
    coordinates; verified against the joint enumeration in
    ``_factorization_check``.
    """

    dim = int(observations.shape[1])
    total = tf.constant(0.0, tf.float64)
    for axis in range(dim):
        result = highdim.independent_panel_sv_mixture_kalman_filter(
            observations[:, axis : axis + 1],
            gamma=gamma[axis],
            beta=beta[axis],
            sigma=sigma[axis],
            mixture=mixture,
            transform_offset=transform_offset,
        )
        total = total + result.log_likelihood
    return total


def _factorization_check(args: argparse.Namespace) -> dict[str, Any]:
    """Verify the coordinate-wise factorization against joint enumeration."""

    observations = _observations(2, seed_base=args.seed_base, horizon=min(6, args.horizon))
    gamma, beta, sigma = _physical_parameters(2)
    mixture = highdim.ksc_1998_log_chi_square_mixture()
    joint = highdim.independent_panel_sv_mixture_kalman_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        mixture=mixture,
        transform_offset=args.ksc_transform_offset,
    ).log_likelihood
    factorized = _panel_mixture_kalman_log_likelihood(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        mixture=mixture,
        transform_offset=args.ksc_transform_offset,
    )
    gap = float(abs((joint - factorized).numpy()))
    if gap > 1e-8:
        raise AssertionError(
            f"panel mixture-Kalman factorization check failed: gap={gap}"
        )
    return {
        "dim": 2,
        "horizon": int(observations.shape[0]),
        "component_count": mixture.component_count,
        "joint_log_likelihood": float(joint.numpy()),
        "factorized_log_likelihood": float(factorized.numpy()),
        "abs_gap": gap,
        "tolerance": 1e-8,
        "status": "passed",
    }


# ---------------------------------------------------------------------------
# TT configuration (matches reviewed p41/p47 fixed-design configs)
# ---------------------------------------------------------------------------


def _convention() -> highdim.MeasureConvention:
    return highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _tt_config(seed: str) -> highdim.FixedBranchFilterConfig:
    convention = _convention()
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 48)],
        convention,
    )
    return highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=2,
            sweep_order=(0,),
            row_budget=512,
            column_budget=128,
            dense_matrix_byte_budget=200_000,
            normal_matrix_byte_budget=100_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], dtype=tf.float64),
                matrix=tf.constant([[8.0]], dtype=tf.float64),
            ),
        ),
        measure_convention=convention,
        deterministic_seed=seed,
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(
                tf.ones([1, product_basis.bases[0].basis_dim, 1], dtype=tf.float64)
            ),
        ),
        fit_quadrature_order=141,
    )


def _centered_finite_difference_score(value_fn, theta: tf.Tensor, step: float) -> tf.Tensor:
    theta_tensor = tf.convert_to_tensor(theta, dtype=tf.float64)
    score = []
    for axis in range(int(theta_tensor.shape[0])):
        direction = tf.one_hot(
            axis, int(theta_tensor.shape[0]), dtype=tf.float64
        ) * tf.constant(step, dtype=tf.float64)
        plus = value_fn(theta_tensor + direction)
        minus = value_fn(theta_tensor - direction)
        score.append((plus - minus) / (2.0 * tf.constant(step, dtype=tf.float64)))
    return tf.stack(score)


def _relative_error(candidate: tf.Tensor, reference: tf.Tensor) -> float:
    value = tf.linalg.norm(candidate - reference) / tf.maximum(
        tf.constant(1.0, dtype=tf.float64),
        tf.linalg.norm(reference),
    )
    return float(value.numpy())


# ---------------------------------------------------------------------------
# Route 1: fixed-variant actual-SV batch TT (dim 1, fixed sigma=1)
# ---------------------------------------------------------------------------


def _collect_batch_tt_row(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate the batched fixed-adjacent TT likelihood on the simulated path.

    Cores are UKF-center-frozen at the truth parameters on the simulated
    dataset (same initializer rule as the reviewed SVX adapter); the same-target
    reference is the exact-transformed dense reference with fixed sigma=1.
    """

    import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator
    from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
        COORDINATE_HALF_WIDTH,
        DEGREE,
        ORDER,
        RANK,
        batched_fixed_tt_likelihood_value_trace,
    )
    from bayesfilter.highdim.filtering import legendre_gauss_nodes_weights

    observations = _observations(1, seed_base=args.seed_base, horizon=args.horizon)
    gamma_panel, beta_panel, _sigma_panel = _physical_parameters(1)
    sigma = tf.ones([1], tf.float64)  # batch TT route is fixed sigma=1
    model = highdim.ExactTransformedSVSSM(sigma=1.0)
    theta_scalar = model.unconstrained_from_physical(
        gamma=gamma_panel[0], beta=beta_panel[0]
    )
    transformed = tf.math.log(tf.square(observations[:, 0]))

    initial, adjacent, initializer = comparator._ukf_initial_cores(
        model=model,
        theta=theta_scalar,
        raw_observations=observations,
        degree=DEGREE,
        order=ORDER,
        rank=RANK,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
    )
    config = comparator._comparator_config(
        degree=DEGREE,
        order=ORDER,
        rank=RANK,
        seed="three-route-sim-batch-tt-center-frozen-v1",
        transition_before_first_observation=False,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
        density_tau=0.0,
        initial_cores=initial,
        adjacent_initial_cores=adjacent,
        initialization_rule=str(initializer["initializer_rule"]),
    )
    nodes, weights = legendre_gauss_nodes_weights(ORDER)
    mesh = tf.meshgrid(nodes, nodes, indexing="ij")
    grid = tf.reshape(tf.stack(mesh, axis=-1), (-1, 2))
    weight_mesh = tf.meshgrid(0.5 * weights, 0.5 * weights, indexing="ij")
    grid_weights = tf.reshape(weight_mesh[0] * weight_mesh[1], (-1,))
    program_tensors = {
        "transformed_observations": transformed,
        "initial_core": initial[0].values,
        "adjacent_core0": adjacent[0].values,
        "adjacent_core1": adjacent[1].values,
        "reference_nodes": nodes,
        "reference_weights": 0.5 * weights,
        "reference_grid": grid,
        "reference_grid_weights": grid_weights,
        "basis_nodes": config.initial.product_basis.evaluate_axis(0, nodes),
        "basis_grid_axis0": config.adjacent.product_basis.evaluate_axis(0, grid[:, 0]),
        "basis_grid_axis1": config.adjacent.product_basis.evaluate_axis(1, grid[:, 1]),
    }

    # Batch of two rows: truth and a nearby offset, exercising batch nativeness.
    theta_batch = tf.stack(
        (theta_scalar, theta_scalar + tf.constant([0.02, 0.03], tf.float64)), axis=0
    )
    trace = batched_fixed_tt_likelihood_value_trace(theta_batch, **program_tensors)
    tt_value_truth = trace.value[0]
    status = {
        key: value.numpy().tolist() for key, value in trace.status.items()
    }

    dense = highdim.exact_transformed_sv_independent_panel_dense_reference(
        observations,
        gamma=gamma_panel[:1],
        beta=beta_panel[:1],
        sigma=sigma,
        order=args.dense_order,
        radius=args.dense_radius,
    )
    return {
        "dim": 1,
        "fixed_sigma": 1.0,
        "horizon": int(observations.shape[0]),
        "theta_truth": [float(v) for v in theta_scalar.numpy()],
        "batch_size": 2,
        "tt_log_likelihood": float(tt_value_truth.numpy()),
        "dense_same_target_log_likelihood": float(dense.log_likelihood.numpy()),
        "same_target_value_gap": float(
            abs((tt_value_truth - dense.log_likelihood).numpy())
        ),
        "initializer_rule": str(initializer["initializer_rule"]),
        "status": status,
        "core_provenance": "ukf_center_frozen_truth_theta_simulated_dataset",
        "nonclaim": (
            "likelihood-only same-target check at truth theta; "
            "not the frozen T10 seed-81101 adapter dataset"
        ),
    }


# ---------------------------------------------------------------------------
# Route 2: exact-transformed Zhao-Cui vs own dense reference
# ---------------------------------------------------------------------------


def _collect_exact_transformed_rows(
    args: argparse.Namespace, dims: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim in dims:
        observations = _observations(dim, seed_base=args.seed_base, horizon=args.horizon)
        gamma, beta, sigma = _physical_parameters(dim)
        theta = _theta_from_physical(gamma, beta)

        dense = highdim.exact_transformed_sv_independent_panel_dense_reference(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            order=args.dense_order,
            radius=args.dense_radius,
        )
        tt = highdim.exact_transformed_sv_independent_panel_zhaocui_tt_filter(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            config=_tt_config(seed=f"three-route-exact-tt-dim-{dim}"),
            branch_seed_prefix=f"three-route-exact-tt-dim-{dim}",
        )
        row: dict[str, Any] = {
            "dim": dim,
            "dense_log_likelihood": float(dense.log_likelihood.numpy()),
            "zhaocui_tt_log_likelihood": float(tt.log_likelihood.numpy()),
            "same_target_value_gap": float(
                abs((tt.log_likelihood - dense.log_likelihood).numpy())
            ),
            "raw_y_log_likelihood_dense": float(
                (
                    dense.log_likelihood
                    + _exact_log_square_jacobian_log_abs_det(observations)
                ).numpy()
            ),
            "transform_offset": 0.0,
        }
        if not args.skip_scores and dim == 1:
            derivative_config = highdim.FixedBranchDerivativeConfig(
                parameter_indices=(0, 1)
            )
            score = highdim.exact_transformed_sv_independent_panel_zhaocui_tt_score(
                observations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                config=_tt_config(seed=f"three-route-exact-tt-score-dim-{dim}"),
                derivative_config=derivative_config,
                branch_seed_prefix=f"three-route-exact-tt-score-dim-{dim}",
            )

            def _value(current_theta: tf.Tensor) -> tf.Tensor:
                current_gamma, current_beta = _physical_from_theta(current_theta)
                return highdim.exact_transformed_sv_independent_panel_zhaocui_tt_filter(
                    observations,
                    gamma=current_gamma,
                    beta=current_beta,
                    sigma=sigma,
                    config=_tt_config(seed=f"three-route-exact-tt-score-dim-{dim}"),
                    branch_seed_prefix=f"three-route-exact-tt-score-dim-{dim}",
                ).log_likelihood

            finite = _centered_finite_difference_score(_value, theta, args.fd_step)
            row["score_relative_error_to_fd"] = _relative_error(score.score, finite)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Route 3: KSC-surrogate Zhao-Cui vs own dense KSC reference
# ---------------------------------------------------------------------------


def _collect_ksc_surrogate_rows(
    args: argparse.Namespace, dims: list[int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mixture = highdim.ksc_1998_log_chi_square_mixture()
    for dim in dims:
        observations = _observations(dim, seed_base=args.seed_base, horizon=args.horizon)
        gamma, beta, sigma = _physical_parameters(dim)
        theta = _theta_from_physical(gamma, beta)

        # Dense same-target KSC reference, coordinate-wise (factorized target).
        dense_total = tf.constant(0.0, tf.float64)
        for axis in range(dim):
            model = highdim.StochasticVolatilitySSM(sigma=sigma[axis])
            theta_axis = model.unconstrained_from_physical(
                gamma=gamma[axis], beta=beta[axis]
            )
            dense_total = dense_total + highdim.scalar_sv_mixture_dense_reference(
                model,
                theta_axis,
                observations[:, axis : axis + 1],
                mixture=mixture,
                order=args.dense_order,
                radius=args.dense_radius,
                transform_offset=args.ksc_transform_offset,
            ).log_likelihood

        tt = highdim.independent_panel_sv_mixture_zhaocui_tt_filter(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            config=_tt_config(seed=f"three-route-ksc-tt-dim-{dim}"),
            mixture=mixture,
            transform_offset=args.ksc_transform_offset,
            branch_seed_prefix=f"three-route-ksc-tt-dim-{dim}",
        )
        kalman = _panel_mixture_kalman_log_likelihood(
            observations,
            gamma=gamma,
            beta=beta,
            sigma=sigma,
            mixture=mixture,
            transform_offset=args.ksc_transform_offset,
        )
        row: dict[str, Any] = {
            "dim": dim,
            "dense_ksc_log_likelihood": float(dense_total.numpy()),
            "zhaocui_tt_log_likelihood": float(tt.log_likelihood.numpy()),
            "kalman_mixture_log_likelihood": float(kalman.numpy()),
            "same_target_value_gap_tt_vs_dense": float(
                abs((tt.log_likelihood - dense_total).numpy())
            ),
            "same_target_value_gap_kalman_vs_dense": float(
                abs((kalman - dense_total).numpy())
            ),
            "raw_y_log_likelihood_dense": float(
                (
                    dense_total
                    + _offset_log_square_jacobian_log_abs_det(
                        observations, args.ksc_transform_offset
                    )
                ).numpy()
            ),
            "transform_offset": args.ksc_transform_offset,
            "mixture_source": mixture.source,
            "nonclaim": "KSC-mixture surrogate target, not the exact actual-SV target",
        }
        if not args.skip_scores and dim == 1:
            derivative_config = highdim.FixedBranchDerivativeConfig(
                parameter_indices=(0, 1)
            )
            score = highdim.independent_panel_sv_mixture_zhaocui_tt_score(
                observations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                config=_tt_config(seed=f"three-route-ksc-tt-score-dim-{dim}"),
                derivative_config=derivative_config,
                mixture=mixture,
                transform_offset=args.ksc_transform_offset,
                branch_seed_prefix=f"three-route-ksc-tt-score-dim-{dim}",
            )

            def _value(current_theta: tf.Tensor) -> tf.Tensor:
                current_gamma, current_beta = _physical_from_theta(current_theta)
                return highdim.independent_panel_sv_mixture_zhaocui_tt_filter(
                    observations,
                    gamma=current_gamma,
                    beta=current_beta,
                    sigma=sigma,
                    config=_tt_config(seed=f"three-route-ksc-tt-score-dim-{dim}"),
                    mixture=mixture,
                    transform_offset=args.ksc_transform_offset,
                    branch_seed_prefix=f"three-route-ksc-tt-score-dim-{dim}",
                ).log_likelihood

            finite = _centered_finite_difference_score(_value, theta, args.fd_step)
            row["score_relative_error_to_fd"] = _relative_error(score.score, finite)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Route 4: dense Kalman refinement ladder with fitted mixtures
# ---------------------------------------------------------------------------


def _collect_refinement_rows(
    args: argparse.Namespace,
    dims: list[int],
    component_counts: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mixtures: list[tuple[int, highdim.SVLogChiSquareGaussianMixture]] = []
    fit_rows: list[dict[str, Any]] = []
    for count in component_counts:
        mixture, fit_diagnostics = _fit_log_chi_square_mixture(count)
        mixtures.append((count, mixture))
        fit_rows.append(
            {**fit_diagnostics, "mixture_manifest": dict(mixture.manifest_payload())}
        )

    rows: list[dict[str, Any]] = []
    for dim in dims:
        observations = _observations(dim, seed_base=args.seed_base, horizon=args.horizon)
        gamma, beta, sigma = _physical_parameters(dim)
        ladder: list[dict[str, Any]] = []
        for count, mixture in mixtures:
            value = _panel_mixture_kalman_log_likelihood(
                observations,
                gamma=gamma,
                beta=beta,
                sigma=sigma,
                mixture=mixture,
                transform_offset=args.ksc_transform_offset,
            )
            ladder.append(
                {
                    "component_count": count,
                    "kalman_log_likelihood": float(value.numpy()),
                    "raw_y_log_likelihood": float(
                        (
                            value
                            + _offset_log_square_jacobian_log_abs_det(
                                observations, args.ksc_transform_offset
                            )
                        ).numpy()
                    ),
                }
            )
        changes = []
        for previous, current in zip(ladder, ladder[1:]):
            denominator = max(abs(previous["kalman_log_likelihood"]), 1.0)
            changes.append(
                {
                    "from_components": previous["component_count"],
                    "to_components": current["component_count"],
                    "relative_value_change": abs(
                        current["kalman_log_likelihood"]
                        - previous["kalman_log_likelihood"]
                    )
                    / denominator,
                }
            )
        stabilized = all(change["relative_value_change"] < 0.01 for change in changes)
        rows.append(
            {
                "dim": dim,
                "ladder": ladder,
                "refinement_changes": changes,
                "stabilized_under_1pct_rule": stabilized,
                "transform_offset": args.ksc_transform_offset,
            }
        )
    return rows, fit_rows


# ---------------------------------------------------------------------------
# Cross-family raw-y consolidation (descriptive only)
# ---------------------------------------------------------------------------


def _cross_family_rows(
    args: argparse.Namespace,
    dims: list[int],
    exact_rows: list[dict[str, Any]],
    ksc_rows: list[dict[str, Any]],
    refinement_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for dim in dims:
        exact = next(row for row in exact_rows if row["dim"] == dim)
        ksc = next(row for row in ksc_rows if row["dim"] == dim)
        refinement = next(row for row in refinement_rows if row["dim"] == dim)
        finest = refinement["ladder"][-1]
        rows.append(
            {
                "dim": dim,
                "raw_y_exact_dense": exact["raw_y_log_likelihood_dense"],
                "raw_y_ksc_dense": ksc["raw_y_log_likelihood_dense"],
                "raw_y_kalman_finest": finest["raw_y_log_likelihood"],
                "ksc_dense_minus_exact_dense": ksc["raw_y_log_likelihood_dense"]
                - exact["raw_y_log_likelihood_dense"],
                "kalman_finest_minus_exact_dense": finest["raw_y_log_likelihood"]
                - exact["raw_y_log_likelihood_dense"],
                "nonclaim": "descriptive approximation-family differences only",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Actual-SV Three-Route Simulation Benchmark",
        "",
        f"- JSON artifact: `{json_path}`",
        f"- Plan: `{result['plan']}`",
        f"- Dims: `{result['dims']}`, horizon `{result['horizon']}`",
        f"- Git commit: `{result['git_commit']}`",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    if result["batch_tt_row"] is not None:
        row = result["batch_tt_row"]
        lines.extend(
            [
                "",
                "## Route 1: fixed-variant actual-SV batch TT (dim 1)",
                "",
                f"- TT log-likelihood: {row['tt_log_likelihood']:.10g}",
                f"- Dense same-target: {row['dense_same_target_log_likelihood']:.10g}",
                f"- Same-target gap: {row['same_target_value_gap']:.6g}",
            ]
        )
    lines.extend(["", "## Route 2: exact-transformed Zhao-Cui", ""])
    for row in result["exact_transformed_rows"]:
        score_part = (
            f", score relerr={row['score_relative_error_to_fd']:.3g}"
            if "score_relative_error_to_fd" in row
            else ""
        )
        lines.append(
            f"- dim {row['dim']}: same-target gap={row['same_target_value_gap']:.6g}"
            + score_part
        )
    lines.extend(["", "## Route 3: KSC-surrogate Zhao-Cui", ""])
    for row in result["ksc_surrogate_rows"]:
        score_part = (
            f", score relerr={row['score_relative_error_to_fd']:.3g}"
            if "score_relative_error_to_fd" in row
            else ""
        )
        lines.append(
            f"- dim {row['dim']}: TT-vs-dense gap={row['same_target_value_gap_tt_vs_dense']:.6g}, "
            f"Kalman-vs-dense gap={row['same_target_value_gap_kalman_vs_dense']:.6g}"
            + score_part
        )
    lines.extend(["", "## Route 4: dense Kalman refinement ladder", ""])
    for row in result["refinement_rows"]:
        parts = ", ".join(
            f"K={step['component_count']}: {step['kalman_log_likelihood']:.8g}"
            for step in row["ladder"]
        )
        lines.append(
            f"- dim {row['dim']}: {parts} | stabilized(<1%)={row['stabilized_under_1pct_rule']}"
        )
    lines.extend(["", "## Cross-family raw-y gaps (descriptive only)", ""])
    for row in result["cross_family_rows"]:
        lines.append(
            f"- dim {row['dim']}: KSC-dense - exact-dense = {row['ksc_dense_minus_exact_dense']:.6g}, "
            f"Kalman(finest) - exact-dense = {row['kalman_finest_minus_exact_dense']:.6g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    args = _parse_args()
    started = time.time()
    dims = _parse_int_list(args.dims, name="dims")
    if any(dim <= 0 or dim > 3 for dim in dims):
        raise ValueError("dims must be between 1 and 3 for this harness")
    component_counts = _parse_int_list(args.mixture_components, name="mixture-components")

    factorization = _factorization_check(args)
    batch_tt_row = None if args.skip_batch_tt else _collect_batch_tt_row(args)
    exact_rows = _collect_exact_transformed_rows(args, dims)
    ksc_rows = _collect_ksc_surrogate_rows(args, dims)
    refinement_rows, mixture_fit_rows = _collect_refinement_rows(
        args, dims, component_counts
    )
    cross_rows = _cross_family_rows(args, dims, exact_rows, ksc_rows, refinement_rows)

    result: dict[str, Any] = {
        "schema_version": "actual_sv_three_route_simulation.v1",
        "plan": PLAN,
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_commit(),
        "dims": dims,
        "horizon": args.horizon,
        "seed_base": args.seed_base,
        "dense_order": args.dense_order,
        "dense_radius": args.dense_radius,
        "ksc_transform_offset": args.ksc_transform_offset,
        "fd_step": args.fd_step,
        "mixture_component_counts": component_counts,
        "fixture_kind": "simulated_exact_actual_sv_paths_iid_coordinates",
        "factorization_check": factorization,
        "batch_tt_row": batch_tt_row,
        "exact_transformed_rows": exact_rows,
        "ksc_surrogate_rows": ksc_rows,
        "refinement_rows": refinement_rows,
        "mixture_fit_rows": mixture_fit_rows,
        "cross_family_rows": cross_rows,
        "wall_time_seconds": time.time() - started,
        "nonclaims": list(NONCLAIMS),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.markdown_output is not None:
        markdown_path = Path(args.markdown_output)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_path, result, output_path)
    print(json.dumps({k: v for k, v in result.items() if k != "mixture_fit_rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
