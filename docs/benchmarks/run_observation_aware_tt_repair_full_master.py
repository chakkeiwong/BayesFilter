"""Bounded end-to-end diagnostic for the observation-aware TT repair.

This driver deliberately uses a scalar stochastic-volatility fixture.  It
retains a particle bank, builds an observation-aware SGQF guide, fits a
fixed-rank square-root TT to the one-step target, samples a TT conditional by
grid CDF inversion, and applies the exact transition/observation correction.
The conditional grid sampler is an explicitly labelled diagnostic extension;
it is not the paper-scale Zhao--Cui production route.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# This master is intentionally a bounded CPU diagnostic.  Hide CUDA before
# TensorFlow import so a plain invocation cannot accidentally become a GPU run.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
    HighDimStatus,
)
from bayesfilter.highdim.fitting import (
    FixedTTFitConfig,
    FixedTTFitSampleBatch,
    FixedTTFitter,
)
from bayesfilter.highdim.models import StochasticVolatilitySSM
from bayesfilter.highdim.tt import TTCore
from bayesfilter.nonlinear.fixed_sgqf_tf import tf_standard_normal_ghq_level_rule


REPO = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO / "docs/benchmarks/artifacts/observation_aware_tt_repair_full_master_20260913"
DEFAULT_RUN = "run-01"
DTYPE = tf.float64
EPS = tf.constant(1.0e-12, dtype=DTYPE)
FIT_RESIDUAL_VETO = 1.0
ESS_VETO = 4.0
GUIDE_MIX = 0.70
RESAMPLE_ESS_FRACTION = 0.50
CDF_INVERSE_VETO = 1.0e-8


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def _normal_log_prob(x: tf.Tensor, loc: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    return tfp.distributions.Normal(loc=loc, scale=scale).log_prob(x)


def _convention() -> MeasureConvention:
    return MeasureConvention(
        density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )


def _build_basis() -> ProductBasis:
    domain = BoundedInterval(-8.0, 8.0)
    return ProductBasis(
        [LegendreBasis1D(domain, 4), LegendreBasis1D(domain, 4)],
        _convention(),
    )


def _fit_config() -> FixedTTFitConfig:
    return FixedTTFitConfig(
        ranks=(1, 2, 1),
        ridge=1.0e-8,
        max_sweeps=2,
        sweep_order=(0, 1),
        row_budget=4096,
        column_budget=256,
        dense_matrix_byte_budget=8_000_000,
        normal_matrix_byte_budget=2_000_000,
        condition_number_warning=1.0e10,
        condition_number_veto=1.0e14,
        holdout_tolerance=10.0,
    )


def _initial_cores(seed: int) -> tuple[TTCore, TTCore]:
    core0 = 0.02 * tf.random.stateless_normal([1, 5, 2], [seed, 11], dtype=DTYPE)
    core1 = 0.02 * tf.random.stateless_normal([2, 5, 1], [seed, 13], dtype=DTYPE)
    core0 = tf.tensor_scatter_nd_update(core0, [[0, 0, 0]], [tf.constant(1.0, DTYPE)])
    core1 = tf.tensor_scatter_nd_update(core1, [[0, 0, 0]], [tf.constant(1.0, DTYPE)])
    return TTCore(core0), TTCore(core1)


def _guide(
    model: StochasticVolatilitySSM,
    theta: tf.Tensor,
    x_prev: tf.Tensor,
    y: tf.Tensor,
    *,
    level: int = 5,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Likelihood-weighted SGQF moments for every retained ancestor."""
    parameters = model.physical_parameters(theta)
    rule = tf_standard_normal_ghq_level_rule(level)
    mu = parameters["gamma"] * x_prev[:, 0]
    cloud = mu[:, None] + parameters["sigma"] * rule.nodes[None, :]
    flat = tf.reshape(cloud, [-1, 1])
    log_likelihood = tf.reshape(model.observation_log_density(theta, flat, y, 0), tf.shape(cloud))
    log_alpha = tf.math.log(rule.weights)[None, :] + log_likelihood
    alpha = tf.nn.softmax(log_alpha, axis=1)
    mean = tf.reduce_sum(alpha * cloud, axis=1)
    variance = tf.reduce_sum(alpha * tf.square(cloud - mean[:, None]), axis=1)
    return mean, tf.maximum(variance, EPS), cloud, alpha


def _guide_response_check(model: StochasticVolatilitySSM, theta: tf.Tensor) -> dict[str, Any]:
    x_prev = tf.constant([[0.0]], DTYPE)
    m0, v0, _, _ = _guide(model, theta, x_prev, tf.constant([0.08], DTYPE))
    m1, v1, _, _ = _guide(model, theta, x_prev, tf.constant([1.25], DTYPE))
    return {
        "mean_low_observation": m0[0],
        "mean_tail_observation": m1[0],
        "variance_low_observation": v0[0],
        "variance_tail_observation": v1[0],
        "mean_response_gap": tf.abs(m1[0] - m0[0]),
        "finite": _finite(tf.stack([m0[0], m1[0], v0[0], v1[0]])),
        "positive_variance": bool(tf.reduce_all(tf.stack([v0[0], v1[0]]) > 0.0).numpy()),
    }


def _fit_step_tt(
    model: StochasticVolatilitySSM,
    theta: tf.Tensor,
    x_prev: tf.Tensor,
    previous_weights: tf.Tensor,
    y: tf.Tensor,
    step: int,
) -> tuple[Any, dict[str, Any]]:
    del previous_weights
    # A regular fixed design over the complete declared domain is needed here:
    # fitting only at the nine SGQF nodes leaves a degree-4 polynomial
    # unconstrained between nodes and can create a squared-TT spike at a domain
    # boundary.  Covering [-8, 8] also prevents extrapolation when a retained
    # particle reaches the proposal-grid boundary.
    prev_design = tf.cast(tf.linspace(-8.0, 8.0, 41), DTYPE)
    next_design = tf.cast(tf.linspace(-8.0, 8.0, 41), DTYPE)
    prev_rep = tf.repeat(prev_design, tf.shape(next_design)[0])
    x_next = tf.tile(next_design, [tf.shape(prev_design)[0]])
    points = tf.stack([prev_rep, x_next], axis=1)
    transition_log = tf.reshape(
        model.transition_log_density(theta, prev_rep[:, None], x_next[:, None], step),
        [-1],
    )
    observation_log = tf.reshape(
        model.observation_log_density(theta, x_next[:, None], y, step),
        [-1],
    )
    log_target = transition_log + observation_log
    shift = tf.reduce_max(log_target)
    target = tf.exp(0.5 * (log_target - shift))
    sample_batch = FixedTTFitSampleBatch(
        points=points,
        target_values=target,
        weights=tf.ones_like(target),
    )
    # Add the observation-weighted SGQF row cloud for the retained ancestors.
    # The regular design controls extrapolation and global conditioning; these
    # rows make the current posterior/observation pair part of the regression.
    _, _, guide_cloud, guide_alpha = _guide(model, theta, x_prev, y)
    guide_prev = tf.broadcast_to(
        x_prev[:, 0, None], tf.shape(guide_cloud)
    )
    guide_points = tf.stack(
        [tf.reshape(guide_prev, [-1]), tf.reshape(guide_cloud, [-1])], axis=1
    )
    guide_transition_log = tf.reshape(
        model.transition_log_density(
            theta,
            tf.reshape(guide_prev, [-1, 1]),
            tf.reshape(guide_cloud, [-1, 1]),
            step,
        ),
        [-1],
    )
    guide_observation_log = tf.reshape(
        model.observation_log_density(
            theta, tf.reshape(guide_cloud, [-1, 1]), y, step
        ),
        [-1],
    )
    guide_target = tf.exp(
        0.5 * (guide_transition_log + guide_observation_log - shift)
    )
    sample_batch = FixedTTFitSampleBatch(
        points=tf.concat([sample_batch.points, guide_points], axis=0),
        target_values=tf.concat([sample_batch.target_values, guide_target], axis=0),
        weights=tf.concat(
            [sample_batch.weights, tf.reshape(guide_alpha, [-1])], axis=0
        ),
    )
    product_basis = _build_basis()
    fit_result = FixedTTFitter().fit(
        product_basis=product_basis,
        samples=sample_batch,
        config=_fit_config(),
        initial_cores=_initial_cores(1200 + step),
        branch_seed=f"observation-aware-full-master-step-{step}",
        measure_convention=_convention(),
        initialization_rule="stateless_random_small_core_plus_constant",
    )
    # Evaluate on deterministic cell midpoints that were not used by the fit.
    # This is an explanatory interpolation check; it is kept separate from
    # the fit residual and is still required to be finite.
    holdout_prev = tf.cast(
        tf.linspace(-8.0 + 16.0 / 82.0, 8.0 - 16.0 / 82.0, 40), DTYPE
    )
    holdout_next = tf.cast(
        tf.linspace(-8.0 + 16.0 / 82.0, 8.0 - 16.0 / 82.0, 40), DTYPE
    )
    holdout_prev_rep = tf.repeat(holdout_prev, tf.shape(holdout_next)[0])
    holdout_next_rep = tf.tile(holdout_next, [tf.shape(holdout_prev)[0]])
    holdout_points = tf.stack([holdout_prev_rep, holdout_next_rep], axis=1)
    holdout_log_target = tf.reshape(
        model.transition_log_density(
            theta, holdout_prev_rep[:, None], holdout_next_rep[:, None], step
        ),
        [-1],
    ) + tf.reshape(
        model.observation_log_density(theta, holdout_next_rep[:, None], y, step),
        [-1],
    )
    holdout_target = tf.exp(0.5 * (holdout_log_target - shift))
    holdout_prediction = fit_result.fitted_tt.evaluate(holdout_points)
    holdout_residual = tf.sqrt(
        tf.reduce_mean(tf.square(holdout_prediction - holdout_target))
    )
    diagnostics = {
        "status": fit_result.status.value,
        "termination_reason": fit_result.termination_reason,
        "fit_residual": fit_result.fit_residual,
        "holdout_residual": holdout_residual,
        "target_log_shift": shift,
        "branch_hash": fit_result.branch_identity.hash.value,
        "rank_tuple": fit_result.fitted_tt.rank_tuple(),
        "finite": (
            fit_result.fit_residual is not None
            and _finite(fit_result.fit_residual)
            and _finite(holdout_residual)
        ),
    }
    return fit_result, diagnostics


def _conditional_grid(
    fitted_tt: Any,
    x_prev: tf.Tensor,
    grid: tf.Tensor,
    guide_mean: tf.Tensor,
    guide_variance: tf.Tensor,
    guide_mix: float = GUIDE_MIX,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    n = int(x_prev.shape[0])
    g = int(grid.shape[0])
    points = tf.stack(
        [tf.broadcast_to(x_prev[:, 0, None], [n, g]), tf.broadcast_to(grid[None, :], [n, g])],
        axis=2,
    )
    raw = tf.square(fitted_tt.evaluate(tf.reshape(points, [-1, 2]))) + EPS
    tt_density = tf.reshape(raw, [n, g])
    dx = grid[1:] - grid[:-1]
    tt_normalizer = tf.reduce_sum(
        0.5 * (tt_density[:, 1:] + tt_density[:, :-1]) * dx[None, :], axis=1
    )
    tt_density = tt_density / tt_normalizer[:, None]
    guide_scale = tf.sqrt(tf.maximum(guide_variance, EPS))
    guide_density = tf.exp(
        _normal_log_prob(
            grid[None, :], guide_mean[:, None], guide_scale[:, None]
        )
    )
    guide_normalizer = tf.reduce_sum(
        0.5 * (guide_density[:, 1:] + guide_density[:, :-1]) * dx[None, :], axis=1
    )
    guide_density = guide_density / guide_normalizer[:, None]
    density = (1.0 - guide_mix) * tt_density + guide_mix * guide_density
    masses = 0.5 * (density[:, 1:] + density[:, :-1]) * dx[None, :]
    normalizer = tf.reduce_sum(masses, axis=1)
    cdf = tf.concat(
        [tf.zeros([n, 1], DTYPE), tf.cumsum(masses, axis=1)], axis=1
    ) / normalizer[:, None]
    return density, normalizer, cdf


def _inverse_piecewise_cdf(
    grid: tf.Tensor,
    density: tf.Tensor,
    normalizer: tf.Tensor,
    cdf: tf.Tensor,
    probabilities: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    samples: list[tf.Tensor] = []
    q_values: list[tf.Tensor] = []
    inverse_errors: list[tf.Tensor] = []
    for i in range(int(probabilities.shape[0])):
        u = tf.clip_by_value(probabilities[i], 1.0e-9, 1.0 - 1.0e-9)
        idx = int(tf.searchsorted(cdf[i], tf.reshape(u, [1]), side="left")[0].numpy())
        idx = max(1, min(idx, int(grid.shape[0]) - 1))
        c0, c1 = cdf[i, idx - 1], cdf[i, idx]
        d0, d1 = density[i, idx - 1], density[i, idx]
        dx = grid[idx] - grid[idx - 1]
        # The CDF uses the trapezoidal integral of a linearly interpolated
        # density.  Invert that quadratic interval mass so the sampled law and
        # the reported q(x) describe the same proposal.
        target_mass = (u - c0) * normalizer[i] / tf.maximum(dx, EPS)
        slope = d1 - d0
        discriminant = tf.maximum(tf.square(d0) + 2.0 * slope * target_mass, 0.0)
        quadratic_frac = 2.0 * target_mass / tf.maximum(
            d0 + tf.sqrt(discriminant), EPS
        )
        linear_frac = target_mass / tf.maximum(d0, EPS)
        frac = tf.where(tf.abs(slope) < 1.0e-14, linear_frac, quadratic_frac)
        frac = tf.clip_by_value(frac, 0.0, 1.0)
        x = grid[idx - 1] + frac * (grid[idx] - grid[idx - 1])
        q = (d0 + frac * (d1 - d0)) / normalizer[i]
        reconstructed_cdf = c0 + dx * (
            d0 * frac + 0.5 * slope * tf.square(frac)
        ) / normalizer[i]
        samples.append(x)
        q_values.append(q)
        inverse_errors.append(tf.abs(reconstructed_cdf - u))
    return tf.stack(samples), tf.stack(q_values), tf.stack(inverse_errors)


def _tt_proposal_step(
    model: StochasticVolatilitySSM,
    theta: tf.Tensor,
    fit_result: Any,
    x_prev: tf.Tensor,
    previous_weights: tf.Tensor,
    y: tf.Tensor,
    step: int,
    grid: tf.Tensor,
    guide_mean: tf.Tensor,
    guide_variance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, Any]]:
    density, normalizer, cdf = _conditional_grid(
        fit_result.fitted_tt, x_prev, grid, guide_mean, guide_variance
    )
    n = int(x_prev.shape[0])
    probabilities = (tf.cast(tf.range(n), DTYPE) + 0.5) / tf.cast(n, DTYPE)
    proposal, q, inverse_errors = _inverse_piecewise_cdf(
        grid, density, normalizer, cdf, probabilities
    )
    x_next_matrix = proposal[:, None]
    transition_log = model.transition_log_density(theta, x_prev, x_next_matrix, step)
    observation_log = model.observation_log_density(theta, x_next_matrix, y, step)
    log_q = tf.math.log(tf.maximum(q, EPS))
    correction = transition_log + observation_log - log_q
    log_weights = tf.math.log(tf.maximum(previous_weights, EPS)) + correction
    weights = tf.nn.softmax(log_weights)
    diagnostics = {
        "cdf_min_increment": tf.reduce_min(cdf[:, 1:] - cdf[:, :-1]),
        "cdf_max_terminal_error": tf.reduce_max(tf.abs(cdf[:, -1] - 1.0)),
        "cdf_inverse_max_error": tf.reduce_max(inverse_errors),
        "normalizer_min": tf.reduce_min(normalizer),
        "proposal_log_density_min": tf.reduce_min(tf.math.log(tf.maximum(q, EPS))),
        "exact_correction_min": tf.reduce_min(correction),
        "exact_correction_max": tf.reduce_max(correction),
        "exact_correction_finite": _finite(correction),
        "effective_sample_size": 1.0 / tf.reduce_sum(tf.square(weights)),
        "finite": _finite(tf.concat([proposal, q, weights], axis=0)),
        "positive_normalizer": bool(tf.reduce_all(normalizer > 0.0).numpy()),
        "monotone_cdf": bool(tf.reduce_all(cdf[:, 1:] >= cdf[:, :-1]).numpy()),
        "normalized_weights": tf.reduce_sum(weights),
        "guide_mix": GUIDE_MIX,
        "cdf_inverse_finite": _finite(inverse_errors),
    }
    return x_next_matrix, weights, diagnostics


def _prior_proposal_step(
    model: StochasticVolatilitySSM,
    theta: tf.Tensor,
    x_prev: tf.Tensor,
    previous_weights: tf.Tensor,
    y: tf.Tensor,
    step: int,
) -> tuple[tf.Tensor, tf.Tensor, dict[str, Any]]:
    parameters = model.physical_parameters(theta)
    n = int(x_prev.shape[0])
    probabilities = (tf.cast(tf.range(n), DTYPE) + 0.5) / tf.cast(n, DTYPE)
    z = tfp.distributions.Normal(tf.constant(0.0, DTYPE), tf.constant(1.0, DTYPE)).quantile(probabilities)
    mu = parameters["gamma"] * x_prev[:, 0]
    x_next = (mu + parameters["sigma"] * z)[:, None]
    q = tf.exp(model.transition_log_density(theta, x_prev, x_next, step))
    log_weights = tf.math.log(tf.maximum(previous_weights, EPS)) + model.observation_log_density(theta, x_next, y, step)
    weights = tf.nn.softmax(log_weights)
    return x_next, weights, {
        "effective_sample_size": 1.0 / tf.reduce_sum(tf.square(weights)),
        "finite": _finite(tf.concat([tf.reshape(x_next, [-1]), q, weights], axis=0)),
        "normalized_weights": tf.reduce_sum(weights),
    }


def _systematic_resample(
    particles: tf.Tensor, weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Deterministic identity-preserving systematic resampling."""
    n = int(particles.shape[0])
    probabilities = (tf.cast(tf.range(n), DTYPE) + 0.5) / tf.cast(n, DTYPE)
    indices = tf.searchsorted(tf.cumsum(weights), probabilities, side="left")
    indices = tf.clip_by_value(indices, 0, n - 1)
    refreshed = tf.gather(particles, indices)
    refreshed_weights = tf.fill([n], tf.constant(1.0 / n, DTYPE))
    return refreshed, refreshed_weights, indices


def _run(args: argparse.Namespace) -> dict[str, Any]:
    model = StochasticVolatilitySSM(sigma=1.0)
    theta = model.unconstrained_from_physical(gamma=0.65, beta=0.4)
    _, observations = model.simulate(theta, final_time=args.horizon - 1, seed=20260913)
    observations = tf.reshape(observations, [args.horizon, 1])
    parameters = model.physical_parameters(theta)
    prior_scale = parameters["sigma"] / tf.sqrt(1.0 - tf.square(parameters["gamma"]))
    x_initial = (
        tf.cast(tf.linspace(-2.5, 2.5, args.particles), DTYPE) * prior_scale
    )[:, None]
    weights_initial = tf.fill([args.particles], tf.constant(1.0 / args.particles, DTYPE))
    guide = _guide_response_check(model, theta)
    grid = tf.cast(tf.linspace(-8.0, 8.0, args.grid), DTYPE)
    tt_x, tt_w = x_initial, weights_initial
    sir_x, sir_w = x_initial, weights_initial
    tt_steps: list[dict[str, Any]] = []
    sir_steps: list[dict[str, Any]] = []
    vetoes: list[str] = []
    if (
        not guide["finite"]
        or not guide["positive_variance"]
        or float(guide["mean_response_gap"]) <= 0.0
    ):
        vetoes.append("sgqf_guide_response")
    for step in range(args.horizon):
        if vetoes:
            break
        y = observations[step]
        fit_result, fit_diag = _fit_step_tt(model, theta, tt_x, tt_w, y, step)
        if fit_result.status is not HighDimStatus.OK or not fit_diag["finite"]:
            vetoes.append(f"step_{step}:tt_fit:{fit_result.status.value}")
            break
        guide_mean, guide_variance, _, _ = _guide(model, theta, tt_x, y)
        if (
            not _finite(tf.concat([guide_mean, guide_variance], axis=0))
            or not bool(tf.reduce_all(guide_variance > 0.0).numpy())
        ):
            vetoes.append(f"step_{step}:sgqf_guide_validity")
            break
        tt_x, tt_w, proposal_diag = _tt_proposal_step(model, theta, fit_result, tt_x, tt_w, y, step, grid, guide_mean, guide_variance)
        step_payload = {"step": step, "observation": y, "fit": fit_diag, "proposal": proposal_diag, "posterior_mean": tf.reduce_sum(tt_w * tt_x[:, 0]), "posterior_variance": tf.reduce_sum(tt_w * tf.square(tt_x[:, 0] - tf.reduce_sum(tt_w * tt_x[:, 0])))}
        tt_steps.append(step_payload)
        if (
            not proposal_diag["finite"]
            or not proposal_diag["positive_normalizer"]
            or not proposal_diag["monotone_cdf"]
            or not proposal_diag["cdf_inverse_finite"]
            or float(proposal_diag["cdf_inverse_max_error"]) > CDF_INVERSE_VETO
            or float(fit_diag["fit_residual"]) > FIT_RESIDUAL_VETO
            or not proposal_diag["exact_correction_finite"]
            or float(proposal_diag["effective_sample_size"]) < ESS_VETO
        ):
            vetoes.append(f"step_{step}:proposal_validity")
            break
        if float(proposal_diag["effective_sample_size"]) < RESAMPLE_ESS_FRACTION * args.particles:
            tt_x, tt_w, indices = _systematic_resample(tt_x, tt_w)
            proposal_diag["resampled"] = True
            proposal_diag["resampled_unique_ancestors"] = tf.size(tf.unique(indices)[0])
        else:
            proposal_diag["resampled"] = False
            proposal_diag["resampled_unique_ancestors"] = args.particles
        sir_x, sir_w, sir_diag = _prior_proposal_step(model, theta, sir_x, sir_w, y, step)
        sir_payload = {"step": step, "observation": y, "proposal": sir_diag, "posterior_mean": tf.reduce_sum(sir_w * sir_x[:, 0]), "posterior_variance": tf.reduce_sum(sir_w * tf.square(sir_x[:, 0] - tf.reduce_sum(sir_w * sir_x[:, 0])))}
        if float(sir_diag["effective_sample_size"]) < RESAMPLE_ESS_FRACTION * args.particles:
            sir_x, sir_w, sir_indices = _systematic_resample(sir_x, sir_w)
            sir_diag["resampled"] = True
            sir_diag["resampled_unique_ancestors"] = tf.size(tf.unique(sir_indices)[0])
        else:
            sir_diag["resampled"] = False
            sir_diag["resampled_unique_ancestors"] = args.particles
        sir_steps.append(sir_payload)
    status = "PASS" if not vetoes and len(tt_steps) == args.horizon else "FAIL"
    situations = []
    for index, observation in enumerate(observations):
        magnitude = float(tf.abs(observation[0]).numpy())
        label = "near_zero" if magnitude < 0.15 else ("tail" if magnitude > 0.75 else "ordinary")
        situations.append({"step": index, "observation": observation, "class": label})
    return {
        "status": status,
        "research_question": "recursive retained-bank SGQF guide plus fixed square-root TT conditional proposal",
        "classification": {
            "sgqf_guide": {
                "classification": "source_faithful_repository_contract",
                "anchors": ["bayesfilter/nonlinear/fixed_sgqf_tf.py:619-634"],
            },
            "tt_fit": {
                "classification": "source_faithful_repository_contract",
                "anchors": ["bayesfilter/highdim/fitting.py:221-290"],
            },
            "conditional_grid_inverse": {
                "classification": "extension_or_invention_diagnostic",
                "anchors": ["docs/plans/observation-aware-tt-repair-full-master-20260913.md"],
            },
            "exact_correction": {
                "classification": "source_faithful_model_density_ratio",
                "anchors": ["bayesfilter/highdim/models.py:320-372"],
            },
        },
        "fixture": {"model": model.manifest_payload(), "theta": theta, "physical_parameters": parameters, "observations": observations, "horizon": args.horizon, "particles": args.particles, "grid_points": args.grid},
        "sgqf_guide_response": guide,
        "tt_steps": tt_steps,
        "prior_sir_steps": sir_steps,
        "heuristic_adversaries": [
            {"id": "prior_proposal_sir", "rationale": "cheap unadjusted transition proposal"},
            {"id": "sgqf_guided_moment", "rationale": "cheap observation-aware moment update"},
        ],
        "conditional_situations": situations,
        "resampling_policy": {"type": "deterministic_systematic", "ess_fraction": RESAMPLE_ESS_FRACTION, "identity_preserved": True},
        "heuristic_dominance_verdict": "descriptive_only; no promotion ranking from one seed",
        "vetoes": vetoes,
        "nonclaims": ["no source-faithful paper-scale KR claim", "no posterior correctness or convergence claim", "no TT superiority claim", "no HMC or production-readiness claim"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--grid", type=int, default=129)
    args = parser.parse_args()
    if args.horizon <= 0 or args.particles < 4 or args.grid < 17 or args.grid % 2 == 0:
        raise SystemExit("horizon>0, particles>=4, and odd grid>=17 are required")
    run_dir = ARTIFACT_ROOT / str(args.run_id)
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing artifact directory: {run_dir}")
    run_dir.mkdir(parents=True)
    start = time.time()
    result = _run(args)
    result["wall_time_seconds"] = time.time() - start
    result["run_id"] = args.run_id
    result["artifact_root"] = str(run_dir)
    manifest = {
        "program": str(Path(__file__).relative_to(REPO)),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip(),
        "command": " ".join([sys.executable, *sys.argv]),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "device_policy": "CPU diagnostic; CUDA_VISIBLE_DEVICES=-1",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_status": "intentionally_hidden_cpu_diagnostic",
        "jit_compile": False,
        "memory_policy": "N/A; CPU-only diagnostic",
        "plan": "docs/plans/observation-aware-tt-repair-full-master-20260913.md",
        "seed": 20260913,
        "run_id": args.run_id,
    }
    required_manifest_fields = (
        "program",
        "git_commit",
        "command",
        "python",
        "tensorflow",
        "tensorflow_probability",
        "cuda_visible_devices",
        "gpu_status",
        "jit_compile",
        "plan",
        "seed",
        "run_id",
    )
    missing_manifest_fields = [
        key for key in required_manifest_fields if key not in manifest
    ]
    if missing_manifest_fields:
        result["vetoes"].append(
            "manifest_missing_fields:" + ",".join(missing_manifest_fields)
        )
        result["status"] = "FAIL"
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "run_manifest.json", manifest)
    for step in result["tt_steps"]:
        _write_json(run_dir / f"tt_step_{int(step['step']):02d}.json", step)
    for step in result["prior_sir_steps"]:
        _write_json(run_dir / f"prior_sir_step_{int(step['step']):02d}.json", step)
    (run_dir / "command.log").write_text(" ".join([sys.executable, *sys.argv]) + "\n")
    print(json.dumps({"status": result["status"], "run_dir": str(run_dir), "vetoes": result["vetoes"], "steps": len(result["tt_steps"])}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
